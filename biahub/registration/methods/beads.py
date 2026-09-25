"""
Beads-based registration module.

Provides functions for registering volumetric imaging data by detecting fluorescent
bead landmarks in moving and reference channels, matching them using graph-based
algorithms, and estimating affine transformations.

Pipeline overview
-----------------
1. **Peak detection** (`peaks_from_beads`): Detect bead positions in both channels.
2. **Matching** (`matches_from_beads`): Find bead correspondences via graph matching
   (Hungarian or descriptor-based) with geometric consistency filtering.
3. **Transform estimation** (`transform_from_matches`): Fit an affine/euclidean/similarity
   transform from matched bead pairs.
4. **Iterative refinement** (`optimize_transform`, `estimate`): Compose the approximate
   transform with bead-based corrections, re-detect peaks, and score until convergence.

Key conventions
---------------
- Coordinates are in ZYX order for 3D data.
- "mov" / "moving" refers to the source channel being aligned.
- "ref" / "reference" refers to the fixed target channel.
- Transforms map from moving space to reference space (forward direction).
"""

from __future__ import annotations

from pathlib import Path

import ants
import click
import dask.array as da
import numpy as np

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform

from biahub.characterize_psf import detect_peaks
from biahub.core.graph_matching import Graph, GraphMatcher
from biahub.core.transform import Transform
from biahub.registration.estimators import (
    EstimationError,
    NodeDetector,
    NodeMatcher,
    ScoreFn,
)
from biahub.settings import AffineTransformSettings, BeadsMatchSettings, DetectPeaksSettings


def overlap_score(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    radius: int = 6,
    verbose: bool = False,
) -> float:
    """
    Compute the overlap fraction between two sets of bead peaks.

    For each reference peak, checks whether any moving peak falls within a
    spherical neighborhood of the given radius (using a KDTree). The score is
    the fraction of reference peaks that have at least one nearby moving peak,
    normalized by the smaller peak set size.

    Parameters
    ----------
    mov_peaks : ArrayLike
        (N_mov, D) array of moving bead coordinates (z, y, x).
    ref_peaks : ArrayLike
        (N_ref, D) array of reference bead coordinates (z, y, x).
    radius : int
        Spherical neighborhood radius in voxels for overlap counting.
    verbose : bool
        If True, prints peak counts and overlap statistics.

    Returns
    -------
    float
        Overlap fraction in [0, 1]. Returns np.nan if either peak set is empty.
    """
    if len(mov_peaks) == 0 or len(ref_peaks) == 0:
        click.echo("No peaks found, returning nan metrics")
        return np.nan

    # ---- Overlap counting using KDTree ----
    mov_tree = cKDTree(mov_peaks)

    ref_peaks_mask = np.zeros(len(ref_peaks), dtype=bool)
    mov_peaks_mask = np.zeros(len(mov_peaks), dtype=bool)

    for i, p in enumerate(ref_peaks):
        idx = mov_tree.query_ball_point(p, r=radius)
        if idx:
            ref_peaks_mask[i] = True
            mov_peaks_mask[idx] = True

    peaks_overlap_count = int(ref_peaks_mask.sum())

    # ---- Overlap fraction ----
    peaks_overlap_fraction = peaks_overlap_count / max(min(len(mov_peaks), len(ref_peaks)), 1)

    if verbose:
        click.echo(f"Mov peaks: {len(mov_peaks)}")
        click.echo(f"Ref peaks: {len(ref_peaks)}")
        click.echo(f"Peaks overlap count: {peaks_overlap_count}")
        click.echo(f"Peaks overlap fraction: {peaks_overlap_fraction}")

    return peaks_overlap_fraction


def peaks_from_beads(
    mov: da.Array,
    ref: da.Array,
    mov_peaks_settings: DetectPeaksSettings,
    ref_peaks_settings: DetectPeaksSettings,
    verbose: bool = False,
    mask_path: Path = None,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Detect peaks in moving and reference channels using the detect_peaks function.

    Parameters
    ----------
    mov : da.Array
        (Z, Y, X) array of the moving channel (Dask array).
    ref : da.Array
        (Z, Y, X) array of the reference channel (Dask array).
    mov_peaks_settings : DetectPeaksSettings
        Settings for the moving peaks.
    ref_peaks_settings : DetectPeaksSettings
        Settings for the reference peaks.
    verbose : bool
        If True, prints detailed logs during the process.
    mask_path : Path
        Path to the mask file.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Tuple of (mov_peaks, ref_peaks).
    """
    if verbose:
        click.echo("Detecting beads in moving dataset")
    # TODO: detecte peaks in the zyx space, use skimage.feature.peak_local_max for 2D
    mov_peaks = detect_peaks(
        mov,
        block_size=mov_peaks_settings.block_size,
        threshold_abs=mov_peaks_settings.threshold_abs,
        nms_distance=mov_peaks_settings.nms_distance,
        min_distance=mov_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo("Detecting beads in reference dataset")
    # TODO: detecte peaks in the zyx space, use skimage.feature.peak_local_max for 2D
    ref_peaks = detect_peaks(
        ref,
        block_size=ref_peaks_settings.block_size,
        threshold_abs=ref_peaks_settings.threshold_abs,
        nms_distance=ref_peaks_settings.nms_distance,
        min_distance=ref_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo(f"Total of peaks in moving dataset: {len(mov_peaks)}")
        click.echo(f"Total of peaks in reference dataset: {len(ref_peaks)}")

    if len(mov_peaks) < 2 or len(ref_peaks) < 2:
        click.echo("Not enough beads detected")
        return
    if mask_path is not None:
        click.echo("Filtering peaks with mask")
        with open_ome_zarr(mask_path) as mask_ds:
            mask_load = np.asarray(mask_ds.data[0, 0])

        # filter the peaks with the mask
        # Keep only peaks whose (y, x) column is clean across all Z slices
        ref_peaks_filtered = []
        for peak in ref_peaks:
            z, y, x = peak.astype(int)
            if (
                0 <= y < mask_load.shape[1]
                and 0 <= x < mask_load.shape[2]
                and not mask_load[:, y, x].any()  # True if all Z are clean at (y, x)
            ):
                ref_peaks_filtered.append(peak)
        ref_peaks = np.array(ref_peaks_filtered)
    return mov_peaks, ref_peaks


def matches_from_beads(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
    verbose: bool = False,
) -> ArrayLike:
    """
    Find bead correspondences between moving and reference peak sets.

    Supports two matching algorithms:
    - "hungarian": Builds k-NN graphs for both peak sets, computes a cost matrix
      based on position distance and edge consistency, then solves the assignment
      problem with the Hungarian algorithm.
    - "match_descriptor": Uses scikit-image's descriptor matching on peak positions.

    After matching, applies geometric consistency filters (distance quantiles,
    direction threshold, angle threshold) to remove outliers.

    Parameters
    ----------
    mov_peaks : ArrayLike
        (N, D) array of moving peak coordinates (D = 2 or 3).
    ref_peaks : ArrayLike
        (M, D) array of reference peak coordinates.
    beads_match_settings : BeadsMatchSettings
        Settings controlling the matching algorithm, graph construction,
        cost matrix weights, and post-match filtering.
    verbose : bool
        If True, prints matching settings and match count.

    Returns
    -------
    ArrayLike
        (K, 2) array of matched index pairs [mov_idx, ref_idx].
    """
    if verbose:
        click.echo(f"Getting matches from beads with settings: {beads_match_settings}")

    if beads_match_settings.algorithm == "match_descriptor":
        mov_graph = Graph.from_nodes(mov_peaks)
        ref_graph = Graph.from_nodes(ref_peaks)

        match_descriptor_settings = beads_match_settings.match_descriptor_settings
        matcher = GraphMatcher(
            algorithm="descriptor",
            cross_check=match_descriptor_settings.cross_check,
            max_ratio=match_descriptor_settings.max_ratio,
            metric=match_descriptor_settings.distance_metric,
            verbose=verbose,
        )

        matches = matcher.match(mov_graph, ref_graph)

    elif beads_match_settings.algorithm == "hungarian":
        hungarian_match_settings = beads_match_settings.hungarian_match_settings
        mov_graph = Graph.from_nodes(
            mov_peaks, mode="knn", k=hungarian_match_settings.edge_graph_settings.k
        )
        ref_graph = Graph.from_nodes(
            ref_peaks, mode="knn", k=hungarian_match_settings.edge_graph_settings.k
        )

        matcher = GraphMatcher(
            algorithm="hungarian",
            weights=hungarian_match_settings.cost_matrix_settings.weights,
            cost_threshold=hungarian_match_settings.cost_threshold,
            cross_check=hungarian_match_settings.cross_check,
            max_ratio=hungarian_match_settings.max_ratio,
            verbose=verbose,
        )

        matches = matcher.match(mov_graph, ref_graph)

    # Filter as part of the pipeline
    elif beads_match_settings.algorithm == "spectral":
        spectral_match_settings = beads_match_settings.spectral_match_settings
        # Spectral matching works from pairwise distances of the raw coordinates; the
        # graph is only a node container here.
        mov_graph = Graph.from_nodes(mov_peaks)
        ref_graph = Graph.from_nodes(ref_peaks)
        matcher = GraphMatcher(
            algorithm="spectral",
            spectral_sigma=spectral_match_settings.sigma,
            spectral_rel_cut=spectral_match_settings.rel_cut,
            spectral_max_iter=spectral_match_settings.max_iter,
            verbose=verbose,
        )
        matches = matcher.match(mov_graph, ref_graph)
    else:
        raise ValueError(f"Unknown matching algorithm: {beads_match_settings.algorithm}")

    matches = matcher.filter_matches(
        matches,
        mov_graph,
        ref_graph,
        angle_threshold=beads_match_settings.filter_matches_settings.angle_threshold,
        min_distance_quantile=beads_match_settings.filter_matches_settings.min_distance_quantile,
        max_distance_quantile=beads_match_settings.filter_matches_settings.max_distance_quantile,
        direction_threshold=beads_match_settings.filter_matches_settings.direction_threshold,
    )

    if verbose:
        click.echo(f"Total of matches: {len(matches)}")

    return matches


def transform_from_matches(
    matches: ArrayLike,
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    affine_transform_settings: AffineTransformSettings,
    ndim: int = 3,
    verbose: bool = False,
) -> tuple[Transform, Transform]:
    """
    Estimate the affine transformation matrix between source and target channels.

    Based on detected bead matches at a specific timepoint.

    Parameters
    ----------
    matches : ArrayLike
        (n, 2) array of matches.
    mov_peaks : ArrayLike
        (n, 2) array of moving peaks.
    ref_peaks : ArrayLike
        (n, 2) array of reference peaks.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    ndim: int
        Number of dimensions.
    verbose : bool
        If True, prints detailed logs during the process.

    Returns
    -------
    tuple[Transform, Transform]
        Tuple of forward and inverse transforms.
    """
    if verbose:
        click.echo(f"Estimating transform with settings: {affine_transform_settings}")
    # Detect dimensionality from peaks
    if ndim not in (2, 3):
        raise ValueError(f"Peaks must be 2D or 3D, got {ndim}D")

    # Create appropriate transform
    if affine_transform_settings.transform_type == "affine":
        transform = AffineTransform(dimensionality=ndim)
    elif affine_transform_settings.transform_type == "euclidean":
        transform = EuclideanTransform(dimensionality=ndim)
    elif affine_transform_settings.transform_type == "similarity":
        transform = SimilarityTransform(dimensionality=ndim)
    else:
        raise ValueError(f"Unknown transform type: {affine_transform_settings.transform_type}")

    # Fit transform
    transform.estimate(mov_peaks[matches[:, 0]], ref_peaks[matches[:, 1]])

    inv_transform = Transform(matrix=transform.inverse.params)
    fwd_transform = Transform(matrix=transform.params)

    return fwd_transform, inv_transform


def optimize_transform(
    transform: Transform,
    mov: da.Array,
    ref: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[Transform, float]:
    """
    Refine a transform by bead matching and evaluate registration quality.

    Applies the current transform to the moving volume, detects beads in both
    the registered moving and reference volumes, matches them, estimates a
    correction transform, and composes it with the input transform. Returns
    the better of the two (original vs corrected) based on overlap score.

    Parameters
    ----------
    transform : Transform
        Current transform to refine (maps moving -> reference space).
    mov : ArrayLike
        Original (unregistered) moving volume (Z, Y, X).
    ref : ArrayLike
        Reference volume (Z, Y, X).
    beads_match_settings : BeadsMatchSettings
        Settings controlling peak detection, matching, and filtering.
    affine_transform_settings : AffineTransformSettings
        Settings for the transform type (affine/euclidean/similarity).
    verbose : bool
        If True, prints quality scores before and after optimization.
    debug : bool
        If True, prints detailed intermediate results (peaks, matches, transforms).

    Returns
    -------
    tuple[Transform, float]
        The best transform and its overlap score.
        Returns (None, -1) if not enough peaks or matches are found.
    """
    mov_ants = ants.from_numpy(mov)
    ref_ants = ants.from_numpy(ref)

    # Step 1: Score the current transform by applying it and measuring peak overlap
    if debug:
        click.echo("Step 1: Scoring current transform (before bead matching)...")
    mov_reg_approx = transform.to_ants().apply_to_image(mov_ants, reference=ref_ants).numpy()
    mov_peaks, ref_peaks = peaks_from_beads(
        mov=mov_reg_approx,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=debug,
    )
    if mov_peaks is None or ref_peaks is None:
        return None, -1

    quality_score_approx = overlap_score(
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        radius=beads_match_settings.qc_settings.score_centroid_mask_radius,
        verbose=debug,
    )

    # Step 2: Match beads and estimate a correction transform
    if debug:
        click.echo("Step 2: Matching beads to estimate correction transform...")
    matches = matches_from_beads(
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        beads_match_settings=beads_match_settings,
        verbose=debug,
    )

    if len(matches) < 3:
        click.echo("Not enough matches found, returning the current transform")
        return None, -1

    fwd_transform, inv_transform = transform_from_matches(
        matches=matches,
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        affine_transform_settings=affine_transform_settings,
        ndim=mov.ndim,
        verbose=debug,
    )
    composed_transform = transform @ inv_transform

    # Step 3: Score the composed (corrected) transform
    if debug:
        click.echo("Step 3: Scoring composed transform (after bead matching)...")
    mov_reg_optimized = (
        composed_transform.to_ants().apply_to_image(mov_ants, reference=ref_ants).numpy()
    )
    mov_peaks_optimized, ref_peaks_optimized = peaks_from_beads(
        mov=mov_reg_optimized,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=debug,
    )

    quality_score_optimized = overlap_score(
        mov_peaks=mov_peaks_optimized,
        ref_peaks=ref_peaks_optimized,
        radius=beads_match_settings.qc_settings.score_centroid_mask_radius,
        verbose=debug,
    )
    if debug:
        click.echo(f"Bead matches: {matches}")
        click.echo(f"Forward transform: {fwd_transform}")
        click.echo(f"Inverse transform: {inv_transform}")
        click.echo(f"Composed transform: {composed_transform}")

    if verbose:
        click.echo(f"Quality score before beads matching: {quality_score_approx}")
        click.echo(f"Quality score after beads matching: {quality_score_optimized}")

    if quality_score_optimized >= quality_score_approx:
        return composed_transform, quality_score_optimized
    else:
        return transform, quality_score_approx


class BeadNodeDetector:
    """Detects bead centroids as local-maxima peaks -- today's only node source."""

    def __init__(self, settings: DetectPeaksSettings):
        self.settings = settings

    def detect(self, array: ArrayLike) -> ArrayLike:
        return detect_peaks(
            np.asarray(array),
            block_size=self.settings.block_size,
            threshold_abs=self.settings.threshold_abs,
            nms_distance=self.settings.nms_distance,
            min_distance=self.settings.min_distance,
        )


class NodeGraphEstimator:
    """TransformEstimator over matched point correspondences.

    Composes a `NodeDetector` (beads today; segmentation centroids or other node
    sources later) with graph matching and transform fitting.

    One pass warps `mov` by the current guess (`Transform.apply`), detects nodes in the
    warped frame, matches, fits the residual correction and composes it back
    (`correction @ seed`: seed first, then correction). Nodes are detected in the warped
    frame, so each pass sees a better-aligned image than the last; `iterations > 1` with
    a `score_fn` repeats the pass from the previous result and returns the best-scoring
    transform (never a later, worse one). Without a `score_fn` the last pass is returned.
    """

    def __init__(
        self,
        mov_detector: NodeDetector,
        ref_detector: NodeDetector,
        beads_match_settings: BeadsMatchSettings,
        affine_transform_settings: AffineTransformSettings,
        iterations: int = 1,
        score_fn: ScoreFn | None = None,
        matcher: NodeMatcher | None = None,
    ):
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self.mov_detector = mov_detector
        self.ref_detector = ref_detector
        self.beads_match_settings = beads_match_settings
        self.affine_transform_settings = affine_transform_settings
        self.iterations = iterations
        self.score_fn = score_fn
        self.matcher = matcher or (
            lambda mov_nodes, ref_nodes: matches_from_beads(
                mov_nodes, ref_nodes, beads_match_settings
            )
        )

    @classmethod
    def from_beads_settings(
        cls,
        beads_match_settings: BeadsMatchSettings,
        affine_transform_settings: AffineTransformSettings,
        iterations: int | None = None,
        score_fn: ScoreFn | None = None,
    ) -> NodeGraphEstimator:
        """Bead-peak detection on both sides with the settings' matcher.

        `iterations` defaults to `beads_match_settings.qc_settings.iterations`. This is
        the single-arm estimator; `engine.build_beads_estimator` composes the vote-ICP
        mode, seed correction and the spectral arm around it.
        """
        return cls(
            mov_detector=BeadNodeDetector(beads_match_settings.source_peaks_settings),
            ref_detector=BeadNodeDetector(beads_match_settings.target_peaks_settings),
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
            iterations=(
                beads_match_settings.qc_settings.iterations
                if iterations is None
                else iterations
            ),
            score_fn=score_fn,
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        current = seed
        best: Transform | None = None
        best_score = -np.inf
        for _ in range(self.iterations):
            try:
                current = self._single_pass(mov, ref, current)
            except EstimationError:
                # A later pass that cannot match (e.g. the previous pass drifted) must not
                # throw away an earlier usable result.
                if best is not None:
                    break
                raise
            if self.score_fn is None:
                best = current
                continue
            score = self.score_fn(current, mov, ref)
            if np.isfinite(score) and score > best_score:
                best, best_score = current, score
        if best is None:
            raise EstimationError(
                f"no finite score in {self.iterations} iteration(s): nodes not detectable "
                "after warping"
            )
        return best

    def _single_pass(
        self, mov: np.ndarray, ref: np.ndarray, seed: Transform | None
    ) -> Transform:
        mov_for_detection = seed.apply(mov, reference=ref) if seed is not None else mov
        mov_nodes = np.asarray(self.mov_detector.detect(mov_for_detection))
        ref_nodes = np.asarray(self.ref_detector.detect(ref))
        if len(mov_nodes) < 3 or len(ref_nodes) < 3:
            raise EstimationError(
                f"too few nodes to fit a transform: {len(mov_nodes)} moving, "
                f"{len(ref_nodes)} reference (need >= 3 each)"
            )
        matches = np.asarray(self.matcher(mov_nodes, ref_nodes))
        if matches.ndim != 2 or len(matches) < 3:
            raise EstimationError(
                f"too few matches to fit a transform: {len(matches)} from "
                f"{len(mov_nodes)} x {len(ref_nodes)} nodes (need >= 3)"
            )
        correction, _inv_correction = transform_from_matches(
            matches,
            mov_nodes,
            ref_nodes,
            self.affine_transform_settings,
            ndim=mov.ndim,
        )
        if not np.all(np.isfinite(correction.matrix)):
            raise EstimationError(
                f"degenerate fit from {len(matches)} matches (non-finite matrix)"
            )
        return correction @ seed if seed is not None else correction
