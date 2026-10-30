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
4. **Iterative refinement** (`NodeGraphEstimator.estimate`): Compose the approximate
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

import click
import dask.array as da
import numpy as np

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from biahub.characterize_psf import detect_peaks
from biahub.core.nodes.costs import MatchingTooLargeError
from biahub.core.nodes.graph import Graph
from biahub.core.nodes.matching import GraphMatcher
from biahub.core.nodes.point_set import vote_drift, vote_icp_register
from biahub.core.transform import Transform
from biahub.registration.estimators import (
    EstimationError,
    NodeDetector,
    NodeMatcher,
    ScoreFn,
    fit_pairs,
)
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    SeedCorrectionSettings,
    VoteIcpSettings,
)


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
        Reference peaks with a moving peak within `radius`, over the smaller peak count:
        usually in [0, 1], but above 1 when the reference has more peaks than the moving
        volume (e.g. 3.0 for 1 moving and 3 reference peaks). np.nan if either set is empty.
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
) -> tuple[ArrayLike, ArrayLike] | None:
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
    """Return the (forward, inverse) transform fitted to matched peaks (`Transform.fit`).

    `matches` are (moving, reference) index pairs into the (N, D) peaks; the transform
    type is the settings'. `ndim` is not needed (the peaks give it) and kept for callers.
    Raises ValueError when the fit fails.
    """
    if verbose:
        click.echo(f"Estimating transform with settings: {affine_transform_settings}")
    fwd_transform = Transform.fit(
        np.asarray(mov_peaks)[matches[:, 0]],
        np.asarray(ref_peaks)[matches[:, 1]],
        affine_transform_settings.transform_type,
    )
    return fwd_transform, fwd_transform.invert()


# How the evaluation pairs this method's landmarks: (radius in µm, axes it is measured on).
# Beads are points: a partner within 1 µm in 3D.
POINT_MATCHING = (1.0, "zyx")


def detect_beads(
    volume: ArrayLike,
    voxel_um: tuple,
    threshold_abs: float,
    block_size: tuple = (8, 8, 8),
    nms_distance: int = 16,
    refine_halfwidth: tuple = (2, 4, 4),
) -> np.ndarray:
    """Return bead centres (N x 3 voxels, z y x) to a fraction of a voxel: refined peaks.

    Peaks sit on whole voxels; each is refined to the intensity centroid of a small box
    around it (background-subtracted), which a residual of a voxel or less needs.
    """
    vol = np.asarray(volume, dtype=np.float32)
    peaks = np.asarray(
        detect_peaks(
            vol,
            block_size=block_size,
            threshold_abs=threshold_abs,
            nms_distance=nms_distance,
            min_distance=0,
        ),
        dtype=int,
    ).reshape(-1, 3)
    background = float(np.median(vol))
    refined = []
    for z, y, x in peaks:
        box = tuple(
            slice(max(c - h, 0), min(c + h + 1, n))
            for c, h, n in zip((z, y, x), refine_halfwidth, vol.shape, strict=True)
        )
        weights = np.clip(vol[box] - background, 0, None)
        if weights.sum() <= 0:
            refined.append([z, y, x])
            continue
        grids = np.meshgrid(*[np.arange(s.start, s.stop) for s in box], indexing="ij")
        refined.append([(g * weights).sum() / weights.sum() for g in grids])
    return np.array(refined, dtype=float).reshape(-1, 3)


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

    Composes a `NodeDetector` per channel (beads, nuclei; segmentation centroids later)
    with graph matching and a fit: the landmark-based methods.

    One pass warps `mov` by the current guess (`Transform.apply`), detects nodes in the
    warped frame, matches, fits the residual correction and composes it back
    (`correction @ seed`: seed first, then correction). Nodes are detected in the warped
    frame, so each pass sees a better-aligned image than the last; `iterations > 1` with
    a `score_fn` repeats the pass from the previous result and returns the best-scoring
    transform (never a later, worse one). Without a `score_fn` the last pass is returned.

    As in the legacy pipeline, a pass keeps its input when the refinement scores lower
    (ties go to the refinement), and a score of exactly 1.0 ends the passes early.
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
        fit=None,
    ):
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self.mov_detector = mov_detector
        self.ref_detector = ref_detector
        self.beads_match_settings = beads_match_settings
        self.affine_transform_settings = affine_transform_settings
        self.iterations = iterations
        self.score_fn = score_fn
        # (matches, mov_nodes, ref_nodes) -> the correction; default: the transform type's
        # full fit (transform_from_matches), as beads use
        self.fit = fit
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
        self,
        mov: ArrayLike,
        ref: ArrayLike,
        seed: Transform | None = None,
        competitor: Transform | None = None,
    ) -> Transform:
        """Refine `seed` over the configured passes; return the best-scoring result.

        `competitor`, when given, is refined too on the first pass and replaces the
        seed's result only if it scores strictly higher; later passes continue from the
        winner. This is the legacy `user_transform` competition, used by propagation to
        let the input seed compete with the previous timepoint's result.
        """
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        if self.score_fn is None:
            if competitor is not None:
                raise ValueError("a competitor seed needs a score_fn to compete")
            current = seed
            for _ in range(self.iterations):
                current = self._single_pass(mov, ref, current)
            return current
        current, current_score = seed, None
        best: Transform | None = None
        best_score = -np.inf
        for i in range(self.iterations):
            candidates, error = [], None
            try:
                candidates.append(self._pass_keeping_input(mov, ref, current, current_score))
            except EstimationError as e:
                error = e
            if i == 0 and competitor is not None:
                try:
                    candidates.append(self._pass_keeping_input(mov, ref, competitor, None))
                except EstimationError:
                    pass
            if not candidates:
                # A later pass that cannot match (e.g. the previous pass drifted) must not
                # throw away an earlier usable result.
                if best is not None:
                    break
                raise error
            current, current_score = candidates[0]
            for transform, score in candidates[1:]:
                if score > current_score:  # ties: the seed's result
                    current, current_score = transform, score
            if current_score > best_score:
                best, best_score = current, current_score
            if current_score == 1.0:  # legacy stops on exactly 1 (overlap can exceed 1)
                break
        if best is None or not np.isfinite(best_score):
            raise EstimationError(
                f"no finite score in {self.iterations} iteration(s): nodes not detectable "
                "after warping"
            )
        return best

    def _pass_keeping_input(
        self, mov, ref, start: Transform | None, start_score: float | None
    ) -> tuple[Transform, float]:
        """One pass from `start`; keep `start` when the refinement scores lower (ties: refinement)."""
        refined = self._single_pass(mov, ref, start)
        refined_score = self._finite(self.score_fn(refined, mov, ref))
        if start is None:
            start = Transform.identity(mov.ndim)
        if start_score is None:
            start_score = self._finite(self.score_fn(start, mov, ref))
        if refined_score >= start_score:
            return refined, refined_score
        return start, start_score

    @staticmethod
    def _finite(score: float | None) -> float:
        return float(score) if score is not None and np.isfinite(score) else -np.inf

    def _single_pass(
        self, mov: np.ndarray, ref: np.ndarray, seed: Transform | None
    ) -> Transform:
        mov_for_detection = seed.apply(mov, reference=ref) if seed is not None else mov
        if seed is not None and getattr(self.mov_detector, "missing_as_nan", False):
            # the warp's padding is no data, not dark: for a detector that leaves NaN out
            mov_for_detection = np.asarray(mov_for_detection, dtype=np.float32).copy()
            mov_for_detection[~seed.coverage(mov.shape, ref.shape)] = np.nan
        mov_nodes = np.asarray(self.mov_detector.detect(mov_for_detection))
        ref_nodes = np.asarray(self.ref_detector.detect(ref))
        if len(mov_nodes) < 3 or len(ref_nodes) < 3:
            raise EstimationError(
                f"too few nodes to fit a transform: {len(mov_nodes)} moving, "
                f"{len(ref_nodes)} reference (need >= 3 each)"
            )
        try:
            matches = np.asarray(self.matcher(mov_nodes, ref_nodes))
        except MatchingTooLargeError as e:  # this matcher can't run here; others may
            raise EstimationError(str(e)) from e
        if matches.ndim != 2 or len(matches) < 3:
            raise EstimationError(
                f"too few matches to fit a transform: {len(matches)} from "
                f"{len(mov_nodes)} x {len(ref_nodes)} nodes (need >= 3)"
            )
        if self.fit is not None:
            correction = self.fit(matches, mov_nodes, ref_nodes)
        else:
            correction = Transform(
                fit_pairs(
                    mov_nodes[matches[:, 0]],
                    ref_nodes[matches[:, 1]],
                    self.affine_transform_settings.transform_type,
                )
            )
        if not np.all(np.isfinite(correction.matrix)):
            raise EstimationError(
                f"degenerate fit from {len(matches)} matches (non-finite matrix)"
            )
        return correction @ seed if seed is not None else correction


class VoteIcpEstimator:
    """TransformEstimator over peak clouds by iterated displacement voting.

    Reach stage: dense detection on the raw moving volume (`vote_peaks_settings`), voted
    ICP from the seed with a wide, shrinking capture radius. Precision stage: re-run at
    short range from that result with the pipeline's own moving detection, kept only on a
    strict score win. Raises `EstimationError` when voting abstains before any fit.
    """

    def __init__(
        self,
        dense_detector: NodeDetector,
        precise_detector: NodeDetector,
        ref_detector: NodeDetector,
        settings: VoteIcpSettings,
        score_fn: ScoreFn,
        transform_type: str = "affine",
    ):
        self.transform_type = transform_type
        self.dense_detector = dense_detector
        self.precise_detector = precise_detector
        self.ref_detector = ref_detector
        self.settings = settings
        self.score_fn = score_fn

    @classmethod
    def from_beads_settings(
        cls,
        beads_match_settings: BeadsMatchSettings,
        score_fn: ScoreFn,
        transform_type: str = "affine",
    ) -> VoteIcpEstimator:
        settings = beads_match_settings.vote_icp_settings
        return cls(
            transform_type=transform_type,
            dense_detector=BeadNodeDetector(settings.vote_peaks_settings),
            precise_detector=BeadNodeDetector(beads_match_settings.source_peaks_settings),
            ref_detector=BeadNodeDetector(beads_match_settings.target_peaks_settings),
            settings=settings,
            score_fn=score_fn,
        )

    def _register(self, mov_peaks, ref_peaks, inverse_seed, initial_radius):
        s = self.settings
        return _vote_icp_register(
            mov_peaks=mov_peaks,
            ref_peaks=ref_peaks,
            initial_transform=inverse_seed,
            initial_capture_radius=initial_radius,
            min_capture_radius=s.min_capture_radius,
            radius_decay=s.radius_decay,
            cluster_radius=s.cluster_radius,
            min_votes=s.min_votes,
            max_iterations=s.max_iterations,
            convergence_translation=s.convergence_translation,
            transform_type=self.transform_type,
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        ndim = mov.ndim
        inverse_seed = (seed.invert() if seed is not None else Transform.identity(ndim)).matrix
        ref_peaks = np.asarray(self.ref_detector.detect(ref))
        dense_peaks = np.asarray(self.dense_detector.detect(mov))
        if min(len(dense_peaks), len(ref_peaks)) < self.settings.min_votes:
            raise EstimationError(
                f"vote_icp: {len(dense_peaks)} dense moving / {len(ref_peaks)} reference "
                f"peaks (need >= {self.settings.min_votes})"
            )
        inverse, info = self._register(
            dense_peaks, ref_peaks, inverse_seed, self.settings.initial_capture_radius
        )
        if inverse is None:
            raise EstimationError(
                f"vote_icp: voting abstained after {info['iterations']} iteration(s)"
            )
        best = Transform(inverse, transform_type=self.transform_type).invert()
        best_score = self.score_fn(best, mov, ref)

        precise_peaks = np.asarray(self.precise_detector.detect(mov))
        if len(precise_peaks) >= self.settings.min_votes:
            precise_inverse, _info = self._register(
                precise_peaks, ref_peaks, inverse, self.settings.min_capture_radius
            )
            if precise_inverse is not None:
                precise = Transform(
                    precise_inverse, transform_type=best.transform_type
                ).invert()
                precise_score = self.score_fn(precise, mov, ref)
                # a finite refinement beats a NaN coarse score (a NaN compares false)
                if np.isfinite(precise_score) and (
                    not np.isfinite(best_score) or precise_score > best_score
                ):
                    best = precise
        return best


class VoteSeedCorrection:
    """A stage that corrects a seed by bead displacement voting, for a chain's first slot.

    Warps `mov` by the seed, detects beads densely and votes for the residual drift;
    "voteseed" composes the densest cluster's mean displacement into the seed, "votefit"
    also fits an affine on the cluster's pairs. Every candidate competes against the
    unchanged seed on the median nearest-neighbour distance of the pipeline's own
    detection, so a bad vote leaves the seed unchanged. Returns the (possibly unchanged)
    seed; without a seed, the identity.
    """

    def __init__(
        self,
        settings: SeedCorrectionSettings,
        mov_detector: NodeDetector,
        ref_detector: NodeDetector,
    ):
        self.settings = settings
        self.mov_detector = mov_detector
        self.ref_detector = ref_detector
        self.dense_detector = BeadNodeDetector(settings.vote_peaks_settings)
        self.last_note: str = ""

    @classmethod
    def from_beads_settings(
        cls, beads_match_settings: BeadsMatchSettings
    ) -> VoteSeedCorrection:
        return cls(
            settings=beads_match_settings.seed_correction_settings,
            mov_detector=BeadNodeDetector(beads_match_settings.source_peaks_settings),
            ref_detector=BeadNodeDetector(beads_match_settings.target_peaks_settings),
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        seed = seed if seed is not None else Transform.identity(mov.ndim)
        settings = self.settings
        ref_peaks = np.asarray(self.ref_detector.detect(ref))
        if len(ref_peaks) < settings.min_votes:
            self.last_note = f"only {len(ref_peaks)} ref peaks; seed unchanged"
            return seed
        ref_tree = cKDTree(ref_peaks)

        def nn_median(transform: Transform) -> float:
            peaks = np.asarray(self.mov_detector.detect(transform.apply(mov, reference=ref)))
            if len(peaks) == 0:
                return np.inf
            return float(np.median(ref_tree.query(peaks)[0]))

        warped = seed.apply(mov, reference=ref)
        dense_peaks = np.asarray(self.dense_detector.detect(warped))
        drift, pairs, n_votes = vote_drift(
            dense_peaks,
            ref_peaks,
            settings.capture_radius,
            settings.cluster_radius,
            settings.min_votes,
        )
        if drift is None:
            self.last_note = (
                f"only {len(dense_peaks)} dense peaks / too few votes; seed unchanged"
            )
            return seed
        # Work in the inverse convention the vote was measured in: the seed maps reference
        # coordinates to moving sample coordinates, so undoing an image drift of +d in the
        # reference frame composes as inverse @ T(-d). Both signs compete with the unchanged seed.
        inverse = seed.invert().matrix
        candidates = {
            "keep": seed,
            "minus": Transform(inverse @ Transform.from_translation(-drift).matrix).invert(),
            "plus": Transform(inverse @ Transform.from_translation(drift).matrix).invert(),
        }
        nn = {name: nn_median(t) for name, t in candidates.items()}
        best = min(nn, key=nn.get)
        corrected = candidates[best]
        note = f"drift={np.round(drift, 1).tolist()} votes={n_votes} pick={best}"
        if settings.mode == "votefit" and len(pairs) >= 4:
            # The cluster's votes are correspondences (warped moving peak p, reference peak
            # q) in the uncorrected seed's warped frame; re-warping should put p at q, i.e.
            # sample point C(q) = p, so the fit composes onto the seed as inverse @ C.
            p = np.asarray([a for a, _ in pairs], dtype=float)
            q = np.asarray([b for _, b in pairs], dtype=float)
            correction = (
                Transform.fit(q, p).matrix
                if len(pairs) >= 6
                else Transform.from_translation((p - q).mean(axis=0)).matrix
            )
            fitted = Transform(inverse @ correction).invert()
            nn_fitted = nn_median(fitted)
            if np.isfinite(nn_fitted) and nn_fitted < nn[best]:
                corrected = fitted
                note += f"; fit accepted n={len(pairs)} nn {nn[best]:.1f}->{nn_fitted:.1f}"
            else:
                note += f"; fit rejected n={len(pairs)} nn {nn[best]:.1f}->{nn_fitted:.1f}"
        self.last_note = note
        return corrected


def _vote_icp_register(**kwargs):
    """`vote_icp_register`, with a fit that fails as a failed estimate."""
    try:
        return vote_icp_register(**kwargs)
    except ValueError as error:
        raise EstimationError(str(error)) from error
