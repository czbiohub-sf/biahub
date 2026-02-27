from pathlib import Path
from typing import Literal, Optional

import ants
import click
import dask.array as da
import numpy as np

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform

from biahub.characterize_psf import detect_peaks

from biahub.cli.utils import (
    _check_nan_n_zeros,
)
from biahub.core.graph_matching import Graph, GraphMatcher
from biahub.core.transform import Transform
from biahub.registration.utils import load_transforms
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
)


def estimate_with_propagation(
    mov_tzyx: da.Array,
    ref_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    mode: Literal["registration", "stabilization"] = "registration",
) -> Transform:
    initial_transform = affine_transform_settings.approx_transform
    T, _, _, _ = mov_tzyx.shape
    for t in range(T):
        if np.sum(mov_tzyx[t]) == 0 or np.sum(ref_tzyx[t]) == 0:
            click.echo(f"Timepoint {t} has no data, skipping")
        else:
            approx_transform = estimate(
                t=t,
                mov=mov_tzyx[t],
                ref=ref_tzyx[t],
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                verbose=verbose,
                output_folder_path=output_folder_path,
                mode=mode,
                user_transform=initial_transform,
            )

            if approx_transform is not None:
                affine_transform_settings.approx_transform = approx_transform.to_list()
            else:
                affine_transform_settings.approx_transform = initial_transform

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
        click.echo('Detecting beads in moving dataset')
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
        click.echo('Detecting beads in reference dataset')
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
        click.echo(f'Total of peaks in moving dataset: {len(mov_peaks)}')
        click.echo(f'Total of peaks in reference dataset: {len(ref_peaks)}')

    if len(mov_peaks) < 2 or len(ref_peaks) < 2:
        click.echo('Not enough beads detected')
        return
    if mask_path is not None:
        print("Filtering peaks with mask")
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
    Get matches from beads using the hungarian algorithm.

    Parameters
    ----------
    mov_peaks : ArrayLike
        (n, 2) array of moving peaks.
    ref_peaks : ArrayLike
        (m, 2) array of reference peaks.
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    verbose : bool
        If True, prints detailed logs during the process.

    Returns
    -------
    ArrayLike
        (n, 2) array of matches.
    """
    if verbose:
        click.echo(f'Getting matches from beads with settings: {beads_match_settings}')

    if beads_match_settings.algorithm == 'match_descriptor':
        mov_graph = Graph.from_nodes(mov_peaks)
        ref_graph = Graph.from_nodes(ref_peaks)

        match_descriptor_settings = beads_match_settings.match_descriptor_settings
        matcher = GraphMatcher(
            algorithm='descriptor',
            cross_check=match_descriptor_settings.cross_check,
            max_ratio=match_descriptor_settings.max_ratio,
            metric=match_descriptor_settings.distance_metric,
            verbose=verbose,
        )

        matches = matcher.match(mov_graph, ref_graph)

    elif beads_match_settings.algorithm == 'hungarian':
        hungarian_match_settings = beads_match_settings.hungarian_match_settings
        mov_graph = Graph.from_nodes(
            mov_peaks, mode='knn', k=hungarian_match_settings.edge_graph_settings.k
        )
        ref_graph = Graph.from_nodes(
            ref_peaks, mode='knn', k=hungarian_match_settings.edge_graph_settings.k
        )

        matcher = GraphMatcher(
            algorithm='hungarian',
            weights=hungarian_match_settings.cost_matrix_settings.weights,
            cost_threshold=hungarian_match_settings.cost_threshold,
            cross_check=hungarian_match_settings.cross_check,
            max_ratio=hungarian_match_settings.max_ratio,
            verbose=verbose,
        )

        matches = matcher.match(mov_graph, ref_graph)

    # Filter as part of the pipeline
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
        click.echo(f'Total of matches: {len(matches)}')

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
    Estimate the affine transformation matrix between source and target channels
    based on detected bead matches at a specific timepoint.

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
    if affine_transform_settings.transform_type == 'affine':
        transform = AffineTransform(dimensionality=ndim)
    elif affine_transform_settings.transform_type == 'euclidean':
        transform = EuclideanTransform(dimensionality=ndim)
    elif affine_transform_settings.transform_type == 'similarity':
        transform = SimilarityTransform(dimensionality=ndim)
    else:
        raise ValueError(f'Unknown transform type: {affine_transform_settings.transform_type}')

    # Fit transform
    transform.estimate(mov_peaks[matches[:, 0]], ref_peaks[matches[:, 1]])

    inv_transform = Transform(matrix=transform.inverse.params)
    fwd_transform = Transform(matrix=transform.params)

    return fwd_transform, inv_transform



def registration_beads_score(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    radius: int = 6,
    verbose: bool = False,
):
    """
    Compute the score for the beads registration based on the overlap fraction and IoU.
    between detected bead peaks from LF (ref) and LS (mov) channels.

    Args:
        ref_peaks: (N_ref, 3) array of LF bead coordinates (z, y, x)
        mov_peaks: (N_mov, 3) array of LS bead coordinates (z, y, x)
        matches:   (M, 2) matched indices (optional)
        radius:    spherical neighborhood radius (voxels)
        ref_shape: optional 3D shape for visualization masks

    Returns:
       score: float

    """

    if len(mov_peaks) == 0 or len(ref_peaks) == 0:
        print("No peaks found, returning nan metrics")
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


def optimize_transform(
    transform: Transform,
    mov: da.Array,
    ref: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[Transform, float, float]:

    mov_ants = ants.from_numpy(mov)
    ref_ants = ants.from_numpy(ref)

    if debug:
        click.echo("Checking quality score before beads matching")
    mov_reg_approx = transform.to_ants().apply_to_image(mov_ants, reference=ref_ants).numpy()
    mov_peaks, ref_peaks = peaks_from_beads(
        mov=mov_reg_approx,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=debug,
    )
    if (len(mov_peaks) is None) or (len(ref_peaks) is None):
        return None, -1

    quality_score_approx = registration_beads_score(
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        radius=beads_match_settings.qc_settings.score_centroid_mask_radius,
        verbose=debug,
    )

    if debug:
        click.echo("Optimizing transform with beads matching")
    matches = matches_from_beads(
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        beads_match_settings=beads_match_settings,
        verbose=debug,
    )

    if len(matches) < 3:
        click.echo('Not enough matches found, returning the current transform')
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

    if debug:
        click.echo("Checking quality score after beads matching")
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

    quality_score_optimized = registration_beads_score(
        mov_peaks=mov_peaks_optimized,
        ref_peaks=ref_peaks_optimized,
        radius=beads_match_settings.qc_settings.score_centroid_mask_radius,
        verbose=debug,
    )

    if verbose:
        click.echo(f"Quality score before beads matching: {quality_score_approx}")
        click.echo(f"Quality score after beads matching: {quality_score_optimized}")
    if debug:
        click.echo(f'Bead matches: {matches}')
        click.echo(f"Forward transform: {fwd_transform}")
        click.echo(f"Inverse transform: {inv_transform}")
        click.echo(f"Composed transform: {composed_transform}")

    if quality_score_optimized >= quality_score_approx:
        return composed_transform, quality_score_optimized
    else:
        return transform, quality_score_approx


def estimate(
    mov: da.Array,
    ref: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_filepath: Path = None,
    user_transform: Transform = None,
    debug: bool = False,
    t : Optional[int] = None,
    fov: Optional[str] = None,
) -> tuple[Transform, float]:
    """
    Estimate the affine transformation between source and target channels
    based on detected bead matches.

    Works for both 2D (Y, X) and 3D (Z, Y, X) arrays.

    Parameters
    ----------
    mov : da.Array
        Moving channel data.
    ref : da.Array
        Reference channel data.
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, prints detailed logs during the process.
    output_filepath : Path
        Path to save the output.
    qc_score_threshold : float
        Threshold score for the quality control.
    t : int
        Timepoint index.
    fov : str
        FOV name.
    Returns
    -------
    Transform
        The estimated transformation.

    Raises
    ------
    ValueError
        If the source or target channel data is missing.
    """

    if _check_nan_n_zeros(mov) or _check_nan_n_zeros(ref):
        click.echo('Beads data is missing')
        return

    initial_transform = Transform(
        matrix=np.asarray(affine_transform_settings.approx_transform)
    )
    transform = initial_transform

    current_iterations = 0
    qc_iterations = beads_match_settings.qc_settings.iterations
    transform_iter_dict = {}

    while current_iterations < qc_iterations:
        click.echo(f"Current iteration: {current_iterations}")
        click.echo("Optimizing current transform")
        optimized_transform, quality_score_optimized = optimize_transform(
            transform=transform,
            mov=mov,
            ref=ref,
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
            verbose=verbose,
            debug=debug,
        )
        transform_iter_dict[current_iterations] = {
            "transform": optimized_transform,
            "quality_score": quality_score_optimized,
        }
        if quality_score_optimized == 1:
            break
        transform = optimized_transform

        if user_transform is not None and current_iterations == 0:
            click.echo("Optimizing user transform:")
            user_transform = Transform(matrix=np.asarray(user_transform))
            optimized_transform_user, quality_score_optimized_user = optimize_transform(
                transform=user_transform,
                mov=mov,
                ref=ref,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                verbose=verbose,
                debug=debug,
            )

            if quality_score_optimized < quality_score_optimized_user:

                transform_iter_dict[current_iterations] = {
                    "transform": optimized_transform_user,
                    "quality_score": quality_score_optimized_user,
                }
                if quality_score_optimized_user == 1:
                    break
                transform = optimized_transform_user

        if transform is None:
            break
        current_iterations += 1

    # get highest quality score
    best_quality_score = max(transform_iter_dict.values(), key=lambda x: x["quality_score"])
    best_transform = best_quality_score["transform"]

    if best_transform is None:
        best_transform = initial_transform
    if verbose:
        click.echo(f"Best transform: {best_transform}")
        click.echo(f"Best quality score: {best_quality_score['quality_score']}")
    if output_filepath:
        click.echo(f"Saving transform to {output_filepath}")
        np.save(output_filepath, best_transform.to_list())

    return best_transform
