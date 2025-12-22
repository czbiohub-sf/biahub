from datetime import datetime
from pathlib import Path
from typing import Literal

import ants
import click
import dask.array as da
import numpy as np
import submitit

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform

from biahub.characterize_psf import detect_peaks
from biahub.cli.parsing import (
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import (
    _check_nan_n_zeros,
    estimate_resources,
)
from biahub.core.graph_matching import Graph, GraphMatcher
from biahub.core.transform import Transform
from biahub.registration.utils import load_transforms
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
)


def estimate_tczyx(
    mov_tczyx: da.Array,
    ref_tczyx: da.Array,
    mov_channel_index: int,
    ref_channel_index: int = None,
    beads_match_settings: BeadsMatchSettings = None,
    affine_transform_settings: AffineTransformSettings = None,
    verbose: bool = False,
    cluster: bool = False,
    sbatch_filepath: Path = None,
    output_folder_path: Path = None,
    mode: Literal["registration", "stabilization"] = "registration",
) -> list[Transform]:
    """
    Perform beads-based temporal registration of 4D data using affine transformations.

    This function calculates timepoint-specific affine transformations to align a source channel
    to a target channel in 4D (T, Z, Y, X) data. It validates, smooths, and interpolates transformations
    across timepoints for consistent registration.

    Parameters
    ----------
    mov_tczyx : da.Array
       4D array (T, C, Z, Y, X) of the moving channel (Dask array).
    ref_tczyx : da.Array
       4D array (T, C, Z, Y, X) of the target channel (Dask array).
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, prints detailed logs of the registration process.
    cluster : bool
        If True, uses the cluster.
    sbatch_filepath : Path
        Path to the sbatch file.
    output_folder_path : Path
        Path to the output folder.
    quality_control : bool
        If True, performs quality control.
    threshold_score : float
        Threshold score for the quality control.
    grid_search : bool
        If True, performs grid search.
    mode : Literal["registration", "stabilization"]
        Mode to perform the registration(between two channels) or stabilization (between a channel over time).
    Returns
    -------
    list[Transform]
        List of affine transformation matrices (4x4), one for each timepoint.
        Invalid or missing transformations are interpolated.

    Notes
    -----
    Each timepoint is processed in parallel using submitit executor.
    Use verbose=True for detailed logging during registration. The verbose output will be saved at the same level as the output zarr.
    """
    mov_tzyx = mov_tczyx[:, mov_channel_index]
    if mode == "stabilization":
        ref_tzyx = mov_tzyx
    elif mode == "registration":
        ref_tzyx = ref_tczyx[:, ref_channel_index]

    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    T, _, _, _ = mov_tzyx.shape
    if affine_transform_settings.use_prev_t_transform:
        estimate_with_propagation(
            mov_tzyx=mov_tzyx,
            ref_tzyx=ref_tzyx,
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
            verbose=verbose,
            output_folder_path=output_transforms_path,
            mode=mode,
        )
    else:
        estimate_independently(
            mov_tzyx=mov_tzyx,
            ref_tzyx=ref_tzyx,
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
            verbose=verbose,
            output_folder_path=output_transforms_path,
            cluster=cluster,
            sbatch_filepath=sbatch_filepath,
            mode=mode,
        )

    transforms = load_transforms(output_transforms_path, T, verbose)

    return transforms


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
        if mode == "stabilization" and t == 0:
            continue
        if np.sum(mov_tzyx[t]) == 0 or np.sum(ref_tzyx[t]) == 0:
            click.echo(f"Timepoint {t} has no data, skipping")
        else:
            click.echo(f"Timepoint {t} has data, estimating transform")
            approx_transform = estimate_tzyx(
                t_idx=t,
                mov_tzyx=mov_tzyx,
                ref_tzyx=ref_tzyx,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                verbose=verbose,
                output_folder_path=output_folder_path,
                mode=mode,
            )

            if approx_transform is not None:
                print(f"Using approx transform for timepoint {t+1}: {approx_transform}")
                affine_transform_settings.approx_transform = approx_transform.to_list()
            else:
                print(f"Using initial transform for timepoint {t+1}: {initial_transform}")
                affine_transform_settings.approx_transform = initial_transform


def estimate_independently(
    mov_tzyx: da.Array,
    ref_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    cluster: str = 'local',
    sbatch_filepath: Path = None,
    mode: Literal["registration", "stabilization"] = "registration",
) -> Transform:
    """
    Calculate the affine transformation matrix between source and target channels
    based on detected bead matches at a specific timepoint.

    This function detects beads in both source and target datasets, matches them,
    and computes an affine transformation to align the two channels. It applies
    the transformation to the source channel and returns the transformed channel.
    """
    T, Z, Y, X = mov_tzyx.shape
    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, 2, Z, Y, X), ram_multiplier=5, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_registration",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 30,
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)
    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")

    # Submit jobs
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for t in range(T):
            job = executor.submit(
                estimate_tzyx,
                t_idx=t,
                mov_tzyx=mov_tzyx,
                ref_tzyx=ref_tzyx,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                verbose=verbose,
                output_folder_path=output_folder_path,
                mode=mode,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    wait_for_jobs_to_finish(jobs)


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
        print(f"Descriptor: {len(matches)} matches")

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
        print(f"Hungarian: {len(matches)} matches")

    # Filter as part of the pipeline
    matches = matcher.filter_matches(
        matches,
        mov_graph,
        ref_graph,
        angle_threshold=beads_match_settings.filter_angle_threshold,
        min_distance_quantile=beads_match_settings.filter_min_distance_threshold,
        max_distance_quantile=beads_match_settings.filter_max_distance_threshold,
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


def estimate_tzyx(
    t_idx: int,
    mov_tzyx: da.Array,
    ref_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    mode: Literal["registration", "stabilization"] = "registration",
) -> Transform:
    """
    Calculate the affine transformation matrix between source and target channels
    based on detected bead matches at a specific timepoint.

    This function detects beads in both source and target datasets, matches them,
    and computes an affine transformation to align the two channels. It applies
    various filtering steps, including angle-based filtering, to improve match quality.

    Parameters
    ----------
    t_idx : int
        Timepoint index to process.
    mov_tzyx : da.Array
       4D array (T, Z, Y, X) of the moving channel (Dask array).
    ref_tzyx : da.Array, optional
       4D array (T, Z, Y, X) of the reference channel (Dask array). If not provided, the first timepoint of the moving channel will be used.
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, prints detailed logs during the process.
    slurm : bool
        If True, uses SLURM for parallel processing.
    output_folder_path : Path
        Path to save the output.

    Returns
    -------
    list | None
        A 4x4 affine transformation matrix as a nested list if successful,
        or None if no valid transformation could be calculated.

    Notes
    -----
    Uses ANTsPy for initial transformation application and bead detection.
    Peaks (beads) are detected using a block-based algorithm with thresholds for source and target datasets.
    Bead matches are filtered based on distance and angular deviation from the dominant direction.
    If fewer than three matches are found after filtering, the function returns None.
    """
    click.echo(f'Processing timepoint: {t_idx}')

    (T, Z, Y, X) = mov_tzyx.shape

    if mode == "stabilization":
        click.echo("Performing stabilization, aka registration over time in the same file.")
        if beads_match_settings.t_reference == "first":
            ref_tzyx = np.broadcast_to(mov_tzyx[0], (T, Z, Y, X)).copy()
        elif beads_match_settings.t_reference == "previous":
            ref_tzyx = np.roll(mov_tzyx, shift=-1, axis=0)
            ref_tzyx[0] = mov_tzyx[0]
        else:
            raise ValueError(
                "Invalid reference. Please use 'first' or 'previous as reference."
            )
    elif mode == "registration":
        click.echo("Performing registration between different files")
    mov_zyx = np.asarray(mov_tzyx[t_idx]).astype(np.float32)
    ref_zyx = np.asarray(ref_tzyx[t_idx]).astype(np.float32)

    if output_folder_path:
        output_folder_path.mkdir(parents=True, exist_ok=True)
        output_filepath = output_folder_path / f"{t_idx}.npy"
    else:
        output_filepath = None

    transform = estimate(
        mov=mov_zyx,
        ref=ref_zyx,
        beads_match_settings=beads_match_settings,
        affine_transform_settings=affine_transform_settings,
        verbose=verbose,
        output_filepath=output_filepath,
    )
    return transform


def estimate(
    mov: da.Array,
    ref: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_filepath: Path = None,
) -> Transform:
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
    initial_transform_ants = initial_transform.to_ants()

    mov_ants = ants.from_numpy(mov)
    ref_ants = ants.from_numpy(ref)

    mov_reg_approx = initial_transform_ants.apply_to_image(
        mov_ants, reference=ref_ants
    ).numpy()

    mov_peaks, ref_peaks = peaks_from_beads(
        mov=mov_reg_approx,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=verbose,
    )

    matches = matches_from_beads(
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        beads_match_settings=beads_match_settings,
        verbose=verbose,
    )

    if len(matches) < 3:
        click.echo('Source and target beads were not matches successfully')
        return

    fwd_transform, inv_transform = transform_from_matches(
        matches=matches,
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        affine_transform_settings=affine_transform_settings,
        ndim=mov.ndim,
        verbose=verbose,
    )

    composed_transform = initial_transform @ inv_transform

    if verbose:
        click.echo(f'Bead matches: {matches}')
        click.echo(f"Forward transform: {fwd_transform}")
        click.echo(f"Inverse transform: {inv_transform}")
        click.echo(f"Composed transform: {composed_transform}")

    if output_filepath:
        print(f"Saving transform to {output_filepath}")
        np.save(output_filepath, composed_transform.matrix)

    return composed_transform
