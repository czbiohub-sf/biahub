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
5. **Parameter tuning** (`optimize_matches`): Grid search over matching settings to find
   the combination that maximizes registration quality.

Key conventions
---------------
- Coordinates are in ZYX order for 3D data.
- "mov" / "moving" refers to the source channel being aligned.
- "ref" / "reference" refers to the fixed target channel.
- Transforms map from moving space to reference space (forward direction).
"""

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Literal

import ants
import click
import dask.array as da
import numpy as np
import submitit

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
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
from biahub.registration.utils import get_aprox_transform, load_transforms
from biahub.settings import AffineTransformSettings, BeadsMatchSettings, DetectPeaksSettings


def optimize_matches(
    mov: ArrayLike,
    ref: ArrayLike,
    approx_transform: Transform,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    param_grid: dict = None,
    verbose: bool = False,
) -> BeadsMatchSettings:
    """
    Optimize BeadsMatchSettings by grid search over matching and filter parameters.

    For each parameter combination: detects peaks in approximately registered space,
    matches them, estimates a correction transform, composes it with the approx transform,
    applies to the full volume via ANTs, re-detects peaks, and scores the overlap.

    Parameters
    ----------
    mov : ArrayLike
        Original (unregistered) moving volume (Z, Y, X).
    ref : ArrayLike
        Reference volume (Z, Y, X).
    approx_transform : Transform
        Initial approximate transform to compose with.
    beads_match_settings : BeadsMatchSettings
        Initial matching settings to use as baseline.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform estimation.
    param_grid : dict, optional
        Dictionary of parameter names to lists of values to search.
        Supported keys: 'min_distance_quantile', 'max_distance_quantile',
        'direction_threshold', 'cost_threshold', 'max_ratio', 'k',
        'weights_dist', 'weights_edge_angle', 'weights_edge_length',
        'weights_pca_dir', 'weights_pca_aniso', 'weights_edge_descriptor'.
    verbose : bool
        If True, prints logs for each trial.

    Returns
    -------
    BeadsMatchSettings
        The settings that produced the best overlap score.
    """
    if param_grid is None:
        param_grid = {
            'min_distance_quantile': [0, 0.01],
            'max_distance_quantile': [0, 0.99],
            'direction_threshold': [0, 50],
            'k': [5, 10],
        }

    score_radius = beads_match_settings.qc_settings.score_centroid_mask_radius

    # Convert volumes to ANTs images once (reused across all trials)
    mov_ants = ants.from_numpy(mov.astype(np.float32))
    ref_ants = ants.from_numpy(ref.astype(np.float32))

    # Apply approximate transform to moving volume and detect peaks once.
    # These peaks are reused for all parameter combinations in the grid search.
    click.echo("Detecting peaks in approximately registered space for grid search...")
    mov_reg_approx = (
        approx_transform.to_ants().apply_to_image(mov_ants, reference=ref_ants).numpy()
    )
    mov_peaks, ref_peaks = peaks_from_beads(
        mov=mov_reg_approx,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=False,
    )
    if mov_peaks is None or ref_peaks is None or len(mov_peaks) < 2 or len(ref_peaks) < 2:
        click.echo("Not enough peaks detected for optimization, returning original settings.")
        return beads_match_settings

    click.echo(
        f"Starting grid search: {len(mov_peaks)} mov peaks, {len(ref_peaks)} ref peaks, "
        f"{np.prod([len(v) for v in param_grid.values()])} parameter combinations."
    )

    ndim = mov_peaks.shape[1]
    best_score = -1.0
    best_settings = beads_match_settings

    grid_keys = list(param_grid.keys())
    grid_values = [param_grid[k] for k in grid_keys]

    def apply_trial_params(trial_settings, trial_params):
        """Apply parameter values from a grid search trial to a BeadsMatchSettings copy."""
        fm = trial_settings.filter_matches_settings
        hm = trial_settings.hungarian_match_settings
        w = hm.cost_matrix_settings.weights
        param_map = {
            'min_distance_quantile': lambda v: setattr(fm, 'min_distance_quantile', v),
            'max_distance_quantile': lambda v: setattr(fm, 'max_distance_quantile', v),
            'direction_threshold': lambda v: setattr(fm, 'direction_threshold', v),
            'cost_threshold': lambda v: setattr(hm, 'cost_threshold', v),
            'max_ratio': lambda v: setattr(hm, 'max_ratio', v),
            'k': lambda v: setattr(hm.edge_graph_settings, 'k', v),
            'weights_dist': lambda v: w.__setitem__('dist', v),
            'weights_edge_angle': lambda v: w.__setitem__('edge_angle', v),
            'weights_edge_length': lambda v: w.__setitem__('edge_length', v),
            'weights_pca_dir': lambda v: w.__setitem__('pca_dir', v),
            'weights_pca_aniso': lambda v: w.__setitem__('pca_aniso', v),
            'weights_edge_descriptor': lambda v: w.__setitem__('edge_descriptor', v),
        }
        for key, val in trial_params.items():
            if key in param_map:
                param_map[key](val)

    for combo in product(*grid_values):
        trial_params = dict(zip(grid_keys, combo))
        trial_settings = beads_match_settings.model_copy(deep=True)
        apply_trial_params(trial_settings, trial_params)

        try:
            matches = matches_from_beads(
                mov_peaks=mov_peaks,
                ref_peaks=ref_peaks,
                beads_match_settings=trial_settings,
                verbose=False,
            )

            if len(matches) < 3:
                continue

            fwd_transform, inv_transform = transform_from_matches(
                matches=matches,
                mov_peaks=mov_peaks,
                ref_peaks=ref_peaks,
                affine_transform_settings=affine_transform_settings,
                ndim=ndim,
                verbose=False,
            )

            # Compose approx_transform with correction and apply to full volume
            composed_transform = approx_transform @ inv_transform
            mov_reg_optimized = (
                composed_transform.to_ants()
                .apply_to_image(mov_ants, reference=ref_ants)
                .numpy()
            )

            # Re-detect peaks and score
            mov_peaks_opt, ref_peaks_opt = peaks_from_beads(
                mov=mov_reg_optimized,
                ref=ref,
                mov_peaks_settings=beads_match_settings.source_peaks_settings,
                ref_peaks_settings=beads_match_settings.target_peaks_settings,
                verbose=False,
            )
            if mov_peaks_opt is None or ref_peaks_opt is None:
                continue

            score = overlap_score(
                mov_peaks=mov_peaks_opt,
                ref_peaks=ref_peaks_opt,
                radius=score_radius,
                verbose=False,
            )

            if np.isnan(score):
                continue

            if verbose:
                click.echo(f"  {trial_params} -> matches={len(matches)}, score={score:.4f}")

            if score > best_score:
                best_score = score
                best_settings = trial_settings

        except Exception as e:
            if verbose:
                click.echo(f"  {trial_params} -> failed: {e}")
            continue

    if verbose:
        click.echo(f"Best score: {best_score:.4f}")
        click.echo(f"Best settings: {best_settings}")

    return best_settings


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
    ref_voxel_size: tuple[float, float, float] = (0.174, 0.1494, 0.1494),
    mov_voxel_size: tuple[float, float, float] = (0.174, 0.1494, 0.1494),
    mode: Literal["registration", "stabilization"] = "registration",
) -> list[Transform]:
    """
    Estimate beads-based registration transforms for all timepoints.

    Orchestrates the full registration pipeline: computes the approximate transform
    (if needed), then estimates per-timepoint transforms either sequentially with
    propagation or independently via SLURM, depending on settings.

    Parameters
    ----------
    mov_tczyx : da.Array
        Moving data (T, C, Z, Y, X).
    ref_tczyx : da.Array
        Reference data (T, C, Z, Y, X).
    mov_channel_index : int
        Channel index in the moving data containing beads.
    ref_channel_index : int, optional
        Channel index in the reference data. Ignored in stabilization mode.
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, filtering, and QC.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type, initial approx transform, and propagation.
    verbose : bool
        If True, prints detailed logs.
    cluster : bool
        If True, submits jobs to SLURM; otherwise runs locally.
    sbatch_filepath : Path, optional
        Path to sbatch file for custom SLURM parameters.
    output_folder_path : Path
        Directory to save per-timepoint transforms and logs.
    ref_voxel_size : tuple[float, float, float]
        Reference voxel size (Z, Y, X) in microns.
    mov_voxel_size : tuple[float, float, float]
        Moving voxel size (Z, Y, X) in microns.
    mode : {"registration", "stabilization"}
        "registration": align two different channels.
        "stabilization": align one channel to itself over time.

    Returns
    -------
    list[Transform]
        One 4x4 affine transform per timepoint.
    """
    mov_tzyx = mov_tczyx[:, mov_channel_index]
    if mode == "stabilization":
        ref_tzyx = mov_tzyx
    elif mode == "registration":
        ref_tzyx = ref_tczyx[:, ref_channel_index]

    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    if affine_transform_settings.compute_approx_transform:
        approx_transform = get_aprox_transform(
            mov_shape=mov_tzyx.shape[-3:],
            ref_shape=ref_tzyx.shape[-3:],
            pre_affine_90degree_rotation=-1,
            pre_affine_fliplr=False,
            verbose=verbose,
            ref_voxel_size=ref_voxel_size,
            mov_voxel_size=mov_voxel_size,
        )
        click.echo("Computed approx transform: ", approx_transform)
        affine_transform_settings.approx_transform = approx_transform.to_list()

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

    transforms = load_transforms(output_transforms_path, mov_tzyx.shape[0], verbose)

    return transforms


def estimate_with_propagation(
    mov_tzyx: da.Array,
    ref_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    mode: Literal["registration", "stabilization"] = "registration",
) -> None:
    """
    Estimate transforms sequentially, propagating each result to the next timepoint.

    Processes timepoints in order (t=0, 1, 2, ...). After each timepoint, the
    estimated transform is used as the approximate transform for the next timepoint.
    This is useful when drift is gradual and cumulative, as each timepoint starts
    from a better initial guess.

    Parameters
    ----------
    mov_tzyx : da.Array
        Moving volume (T, Z, Y, X).
    ref_tzyx : da.Array
        Reference volume (T, Z, Y, X).
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, and filtering.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type and initial approximate transform.
        Modified in-place: approx_transform is updated after each timepoint.
    verbose : bool
        If True, prints progress for each timepoint.
    output_folder_path : Path
        Directory to save per-timepoint transform .npy files.
    mode : {"registration", "stabilization"}
        "registration": align moving to reference channel.
        "stabilization": align moving channel to itself over time.
    """
    initial_transform = affine_transform_settings.approx_transform
    T, _, _, _ = mov_tzyx.shape
    for t in range(T):
        if mode == "stabilization" and t == 0:
            continue
        if np.sum(mov_tzyx[t]) == 0 or np.sum(ref_tzyx[t]) == 0:
            click.echo(f"Timepoint {t} has no data, skipping")
        else:
            approx_transform = estimate_tzyx(
                t_idx=t,
                mov_tzyx=mov_tzyx,
                ref_tzyx=ref_tzyx,
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
) -> None:
    """
    Estimate transforms for all timepoints independently via SLURM.

    Each timepoint is submitted as an independent job using submitit. All jobs
    use the same approximate transform as their starting point (no propagation).
    Suitable for large datasets where timepoints can be processed in parallel.

    Parameters
    ----------
    mov_tzyx : da.Array
        Moving volume (T, Z, Y, X).
    ref_tzyx : da.Array
        Reference volume (T, Z, Y, X).
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, and filtering.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type and initial approximate transform.
    verbose : bool
        If True, prints progress for each timepoint.
    output_folder_path : Path
        Directory to save per-timepoint transform .npy files.
    cluster : str
        Submitit cluster backend ('local', 'slurm', etc.).
    sbatch_filepath : Path, optional
        Path to sbatch file for custom SLURM parameters.
    mode : {"registration", "stabilization"}
        "registration": align moving to reference channel.
        "stabilization": align moving channel to itself over time.
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

    slurm_out_path = output_folder_path.parent / "slurm_output"
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


def estimate_tzyx(
    t_idx: int,
    mov_tzyx: da.Array,
    ref_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    mode: Literal["registration", "stabilization"] = "registration",
    user_transform: Transform = None,
) -> Transform:
    """
    Estimate the affine transform for a single timepoint.

    Extracts the 3D volumes for the given timepoint, sets up the reference
    depending on the mode (registration vs stabilization), and delegates to
    `estimate()` for iterative bead-based transform estimation.

    Parameters
    ----------
    t_idx : int
        Timepoint index to process.
    mov_tzyx : da.Array
        Moving volume (T, Z, Y, X).
    ref_tzyx : da.Array
        Reference volume (T, Z, Y, X). Ignored in stabilization mode.
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, and filtering.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type and initial approximate transform.
    verbose : bool
        If True, prints detailed logs during the process.
    output_folder_path : Path, optional
        Directory to save the transform as ``{t_idx}.npy``.
    mode : {"registration", "stabilization"}
        "registration": align moving to reference (different channels).
        "stabilization": align moving channel to itself over time,
        using t_reference setting ("first" or "previous").
    user_transform : Transform, optional
        Alternative initial transform to compete with the default on iteration 0.

    Returns
    -------
    Transform or None
        The estimated 4x4 affine transform, or None if estimation failed.
    """
    click.echo("........................................................................")
    click.echo(f'Processing timepoint: {t_idx}')

    (T, Z, Y, X) = mov_tzyx.shape

    if mode == "stabilization":
        click.echo("Performing stabilization, aka registration over time in the same file.")
        if affine_transform_settings.t_reference == "first":
            ref_tzyx = np.broadcast_to(mov_tzyx[0], (T, Z, Y, X)).copy()
        elif affine_transform_settings.t_reference == "previous":
            ref_tzyx = np.roll(mov_tzyx, shift=-1, axis=0)
            ref_tzyx[0] = mov_tzyx[0]
        else:
            raise ValueError(
                "Invalid reference. Please use 'first' or 'previous' as reference."
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
        user_transform=user_transform,
    )
    return transform


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
    if (len(mov_peaks) is None) or (len(ref_peaks) is None):
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
        click.echo(f'Bead matches: {matches}')
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


def estimate(
    mov: da.Array,
    ref: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_filepath: Path = None,
    user_transform: Transform = None,
    debug: bool = False,
) -> Transform:
    """
    Estimate the best affine transformation between moving and reference volumes.

    Iteratively refines the transform by detecting beads, matching them, estimating
    a correction, and scoring the result. Supports an optional user-provided
    transform that competes with the computed one on the first iteration.

    Works for both 2D (Y, X) and 3D (Z, Y, X) arrays.

    Parameters
    ----------
    mov : ArrayLike
        Moving channel volume (Z, Y, X) or (Y, X).
    ref : ArrayLike
        Reference channel volume (Z, Y, X) or (Y, X).
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, filtering, and QC iterations.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type and initial approximate transform.
    verbose : bool
        If True, prints the best transform and score at the end.
    output_filepath : Path, optional
        If provided, saves the best transform matrix as a .npy file.
    user_transform : Transform, optional
        An alternative initial transform (e.g. from a previous timepoint).
        Tested on the first iteration; used if it scores better.
    debug : bool
        If True, passes debug flag to optimize_transform for detailed logging.

    Returns
    -------
    Transform
        The best transform found across all iterations. Falls back to the
        initial approximate transform if no valid optimization was found.
    """

    if _check_nan_n_zeros(mov) or _check_nan_n_zeros(ref):
        click.echo('Skipping: moving or reference data contains only NaN/zeros.')
        return

    initial_transform = Transform(
        matrix=np.asarray(affine_transform_settings.approx_transform)
    )
    transform = initial_transform

    current_iterations = 0
    qc_iterations = beads_match_settings.qc_settings.iterations
    transform_iter_dict = {}

    while current_iterations < qc_iterations:
        click.echo(
            f"Iteration {current_iterations + 1}/{qc_iterations}: "
            "optimizing transform via bead matching..."
        )
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
