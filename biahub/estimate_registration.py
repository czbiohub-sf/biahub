import os
import shutil

from datetime import datetime
from pathlib import Path
from typing import Literal, Union

import ants
import click
import dask.array as da
import napari
import numpy as np
import submitit

from iohub import open_ome_zarr
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import match_descriptors
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from waveorder.focus import focus_from_transverse_band


from biahub.registration.utils import evaluate_transforms, plot_translations
from biahub.settings import (
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)


from biahub.characterize_psf import detect_peaks
from biahub.cli.parsing import (
    config_filepath,
    local,
    output_filepath,
    sbatch_filepath,
    sbatch_to_submitit,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import (
    _check_nan_n_zeros,
    estimate_resources,
    model_to_yaml,
    yaml_to_model,
)
from biahub.optimize_registration import _optimize_registration
from biahub.register import (
    convert_transform_to_ants,
    convert_transform_to_numpy,
    get_3D_fliplr_matrix,
    get_3D_rescaling_matrix,
    get_3D_rotation_matrix,
)
from biahub.settings import (
    AffineTransformSettings,
    AntsRegistrationSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)



def shrink_slice(s: slice, shrink_fraction: float = 0.1, min_width: int = 5) -> slice:
    """
    Shrink a slice by a fraction of its length.

    Parameters
    ----------
    s : slice
        The slice to shrink.
    shrink_fraction : float
        The fraction of the slice to shrink.
    min_width : int
        The minimum width of the slice.

    Returns
    -------
    slice
        The shrunk slice.
    Notes
    -----
    If the slice is too small, return the original slice.

    """
    start = s.start or 0
    stop = s.stop or 0
    length = stop - start
    if length <= min_width:
        return slice(start, stop)

    shrink = int(length * shrink_fraction)
    new_start = start + shrink
    new_stop = stop - shrink
    if new_stop <= new_start:
        return slice(start, stop)
    return slice(new_start, new_stop)


def ants_registration(
    source_data_tczyx: da.Array,
    target_data_tczyx: da.Array,
    source_channel_index: int | list[int],
    target_channel_index: int,
    ants_registration_settings: AntsRegistrationSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    cluster: str = 'local',
    sbatch_filepath: Path = None,
) -> list[ArrayLike]:
    """
    Perform ants registration of two volumetric image channels.

    This function calculates timepoint-specific affine transformations to align a source channel
    to a target channel in 4D (T, Z, Y, X) data. It validates, smooths, and interpolates transformations
    across timepoints for consistent registration.

    Parameters
    ----------
    source_data_tczyx : da.Array
       4D array (T, C, Z, Y, X) of the source channel (Dask array).
    target_data_tczyx : da.Array
       4D array (T, C, Z, Y, X) of the target channel (Dask array).
    source_channel_index : int | list[int]
        Index of the source channel.
    target_channel_index : int
        Index of the target channel.
    ants_registration_settings : AntsRegistrationSettings
        Settings for the ants registration.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, prints detailed logs of the registration process.
    output_folder_path : Path
        Path to the output folder.
    cluster : str
        Cluster to use.
    sbatch_filepath : Path
        Path to the sbatch file.

    Returns
    -------
    list[ArrayLike]
        List of affine transformation matrices (4x4), one for each timepoint.
        Invalid or missing transformations are interpolated.

    Notes
    -----
    Each timepoint is processed in parallel using submitit executor.
    Use verbose=True for detailed logging during registration. The verbose output will be saved at the same level as the output zarr.
    """
    T, C, Z, Y, X = source_data_tczyx.shape
    initial_tform = np.asarray(affine_transform_settings.approx_transform)

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, 2, Z, Y, X), ram_multiplier=16, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_registration",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 30,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM estimate regstration jobs with resources: {slurm_args}")
    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    click.echo('Computing registration transforms...')
    # NOTE: ants is mulitthreaded so no need for multiprocessing here
    # Submit jobs
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for t in range(T):
            job = executor.submit(
                _optimize_registration,
                source_data_tczyx[t],
                target_data_tczyx[t],
                initial_tform=initial_tform,
                source_channel_index=source_channel_index,
                target_channel_index=target_channel_index,
                crop=True,
                target_mask_radius=0.8,
                clip=True,
                sobel_fitler=ants_registration_settings.sobel_filter,
                verbose=verbose,
                slurm=True,
                output_folder_path=output_transforms_path,
                t_idx=t,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    wait_for_jobs_to_finish(jobs)

    # Load the transforms
    transforms = []
    for t in range(T):
        file_path = output_transforms_path / f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
            click.echo(f"Transform for timepoint {t} not found.")
        else:
            T_zyx_shift = np.load(file_path).tolist()
            transforms.append(T_zyx_shift)

    if len(transforms) != T:
        raise ValueError(
            f"Number of transforms {len(transforms)} does not match number of timepoints {T}"
        )

    # Remove the output temporary folder
    shutil.rmtree(output_transforms_path)

    return transforms


def beads_based_registration(
    source_channel_tzyx: da.Array,
    target_channel_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings = None,
    affine_transform_settings: AffineTransformSettings = None,
    verbose: bool = False,
    cluster: bool = False,
    sbatch_filepath: Path = None,
    output_folder_path: Path = None,
) -> list[ArrayLike]:
    """
    Perform beads-based temporal registration of 4D data using affine transformations.

    This function calculates timepoint-specific affine transformations to align a source channel
    to a target channel in 4D (T, Z, Y, X) data. It validates, smooths, and interpolates transformations
    across timepoints for consistent registration.

    Parameters
    ----------
    source_channel_tzyx : da.Array
       4D array (T, Z, Y, X) of the source channel (Dask array).
    target_channel_tzyx : da.Array
       4D array (T, Z, Y, X) of the target channel (Dask array).
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

    Returns
    -------
    list[ArrayLike]
        List of affine transformation matrices (4x4), one for each timepoint.
        Invalid or missing transformations are interpolated.

    Notes
    -----
    Each timepoint is processed in parallel using submitit executor.
    Use verbose=True for detailed logging during registration. The verbose output will be saved at the same level as the output zarr.
    """

    (T, Z, Y, X) = source_channel_tzyx.shape

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

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")
    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for t in range(T):
            job = executor.submit(
                estimate_transform_from_beads,
                source_channel_tzyx=source_channel_tzyx,
                target_channel_tzyx=target_channel_tzyx,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                verbose=verbose,
                slurm=True,
                output_folder_path=output_transforms_path,
                t_idx=t,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    wait_for_jobs_to_finish(jobs)

    transforms = []
    for t in range(T):
        file_path = output_transforms_path / f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
        else:
            T_zyx_shift = np.load(file_path).tolist()
            transforms.append(T_zyx_shift)

    # Remove the output temporary folder
    shutil.rmtree(output_transforms_path)

    return transforms


def get_local_pca_features(
    points: ArrayLike,
    edges: list[tuple[int, int]],
) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute dominant direction and anisotropy for each point using PCA,
    using neighborhoods defined by existing graph edges.

    Parameters
    ----------
    points : ArrayLike
        (n, 2) array of points.
    edges : list[tuple[int, int]]
        List of edges (i, j) in the graph.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        - directions : (n, 3) array of dominant directions.
        - anisotropy : (n,) array of anisotropy.

    Notes
    -----
    The PCA features are computed as the dominant direction and anisotropy of the local neighborhood of each point.
    The direction is the first principal component of the local neighborhood.
    The anisotropy is the ratio of the first to third principal component of the local neighborhood.
    """
    n = len(points)
    directions = np.zeros((n, 3))
    anisotropy = np.zeros(n)

    # Build neighbor list from edges
    from collections import defaultdict

    neighbor_map = defaultdict(list)
    for i, j in edges:
        neighbor_map[i].append(j)

    for i in range(n):
        neighbors = neighbor_map[i]
        if not neighbors:
            directions[i] = np.nan
            anisotropy[i] = np.nan
            continue

        local_points = points[neighbors].astype(np.float32)
        local_points -= local_points.mean(axis=0)
        _, S, Vt = np.linalg.svd(local_points, full_matrices=False)

        directions[i] = Vt[0] if Vt.shape[0] > 0 else np.zeros(3)
        anisotropy[i] = S[0] / (S[2] + 1e-5) if len(S) >= 3 else 0.0

    return directions, anisotropy


def get_edge_descriptors(
    points: ArrayLike,
    edges: list[tuple[int, int]],
) -> ArrayLike:
    """
    Compute edge descriptors for a set of points.

    Parameters
    ----------
    points : ArrayLike
        (n, 2) array of points.
    edges : list[tuple[int, int]]
        List of edges (i, j) in the graph.

    Returns
    -------
    ArrayLike
        (n, 4) array of edge descriptors.
        Each row contains:
        - mean length
        - std length
        - mean angle
        - std angle

    Notes
    -----
    The edge descriptors are computed as the mean and standard deviation of the lengths and angles of the edges.
    """
    n = len(points)
    desc = np.zeros((n, 4))
    for i in range(n):
        neighbors = [j for a, j in edges if a == i]
        if not neighbors:
            continue
        vectors = points[neighbors] - points[i]
        lengths = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        desc[i, 0] = np.mean(lengths)
        desc[i, 1] = np.std(lengths)
        desc[i, 2] = np.mean(angles)
        desc[i, 3] = np.std(angles)
    return desc


def get_edge_attrs(
    points: ArrayLike,
    edges: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """
    Compute edge distances and angles for a set of points.

    Parameters
    ----------
    points : ArrayLike
        (n, 2) array of points.
    edges : list[tuple[int, int]]
        List of edges (i, j) in the graph.

    Returns
    -------
    tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]
        - distances : dict[tuple[int, int], float]
        - angles : dict[tuple[int, int], float]

    """
    distances, angles = {}, {}
    for i, j in edges:
        vec = points[j] - points[i]
        d = np.linalg.norm(vec)
        angle = np.arctan2(vec[1], vec[0])
        distances[(i, j)] = distances[(j, i)] = d
        angles[(i, j)] = angles[(j, i)] = angle
    return distances, angles


def match_hungarian_local_cost(
    i: int,
    j: int,
    s_neighbors: list[int],
    t_neighbors: list[int],
    source_attrs: dict[tuple[int, int], float],
    target_attrs: dict[tuple[int, int], float],
    default_cost: float,
) -> float:
    """
    Match neighbor edges between two graphs using the Hungarian algorithm for local cost estimation.
    The cost is the mean of the absolute differences between the source and target edge attributes.

    Parameters
    ----------
    i : int
        Index of the source edge.
    j : int
        Index of the target edge.
    s_neighbors : list[int]
        List of source neighbors.
    t_neighbors : list[int]
        List of target neighbors.
    source_attrs : dict[tuple[int, int], float]
        Dictionary of source edge attributes.
    target_attrs : dict[tuple[int, int], float]
        Dictionary of target edge attributes.
    """
    C = np.full((len(s_neighbors), len(t_neighbors)), default_cost)

    # compute cost matrix
    for ii, sn in enumerate(s_neighbors):
        # get target neighbors
        for jj, tn in enumerate(t_neighbors):
            s_edge = (i, sn)
            t_edge = (j, tn)
            if s_edge in source_attrs and t_edge in target_attrs:
                C[ii, jj] = abs(source_attrs[s_edge] - target_attrs[t_edge])

    # use hungarian algorithm to find the best match
    row_ind, col_ind = linear_sum_assignment(C)
    # get the mean of the matched costs
    matched_costs = C[row_ind, col_ind]
    # return the mean of the matched costs

    return matched_costs.mean() if len(matched_costs) > 0 else default_cost


def compute_edge_consistency_cost(
    n: int,
    m: int,
    source_attrs: dict[tuple[int, int], float],
    target_attrs: dict[tuple[int, int], float],
    source_edges: list[tuple[int, int]],
    target_edges: list[tuple[int, int]],
    default: float = 1e6,
    hungarian: bool = True,
) -> ArrayLike:
    """
    Compute the cost matrix for matching edges between two graphs.

    Parameters
    ----------
    n : int
        Number of source edges.
    m : int
        Number of target edges.
    source_attrs : dict[tuple[int, int], float]
        Dictionary of source edge attributes.
    target_attrs : dict[tuple[int, int], float]
        Dictionary of target edge attributes.
    source_edges : list[tuple[int, int]]
        List of edges (i, j) in source graph.
    target_edges : list[tuple[int, int]]
        List of edges (i, j) in target graph.
    default : float
        Default value for the cost matrix.
    hungarian : bool
        Whether to use the Hungarian algorithm for local cost estimation.
        If False, the cost matrix is computed as the mean of the absolute differences between the source and target edge attributes.
        If True, the cost matrix is computed as the mean of the absolute differences between the source and target edge attributes using the Hungarian algorithm.

    Returns
    -------
    ArrayLike
        Cost matrix of shape (n, m).

    Notes
    -----
    The cost matrix is computed as the mean of the absolute differences between the source and target edge attributes.
    """
    cost_matrix = np.full((n, m), default)
    for i in range(n):
        # get source neighbors
        s_neighbors = [j for a, j in source_edges if a == i]
        for j in range(m):
            # get target neighbors
            t_neighbors = [k for a, k in target_edges if a == j]
            if hungarian:
                # hungarian algorithm based cost estimation
                cost_matrix[i, j] = match_hungarian_local_cost(
                    i, j, s_neighbors, t_neighbors, source_attrs, target_attrs, default
                )
            else:
                # position based cost estimation (mean of the absolute differences between the source and target edge attributes)
                common_len = min(len(s_neighbors), len(t_neighbors))
                diffs = []
                for k in range(common_len):
                    s_edge = (i, s_neighbors[k])
                    t_edge = (j, t_neighbors[k])
                    if s_edge in source_attrs and t_edge in target_attrs:
                        v1 = source_attrs[s_edge]
                        v2 = target_attrs[t_edge]
                        diff = np.abs(v1 - v2)
                        diffs.append(diff)
                cost_matrix[i, j] = np.mean(diffs) if diffs else default

    return cost_matrix


def compute_cost_matrix(
    source_peaks: ArrayLike,
    target_peaks: ArrayLike,
    source_edges: list[tuple[int, int]],
    target_edges: list[tuple[int, int]],
    weights: dict[str, float] = None,
    distance_metric: str = 'euclidean',
    normalize: bool = False,
) -> ArrayLike:
    """
    Compute a cost matrix for matching peaks between two graphs based on:
    - Euclidean or other distance between peaks
    - Consistency in edge distances
    - Consistency in edge angles
    - PCA features
    - Edge descriptors

    Parameters
    ----------
    source_peaks : ArrayLike
        (n, 2) array of source node coordinates.
    target_peaks : ArrayLike
        (m, 2) array of target node coordinates.
    source_edges : list[tuple[int, int]]
        List of edges (i, j) in source graph.
    target_edges : list[tuple[int, int]]
        List of edges (i, j) in target graph.
    weights : dict[str, float]
        Weights for different cost components.
    distance_metric : str
        Metric for direct point-to-point distances.
    normalize : bool
        Whether to normalize the cost matrix.

    Notes
    -----
    The cost matrix is computed as the sum of the weighted costs for each component.
    The weights are defined in the `weights` parameter.
    The default weights are:
    - dist: 0.5
    - edge_angle: 1.0
    - edge_length: 1.0
    - pca_dir: 0.0
    - pca_aniso: 0.0
    - edge_descriptor: 0.0

    Returns
    -------
    ArrayLike
        Cost matrix of shape (n, m).
    """
    n, m = len(source_peaks), len(target_peaks)
    C_total = np.zeros((n, m))

    # --- Default weights ---
    default_weights = {
        "dist": 0.5,
        "edge_angle": 1.0,
        "edge_length": 1.0,
        "pca_dir": 0.0,
        "pca_aniso": 0.0,
        "edge_descriptor": 0.0,
    }
    if weights is None:
        weights = default_weights
    else:
        weights = {**default_weights, **weights}  # override defaults

    # --- Base distance cost ---
    if weights["dist"] > 0:
        C_dist = cdist(source_peaks, target_peaks, metric=distance_metric)
        if normalize:
            C_dist /= C_dist.max()
        C_total += weights["dist"] * C_dist

    # --- Edge angle and length costs ---
    source_dists, source_angles = get_edge_attrs(source_peaks, source_edges)
    target_dists, target_angles = get_edge_attrs(target_peaks, target_edges)

    if weights["edge_length"] > 0:
        C_edge_len = compute_edge_consistency_cost(
            n=n,
            m=m,
            source_attrs=source_dists,
            target_attrs=target_dists,
            source_edges=source_edges,
            target_edges=target_edges,
            default=1e6,
        )
        if normalize:
            C_edge_len /= C_edge_len.max()
        C_total += weights["edge_length"] * C_edge_len

    if weights["edge_angle"] > 0:
        C_edge_ang = compute_edge_consistency_cost(
            n=n,
            m=m,
            source_attrs=source_angles,
            target_attrs=target_angles,
            source_edges=source_edges,
            target_edges=target_edges,
            default=np.pi,
        )
        if normalize:
            C_edge_ang /= np.pi
        C_total += weights["edge_angle"] * C_edge_ang

    # --- PCA features ---
    if weights["pca_dir"] > 0 or weights["pca_aniso"] > 0:
        dirs_s, aniso_s = get_local_pca_features(source_peaks, source_edges)
        dirs_t, aniso_t = get_local_pca_features(target_peaks, target_edges)

        if weights["pca_dir"] > 0:
            dot = np.clip(np.dot(dirs_s, dirs_t.T), -1.0, 1.0)
            C_dir = 1 - np.abs(dot)
            if normalize:
                C_dir /= C_dir.max()
            C_total += weights["pca_dir"] * C_dir

        if weights["pca_aniso"] > 0:
            C_aniso = np.abs(aniso_s[:, None] - aniso_t[None, :])
            if normalize:
                C_aniso /= C_aniso.max()
            C_total += weights["pca_aniso"] * C_aniso
    # --- Edge descriptors ---
    if weights["edge_descriptor"] > 0:
        desc_s = get_edge_descriptors(source_peaks, source_edges)
        desc_t = get_edge_descriptors(target_peaks, target_edges)
        C_desc = cdist(desc_s, desc_t)
        if normalize:
            C_desc /= C_desc.max()
        C_total += weights["edge_descriptor"] * C_desc

    return C_total


def build_edge_graph(
    points: ArrayLike,
    mode: Literal["knn", "radius", "full"] = "knn",
    k: int = 5,
    radius: float = 30.0,
) -> list[tuple[int, int]]:
    """
    Build a set of edges for a graph based on a given strategy.

    Parameters
    ----------
    points : ArrayLike
        (N, 3) array of 3D point coordinates.
    mode : Literal["knn", "radius", "full"]
        Mode for building the edge graph.
    k : int
        Number of neighbors if mode == "knn".
    radius : float
        Distance threshold if mode == "radius".

    Returns
    -------
    list[tuple[int, int]]
        List of (i, j) index pairs representing edges.
    """
    n = len(points)
    if n <= 1:
        return []

    if mode == "knn":
        k_eff = min(k + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k_eff).fit(points)
        _, indices = nbrs.kneighbors(points)
        edges = [(i, j) for i in range(n) for j in indices[i] if i != j]

    elif mode == "radius":
        graph = radius_neighbors_graph(
            points, radius=radius, mode='connectivity', include_self=False
        )
        if graph.nnz == 0:
            return []
        edges = [(i, j) for i in range(n) for j in graph[i].nonzero()[1]]

    elif mode == "full":
        edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    return edges


def match_hungarian_global_cost(
    C: ArrayLike,
    cost_threshold: float = 1e5,
    dummy_cost: float = 1e6,
    max_ratio: float = None,
) -> ArrayLike:
    """
    Runs Hungarian matching with padding for unequal-sized graphs,
    optionally applying max_ratio filtering similar to match_descriptors.

    Parameters
    ----------
    C : ArrayLike
        Cost matrix of shape (n_A, n_B).
    cost_threshold : float
        Maximum cost to consider a valid match.
    dummy_cost : float
        Cost assigned to dummy nodes (must be > cost_threshold).
    max_ratio : float, optional
        Maximum allowed ratio between best and second-best cost.

    Returns
    -------
    ArrayLike
        Array of shape (N_matches, 2) with valid (A_idx, B_idx) pairs.
    """
    n_A, n_B = C.shape
    n = max(n_A, n_B)

    # Pad cost matrix to square shape
    C_padded = np.full((n, n), fill_value=dummy_cost)
    C_padded[:n_A, :n_B] = C

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(C_padded)

    matches = []
    for i, j in zip(row_ind, col_ind):
        if i >= n_A or j >= n_B:
            continue  # matched with dummy
        if C[i, j] >= cost_threshold:
            continue  # too costly

        if max_ratio is not None:
            # Find second-best match for i
            costs_i = C[i, :]
            sorted_costs = np.sort(costs_i)
            if len(sorted_costs) > 1:
                second_best = sorted_costs[1]
                ratio = C[i, j] / (second_best + 1e-10)  # avoid division by zero
                if ratio > max_ratio:
                    continue  # reject if not sufficiently better
            # else (only one candidate) => accept by default

        matches.append((i, j))

    return np.array(matches)


def detect_bead_peaks(
    source_channel_zyx: da.Array,
    target_channel_zyx: da.Array,
    source_peaks_settings: DetectPeaksSettings,
    target_peaks_settings: DetectPeaksSettings,
    verbose: bool = False,
    filter_dirty_peaks: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Detect peaks in source and target channels using the detect_peaks function.

    Parameters
    ----------
    source_channel_zyx : da.Array
        (T, Z, Y, X) array of the source channel (Dask array).
    target_channel_zyx : da.Array
        (T, Z, Y, X) array of the target channel (Dask array).
    source_peaks_settings : DetectPeaksSettings
        Settings for the source peaks.
    target_peaks_settings : DetectPeaksSettings
        Settings for the target peaks.
    verbose : bool
        If True, prints detailed logs during the process.
    filter_dirty_peaks : bool
        If True, filters the dirty peaks.
    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Tuple of (source_peaks, target_peaks).
    """
    if verbose:
        click.echo('Detecting beads in source dataset')

    source_peaks = detect_peaks(
        source_channel_zyx,
        block_size=source_peaks_settings.block_size,
        threshold_abs=source_peaks_settings.threshold_abs,
        nms_distance=source_peaks_settings.nms_distance,
        min_distance=source_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo('Detecting beads in target dataset')

    target_peaks = detect_peaks(
        target_channel_zyx,
        block_size=target_peaks_settings.block_size,
        threshold_abs=target_peaks_settings.threshold_abs,
        nms_distance=target_peaks_settings.nms_distance,
        min_distance=target_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo(f'Total of peaks in source dataset: {len(source_peaks)}')
        click.echo(f'Total of peaks in target dataset: {len(target_peaks)}')

    if len(source_peaks) < 2 or len(target_peaks) < 2:
        click.echo('Not enough beads detected')
        return
    if filter_dirty_peaks:
        print("Filtering dirty peaks")
        with open_ome_zarr(
            Path(
                "/hpc/projects/intracellular_dashboard/viral-sensor/dirty_on_mantis/lf_mask_2025_05_01_A549_DENV_sensor_DENV_T_9_0.zarr/C/1/000000"
            )
        ) as dirty_mask_ds:
            dirty_mask_load = np.asarray(dirty_mask_ds.data[0, 0])

        # filter the dirty peaks
        # Keep only peaks whose (y, x) column is clean across all Z slices
        target_peaks_filtered = []
        for peak in target_peaks:
            z, y, x = peak.astype(int)
            if (
                0 <= y < dirty_mask_load.shape[1]
                and 0 <= x < dirty_mask_load.shape[2]
                and not dirty_mask_load[:, y, x].any()  # True if all Z are clean at (y, x)
            ):
                target_peaks_filtered.append(peak)
        target_peaks = np.array(target_peaks_filtered)
    return source_peaks, target_peaks


def get_matches_from_hungarian(
    source_peaks: ArrayLike,
    target_peaks: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
    verbose: bool = False,
) -> ArrayLike:
    """
    Get matches from beads using the hungarian algorithm.
    Parameters
    ----------
    source_peaks : ArrayLike
        (n, 2) array of source peaks.
    target_peaks : ArrayLike
        (m, 2) array of target peaks.
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    verbose : bool
        If True, prints detailed logs during the process.

    Returns
    -------
    ArrayLike
        (n, 2) array of matches.
    """
    hungarian_settings = beads_match_settings.hungarian_match_settings
    cost_settings = hungarian_settings.cost_matrix_settings
    edge_settings = hungarian_settings.edge_graph_settings
    source_edges = build_edge_graph(
        source_peaks, mode=edge_settings.method, k=edge_settings.k, radius=edge_settings.radius
    )
    target_edges = build_edge_graph(
        target_peaks, mode=edge_settings.method, k=edge_settings.k, radius=edge_settings.radius
    )

    if hungarian_settings.cross_check:
        # Step 1: A → B
        C_ab = compute_cost_matrix(
            source_peaks,
            target_peaks,
            source_edges,
            target_edges,
            weights=cost_settings.weights,
            distance_metric=hungarian_settings.distance_metric,
            normalize=cost_settings.normalize,
        )

        matches_ab = match_hungarian_global_cost(
            C_ab,
            cost_threshold=np.quantile(C_ab, hungarian_settings.cost_threshold),
            max_ratio=hungarian_settings.max_ratio,
        )

        # Step 2: B → A (swap arguments)
        C_ba = compute_cost_matrix(
            target_peaks,
            source_peaks,
            target_edges,
            source_edges,
            weights=cost_settings.weights,
            distance_metric=hungarian_settings.distance_metric,
            normalize=cost_settings.normalize,
        )

        matches_ba = match_hungarian_global_cost(
            C_ba,
            cost_threshold=np.quantile(C_ba, hungarian_settings.cost_threshold),
            max_ratio=hungarian_settings.max_ratio,
        )

        # Step 3: Invert matches_ba to compare
        reverse_map = {(j, i) for i, j in matches_ba}

        # Step 4: Keep only symmetric matches
        matches = np.array([[i, j] for i, j in matches_ab if (i, j) in reverse_map])
    else:
        # without cross-check

        C = compute_cost_matrix(
            source_peaks,
            target_peaks,
            source_edges,
            target_edges,
            weights=cost_settings.weights,
            distance_metric=hungarian_settings.distance_metric,
            normalize=cost_settings.normalize,
        )

        matches = match_hungarian_global_cost(
            C,
            cost_threshold=np.quantile(C, hungarian_settings.cost_threshold),
            max_ratio=hungarian_settings.max_ratio,
        )
    return matches


def get_matches_from_beads(
    source_peaks: ArrayLike,
    target_peaks: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
    verbose: bool = False,
) -> ArrayLike:
    """
    Get matches from beads using the hungarian algorithm.

    Parameters
    ----------
    source_peaks : ArrayLike
        (n, 2) array of source peaks.
    target_peaks : ArrayLike
        (m, 2) array of target peaks.
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
        match_descriptor_settings = beads_match_settings.match_descriptor_settings
        matches = match_descriptors(
            source_peaks,
            target_peaks,
            metric=match_descriptor_settings.distance_metric,
            max_ratio=match_descriptor_settings.max_ratio,
            cross_check=match_descriptor_settings.cross_check,
        )

    elif beads_match_settings.algorithm == 'hungarian':
        matches = get_matches_from_hungarian(
            source_peaks=source_peaks,
            target_peaks=target_peaks,
            beads_match_settings=beads_match_settings,
            verbose=verbose,
        )

    if verbose:
        click.echo(f'Total of matches: {len(matches)}')

    return matches


def filter_matches(
    matches: ArrayLike,
    source_peaks: ArrayLike,
    target_peaks: ArrayLike,
    angle_threshold: float = 30,
    distance_threshold: float = 0.95,
    verbose: bool = False,
) -> ArrayLike:
    """
    Filter matches based on the angle and distance thresholds.

    Parameters
    ----------
    matches : ArrayLike
        (n, 2) array of matches.
    source_peaks : ArrayLike
        (n, 2) array of source peaks.
    target_peaks : ArrayLike
        (n, 2) array of target peaks.
    angle_threshold : float
        Angle threshold in degrees.
    distance_threshold : float
        Distance threshold.
    verbose : bool
        If True, prints detailed logs during the process.

    Returns
    -------
    ArrayLike
        (n, 2) array of filtered matches.

    Notes
    -----
    Uses the angle and distance thresholds to filter matches.
    The angle threshold is the maximum allowed angle between the source and target peaks.
    The distance threshold is the maximum allowed distance between the source and target peaks.
    The dominant angle is the angle that appears most frequently in the matches.
    """
    if distance_threshold:
        click.echo(f'Filtering matches with distance threshold: {distance_threshold}')
        dist = np.linalg.norm(
            source_peaks[matches[:, 0]] - target_peaks[matches[:, 1]], axis=1
        )
        matches = matches[dist < np.quantile(dist, distance_threshold), :]

    if verbose:
        click.echo(f'Total of matches after distance filtering: {len(matches)}')

    if angle_threshold:
        click.echo(f'Filtering matches with angle threshold: {angle_threshold}')
        vectors = target_peaks[matches[:, 1]] - source_peaks[matches[:, 0]]
        angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles_deg = np.degrees(angles_rad)

        bins = np.linspace(-180, 180, 36)  # 10-degree bins
        hist, bin_edges = np.histogram(angles_deg, bins=bins)

        dominant_bin_index = np.argmax(hist)
        dominant_angle = (
            bin_edges[dominant_bin_index] + bin_edges[dominant_bin_index + 1]
        ) / 2

        filtered_indices = np.where(np.abs(angles_deg - dominant_angle) <= angle_threshold)[0]

        matches = matches[filtered_indices]

    if verbose:
        click.echo(f'Total of matches after angle filtering: {len(matches)}')

    return matches


def estimate_transform(
    matches: ArrayLike,
    source_peaks: ArrayLike,
    target_peaks: ArrayLike,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
) -> ArrayLike:
    """
    Estimate the affine transformation matrix between source and target channels
    based on detected bead matches at a specific timepoint.

    Parameters
    ----------
    matches : ArrayLike
        (n, 2) array of matches.
    source_peaks : ArrayLike
        (n, 2) array of source peaks.
    target_peaks : ArrayLike
        (n, 2) array of target peaks.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, prints detailed logs during the process.

    Returns
    -------
    ArrayLike
        (4, 4) array of the affine transformation matrix.
    """
    if verbose:
        click.echo(f"Estimating transform with settings: {affine_transform_settings}")

    if affine_transform_settings.transform_type == 'affine':
        tform = AffineTransform(dimensionality=3)

    elif affine_transform_settings.transform_type == 'euclidean':
        tform = EuclideanTransform(dimensionality=3)

    elif affine_transform_settings.transform_type == 'similarity':
        tform = SimilarityTransform(dimensionality=3)

    else:
        raise ValueError(f'Unknown transform type: {affine_transform_settings.transform_type}')

    tform.estimate(source_peaks[matches[:, 0]], target_peaks[matches[:, 1]])

    return tform


def estimate_transform_from_beads(
    t_idx: int,
    source_channel_tzyx: da.Array,
    target_channel_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    slurm: bool = False,
    output_folder_path: Path = None,
) -> list | None:
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
    source_channel_tzyx : da.Array
       4D array (T, Z, Y, X) of the source channel (Dask array).
    target_channel_tzyx : da.Array
       4D array (T, Z, Y, X) of the target channel (Dask array).
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

    source_channel_zyx = np.asarray(source_channel_tzyx[t_idx]).astype(np.float32)
    target_channel_zyx = np.asarray(target_channel_tzyx[t_idx]).astype(np.float32)

    if _check_nan_n_zeros(source_channel_zyx) or _check_nan_n_zeros(target_channel_zyx):
        click.echo(f'Beads data is missing at timepoint {t_idx}')
        return

    approx_tform = np.asarray(affine_transform_settings.approx_transform)
    source_data_ants = ants.from_numpy(source_channel_zyx)
    target_data_ants = ants.from_numpy(target_channel_zyx)
    source_data_reg = (
        convert_transform_to_ants(approx_tform)
        .apply_to_image(source_data_ants, reference=target_data_ants)
        .numpy()
    )

    source_peaks, target_peaks = detect_bead_peaks(
        source_channel_zyx=source_data_reg,
        target_channel_zyx=target_channel_zyx,
        source_peaks_settings=beads_match_settings.source_peaks_settings,
        target_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=verbose,
    )

    matches = get_matches_from_beads(
        source_peaks=source_peaks,
        target_peaks=target_peaks,
        beads_match_settings=beads_match_settings,
        verbose=verbose,
    )

    matches = filter_matches(
        matches=matches,
        source_peaks=source_peaks,
        target_peaks=target_peaks,
        angle_threshold=beads_match_settings.filter_angle_threshold,
        distance_threshold=beads_match_settings.filter_distance_threshold,
    )

    if len(matches) < 3:
        click.echo(
            f'Source and target beads were not matches successfully for timepoint {t_idx}'
        )
        return

    tform = estimate_transform(
        matches=matches,
        source_peaks=source_peaks,
        target_peaks=target_peaks,
        affine_transform_settings=affine_transform_settings,
        verbose=verbose,
    )
    compount_tform = np.asarray(approx_tform) @ tform.inverse.params

    if verbose:
        click.echo(f'Matches: {matches}')
        click.echo(f"tform.params: {tform.params}")
        click.echo(f"tform.inverse.params: {tform.inverse.params}")
        click.echo(f"compount_tform: {compount_tform}")

    if slurm:
        print(f"Saving transform to {output_folder_path}")
        output_folder_path.mkdir(parents=True, exist_ok=True)
        np.save(output_folder_path / f"{t_idx}.npy", compount_tform)

    return compount_tform.tolist()


def estimate_registration(
    source_position_dirpaths: list[str],
    target_position_dirpaths: list[str],
    output_filepath: str,
    config_filepath: str,
    registration_target_channel: str,
    registration_source_channel: list[str],
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate the affine transformation between a source and target image for registration.

    This command-line tool uses either bead-based or user-assisted methods to estimate registration
    parameters for aligning source (moving) and target (fixed) images. The output is a configuration
    file that can be used with subsequent tools (`stabilize` and `register`).

    Parameters
    ----------
    source_position_dirpaths : list[str]
        List of file paths to the source channel data in OME-Zarr format.
    target_position_dirpaths : list[str]
        List of file paths to the target channel data in OME-Zarr format.
    output_filepath : str
        Path to save the estimated registration configuration file (YAML).
    num_processes : int
        Number of processes for parallel computations (used in bead-based registration).
    config_filepath : str
        Path to the YAML configuration file for the registration settings.
    registration_target_channel : str
        Name of the target channel to be used when registration params are applied.
    registration_source_channels : list[str]
        List of source channel names to be used when registration params are applied.

    Returns
    -------
    None
        Writes the estimated registration parameters to the specified output file.
    """
    output_dir = Path(output_filepath).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    click.echo(f"Settings: {settings}")

    target_channel_name = settings.target_channel_name
    source_channel_name = settings.source_channel_name
    registration_source_channels = registration_source_channel

    if registration_target_channel is None:
        registration_target_channel = target_channel_name
    if len(registration_source_channels) == 0:
        registration_source_channels = [source_channel_name]

    click.echo(f"Target channel: {target_channel_name}")
    click.echo(f"Source channel: {source_channel_name}")

    with open_ome_zarr(source_position_dirpaths[0], mode="r") as source_channel_position:
        source_channels = source_channel_position.channel_names
        source_channel_index = source_channels.index(source_channel_name)
        source_channel_name = source_channels[source_channel_index]
        source_data = source_channel_position.data.dask_array()
        source_channel_data = source_data[:, source_channel_index]
        source_channel_voxel_size = source_channel_position.scale[-3:]

    with open_ome_zarr(target_position_dirpaths[0], mode="r") as target_channel_position:
        target_channels = target_channel_position.channel_names
        target_channel_index = target_channels.index(target_channel_name)
        target_channel_name = target_channels[target_channel_index]
        target_data = target_channel_position.data.dask_array()
        target_channel_data = target_data[:, target_channel_index]
        voxel_size = target_channel_position.scale
        target_channel_voxel_size = voxel_size[-3:]

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"
    eval_transform_settings = settings.eval_transform_settings

    if settings.estimation_method == "beads":
        transforms = beads_based_registration(
            source_channel_tzyx=source_channel_data,
            target_channel_tzyx=target_channel_data,
            beads_match_settings=settings.beads_match_settings,
            affine_transform_settings=settings.affine_transform_settings,
            verbose=settings.verbose,
            cluster=cluster,
            sbatch_filepath=sbatch_filepath,
            output_folder_path=output_dir,
        )

    elif settings.estimation_method == "ants":
        transforms = ants_registration(
            source_data_tczyx=source_data,
            target_data_tczyx=target_data,
            source_channel_index=source_channel_index,
            target_channel_index=target_channel_index,
            ants_registration_settings=settings.ants_registration_settings,
            affine_transform_settings=settings.affine_transform_settings,
            sbatch_filepath=sbatch_filepath,
            cluster=cluster,
            verbose=settings.verbose,
            output_folder_path=output_dir,
        )

    elif settings.estimation_method == "manual":
        transforms = user_assisted_registration(
            source_channel_volume=np.asarray(
                source_channel_data[settings.manual_registration_settings.time_index]
            ),
            source_channel_name=source_channel_name,
            source_channel_voxel_size=source_channel_voxel_size,
            target_channel_volume=np.asarray(
                target_channel_data[settings.manual_registration_settings.time_index]
            ),
            target_channel_name=target_channel_name,
            target_channel_voxel_size=target_channel_voxel_size,
            similarity=(
                True
                if settings.affine_transform_settings.transform_type == "similarity"
                else False
            ),
            pre_affine_90degree_rotation=settings.manual_registration_settings.affine_90degree_rotation,
            pre_affine_fliplr=settings.manual_registration_settings.affine_fliplr,
        )

    else:
        raise ValueError(
            f"Unknown estimation method: {settings.estimation_method}. "
            "Supported methods are 'beads', 'ants', and 'manual'."
        )

    if len(transforms) == 1:
        if eval_transform_settings:
            click.echo("One transform was estimated, no need to evaluate")
        transform = transforms[0]
        model = RegistrationSettings(
            source_channel_names=registration_source_channels,
            target_channel_name=registration_target_channel,
            affine_transform_zyx=transform,
        )

    else:
        if eval_transform_settings:
            transforms = evaluate_transforms(
                transforms=transforms,
                shape_zyx=source_channel_data.shape[-3:],
                validation_window_size=eval_transform_settings.validation_window_size,
                validation_tolerance=eval_transform_settings.validation_tolerance,
                interpolation_window_size=eval_transform_settings.interpolation_window_size,
                interpolation_type=eval_transform_settings.interpolation_type,
                verbose=settings.verbose,
            )

        model = StabilizationSettings(
            stabilization_estimation_channel='',
            stabilization_type='xyz',
            stabilization_channels=registration_source_channels,
            affine_transform_zyx_list=transforms,
            time_indices='all',
            output_voxel_size=voxel_size,
        )
        if settings.verbose:
            plot_translations(
                transforms_zyx=transforms,
                output_filepath=output_dir
                / "translation_plots"
                / f"{settings.estimation_method}_registration.png",
            )

    model_to_yaml(model, output_filepath)

    click.echo(f"Registration settings saved to {output_dir.resolve()}")


@click.command("estimate-registration")
@source_position_dirpaths()
@target_position_dirpaths()
@output_filepath()
@config_filepath()
@sbatch_filepath()
@local()
@click.option(
    "--registration-target-channel",
    "-rt",
    type=str,
    help="Name of the target channel to be used when registration params are applied. If not provided, the target channel from the config file will be used.",
    required=False,
)
@click.option(
    "--registration-source-channel",
    "-rs",
    type=str,
    multiple=True,
    help="Name of the source channels to be used when registration params are applied. May be passed multiple times. If not provided, the source channels from the config file will be used.",
    required=False,
)
def estimate_registration_cli(
    source_position_dirpaths: list[str],
    target_position_dirpaths: list[str],
    output_filepath: str,
    config_filepath: Path,
    registration_target_channel: str,
    registration_source_channel: list[str],
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate the affine transformation between a source and target image for registration.

    This command-line tools estimates the registration transforms between a source (moving) and target (fixed) image
    using either (1) user input, (2) images or registration beads, or (3) image features via the ANTS registration library.
    The output is a configuration file that can be used with subsequent tools (`stabilize` and `register`).

    User-assisted registration requires manual selection of corresponding features in source and target images.
    Bead-based registration uses detected bead matches across timepoints to compute affine transformations.
    ANTs-based registration uses the ANTsPy library to estimate transformations based on image features. Optionally,
    a Sobel filter may be applied to the data to enhance feature detection between label-free and fluorescent channels.

    Example:
    >> biahub estimate-registration
        -s ./acq_name_labelfree_reconstructed.zarr/0/0/0   # Source channel OME-Zarr data path
        -t ./acq_name_lightsheet_deskewed.zarr/0/0/0       # Target channel OME-Zarr data path
        -o ./output.yml                                    # Output configuration file path
        --config ./config.yml                              # Path to input configuration file
        --registration-target-channel "Phase3D"            # Name of the target channel
        --registration-source-channel "GFP"                # Names of source channel
        --registration-source-channel "mCherry"            # Names of another source channel
    """

    estimate_registration(
        source_position_dirpaths=source_position_dirpaths,
        target_position_dirpaths=target_position_dirpaths,
        output_filepath=output_filepath,
        config_filepath=config_filepath,
        registration_target_channel=registration_target_channel,
        registration_source_channel=registration_source_channel,
        sbatch_filepath=sbatch_filepath,
        local=local,
    )


if __name__ == "__main__":
    estimate_registration_cli()
