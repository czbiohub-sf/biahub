from functools import partial
from multiprocessing import Pool
import subprocess
import time
from pathlib import Path
import ants
import click
import dask.array as da
import napari
import numpy as np
import submitit
from datetime import datetime
from scipy.ndimage import gaussian_filter
from iohub import open_ome_zarr
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import match_descriptors
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform
from sklearn.neighbors import NearestNeighbors
from waveorder.focus import focus_from_transverse_band
import os
from biahub.analysis.AnalysisSettings import (
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)
from biahub.analysis.analyze_psf import detect_peaks

from biahub.analysis.register import (
    convert_transform_to_ants,
    convert_transform_to_numpy,
    get_3D_rescaling_matrix,
    get_3D_rotation_matrix,
)
from biahub.cli.parsing import (
    config_filepath,
    num_processes,
    output_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
    sbatch_filepath,
    sbatch_to_submitit,
    local
)
from biahub.cli.utils import estimate_resources, _check_nan_n_zeros, model_to_yaml, yaml_to_model

# TODO: see if at some point these globals should be hidden or exposed.
NA_DETECTION_SOURCE = 1.35
NA_DETECTION_TARGET = 1.35
WAVELENGTH_EMISSION_SOURCE_CHANNEL = 0.45  # in um
WAVELENGTH_EMISSION_TARGET_CHANNEL = 0.6  # in um
FOCUS_SLICE_ROI_WIDTH = 150  # size of central ROI used to find focal slice

COLOR_CYCLE = [
    "white",
    "cyan",
    "lime",
    "orchid",
    "blue",
    "orange",
    "yellow",
    "magenta",
]

def wait_for_jobs_to_finish(job_ids, sleep_time=60):
    """Wait for SLURM jobs to finish."""
    print(f"Waiting for jobs: {', '.join(job_ids)} to finish...")
    while True:
        result = subprocess.run(
            ["squeue", "--job", ",".join(job_ids)], stdout=subprocess.PIPE, text=True
        )
        if len(result.stdout.strip().split("\n")) <= 1:  # No jobs found
            print("All jobs completed.")
            break
        else:
            print("Jobs still running...")
            time.sleep(sleep_time)  # Wait sleep_time seconds before checking again

def user_assisted_registration(
    source_channel_volume,
    source_channel_name,
    source_channel_voxel_size,
    target_channel_volume,
    target_channel_name,
    target_channel_voxel_size,
    similarity=False,
    pre_affine_90degree_rotation=0,
):
    """
    Perform user-assisted registration of two volumetric image channels.

    This function allows users to manually annotate corresponding features between
    a source and target channel to calculate an affine transformation matrix for registration.

    Parameters:
    - source_channel_volume (np.ndarray): 3D array (Z, Y, X) of the source channel.
    - source_channel_name (str): Name of the source channel for display purposes.
    - source_channel_voxel_size (tuple): Voxel size of the source channel (Z, Y, X).
    - target_channel_volume (np.ndarray): 3D array (Z, Y, X) of the target channel.
    - target_channel_name (str): Name of the target channel for display purposes.
    - target_channel_voxel_size (tuple): Voxel size of the target channel (Z, Y, X).
    - similarity (bool): If True, use a similarity transform (rotation, translation, scaling);
                         if False, use an Euclidean transform (rotation, translation).
    - pre_affine_90degree_rotation (int): Number of 90-degree rotations to apply to the source channel
                                          before registration.

    Returns:
    - np.ndarray: The estimated 4x4 affine transformation matrix for registering the source channel to the target channel.

    Notes:
    - The function uses a Napari viewer for manual feature annotation.
    - Scaling factors for voxel size differences between source and target are calculated and applied.
    - Users must annotate at least three corresponding points in both channels for the transform calculation.
    - Two types of transformations are supported: similarity (with scaling) and Euclidean (no scaling).
    - The function visually displays intermediate and final results in Napari for user validation.
    """

    # Find the in-focus slice
    source_channel_Z, source_channel_Y, source_channel_X = source_channel_volume.shape[-3:]
    target_channel_Z, target_channel_Y, target_channel_X = target_channel_volume.shape[-3:]

    source_channel_focus_idx = focus_from_transverse_band(
        source_channel_volume[
            :,
            source_channel_Y // 2
            - FOCUS_SLICE_ROI_WIDTH : source_channel_Y // 2
            + FOCUS_SLICE_ROI_WIDTH,
            source_channel_X // 2
            - FOCUS_SLICE_ROI_WIDTH : source_channel_X // 2
            + FOCUS_SLICE_ROI_WIDTH,
        ],
        NA_det=NA_DETECTION_SOURCE,
        lambda_ill=WAVELENGTH_EMISSION_SOURCE_CHANNEL,
        pixel_size=source_channel_voxel_size[-1],
    )

    target_channel_focus_idx = focus_from_transverse_band(
        target_channel_volume[
            :,
            target_channel_Y // 2
            - FOCUS_SLICE_ROI_WIDTH : target_channel_Y // 2
            + FOCUS_SLICE_ROI_WIDTH,
            target_channel_X // 2
            - FOCUS_SLICE_ROI_WIDTH : target_channel_X // 2
            + FOCUS_SLICE_ROI_WIDTH,
        ],
        NA_det=NA_DETECTION_TARGET,
        lambda_ill=WAVELENGTH_EMISSION_TARGET_CHANNEL,
        pixel_size=target_channel_voxel_size[-1],
    )

    if source_channel_focus_idx not in (0, source_channel_Z - 1):
        click.echo(f"Best source channel focus slice: {source_channel_focus_idx}")
    else:
        source_channel_focus_idx = source_channel_Z // 2
        click.echo(
            f"Could not determine best source channel focus slice, using {source_channel_focus_idx}"
        )

    if target_channel_focus_idx not in (0, target_channel_Z - 1):
        click.echo(f"Best target channel focus slice: {target_channel_focus_idx}")
    else:
        target_channel_focus_idx = target_channel_Z // 2
        click.echo(
            f"Could not determine best target channel focus slice, using {target_channel_focus_idx}"
        )

    # Calculate scaling factors for displaying data
    scaling_factor_z = source_channel_voxel_size[-3] / target_channel_voxel_size[-3]
    scaling_factor_yx = source_channel_voxel_size[-1] / target_channel_voxel_size[-1]
    click.echo(
        f"Z scaling factor: {scaling_factor_z:.3f}; XY scaling factor: {scaling_factor_yx:.3f}\n"
    )

    # Add layers to napari with and transform
    # Rotate the image if needed here
    # Convert to ants objects
    source_zyx_ants = ants.from_numpy(source_channel_volume.astype(np.float32))
    target_zyx_ants = ants.from_numpy(target_channel_volume.astype(np.float32))

    scaling_affine = get_3D_rescaling_matrix(
        (target_channel_Z, target_channel_Y, target_channel_X),
        (scaling_factor_z, scaling_factor_yx, scaling_factor_yx),
        (target_channel_Z, target_channel_Y, target_channel_X),
    )
    rotate90_affine = get_3D_rotation_matrix(
        (source_channel_Z, source_channel_Y, source_channel_X),
        90 * pre_affine_90degree_rotation,
        (target_channel_Z, target_channel_Y, target_channel_X),
    )
    compound_affine = scaling_affine @ rotate90_affine
    tx_manual = convert_transform_to_ants(compound_affine).invert()

    source_zxy_pre_reg = tx_manual.apply_to_image(source_zyx_ants, reference=target_zyx_ants)

    # Get a napari viewer
    viewer = napari.Viewer()

    viewer.add_image(target_channel_volume, name=f"target_{target_channel_name}")
    points_target_channel = viewer.add_points(
        ndim=3, name=f"pts_target_{target_channel_name}", size=20, face_color=COLOR_CYCLE[0]
    )

    source_layer = viewer.add_image(
        source_zxy_pre_reg.numpy(),
        name=f"source_{source_channel_name}",
        blending='additive',
        colormap='green',
    )
    points_source_channel = viewer.add_points(
        ndim=3, name=f"pts_source_{source_channel_name}", size=20, face_color=COLOR_CYCLE[0]
    )

    # setup viewer
    viewer.layers.selection.active = points_source_channel
    viewer.grid.enabled = False
    viewer.grid.stride = 2
    viewer.grid.shape = (-1, 2)
    points_source_channel.mode = "add"
    points_target_channel.mode = "add"

    # Manual annotation of features
    def next_on_click(layer, event, in_focus):
        if layer.mode == "add":
            if layer is points_source_channel:
                next_layer = points_target_channel
                # Change slider value
                if len(next_layer.data) < 1:
                    prev_step_target_channel = (
                        in_focus[1],
                        0,
                        0,
                    )
                else:
                    prev_step_target_channel = (next_layer.data[-1][0], 0, 0)
                # Add a point to the active layer
                # viewer.cursor.position is return in world coordinates
                # point position needs to be converted to data coordinates before plotting
                # on top of layer
                cursor_position_data_coords = layer.world_to_data(viewer.cursor.position)
                layer.add(cursor_position_data_coords)

                # Change the colors
                current_index = COLOR_CYCLE.index(layer.current_face_color)
                next_index = (current_index) % len(COLOR_CYCLE)
                next_color = COLOR_CYCLE[next_index]
                next_layer.current_face_color = next_color

                # Switch to the next layer
                next_layer.mode = "add"
                layer.selected_data = {}
                viewer.layers.selection.active = next_layer
                viewer.dims.current_step = prev_step_target_channel

            else:
                next_layer = points_source_channel
                # Change slider value
                if len(next_layer.data) < 1:
                    prev_step_source_channel = (
                        in_focus[0] * scaling_factor_z,
                        0,
                        0,
                    )
                else:
                    # TODO: this +1 is not clear to me?
                    prev_step_source_channel = (next_layer.data[-1][0], 0, 0)
                cursor_position_data_coords = layer.world_to_data(viewer.cursor.position)
                layer.add(cursor_position_data_coords)
                # Change the colors
                current_index = COLOR_CYCLE.index(layer.current_face_color)
                next_index = (current_index + 1) % len(COLOR_CYCLE)
                next_color = COLOR_CYCLE[next_index]
                next_layer.current_face_color = next_color

                # Switch to the next layer
                next_layer.mode = "add"
                layer.selected_data = {}
                viewer.layers.selection.active = next_layer
                viewer.dims.current_step = prev_step_source_channel

    # Bind the mouse click callback to both point layers
    in_focus = (source_channel_focus_idx, target_channel_focus_idx)

    def lambda_callback(layer, event):
        return next_on_click(layer=layer, event=event, in_focus=in_focus)

    viewer.dims.current_step = (
        in_focus[0] * scaling_factor_z,
        0,
        0,
    )
    points_source_channel.mouse_drag_callbacks.append(lambda_callback)
    points_target_channel.mouse_drag_callbacks.append(lambda_callback)

    input(
        "Add at least three points in the two channels by sequentially clicking "
        + "on a feature in the source channel and its corresponding feature in target channel. "
        + "Select grid mode if you prefer side-by-side view. "
        + "Press <enter> when done..."
    )

    # Get the data from the layers
    pts_source_channel_data = points_source_channel.data
    pts_target_channel_data = points_target_channel.data

    # Estimate the affine transform between the points xy to make sure registration is good
    if similarity:
        # Similarity transform (rotation, translation, scaling)
        transform = SimilarityTransform()
        transform.estimate(pts_source_channel_data, pts_target_channel_data)
        manual_estimated_transform = transform.params @ compound_affine

    else:
        # Euclidean transform (rotation, translation) limiting this dataset's scale and just z-translation
        transform = EuclideanTransform()
        transform.estimate(pts_source_channel_data[:, 1:], pts_target_channel_data[:, 1:])
        yx_points_transformation_matrix = transform.params

        z_translation = pts_target_channel_data[0, 0] - pts_source_channel_data[0, 0]

        z_scale_translate_matrix = np.array([[1, 0, 0, z_translation]])

        # 2D to 3D matrix
        euclidian_transform = np.vstack(
            (
                z_scale_translate_matrix,
                np.insert(yx_points_transformation_matrix, 0, 0, axis=1),
            )
        )  # Insert 0 in the third entry of each row
        manual_estimated_transform = euclidian_transform @ compound_affine

    tx_manual = convert_transform_to_ants(manual_estimated_transform).invert()

    source_zxy_manual_reg = tx_manual.apply_to_image(
        source_zyx_ants, reference=target_zyx_ants
    )

    click.echo("\nShowing registered source image in magenta")
    viewer.grid.enabled = False
    viewer.add_image(
        source_zxy_manual_reg.numpy(),
        name=f"registered_{source_channel_name}",
        colormap="magenta",
        blending='additive',
    )
    # Cleanup
    viewer.layers.remove(points_source_channel)
    viewer.layers.remove(points_target_channel)
    source_layer.visible = False

    # Ants affine transforms
    tform = convert_transform_to_numpy(tx_manual)
    click.echo(f'Estimated affine transformation matrix:\n{tform}\n')
    input("Press <Enter> to close the viewer and exit...")
    viewer.close()

    return tform


def beads_based_registration(
    source_channel_tzyx: da.Array,
    target_channel_tzyx: da.Array,
    approx_tform: list,
    match_algorithm: str = 'hungarian',
    match_filter_angle_threshold: float = 0,
    match_max_ratio: float = 0.6,
    transform_type: str = 'affine',
    hungarian_knn_k: int = 5,
    validation_window_size: int = 10,
    validation_tolerance: float = 100.0,
    interpolation_window_size: int = 3,
    interpolation_type: str = 'linear',
    verbose: bool = False,
    cluster: bool = False,
    sbatch_filepath: Path = None,
    output_folder_path: Path = None,
):
    """
    Perform beads-based temporal registration of 4D data using affine transformations.

    This function calculates timepoint-specific affine transformations to align a source channel
    to a target channel in 4D (T, Z, Y, X) data. It validates, smooths, and interpolates transformations
    across timepoints for consistent registration.

    Parameters:
    - source_channel_tzyx (da.Array): 4D array (T, Z, Y, X) of the source channel (Dask array).
    - target_channel_tzyx (da.Array): 4D array (T, Z, Y, X) of the target channel (Dask array).
    - approx_tform (list): Initial approximate affine transform (4x4 matrix) for guiding registration.
    - num_processes (int): Number of parallel processes for transformation computation.
    - window_size (int): Size of the moving window for smoothing transformations.
    - tolerance (float): Maximum allowed difference between consecutive transformations for validation.
    - angle_threshold (int): Threshold for filtering outliers in detected bead matches (in degrees).
    - verbose (bool): If True, prints detailed logs of the registration process.

    Returns:
    - transforms (list): List of affine transformation matrices (4x4), one for each timepoint.
                         Invalid or missing transformations are interpolated.

    Notes:
    - Each timepoint is processed in parallel using a multiprocessing pool.
    - Transformations are smoothed with a moving average window and validated against a reference.
    - Missing transformations are interpolated linearly across timepoints.
    - Use verbose=True for detailed logging during registration.
    """



    (T, Z, Y, X) = source_channel_tzyx.shape


    # Compute transformations in parallel

    num_cpus, gb_ram_per_cpu = estimate_resources(shape=(T,2,Z,Y,X), ram_multiplier=5, max_num_cpus=16)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_focus_z",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 5,
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

    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")
    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    jobs = []
    with executor.batch():
        for t in range(T):
            job = executor.submit(_get_tform_from_beads,
                approx_tform=approx_tform,
                source_channel_tzyx=source_channel_tzyx,
                target_channel_tzyx=target_channel_tzyx,
                match_filter_angle_threshold=match_filter_angle_threshold,
                hungarian_knn_k=hungarian_knn_k,
                verbose=verbose,
                match_algorithm=match_algorithm,
                transform_type=transform_type,
                slurm = True,
                output_folder_path=output_transforms_path,
                match_max_ratio=match_max_ratio,
                t_idx=t)
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")
    # Wait for all jobs to finish
    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)
    # Get list of .npy transform files


    # Load and collect all transform arrays
    transforms = []

    for t in range(T):
        file_path = output_transforms_path/f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
        else:
            T_zyx_shift = np.load(file_path).tolist()
            transforms.append(T_zyx_shift)

    click.echo(f"Before validate Transforms: {transforms[0]}")

    # # Validate and filter transforms
    transforms = _validate_transforms(
        transforms=transforms,
        window_size=validation_window_size,
        tolerance=validation_tolerance,
        Z=Z,
        Y=Y,
        X=X,
        verbose=verbose,
    )
    click.echo(f"After validation transforms: {transforms[0]}")


    # Interpolate missing transforms
    transforms = _interpolate_transforms(
        transforms=transforms,
        window_size=interpolation_window_size,
        interpolation_type=interpolation_type,
        verbose=verbose,
    )
    click.echo(f"After interpolation transforms: {transforms[0]}")


    return transforms


def _validate_transforms(transforms, Z, Y, X, window_size=10, tolerance=100.0, verbose=False):
    """
    Validate a list of affine transformation matrices by smoothing and filtering.

    This function validates a list of affine transformation matrices by smoothing them
    with a moving average window and filtering out invalid or inconsistent transformations based on a tolerance threshold.

    Parameters:
    - transforms (list): List of affine transformation matrices (4x4), one for each timepoint.
    - window_size (int): Size of the moving window for smoothing transformations.
    - tolerance (float): Maximum allowed difference between consecutive transformations for validation.
    - Z (int): Number of slices in the Z dimension.
    - Y (int): Number of pixels in the Y dimension.
    - X (int): Number of pixels in the X dimension.
    - verbose (bool): If True, prints detailed logs of the validation process.

    Returns:
    - list: List of affine transformation matrices with invalid or inconsistent values
            replaced by None.
    """
    valid_transforms = []
    reference_transform = None
    for i in range(len(transforms)):
        if transforms[i] is not None:
            if reference_transform is None or _check_transform_difference(
                transforms[i], reference_transform, (Z, Y, X), tolerance, verbose
            ):
                valid_transforms.append(transforms[i])
                if len(valid_transforms) > window_size:
                    valid_transforms.pop(0)
                reference_transform = np.mean(valid_transforms, axis=0)
                if verbose:
                    click.echo(f"Transform at timepoint {i} is valid")
            else:
                if verbose:
                    click.echo(f'Transform at timepoint {i} will be interpolated')
                transforms[i] = None

        else:
            if verbose:
                click.echo(f'Transform at timepoint {i} is None, will be interpolated')
            transforms[i] = None
    return transforms


def _interpolate_transforms(
    transforms, window_size=3, interpolation_type='linear', verbose=False
):
    """
    Interpolate missing transforms (None) in a list of affine transformation matrices.

    Parameters:
    - transforms (list of 4x4 arrays or None): One transform per timepoint.
    - window (int): Local window radius for interpolation. If 0, global interpolation is used.

    Returns:
    - list: Transforms with missing values filled via linear interpolation.
    """
    n = len(transforms)
    valid_indices = [i for i, t in enumerate(transforms) if t is not None]
    valid_transforms = [np.array(transforms[i]) for i in valid_indices]

    if not valid_indices or len(valid_indices) < 2:
        raise ValueError("At least two valid transforms are required for interpolation.")

    missing_indices = [i for i in range(n) if transforms[i] is None]

    if not missing_indices:
        return transforms  # nothing to do
    if verbose:
        click.echo(f"Interpolating missing transforms at timepoints: {missing_indices}")

    if window_size > 0:
        for idx in missing_indices:
            # Define local window
            start = max(0, idx - window_size)
            end = min(n, idx + window_size + 1)

            local_x = []
            local_y = []

            for j in range(start, end):
                if transforms[j] is not None:
                    local_x.append(j)
                    local_y.append(np.array(transforms[j]))

            if len(local_x) < 2:
                if verbose:
                    click.echo(
                        f"Skipping timepoint {idx}: only {len(local_x)} neighbors found."
                    )
                continue

            f = interp1d(
                local_x, local_y, axis=0, kind=interpolation_type, fill_value='extrapolate'
            )
            transforms[idx] = f(idx).tolist()
            if verbose:
                click.echo(f"Interpolated timepoint {idx} using neighbors: {local_x}")

    else:
        # Global interpolation using all valid transforms
        f = interp1d(
            valid_indices, valid_transforms, axis=0, kind='linear', fill_value='extrapolate'
        )
        transforms = [
            f(i).tolist() if transforms[i] is None else transforms[i] for i in range(n)
        ]

    return transforms


def _check_transform_difference(tform1, tform2, shape, threshold=5.0, verbose=False):
    """
    Evaluate the difference between two affine transforms by calculating the
    Mean Squared Error (MSE) of a grid of points transformed by each matrix.

    Parameters:
    - tform1: First affine transform (4x4 matrix).
    - tform2: Second affine transform (4x4 matrix).
    - shape: Shape of the source (i.e. moving) volume (Z, Y, X).
    - threshold: The maximum allowed MSE difference.
    - verbose: Flag to print the MSE difference.

    Returns:
    - bool: True if the MSE difference is within the threshold, False otherwise.
    """
    tform1 = np.array(tform1)
    tform2 = np.array(tform2)
    (Z, Y, X) = shape

    zz, yy, xx = np.meshgrid(
        np.linspace(0, Z - 1, 10), np.linspace(0, Y - 1, 10), np.linspace(0, X - 1, 10)
    )

    grid_points = np.vstack([zz.ravel(), yy.ravel(), xx.ravel(), np.ones(zz.size)]).T

    points_tform1 = np.dot(tform1, grid_points.T).T
    points_tform2 = np.dot(tform2, grid_points.T).T

    differences = np.linalg.norm(points_tform1[:, :3] - points_tform2[:, :3], axis=1)
    mse = np.mean(differences)

    if verbose:
        click.echo(f'MSE of transformed points: {mse:.2f}; threshold: {threshold:.2f}')
    return mse <= threshold


def _compute_cost_matrix(
    source_peaks,
    target_peaks,
    source_edges,
    target_edges,
    distance_metric='euclidean',
    distance_weight=0.5,
    nodes_angle_weight=1.0,
    nodes_distance_weight=1.0,
):
    """
    Compute a cost matrix for matching peaks between two graphs based on:
    - Euclidean or other distance between peaks
    - Consistency in edge distances
    - Consistency in edge angles

    Parameters:
    - source_peaks (ndarray): (n, 2) array of source node coordinates.
    - target_peaks (ndarray): (m, 2) array of target node coordinates.
    - source_edges (list of tuple): List of edges (i, j) in source graph.
    - target_edges (list of tuple): List of edges (i, j) in target graph.
    - distance_metric (str): Metric for direct point-to-point distances.
    - distance_weight (float): Weight for point distance cost.
    - nodes_angle_weight (float): Weight for angular consistency cost.
    - nodes_distance_weight (float): Weight for local edge distance cost.

    Returns:
    - ndarray: Cost matrix of shape (n, m).
    """
    n, m = len(source_peaks), len(target_peaks)

    def compute_edge_attributes(peaks, edges):
        distances = {}
        angles = {}
        for i, j in edges:
            vec = peaks[j] - peaks[i]
            d = np.linalg.norm(vec)
            angle = np.arctan2(vec[1], vec[0])
            distances[(i, j)] = distances[(j, i)] = d
            angles[(i, j)] = angles[(j, i)] = angle
        return distances, angles

    def local_edge_costs(
        source_edges, target_edges, source_attrs, target_attrs, attr='distance', default=1e6
    ):
        cost_matrix = np.full((n, m), default)
        for i in range(n):
            s_neighbors = [j for a, j in source_edges if a == i]
            for j in range(m):
                t_neighbors = [k for a, k in target_edges if a == j]
                common_len = min(len(s_neighbors), len(t_neighbors))
                diffs = []
                for k in range(common_len):
                    s_edge = (i, s_neighbors[k])
                    t_edge = (j, t_neighbors[k])
                    if s_edge in source_attrs and t_edge in target_attrs:
                        v1 = source_attrs[s_edge]
                        v2 = target_attrs[t_edge]
                        if attr == 'angle':
                            diff = np.abs(v1 - v2)
                        else:  # distance
                            diff = np.abs(v1 - v2)
                        diffs.append(diff)
                cost_matrix[i, j] = np.mean(diffs) if diffs else default
        return cost_matrix

    # Compute direct point-wise distance
    C_dist = cdist(source_peaks, target_peaks, metric=distance_metric)

    # Compute edge distances and angles
    source_dists, source_angles = compute_edge_attributes(source_peaks, source_edges)
    target_dists, target_angles = compute_edge_attributes(target_peaks, target_edges)

    # Compute local consistency costs
    C_dist_node = local_edge_costs(
        source_edges, target_edges, source_dists, target_dists, attr='distance', default=1e6
    )
    C_angle_node = local_edge_costs(
        source_edges, target_edges, source_angles, target_angles, attr='angle', default=np.pi
    )

    # Combine all costs
    C_total = (
        distance_weight * C_dist
        + nodes_angle_weight * C_angle_node
        + nodes_distance_weight * C_dist_node
    )

    return C_total


def _knn_edges(points, k=5):
    n_points = len(points)
    if n_points <= 1:
        return []

    k_eff = min(k, n_points - 1)  # Prevent k >= n
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(points)  # +1 includes self
    _, indices = nbrs.kneighbors(points)

    edges = [(i, j) for i, neighbors in enumerate(indices) for j in neighbors if i != j]
    return edges

def match_hungarian(
    C,
    cost_threshold=1e5,
    dummy_cost=1e6,
    max_ratio=None,
):
    """
    Runs Hungarian matching with padding for unequal-sized graphs,
    optionally applying max_ratio filtering similar to match_descriptors.

    Parameters:
        C (ndarray): Cost matrix of shape (n_A, n_B).
        cost_threshold (float): Maximum cost to consider a valid match.
        dummy_cost (float): Cost assigned to dummy nodes (must be > cost_threshold).
        max_ratio (float, optional): Maximum allowed ratio between best and second-best cost.

    Returns:
        matches (ndarray): Array of shape (N_matches, 2) with valid (A_idx, B_idx) pairs.
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


def peaks_to_image(peaks, shape, sigma=2.0):
    """Convert peak coordinates to a blurred image."""
    img = np.zeros(shape, dtype=np.float32)
    for z, y, x in peaks:
        z, y, x = map(int, [z, y, x])
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            img[z, y, x] = 1.0
    return gaussian_filter(img, sigma=sigma)


def register_peak_clouds_ants(source_peaks, target_peaks, shape, sigma=10.0, transform_type="Affine"):
    source_img = peaks_to_image(source_peaks, shape, sigma=sigma)
    target_img = peaks_to_image(target_peaks, shape, sigma=sigma)

    ants_source = ants.from_numpy(source_img)
    ants_target = ants.from_numpy(target_img)

    reg = ants.registration(
        fixed=ants_target,
        moving=ants_source,
        type_of_transform="Similarity",
        aff_metric="MI",
        verbose=True,
    
    )

    return reg["fwdtransforms"], reg["fwdtransforms"][0]  # ANTs filenames + object

def extract_patch(volume, center, patch_size=11):
    """Extract a cubic patch centered at `center` from `volume`."""
    half = patch_size // 2
    z, y, x = map(int, center)
    patch = volume[
        max(z - half, 0): z + half + 1,
        max(y - half, 0): y + half + 1,
        max(x - half, 0): x + half + 1
    ]
    return patch

def compute_mi(p1, p2, bins=32):
    """Compute mutual information between two patches."""
    hgram, _, _ = np.histogram2d(p1.ravel(), p2.ravel(), bins=bins)
    pxy = hgram / np.sum(hgram)
    px = np.sum(pxy, axis=1)  # marginal for x
    py = np.sum(pxy, axis=0)  # marginal for y
    px_py = np.outer(px, py)
    nz = pxy > 0
    return np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))

def match_mutual_information(source_peaks, target_peaks, source_vol, target_vol, patch_size=11):
    """Match peaks using mutual information between local patches."""
    n_source, n_target = len(source_peaks), len(target_peaks)
    mi_matrix = np.zeros((n_source, n_target))

    for i, s_peak in enumerate(source_peaks):
        patch_s = extract_patch(source_vol, s_peak, patch_size=patch_size)
        for j, t_peak in enumerate(target_peaks):
            patch_t = extract_patch(target_vol, t_peak, patch_size=patch_size)
            if patch_s.shape == patch_t.shape:
                mi_matrix[i, j] = compute_mi(patch_s, patch_t)
            else:
                mi_matrix[i, j] = -np.inf  # incompatible shapes, ignore

    # Hungarian matching: convert similarity to cost
    cost_matrix = -mi_matrix
    matches = match_hungarian(cost_matrix)
    return matches
def _get_tform_from_beads(
    t_idx: int,
    approx_tform: list,
    source_channel_tzyx: da.Array,
    target_channel_tzyx: da.Array,
    source_block_size: list = [32, 16, 16],
    source_threshold_abs: int = 110,
    source_nms_distance: int = 16,
    source_min_distance: int = 0,
    target_block_size: list = [32, 16, 16],
    target_threshold_abs: float = 0.8,
    target_nms_distance: int = 16,
    target_min_distance: int = 0,
    match_algorithm: str = 'hungarian',
    match_cross_check: bool = True,
    match_metric: str = 'euclidean',
    match_max_ratio: float = 0.6,
    match_filter_angle_threshold: float = 0,
    transform_type: str = 'affine',
    hungarian_knn_k: int = 5,
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

    Parameters:
    - approx_tform (list): Approximate initial affine transformation matrix (4x4).
    - source_channel_tzyx (da.Array): 4D array (T, Z, Y, X) of the source channel (Dask array).
    - target_channel_tzyx (da.Array): 4D array (T, Z, Y, X) of the target channel (Dask array).
    - angle_threshold (int): Threshold (in degrees) to filter bead matches based on direction.
    - verbose (bool): If True, prints detailed logs during the process.
    - t_idx (int): Timepoint index to process.
    - source_block_size (list): Block size for bead detection in the source dataset.
    - source_threshold_abs (int): Threshold for bead detection in the source dataset.
    - source_nms_distance (int): Non-maximum suppression distance for source dataset.
    - source_min_distance (int): Minimum distance between beads in the source dataset.
    - target_block_size (list): Block size for bead detection in the target dataset.
    - target_threshold_abs (float): Threshold for bead detection in the target dataset.
    - target_nms_distance (int): Non-maximum suppression distance for target dataset.
    - target_min_distance (int): Minimum distance between beads in the target dataset.
    - transform_type (str): Type of transformation to apply (Affine, Similarity, Euclidean).
    - match_algorithm (str): Matching algorithm to use (match_descriptor, hungarian).
    - match_cross_check (bool): If True, perform cross-checking of matches.
    - match_metric (str): Distance metric to use for matching (euclidean, manhattan, cosine).
    - match_max_ratio (float): Maximum ratio of the second-best match to the best match, in match_descriptor.

    Returns:
    - list | None: A 4x4 affine transformation matrix as a nested list if successful,
                   or None if no valid transformation could be calculated.

    Notes:
    - Uses ANTsPy for initial transformation application and bead detection.
    - Peaks (beads) are detected using a block-based algorithm with thresholds for source and target datasets.
    - Bead matches are filtered based on distance and angular deviation from the dominant direction.
    - If fewer than three matches are found after filtering, the function returns None.
    """

    approx_tform = np.asarray(approx_tform)
    source_channel_zyx = np.asarray(source_channel_tzyx[t_idx]).astype(np.float32)
    target_channel_zyx = np.asarray(target_channel_tzyx[t_idx]).astype(np.float32)

    if _check_nan_n_zeros(source_channel_zyx) or _check_nan_n_zeros(target_channel_zyx):
        click.echo(f'Beads data is missing at timepoint {t_idx}')
        return

    source_data_ants = ants.from_numpy(source_channel_zyx)
    target_data_ants = ants.from_numpy(target_channel_zyx)
    source_data_reg = (
        convert_transform_to_ants(approx_tform)
        .apply_to_image(source_data_ants, reference=target_data_ants)
        .numpy()
    )
    click.echo(f'Detecting beads for timepoint {t_idx}')
    if verbose:
        click.echo('Detecting beads in source dataset:')

    source_peaks = detect_peaks(
        source_data_reg,
        block_size=source_block_size,
        threshold_abs=source_threshold_abs,
        nms_distance=source_nms_distance,
        min_distance=source_min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo('Detecting beads in target dataset:')

    target_peaks = detect_peaks(
        target_channel_zyx,
        block_size=target_block_size,
        threshold_abs=target_threshold_abs,
        nms_distance=target_nms_distance,
        min_distance=target_min_distance,
        verbose=verbose,
    )

    # Skip if there is no peak detected
    if len(source_peaks) < 2 or len(target_peaks) < 2:
        click.echo(f'No beads were detected at timepoint {t_idx}')
        return

    if match_algorithm == 'match_descriptor':
        print("Using match descriptor")

        # Match peaks, excluding top 5% of distances as outliers
        matches = match_descriptors(
            source_peaks,
            target_peaks,
            metric=match_metric,
            max_ratio=match_max_ratio,
            cross_check=match_cross_check,
        )
    elif match_algorithm == 'hungarian':

        source_edges = _knn_edges(source_peaks, k=hungarian_knn_k)
        target_edges = _knn_edges(target_peaks, k=hungarian_knn_k)

        if match_cross_check:
            # Step 1: A → B
            C_ab = _compute_cost_matrix(
                source_peaks,
                target_peaks,
                source_edges,
                target_edges
            )
            matches_ab = match_hungarian(C_ab, cost_threshold=np.quantile(C_ab, 0.10), max_ratio=match_max_ratio)

            # Step 2: B → A (swap arguments)
            C_ba = _compute_cost_matrix(
                target_peaks,
                source_peaks,
                target_edges,
                source_edges,
                distance_metric=match_metric,
            )
            matches_ba = match_hungarian(C_ba, cost_threshold=np.quantile(C_ba, 0.10), max_ratio=match_max_ratio)

            # Step 3: Invert matches_ba to compare
            reverse_map = {(j, i) for i, j in matches_ba}

            # Step 4: Keep only symmetric matches
            matches = np.array([[i, j] for i, j in matches_ab if (i, j) in reverse_map])
        else:
            # # Compute cost matrix
            C = _compute_cost_matrix(source_peaks, target_peaks, source_edges, target_edges)

            matches = match_hungarian(C, cost_threshold=np.quantile(C, 0.10))
            
    elif match_algorithm == 'mutual_info_gauss':
        click.echo('Using mutual information with Gaussian smoothing')
        shape = source_data_reg.shape  # Z, Y, X
        ants_tforms, ants_affine_path = register_peak_clouds_ants(
            source_peaks=source_peaks,
            target_peaks=target_peaks,
            shape=shape,
            sigma=2.0,
            transform_type='Similarity',
        )

        ants_tform = ants.read_transform(ants_affine_path)
        tform_np = convert_transform_to_numpy(ants_tform)

        compount_tform = approx_tform @ tform_np
        if slurm:
            output_folder_path.mkdir(parents=True, exist_ok=True)
            np.save(output_folder_path/ f"{t_idx}.npy", compount_tform)

        return compount_tform.tolist()
    elif match_algorithm == 'mutual_information':
        matches = match_mutual_information(
            source_peaks=source_peaks,
            target_peaks=target_peaks,
            source_vol=source_data_reg,
            target_vol=target_channel_zyx,
            patch_size=11,
        )

    if verbose:
        click.echo(f'Total of matches at time point {t_idx}: {len(matches)}')
    dist = np.linalg.norm(source_peaks[matches[:, 0]] - target_peaks[matches[:, 1]], axis=1)
    matches = matches[dist < np.quantile(dist, 0.95), :]
    if verbose:
        click.echo(
            f'Total of matches after distance filtering at time point {t_idx}: {len(matches)}'
        )
    if match_filter_angle_threshold:

        # Calculate vectors between matches
        vectors = target_peaks[matches[:, 1]] - source_peaks[matches[:, 0]]

        # Compute angles in radians relative to the x-axis
        angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Convert to degrees for easier interpretation
        angles_deg = np.degrees(angles_rad)

        # Create a histogram of angles
        bins = np.linspace(-180, 180, 36)  # 10-degree bins
        hist, bin_edges = np.histogram(angles_deg, bins=bins)

        # Find the dominant bin
        dominant_bin_index = np.argmax(hist)
        dominant_angle = (
            bin_edges[dominant_bin_index] + bin_edges[dominant_bin_index + 1]
        ) / 2

        # Filter matches within ±filter_angle_threshold degrees of the dominant direction, which may need finetuning
        filtered_indices = np.where(
            np.abs(angles_deg - dominant_angle) <= match_filter_angle_threshold
        )[0]
        matches = matches[filtered_indices]

        if verbose:
            click.echo(
                f'Total of matches after angle filtering at time point {t_idx}: {len(matches)}'
            )

    if len(matches) < 3:
        click.echo(
            f'Source and target beads were not matches successfully for timepoint {t_idx}'
        )
        return

    # Affine transform performs better than Euclidean
    if transform_type == 'affine':
        tform = AffineTransform(dimensionality=3)
    elif transform_type == 'euclidean':
        tform = EuclideanTransform(dimensionality=3)
    elif transform_type == 'similarity':
        tform = SimilarityTransform(dimensionality=3)
    else:
        raise ValueError(f'Unknown transform type: {transform_type}')
    
  
    tform.estimate(source_peaks[matches[:, 0]], target_peaks[matches[:, 1]])
    compount_tform = np.asarray(approx_tform) @ tform.inverse.params
    click.echo(f'Matches: {matches}')
    click.echo(f"tform.params: {tform.params}")
    click.echo(f" tform.inverse.params: { tform.inverse.params}")
    click.echo(f"compount_tform: {compount_tform}")

    if slurm:
        output_folder_path.mkdir(parents=True, exist_ok=True)
        np.save(output_folder_path/ f"{t_idx}.npy", compount_tform)

    return compount_tform.tolist()

def _get_z_shift_at_t(t_idx, phase_tzyx, fluoresc_tzyx, voxel_size, output_folder_path, ):
    from waveorder.focus import focus_from_transverse_band

    lf_img = np.asarray(phase_tzyx[t_idx]).astype(np.float32)
    ls_img = np.asarray(fluoresc_tzyx[t_idx]).astype(np.float32)

    (Z, Y, X) = lf_img.shape
    click.echo(f"Timepoint {t_idx}: LF shape: {lf_img.shape}, LS shape: {ls_img.shape}")
    # Focus detection
    lf_focus_idx = 22
    ls_focus_idx = 25
    click.echo(f"Phase focus index: {lf_focus_idx}")
    click.echo(f"Fluorescence focus index: {ls_focus_idx}")

    z_shift = (lf_focus_idx - ls_focus_idx)
    
    affine = np.eye(4)
    affine[0, 3] = z_shift  # phase is Z-first

    # Save output
    output_folder_path.mkdir(parents=True, exist_ok=True)
    np.save(output_folder_path / f"{t_idx}.npy", affine)

    return affine.tolist()

def z_shift_based_registration(
    phase_tzyx,
    fluoresc_tzyx,
    voxel_size,
    cluster,
    output_folder_path: Path,
    interpolation_window_size=3,
    interpolation_type='linear',
    verbose=False,
    sbatch_filepath=None,
):
    """
    Estimate per-timepoint Z shifts between Phase and Fluorescence volumes.
    Supports local or SLURM+Submitit parallelization.
    """

    (T, Z, Y, X) = phase_tzyx.shape


    # Compute transformations in parallel

    num_cpus, gb_ram_per_cpu = estimate_resources(shape=(T,2,Z,Y,X), ram_multiplier=5, max_num_cpus=16)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_focus_z",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 5,
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

    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")
    output_transforms_path = output_folder_path / "z_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    jobs = []
    with executor.batch():
        for t in range(T):
            job = executor.submit(
                _get_z_shift_at_t,
                t_idx=t,
                phase_tzyx=phase_tzyx,
                fluoresc_tzyx=fluoresc_tzyx,
                voxel_size=voxel_size,
                output_folder_path=output_transforms_path,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    # Wait for jobs
    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)

    # Load transforms
    transforms = []
    for t in range(T):
        file_path = output_transforms_path / f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
        else:
            transforms.append(np.load(file_path).tolist())

    # Interpolate missing
    transforms = _interpolate_transforms(
        transforms=transforms,
        window_size=interpolation_window_size,
        interpolation_type=interpolation_type,
        verbose=verbose,
    )

    return transforms

@click.command()
@source_position_dirpaths()
@target_position_dirpaths()
@output_filepath()
@num_processes()
@config_filepath()
@sbatch_filepath()
@local()
@click.option(
    "--registration-target-channel",
    "-rt",
    type=str,
    help="Name of the target channel to be used when registration params are applied.",
    required=False,
)
@click.option(
    "--registration-source-channel",
    "-rs",
    type=str,
    multiple=True,
    help="Name of the source channels to be used when registration params are applied. May be passed multiple times.",
    required=False,
)
def estimate_registration(
    source_position_dirpaths,
    target_position_dirpaths,
    output_filepath,
    num_processes,
    config_filepath,
    registration_target_channel,
    registration_source_channel,
    sbatch_filepath: str = None,
    local: bool = False,
):

    """
    Estimate the affine transformation between a source and target image for registration.

    This command-line tool uses either bead-based or user-assisted methods to estimate registration
    parameters for aligning source (moving) and target (fixed) images. The output is a configuration
    file that can be used with subsequent tools (`stabilize` and `register`).

    Parameters:
    - source_position_dirpaths (list): List of file paths to the source channel data in OME-Zarr format.
    - target_position_dirpaths (list): List of file paths to the target channel data in OME-Zarr format.
    - output_filepath (str): Path to save the estimated registration configuration file (YAML).
    - num_processes (int): Number of processes for parallel computations (used in bead-based registration).
    - config_filepath (str): Path to the YAML configuration file for the registration settings.
    - registration_target_channel (str): Name of the target channel to be used when registration params are applied.
    - registration_source_channels (list): List of source channel names to be used when registration params are applied.

    Returns:
    - None: Writes the estimated registration parameters to the specified output file.

    Notes:
    - Bead-based registration uses detected bead matches across timepoints to compute affine transformations.
    - User-assisted registration requires manual selection of corresponding features in source and target images.
    - If registration_target_channel and registration_source_channels are not provided, the target and source channels
    used for parameter estimation will be used.
    - The output configuration is essential for downstream processing in multi-modal image registration workflows.

    Example:
    >> biahub estimate-registration
        -s ./acq_name_labelfree_reconstructed.zarr/0/0/0   # Source channel OME-Zarr data path
        -t ./acq_name_lightsheet_deskewed.zarr/0/0/0       # Target channel OME-Zarr data path
        -o ./output.yml                                    # Output configuration file path
        --config ./config.yml                              # Path to input configuration file
        --num-processes 4                                  # Number of processes for parallel bead detection
        --registration-target-channel "Phase3D"            # Name of the target channel
        --registration-source-channel "GFP"                # Names of source channel
        --registration-source-channel "mCherry"            # Names of another source channel
    """

    settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    click.echo(f"Settings: {settings}")
    target_channel_name = settings.target_channel_name
    source_channel_name = settings.source_channel_name
    affine_90degree_rotation = settings.affine_90degree_rotation
    affine_transform_type = settings.affine_transform_type
    registration_source_channels = registration_source_channel  # rename for clarity
    output_dir = output_filepath.parent
    output_dir.mkdir(parents=True, exist_ok=True)

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
        source_channel_data = source_channel_position.data.dask_array()[
            :, source_channel_index
        ]
        source_channel_voxel_size = source_channel_position.scale[-3:]
    with open_ome_zarr(target_position_dirpaths[0], mode="r") as target_channel_position:
        target_channels = target_channel_position.channel_names
        target_channel_index = target_channels.index(target_channel_name)
        target_channel_name = target_channels[target_channel_index]
        target_channel_data = target_channel_position.data.dask_array()[
            :, target_channel_index
        ]
        voxel_size = target_channel_position.scale
        target_channel_voxel_size = voxel_size[-3:]

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    if settings.estimation_method == "beads":
        # Register using bead images
        transforms = beads_based_registration(
            source_channel_tzyx=source_channel_data,
            target_channel_tzyx=target_channel_data,
            approx_tform=np.asarray(settings.approx_affine_transform),
            match_algorithm=settings.match_algorithm,
            match_filter_angle_threshold=settings.match_filter_angle_threshold,
            match_max_ratio=settings.match_max_ratio,
            hungarian_knn_k=settings.hungarian_knn_k,
            transform_type=affine_transform_type,
            validation_window_size=settings.affine_transform_validation_window_size,
            validation_tolerance=settings.affine_transform_validation_tolerance,
            interpolation_window_size=settings.affine_transform_interpolation_window_size,
            interpolation_type=settings.affine_transform_interpolation_type,
            verbose=settings.verbose,
            cluster=cluster,
            sbatch_filepath=sbatch_filepath,
            output_folder_path=output_dir,
        )

        model = StabilizationSettings(
            stabilization_estimation_channel='',
            stabilization_type='xyz',
            stabilization_channels=registration_source_channels,
            affine_transform_zyx_list=transforms,
            time_indices='all',
            output_voxel_size=voxel_size,
        )
    elif settings.estimation_method == "z_shift":
        # Register using Z-focus shift only
        transforms = z_shift_based_registration(
            phase_tzyx=source_channel_data,
            fluoresc_tzyx=target_channel_data,
            voxel_size=voxel_size,
            cluster=cluster,
            output_folder_path=output_dir,
            interpolation_window_size=settings.affine_transform_interpolation_window_size,
            interpolation_type=settings.affine_transform_interpolation_type,
            verbose=settings.verbose,
            sbatch_filepath=sbatch_filepath,
        )
        model = StabilizationSettings(
            stabilization_estimation_channel='',
            stabilization_type='z',  # NOTE: different from 'xyz'
            stabilization_channels=registration_source_channels,
            affine_transform_zyx_list=transforms,
            time_indices='all',
            output_voxel_size=voxel_size,
        )

    else:
        # Register based on user input
        transform = user_assisted_registration(
            source_channel_volume=np.asarray(source_channel_data[settings.time_index]),
            source_channel_name=source_channel_name,
            source_channel_voxel_size=source_channel_voxel_size,
            target_channel_volume=np.asarray(target_channel_data[settings.time_index]),
            target_channel_name=target_channel_name,
            target_channel_voxel_size=target_channel_voxel_size,
            similarity=True if affine_transform_type == "similarity" else False,
            pre_affine_90degree_rotation=affine_90degree_rotation,
        )

        model = RegistrationSettings(
            source_channel_names=registration_source_channels,
            target_channel_name=registration_target_channel,
            affine_transform_zyx=transform.tolist(),
        )

    click.echo(f"Writing registration parameters to {output_filepath}")
    model_to_yaml(model, output_filepath)

   
    
    #plot z component of the transforms over time
    if settings.estimation_method == "beads" and settings.verbose:
        click.echo("Plotting translations over time")
        import matplotlib.pyplot as plt

        transforms = np.array(transforms)

        z_transforms = transforms[:, 0, 3] #->ZYX
        y_transforms = transforms[:, 1, 3] #->ZYX
        x_transforms = transforms[:, 2, 3] #->ZYX

        plt.plot(z_transforms)
        plt.legend(["Z-Translation"])
        plt.xlabel("Timepoint")
        plt.ylabel("Translations")
        plt.title("Translations Over Time")
        plt.grid()
        # Save the figure
        plt.savefig(output_dir/"Z_translation_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.plot(y_transforms)
        plt.legend(["Y-Translation"])
        plt.xlabel("Timepoint")
        plt.ylabel("Translations")
        plt.title("Translations Over Time")
        plt.grid()
        # Save the figure
        plt.savefig(output_dir/"Y_translation_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.plot(x_transforms) 
        plt.legend(["X-Translation"])
        plt.xlabel("Timepoint")
        plt.ylabel("Translations")
        plt.title("Translations Over Time")
        plt.grid()
        # Save the figure
        plt.savefig(output_dir/"X_translation_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        

if __name__ == "__main__":
    estimate_registration()
