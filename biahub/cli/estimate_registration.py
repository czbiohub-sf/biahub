from functools import partial
from multiprocessing import Pool

import ants
import click
import dask.array as da
import napari
import numpy as np

from iohub import open_ome_zarr
from scipy.interpolate import interp1d
from skimage.feature import match_descriptors
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform
from waveorder.focus import focus_from_transverse_band

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
)
from biahub.cli.utils import _check_nan_n_zeros, model_to_yaml, yaml_to_model

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
    num_processes: int,
    window_size: int,
    tolerance: float,
    angle_threshold: int,
    verbose: bool = False,
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
    with Pool(num_processes) as pool:
        transforms = pool.map(
            partial(
                _get_tform_from_beads,
                approx_tform,
                source_channel_tzyx,
                target_channel_tzyx,
                angle_threshold,
                verbose,
            ),
            range(T),
        )

    # Check and filter transforms
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

    # Interpolate missing transforms
    x, y = zip(*[(i, transforms[i]) for i in range(T) if transforms[i] is not None])
    if len(transforms) - len(x) > 0:
        _x = [i for i in range(T) if i not in x]
        click.echo(f"Interpolating missing transforms at timepoints: {_x}")
        f = interp1d(x, y, axis=0, kind="linear", fill_value="extrapolate")
        transforms = f(range(T)).tolist()

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
        click.echo(f'Mean Squared Error of transformed points: {mse}')
    return mse <= threshold


def _get_tform_from_beads(
    approx_tform: list,
    source_channel_tzyx: da.Array,
    target_channel_tzyx: da.Array,
    angle_threshold: int,
    verbose: bool,
    t_idx: int,
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
        block_size=[32, 16, 16],
        threshold_abs=110,
        nms_distance=16,
        min_distance=0,
        verbose=verbose,
    )
    if verbose:
        click.echo('Detecting beads in target dataset:')
    target_peaks = detect_peaks(
        target_channel_zyx,
        block_size=[32, 16, 16],
        threshold_abs=0.8,
        nms_distance=16,
        min_distance=0,
        verbose=verbose,
    )

    # Skip if there is no peak detected
    if len(source_peaks) < 2 or len(target_peaks) < 2:
        click.echo(f'No beads were detected at timepoint {t_idx}')
        return

    # Match peaks, excluding top 5% of distances as outliers
    matches = match_descriptors(
        source_peaks, target_peaks, metric='euclidean', max_ratio=0.6, cross_check=True
    )
    if verbose:
        click.echo(f'Total of matches at time point {t_idx}: {len(matches)}')
    dist = np.linalg.norm(source_peaks[matches[:, 0]] - target_peaks[matches[:, 1]], axis=1)
    matches = matches[dist < np.quantile(dist, 0.95), :]
    if verbose:
        click.echo(
            f'Total of matches after distance filtering at time point {t_idx}: {len(matches)}'
        )

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
    dominant_angle = (bin_edges[dominant_bin_index] + bin_edges[dominant_bin_index + 1]) / 2

    # Filter matches within ±30 degrees of the dominant direction, which may need finetuning

    filtered_indices = np.where(np.abs(angles_deg - dominant_angle) <= angle_threshold)[0]
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
    tform = AffineTransform(dimensionality=3)
    tform.estimate(source_peaks[matches[:, 0]], target_peaks[matches[:, 1]])
    compount_tform = approx_tform @ tform.inverse.params

    return compount_tform.tolist()


@click.command()
@source_position_dirpaths()
@target_position_dirpaths()
@output_filepath()
@num_processes()
@config_filepath()
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

    target_channel_name = settings.target_channel_name
    source_channel_name = settings.source_channel_name
    affine_90degree_rotation = settings.affine_90degree_rotation
    affine_transform_type = settings.affine_transform_type
    registration_source_channels = registration_source_channel  # rename for clarity
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

    if settings.estimation_method == "beads":
        # Register using bead images
        transforms = beads_based_registration(
            source_channel_data,
            target_channel_data,
            approx_tform=np.asarray(settings.approx_affine_transform),
            num_processes=num_processes,
            window_size=settings.affine_transform_window_size,
            tolerance=settings.affine_transform_tolerance,
            angle_threshold=settings.filtering_angle_threshold,
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
    else:
        # Register based on user input
        transform = user_assisted_registration(
            source_channel_volume=np.asarray(source_channel_data[settings.time_index]),
            source_channel_name=source_channel_name,
            source_channel_voxel_size=source_channel_voxel_size,
            target_channel_volume=np.asarray(target_channel_data[settings.time_index]),
            target_channel_name=target_channel_name,
            target_channel_voxel_size=target_channel_voxel_size,
            similarity=True if affine_transform_type == "Similarity" else False,
            pre_affine_90degree_rotation=affine_90degree_rotation,
        )

        model = RegistrationSettings(
            source_channel_names=registration_source_channels,
            target_channel_name=registration_target_channel,
            affine_transform_zyx=transform.tolist(),
        )

    click.echo(f"Writing registration parameters to {output_filepath}")
    model_to_yaml(model, output_filepath)


if __name__ == "__main__":
    estimate_registration()
