from pathlib import Path

import ants
import click
import numpy as np

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from skimage.transform import EuclideanTransform, SimilarityTransform
from waveorder.focus import focus_from_transverse_band

from biahub.cli.parsing import (
    config_filepath,
    local,
    output_filepath,
    sbatch_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.utils import (
    get_submitit_cluster,
    model_to_yaml,
    yaml_to_model,
)
from biahub.registration.utils import (
    convert_transform_to_ants,
    convert_transform_to_numpy,
    evaluate_transforms,
    get_3D_fliplr_matrix,
    get_3D_rescaling_matrix,
    get_3D_rotation_matrix,
    plot_translations,
)
from biahub.settings import (
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)

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
    source_channel_volume: ArrayLike,
    source_channel_name: str,
    source_channel_voxel_size: tuple[float, float, float],
    target_channel_volume: ArrayLike,
    target_channel_name: str,
    target_channel_voxel_size: tuple[float, float, float],
    similarity: bool = False,
    pre_affine_90degree_rotation: int = 0,
    pre_affine_fliplr: bool = False,
) -> list[ArrayLike]:
    """
    Perform user-assisted registration of two volumetric image channels.

    This function allows users to manually annotate corresponding features between
    a source and target channel to calculate an affine transformation matrix for registration.

    Parameters
    ----------
    source_channel_volume : ArrayLike
       3D array (Z, Y, X) of the source channel.
    source_channel_name : str
        Name of the source channel for display purposes.
    source_channel_voxel_size : tuple[float, float, float]
        Voxel size of the source channel (Z, Y, X).
    target_channel_volume : ArrayLike
       3D array (Z, Y, X) of the target channel.
    target_channel_name : str
        Name of the target channel for display purposes.
    target_channel_voxel_size : tuple[float, float, float]
        Voxel size of the target channel (Z, Y, X).
    similarity : bool
        If True, use a similarity transform (rotation, translation, scaling);
                         if False, use an Euclidean transform (rotation, translation).
    pre_affine_90degree_rotation : int
        Number of 90-degree rotations to apply to the source channel before registration.
    pre_affine_fliplr : bool
        If True, apply left-right flip to the source channel before registration.
        Note: Flip is applied first, then rotation.

    Returns
    -------
    ArrayLike
        The estimated 4x4 affine transformation matrix for registering the source channel to the target channel.

    Notes
    -----
    - The function uses a napari viewer for manual feature annotation.
    - Scaling factors for voxel size differences between source and target are calculated and applied.
    - Users must annotate at least three corresponding points in both channels for the transform calculation.
    - Two types of transformations are supported: similarity (with scaling) and Euclidean (no scaling).
    - The function visually displays intermediate and final results in napari for user validation.
    """
    # Find the in-focus slice
    source_channel_Z, source_channel_Y, source_channel_X = source_channel_volume.shape[-3:]
    target_channel_Z, target_channel_Y, target_channel_X = target_channel_volume.shape[-3:]

    source_channel_focus_idx = focus_from_transverse_band(
        source_channel_volume[
            :,
            source_channel_Y // 2 - FOCUS_SLICE_ROI_WIDTH : source_channel_Y // 2
            + FOCUS_SLICE_ROI_WIDTH,
            source_channel_X // 2 - FOCUS_SLICE_ROI_WIDTH : source_channel_X // 2
            + FOCUS_SLICE_ROI_WIDTH,
        ],
        NA_det=NA_DETECTION_SOURCE,
        lambda_ill=WAVELENGTH_EMISSION_SOURCE_CHANNEL,
        pixel_size=source_channel_voxel_size[-1],
    )

    target_channel_focus_idx = focus_from_transverse_band(
        target_channel_volume[
            :,
            target_channel_Y // 2 - FOCUS_SLICE_ROI_WIDTH : target_channel_Y // 2
            + FOCUS_SLICE_ROI_WIDTH,
            target_channel_X // 2 - FOCUS_SLICE_ROI_WIDTH : target_channel_X // 2
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

    # Apply flip transformation if requested (flip happens first)
    if pre_affine_fliplr:
        fliplr_affine = get_3D_fliplr_matrix(
            (source_channel_Z, source_channel_Y, source_channel_X),
            (target_channel_Z, target_channel_Y, target_channel_X),
        )
    else:
        fliplr_affine = np.eye(4)

    compound_affine = scaling_affine @ rotate90_affine @ fliplr_affine
    tx_manual = convert_transform_to_ants(compound_affine).invert()

    source_zxy_pre_reg = tx_manual.apply_to_image(source_zyx_ants, reference=target_zyx_ants)

    import napari

    viewer = napari.Viewer()

    viewer.add_image(target_channel_volume, name=f"target_{target_channel_name}")
    points_target_channel = viewer.add_points(
        ndim=3, name=f"pts_target_{target_channel_name}", size=20, face_color=COLOR_CYCLE[0]
    )

    source_layer = viewer.add_image(
        source_zxy_pre_reg.numpy(),
        name=f"source_{source_channel_name}",
        blending="additive",
        colormap="green",
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
        blending="additive",
    )
    # Cleanup
    viewer.layers.remove(points_source_channel)
    viewer.layers.remove(points_target_channel)
    source_layer.visible = False

    # Ants affine transforms
    tform = convert_transform_to_numpy(tx_manual)
    click.echo(f"Estimated affine transformation matrix:\n{tform}\n")
    input("Press <Enter> to close the viewer and exit...")
    viewer.close()

    return [tform.tolist()]


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
    num_threads : int
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
    cluster = get_submitit_cluster(local)
    eval_transform_settings = settings.eval_transform_settings

    if settings.estimation_method == "beads":
        from biahub.registration.beads import estimate_tczyx

        transforms = estimate_tczyx(
            mov_tczyx=source_data,
            ref_tczyx=target_data,
            mov_channel_index=source_channel_index,
            ref_channel_index=target_channel_index,
            beads_match_settings=settings.beads_match_settings,
            affine_transform_settings=settings.affine_transform_settings,
            verbose=settings.verbose,
            cluster=cluster,
            sbatch_filepath=sbatch_filepath,
            output_folder_path=output_dir,
            ref_voxel_size=target_channel_voxel_size,
            mov_voxel_size=source_channel_voxel_size,
        )

    elif settings.estimation_method == "ants":
        from biahub.registration.ants import estimate_tczyx

        transforms = estimate_tczyx(
            mov_tczyx=source_data,
            ref_tczyx=target_data,
            mov_channel_index=source_channel_index,
            ref_channel_index=target_channel_index,
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
            stabilization_estimation_channel=target_channel_name,
            stabilization_type="affine",
            stabilization_method=settings.estimation_method,
            stabilization_channels=[source_channel_name, target_channel_name],
            affine_transform_zyx_list=transforms,
            time_indices="all",
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
    """Estimate the affine transformation between a source and target image for registration.

    This command-line tools estimates the registration transforms between a source (moving) and target (fixed) image
    using either (1) user input, (2) images or registration beads, or (3) image features via the ANTS registration library.
    The output is a configuration file that can be used with subsequent tools (`stabilize` and `register`).

    User-assisted registration requires manual selection of corresponding features in source and target images.
    Bead-based registration uses detected bead matches across timepoints to compute affine transformations.
    ANTs-based registration uses the ANTsPy library to estimate transformations based on image features. Optionally,
    a Sobel filter may be applied to the data to enhance feature detection between label-free and fluorescent channels.

    >>> biahub estimate-registration \
        -s ./acq_name_labelfree_reconstructed.zarr/0/0/0 \
        -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 \
        -o ./output.yml \
        --config ./config.yml \
        --registration-target-channel "Phase3D" \
        --registration-source-channel "GFP" \
        --registration-source-channel "mCherry"
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
