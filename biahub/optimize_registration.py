from pathlib import Path

import ants
import click
import napari
import numpy as np

from iohub import open_ome_zarr
from skimage import filters

from biahub.cli.parsing import (
    config_filepaths,
    output_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.utils import _check_nan_n_zeros, model_to_yaml, yaml_to_model
from biahub.register import (
    convert_transform_to_ants,
    convert_transform_to_numpy,
    find_lir,
)
from biahub.settings import RegistrationSettings


def _optimize_registration(
    source_czyx: np.ndarray,
    target_czyx: np.ndarray,
    initial_tform: np.ndarray,
    source_channel_index: int | list = 0,
    target_channel_index: int = 0,
    crop: bool = False,
    clip: bool = False,
    sobel_fitler: bool = False,
    verbose: bool = False,
    slurm: bool = False,
    t_idx: int = 0,
    output_folder_path: str | None = None,
) -> np.ndarray | None:

    source_czyx = np.asarray(source_czyx).astype(np.float32)
    target_czyx = np.asarray(target_czyx).astype(np.float32)
    if _check_nan_n_zeros(source_czyx) or _check_nan_n_zeros(target_czyx):
        return None

    t_form_ants = convert_transform_to_ants(initial_tform)

    target_zyx = target_czyx[target_channel_index]
    if target_zyx.ndim != 3:
        raise ValueError(f"Expected 3D target channel, got shape {target_zyx.shape}")
    target_ants_pre_crop = ants.from_numpy(target_zyx)

    if not isinstance(source_channel_index, list):
        source_channel_index = [source_channel_index]
    source_channels = []
    for idx in source_channel_index:
        # Cropping, clipping, and filtering are applied after registration with initial_tform
        _source_channel = np.asarray(source_czyx[idx]).astype(np.float32)
        if _source_channel.ndim != 3:
            raise ValueError(f"Expected 3D source channel, got shape {_source_channel.shape}")
        source_channel = t_form_ants.apply_to_image(
            ants.from_numpy(_source_channel), reference=target_ants_pre_crop
        ).numpy()
        if source_channel.ndim != 3:
            raise ValueError(
                f"apply_to_image returned non-3D array: {source_channel.shape}.\n"
                "This is likely caused by mismatched input shape or invalid transform/reference."
            )
        source_channels.append(source_channel)

    _offset = np.zeros(3, dtype=np.float32)
    if crop:
        mask = np.astype((target_zyx != 0) & (source_channels[0] != 0), np.uint8)
        z_slice, y_slice, x_slice = find_lir(mask, plot=False)
        _offset = np.asarray(
            [_s.start for _s in (z_slice, y_slice, x_slice)], dtype=np.float32
        )
        target_zyx = target_zyx[z_slice, y_slice, x_slice]
        source_channels = [_channel[z_slice, y_slice, x_slice] for _channel in source_channels]

    # TODO: hardcoded clipping limits
    if clip:
        target_zyx = np.clip(target_zyx, 0, 0.5)
        source_channels = [
            np.clip(_channel, 110, np.quantile(_channel, 0.99)) for _channel in source_channels
        ]

    if sobel_fitler:
        target_zyx = filters.sobel(target_zyx)
        source_channels = [filters.sobel(_channel) for _channel in source_channels]

    source_zyx = np.sum(source_channels, axis=0)
    target_ants = ants.from_numpy(target_zyx)
    source_ants = ants.from_numpy(source_zyx)

    reg = ants.registration(
        fixed=target_ants,
        moving=source_ants,
        type_of_transform="Similarity",
        aff_shrink_factors=(6, 3, 1),
        aff_iterations=(2100, 1200, 50),
        aff_smoothing_sigmas=(2, 1, 0),
        verbose=verbose,
    )

    tx_opt_mat = ants.read_transform(reg["fwdtransforms"][0])
    tx_opt_numpy = convert_transform_to_numpy(tx_opt_mat)

    # Account for tx_opt being estimated at a crop rather than starting at the origin,
    # i.e. (0, 0, 0) of the image.
    shift_to_roi_np = np.eye(4)
    shift_to_roi_np[:3, -1] = _offset
    shift_back_np = np.eye(4)
    shift_back_np[:3, -1] = -_offset
    composed_matrix = initial_tform @ shift_to_roi_np @ tx_opt_numpy @ shift_back_np

    if slurm:
        output_folder_path.mkdir(parents=True, exist_ok=True)
        np.save(output_folder_path / f"{t_idx}.npy", composed_matrix)

    return composed_matrix


@click.command("optimize-registration")
@source_position_dirpaths()
@target_position_dirpaths()
@config_filepaths()
@output_filepath()
@click.option(
    "--display-viewer",
    "-d",
    is_flag=True,
    help="Display the registered channels in a napari viewer",
)
def optimize_registration_cli(
    source_position_dirpaths,
    target_position_dirpaths,
    config_filepaths,
    output_filepath,
    display_viewer,
):
    """
    Optimize the affine transform between source and target channels using ANTs library.

    Start by generating an initial affine transform with `estimate-registration`.

    >> biahub optimize-registration -s ./acq_name_virtual_staining_reconstructed.zarr/0/0/0 -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 -c ./transform.yml -o ./optimized_transform.yml -d -v
    """

    if len(config_filepaths) == 1:
        config_filepath = Path(config_filepaths[0])
    else:
        raise ValueError(
            "Only one configuration file is supported for optimize_registration. Please provide a single configuration file."
        )

    settings = yaml_to_model(config_filepath, RegistrationSettings)
    t_idx = settings.time_indices
    # if time_indices not int type
    if not isinstance(t_idx, int):
        print(
            "Time index 'all' is not supported for optimize-registration, using first time index"
        )
        t_idx = 0

    # Load the source volume
    with open_ome_zarr(source_position_dirpaths[0]) as source_position:
        # NOTE: using the first channel in the config to register
        source_channel_names = source_position.channel_names
        source_channel_index = source_channel_names.index(settings.source_channel_names[0])
        source_data_czyx = np.asarray(source_position.data[t_idx])
        print("Source data shape:", source_data_czyx.shape)

    # Load the target volume
    with open_ome_zarr(target_position_dirpaths[0]) as target_position:
        target_channel_names = target_position.channel_names
        target_channel_index = target_channel_names.index(settings.target_channel_name)
        target_data_czyx = np.asarray(target_position.data[t_idx])
        print("Target data shape:", target_data_czyx.shape)
    click.echo(
        f"\nOptimizing registration using source channel {source_channel_names[source_channel_index]} and target channel {target_channel_names[target_channel_index]}"
    )

    approx_tform = np.asarray(settings.affine_transform_zyx, dtype=np.float32)

    source_data_zyx = source_data_czyx[source_channel_index]
    target_data_zyx = target_data_czyx[target_channel_index]

    composed_matrix = _optimize_registration(
        source_czyx=source_data_czyx,
        target_czyx=target_data_czyx,
        initial_tform=approx_tform,
        source_channel_index=source_channel_index,
        target_channel_index=target_channel_index,
        crop=True,
        verbose=settings.verbose,
    )

    # Saving the parameters
    click.echo(f"Writing registration parameters to {output_filepath}")
    # copy config settings and modify only ones that change
    output_settings = settings.model_copy()
    output_settings.affine_transform_zyx = composed_matrix.tolist()
    model_to_yaml(output_settings, output_filepath)

    if display_viewer:
        click.echo("Initializing napari viewer...")
        approx_tform_ants = convert_transform_to_ants(approx_tform)
        composed_matrix_ants = convert_transform_to_ants(composed_matrix)
        source_zyx_ants = ants.from_numpy(source_data_zyx.astype(np.float32))
        target_zyx_ants = ants.from_numpy(target_data_zyx.astype(np.float32))
        source_pre_optim = approx_tform_ants.apply_to_image(
            source_zyx_ants, reference=target_zyx_ants
        )
        source_post_optim = composed_matrix_ants.apply_to_image(
            source_zyx_ants, reference=target_zyx_ants
        )

        viewer = napari.Viewer()
        source_pre_opt_layer = viewer.add_image(
            source_pre_optim.numpy(),
            name="source_pre_optimization",
            colormap="cyan",
            opacity=0.5,
        )
        source_pre_opt_layer.visible = False

        viewer.add_image(
            source_post_optim.numpy(),
            name="source_post_optimization",
            colormap="cyan",
            blending="additive",
        )
        viewer.add_image(
            target_data_zyx,
            name="target",
            colormap="magenta",
            blending="additive",
        )

        input("\n Displaying registered channels. Press <enter> to close...")


if __name__ == "__main__":
    optimize_registration_cli()
