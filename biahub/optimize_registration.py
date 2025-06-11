import ants
import click
import napari
import numpy as np

from iohub import open_ome_zarr
from skimage import filters

from biahub.cli.parsing import (
    config_filepath,
    output_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.utils import _check_nan_n_zeros, model_to_yaml, yaml_to_model
from biahub.register import (
    convert_transform_to_ants,
    convert_transform_to_numpy,
    find_overlapping_volume,
)
from biahub.settings import RegistrationSettings

# TODO: maybe a CLI call?
T_IDX = 0


def _optimize_registration(
    source_czyx: np.ndarray,
    target_czyx: np.ndarray,
    initial_tform: np.ndarray,
    source_channel_index: int | list = 0,
    target_channel_index: int = 0,
    z_slice: slice = slice(None),
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
    clip: bool = False,
    sobel_fitler: bool = False,
    verbose: bool = False,
) -> np.ndarray | None:
    if _check_nan_n_zeros(source_czyx) or _check_nan_n_zeros(target_czyx):
        return None

    t_form_ants = convert_transform_to_ants(initial_tform)

    # TODO: hardcoded values
    _target_channel = np.asarray(target_czyx[target_channel_index]).astype(np.float32)
    target_ants_pre_crop = ants.from_numpy(_target_channel)
    target_zyx = _target_channel[z_slice, y_slice, x_slice]
    if clip:
        target_zyx = np.clip(target_zyx, 0, 0.5)
    if sobel_fitler:
        target_zyx = filters.sobel(target_zyx)
    target_ants = ants.from_numpy(target_zyx)

    if not isinstance(source_channel_index, list):
        source_channel_index = [source_channel_index]
    source_channels = []
    for idx in source_channel_index:
        # Cropping, clipping, and filtering are applied after registration with initial_tform
        _source_channel = np.asarray(source_czyx[idx]).astype(np.float32)
        source_pre_optim = t_form_ants.apply_to_image(
            ants.from_numpy(_source_channel), reference=target_ants_pre_crop
        )
        source_channel = source_pre_optim.numpy()[z_slice, y_slice, x_slice]
        if clip:
            source_channel = np.clip(source_channel, 110, np.quantile(source_channel, 0.99))
        if sobel_fitler:
            source_channel = filters.sobel(source_channel)
        source_channels.append(source_channel)
    source_zyx = np.sum(source_channels, axis=0)
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
    composed_matrix = initial_tform @ tx_opt_numpy

    return composed_matrix


@click.command("optimize-registration")
@source_position_dirpaths()
@target_position_dirpaths()
@config_filepath()
@output_filepath()
@click.option(
    "--display-viewer",
    "-d",
    is_flag=True,
    help="Display the registered channels in a napari viewer",
)
@click.option(
    "--optimizer-verbose",
    "-v",
    is_flag=True,
    help="Show verbose output of optimizer",
)
def optimize_registration_cli(
    source_position_dirpaths,
    target_position_dirpaths,
    config_filepath,
    output_filepath,
    display_viewer,
    optimizer_verbose,
):
    """
    Optimize the affine transform between source and target channels using ANTs library.

    Start by generating an initial affine transform with `estimate-registration`.

    >> biahub optimize-registration -s ./acq_name_virtual_staining_reconstructed.zarr/0/0/0 -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 -c ./transform.yml -o ./optimized_transform.yml -d -v
    """

    settings = yaml_to_model(config_filepath, RegistrationSettings)

    # Load the source volume
    with open_ome_zarr(source_position_dirpaths[0]) as source_position:
        source_channel_names = source_position.channel_names
        # NOTE: using the first channel in the config to register
        source_channel_index = source_channel_names.index(settings.source_channel_names[0])
        source_channel_name = source_channel_names[source_channel_index]
        source_data_zyx = source_position.data[T_IDX, source_channel_index]

    # Load the target volume
    with open_ome_zarr(target_position_dirpaths[0]) as target_position:
        target_channel_names = target_position.channel_names
        target_channel_index = target_channel_names.index(settings.target_channel_name)
        target_channel_name = target_channel_names[target_channel_index]
        target_data_zyx = target_position.data[T_IDX, target_channel_index]

    click.echo(
        f"\nOptimizing registration using source channel {source_channel_name} and target channel {target_channel_name}"
    )

    approx_tform = np.array(settings.affine_transform_zyx)
    # Crop only to the overlapping regio, zero padding interfereces with registration
    z_slice, y_slice, x_slice = find_overlapping_volume(
        target_data_zyx.shape, source_data_zyx.shape, approx_tform
    )

    composed_matrix = _optimize_registration(
        source_data_zyx,
        target_data_zyx,
        approx_tform,
        z_slice=z_slice,
        y_slice=y_slice,
        x_slice=x_slice,
        verbose=optimizer_verbose,
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
