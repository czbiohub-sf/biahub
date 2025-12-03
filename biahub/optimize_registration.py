import ants
import click
import napari
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import (
    config_filepath,
    output_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.utils import model_to_yaml, yaml_to_model
from biahub.registration.utils import convert_transform_to_ants
from biahub.settings import RegistrationSettings
from biahub.registration.utils import convert_transform_to_ants

from biahub.registration.ants import _optimize_registration

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
def optimize_registration_cli(
    source_position_dirpaths,
    target_position_dirpaths,
    config_filepath,
    output_filepath,
    display_viewer,
):
    """
    Optimize the affine transform between source and target channels using ANTs library.

    Start by generating an initial affine transform with `estimate-registration`.

    >> biahub optimize-registration -s ./acq_name_virtual_staining_reconstructed.zarr/0/0/0 -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 -c ./transform.yml -o ./optimized_transform.yml -d -v
    """
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
