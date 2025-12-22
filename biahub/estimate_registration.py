from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import (
    config_filepath,
    local,
    output_filepath,
    sbatch_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.utils import (
    model_to_yaml,
    yaml_to_model,
)
from biahub.registration.manual import user_assisted_registration
from biahub.registration.utils import evaluate_transforms, plot_translations
from biahub.settings import (
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)


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
        from biahub.registration.methods.bead_matching import estimate_tczyx

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
