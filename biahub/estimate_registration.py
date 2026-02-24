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
    estimate_resources,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.registration.manual import user_assisted_registration
from biahub.registration.utils import evaluate_transforms, plot_translations
from biahub.settings import (
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)
from biahub.cli.parsing import (
    sbatch_to_submitit,
)
from pathlib import Path
from typing import List

import click
import numpy as np
import submitit
from iohub.ngff import open_ome_zarr
from tqdm import tqdm

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import yaml_to_model
from biahub.registration.utils import (
    evaluate_transforms,
    save_transforms,
)
from biahub.settings import (
    EstimateStabilizationSettings,
    StabilizationSettings,
)

def estimate_transform(
    ref_position_dirpath: str,
    mov_position_dirpath: str,
    method: str,
    output_dirpath: str,
    sbatch_filepath: str = None,
    cluster: str = "local",
    verbose: bool = False,
) -> None:

    function_to_run = None
    match method:
        case "manual":
            # currently only for one timepoint, should be like that,
            #  or maybe allow the user to run it for different timepoints,
            #  interp between them if needed and show the total registration
            from biahub.registration.manual import user_assisted_registration
            function_to_run = user_assisted_registration
        case "ants":
            from biahub.registration.ants import estimate_tczyx
            #currentlyslurm over timepoints
            # should be able to run over positions too
            function_to_run = estimate_tczyx
        case "beads":
            #slurm over timepoints or sequentially (necessary for best performance).
            #  usually used only in on the beads fov, but better to allow more flexibility
            from biahub.registration.beads import estimate_independently
            function_to_run = estimate_independently
        case "pcc":
            #currently slurm over positions, but can be over time too, better to allow both
            from biahub.registration.phase_cross_correlation import estimate_tczyx

            function_to_run = estimate_tczyx
        case "focus-finding":
            #currently slurm over positions, but can be over time too, better to allow both
            from biahub.registration.z_focus_finding import estimate_tczyx
            function_to_run = estimate_tczyx
        case "stackreg":
            #currently slurm over positions, but can be over time too, better to allow both
            from biahub.registration.stackreg import estimate_tczyx
            function_to_run = estimate_tczyx
        case _:
            raise ValueError(f"Unknown method: {method}")

    return function_to_run

def estimate_registration_parallel_fovs_and_time(
    ref_position_dirpath: str,
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    mov_position_dirpath: str = None,
    cluster: str = "local",
) -> None:

    settings = yaml_to_model(config_filepath, RegistrationSettings)
    click.echo(f"Settings: {settings}")

    fov = settings.fov
    methods_kwargs = {}
    ref_position_dirpath = Path(ref_position_dirpath) / fov


    output_dirpath.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_dirpath / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        shape = dataset.data.shape
        T, C, Z, Y, X = shape

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, C, Z, Y, X), ram_multiplier=16, max_num_cpus=16
    )

    slurm_args = {
        "slurm_job_name": "estimate_registration",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 60,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM xyz PCC jobs with resources: {slurm_args}")
    transforms_out_path = output_dirpath / "transforms_per_position"
    transforms_out_path.mkdir(parents=True, exist_ok=True)
  
    jobs = []
    
    function_to_run = estimate_transform(method=settings.method)


    ## If you want to avoid the nested complexity entirely, you could flatten it: submit one job per (position,
   #t) pair from the CLI, then collect and save results per position after all jobs finish. More jobs but
  #simpler dependency graph.
    with submitit.helpers.clean_env(), executor.batch():
        for fov in ref_position_dirpath:
            for t in range(T):
                job = executor.submit(
                    function_to_run,
                    t, 
                    fov, 
                    **methods_kwargs
                )
                jobs.append(job)

    # wait_for_jobs_to_finish(jobs)


    # transform_files = list(transforms_out_path.glob("*.npy"))

    # fov_transforms = {}
    # for file_path in transform_files:
    #     fov_filename = file_path.stem
    #     fov_transforms[fov_filename] = np.load(file_path).tolist()


    # evaluate transforms
    # save transforms

    # plots
    # qc?

    pass

def estimate_registration_parallel_timepoints():
    pass


def estimate_stabilization(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = False,
) -> None:
    """
    Estimate the stabilization matrices for a list of positions.

    Parameters
    ----------
    input_position_dirpaths : list[str]
        Paths to the input position directories.
    output_filepath : str
        Path to the output file.
    config_filepath : str
        Path to the configuration file.
    sbatch_filepath : str
        Path to the sbatch file.
    local : bool
        If True, run locally.

    Returns
    -------
    None

    Notes
    -----
    The verbose output will be saved at the same level as the output zarr.
    """

    # Load the settings
    config_filepath = Path(config_filepath)

    settings = yaml_to_model(config_filepath, EstimateStabilizationSettings)
    click.echo(f"Settings: {settings}")

    verbose = settings.verbose
    stabilization_estimation_channel = settings.stabilization_estimation_channel
    stabilization_type = settings.stabilization_type
    stabilization_method = settings.stabilization_method

    output_dirpath = Path(output_dirpath)
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Channel names to process
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        channel_names = dataset.channel_names
        voxel_size = dataset.scale
        channel_index = channel_names.index(stabilization_estimation_channel)
        T, C, Z, Y, X = dataset.data.shape

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Load the evaluation settings
    eval_transform_settings = settings.eval_transform_settings

    if "xyz" == stabilization_type:
        if stabilization_method == "focus-finding":
            click.echo(
                "Estimating xyz stabilization parameters with focus finding and stack registration"
            )
            from biahub.registration.z_focus_finding import estimate_z_stabilization

            z_transforms_dict = estimate_z_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                focus_finding_settings=settings.focus_finding_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )
            from biahub.registration.stackreg import estimate_xy_stabilization

            xy_transforms_dict = estimate_xy_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                stack_reg_settings=settings.stack_reg_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )

            try:

                for fov, xy_transforms in tqdm(
                    xy_transforms_dict.items(), desc="Processing FOVs"
                ):

                    z_transforms = np.asarray(z_transforms_dict[fov])
                    xy_transforms = np.asarray(xy_transforms)

                    if xy_transforms.shape[0] != z_transforms.shape[0]:
                        raise ValueError(
                            "The number of translation matrices and z drift matrices must be the same"
                        )

                    xyz_transforms = np.asarray(
                        [a @ b for a, b in zip(xy_transforms, z_transforms)]
                    ).tolist()

                    if eval_transform_settings:
                        xyz_transforms = evaluate_transforms(
                            transforms=xyz_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )
                        z_transforms = evaluate_transforms(
                            transforms=z_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )
                        xy_transforms = evaluate_transforms(
                            transforms=xy_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )

                    save_transforms(
                        model=model,
                        transforms=xyz_transforms,
                        output_filepath_settings=output_dirpath
                        / "xyz_stabilization_settings"
                        / f"{fov}.yml",
                        output_filepath_plot=output_dirpath
                        / "translation_plots"
                        / f"{fov}.png",
                        verbose=verbose,
                    )
                    save_transforms(
                        model=model,
                        transforms=z_transforms,
                        output_filepath_settings=output_dirpath
                        / "z_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                    )
                    save_transforms(
                        model=model,
                        transforms=xy_transforms,
                        output_filepath_settings=output_dirpath
                        / "xy_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                    )

            except Exception as e:
                click.echo(
                    f"Error estimating {stabilization_type} stabilization parameters: {e}"
                )

        elif stabilization_method == "beads":

            from biahub.registration.beads import estimate_tczyx

            click.echo("Estimating xyz stabilization parameters with beads")
            with open_ome_zarr(input_position_dirpaths[0], mode="r") as beads_position:
                source_channels = beads_position.channel_names
                source_channel_index = source_channels.index(stabilization_estimation_channel)
                channel_tczyx = beads_position.data.dask_array()

            xyz_transforms = estimate_tczyx(
                mov_tczyx=channel_tczyx,
                ref_tczyx=channel_tczyx,
                mov_channel_index=source_channel_index,
                ref_channel_index=source_channel_index,
                beads_match_settings=settings.beads_match_settings,
                affine_transform_settings=settings.affine_transform_settings,
                verbose=verbose,
                output_folder_path=output_dirpath,
                mode="stabilization",
                cluster=cluster,
                sbatch_filepath=sbatch_filepath,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )

            if eval_transform_settings:
                xyz_transforms = evaluate_transforms(
                    transforms=xyz_transforms,
                    shape_zyx=(Z, Y, X),
                    validation_window_size=eval_transform_settings.validation_window_size,
                    validation_tolerance=eval_transform_settings.validation_tolerance,
                    interpolation_window_size=eval_transform_settings.interpolation_window_size,
                    interpolation_type=eval_transform_settings.interpolation_type,
                    verbose=verbose,
                )

            save_transforms(
                model=model,
                transforms=xyz_transforms,
                output_filepath_settings=output_dirpath / "xyz_stabilization_settings.yml",
                verbose=verbose,
                output_filepath_plot=output_dirpath / "translation_plots" / "beads.png",
            )

        elif stabilization_method == "phase-cross-corr":
            click.echo("Estimating xyz stabilization parameters with phase cross correlation")

            from biahub.registration.phase_cross_correlation import (
                estimate_xyz_stabilization_pcc,
            )

            xyz_transforms_dict = estimate_xyz_stabilization_pcc(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                phase_cross_corr_settings=settings.phase_cross_corr_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )

            try:
                for fov, xyz_transforms in tqdm(
                    xyz_transforms_dict.items(), desc="Processing FOVs"
                ):
                    if eval_transform_settings:
                        xyz_transforms = evaluate_transforms(
                            transforms=xyz_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )

                    save_transforms(
                        model=model,
                        transforms=xyz_transforms,
                        output_filepath_settings=output_dirpath
                        / "xyz_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                        output_filepath_plot=output_dirpath
                        / "translation_plots"
                        / f"{fov}.png",
                    )
            except Exception as e:
                click.echo(
                    f"Error estimating {stabilization_type} stabilization parameters: {e}"
                )

    # Estimate z drift
    if "z" == stabilization_type and stabilization_method == "focus-finding":
        click.echo("Estimating z stabilization parameters with focus finding")

        from biahub.registration.z_focus_finding import estimate_z_stabilization

        z_transforms_dict = estimate_z_stabilization(
            input_position_dirpaths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            channel_index=channel_index,
            focus_finding_settings=settings.focus_finding_settings,
            sbatch_filepath=sbatch_filepath,
            cluster=cluster,
            verbose=verbose,
        )

        model = StabilizationSettings(
            stabilization_type=settings.stabilization_type,
            stabilization_method=settings.stabilization_method,
            stabilization_estimation_channel=settings.stabilization_estimation_channel,
            stabilization_channels=settings.stabilization_channels,
            affine_transform_zyx_list=[],
            time_indices="all",
            output_voxel_size=voxel_size,
        )

        try:
            for fov, z_transforms in tqdm(z_transforms_dict.items(), desc="Processing FOVs"):
                if eval_transform_settings:
                    z_transforms = evaluate_transforms(
                        transforms=z_transforms,
                        shape_zyx=(Z, Y, X),
                        validation_window_size=eval_transform_settings.validation_window_size,
                        validation_tolerance=eval_transform_settings.validation_tolerance,
                        interpolation_window_size=eval_transform_settings.interpolation_window_size,
                        interpolation_type=eval_transform_settings.interpolation_type,
                        verbose=verbose,
                    )

                save_transforms(
                    model=model,
                    transforms=z_transforms,
                    output_filepath_settings=output_dirpath
                    / "z_stabilization_settings"
                    / f"{fov}.yml",
                    verbose=verbose,
                    output_filepath_plot=output_dirpath / "translation_plots" / f"{fov}.png",
                )
        except Exception as e:
            click.echo(f"Error estimating {stabilization_type} stabilization parameters: {e}")

    # Estimate yx drift
    if "xy" == stabilization_type:
        if stabilization_method == "focus-finding":
            click.echo(
                "Estimating xy stabilization parameters with focus finding and stack registration"
            )
            from biahub.registration.stackreg import estimate_xy_stabilization

            xy_transforms_dict = estimate_xy_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                stack_reg_settings=settings.stack_reg_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )
            try:
                for fov, xy_transforms in tqdm(
                    xy_transforms_dict.items(), desc="Processing FOVs"
                ):
                    if eval_transform_settings:
                        xy_transforms = evaluate_transforms(
                            transforms=xy_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )

                    save_transforms(
                        model=model,
                        transforms=xy_transforms,
                        output_filepath_settings=output_dirpath
                        / "xy_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                        output_filepath_plot=output_dirpath
                        / "translation_plots"
                        / f"{fov}.png",
                    )
            except Exception as e:
                click.echo(
                    f"Error estimating {stabilization_type} stabilization parameters: {e}"
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

@click.command("estimate-stabilization")
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
def estimate_stabilization_cli(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepath: Path,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate translation matrices for XYZ stabilization of a timelapse dataset.

    Stabilization parameters may be computed for the XY, Z, or XYZ dimensions using
    focus finding, beads, or phase cross correlation methods.

    Example usage:
    biahub estimate-stabilization -i ./timelapse.zarr/0/0/0 -o ./stabilization.yml  -c ./config.yml -s ./sbatch.sh --local --verbose

    """
    estimate_stabilization(
        input_position_dirpaths=input_position_dirpaths,
        output_dirpath=output_dirpath,
        config_filepath=config_filepath,
        sbatch_filepath=sbatch_filepath,
        local=local,
    )

