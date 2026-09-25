import itertools
import shutil

from datetime import datetime
from pathlib import Path
from typing import Literal

import click
import numpy as np
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr
from numpy.typing import ArrayLike
from pystackreg import StackReg
from tqdm import tqdm
from waveorder.focus import focus_from_transverse_band

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.registration.utils import (
    evaluate_transforms,
    save_transforms,
)
from biahub.settings import (
    EstimateRegistrationSettings,
    EstimateStabilizationSettings,
    FocusFindingSettings,
    StabilizationSettings,
    StackRegSettings,
)
from biahub.utils.cluster import estimate_resources, get_submitit_cluster
from biahub.utils.config import yaml_to_model

NA_DET = 1.35
LAMBDA_ILL = 0.500


def remove_beads_fov_from_path_list(
    position_dirpaths: list[Path],
    skip_beads_fov: str,
) -> list[Path]:
    """
    Remove the beads FOV from the input data paths.

    Parameters
    ----------
    position_dirpaths : list[Path]
        Paths to the input position directories.
    skip_beads_fov : str
        Beads FOV to skip.

    Returns
    -------
    list[Path]
        Paths to the input position directories without the beads FOV.
    """
    if skip_beads_fov != "0":
        click.echo(f"Removing beads FOV {skip_beads_fov} from input data paths")
        position_dirpaths = [
            path for path in position_dirpaths if skip_beads_fov not in str(path)
        ]
    return position_dirpaths


def estimate_xy_stabilization_per_position(
    input_position_dirpath: Path,
    output_folder_path: Path,
    df_z_focus_path: Path,
    channel_index: int,
    center_crop_xy: list[int, int],
    t_reference: str = "previous",
    verbose: bool = False,
) -> ArrayLike:
    """
    Estimate the xy stabilization for a single position.

    Parameters
    ----------
    input_position_dirpath : Path
        Path to the input position directory.
    output_folder_path : Path
        Path to the output folder.
    df_z_focus_path : Path
        Path to the input focus CSV file.
    channel_index : int
        Index of the channel to process.
    center_crop_xy : list[int, int]
        Size of the crop in the XY plane.
    t_reference : str
        Reference timepoint.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Transformation matrix.
    """
    with open_ome_zarr(input_position_dirpath) as input_position:
        T, _, _, Y, X = input_position.data.shape
        x_idx = slice(X // 2 - center_crop_xy[0] // 2, X // 2 + center_crop_xy[0] // 2)
        y_idx = slice(Y // 2 - center_crop_xy[1] // 2, Y // 2 + center_crop_xy[1] // 2)

        if verbose:
            click.echo(f"Reading focus index from {df_z_focus_path}")
        df = pd.read_csv(df_z_focus_path)
        pos_idx = str(Path(*input_position_dirpath.parts[-3:]))
        focus_idx = df[df["position"] == pos_idx]["focus_idx"]
        focus_idx = focus_idx.replace(0, np.nan).ffill().fillna(focus_idx.mean())

        z_idx = focus_idx.astype(int).to_list()

        if verbose:
            click.echo("Calculating xy stabilization...")
        # Get the data for the specified channel and crop
        tyx_data = np.stack(
            [
                input_position[0][t, channel_index, z, y_idx, x_idx]
                for t, z in zip(range(T), z_idx, strict=True)
            ]
        )
        tyx_data = np.clip(tyx_data, a_min=0, a_max=None)

        sr = StackReg(StackReg.TRANSLATION)
        T_stackreg = sr.register_stack(tyx_data, reference=t_reference, axis=0)

        # Swap translation directions: (x, y) -> (y, x)
        for tform in T_stackreg:
            tform[0, 2], tform[1, 2] = tform[1, 2], tform[0, 2]

        transform = np.zeros((T_stackreg.shape[0], 4, 4))
        transform[:, 1:4, 1:4] = T_stackreg
        transform[:, 0, 0] = 1
        # save the transforms as
        position_filename = str(Path(*input_position_dirpath.parts[-3:]))
        position_filename = position_filename.replace("/", "_")

        np.save(output_folder_path / f"{position_filename}.npy", transform.astype(np.float32))

    return transform


def estimate_xy_stabilization(
    input_position_dirpaths: list[Path],
    output_folder_path: Path,
    stack_reg_settings: StackRegSettings,
    channel_index: int = 0,
    sbatch_filepath: Path | None = None,
    cluster: str = "local",
    verbose: bool = False,
) -> dict[str, list[ArrayLike]]:
    """
    Estimate XY stabilization using StackReg.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to the input position directories.
    output_folder_path : Path
        Path to the output folder.
    stack_reg_settings : StackRegSettings
        Settings for the stack registration.
    channel_index : int
        Index of the channel to process.
    sbatch_filepath : Path
        Path to the sbatch file.
    cluster : str
        Cluster to use.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    dict[str, list[ArrayLike]]
        Dictionary of the xy stabilization for each position.
    """
    input_position_dirpaths = remove_beads_fov_from_path_list(
        input_position_dirpaths, stack_reg_settings.skip_beads_fov
    )

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(exist_ok=True)

    # Estimate resources from a sample dataset
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        shape = dataset.data.shape  # (T, C, Z, Y, X)

    df_focus_path = output_folder_path / "positions_focus.csv"

    if df_focus_path.exists():
        click.echo("Using existing Z focus index file.")
    else:
        click.echo("Estimating Z focus positions...")

        estimate_z_stabilization(
            input_position_dirpaths=input_position_dirpaths,
            output_folder_path=output_folder_path,
            channel_index=channel_index,
            sbatch_filepath=sbatch_filepath,
            cluster=cluster,
            verbose=verbose,
            estimate_z_index=True,
            focus_finding_settings=stack_reg_settings.focus_finding_settings,
        )

    _, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=16, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_focus_z",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 10,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")
    output_transforms_path = output_folder_path / "xy_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_dirpath in input_position_dirpaths:
            job = executor.submit(
                estimate_xy_stabilization_per_position,
                input_position_dirpath=input_position_dirpath,
                output_folder_path=output_transforms_path,
                df_z_focus_path=df_focus_path,
                channel_index=channel_index,
                center_crop_xy=stack_reg_settings.center_crop_xy,
                t_reference=stack_reg_settings.t_reference,
                verbose=verbose,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    wait_for_jobs_to_finish(jobs)

    transforms_paths = list(output_transforms_path.glob("*.npy"))
    fov_transforms = {}

    for file_path in transforms_paths:
        fov_filename = file_path.stem
        fov_transforms[fov_filename] = np.load(file_path).tolist()

    shutil.rmtree(output_transforms_path)

    return fov_transforms


def estimate_z_focus_per_position(
    input_position_dirpath: Path,
    input_channel_indices: tuple[int, ...],
    center_crop_xy: list[int, int],
    output_path_focus_csv: Path,
    output_path_transform: Path,
    verbose: bool = False,
) -> None:
    """
    Estimate the z-focus for each timepoint and channel.

    Parameters
    ----------
    input_position_dirpath : Path
        Path to the input position directory.
    input_channel_indices : Tuple[int, ...]
        Indices of the channels to process.
    center_crop_xy : list[int, int]
        Size of the crop in the XY plane.
    output_path_focus_csv : Path
        Path to the output focus CSV file.
    output_path_transform : Path
        Path to the output transform file.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    None
    """
    position, time_idx, channel, focus_idx = [], [], [], []

    with open_ome_zarr(input_position_dirpath) as dataset:
        channel_names = dataset.channel_names
        T, _, Z, Y, X = dataset[0].shape
        _, _, _, _, pixel_size = dataset.scale

        for tc_idx in itertools.product(range(T), input_channel_indices):
            data_zyx = dataset.data[tc_idx][
                :,
                Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2,
                X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2,
            ]

            # if the FOV is empty, set the focal plane to 0
            if np.sum(data_zyx) == 0:
                z_idx = 0
            else:
                z_idx = focus_from_transverse_band(
                    data_zyx,
                    NA_det=NA_DET,
                    lambda_ill=LAMBDA_ILL,
                    pixel_size=pixel_size,
                )
                click.echo(
                    f"Estimating focus for timepoint {tc_idx[0]} and channel {tc_idx[1]}: {z_idx}"
                )

            position.append(str(Path(*input_position_dirpath.parts[-3:])))
            time_idx.append(tc_idx[0])
            channel.append(channel_names[tc_idx[1]])
            focus_idx.append(z_idx)

    df = pd.DataFrame(
        {
            "position": position,
            "time_idx": time_idx,
            "channel": channel,
            "focus_idx": focus_idx,
        }
    )

    output_path_focus_csv.mkdir(parents=True, exist_ok=True)
    if verbose:
        click.echo(f"Saving focus finding results to {output_path_focus_csv}")

    position_filename = str(Path(*input_position_dirpath.parts[-3:])).replace("/", "_")
    output_csv = output_path_focus_csv / f"{position_filename}.csv"
    df.to_csv(output_csv, index=False)

    # Compute Z drifts
    z_focus_shift = [np.eye(4)]

    z_val = next((v for v in focus_idx if v != 0), None)
    if z_val is None:
        raise ValueError("Z index of focus reference is None, focus_idx contains only zeros")

    for z_val_next in focus_idx[1:]:
        shift = np.eye(4)
        # Set the translation components of the transform
        shift[0, 3] = z_val_next - z_val
        z_focus_shift.append(shift)

    transform = np.array(z_focus_shift)

    # Save the transform
    output_path_transform.mkdir(parents=True, exist_ok=True)
    np.save(output_path_transform / f"{position_filename}.npy", transform)

    if verbose:
        click.echo(f"Saved Z transform matrices to {output_path_transform}")


def get_mean_z_positions(
    dataframe_path: Path,
    verbose: bool = False,
    method: Literal["mean", "median"] = "mean",
) -> None:
    """
    Get the mean or median z-focus for each timepoint.

    Parameters
    ----------
    dataframe_path : Path
        Path to the input focus CSV file.
    verbose : bool
        If True, print verbose output.
    method : Literal["mean", "median"]
        Method to use for averaging the z-focus.

    Returns
    -------
    np.ndarray
        Array of the mean or median z-focus for each timepoint.
    """
    df = pd.read_csv(dataframe_path)

    df = df.sort_values("time_idx")

    # When focus finding fails, it may return 0, which here is replaced with NaN
    # before calculating the mean focus index per position
    df["focus_idx"] = df["focus_idx"].replace(0, np.nan)

    # Get the mean of positions for each time point
    if method == "mean":
        average_focus_idx = df.groupby("time_idx")["focus_idx"].mean().reset_index()
    elif method == "median":
        average_focus_idx = df.groupby("time_idx")["focus_idx"].median().reset_index()

    if verbose:
        import matplotlib.pyplot as plt

        plt.plot(average_focus_idx["focus_idx"], linestyle="--", label="mean of all positions")
        plt.xlabel("Time index")
        plt.ylabel("Focus index")
        plt.ylim(0, 100)
        plt.legend()
        plt.savefig(dataframe_path.parent / "z_drift.png")

    return average_focus_idx["focus_idx"].values


def estimate_z_stabilization(
    input_position_dirpaths: list[Path],
    output_folder_path: Path,
    focus_finding_settings: FocusFindingSettings,
    channel_index: int,
    sbatch_filepath: Path | None = None,
    cluster: str = "local",
    verbose: bool = False,
    estimate_z_index: bool = False,
) -> dict[str, list[ArrayLike]]:
    """
    Estimate the z stabilization for a list of positions.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to the input position directories.
    output_folder_path : Path
        Path to the output folder.
    focus_finding_settings : FocusFindingSettings
        Settings for the focus finding.
    channel_index : int
        Index of the channel to process.
    sbatch_filepath : Path
        Path to the sbatch file.
    cluster : str
        Cluster to use.
    verbose : bool
        If True, print verbose output.
    estimate_z_index : bool
        If True, estimate the z index and save the focus csv without saving the transforms (for xy stabilization).

    Returns
    -------
    dict[str, list[ArrayLike]]
        Dictionary of the z stabilization for each position.
    """
    input_position_dirpaths = remove_beads_fov_from_path_list(
        input_position_dirpaths, focus_finding_settings.skip_beads_fov
    )

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(exist_ok=True)

    # Estimate resources from a sample dataset
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        shape = dataset.data.shape  # (T, C, Z, Y, X)

    _, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=16, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_focus_z",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 30,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")
    output_folder_focus_path = output_folder_path / "z_focus_positions"
    output_folder_focus_path.mkdir(parents=True, exist_ok=True)

    output_transforms_path = output_folder_path / "z_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    jobs = []

    with submitit.helpers.clean_env(), executor.batch():
        for input_position_dirpath in input_position_dirpaths:
            job = executor.submit(
                estimate_z_focus_per_position,
                input_position_dirpath=input_position_dirpath,
                input_channel_indices=(channel_index,),
                center_crop_xy=focus_finding_settings.center_crop_xy,
                output_path_focus_csv=output_folder_focus_path,
                output_path_transform=output_transforms_path,
                verbose=verbose,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    wait_for_jobs_to_finish(jobs)

    # Load the focus CSV files and concatenate them
    focus_csvs_path = list(output_folder_focus_path.glob("*.csv"))
    if len(focus_csvs_path) != len(input_position_dirpaths):
        click.echo(
            f"Warning: {len(focus_csvs_path)} focus CSV files found for {len(input_position_dirpaths)} input data paths."
        )
    df = pd.concat([pd.read_csv(f) for f in focus_csvs_path])

    # Check if the existing focus CSV file exists
    if Path(output_folder_path / "positions_focus.csv").exists():
        click.echo("Using existing focus CSV file.")
        df_old = pd.read_csv(output_folder_path / "positions_focus.csv")
        df = pd.concat([df, df_old])
        df = df.drop_duplicates(subset=["position", "time_idx"])
    df = df.sort_values(["position", "time_idx"])
    df.to_csv(output_folder_path / "positions_focus.csv", index=False)

    # Remove the output temporary folder
    shutil.rmtree(output_folder_focus_path)

    if estimate_z_index:
        shutil.rmtree(output_transforms_path)
        return

    if focus_finding_settings.average_across_wells:
        z_drift_offsets = get_mean_z_positions(
            dataframe_path=output_folder_path / "positions_focus.csv",
            method=focus_finding_settings.average_across_wells_method,
            verbose=verbose,
        )

        # Initialize the z-focus shift
        z_focus_shift = [np.eye(4)]
        z_val = next((v for v in z_drift_offsets if v != 0 and not np.isnan(v)), None)
        if z_val is None:
            raise ValueError(
                "Z index of focus reference is None; no valid (non-zero, non-NaN) z-index found in z_drift_offsets"
            )
        transform = {}

        # Compute the z-focus shift for each timepoint
        for z_val_next in z_drift_offsets[1:]:
            # Set the translation components of the transform
            shift = np.eye(4)
            shift[0, 3] = z_val_next - z_val
            z_focus_shift.append(shift)
        transform["average"] = np.array(z_focus_shift).tolist()

        if verbose:
            click.echo(f"Saving z focus shift matrices to {output_folder_path}")
            np.save(output_folder_path / "z_focus_shift.npy", transform["average"])

        return transform
    else:
        # Load the transforms
        transforms_paths = list(output_transforms_path.glob("*.npy"))
        fov_transforms = {}

        for file_path in transforms_paths:
            transform = np.load(file_path).tolist()
            fov_filename = file_path.stem
            fov_transforms[fov_filename] = transform

        # Remove the output temporary folder
        shutil.rmtree(output_transforms_path)

    return fov_transforms


def estimate_stabilization(
    input_position_dirpaths: list[str],
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
    cluster = get_submitit_cluster(local)

    # Load the evaluation settings
    eval_transform_settings = settings.eval_transform_settings

    if "xyz" == stabilization_type:
        if stabilization_method == "focus-finding":
            click.echo(
                "Estimating xyz stabilization parameters with focus finding and stack registration"
            )

            z_transforms_dict = estimate_z_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                focus_finding_settings=settings.focus_finding_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

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
                        [a @ b for a, b in zip(xy_transforms, z_transforms, strict=True)]
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
            from biahub.estimate_transform import estimate_transform_series
            from biahub.registration.legacy import legacy_pull_from_forward

            click.echo("Estimating xyz stabilization parameters with beads")
            engine_settings = EstimateRegistrationSettings(
                target_channel_name=stabilization_estimation_channel,
                source_channel_name=stabilization_estimation_channel,
                estimation_method="beads",
                beads_match_settings=settings.beads_match_settings,
                affine_transform_settings=settings.affine_transform_settings,
                verbose=verbose,
            )
            _result, _time_indices, forward_transforms = estimate_transform_series(
                input_position_dirpaths[0],
                input_position_dirpaths[0],
                engine_settings,
                output_dirpath,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                reference_kind=settings.affine_transform_settings.t_reference,
            )
            xyz_transforms = [
                legacy_pull_from_forward(transform) for transform in forward_transforms
            ]

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

            from concurrent.futures import ThreadPoolExecutor

            from biahub.estimate_transform import estimate_transform_series
            from biahub.registration.legacy import legacy_pull_from_forward

            pcc_settings = settings.phase_cross_corr_settings
            engine_settings = EstimateRegistrationSettings(
                target_channel_name=stabilization_estimation_channel,
                source_channel_name=stabilization_estimation_channel,
                estimation_method="phase-cross-corr",
                phase_cross_corr_settings=pcc_settings,
                affine_transform_settings=settings.affine_transform_settings,
                verbose=verbose,
            )
            positions = remove_beads_fov_from_path_list(
                input_position_dirpaths, pcc_settings.skip_beads_fov
            )

            def estimate_position(position_dirpath):
                fov = "_".join(Path(position_dirpath).parts[-3:])
                _result, _time_indices, forward = estimate_transform_series(
                    position_dirpath,
                    position_dirpath,
                    engine_settings,
                    output_dirpath / "estimate_transform" / fov,
                    sbatch_filepath=sbatch_filepath,
                    cluster=cluster,
                    reference_kind=pcc_settings.t_reference,
                )
                return fov, [legacy_pull_from_forward(t) for t in forward]

            # One driver per position, concurrently: each fans its own timepoints out.
            with ThreadPoolExecutor(max_workers=max(1, len(positions))) as pool:
                xyz_transforms_dict = dict(pool.map(estimate_position, positions))

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


@click.command("estimate-stabilization")
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
def estimate_stabilization_cli(
    input_position_dirpaths: list[str],
    output_dirpath: str,
    config_filepath: Path,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """Estimate translation matrices for XYZ stabilization of a timelapse dataset.

    Stabilization parameters may be computed for the XY, Z, or XYZ dimensions using
    focus finding, beads, or phase cross correlation methods.

    >>> biahub estimate-stabilization \
        -i ./timelapse.zarr/0/0/0 \
        -o ./stabilization.yml \
        -c ./config.yml \
        -s ./sbatch.sh \
        --local --verbose
    """
    estimate_stabilization(
        input_position_dirpaths=input_position_dirpaths,
        output_dirpath=output_dirpath,
        config_filepath=config_filepath,
        sbatch_filepath=sbatch_filepath,
        local=local,
    )


if __name__ == "__main__":
    estimate_stabilization_cli()
