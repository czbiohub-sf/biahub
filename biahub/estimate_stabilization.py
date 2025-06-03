import itertools
import multiprocessing as mp

from functools import partial
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd

from iohub.ngff import open_ome_zarr
from pystackreg import StackReg
from waveorder.focus import focus_from_transverse_band

from biahub.cli.parsing import input_position_dirpaths, output_filepath, config_filepath, sbatch_filepath, local
from biahub.cli.utils import model_to_yaml, yaml_to_model
from biahub.settings import StabilizationSettings

NA_DET = 1.35
LAMBDA_ILL = 0.500

def estimate_position_focus(
    input_data_path: Path,
    input_channel_indices: Tuple[int, ...],
    crop_size_xy: list[int, int],
    output_path_focus_csv: Path,
    output_path_transform: Path,
    verbose: bool = False,
) -> None:
 
    position, time_idx, channel, focus_idx = [], [], [], []

    with open_ome_zarr(input_data_path) as dataset:
        channel_names = dataset.channel_names
        T, _, Z, Y, X = dataset[0].shape
        _, _, _, _, pixel_size = dataset.scale

        for tc_idx in itertools.product(range(T), input_channel_indices):
            data_zyx = dataset.data[tc_idx][
                :,
                Y // 2 - crop_size_xy[1] // 2 : Y // 2 + crop_size_xy[1] // 2,
                X // 2 - crop_size_xy[0] // 2 : X // 2 + crop_size_xy[0] // 2,
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

            position.append(str(Path(*input_data_path.parts[-3:])))
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

    position_filename = str(Path(*input_data_path.parts[-3:])).replace("/", "_")
    output_csv = output_path_focus_csv / f"{position_filename}.csv"
    df.to_csv(output_csv, index=False)

    # ---- Generate and save Z transformation matrix per timepoint

     # Compute Z drifts


    z_focus_shift = [np.eye(4)]
    z_val = focus_idx[0]
    for z_val_next in focus_idx[1:]:
        shift = np.eye(4)
        shift[0, 3] = z_val_next - z_val
        z_focus_shift.append(shift)
    T_z_drift_mats = np.array(z_focus_shift)

    
    output_path_transform.mkdir(parents=True, exist_ok=True)
    np.save(output_path_transform/ f"{position_filename}.npy", T_z_drift_mats)

    if verbose:
        click.echo(f"Saved Z transform matrices to {output_path_transform}")



def get_mean_z_positions(dataframe_path: Path, method: Literal["mean", "median"] = "mean", verbose: bool = False) -> None:
    """
    Get the mean or median of the focus index for each timepoint.
    
    
    """
    df = pd.read_csv(dataframe_path)

    # Sort the DataFrame based on 'time_idx'
    df = df.sort_values("time_idx")

    # TODO: this is a hack to deal with the fact that the focus finding function returns 0 if it fails
    df["focus_idx"] = df["focus_idx"].replace(0, np.nan).ffill()

    # Get the mean of positions for each time point
    if method == "mean":
        average_focus_idx = df.groupby("time_idx")["focus_idx"].mean().reset_index()
    elif method == "median":
        average_focus_idx = df.groupby("time_idx")["focus_idx"].median().reset_index()
    else:
        raise ValueError("Unknown averaging method.")

    return average_focus_idx["focus_idx"].values


def estimate_z_stabilization(
    input_data_paths: list[Path],
    output_folder_path: Path,
    channel_index: int,
    crop_size_xy: list[int],
    sbatch_filepath: Optional[Path] = None,
    cluster: str = "local",
    estimate_z_index: bool = False,
    z_focus_files_path: bool = False,
    verbose: bool = False,

) -> np.ndarray:
    """
    Submit SLURM jobs to estimate per-position Z focus and return averaged drift matrices.
    """

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    output_folder_focus_path = output_folder_path / "z_focus_positions"
    output_folder_focus_path.mkdir(parents=True, exist_ok=True)

    output_transforms_path = output_folder_path / "z_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    # Estimate resources from a sample dataset
    with open_ome_zarr(input_data_paths[0]) as dataset:
        shape = dataset.data.shape  # (T, C, Z, Y, X)

    num_cpus, gb_ram_per_cpu = estimate_resources(shape=shape, ram_multiplier=16, max_num_cpus=16)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_stabilization_z",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 30,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))
    
    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo('Submitting SLURM jobs...')
    jobs = []

    with executor.batch():
        for input_data_path in input_data_paths:
            job = executor.submit(
                estimate_position_focus,
                input_data_path=input_data_path,
                input_channel_indices=(channel_index,),
                crop_size_xy=crop_size_xy,
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

    # Wait for all jobs to finish
    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)

    # Aggregate results
    z_focus_files_path = list(output_folder_focus_path.glob("*.csv"))
    
    # Check length of z_focus_files_path
    if len(z_focus_files_path) == 0:
        click.echo("No focus CSV files found. Exiting.")
        return
    elif len(z_focus_files_path) != len(input_data_paths):
        click.echo(f"Warning: {len(z_focus_files_path)} focus CSV files found for {len(input_data_paths)} input data paths.")

    df = pd.concat([pd.read_csv(f) for f in z_focus_files_path])
    # sort by position and time_idx
    if Path(output_folder_path / "positions_focus.csv").exists():
        click.echo("Update existing focus CSV file.")
        # read the existing CSV file
        df_old = pd.read_csv(output_folder_path / "positions_focus.csv")
        # concatenate the new and old dataframes
        df = pd.concat([df, df_old])
        # drop duplicates
        df = df.drop_duplicates(subset=["position", "time_idx"])
        # sort by position and time_idx
    df = df.sort_values(["position", "time_idx"])
    # Save the results to a CSV file
    df.to_csv(output_folder_path / "positions_focus.csv", index=False)

    shutil.rmtree(output_folder_focus_path)

    if estimate_z_index:
        shutil.rmtree(output_transforms_path)
        return
    
    if global_z_focus_index:
    # # Compute Z drifts
        z_drift_offsets = get_mean_z_positions(
            dataframe_path=output_folder_path / "positions_focus.csv",
            method="median",
            verbose=verbose,
        )

        z_focus_shift = [np.eye(4)]
        z_val = z_drift_offsets[0]
        transform = {}
        for z_val_next in z_drift_offsets[1:]:
            shift = np.eye(4)
            shift[0, 3] = z_val_next - z_val
            z_focus_shift.append(shift)
        transform["z_focus_files_path"] = np.array(z_focus_shift).tolist()

        if verbose:
            click.echo(f"Saving z focus shift matrices to {output_folder_path}")
            np.save(output_folder_path / "z_transforms.npy", transform["z_focus_files_path"])

        return transform
    else:
        # Get list of .npy transform files
        transforms_paths = list(output_transforms_path.glob("*.npy"))

        # Load and collect all transform arrays
        fov_transforms = {}

        for file_path in transforms_paths:
            T_zyx_shift = np.load(file_path).tolist()
            fov_filename = file_path.stem
            # 
            fov_transforms[fov_filename] = T_zyx_shift
        click.echo(f"Saved {len(fov_transforms)} z transform matrices to {output_folder_path}")
        shutil.rmtree(output_transforms_path)

    return fov_transforms

def get_mean_z_positions(dataframe_path: Path, verbose: bool = False) -> None:
    df = pd.read_csv(dataframe_path)

    # Sort the DataFrame based on 'time_idx'
    df = df.sort_values("time_idx")

    # TODO: this is a hack to deal with the fact that the focus finding function returns 0 if it fails
    df["focus_idx"] = df["focus_idx"].replace(0, np.nan).ffill()

    # Get the mean of positions for each time point
    average_focus_idx = df.groupby("time_idx")["focus_idx"].mean().reset_index()

    if verbose:
        import matplotlib.pyplot as plt

        # Get the moving average of the focus_idx
        plt.plot(average_focus_idx["focus_idx"], linestyle="--", label="mean of all positions")
        plt.xlabel('Time index')
        plt.ylabel('Focus index')
        plt.legend()
        plt.savefig(dataframe_path.parent / "z_drift.png")

    return average_focus_idx["focus_idx"].values



def estimate_xy_stabilization(
    input_data_paths: Path,
    output_folder_path: Path,
    c_idx: int = 0,
    crop_size_xy: list[int, int] = (400, 400),
    verbose: bool = False,
) -> np.ndarray:
    input_data_path = input_data_paths[0]
    input_position = open_ome_zarr(input_data_path)
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    focus_params = {
        "NA_det": NA_DET,
        "lambda_ill": LAMBDA_ILL,
        "pixel_size": input_position.scale[-1],
    }

    # Get metadata
    T, C, Z, Y, X = input_position.data.shape
    x_idx = slice(X // 2 - crop_size_xy[0] // 2, X // 2 + crop_size_xy[0] // 2)
    y_idx = slice(Y // 2 - crop_size_xy[1] // 2, Y // 2 + crop_size_xy[1] // 2)

    if (output_folder_path / "positions_focus.csv").exists():
        df = pd.read_csv(output_folder_path / "positions_focus.csv")
        pos_idx = str(Path(*input_data_path.parts[-3:]))
        focus_idx = df[df["position"] == pos_idx]["focus_idx"]
        # forward fill 0 values, when replace remaining NaN with the mean
        focus_idx = focus_idx.replace(0, np.nan).ffill()
        focus_idx = focus_idx.fillna(focus_idx.mean())
        z_idx = focus_idx.astype(int).to_list()
    else:
        z_idx = [
            focus_from_transverse_band(
                input_position[0][
                    0,
                    c_idx,
                    :,
                    y_idx,
                    x_idx,
                ],
                **focus_params,
            )
        ] * T
        if verbose:
            click.echo(f"Estimated in-focus slice: {z_idx}")

    # Load timelapse and ensure negative values are not present
    tyx_data = np.stack(
        [
            input_position[0][_t_idx, c_idx, _z_idx, y_idx, x_idx]
            for _t_idx, _z_idx in zip(range(T), z_idx)
        ]
    )
    tyx_data = np.clip(tyx_data, a_min=0, a_max=None)

    # register each frame to the previous (already registered) one
    # this is what the original StackReg ImageJ plugin uses
    sr = StackReg(StackReg.TRANSLATION)

    T_stackreg = sr.register_stack(tyx_data, reference="previous", axis=0)

    # Swap values in the array since stackreg is xy and we need yx
    for subarray in T_stackreg:
        subarray[0, 2], subarray[1, 2] = subarray[1, 2], subarray[0, 2]

    T_zyx_shift = np.zeros((T_stackreg.shape[0], 4, 4))
    T_zyx_shift[:, 1:4, 1:4] = T_stackreg
    T_zyx_shift[:, 0, 0] = 1

    # Save the translation matrices
    if verbose:
        click.echo(f"Saving translation matrices to {output_folder_path}")
        yx_shake_translation_tx_filepath = (
            output_folder_path / "yx_shake_translation_tx_ants.npy"
        )
        np.save(yx_shake_translation_tx_filepath, T_zyx_shift)

    input_position.close()

    return T_zyx_shift


@click.command("estimate-stabilization")
@input_position_dirpaths()
@output_filepath()
@config_filepath()
@sbatch_filepath()
@local()
def estimate_stabilization_cli(
    input_position_dirpaths: List[str],
    output_filepath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate the Z and/or XY timelapse stabilization matrices.

    This function estimates xy and z drifts and returns the affine matrices per timepoint taking t=0 as reference saved as a yaml file.
    The level of verbosity can be controlled with the verbose flag.
    The size of the crop in xy can be specified with the crop-size-xy option.

    Example usage:
    biahub stabilization -i ./timelapse.zarr/0/0/0 -o ./stabilization.yml -y -z -v --crop-size-xy 300 300

    Note: the verbose output will be saved at the same level as the output zarr.
    """
    # Load the settings
    config_filepath = Path(config_filepath)

    settings = yaml_to_model(config_filepath, EstimateStabilizationSettings)
    click.echo(f"Settings: {settings}")

    if not (stabilize_xy or stabilize_z):
        raise ValueError("At least one of 'stabilize_xy' or 'stabilize_z' must be selected")

    if output_filepath.suffix not in [".yml", ".yaml"]:
        raise ValueError("Output file must be a yaml file")

    output_dirpath = output_filepath.parent
    output_dirpath.mkdir(parents=True, exist_ok=True)

    verbose = settings.verbose
    crop_size_xy = settings.crop_size_xy
    estimate_stabilization_channel = settings.estimate_stabilization_channel
    stabilization_type = settings.stabilization_type
    stabilization_method = settings.stabilization_method
    skip_beads_fov = settings.skip_beads_fov
    average_across_wells = settings.average_across_wells


    if skip_beads_fov != '0':
        # Remove the beads FOV from the input data paths
        click.echo(f"Removing beads FOV {skip_beads_fov} from input data paths")
        input_position_dirpaths = [
            path for path in input_position_dirpaths if skip_beads_fov not in str(path)
        ]

  

    # Channel names to process
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        channel_names = dataset.channel_names
        voxel_size = dataset.scale
        channel_index = channel_names.index(estimate_stabilization_channel)
        T, C, Z, Y, X = dataset.data.shape

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # # Channel names to process
    # stabilization_channel_names = []
    # with open_ome_zarr(input_position_dirpaths[0]) as dataset:
    #     channel_names = dataset.channel_names
    #     voxel_size = dataset.scale
    # if len(stabilization_channel_indices) < 1:
    #     stabilization_channel_indices = range(len(channel_names))
    #     stabilization_channel_names = channel_names
    # else:
    #     # Make the input a list
    #     stabilization_channel_indices = list(stabilization_channel_indices)
    #     stabilization_channel_names = []
    #     # Check the channel indeces are valid
    #     for c_idx in stabilization_channel_indices:
    #         if c_idx not in range(len(channel_names)):
    #             raise ValueError(
    #                 f"Channel index {c_idx} is not valid. Please provide channel indeces from 0 to {len(channel_names)-1}"
    #             )
    #         else:
    #             stabilization_channel_names.append(channel_names[c_idx])

    # Estimate stabilization parameters
    if stabilization_type == "z":
        if stabilization_method == "focus-finding":
            click.echo("Estimating z stabilization parameters")

            T_z_drift_mats_dict = estimate_z_stabilization(
                    input_data_paths=input_position_dirpaths,
                    output_folder_path=output_dirpath,
                    channel_index=channel_index,
                    crop_size_xy=crop_size_xy,
                    sbatch_filepath=sbatch_filepath,
                    global_z_focus_index=global_z_focus_index,
                    cluster=cluster,
                    verbose=verbose,
                )
            os.makedirs(output_dirpath / "z_stabilization_settings", exist_ok=True)
            if verbose:
                os.makedirs(output_dirpath / "translation_plots", exist_ok=True)
            # save each FOV separately
            for fov, transforms in T_z_drift_mats_dict.items():
                # Validate and filter transforms
                transforms = _validate_transforms(
                    transforms=transforms,
                    window_size=settings.affine_transform_validation_window_size,
                    tolerance=settings.affine_transform_validation_tolerance,
                    Z=Z,
                    Y=Y,
                    X=X,
                    verbose=verbose,
                )
                # Interpolate missing transforms
                transforms = _interpolate_transforms(
                    transforms=transforms,
                    window_size=settings.affine_transform_interpolation_window_size,
                    interpolation_type=settings.affine_transform_interpolation_type,
                    verbose=verbose,
                )
                output_filepath_fov = output_dirpath / "z_stabilization_settings" / f"{fov}.yml"
                # Save the combined matrices
                model = StabilizationSettings(
                    stabilization_type=stabilization_type,
                    stabilization_method=stabilization_method,
                    stabilization_estimation_channel=estimate_stabilization_channel,
                    stabilization_channels=settings.stabilization_channels,
                    affine_transform_zyx_list=transforms,
                    time_indices="all",
                    output_voxel_size=voxel_size,
                )
                model_to_yaml(model, output_filepath_fov)

                if verbose:
                    os.makedirs(output_dirpath / "translation_plots", exist_ok=True)

                    transforms = np.array(transforms)

                    z_transforms = transforms[:, 0, 3] #->ZYX
                    y_transforms = transforms[:, 1, 3] #->ZYX
                    x_transforms = transforms[:, 2, 3] #->ZYX

                    plt.plot(z_transforms)
                    plt.plot(x_transforms)
                    plt.plot(y_transforms)
                    plt.legend(["Z-Translation", "X-Translation", "Y-Translation"])
                    plt.xlabel("Timepoint")
                    plt.ylabel("Translations")
                    plt.title("Translations Over Time")
                    plt.grid()
                    # Save the figure
                    plt.savefig(output_dirpath/"translation_plots"/f"{fov}.png", dpi=300, bbox_inches='tight')
                    plt.close()

        else:
            raise ValueError(f"Invalid stabilization method: {stabilization_method}")
    elif stabilization_type == "xy":
        if stabilization_method == "focus-finding":
            pass
        else:
            raise ValueError(f"Invalid stabilization method: {stabilization_method}")
    elif stabilization_type == "xyz":
        if stabilization_method == "focus-finding":
        elif stabilization_method == "phase-cross-corr":
        elif stabilization_method == "match_peaks":
            pass
        else:
            raise ValueError(f"Invalid stabilization method: {stabilization_method}")
    else:
        raise ValueError(f"Invalid stabilization type: {stabilization_type}")
        

    

    if stabilize_z:
        click.echo("Estimating z stabilization parameters")
        T_z_drift_mats = estimate_z_stabilization(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            z_drift_channel_idx=channel_index,
            num_processes=num_processes,
            crop_size_xy=crop_size_xy,
            verbose=verbose,
        )
        stabilization_type = "z"

    # Estimate yx drift
    if stabilize_xy:
        click.echo("Estimating xy stabilization parameters")
        T_translation_mats = estimate_xy_stabilization(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            c_idx=channel_index,
            crop_size_xy=crop_size_xy,
            verbose=verbose,
        )
        stabilization_type = "xy"

    if stabilize_z and stabilize_xy:
        if T_translation_mats.shape[0] != T_z_drift_mats.shape[0]:
            raise ValueError(
                "The number of translation matrices and z drift matrices must be the same"
            )
        combined_mats = np.array([a @ b for a, b in zip(T_translation_mats, T_z_drift_mats)])
        stabilization_type = "xyz"

    # NOTE: we've checked that one of the two conditions below is true
    elif stabilize_z:
        combined_mats = T_z_drift_mats
    elif stabilize_xy:
        combined_mats = T_translation_mats

    # Save the combined matrices
    model = StabilizationSettings(
        stabilization_type=stabilization_type,
        stabilization_estimation_channel=channel_names[channel_index],
        stabilization_channels=stabilization_channel_names,
        affine_transform_zyx_list=combined_mats.tolist(),
        time_indices="all",
        output_voxel_size=voxel_size,
    )
    model_to_yaml(model, output_filepath)


if __name__ == "__main__":
    estimate_stabilization_cli()
