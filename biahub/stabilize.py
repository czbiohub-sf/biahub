from pathlib import Path
from typing import List

import ants
import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as R

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepaths,
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    copy_n_paste_czyx,
    create_empty_hcs_zarr,
    estimate_resources,
    process_single_position_v2,
    yaml_to_model,
)
from biahub.register import convert_transform_to_ants
from biahub.settings import StabilizationSettings


def apply_stabilization_transform(
    zyx_data: np.ndarray,
    list_of_shifts: list[np.ndarray],
    t_idx: int,
    output_shape: tuple[int, int, int] = None,
):
    """
    Apply stabilization transformations to 3D or 4D volumetric data.

    This function applies a time-indexed stabilization transformation to a single 3D (Z, Y, X) volume
    or a 4D (C, Z, Y, X) volume using a precomputed list of transformations.

    Parameters:
    - zyx_data (np.ndarray): Input 3D (Z, Y, X) or 4D (C, Z, Y, X) volumetric data.
    - list_of_shifts (list[np.ndarray]): List of transformation matrices (one per time index).
    - t_idx (int): Time index corresponding to the transformation to apply.
    - output_shape (tuple[int, int, int], optional): Desired shape of the output stabilized volume.
                                                     If None, the shape of `zyx_data` is used.
                                                     Defaults to None.

    Returns:
    - np.ndarray: The stabilized 3D (Z, Y, X) or 4D (C, Z, Y, X) volume.

    Notes:
    - If `zyx_data` is 4D, the function recursively applies stabilization to each channel (C).
    - Uses ANTsPy for applying the transformation to the input data.
    - Handles `NaN` values in the input by replacing them with 0 before applying the transformation.
    - Echoes the transformation matrix for debugging purposes when verbose logging is enabled.
    """

    if output_shape is None:
        output_shape = zyx_data.shape[-3:]

    # Get the transformation matrix for the current time index
    tx_shifts = convert_transform_to_ants(list_of_shifts[t_idx])

    if zyx_data.ndim == 4:
        stabilized_czyx = np.zeros((zyx_data.shape[0],) + output_shape, dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            stabilized_czyx[c] = apply_stabilization_transform(
                zyx_data[c], list_of_shifts, t_idx, output_shape
            )
        return stabilized_czyx
    else:
        click.echo(f'shifting matrix with t_idx:{t_idx} \n{list_of_shifts[t_idx]}')
        target_zyx_ants = ants.from_numpy(np.zeros((output_shape), dtype=np.float32))

        zyx_data = np.nan_to_num(zyx_data, nan=0)
        zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
        stabilized_zyx = tx_shifts.apply_to_image(
            zyx_data_ants, reference=target_zyx_ants
        ).numpy()

    return stabilized_zyx


@click.command("stabilize")
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
@monitor()
def stabilize_cli(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepaths: list[str],
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True,
):
    """
    Stabilize a timelapse dataset by applying spatial transformations.

    This function stabilizes a timelapse dataset based on precomputed transformations or
    configuration settings. It supports both local processing and SLURM-based distributed
    processing and outputs a Zarr dataset with stabilized channels.

    Parameters:
    - input_position_dirpaths (List[str]): List of file paths to the input OME-Zarr datasets for each position.
    - output_dirpath (str): Directory path to save the stabilized output dataset.
    - config_filepaths (list[str]): Paths to the YAML configuration files containing transformation settings.
    - sbatch_filepath (str, optional): Path to a SLURM sbatch file to override default SLURM settings. Defaults to None.
    - local (bool, optional): If True, runs the stabilization process locally instead of submitting to SLURM. Defaults to False.

    Returns:
    - None: Writes the stabilized dataset to the specified output directory.

    Notes:
    - The function applies stabilization based on affine transformations specified in the configuration file.
    - Stabilization can estimate both YX and Z drifts and handles multi-channel data.
    - Input and output datasets must follow the OME-Zarr format.

    Example:
    >> biahub stabilize-timelapse
        -i ./timelapse.zarr/0/0/0               # Input timelapse dataset
        -o ./stabilized_timelapse.zarr          # Output directory for stabilized data
        -c ./file_w_matrices.yml                # Configuration file with transformation matrices
        -v                                      # Verbose mode for detailed logs
        --local                                 # Run locally instead of submitting to SLURM

    """

    # Single config file for all FOVs
    if len(config_filepaths) == 1:
        config_filepath = Path(config_filepaths[0])
    else:
        config_filepath = None

    settings = yaml_to_model(config_filepaths[0], StabilizationSettings)

    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"
    # Load the config file

    combined_mats = settings.affine_transform_zyx_list
    combined_mats = np.array(combined_mats)
    # stabilization_channels = settings.stabilization_channels

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape
        channel_names = dataset.channel_names
        stabilization_channels = channel_names
        # for stabilization_channels in stabilization_channels:
        # if channel not in channel_names:
        #     raise ValueError(f"Channel <{channel}> not found in the input data")

        # NOTE: these can be modified to crop the output
        Z_slice, Y_slice, X_slice = (
            slice(0, Z),
            slice(0, Y),
            slice(0, X),
        )
        Z = Z_slice.stop - Z_slice.start

    # Get the rotation matrix

    # Extract 3x3 rotation/scaling matrix
    R_matrix = combined_mats[0][:3, :3]

    # Remove scaling using SVD
    U, _, Vt = svd(R_matrix)
    R_pure = np.dot(U, Vt)

    # Convert to Euler angles
    rotation = R.from_matrix(R_pure)
    euler_angles = rotation.as_euler('xyz', degrees=True)  # XYZ order, in degrees

    if np.isclose(euler_angles[0], 90, atol=10):
        X = Y_slice.stop - Y_slice.start
        Y = X_slice.stop - X_slice.start
    else:
        Y = Y_slice.stop - Y_slice.start
        X = X_slice.stop - X_slice.start

    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    # Attempted to calculate the new scale from the input affine transform,
    # but chose to use the `output_voxel_size` instead to ensure consistency in scale.
    # The computed scale value was close to the desired voxel size but not exact.

    # transform_t0_sy calculates the scale factor for the YZ plane (shear factor)
    # derived from the affine transform matrix.
    # It uses the [2][1] element of the first affine matrix (T0) and rounds it to 3 decimal places.
    # transform_t0_sy = np.abs(settings.affine_transform_zyx_list[0][2][1]).round(3)

    # Calculate the new scale for the dataset:
    # The first three elements represent the spatial scaling factors for X, Y, and Z dimensions.
    # The last two elements adjust the temporal scaling based on `transform_t0_sy`.
    # Note: Temporal scaling is applied to dimensions 3 and 4.
    # new_scale = [
    #     scale_dataset[0],  # X-dimension scaling factor (remains unchanged)
    #     scale_dataset[1],  # Y-dimension scaling factor (remains unchanged)
    #     scale_dataset[2],  # Z-dimension scaling factor (remains unchanged)
    #     scale_dataset[3] * transform_t0_sy,  # Adjust temporal scaling for the 4th dimension.
    #     scale_dataset[4] * transform_t0_sy   # Adjust temporal scaling for the 5th dimension.
    # ]

    output_metadata = {
        "shape": (len(time_indices), len(channel_names), Z, Y, X),
        "chunks": None,
        "scale": settings.output_voxel_size,
        "channel_names": channel_names,
        "dtype": np.float32,
    }

    # Create the output zarr mirroring input_position_dirpaths
    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        **output_metadata,
    )

    stabilize_zyx_args = {"list_of_shifts": combined_mats}
    copy_n_paste_kwargs = {"czyx_slicing_params": ([Z_slice, Y_slice, X_slice])}

    # Estimate resources

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "stabilize",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 20,
        "slurm_partition": "preempted",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo('Submitting SLURM jobs...')
    jobs = []

    with executor.batch():
        # apply stabilization to channels in the chosen channels and else copy the rest
        for input_position_path in input_position_dirpaths:
            if not config_filepath:
                fov = "_".join(input_position_path.parts[-3:])
                config_filepath = [p for p in config_filepaths if fov in p.name][0]

            settings = yaml_to_model(config_filepath, StabilizationSettings)
            # Use settings for this FOV
            combined_mats = np.array(settings.affine_transform_zyx_list)
            stabilize_zyx_args = {"list_of_shifts": combined_mats}
            for channel_name in channel_names:
                if channel_name in stabilization_channels:
                    job = executor.submit(
                        process_single_position_v2,
                        apply_stabilization_transform,
                        input_data_path=input_position_path,  # source store
                        output_path=output_dirpath,
                        time_indices=time_indices,
                        output_shape=(Z, Y, X),
                        input_channel_idx=[channel_names.index(channel_name)],
                        output_channel_idx=[channel_names.index(channel_name)],
                        num_processes=int(
                            slurm_args["slurm_cpus_per_task"]
                        ),  # parallel processing over time
                        **stabilize_zyx_args,
                    )
                else:
                    job = executor.submit(
                        process_single_position_v2,
                        copy_n_paste_czyx,
                        input_data_path=input_position_path,  # target store
                        output_path=output_dirpath,
                        time_indices=time_indices,
                        input_channel_idx=[channel_names.index(channel_name)],
                        output_channel_idx=[channel_names.index(channel_name)],
                        num_processes=int(slurm_args["slurm_cpus_per_task"]),
                        **copy_n_paste_kwargs,
                    )

                jobs.append(job)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("stabilize")
@input_position_dirpaths()
@output_dirpath()
@config_filepaths()
@sbatch_filepath()
@local()
def stabilize_cli(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepaths: list[str],
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Stabilize a timelapse dataset by applying spatial transformations.

    This function stabilizes a timelapse dataset based on precomputed transformations or
    configuration settings. It supports both local processing and SLURM-based distributed
    processing and outputs a Zarr dataset with stabilized channels.

    Parameters:
    - input_position_dirpaths (List[str]): List of file paths to the input OME-Zarr datasets for each position.
    - output_dirpath (str): Directory path to save the stabilized output dataset.
    - config_filepaths (list[str]): Paths to the YAML configuration files containing transformation settings.
    - sbatch_filepath (str, optional): Path to a SLURM sbatch file to override default SLURM settings. Defaults to None.
    - local (bool, optional): If True, runs the stabilization process locally instead of submitting to SLURM. Defaults to False.

    Returns:
    - None: Writes the stabilized dataset to the specified output directory.

    Notes:
    - The function applies stabilization based on affine transformations specified in the configuration file.
    - Stabilization can estimate both YX and Z drifts and handles multi-channel data.
    - Input and output datasets must follow the OME-Zarr format.

    Example:
    >> biahub stabilize-timelapse
        -i ./timelapse.zarr/0/0/0               # Input timelapse dataset
        -o ./stabilized_timelapse.zarr          # Output directory for stabilized data
        -c ./file_w_matrices.yml                # Configuration file with transformation matrices
        -v                                      # Verbose mode for detailed logs
        --local                                 # Run locally instead of submitting to SLURM

    """
    stabilize(
        input_position_dirpaths=input_position_dirpaths,
        output_dirpath=output_dirpath,
        config_filepaths=config_filepaths,
        sbatch_filepath=sbatch_filepath,
        local=local,
    )


if __name__ == "__main__":
    stabilize_cli()
