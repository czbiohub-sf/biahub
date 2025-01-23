from pathlib import Path
from typing import List

import ants
import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as R

from biahub.analysis.AnalysisSettings import StabilizationSettings
from biahub.analysis.register import convert_transform_to_ants
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    copy_n_paste_czyx,
    create_empty_hcs_zarr,
    process_single_position_v2,
    yaml_to_model,
)


def apply_stabilization_transform(
    zyx_data: np.ndarray,
    list_of_shifts: list[np.ndarray],
    t_idx: int,
    output_shape: tuple[int, int, int] = None,
):
    """Apply stabilization to a single zyx"""
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


@click.command()
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
def stabilize(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Stabilize the timelapse input based on single position and channel.

    This function applies stabilization to the input data. It can estimate both yx and z drifts.
    The level of verbosity can be controlled with the stabilization_verbose flag.
    The size of the crop in xy can be specified with the crop-size-xy option.

    Example usage:
    biahub stabilize-timelapse -i ./timelapse.zarr/0/0/0 -o ./stabilized_timelapse.zarr -c ./file_w_matrices.yml -v

    """
    assert config_filepath.suffix == ".yml", "Config file must be a yaml file"

    # Convert to Path objects
    config_filepath = Path(config_filepath)
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"
    # Load the config file
    settings = yaml_to_model(config_filepath, StabilizationSettings)

    combined_mats = settings.affine_transform_zyx_list
    combined_mats = np.array(combined_mats)
    stabilization_channels = settings.stabilization_channels

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape
        channel_names = dataset.channel_names
        for channel in stabilization_channels:
            if channel not in channel_names:
                raise ValueError(f"Channel <{channel}> not found in the input data")

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

    # Logic to parse time indices
    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    # transform_t0_sy = np.abs(settings.affine_transform_zyx_list[0][2][1]).round(3)

    # # Calculate scale
    # new_scale = [
    #     scale_dataset[0],
    #     scale_dataset[1],
    #     scale_dataset[2],
    #     scale_dataset[3] * transform_t0_sy,
    #     scale_dataset[4] * transform_t0_sy,
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
    gb_per_element = 4 / 2**30  # bytes_per_float32 / bytes_per_gb
    num_cpus = np.min([T * C, 16])
    input_memory = num_cpus * Z * Y * X * gb_per_element
    gb_ram_request = np.ceil(np.max([1, input_memory])).astype(int)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "stabilize",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 60,
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


if __name__ == "__main__":
    stabilize()
