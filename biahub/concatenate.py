import glob

from pathlib import Path

import click
import numpy as np
import submitit

from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position
from natsort import natsorted

from biahub.cli.parsing import (
    config_filepath,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import copy_n_paste, estimate_resources, get_output_paths, yaml_to_model
from biahub.settings import ConcatenateSettings


def get_path_slice_param(slice_param, path_index, total_paths):
    """
    Determine the slice parameter for a specific path.

    Args:
        slice_param: The slice parameter from settings (can be 'all', a single slice range, or per-path specifications)
        path_index: The index of the current path
        total_paths: The total number of paths

    Returns:
        The slice parameter for the current path
    """
    # Handle 'all' case
    if slice_param == "all":
        return "all"

    # Handle single slice range [start, end]
    if isinstance(slice_param, list):
        if len(slice_param) == 2 and all(isinstance(i, int) for i in slice_param):
            return slice_param
        else:
            return (
                slice_param[path_index] if path_index < len(slice_param) else slice_param[-1]
            )

    # Handle any other case
    return slice_param


def create_path_slicing_params(path_z_slice, path_y_slice, path_x_slice, dataset_shape):
    """
    Create slicing parameters for a specific path.

    Args:
        path_z_slice: The Z slice parameter for the path
        path_y_slice: The Y slice parameter for the path
        path_x_slice: The X slice parameter for the path
        dataset_shape: The shape of the dataset

    Returns:
        A list of slice objects [z_slice, y_slice, x_slice]
    """
    z_slice = get_slice(path_z_slice, dataset_shape[2])
    y_slice = get_slice(path_y_slice, dataset_shape[3])
    x_slice = get_slice(path_x_slice, dataset_shape[4])
    return [z_slice, y_slice, x_slice]


def get_channel_combiner_metadata(
    data_paths_list: list[str],
    processing_channel_names: list[str],
    slicing_params: list,
):
    """
    Get metadata for channel combination.

    Args:
        data_paths_list: List of data paths
        processing_channel_names: List of channel names to process
        slicing_params: List of slicing parameters [Z_slice, Y_slice, X_slice]

    Returns:
        Tuple of (all_data_paths, all_channel_names, input_channel_idx, output_channel_idx, all_slicing_params)
    """
    all_data_paths = []
    all_channel_names = []
    input_channel_idx = []
    output_channel_idx = []
    out_chan_idx_counter = 0
    all_slicing_params = []

    # Unpack slicing parameters
    z_slice_param, y_slice_param, x_slice_param = slicing_params

    # Expand the data paths
    expanded_paths = []
    for paths in data_paths_list:
        expanded_paths.append([Path(path) for path in natsorted(glob.glob(paths))])

    # Flatten the expanded paths
    all_data_paths = [path for paths in expanded_paths for path in paths]

    # For each original path, determine the appropriate slice specifications
    for i, (paths, per_datapath_channels) in enumerate(
        zip(expanded_paths, processing_channel_names)
    ):
        # NOTE: taking first file as sample to get the channel names
        dataset = open_ome_zarr(paths[0])
        channel_names = dataset.channel_names

        # Determine the slice specifications for this path
        path_z_slice = get_path_slice_param(z_slice_param, i, len(data_paths_list))
        path_y_slice = get_path_slice_param(y_slice_param, i, len(data_paths_list))
        path_x_slice = get_path_slice_param(x_slice_param, i, len(data_paths_list))

        # Create slicing parameters for each path in this group
        for _ in range(len(paths)):
            slicing_params = create_path_slicing_params(
                path_z_slice, path_y_slice, path_x_slice, dataset.data.shape
            )
            all_slicing_params.append(slicing_params)

        # Parse channels
        output_channel_indices = []
        input_channel_indices = []

        if per_datapath_channels == "all":
            per_datapath_channels = channel_names

        for channel in per_datapath_channels:
            if channel in channel_names:
                # If the channel already exists in the list, we don't want to add it again
                if channel not in all_channel_names:
                    all_channel_names.append(channel)
                    output_channel_indices.append(out_chan_idx_counter)
                    out_chan_idx_counter += 1
                else:
                    click.echo(
                        f"Warning: Channel {channel} already exists. Skipping and using index from the first entry."
                    )
                    # Set the out_chan_idx_counter to the index of the channel in the all_channel_names list
                    out_chan_idx_counter = all_channel_names.index(channel)
                    output_channel_indices.append(out_chan_idx_counter)
                input_channel_indices.append(channel_names.index(channel))

        dataset.close()

        # Create a list of len paths
        input_channel_idx.extend([input_channel_indices for _ in paths])
        output_channel_idx.extend([output_channel_indices for _ in paths])

    # Validate that all slicing parameters produce the same output size
    if len(all_slicing_params) > 1:
        validate_slicing_params_zyx(all_slicing_params)

    click.echo(f"Channel names: {all_channel_names}")
    click.echo(f"Input channel indices: {input_channel_idx}")
    click.echo(f"Output channel indices: {output_channel_idx}")

    return (
        all_data_paths,
        all_channel_names,
        input_channel_idx,
        output_channel_idx,
        all_slicing_params,
    )


def get_slice(slice_param, max_value: int):
    """
    Convert slice parameters to slice objects.

    Args:
        slice_param: Can be 'all' or a single slice range [start, end]
        max_value: Maximum value for the dimension

    Returns:
        A slice object
    """
    # Handle 'all' case
    if slice_param == "all":
        return slice(0, max_value)

    # Handle single slice range [start, end]
    if (
        isinstance(slice_param, list)
        and len(slice_param) == 2
        and all(isinstance(i, int) for i in slice_param)
    ):
        return slice(*slice_param)

    raise ValueError(f"Invalid slice parameter: {slice_param}")


def validate_slicing_params_zyx(slicing_params_zyx_list: list[list[slice, slice, slice]]):
    """
    Validate that all slicing parameters are the same for a given dimension
    """
    first_slice_size = calculate_cropped_size(slicing_params_zyx_list[0])
    for i, slice_obj in enumerate(slicing_params_zyx_list[1:], 1):
        slice_size = calculate_cropped_size(slice_obj)
        if slice_size != first_slice_size:
            raise ValueError(
                f"Inconsistent slice sizes detected. Path 0 has size {first_slice_size}, "
                f"but path {i} has size {slice_size}. All paths must have the same slice size."
            )


def calculate_cropped_size(
    slice_params_zyx: list[slice, slice, slice]
) -> tuple[int, int, int]:
    """
    Calculate the size of a dimension after cropping.

    Args:
        slice_params_zyx: A list of slice parameters for the Z, Y, and X dimensions

    Returns:
        A tuple of the size of the dimension after cropping for the Z, Y, and X dimensions
    """
    # Calculate the size of each dimension by taking the absolute difference between stop and start
    z_size = abs(slice_params_zyx[0].stop - slice_params_zyx[0].start)
    y_size = abs(slice_params_zyx[1].stop - slice_params_zyx[1].start)
    x_size = abs(slice_params_zyx[2].stop - slice_params_zyx[2].start)

    cropped_shape_zyx = (z_size, y_size, x_size)
    click.echo(f"Output ZYX shape after cropping: {cropped_shape_zyx}")

    return cropped_shape_zyx


def concatenate(
    settings: ConcatenateSettings,
    output_dirpath: Path,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Concatenate datasets (with optional cropping)

    >> biahub concatenate -c ./concat.yml -o ./output_concat.zarr -j 8
    """
    slurm_out_path = output_dirpath.parent / "slurm_output"

    slicing_params = [
        settings.Z_slice,
        settings.Y_slice,
        settings.X_slice,
    ]
    (
        all_data_paths,
        all_channel_names,
        input_channel_idx_list,
        output_channel_idx_list,
        all_slicing_params,
    ) = get_channel_combiner_metadata(
        settings.concat_data_paths, settings.channel_names, slicing_params
    )
    output_position_paths_list = get_output_paths(
        all_data_paths,
        output_dirpath,
        ensure_unique_positions=settings.ensure_unique_positions,
    )

    all_shapes = []
    all_dtypes = []
    all_voxel_sizes = []
    for path in all_data_paths:
        with open_ome_zarr(path) as dataset:
            all_shapes.append(dataset.data.shape)
            all_dtypes.append(dataset.data.dtype)
            all_voxel_sizes.append(dataset.scale[-3:])

    # Only check for shape compatibility when using 'all' for slicing
    if (
        settings.Z_slice == "all"
        and settings.Y_slice == "all"
        and settings.X_slice == "all"
        and not all([shape[-3:] == all_shapes[0][-3:] for shape in all_shapes])
    ):
        raise ValueError(
            "Datasets have different shapes. All ZYX shapes must match to concatenate when using 'all' for slicing."
        )

    # Check the voxel sizes
    if not all([voxel_size == all_voxel_sizes[0] for voxel_size in all_voxel_sizes]):
        click.echo(
            "Warning: Datasets have different voxel sizes. Taking the first voxel size."
        )

    T, C, Z, Y, X = all_shapes[0]
    output_voxel_size = all_voxel_sizes[0]

    if all([dtype == all_dtypes[0] for dtype in all_dtypes]):
        dtype = all_dtypes[0]
    else:
        click.echo("Warning: not all dtypes match. Casting data at float32.")
        dtype = np.float32

    # Logic to parse time indices
    if settings.time_indices == "all":
        if not all([shape[0] == T for shape in all_shapes]):
            click.echo(
                "Warning: Datasets have different number of time points. Taking the smallest number of time points."
            )
        T = min([shape[0] for shape in all_shapes])
        input_time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        input_time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        input_time_indices = [settings.time_indices]

    # If input shapes are different but slicing is specified, inform the user
    if not all([shape[-3:] == all_shapes[0][-3:] for shape in all_shapes]):
        click.echo(
            "Warning: Datasets have different shapes, but slicing parameters are specified. Will validate output shapes after cropping."
        )

    cropped_shape_zyx = calculate_cropped_size(all_slicing_params[0])

    # Ensure that the cropped shape is within the bounds of the original shape
    if cropped_shape_zyx[0] > Z or cropped_shape_zyx[1] > Y or cropped_shape_zyx[2] > X:
        raise ValueError("The cropped shape is larger than the original shape.")

    # TODO: make assertion for chunk size?
    if settings.chunks_czyx is not None:
        chunk_size = [1] + list(settings.chunks_czyx)
    else:
        chunk_size = settings.chunks_czyx

    # Logic for creation of zarr and metadata
    output_metadata = {
        "shape": (len(input_time_indices), len(all_channel_names)) + tuple(cropped_shape_zyx),
        "chunks": chunk_size,
        "scale": (1,) * 2 + tuple(output_voxel_size),
        "channel_names": all_channel_names,
        "dtype": dtype,
    }

    # Create the output zarr mirroring source_position_dirpaths
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in output_position_paths_list],
        **output_metadata,
    )

    # Estimate resources
    num_cpus, gb_ram_per_cpu = estimate_resources(shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16)
    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "concatenate",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
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
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting SLURM jobs...")
    jobs = []

    with executor.batch():
        for i, (
            input_position_path,
            output_position_path,
            input_channel_idx,
            output_channel_idx,
            zyx_slicing_params,
        ) in enumerate(
            zip(
                all_data_paths,
                output_position_paths_list,
                input_channel_idx_list,
                output_channel_idx_list,
                all_slicing_params,
            )
        ):
            # Create slicing parameters for this specific path
            copy_n_paste_kwargs = {"zyx_slicing_params": zyx_slicing_params}

            job = executor.submit(
                process_single_position,
                copy_n_paste,
                input_position_path=input_position_path,
                output_position_path=output_position_path,
                input_channel_indices=input_channel_idx,
                output_channel_indices=output_channel_idx,
                input_time_indices=input_time_indices,
                output_time_indices=list(range(len(input_time_indices))),
                num_processes=int(slurm_args["slurm_cpus_per_task"]),
                **copy_n_paste_kwargs,
            )
            jobs.append(job)

    # monitor_jobs(jobs, all_data_paths)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))


@click.command("concatenate")
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
def concatenate_cli(
    config_filepath: str,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Concatenate datasets (with optional cropping)

    >> biahub concatenate -c ./concat.yml -o ./output_concat.zarr -j 8
    """
    concatenate(
        settings=yaml_to_model(config_filepath, ConcatenateSettings),
        output_dirpath=Path(output_dirpath),
        sbatch_filepath=sbatch_filepath,
        local=local,
    )


if __name__ == "__main__":
    concatenate_cli()
