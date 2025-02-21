import glob

from pathlib import Path

import click
import numpy as np
import submitit

from iohub import open_ome_zarr
from natsort import natsorted

from biahub.analysis.AnalysisSettings import ConcatenateSettings
from biahub.cli.parsing import (
    config_filepath,
    local,
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


def get_channel_combiner_metadata(
    data_paths_list: list[str], processing_channel_names: list[str]
):
    all_data_paths = []
    all_channel_names = []
    input_channel_idx = []
    output_channel_idx = []
    out_chan_idx_counter = 0
    for paths, per_datapath_channels in zip(data_paths_list, processing_channel_names):
        # Parse the data paths
        parsed_paths = [Path(path) for path in natsorted(glob.glob(paths))]
        all_data_paths.extend(parsed_paths)

        output_channel_indices = []
        input_channel_indices = []

        # NOTE: taking first file as sample to get the channel names
        dataset = open_ome_zarr(parsed_paths[0])
        channel_names = dataset.channel_names

        # Parse channels
        if per_datapath_channels == 'all':
            for i in range(len(channel_names)):
                input_channel_indices.append(i)
            for i in range(out_chan_idx_counter, out_chan_idx_counter + len(channel_names)):
                output_channel_indices.append(i)
            out_chan_idx_counter += len(channel_names)
            all_channel_names.extend(channel_names)

        elif isinstance(per_datapath_channels, list):
            for channel in per_datapath_channels:
                if channel in channel_names:
                    # If the channel already exists in the list, we don't want to add it again
                    if channel not in all_channel_names:
                        all_channel_names.append(channel)
                        output_channel_indices.append(out_chan_idx_counter)
                        out_chan_idx_counter += 1
                    else:
                        # Set the out_chan_idx_counter to the index of the channel in the all_channel_names list
                        out_chan_idx_counter = all_channel_names.index(channel)
                        output_channel_indices.append(out_chan_idx_counter)
                    input_channel_indices.append(channel_names.index(channel))

        dataset.close()

        # Create a list of len paths
        input_channel_idx.extend([input_channel_indices for _ in parsed_paths])
        output_channel_idx.extend([output_channel_indices for _ in parsed_paths])

    click.echo(f"Channel names: {all_channel_names}")
    click.echo(f"Input channel indices: {input_channel_idx}")
    click.echo(f"Output channel indices: {output_channel_idx}")

    return all_data_paths, all_channel_names, input_channel_idx, output_channel_idx


def get_slice(slice_param, max_value):
    return slice(0, max_value) if slice_param == 'all' else slice(*slice_param)


@click.command()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
def concatenate(
    config_filepath: str, output_dirpath: str, sbatch_filepath: str = None, local: bool = False
):
    """
    Concatenate datasets (with optional cropping)

    >> biahub concatenate -c ./concat.yml -o ./output_concat.zarr -j 8
    """
    # Convert to Path objects
    config_filepath = Path(config_filepath)
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    settings = yaml_to_model(config_filepath, ConcatenateSettings)

    (
        all_data_paths,
        all_channel_names,
        input_channel_idx_list,
        output_channel_idx_list,
    ) = get_channel_combiner_metadata(settings.concat_data_paths, settings.channel_names)

    all_shapes = []
    all_dtypes = []
    all_voxel_sizes = []
    for path in all_data_paths:
        with open_ome_zarr(path) as dataset:
            all_shapes.append(dataset.data.shape)
            all_dtypes.append(dataset.data.dtype)
            all_voxel_sizes.append(dataset.scale[-3:])

    if not all([shape == all_shapes[0] for shape in all_shapes]):
        raise ValueError("All shapes must match")
    if not all([voxel_size == all_voxel_sizes[0] for voxel_size in all_voxel_sizes]):
        raise ValueError("All voxel sizes must match")
    T, C, Z, Y, X = all_shapes[0]
    output_voxel_size = all_voxel_sizes[0]

    if all([dtype == all_dtypes[0] for dtype in all_dtypes]):
        dtype = all_dtypes[0]
    else:
        click.echo("Warning: not all dtypes match. Casting data at float32.")
        dtype = np.float32

    # Logic to parse time indices
    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    # Crop the data
    Z_slice = get_slice(settings.Z_slice, Z)
    Y_slice = get_slice(settings.Y_slice, Y)
    X_slice = get_slice(settings.X_slice, X)

    cropped_shape_zyx = (
        abs(Z_slice.stop - Z_slice.start),
        abs(Y_slice.stop - Y_slice.start),
        abs(X_slice.stop - X_slice.start),
    )

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
        "shape": (len(time_indices), len(all_channel_names)) + tuple(cropped_shape_zyx),
        "chunks": chunk_size,
        "scale": (1,) * 2 + tuple(output_voxel_size),
        "channel_names": all_channel_names,
        "dtype": dtype,
    }

    # Create the output zarr mirroring source_position_dirpaths
    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in all_data_paths],
        **output_metadata,
    )

    copy_n_paste_kwargs = {"czyx_slicing_params": ([Z_slice, Y_slice, X_slice])}

    # Estimate resources
    num_cpus, gb_ram_per_cpu = estimate_resources(shape=[T, C, Z, Y, X], ram_multiplier=16)
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

    click.echo('Submitting SLURM jobs...')
    jobs = []

    with executor.batch():
        for input_position_path, input_channel_idx, output_channel_idx in zip(
            all_data_paths, input_channel_idx_list, output_channel_idx_list
        ):
            job = executor.submit(
                process_single_position_v2,
                copy_n_paste_czyx,
                input_data_path=input_position_path,
                output_path=output_dirpath,
                time_indices=time_indices,
                input_channel_idx=input_channel_idx,
                output_channel_idx=output_channel_idx,
                num_processes=int(slurm_args["slurm_cpus_per_task"]),
                **copy_n_paste_kwargs,
            )
            jobs.append(job)

    # monitor_jobs(jobs, all_data_paths)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))


if __name__ == "__main__":
    concatenate()
