from pathlib import Path

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.analysis.AnalysisSettings import ProcessingSettings
from biahub.analysis.imgproc import process_czyx
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import get_output_paths, yaml_to_model


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
@click.option(
    "--num-processes",
    "-j",
    default=1,
    help="Number of parallel processes. Default is 1.",
    required=False,
    type=int,
)
def process_w_imports(
    input_position_dirpaths,
    config_filepath,
    output_dirpath,
    sbatch_filepath,
    local,
    num_processes,
):
    """
    Process data with functions specified in the config file.

    Example usage:
    biahub process_w_imports -i ./timelapse.zarr/0/0/0 -c ./process_params.yml -o ./processed_timelapse.zarr -j 1
    """
    # Convert to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)
    slurm_out_path = output_dirpath.parent / "slurm_output"
    # Handle single position or wildcard filepath
    output_position_paths = get_output_paths(input_position_dirpaths, output_dirpath)

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape
        channel_names = dataset.channel_names
        scale_dataset = dataset.scale

        settings = yaml_to_model(config_filepath, ProcessingSettings)

        if settings.processing_functions is not None:
            for proc in settings.processing_functions:
                # Replace the channel name with the channel index
                if proc.channel is not None:
                    proc.channel = channel_names.index(proc.channel)
                else:
                    raise ValueError("Channel must be specified for preprocessing functions")
        else:
            raise ValueError("Processing functions must be specified")

        # FIXME: temporary for binning functions
        if any(
            proc.function.__name__ == "binning_czyx" for proc in settings.processing_functions
        ):
            click.echo("Binning output shape is hard")
            # Get the binning factor from the first binning function found
            for proc in settings.processing_functions:
                if proc.function.__name__ == "binning_czyx":
                    binning_factor = proc.kwargs.get("binning_factor_zyx", (1, 4, 4))
                    click.echo(f"Binning factor: {binning_factor}")
                    break

            # Calculate new dimensions after binning
            new_Z = Z // binning_factor[0]
            new_Y = Y // binning_factor[1]
            new_X = X // binning_factor[2]

            output_shape = (T, C, new_Z, new_Y, new_X)
            new_scale = [
                scale_dataset[0],  # T
                scale_dataset[1],  # C
                scale_dataset[2] * binning_factor[0],  # Z
                scale_dataset[3] * binning_factor[1],  # Y
                scale_dataset[4] * binning_factor[2],  # X
            ]
        else:
            # Calculate output shape based on processing functions
            # For now, we'll assume the shape stays the same
            # TODO: Add logic to calculate output shape based on processing functions

            output_shape = (T, C, Z, Y, X)
            new_scale = scale_dataset

    # TODO: should the channels be the same as the input channels?
    output_metadata = {
        "shape": output_shape,
        "chunks": None,
        "scale": new_scale,
        "channel_names": channel_names,
        "dtype": np.float32,
    }

    # Create the output zarr mirroring input_position_dirpaths
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        **output_metadata,
    )

    process_args = {
        "processing_functions": settings.processing_functions,
    }

    # Estimate Resources
    gb_per_element = 4 / 2**30  # bytes_per_float32 / bytes_per_gb
    num_cpus = np.min([T * C, 16])
    input_memory = num_cpus * Z * Y * X * gb_per_element
    gb_ram_request = np.ceil(np.max([1, input_memory])).astype(int)
    slurm_time = np.ceil(np.max([60, T * 0.75])).astype(int)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "process_data",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": slurm_time,
        "slurm_partition": "cpu",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    slurm_out_path = output_dirpath.parent / "slurm_output"
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    # TODO: perhaps these should be exposed as parameters in config
    jobs = []
    with executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    process_czyx,
                    input_position_path=input_position_path,
                    output_position_path=output_position_path,
                    input_time_indices=list(range(T)),
                    input_channel_indices=[list(range(C))],  # Process all channels
                    output_channel_indices=[list(range(C))],
                    num_processes=np.max([1, num_cpus - 3]),
                    **process_args,
                )
            )

    monitor_jobs(jobs, input_position_dirpaths)


if __name__ == "__main__":
    process_w_imports()
