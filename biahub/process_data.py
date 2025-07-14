from pathlib import Path
from typing import Literal, Sequence

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

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
from biahub.settings import ProcessingFunctions, ProcessingImportFuncSettings


def binning_czyx(
    czyx_data: np.ndarray,
    binning_factor_zyx: Sequence[int] = [1, 2, 2],
    mode: Literal['sum', 'mean'] = 'sum',
) -> np.ndarray:
    """
    Binning via summing or averaging pixels within bin windows

    Parameters
    ----------
    czyx_data : np.ndarray
        Input array to bin in CZYX format
    binning_factor_zyx : Sequence[int]
        Binning factor in each dimension (Z, Y, X). Can be list or tuple.
    mode : str
        'sum' for sum binning or 'mean' for mean binning

    Returns
    -------
    np.ndarray
        Binned array with shape (C, new_Z, new_Y, new_X) with same dtype as input.
        For sum mode, values are normalized to span [0, dtype.max] for integer types
        or [0, 65535] for float types.
        For mean mode, values are averaged within bins.
    """
    # Calculate new dimensions after binning
    C = czyx_data.shape[0]
    new_z = czyx_data.shape[1] // binning_factor_zyx[0]
    new_y = czyx_data.shape[2] // binning_factor_zyx[1]
    new_x = czyx_data.shape[3] // binning_factor_zyx[2]

    # Use float32 for intermediate calculations to avoid overflow
    output = np.zeros((C, new_z, new_y, new_x), dtype=np.float32)

    for c in range(C):
        # Reshape to group pixels that will be binned together
        reshaped = (
            czyx_data[c]
            .astype(np.float32)
            .reshape(
                new_z,
                binning_factor_zyx[0],
                new_y,
                binning_factor_zyx[1],
                new_x,
                binning_factor_zyx[2],
            )
        )

        if mode == 'sum':
            output[c] = reshaped.sum(axis=(1, 3, 5))
            # Normalize sum to [0, max_val] where max_val is dtype dependent
            if output[c].max() > 0:  # Avoid division by zero
                if np.issubdtype(czyx_data.dtype, np.integer):
                    max_val = np.iinfo(czyx_data.dtype).max
                else:
                    max_val = np.iinfo(np.uint16).max  # Normalize floats to uint16 range
                output[c] = (
                    (output[c] - output[c].min())
                    * max_val
                    / (output[c].max() - output[c].min())
                )
        elif mode == 'mean':
            output[c] = reshaped.mean(axis=(1, 3, 5))
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sum' or 'mean'.")

    # For mean mode and integer dtypes, scale to dtype range
    if mode == 'mean' and np.issubdtype(czyx_data.dtype, np.integer):
        dtype_info = np.iinfo(czyx_data.dtype)
        output = output * dtype_info.max / output.max()

    # Convert back to original dtype
    return output.astype(czyx_data.dtype)


def process_czyx(
    czyx_data: np.ndarray,
    processing_functions: list[ProcessingFunctions],
) -> np.ndarray:
    """
    Process a CZYX image using processing functions

    Parameters
    ----------
    czyx_data : np.ndarray
        A CZYX image to process
    processing_functions : list[ProcessingFunctions]
        A list of processing functions to apply with their configurations

    Returns
    -------
    np.ndarray
        A processed CZYX image
    """
    # Apply processing functions
    for proc in processing_functions:

        func = proc.function  # ImportString automatically resolves the function
        kwargs = proc.kwargs
        c_idx = proc.channel

        click.echo(f'Processing with {func.__name__} with kwargs {kwargs} to channel {c_idx}')
        # TODO should we ha
        czyx_data = func(czyx_data, **kwargs)

    return czyx_data


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
def process_w_imports_cli(
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
    biahub process-w-imports -i ./timelapse.zarr/0/0/0 -c ./process_params.yml -o ./processed_timelapse.zarr -j 1
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

        settings = yaml_to_model(config_filepath, ProcessingImportFuncSettings)

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
    process_w_imports_cli()
