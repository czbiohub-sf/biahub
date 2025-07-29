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
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.resolve_function import resolve_function
from biahub.cli.utils import estimate_resources, get_output_paths, yaml_to_model
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


CUSTOM_FUNCTIONS = {
    "biahub.process_data.binning_czyx": binning_czyx,
}


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
        func = resolve_function(
            proc.function, custom_functions=CUSTOM_FUNCTIONS
        )  # Resolve function string to callable
        kwargs = proc.kwargs
        if len(proc.input_channels) == 1:
            c_idx = proc.input_channels[0]
        else:
            raise ValueError("Only one input channel is supported for now")

        click.echo(f'Processing with {func.__name__} with kwargs {kwargs} to channel {c_idx}')
        czyx_data = func(czyx_data, **kwargs)

    return czyx_data


def process_with_config(
    input_position_dirpaths: Sequence[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: Path | None = None,
    local: bool = False,
    monitor: bool = True,
) -> None:
    """
    Process data with functions specified in the config file.

    Parameters
    ----------
    input_position_dirpaths : Sequence[Path]
        Input position directory paths
    config_filepath : Path
        Path to the configuration file
    output_dirpath : Path
        Output directory path
    sbatch_filepath : Path | None, optional
        Path to the SLURM batch file, by default None
    local : bool, optional
        Whether to run locally or submit to SLURM, by default False
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
        if not settings.processing_functions:
            raise ValueError("Processing functions must be specified")
        for proc in settings.processing_functions:
            # Replace the channel name with the channel index
            if proc.input_channels is not None and len(proc.input_channels) == 1:
                proc.input_channels[0] = channel_names.index(proc.input_channels[0])
            else:
                raise ValueError("Channel must be specified for preprocessing functions")
            # Resolve function and check if it's callable
            try:
                resolved_func = resolve_function(
                    proc.function, custom_functions=CUSTOM_FUNCTIONS
                )
                if not callable(resolved_func):
                    raise ValueError(f"Function {proc.function} is not callable")
            except ValueError as e:
                raise ValueError(f"Function {proc.function} could not be resolved: {e}")
    else:
        raise ValueError("Processing functions must be specified")

    # FIXME: temporary for binning functions
    if any(
        proc.function == "biahub.process_data.binning_czyx"
        for proc in settings.processing_functions
    ):
        # Get the binning factor from the first binning function found
        for proc in settings.processing_functions:
            if proc.function == "biahub.process_data.binning_czyx":
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
    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=output_shape, dtype=np.float32, ram_multiplier=4, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "process_data",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 60,
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

    # TODO: perhaps T/C processing indices should be exposed as params in config
    jobs = []
    with submitit.helpers.clean_env():
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
                        num_processes=slurm_args["slurm_cpus_per_task"],
                        **process_args,
                    )
                )

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("process-with-config")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
@monitor()
def process_with_config_cli(
    input_position_dirpaths: Sequence[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: Path | None = None,
    local: bool = False,
    monitor: bool = True,
) -> None:
    """
    Process data with functions specified in the config file.

    Example usage:
    biahub process-with-config -i ./timelapse.zarr/0/0/0 -c ./process_params.yml -o ./processed_timelapse.zarr
    """
    process_with_config(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        local=local,
        monitor=monitor,
    )


if __name__ == "__main__":
    process_with_config_cli()
