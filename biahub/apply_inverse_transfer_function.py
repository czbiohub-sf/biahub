import itertools
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Final

import click
import numpy as np
import submitit
import torch
import torch.multiprocessing as mp
from iohub import open_ome_zarr

from waveorder.cli import apply_inverse_models, jobs_mgmt
from waveorder.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
)
from waveorder.cli.parsing import transfer_function_dirpath
from waveorder.cli.apply_inverse_transfer_function import apply_inverse_transfer_function_single_position
from waveorder.settings import ReconstructionSettings
from biahub.cli.utils import yaml_to_model, create_empty_hcs_zarr

@click.command()
@input_position_dirpaths()
@transfer_function_dirpath()
@config_filepath()
@output_dirpath()
def apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    num_processes,
) -> None:
    """
    Apply an inverse transfer function to a dataset using a configuration file.

    Applies a transfer function to all positions in the list `input-position-dirpaths`,
    so all positions must have the same TCZYX shape.

    Appends channels to ./output.zarr, so multiple reconstructions can fill a single store.

    See /examples for example configuration files.

    >> waveorder apply-inv-tf -i ./input.zarr/*/*/* -t ./transfer-function.zarr -c /examples/birefringence.yml -o ./output.zarr
    """
    output_metadata = get_reconstruction_output_metadata(
        input_position_dirpaths[0], config_filepath
    )
    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        **output_metadata,
    )
    # Initialize torch num of threads and interoeration operations
    if num_processes > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # Estimate resources
    with open_ome_zarr(input_position_dirpaths[0]) as input_dataset:
        T, C, Z, Y, X = input_dataset["0"].shape

    settings = yaml_to_model(config_filepath, ReconstructionSettings)
    gb_ram_request = 0
    gb_per_element = 4 / 2**30  # bytes_per_float32 / bytes_per_gb
    voxel_resource_multiplier = 4
    fourier_resource_multiplier = 32
    input_memory = Z * Y * X * gb_per_element

    if settings.birefringence is not None:
        gb_ram_request += input_memory * voxel_resource_multiplier
    if settings.phase is not None:
        gb_ram_request += input_memory * fourier_resource_multiplier
    if settings.fluorescence is not None:
        gb_ram_request += input_memory * fourier_resource_multiplier

    gb_ram_request = np.ceil(
        np.max([1, ram_multiplier * gb_ram_request])
    ).astype(int)
    cpu_request = np.min([32, num_processes])
    num_jobs = len(input_position_dirpaths)

    # Prepare and submit jobs
    click.echo(
        f"Preparing {num_jobs} job{'s, each with' if num_jobs > 1 else ' with'} "
        f"{cpu_request} CPU{'s' if cpu_request > 1 else ''} and "
        f"{gb_ram_request} GB of memory per CPU."
    )

    name_without_ext = os.path.splitext(Path(output_dirpath).name)[0]
    executor_folder = os.path.join(
        Path(output_dirpath).parent.absolute(), name_without_ext + "_logs"
    )
    executor = submitit.AutoExecutor(folder=Path(executor_folder))

    executor.update_parameters(
        slurm_array_parallelism=np.min([50, num_jobs]),
        slurm_mem_per_cpu=f"{gb_ram_request}G",
        slurm_cpus_per_task=cpu_request,
        slurm_time=60,
        slurm_partition="cpu",
        timeout_min=jobs_mgmt.JOBS_TIMEOUT,
        # more slurm_*** resource parameters here
    )

    jobs = []
    with executor.batch():
        for input_position_dirpath in input_position_dirpaths:
            job = executor.submit(
                apply_inverse_transfer_function_single_position,
                input_position_dirpath,
                transfer_function_dirpath,
                config_filepath,
                output_dirpath / Path(*input_position_dirpath.parts[-3:]),
                num_processes,
                output_metadata["channel_names"],
            )
            jobs.append(job)
    
    click.echo(
        f"{num_jobs} job{'s' if num_jobs > 1 else ''} submitted {'locally' if executor.cluster == 'local' else 'via ' + executor.cluster}."
    )
    

    monitor_jobs(jobs, input_position_dirpaths, doPrint)


