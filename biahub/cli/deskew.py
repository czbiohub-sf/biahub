from pathlib import Path
from typing import List

import click
import numpy as np
import submitit
import torch

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import process_single_position

from biahub.analysis.AnalysisSettings import DeskewSettings
from biahub.analysis.deskew import deskew_data, get_deskewed_data_shape
from biahub.cli import utils
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import yaml_to_model

# Needed for multiprocessing with GPUs
# https://github.com/pytorch/pytorch/issues/40403#issuecomment-1422625325
torch.multiprocessing.set_start_method('spawn', force=True)


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
def deskew(
    input_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Deskew a single position across T and C axes using a configuration file
    generated by estimate_deskew.py

    >> biahub deskew \
        -i ./input.zarr/*/*/* \
        -c ./deskew_params.yml \
        -o ./output.zarr
    """

    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

    # Handle single position or wildcard filepath
    output_position_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)

    # Get the deskewing parameters
    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        settings = yaml_to_model(config_filepath, DeskewSettings)
        deskewed_shape, voxel_size = get_deskewed_data_shape(
            (Z, Y, X),
            settings.ls_angle_deg,
            settings.px_to_scan_ratio,
            settings.keep_overhang,
            settings.average_n_slices,
            settings.pixel_size_um,
        )

        # Create a zarr store output to mirror the input
        utils.create_empty_zarr(
            input_position_dirpaths,
            output_dirpath,
            output_zyx_shape=deskewed_shape,
            chunk_zyx_shape=None,
            voxel_size=voxel_size,
        )

    deskew_args = {
        'ls_angle_deg': settings.ls_angle_deg,
        'px_to_scan_ratio': settings.px_to_scan_ratio,
        'keep_overhang': settings.keep_overhang,
        'average_n_slices': settings.average_n_slices,
        'extra_metadata': {'deskew': settings.model_dump()},
    }

    # Estimate resources
    gb_per_element = 4 / 2**30  # bytes_per_float32 / bytes_per_gb
    num_cpus = np.min([T * C, 16])
    input_memory = num_cpus * Z * Y * X * gb_per_element
    gb_ram_request = np.ceil(np.max([1, input_memory])).astype(int)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "deskew",
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
    executor = submitit.AutoExecutor(folder="logs", cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    _czyx_deskew_data,
                    input_position_path,
                    output_position_path,
                    num_processes=slurm_args["slurm_cpus_per_task"],
                    **deskew_args,
                )
            )

    monitor_jobs(jobs, input_position_dirpaths)


# Adapt ZYX function to CZYX
# Needs to be a top-level function for multiprocessing pickling
def _czyx_deskew_data(data, **kwargs):
    return deskew_data(data[0], **kwargs)[None]


if __name__ == "__main__":
    deskew()
