from pathlib import Path
from typing import List

import click
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import process_single_position, create_empty_plate

import numpy as np
from scipy.ndimage import map_coordinates
from biahub.cli import utils
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, yaml_to_model

def apply_poly_transform(data: np.ndarray, xcenter, ycenter, list_fact, order, mode):
    # fuction adapted from discorpy
    # https://github.com/DiamondLightSource/discorpy
    # data is expected to be CZYX ND array

    assert data.ndim == 4, 'Data must be a 4 dimensional CZYX array'

    (height, width) = data.shape[-2:]
    xu_list = np.arange(width) - xcenter
    yu_list = np.arange(height) - ycenter
    xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
    ru_mat = np.sqrt(xu_mat ** 2 + yu_mat ** 2)
    fact_mat = np.sum(np.asarray(
        [factor * ru_mat ** i for i, factor in enumerate(list_fact)]), axis=0)
    xd_mat = np.float32(np.clip(xcenter + fact_mat * xu_mat, 0, width - 1))
    yd_mat = np.float32(np.clip(ycenter + fact_mat * yu_mat, 0, height - 1))
    indices = np.reshape(yd_mat, (-1, 1)), np.reshape(xd_mat, (-1, 1))
    mat_flat = np.reshape(np.squeeze(data), (np.prod(data.shape[:-2]), height, width))
    corrected_mat = np.array([map_coordinates(slice, indices, order=order, mode=mode) for i, slice in enumerate(mat_flat)])
    return corrected_mat.reshape(data.shape)

@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@click.option(
    "--channel-name",
    required=True,
    type=str,
)
@sbatch_filepath()
@local()
def correct_distortion(
    input_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
    channel_name: str,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    """

    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    # Handle single position or wildcard filepath
    output_position_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)

    # Get the deskewing parameters
    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        scale = input_dataset.scale
        input_channel_names = input_dataset.channel_names

    # Create a zarr store output to mirror the input
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(p).parts[-3:] for p in input_position_dirpaths],
        channel_names=[channel_name],
        shape=(T, 1, Z, Y, X),  ## TODO: Fix hardcoded value
        chunks=None,
        scale=scale,
    )

    # Read distortion parameters
    with open(config_filepath, "r") as file:
        coeffs = [float(line.strip().rsplit(' ', -1)[-1]) for line in file]
        xcenter, ycenter, *list_fact = coeffs

    # Estimate resources
    num_cpus, gb_ram = estimate_resources(shape=(T, C, Z, Y, X), ram_multiplier=12)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "correct-distortion",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 60,
        "slurm_partition": "preempted",
    }

    func_args = {
        'xcenter': xcenter,
        'ycenter': ycenter,
        'list_fact': list_fact,
        'order': 1,
        'mode': 'constant'
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

    jobs = []
    with executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    apply_poly_transform,
                    input_position_path,
                    output_position_path,
                    input_channel_indices=[[input_channel_names.index(channel_name)]],
                    output_channel_indices=[[0]],  ## TODO: Fix hardcoded value
                    num_processes=slurm_args["slurm_cpus_per_task"],
                    **func_args,
                )
            )


if __name__ == "__main__":
    correct_distortion()
