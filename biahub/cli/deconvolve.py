from pathlib import Path
from typing import List

import click
import numpy as np
import submitit

from iohub import open_ome_zarr
from iohub.ngff.utils import process_single_position, create_empty_plate

from biahub.analysis.AnalysisSettings import DeconvolveSettings
from biahub.cli.parsing import (
    _str_to_path,
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.analysis.deconvolve import compute_tranfser_function, deconvolve_data
from biahub.cli.monitor import monitor_jobs
from biahub.cli.utils import yaml_to_model, get_output_paths


@click.command()
@input_position_dirpaths()
@click.option(
    "--psf-dirpath",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    callback=_str_to_path,
    help="Path to psf.zarr",
)
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
def deconvolve(
    input_position_dirpaths: List[str],
    psf_dirpath: str,
    config_filepath: str,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Deconvolve across T and C axes using a PSF and a configuration file

    >> biahub deconvolve -i ./input.zarr/*/*/* -p ./psf.zarr -c ./deconvolve_params.yml -o ./output.zarr
    """
    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)
    slurm_out_path = output_dirpath.parent / "slurm_output"
    output_position_paths = get_output_paths(input_position_dirpaths, output_dirpath)

    # Read config file
    settings = yaml_to_model(config_filepath, DeconvolveSettings)

    # Get input dataset metadata
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        channel_names = input_dataset.channel_names
        shape = input_dataset.data.shape
        scale = input_dataset.scale
        T, C, Z, Y, X = shape

    # Create output zarr store
    click.echo("Creating empty output zarr...")
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        channel_names=channel_names,
        shape=shape,
        scale=scale
    )

    # Compute transfer function
    click.echo("Computing transfer function...")
    with open_ome_zarr(Path(psf_dirpath, '0/0/0'), mode="r") as psf_dataset:
        if scale[-3:] != psf_dataset.scale[-3:]:
            click.echo(
                f"Warning: PSF scale: {scale[-3:]} does not match data scale: {scale[-3:]}. "
                "Consider resampling the PSF."
            )
        psf_data = psf_dataset.data[0, 0]

    zyx_padding = np.array(shape[-3:]) - np.array(psf_data.shape)
    pad_width = [(x // 2, x // 2) if x % 2 == 0 else (x // 2, x // 2 + 1) for x in zyx_padding]
    padded_average_psf = np.pad(
        psf_data, pad_width=pad_width, mode="constant", constant_values=0
    )
    transfer_function = compute_tranfser_function(padded_average_psf)

    # Estimate resources
    gb_per_element = 4 / 2**30  # bytes_per_float32 / bytes_per_gb
    num_cpus = np.min([T * C, 16])
    input_memory = num_cpus * Z * Y * X * gb_per_element
    gb_ram_request = np.ceil(np.max([1, input_memory])).astype(int)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "deconvolve",
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
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo('Submitting SLURM jobs...')
    jobs = []
    with executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths
        ):
            job = executor.submit(
                process_single_position,
                deconvolve_data,
                input_position_path,
                output_position_path,
                num_processes=slurm_args["slurm_cpus_per_task"],
                transfer_function=transfer_function,
                regularization_strength=settings.regularization_strength,
            )
            jobs.append(job)

    monitor_jobs(jobs, input_position_dirpaths)

if __name__ == "__main__":
    deconvolve()
