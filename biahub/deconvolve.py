from pathlib import Path
from typing import List

import click
import numpy as np
import submitit
import torch

from iohub import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from iohub.ngff.utils import create_empty_plate, process_single_position
from waveorder.models.isotropic_fluorescent_thick_3d import apply_inverse_transfer_function

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    _str_to_path,
    config_filepath,
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, get_output_paths, yaml_to_model
from biahub.settings import DeconvolveSettings


def compute_tranfser_function(
    psf_zyx_data: np.ndarray,
    output_zyx_shape: tuple,
) -> np.ndarray:
    zyx_padding = np.array(output_zyx_shape) - np.array(psf_zyx_data.shape)
    pad_width = [(x // 2, x // 2) if x % 2 == 0 else (x // 2, x // 2 + 1) for x in zyx_padding]
    padded_psf_data = np.pad(
        psf_zyx_data, pad_width=pad_width, mode="constant", constant_values=0
    )

    transfer_function = torch.abs(torch.fft.fftn(torch.tensor(padded_psf_data)))
    transfer_function /= torch.max(transfer_function)

    return transfer_function.numpy()


def deconvolve(
    czyx_raw_data: np.ndarray,
    transfer_function: torch.Tensor = None,
    transfer_function_store_path: str = None,
    regularization_strength: float = 1e-3,
) -> np.ndarray:
    if transfer_function is None:
        with open_ome_zarr(transfer_function_store_path, layout='fov', mode='r') as ds:
            transfer_function = torch.tensor(ds.data[0, 0])

    output = []
    for zyx_raw_data in czyx_raw_data:
        zyx_decon_data = apply_inverse_transfer_function(
            torch.tensor(zyx_raw_data),
            transfer_function,
            z_padding=0,
            regularization_strength=regularization_strength,
        )
        output.append(zyx_decon_data.numpy())

    return np.stack(output)


@click.command("deconvolve")
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
@monitor()
def deconvolve_cli(
    input_position_dirpaths: List[str],
    psf_dirpath: str,
    config_filepath: str,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True,
):
    """
    Deconvolve across T and C axes using a PSF and a configuration file

    >> biahub deconvolve -i ./input.zarr/*/*/* -p ./psf.zarr -c ./deconvolve_params.yml -o ./output.zarr
    """
    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)
    slurm_out_path = output_dirpath.parent / "slurm_output"
    transfer_function_store_path = output_dirpath.parent / "transfer_function.zarr"
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
        scale=scale,
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

    transfer_function = compute_tranfser_function(psf_data, output_zyx_shape=shape[-3:])
    with open_ome_zarr(
        transfer_function_store_path, layout='fov', mode='w-', channel_names=['PSF']
    ) as psf_output_dataset:
        psf_output_dataset.create_image(
            '0',
            transfer_function[None, None],
            chunks=(1, 1, 256) + shape[-2:],
            transform=[TransformationMeta(type='scale', scale=psf_dataset.scale)],
        )

    # Estimate resources
    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=10, max_num_cpus=16
    )
    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "deconvolve",
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
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo('Submitting SLURM jobs...')
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths
        ):
            job = executor.submit(
                process_single_position,
                deconvolve,
                str(input_position_path),
                str(output_position_path),
                num_processes=int(slurm_args["slurm_cpus_per_task"]),
                transfer_function_store_path=str(transfer_function_store_path),
                regularization_strength=float(settings.regularization_strength),
            )
            jobs.append(job)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


if __name__ == "__main__":
    deconvolve_cli()
