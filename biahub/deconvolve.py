from pathlib import Path

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
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    estimate_resources,
    get_output_paths,
    get_submitit_cluster,
    resolve_ome_zarr_version,
    yaml_to_model,
)
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
        with open_ome_zarr(transfer_function_store_path, layout="fov", mode="r") as ds:
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


def _load_settings(config_filepath: Path) -> DeconvolveSettings:
    return yaml_to_model(config_filepath, DeconvolveSettings)


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    settings: DeconvolveSettings,
) -> tuple[tuple[int, ...], list[str], tuple]:
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        channel_names = input_dataset.channel_names
        shape = input_dataset.data.shape
        scale = input_dataset.scale

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        channel_names=channel_names,
        shape=shape,
        scale=scale,
        version=resolve_ome_zarr_version(
            input_position_dirpaths[0], settings.output_ome_zarr_version
        ),
        metadata_sources=input_plate,
    )

    return shape, channel_names, scale


def _compute_and_save_transfer_function(
    psf_dirpath: Path,
    output_dirpath: Path,
    output_zyx_shape: tuple,
) -> Path:
    transfer_function_store_path = output_dirpath.parent / "transfer_function.zarr"

    with open_ome_zarr(Path(psf_dirpath, "0/0/0"), mode="r") as psf_dataset:
        psf_scale = psf_dataset.scale
        psf_data = psf_dataset.data[0, 0]

    transfer_function = compute_tranfser_function(psf_data, output_zyx_shape=output_zyx_shape)
    with open_ome_zarr(
        transfer_function_store_path, layout="fov", mode="w-", channel_names=["PSF"]
    ) as psf_output_dataset:
        psf_output_dataset.create_image(
            "0",
            transfer_function[None, None],
            chunks=(1, 1, 256) + output_zyx_shape[-2:],
            transform=[TransformationMeta(type="scale", scale=psf_scale)],
        )

    return transfer_function_store_path


@click.command("deconvolve")
@input_position_dirpaths()
@click.option(
    "--psf-dirpath",
    "-p",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to psf.zarr (required for --init)",
)
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
def deconvolve_cli(
    input_position_dirpaths: list[str],
    psf_dirpath: str,
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
):
    r"""Deconvolve across T and C axes using a PSF and a configuration file.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub deconvolve -i ./input.zarr/*/*/* -p ./psf.zarr -c ./deconvolve.yml -o ./output.zarr

    \b
    Initialize the output plate only (computes transfer function):
    >>> biahub deconvolve --init -i ./input.zarr/*/*/* -p ./psf.zarr -c ./deconvolve.yml -o ./output.zarr

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub deconvolve --cluster debug -i ./input.zarr/A/1/0 -c ./deconvolve.yml -o ./output.zarr

    """  # noqa: D301
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"
    transfer_function_store_path = output_dirpath.parent / "transfer_function.zarr"

    settings = _load_settings(config_filepath)

    if init_only:
        if psf_dirpath is None:
            raise click.UsageError("--psf-dirpath / -p is required when using --init")
        psf_dirpath = Path(psf_dirpath)

        shape, channel_names, scale = _init_output_plate(
            input_position_dirpaths, output_dirpath, settings
        )
        T, C, Z, Y, X = shape

        click.echo("Computing transfer function...")
        _compute_and_save_transfer_function(psf_dirpath, output_dirpath, shape[-3:])

        num_cpus, gb_ram = estimate_resources(
            shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
        )
        click.echo(f"RESOURCES:{num_cpus} {num_cpus * gb_ram}")
        click.echo(f"Initialized {output_dirpath} ({len(input_position_dirpaths)} positions)")
        return

    # Full run path: output plate and TF must already exist (created by --init)
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        shape = input_dataset.data.shape
        T, C, Z, Y, X = shape

    output_position_paths = get_output_paths(input_position_dirpaths, output_dirpath)

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
    )

    slurm_args = {
        "slurm_job_name": "deconvolve",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 60,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)

    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths, strict=True
        ):
            job = executor.submit(
                process_single_position,
                deconvolve,
                str(input_position_path),
                str(output_position_path),
                num_workers=int(slurm_args["slurm_cpus_per_task"]),
                transfer_function_store_path=str(transfer_function_store_path),
                regularization_strength=float(settings.regularization_strength),
            )
            jobs.append(job)

    job_ids = [job.job_id for job in jobs]

    slurm_out_path.mkdir(exist_ok=True)
    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # DebugExecutor is lazy: run each job in the foreground.
    if resolved_cluster == "debug":
        for job in jobs:
            job.wait()
        click.echo("Deconvolution complete")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


if __name__ == "__main__":
    deconvolve_cli()
