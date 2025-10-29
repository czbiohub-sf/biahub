from pathlib import Path
from typing import List, Optional

import click
import submitit

from iohub import open_ome_zarr

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.settings import ConvertSettings


def convert_position(
    input_position_path: Path,
    output_store_path: Path,
) -> None:
    """
    Convert a single position from Zarr V2 OME-NGFF v0.4 to V3 OME-NGFF v0.5.

    Parameters
    ----------
    input_position_path : Path
        Path to input position in V2 store
    output_store_path : Path
        Path to output V3 store

    Notes
    -----
    - iohub automatically handles compression (Blosc with zstd)
    - Metadata (transforms, axes, channels) is automatically preserved
    - Version is always "0.5"
    """
    row, col, fov = input_position_path.parts[-3:]
    click.echo(f"Converting position: {row}/{col}/{fov}")

    # Read input V2 data
    with open_ome_zarr(input_position_path, mode="r") as old_position:
        old_image = old_position["0"]
        click.echo(f"\tShape: {old_image.shape}")

        # Open output V3 store and copy data to pre-created position
        with open_ome_zarr(output_store_path, mode="r+") as new_dataset:
            new_position = new_dataset[f"{row}/{col}/{fov}"]
            new_image = new_position["0"]

            new_image[:] = old_image[:]

    click.echo(f"✓ Completed: {row}/{col}/{fov}")


@click.command("convert")
@input_position_dirpaths()
@output_dirpath()
@click.option(
    "--config-filepath",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    default=None,
    help="Optional YAML config for chunks/shards. If not provided, uses input chunks and no sharding.",
)
@sbatch_filepath()
@local()
@monitor()
def convert_cli(
    input_position_dirpaths: List[Path],
    output_dirpath: Path,
    config_filepath: Optional[Path] = None,
    sbatch_filepath: Optional[str] = None,
    local: bool = False,
    monitor: bool = False,
) -> None:
    """
    Convert Zarr V2 OME-NGFF v0.4 to V3 OME-NGFF v0.5.

    This command converts HCS plate stores from Zarr V2 format (OME-NGFF v0.4)
    to Zarr V3 format (OME-NGFF v0.5). Conversion is parallelized per position
    for efficient processing on SLURM clusters.

    Features:
    - Automatic compression with Blosc+zstd
    - Preserves all metadata (transforms, axes, channels)
    - Optional custom chunking and Zarr V3 sharding
    - Parallelizable across positions

    Examples:
        # Basic conversion with defaults
        biahub convert -i ./input_v2.zarr/*/*/* -o ./output_v3.zarr

        # With custom chunks and sharding
        biahub convert -i ./input_v2.zarr/*/*/* -o ./output_v3.zarr -c config.yaml

        # Local execution (no SLURM)
        biahub convert -i ./input_v2.zarr/*/*/* -o ./output_v3.zarr --local

        # With SLURM config and monitoring
        biahub convert -i ./input_v2.zarr/*/*/* -o ./output_v3.zarr \\
            -c config.yaml -sb slurm_config.sh -m
    """
    click.echo("Starting Zarr V2 to V3 Conversion")

    # Load settings
    if config_filepath:
        click.echo(f"Loading config from: {config_filepath}")
        settings = yaml_to_model(config_filepath, ConvertSettings)
    else:
        click.echo("Using default settings (input chunks, no sharding)")
        settings = ConvertSettings()

    # Pre-create all positions with their individual sizes
    click.echo(f"\nPre-creating {len(input_position_dirpaths)} positions in output store...")
    click.echo(f"Output store: {output_dirpath}")

    first_shape = None
    for idx, input_path in enumerate(input_position_dirpaths):
        with open_ome_zarr(input_path, mode="r") as old_position:
            old_image = old_position["0"]
            shape = old_image.shape
            chunks = settings.chunks or old_image.chunks
            row, col, fov = input_path.parts[-3:]

            if idx == 0:
                first_shape = shape
                # Create plate on first position
                new_plate = open_ome_zarr(
                    output_dirpath,
                    layout="hcs",
                    mode="w",
                    channel_names=old_position.channel_names,
                    axes=old_position.axes,
                    version="0.5",
                )
            else:
                new_plate = open_ome_zarr(output_dirpath, mode="r+")

            with new_plate:
                # Create position
                new_position = new_plate.create_position(row, col, fov)

                # Create empty array with exact shape from V2
                new_position.create_zeros(
                    name="0",
                    shape=shape,
                    dtype=old_image.dtype,
                    chunks=chunks,
                    shards_ratio=settings.shards_ratio,
                    transform=old_position.metadata.multiscales[0]
                    .datasets[0]
                    .coordinate_transformations,
                )

            if (idx + 1) % 10 == 0:
                click.echo(f"  Created {idx + 1}/{len(input_position_dirpaths)} positions...")

    click.echo(f"✓ All {len(input_position_dirpaths)} positions pre-created")

    # Prepare job arguments list (one job per position)
    click.echo(f"\nPreparing {len(input_position_dirpaths)} conversion jobs...")
    job_args_list = [
        (input_path, output_dirpath, settings) for input_path in input_position_dirpaths
    ]

    # Estimate resources using first position shape
    num_cpus, gb_ram = estimate_resources(shape=first_shape, ram_multiplier=3, max_num_cpus=16)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "convert",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 60,
        "slurm_partition": "cpu",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        click.echo(f"Loading SLURM config from: {sbatch_filepath}")
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    cluster = "local" if local else "slurm"
    click.echo(f"Execution mode: {cluster}")
    click.echo(f"Resources: {num_cpus} CPUs, {gb_ram}GB RAM per task")

    # Prepare and submit jobs
    slurm_out_path = Path(output_dirpath).parent / "slurm_output"
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    click.echo("Submitting jobs...")
    with submitit.helpers.clean_env(), executor.batch():
        for job_args in job_args_list:
            jobs.append(
                executor.submit(
                    convert_position,
                    *job_args,
                )
            )

    job_ids = [job.job_id for job in jobs]

    # Log job IDs
    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    click.echo(f"Submitted {len(jobs)} jobs")
    click.echo(f"Job IDs logged to: {log_path}")

    if monitor:
        click.echo("Monitoring jobs...")
        monitor_jobs(jobs, input_position_dirpaths)

    click.echo("Conversion complete!")


if __name__ == "__main__":
    convert_cli()
