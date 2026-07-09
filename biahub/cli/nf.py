import logging
import shutil

from pathlib import Path

import click

from iohub.ngff import open_ome_zarr

from biahub.cli.utils import echo_resources, estimate_resources

logger = logging.getLogger(__name__)


@click.group("nf")
def nf_cli():
    """Nextflow-oriented utility commands.

    Generic helpers shared across Nextflow pipelines. Step-specific init/run
    logic lives on each step's own CLI command (e.g. ``biahub deskew``).
    """


@nf_cli.command("list-positions")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
def list_positions(input_zarr: str):
    """List position keys in a plate zarr (one per line, for Nextflow fan-out)."""
    with open_ome_zarr(input_zarr, mode="r") as plate:
        for name, _ in plate.positions():
            click.echo(name)


@nf_cli.command("init-resources")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--ram-multiplier", "-r", required=True, type=float)
@click.option("--max-num-cpus", default=16, type=int)
@click.option("--time-minutes", default=60, type=int)
def init_resources(
    input_zarr: str, ram_multiplier: float, max_num_cpus: int, time_minutes: int
):
    """Estimate CPU/memory resources from input zarr shape (for Nextflow fan-out)."""
    with open_ome_zarr(input_zarr, mode="r") as plate:
        first_pos = next(plate.positions())[1]
        shape = first_pos.data.shape
    num_cpus, mem_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=ram_multiplier, max_num_cpus=max_num_cpus
    )
    echo_resources(num_cpus, num_cpus * mem_per_cpu, time_minutes)


@nf_cli.command("clean-temp")
@click.argument("temp_dir", type=click.Path())
def clean_temp(temp_dir: str):
    """Remove a temp directory if it exists (idempotent pre-retry cleanup)."""
    path = Path(temp_dir)
    if path.exists():
        shutil.rmtree(path)
        logger.info(f"Removed stale temp directory: {path}")
    else:
        logger.info(f"No temp directory to clean: {path}")


@nf_cli.command("clean-intermediates")
@click.option("--output-dir", "-o", required=True, type=click.Path())
@click.option("--dataset-name", "-d", required=True, type=str)
@click.option(
    "--intermediate-dir",
    "-i",
    multiple=True,
    required=True,
    help="Step directory name to clean (repeatable, e.g. -i 0-flatfield -i 1-deskew).",
)
def clean_intermediates(output_dir: str, dataset_name: str, intermediate_dir: tuple[str, ...]):
    """Delete intermediate zarrs after successful assembly."""
    for dirname in intermediate_dir:
        zarr_path = Path(output_dir) / dirname / f"{dataset_name}.zarr"
        if zarr_path.exists():
            shutil.rmtree(zarr_path)
            logger.info(f"Deleted intermediate zarr: {zarr_path}")
        else:
            logger.info(f"Intermediate zarr not found (skipping): {zarr_path}")
