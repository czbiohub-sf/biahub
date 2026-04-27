import logging
import re
import shutil

from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group("qc")
def nf_qc_cli():
    """QC adapter commands for Nextflow pipelines."""


@nf_qc_cli.command("consolidate")
@click.option(
    "--step-zarr",
    "-s",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Step zarr path to copy QC parquets from (repeat for each step).",
)
@click.option(
    "--assembly-zarr",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="Assembly zarr to copy QC parquets into.",
)
def consolidate(step_zarr: tuple[str, ...], assembly_zarr: str) -> None:
    """Copy per-position QC parquets from step zarrs into the assembly zarr.

    Finds all ``tables/qc/stage*`` parquets in each step zarr and copies them
    to the matching position directory in the assembly zarr, preserving the
    relative path structure.
    """
    assembly = Path(assembly_zarr)
    copied = 0
    for zarr_str in step_zarr:
        zarr = Path(zarr_str)
        for src in zarr.rglob("tables/qc/stage*"):
            if not src.is_file():
                continue
            rel = src.relative_to(zarr)
            dst = assembly / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    logger.info("Copied %d QC parquet file(s) into %s", copied, assembly_zarr)


@nf_qc_cli.command("log-summary")
@click.argument("summary_files", nargs=-1, type=click.Path(exists=True))
def log_summary(summary_files: tuple[str, ...]) -> None:
    """Print a consolidated QC gate summary from per-stage summary files.

    Each summary file should contain a ``QC_SUMMARY`` line captured from
    ``imaging-qc run --mode gate_only`` stderr.  Files containing only
    ``no_summary`` are silently skipped.
    """
    lines: list[str] = []
    for path_str in summary_files:
        text = Path(path_str).read_text().strip()
        if text and text != "no_summary":
            lines.append(text)

    click.echo("")
    click.echo("========================================")
    click.echo("  QC Gate Summary")
    click.echo("========================================")

    if not lines:
        click.echo("  No gate summary data available.")
    else:
        for line in lines:
            click.echo(f"  {line}")

    click.echo("")

    has_failures = any(re.search(r"fail=[1-9]", line) for line in lines)
    if has_failures:
        click.echo("  WARNING: Gate failures detected in one or more QC stages.")
        click.echo("  Review the QC report for details.")
    else:
        click.echo("  All QC gates passed.")

    click.echo("========================================")
    click.echo("")
