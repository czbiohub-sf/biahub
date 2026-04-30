import logging
import re
import shutil

from pathlib import Path

import click
import yaml

from biahub.cli.utils import plan_single_position, read_plate_metadata

logger = logging.getLogger(__name__)


def _resolve_qc_config(config_path: Path) -> dict:
    """Load a QC YAML config, resolving Hydra-style ``defaults`` from the same dir."""
    raw = yaml.safe_load(config_path.read_text())
    defaults = raw.pop("defaults", [])
    merged: dict = {}
    for entry in defaults:
        if isinstance(entry, str) and entry != "_self_":
            base_path = config_path.parent / f"{entry}.yaml"
            if base_path.exists():
                base = yaml.safe_load(base_path.read_text()) or {}
                merged.update(base)
    merged.update(raw)
    return merged


@click.group("qc")
def nf_qc_cli():
    """QC adapter commands for Nextflow pipelines."""


@nf_qc_cli.command("init-chunks")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
def init_chunks(input_zarr: str) -> None:
    """Emit (position, start, end, chunk_id) CSV rows for Nextflow QC fan-out.

    Produces a headerless CSV with one row per (position, time-chunk) pair.
    Nextflow consumes this directly via ``splitCsv``.
    """
    position_keys, _, _, _ = read_plate_metadata(input_zarr)
    first_position = Path(input_zarr) / "/".join(position_keys[0])
    items = plan_single_position(first_position, first_position)

    chunks: list[tuple[int, int, str]] = []
    seen: set[tuple[int, ...]] = set()
    for item in items:
        key = tuple(item.input_time_indices)
        if key in seen:
            continue
        seen.add(key)
        start = item.input_time_indices[0]
        end = item.input_time_indices[-1] + 1
        chunks.append((start, end, f"t{start}-{end - 1}"))

    for pos_key in position_keys:
        position = "/".join(pos_key)
        for start, end, chunk_id in chunks:
            click.echo(f"{position},{start},{end},{chunk_id}")


@nf_qc_cli.command("init-qc-fanout")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option(
    "--chunk-size",
    type=int,
    default=10,
    show_default=True,
    help="Number of timepoints per QC chunk. Merges shard-aligned batches.",
)
def init_qc_fanout(input_zarr: str, config: str, chunk_size: int) -> None:
    """Emit fan-out rows for QC metric groups, respecting scope.

    Reads metric_groups from the QC config and zarr metadata to produce
    a headerless CSV consumed by Nextflow via ``splitCsv``.

    Position-scoped groups emit one row per (position, time-chunk)::

        position,group_name,start,end,chunk_id

    Temporal-scoped groups emit one row per position (no chunking)::

        position,group_name,,,
    """
    cfg = _resolve_qc_config(Path(config))
    groups = cfg.get("metric_groups", {})
    if not groups:
        return

    position_keys, _, _, _ = read_plate_metadata(input_zarr)
    positions = ["/".join(pk) for pk in position_keys]

    has_position_scope = any(
        g.get("scope", "position") == "position" for g in groups.values()
    )
    chunks: list[tuple[int, int, str]] = []
    if has_position_scope:
        first_position = Path(input_zarr) / "/".join(position_keys[0])
        items = plan_single_position(first_position, first_position)
        seen: set[tuple[int, ...]] = set()
        shard_chunks: list[tuple[int, int]] = []
        for item in items:
            key = tuple(item.input_time_indices)
            if key in seen:
                continue
            seen.add(key)
            start = item.input_time_indices[0]
            end = item.input_time_indices[-1] + 1
            shard_chunks.append((start, end))
        shard_chunks.sort()
        for i in range(0, len(shard_chunks), chunk_size):
            batch = shard_chunks[i : i + chunk_size]
            merged_start = batch[0][0]
            merged_end = batch[-1][1]
            chunks.append(
                (merged_start, merged_end, f"t{merged_start}-{merged_end - 1}")
            )

    for group_name, group_def in groups.items():
        scope = group_def.get("scope", "position")
        for pos in positions:
            if scope == "position":
                for start, end, chunk_id in chunks:
                    click.echo(f"{pos},{group_name},{start},{end},{chunk_id}")
            else:
                click.echo(f"{pos},{group_name},,,")


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
