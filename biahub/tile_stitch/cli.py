"""Command-line entry point for distributed tile reconstruction."""

import logging

from pathlib import Path

import click
import yaml

from biahub.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath

logger = logging.getLogger(__name__)


def _resolve_time_indices(spec: int | list[int] | str, n_t: int) -> list[int]:
    """Normalize configured time indices.

    Parameters
    ----------
    spec : int, list[int], or str
        One index, explicit indices, or ``"all"``.
    n_t : int
        Number of available timepoints.

    Returns
    -------
    list[int]
        Explicit timepoint indices.
    """
    if spec == "all":
        return list(range(n_t))
    if isinstance(spec, int):
        return [spec]
    return list(spec)


@click.command("tile-stitch")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def tile_stitch_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
) -> None:
    """Reconstruct and stitch configured channels over a Monarch actor mesh.

    Parameters
    ----------
    input_position_dirpaths : list[pathlib.Path]
        Input OME-Zarr positions.
    config_filepath : pathlib.Path
        Tile-stitch YAML configuration.
    output_dirpath : pathlib.Path
        Output OME-Zarr path.

    Raises
    ------
    click.UsageError
        If input positions have incompatible shapes or layout.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    from biahub.settings import TileStitchReconSettings
    from biahub.tile_stitch.orchestration import (
        PreparationError,
        create_stitch_output,
        execute_stitch_run,
        prepare_stitch_run,
    )

    config = TileStitchReconSettings.model_validate(
        yaml.safe_load(config_filepath.read_text())
    )
    try:
        prepared = prepare_stitch_run(
            input_position_dirpaths,
            config,
            output_dirpath,
        )
    except PreparationError as exc:
        raise click.UsageError(str(exc)) from exc

    create_stitch_output(prepared)
    logger.info(
        "output created: %s | units=%d | full_shape=%s | chunks=%s",
        prepared.final_output,
        len(prepared.work_units),
        prepared.full_shape,
        prepared.chunk_shape,
    )
    result = execute_stitch_run(prepared)
    logger.info(
        "tile-stitch done: %d units, total=%.1fs",
        len(prepared.work_units),
        result["total_s"],
    )
    click.echo(f"tile-stitch complete: {prepared.final_output}")
