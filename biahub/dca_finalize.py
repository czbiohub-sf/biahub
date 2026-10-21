"""Write DCA metadata onto every FOV of an assembled plate.

``biahub dca-finalize`` is metadata-only. It edits each position's ``zarr.json`` attributes
in place and writes no pixel data, so it takes no ``-o``, ``--cluster`` or ``--sbatch``.

For each FOV it builds one metadata row (the well's sample-sheet row, the store facts and
the derived columns), then resolves and validates it against the base config plus the
plate's confirmed file. Normalization statistics come from the plate QC table. The run is
two-pass: if any FOV has an error, nothing is written and every error is listed.
"""

import tomllib

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import pandas as pd

from dca_helpers.metadata import (
    DcaValidationError,
    load_dca_config,
    merge_dca_attributes,
    resolve_dca,
    validate_dca,
    write_normalization_statistics,
)
from iohub.ngff import open_ome_zarr

from biahub.cli.parsing import config_filepath, input_position_dirpaths, position_key
from biahub.dca_metadata import AirtableClient, metadata_rows
from biahub.dca_stats import (
    STATS_METADATA,
    derived_columns,
    load_qc_statistics,
    position_statistics,
    store_facts,
    time_increment_issue,
)
from biahub.settings import DcaFinalizeSettings

PROVENANCE_KEY = "biahub-dca-finalize"


class DcaFinalizeError(Exception):
    """Raised when any FOV cannot be finalized; nothing has been written.

    Parameters
    ----------
    issues
        One message per problem, each prefixed with the position (or plate) path.
    confirmed_paths
        The confirmed file of each plate that had an issue, where missing values are filled.
    """

    def __init__(self, issues: list[str], confirmed_paths: list[Path] | None = None):
        self.issues = issues
        self.confirmed_paths = confirmed_paths or []
        super().__init__("\n".join(issues))


@dataclass(frozen=True)
class _PlateSources:
    block: dict[str, Any] | None
    rows: dict[str, dict[str, str]] | None
    qc: pd.DataFrame | None
    derived: dict[str, str]


def _group_by_plate(position_dirpaths: list[Path]) -> dict[Path, list[Path]]:
    plates: dict[Path, list[Path]] = defaultdict(list)
    for dirpath in position_dirpaths:
        plates[Path(dirpath).parents[2]].append(Path(dirpath))
    return dict(plates)


def _load_block(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        try:
            tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"{path}: {error}") from error
    return load_dca_config(*paths)


def _load_plate_sources(
    plate: Path, settings: DcaFinalizeSettings, client: AirtableClient | None
) -> tuple[_PlateSources, list[str]]:
    issues = []
    confirmed = settings.confirmed_path(plate)
    try:
        block = _load_block(
            [settings.base_config, *([confirmed] if confirmed.exists() else [])]
        )
    except ValueError as error:
        issues.append(f"{plate}: {error}")
        block = None
    try:
        rows = metadata_rows(plate, settings, client=client)
    except (ValueError, FileNotFoundError) as error:
        issues.append(f"{plate}: {error}")
        rows = None
    try:
        qc = load_qc_statistics(plate, settings.qc_stage)
    except ValueError as error:
        issues.append(f"{plate}: {error}")
        qc = None
    return _PlateSources(block, rows, qc, derived_columns(plate)), issues


def _dca_block(payload: dict[str, Any], stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    merge_dca_attributes(
        attrs,
        {k: v for k, v in payload.items() if k != "channels"},
        payload.get("channels"),
    )
    write_normalization_statistics(attrs, stats, metadata=STATS_METADATA)
    return attrs["dca"]


def _check_position(
    dirpath: Path, plate: Path, sources: _PlateSources
) -> tuple[dict[str, Any] | None, list[str]]:
    where = str(dirpath)
    key = position_key(dirpath)
    well = key.rsplit("/", 1)[0]
    with open_ome_zarr(dirpath, mode="r") as position:
        channel_names = list(position.channel_names)
        facts = store_facts(plate, key, position)

    issues = []
    payload = stats = None
    table_row = None if sources.rows is None else sources.rows.get(well)
    if sources.rows is not None and table_row is None:
        issues.append(f"{where}: no metadata row for well {well}")
    if sources.block is not None and table_row is not None:
        row = table_row | facts | sources.derived
        try:
            resolved = resolve_dca(sources.block, row, channel_keys=channel_names, where=where)
            payload = validate_dca(resolved, where=where)
        except DcaValidationError as error:
            issues.extend(f"{where}: {issue}" for issue in error.issues)
        except ValueError as error:
            issues.append(f"{where}: {error}")
        try:
            increment = time_increment_issue(table_row, facts)
        except ValueError as error:
            increment = f"acquisition.time_increment_s: {error}"
        if increment is not None:
            issues.append(f"{where}: {increment}")
    if sources.qc is not None:
        try:
            stats = position_statistics(sources.qc, key, channel_names)
        except ValueError as error:
            issues.append(f"{where}: {str(error).removeprefix(f'{key}: ')}")

    if issues or payload is None or stats is None:
        return None, issues
    try:
        return _dca_block(payload, stats), []
    except ValueError as error:
        return None, [f"{where}: {error}"]


def dca_finalize(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    airtable_client: AirtableClient | None = None,
) -> None:
    """Resolve, validate and write ``attributes.dca`` for every FOV.

    Parameters
    ----------
    input_position_dirpaths
        Positions of one or more assembled plates (``plate.zarr/A/1/0``).
    config_filepath
        YAML of ``DcaFinalizeSettings``; relative paths resolve against its directory.
    airtable_client
        Overrides the default AirTable client when the sample sheet comes from AirTable.

    Raises
    ------
    DcaFinalizeError
        If any FOV has an unresolved or invalid field, a missing table row, missing
        statistics or an inconsistent time increment. No position is written.
    """
    settings = DcaFinalizeSettings.from_yaml(config_filepath)
    plates = _group_by_plate(input_position_dirpaths)

    issues: list[str] = []
    failing_plates: list[Path] = []
    blocks: dict[Path, dict[str, Any]] = {}
    for plate, dirpaths in plates.items():
        sources, plate_issues = _load_plate_sources(plate, settings, airtable_client)
        for dirpath in dirpaths:
            block, position_issues = _check_position(dirpath, plate, sources)
            plate_issues.extend(position_issues)
            if block is not None:
                blocks[dirpath] = block
        if plate_issues:
            issues.extend(plate_issues)
            failing_plates.append(plate)
    if issues:
        confirmed = dict.fromkeys(settings.confirmed_path(plate) for plate in failing_plates)
        raise DcaFinalizeError(issues, list(confirmed))

    provenance = settings.model_dump(mode="json")
    for dirpath, block in blocks.items():
        with open_ome_zarr(dirpath, mode="r+") as position:
            position.zattrs.update({"dca": block, PROVENANCE_KEY: provenance})
    click.echo(f"Wrote attributes.dca to {len(blocks)} positions")


@click.command("dca-finalize")
@input_position_dirpaths()
@config_filepath()
def dca_finalize_cli(input_position_dirpaths: list[Path], config_filepath: Path):
    """Write DCA metadata onto every FOV of an assembled plate (metadata-only).

    \b
    Finalize a whole plate:
    >>> biahub dca-finalize -i ./plate.zarr/*/*/* -c ./dca_finalize.yml
    """  # noqa: D301
    try:
        dca_finalize(
            input_position_dirpaths=input_position_dirpaths,
            config_filepath=config_filepath,
        )
    except DcaFinalizeError as error:
        lines = [*error.issues]
        lines.extend(f"fill the missing values in {path}" for path in error.confirmed_paths)
        raise click.ClickException("\n".join(lines)) from error


if __name__ == "__main__":
    dca_finalize_cli()
