"""Per-well infectomics metadata for ``dca-finalize``, from a sample sheet or AirTable."""

import os

from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from pyairtable import Api
from pyairtable.formulas import match

from biahub.settings import AirtableSettings, DcaFinalizeSettings

TABLE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "well",
    "cell_line",
    "marker",
    "organelle",
    "perturbation",
    "moi",
    "time_interval_min",
    "hours_post_perturbation",
)

AIRTABLE_FIELDS: dict[str, str] = {
    "dataset": "dataset",
    "well": "well_id",
    "cell_line": "cell_line",
    "marker": "marker",
    "organelle": "organelle",
    "perturbation": "perturbation",
    "moi": "moi",
    "time_interval_min": "time_interval_min",
    "hours_post_perturbation": "hours_post_perturbation",
}


class AirtableClient(Protocol):
    """Source of raw AirTable records (pyairtable ``RecordDict``) for one dataset."""

    def records(self, dataset: str) -> list[dict[str, Any]]: ...


class PyairtableClient:
    """Default client wrapping ``pyairtable``; reads the token from an env var."""

    def __init__(self, settings: AirtableSettings):
        token = os.environ.get(settings.token_env)
        if not token:
            raise RuntimeError(
                f"AirTable token not found: set the environment variable {settings.token_env}"
            )
        self._table = Api(token).table(settings.base_id, settings.table)
        self._dataset_field = AIRTABLE_FIELDS["dataset"]

    def records(self, dataset: str) -> list[dict[str, Any]]:
        return self._table.all(formula=match({self._dataset_field: dataset}))


def blank_row() -> dict[str, str]:
    """Return a row with every table column set to an empty string."""
    return {column: "" for column in TABLE_COLUMNS}


def _index_by_well(rows: list[dict[str, str]], origin: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        well = row["well"]
        if well in indexed:
            raise ValueError(f"Duplicate well {well} in {origin}")
        indexed[well] = row
    return indexed


def read_metadata_table(path: Path) -> dict[str, dict[str, str]]:
    """Read a CSV or TSV sample sheet into rows keyed by well (e.g. ``"C/2"``)."""
    path = Path(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    frame = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    missing = [column for column in TABLE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    rows = frame[list(TABLE_COLUMNS)].to_dict(orient="records")
    return _index_by_well(rows, str(path))


def _field_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return _field_to_str(value.get("name", ""))
    if isinstance(value, list):
        return ",".join(_field_to_str(item) for item in value)
    return str(value)


def _record_to_row(record: dict[str, Any]) -> dict[str, str]:
    fields = record.get("fields", {})
    row = blank_row()
    for column, field in AIRTABLE_FIELDS.items():
        row[column] = _field_to_str(fields.get(field))
    if not row["well"]:
        raise ValueError(
            f"AirTable record {record.get('id')} has no {AIRTABLE_FIELDS['well']}"
        )
    return row


def airtable_rows(client: AirtableClient, dataset: str) -> dict[str, dict[str, str]]:
    """Map AirTable records for ``dataset`` to rows keyed by well."""
    rows = [_record_to_row(record) for record in client.records(dataset)]
    return _index_by_well(rows, f"AirTable dataset {dataset}")


def metadata_rows(
    plate: Path, settings: DcaFinalizeSettings, client: AirtableClient | None = None
) -> dict[str, dict[str, str]]:
    """Return one metadata row per well, from a sample sheet if present, else AirTable.

    ``client`` overrides the default pyairtable client, which is built only when
    no sample sheet exists and ``settings.airtable`` is set.
    """
    plate = Path(plate)
    table = settings.metadata_table_path(plate)
    if table is not None:
        return read_metadata_table(table)
    if settings.airtable is not None:
        client = client or PyairtableClient(settings.airtable)
        return airtable_rows(client, plate.stem)
    expected = plate.parent / f"{plate.stem}.dca-metadata.csv"
    raise FileNotFoundError(
        f"No metadata source: expected {expected} (or .tsv), or set airtable in the settings"
    )
