from pathlib import Path
from typing import Any

import pytest

from biahub.dca_metadata import (
    AIRTABLE_FIELDS,
    TABLE_COLUMNS,
    PyairtableClient,
    airtable_rows,
    blank_row,
    metadata_rows,
    read_metadata_table,
)
from biahub.settings import AirtableSettings, DcaFinalizeSettings

ROW_C2 = ["ds", "C/2", "A549", "TOMM20", "mitochondria", "ZIKV", "5", "30", "24"]
ROW_B1 = ["ds", "B/1", "A549", "SEC61B", "endoplasmic_reticulum", "mock", "0", "30", "24"]


def write_table(path: Path, rows: list[list[str]], sep: str, columns=TABLE_COLUMNS) -> Path:
    lines = [sep.join(columns)] + [sep.join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def pyairtable_record(well: str, **overrides: Any) -> dict[str, Any]:
    fields = {
        AIRTABLE_FIELDS["dataset"]: "ds",
        AIRTABLE_FIELDS["well"]: well,
        AIRTABLE_FIELDS["cell_line"]: ["A549"],
        AIRTABLE_FIELDS["marker"]: "TOMM20",
        AIRTABLE_FIELDS["organelle"]: "mitochondria",
        AIRTABLE_FIELDS["perturbation"]: "ZIKV",
        AIRTABLE_FIELDS["moi"]: 5,
        AIRTABLE_FIELDS["time_interval_min"]: 30,
        AIRTABLE_FIELDS["hours_post_perturbation"]: 24.5,
    }
    fields.update(overrides)
    return {
        "id": "recXXXXXXXXXXXXXX",
        "createdTime": "2024-01-01T00:00:00.000Z",
        "fields": fields,
    }


class FakeClient:
    def __init__(self, records: list[dict[str, Any]]):
        self._records = records
        self.calls: list[str] = []

    def records(self, dataset: str) -> list[dict[str, Any]]:
        self.calls.append(dataset)
        return self._records


def test_read_csv_and_tsv(tmp_path):
    rows = [ROW_C2, ROW_B1]
    csv = read_metadata_table(write_table(tmp_path / "t.csv", rows, ","))
    tsv = read_metadata_table(write_table(tmp_path / "t.tsv", rows, "\t"))
    assert csv == tsv
    assert set(csv) == {"C/2", "B/1"}
    assert csv["C/2"]["marker"] == "TOMM20"
    assert csv["B/1"]["moi"] == "0"


def test_duplicate_well_raises(tmp_path):
    path = write_table(tmp_path / "t.csv", [ROW_C2, ROW_C2], ",")
    with pytest.raises(ValueError, match="C/2"):
        read_metadata_table(path)


def test_missing_column_raises(tmp_path):
    columns = tuple(c for c in TABLE_COLUMNS if c != "marker")
    rows = [[v for c, v in zip(TABLE_COLUMNS, ROW_C2, strict=True) if c != "marker"]]
    path = write_table(tmp_path / "t.csv", rows, ",", columns)
    with pytest.raises(ValueError, match="marker"):
        read_metadata_table(path)


def test_blank_row_has_every_column():
    assert blank_row() == {c: "" for c in TABLE_COLUMNS}


def test_airtable_fields_cover_known_columns():
    assert set(AIRTABLE_FIELDS) <= set(TABLE_COLUMNS)


def test_airtable_rows_map_pyairtable_record_shape():
    client = FakeClient([pyairtable_record("C/2")])
    rows = airtable_rows(client, "ds")
    assert client.calls == ["ds"]
    assert rows["C/2"] == {
        "dataset": "ds",
        "well": "C/2",
        "cell_line": "A549",
        "marker": "TOMM20",
        "organelle": "mitochondria",
        "perturbation": "ZIKV",
        "moi": "5",
        "time_interval_min": "30",
        "hours_post_perturbation": "24.5",
    }


def test_airtable_missing_field_gives_empty_string():
    record = pyairtable_record("C/2")
    del record["fields"][AIRTABLE_FIELDS["moi"]]
    rows = airtable_rows(FakeClient([record]), "ds")
    assert rows["C/2"]["moi"] == ""


def test_airtable_multi_select_cell_line_joined():
    record = pyairtable_record("C/2", **{AIRTABLE_FIELDS["cell_line"]: ["A549", "HEK293T"]})
    rows = airtable_rows(FakeClient([record]), "ds")
    assert rows["C/2"]["cell_line"] == "A549,HEK293T"


def test_airtable_duplicate_well_raises():
    client = FakeClient([pyairtable_record("C/2"), pyairtable_record("C/2")])
    with pytest.raises(ValueError, match="C/2"):
        airtable_rows(client, "ds")


def test_airtable_record_without_well_raises():
    record = pyairtable_record("C/2")
    del record["fields"][AIRTABLE_FIELDS["well"]]
    with pytest.raises(ValueError, match="recXXXXXXXXXXXXXX"):
        airtable_rows(FakeClient([record]), "ds")


def test_table_next_to_plate_wins_over_airtable(tmp_path):
    plate = tmp_path / "plate.zarr"
    write_table(tmp_path / "plate.dca-metadata.csv", [ROW_C2], ",")
    client = FakeClient([pyairtable_record("B/1")])
    settings = DcaFinalizeSettings(
        base_config="b.toml", airtable=AirtableSettings(base_id="b", table="t")
    )
    rows = metadata_rows(plate, settings, client)
    assert set(rows) == {"C/2"}
    assert client.calls == []


def test_airtable_used_when_no_table(tmp_path):
    plate = tmp_path / "plate.zarr"
    client = FakeClient([pyairtable_record("C/2"), pyairtable_record("B/1")])
    settings = DcaFinalizeSettings(
        base_config="b.toml", airtable=AirtableSettings(base_id="b", table="t")
    )
    rows = metadata_rows(plate, settings, client)
    assert set(rows) == {"C/2", "B/1"}
    assert client.calls == ["plate"]


def test_no_source_raises(tmp_path):
    plate = tmp_path / "plate.zarr"
    settings = DcaFinalizeSettings(base_config="b.toml")
    expected = tmp_path / "plate.dca-metadata.csv"
    with pytest.raises(FileNotFoundError, match=str(expected)):
        metadata_rows(plate, settings, None)


def test_default_client_missing_token_names_variable(monkeypatch):
    monkeypatch.delenv("BIAHUB_TEST_AIRTABLE_TOKEN", raising=False)
    settings = AirtableSettings(base_id="b", table="t", token_env="BIAHUB_TEST_AIRTABLE_TOKEN")
    with pytest.raises(RuntimeError, match="BIAHUB_TEST_AIRTABLE_TOKEN"):
        PyairtableClient(settings)
