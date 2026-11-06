import json
import logging

from pathlib import Path
from typing import Any

import pytest

from biahub.dca_finalize import dca_finalize
from biahub.dca_metadata import AIRTABLE_FIELDS
from tests.dca_plate import (
    RELEASE_DATE,
    WELLS,
    attributes,
    build_plate,
    confirmed_path,
    run,
    sheet_path,
    table_row,
    write_config,
    write_table,
)

UNSOURCED_EXPERIMENTER = (
    "[dca.study]\n"
    'experimenter = [{ first_name = "Ada", last_name = "Example", institution = "Example Lab" }]\n'
    f'release_date = "{RELEASE_DATE}"\n'
    "[dca.provenance.study.release_date]\n"
    'source_kind = "human"\n'
)


@pytest.fixture
def plate(tmp_path, create_custom_plate) -> Path:
    return build_plate(tmp_path, create_custom_plate)


@pytest.fixture
def config(tmp_path) -> Path:
    return write_config(tmp_path)


class SheetClient:
    """AirTable client serving each sample-sheet row as a well-level record."""

    def __init__(self, rows: list[dict[str, str]]):
        self._rows = rows

    def records(self, dataset: str) -> list[dict[str, Any]]:
        return [
            {"id": f"rec{i}", "fields": {AIRTABLE_FIELDS[k]: v for k, v in row.items()}}
            for i, row in enumerate(self._rows)
            if row["dataset"] == dataset
        ]


def test_provenance_has_no_local_paths_or_airtable_ids(plate, config, tmp_path):
    sheet_path(plate).unlink()
    config.write_text(
        "base_config: dca_base.toml\n"
        "airtable:\n  base_id: appPRIVATEBASE0001\n  table: tblPRIVATETABLE001\n"
    )
    client = SheetClient([table_row(well) for well in WELLS])
    dca_finalize([plate / well / "0" for well in WELLS], config, airtable_client=client)
    provenance = attributes(plate, "A/1")["biahub-dca-finalize"]
    assert provenance["metadata_source"] == {"kind": "airtable"}
    text = json.dumps(provenance)
    for private in (str(tmp_path), "appPRIVATEBASE0001", "tblPRIVATETABLE001"):
        assert private not in text


def test_shared_should_gap_logged_once_per_plate(plate, config, caplog):
    confirmed_path(plate).write_text(UNSOURCED_EXPERIMENTER)
    with caplog.at_level(logging.WARNING):
        result = run(plate, config)
    assert result.exit_code == 0, result.output
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    gaps = [m for m in messages if "'study.experimenter' has no provenance record" in m]
    assert len(gaps) == 1, messages
    assert gaps[0].startswith(f"{plate}: ")
    assert f"({len(WELLS)} of {len(WELLS)} positions)" in gaps[0]
    assert not [r for r in caplog.records if r.name.startswith("dca_helpers")]


def error_lines(result) -> list[str]:
    return result.output.strip().removeprefix("Error: ").splitlines()


def test_missing_base_config_is_a_plate_issue(plate, config):
    missing = config.parent / "missing.toml"
    config.write_text("base_config: missing.toml\n")
    result = run(plate, config)
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == 1
    assert error_lines(result)[0].startswith(f"{plate}: ")
    assert str(missing) in error_lines(result)[0]
    for well in WELLS:
        assert "dca" not in attributes(plate, well)


def test_airtable_client_error_does_not_abort_other_plates(
    plate, config, tmp_path, create_custom_plate, monkeypatch
):
    monkeypatch.delenv("BIAHUB_TEST_AIRTABLE_TOKEN", raising=False)
    other = build_plate(tmp_path, create_custom_plate, "2026_01_03_other_plate")
    sheet_path(plate).unlink()
    write_table(sheet_path(other), [table_row("A/1", dataset=other.stem)])
    config.write_text(
        "base_config: dca_base.toml\n"
        "airtable:\n  base_id: b\n  table: t\n  token_env: BIAHUB_TEST_AIRTABLE_TOKEN\n"
    )
    result = run(plate, config, other)
    assert isinstance(result.exception, SystemExit), result.exception
    lines = error_lines(result)
    assert any(
        line.startswith(f"{plate}: ") and "BIAHUB_TEST_AIRTABLE_TOKEN" in line
        for line in lines
    )
    assert f"{other / 'B' / '1' / '0'}: no metadata row for well B/1" in lines
