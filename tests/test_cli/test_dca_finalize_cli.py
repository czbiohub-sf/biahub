import hashlib
import json
import logging
import shutil

from pathlib import Path
from typing import Any

import pytest

from click.testing import CliRunner
from dca_helpers.validation.runner import validate
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli
from biahub.dca_finalize import dca_finalize
from biahub.dca_metadata import AIRTABLE_FIELDS
from tests.dca_plate import (
    BASE_CONFIG,
    CHANNELS,
    CONFIRMED,
    STEM,
    TIME_INCREMENT_S,
    WELLS,
    build_plate,
    confirmed_path,
    p1_value,
    sheet_path,
    table_row,
    write_qc,
)
from tests.dca_plate import write_table as write_sheet

OTHER_STEM = "2026_01_03_other_plate"


def write_table(plate: Path, rows: list[dict[str, str]]) -> None:
    write_sheet(sheet_path(plate), rows)


@pytest.fixture
def plate(tmp_path, create_custom_plate) -> Path:
    return build_plate(tmp_path, create_custom_plate)


@pytest.fixture
def config(tmp_path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    shutil.copy(BASE_CONFIG, config_dir / "dca_base.toml")
    path = config_dir / "dca_finalize.yml"
    path.write_text("base_config: dca_base.toml\n")
    return path


def positions(plate: Path) -> list[str]:
    return [str(plate / well / "0") for well in WELLS]


def run(plate: Path, config: Path, *more_plates: Path):
    dirpaths = [d for p in (plate, *more_plates) for d in positions(p)]
    return CliRunner().invoke(cli, ["dca-finalize", "-i", *dirpaths, "-c", str(config)])


def file_record(path: Path) -> dict[str, str]:
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def error_lines(result) -> list[str]:
    return result.output.strip().removeprefix("Error: ").splitlines()


def attributes(plate: Path, well: str) -> dict:
    with open_ome_zarr(plate / well / "0", mode="r") as position:
        return dict(position.zattrs)


def test_dca_finalize_writes_block(plate, config, caplog):
    with caplog.at_level(logging.WARNING, logger="dca_helpers"):
        result = run(plate, config)
    assert result.exit_code == 0, result.output
    assert [r.getMessage() for r in caplog.records] == []
    for well in WELLS:
        attrs = attributes(plate, well)
        dca = attrs["dca"]
        assert [c["name"] for c in dca["channels"]] == list(CHANNELS)
        stats = dca["normalization_statistics"]
        for c in range(len(CHANNELS)):
            assert stats[str(c)]["dataset_statistics"]["p1"] == p1_value(f"{well}/0", c)
        assert stats["metadata"]["source"] == "imaging-qc intensity_stats"
        assert dca["biosample"]["group_id"] == f"{STEM}/{well}"
        assert dca["acquisition"]["time_increment_s"] == TIME_INCREMENT_S
        assert attrs["biahub-dca-finalize"] == {
            "base_config": file_record(config.parent / "dca_base.toml"),
            "confirmed_config": file_record(confirmed_path(plate)),
            "metadata_source": {"kind": "csv", "name": sheet_path(plate).name},
            "qc_stage": 5,
        }


def test_relative_config_paths_resolve_outside_cwd(plate, config, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = run(plate, config)
    assert result.exit_code == 0, result.output
    written = attributes(plate, "A/1")["biahub-dca-finalize"]["base_config"]
    assert written == file_record(config.parent / "dca_base.toml")


def test_finalized_plate_passes_validator(plate, config):
    assert run(plate, config).exit_code == 0
    validation = validate(str(plate), show_progress=False)
    failures = [
        (r.node_path, i.severity.value, i.message)
        for r in validation.results
        for i in r.issues
        if not r.passed
    ]
    assert validation.results
    assert all(r.passed for r in validation.results), failures


def test_missing_well_row_reports_fields(plate, config):
    write_table(plate, [table_row("A/1")])
    result = run(plate, config)
    assert result.exit_code != 0
    assert error_lines(result) == [
        f"{plate / 'B' / '1' / '0'}: no metadata row for well B/1",
        f"fill the missing values in {confirmed_path(plate)}",
    ]
    for well in WELLS:
        assert "dca" not in attributes(plate, well)


def test_unresolved_field_lists_position_and_field(plate, config):
    confirmed_path(plate).unlink()
    result = run(plate, config)
    assert result.exit_code != 0
    for well in WELLS:
        assert f"{plate / well / '0'}: study.experimenter" in result.output


def test_resolver_errors_are_collected(plate, config):
    write_table(plate, [table_row("A/1", marker=""), table_row("B/1", perturbation="HIV")])
    result = run(plate, config)
    assert result.exit_code != 0
    assert f"{plate / 'A' / '1' / '0'}: " in result.output
    assert "raw GFP EX488 EM525-45" in result.output
    assert f"{plate / 'B' / '1' / '0'}: " in result.output
    assert "HIV" in result.output


def test_time_increment_mismatch_reported(plate, config):
    write_table(plate, [table_row("A/1", time_interval_min="10"), table_row("B/1")])
    result = run(plate, config)
    assert result.exit_code != 0
    assert "table says 600 s, store T scale says 1800 s" in result.output


def test_confirmed_file_overrides_table(plate, config):
    confirmed_path(plate).write_text(
        CONFIRMED
        + '[dca.biosample]\ncell_line_name = "A549 (confirmed)"\n'
        + '[dca.provenance.biosample.cell_line_name]\nsource_kind = "human"\n'
    )
    result = run(plate, config)
    assert result.exit_code == 0, result.output
    assert attributes(plate, "A/1")["dca"]["biosample"]["cell_line_name"] == "A549 (confirmed)"


def test_missing_stats_reported_with_metadata_gaps(plate, config):
    write_qc(plate, skip={"A/1/0": "p1", "B/1/0": "p99"})
    write_table(plate, [table_row("A/1")])
    result = run(plate, config)
    assert result.exit_code != 0
    assert error_lines(result) == [
        f"{plate / 'A' / '1' / '0'}: channel 'BF' has no intensity_stats rows for p1",
        f"{plate / 'B' / '1' / '0'}: no metadata row for well B/1",
        f"{plate / 'B' / '1' / '0'}: channel 'BF' has no intensity_stats rows for p99",
        f"fill the missing values in {confirmed_path(plate)}",
    ]
    for well in WELLS:
        assert "dca" not in attributes(plate, well)


def test_non_finite_statistics_write_nothing(plate, config):
    write_qc(plate, p1={"B/1/0": float("inf")})
    result = run(plate, config)
    assert result.exit_code != 0
    assert f"{plate / 'B' / '1' / '0'}: " in result.output
    assert "not finite" in result.output
    for well in WELLS:
        assert "dca" not in attributes(plate, well)


def test_unreadable_confirmed_file_named_in_error(plate, config):
    confirmed_path(plate).write_text("[dca.study\n")
    result = run(plate, config)
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == 1
    assert error_lines(result)[0].startswith(f"{plate}: {confirmed_path(plate)}: ")
    for well in WELLS:
        assert "dca" not in attributes(plate, well)


def test_confirmed_hint_lists_each_failing_plate_once(
    plate, config, tmp_path, create_custom_plate
):
    other = build_plate(tmp_path, create_custom_plate, OTHER_STEM)
    write_table(plate, [])
    result = run(plate, config, other)
    assert result.exit_code != 0
    hints = [line for line in error_lines(result) if line.startswith("fill the missing")]
    assert hints == [f"fill the missing values in {confirmed_path(plate)}"]


def test_dca_finalize_is_idempotent(plate, config):
    assert run(plate, config).exit_code == 0
    first = attributes(plate, "A/1")
    assert run(plate, config).exit_code == 0
    second = attributes(plate, "A/1")
    assert second["dca"] == first["dca"]
    assert second["biahub-concatenate"] == {"marker": "kept"}
    assert second == first


def test_rerun_drops_fields_no_longer_resolved(plate, config):
    assert run(plate, config).exit_code == 0
    assert "perturbation" in attributes(plate, "A/1")["dca"]
    write_table(
        plate,
        [
            table_row("A/1", perturbation="", moi="", hours_post_perturbation=""),
            table_row("B/1"),
        ],
    )
    assert run(plate, config).exit_code == 0
    assert "perturbation" not in attributes(plate, "A/1")["dca"]


def test_shared_sheet_serves_each_plate_its_own_rows(
    plate, config, tmp_path, create_custom_plate
):
    other = build_plate(tmp_path, create_custom_plate, OTHER_STEM)
    shared = tmp_path / "shared.dca-metadata.csv"
    for p in (plate, other):
        sheet_path(p).unlink()
    write_sheet(shared, [table_row(well) for well in WELLS])
    config.write_text(f"base_config: dca_base.toml\nmetadata_table: {shared}\n")

    result = run(plate, config, other)
    assert result.exit_code != 0
    for well in WELLS:
        assert f"{other / well / '0'}: no metadata row for well {well}" in result.output
        assert f"{plate / well / '0'}:" not in result.output
        for p in (plate, other):
            assert "dca" not in attributes(p, well)

    rows = [table_row(well) for well in WELLS]
    rows += [table_row(well, dataset=OTHER_STEM, marker="SEC61B") for well in WELLS]
    write_sheet(shared, rows)
    result = run(plate, config, other)
    assert result.exit_code == 0, result.output
    markers = {
        p.stem: attributes(p, "A/1")["dca"]["channels"][1]["biological_annotation"]["marker"]
        for p in (plate, other)
    }
    assert markers == {STEM: "TOMM20", OTHER_STEM: "SEC61B"}


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
