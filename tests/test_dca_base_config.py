from pathlib import Path

import pytest

from dca_helpers.metadata import (
    STORE_COLUMNS,
    DcaValidationError,
    load_dca_config,
    merge_dca_attributes,
    resolve_dca,
    validate_dca,
    write_normalization_statistics,
)
from dca_helpers.validation.runner import validate
from iohub.ngff import open_ome_zarr

from biahub.dca_metadata import TABLE_COLUMNS
from biahub.dca_stats import DERIVED_COLUMNS

BASE_CONFIG = Path(__file__).parents[1] / "nextflow" / "configs" / "a549" / "dca_base.toml"
SIX = [
    "BF",
    "raw GFP EX488 EM525-45",
    "raw mCherry EX561 EM600-37",
    "Phase3D",
    "nuclei_prediction",
    "membrane_prediction",
]
FIVE = [name for name in SIX if "mCherry" not in name]
UNSOURCED = ["study.experimenter"]
STATS = {"p1": 0.0, "p5": 1.0, "p95": 9.0, "p99": 10.0, "mean": 5.0, "std": 2.0}


def full_row(**overrides: str) -> dict[str, str]:
    row = {
        "dataset": "synthetic_plate",
        "well": "A/1",
        "cell_line": "A549",
        "marker": "TOMM20",
        "organelle": "mitochondria",
        "perturbation": "DENV",
        "moi": "1.5",
        "time_interval_min": "30",
        "hours_post_perturbation": "3",
        "store_dataset_name": "2026_01_02_synthetic_plate",
        "store_group_id": "2026_01_02_synthetic_plate/A/1",
        "store_name_date": "2026-01-02",
        "store_time_increment_s": "1800.0",
        "dataset_year": "2026",
    }
    row.update(overrides)
    assert set(row) == set(TABLE_COLUMNS) | STORE_COLUMNS | set(DERIVED_COLUMNS)
    return row


def confirmed_file(tmp_path: Path) -> Path:
    path = tmp_path / "plate.dca-confirmed.toml"
    path.write_text(
        "[dca.study]\n"
        'experimenter = [{ first_name = "Ada", last_name = "Example", institution = "Example Lab" }]\n'
    )
    return path


def resolved(channels: list[str], row: dict[str, str], *overlays: Path) -> dict:
    block = load_dca_config(BASE_CONFIG, *overlays)
    return resolve_dca(block, row, channel_keys=channels, where="t")


def test_a549_base_config_validates(tmp_path):
    out = validate_dca(resolved(SIX, full_row(), confirmed_file(tmp_path)), where="t")
    assert out["study"]["license"] == "CC-BY-4.0"


def test_a549_base_config_five_channel_plate(tmp_path):
    out = validate_dca(resolved(FIVE, full_row(), confirmed_file(tmp_path)), where="t")
    assert [(c["index"], c["name"]) for c in out["channels"]] == list(enumerate(FIVE))


def test_mock_well_without_dose_validates(tmp_path):
    row = full_row(perturbation="mock", moi="", hours_post_perturbation="")
    out = validate_dca(resolved(SIX, row, confirmed_file(tmp_path)), where="t")
    assert out["perturbation"]["control_class"] == "negcon"
    assert "concentration" not in out["perturbation"]


def test_blank_marker_names_the_channel(tmp_path):
    with pytest.raises(ValueError, match="raw GFP EX488 EM525-45"):
        resolved(SIX, full_row(marker=""), confirmed_file(tmp_path))


def test_blank_perturbation_has_no_orphaned_provenance(tmp_path):
    row = full_row(perturbation="", moi="", hours_post_perturbation="")
    out = validate_dca(resolved(SIX, row, confirmed_file(tmp_path)), where="t")
    assert "perturbation" not in out


def test_base_config_alone_reports_unsourced_fields():
    with pytest.raises(DcaValidationError) as error:
        validate_dca(resolved(SIX, full_row()), where="t")
    assert sorted(error.value.fields) == UNSOURCED


def test_dateless_name_reports_release_date(tmp_path):
    row = full_row(store_name_date="", dataset_year="")
    with pytest.raises(DcaValidationError) as error:
        validate_dca(resolved(SIX, row, confirmed_file(tmp_path)), where="t")
    for field in ("study.release_date", "study.year_imaged"):
        assert field in error.value.fields
    assert any("acquisition.acquisition_date" in issue for issue in error.value.issues)


def test_base_config_has_no_internal_paths():
    text = BASE_CONFIG.read_text()
    for forbidden in ("/hpc/", "s3://", "Confluence"):
        assert forbidden not in text


def test_a549_base_config_passes_repo_validator(tmp_path):
    row = full_row()
    block = validate_dca(resolved(SIX, row, confirmed_file(tmp_path)), where="t")
    channels = block.pop("channels")
    plate_path = tmp_path / "2026_01_02_synthetic_plate.ome.zarr"
    with open_ome_zarr(plate_path, layout="hcs", mode="w", channel_names=SIX) as plate:
        position = plate.create_position("A", "1", "0")
        position.create_zeros(
            "0", shape=(2, 6, 16, 128, 128), chunks=(1, 1, 16, 128, 128), dtype="uint16"
        )
        attrs = dict(position.zattrs)
        merge_dca_attributes(attrs, block, channels)
        write_normalization_statistics(
            attrs, {str(i): {"dataset_statistics": STATS} for i in range(len(SIX))}
        )
        position.zattrs.update(attrs)
    run = validate(str(plate_path), show_progress=False)
    failures = [
        (result.node_path, issue.severity.value, issue.message)
        for result in run.results
        for issue in result.issues
        if not result.passed
    ]
    assert run.results, "validator found no nodes"
    assert all(r.passed for r in run.results)
    assert any(r.parent_path is not None for r in run.results)
    assert not failures
