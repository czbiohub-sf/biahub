import shutil

from pathlib import Path

import pandas as pd
import pytest

from click.testing import CliRunner
from dca_helpers.validation.runner import validate
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli
from biahub.dca_metadata import TABLE_COLUMNS

BASE_CONFIG = Path(__file__).parents[2] / "nextflow" / "configs" / "a549" / "dca_base.toml"
CHANNELS = (
    "BF",
    "raw GFP EX488 EM525-45",
    "raw mCherry EX561 EM600-37",
    "Phase3D",
    "nuclei_prediction",
    "membrane_prediction",
)
STEM = "2026_01_02_synthetic_plate"
WELLS = ("A/1", "B/1")
TIME_POINTS = 2
TIME_INCREMENT_S = 1800.0
STAT_KEYS = ("p1", "p5", "p95", "p99", "mean", "std", "median", "iqr", "p95_p5", "p99_p1")
CONFIRMED = (
    "[dca.study]\n"
    'experimenter = [{ first_name = "Ada", last_name = "Example", institution = "Example Lab" }]\n'
    "[dca.provenance.study.experimenter]\n"
    'source_kind = "human"\n'
)


def p1_value(position_key: str, channel_index: int) -> float:
    return float(10 * WELLS.index(position_key.rsplit("/", 1)[0]) + channel_index)


def table_row(well: str, **overrides: str) -> dict[str, str]:
    row = {
        "dataset": STEM,
        "well": well,
        "cell_line": "A549",
        "marker": "TOMM20",
        "organelle": "mitochondria",
        "perturbation": "DENV",
        "moi": "1.5",
        "time_interval_min": "30",
        "hours_post_perturbation": "3",
    }
    row.update(overrides)
    return row


def write_table(plate: Path, rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    frame.to_csv(plate.parent / f"{STEM}.dca-metadata.csv", index=False)


def qc_rows(position_key: str, skip_metric: str | None = None) -> list[dict]:
    rows = []
    for c, name in enumerate(CHANNELS):
        for t in range(TIME_POINTS):
            for key in STAT_KEYS:
                if key == skip_metric:
                    continue
                value = (
                    p1_value(position_key, c) if key == "p1" else 1.0 + STAT_KEYS.index(key)
                )
                rows.append(
                    {
                        "store_path": f"{STEM}.zarr",
                        "dataset": STEM,
                        "batch": "b",
                        "plate": STEM,
                        "well": position_key.rsplit("/", 1)[0],
                        "position": position_key,
                        "field": position_key.rsplit("/", 1)[1],
                        "T": t,
                        "Z": None,
                        "C": c,
                        "channel_name": name,
                        "channel_type": "labelfree",
                        "metric_name": f"intensity_stats_{key}",
                        "value": value,
                        "granularity": "volume",
                        "stage": 5,
                        "source": "imaging-qc",
                        "timestamp": "2026-09-25T00:00:00",
                    }
                )
    return rows


def write_qc(plate: Path, skip: dict[str, str] | None = None) -> None:
    skip = skip or {}
    rows = [row for well in WELLS for row in qc_rows(f"{well}/0", skip.get(f"{well}/0"))]
    frame = pd.DataFrame(rows)
    frame["T"] = frame["T"].astype("Int64")
    frame["Z"] = frame["Z"].astype("Int64")
    frame["stage"] = frame["stage"].astype("int64")
    qc_dir = plate / "tables" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(qc_dir / "stage5_position.parquet")


@pytest.fixture
def plate(tmp_path, create_custom_plate) -> Path:
    build = tmp_path / "build"
    build.mkdir()
    built_path, dataset = create_custom_plate(
        build,
        position_list=[(*well.split("/"), "0") for well in WELLS],
        channel_names=CHANNELS,
        time_points=TIME_POINTS,
        z_size=16,
        y_size=128,
        x_size=128,
    )
    dataset.close()
    plate_path = tmp_path / "data" / f"{STEM}.zarr"
    plate_path.parent.mkdir()
    shutil.move(built_path, plate_path)
    for well in WELLS:
        with open_ome_zarr(plate_path / well / "0", mode="r+") as position:
            position.set_scale("0", "t", TIME_INCREMENT_S)
            position.zattrs["biahub-concatenate"] = {"marker": "kept"}
    write_table(plate_path, [table_row(well) for well in WELLS])
    write_qc(plate_path)
    (plate_path.parent / f"{STEM}.dca-confirmed.toml").write_text(CONFIRMED)
    return plate_path


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


def run(plate: Path, config: Path):
    return CliRunner().invoke(
        cli, ["dca-finalize", "-i", *positions(plate), "-c", str(config)]
    )


def attributes(plate: Path, well: str) -> dict:
    with open_ome_zarr(plate / well / "0", mode="r") as position:
        return dict(position.zattrs)


def test_dca_finalize_writes_block(plate, config):
    result = run(plate, config)
    assert result.exit_code == 0, result.output
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
        assert Path(attrs["biahub-dca-finalize"]["base_config"]).is_absolute()
        assert attrs["biahub-dca-finalize"]["qc_stage"] == 5


def test_relative_config_paths_resolve_outside_cwd(plate, config, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = run(plate, config)
    assert result.exit_code == 0, result.output
    written = attributes(plate, "A/1")["biahub-dca-finalize"]["base_config"]
    assert Path(written) == config.parent / "dca_base.toml"


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
    assert not failures


def test_missing_well_row_reports_fields(plate, config):
    write_table(plate, [table_row("A/1")])
    result = run(plate, config)
    assert result.exit_code != 0
    assert "no metadata row for well B/1" in result.output
    assert str(plate / "B" / "1" / "0") in result.output
    last_line = result.output.strip().splitlines()[-1]
    assert (
        last_line
        == f"fill the missing values in {plate.parent / f'{STEM}.dca-confirmed.toml'}"
    )
    for well in WELLS:
        assert "dca" not in attributes(plate, well)


def test_unresolved_field_lists_position_and_field(plate, config):
    (plate.parent / f"{STEM}.dca-confirmed.toml").unlink()
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
    (plate.parent / f"{STEM}.dca-confirmed.toml").write_text(
        CONFIRMED
        + '[dca.biosample]\ncell_line_name = "A549 (confirmed)"\n'
        + '[dca.provenance.biosample.cell_line_name]\nsource_kind = "human"\n'
    )
    result = run(plate, config)
    assert result.exit_code == 0, result.output
    assert attributes(plate, "A/1")["dca"]["biosample"]["cell_line_name"] == "A549 (confirmed)"


def test_missing_stats_reported_with_metadata_gaps(plate, config):
    write_qc(plate, skip={"A/1/0": "p1"})
    write_table(plate, [table_row("A/1")])
    result = run(plate, config)
    assert result.exit_code != 0
    assert "intensity_stats rows for p1" in result.output
    assert str(plate / "A" / "1" / "0") in result.output
    assert "no metadata row for well B/1" in result.output
    for well in WELLS:
        assert "dca" not in attributes(plate, well)


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
