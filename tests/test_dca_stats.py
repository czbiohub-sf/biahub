from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dca_helpers.metadata import STORE_COLUMNS
from iohub.ngff import open_ome_zarr

from biahub.dca_stats import (
    MissingStatisticsError,
    dataset_date,
    load_qc_statistics,
    position_statistics,
    store_facts,
    time_increment_issue,
)

POSITION = "C/2/0001"
CHANNELS = ["Phase3D", "nuclei_prediction"]


def _row(channel: str, t: int | None, metric: str, value: float, position: str = POSITION):
    return {
        "store_path": "plate.zarr",
        "dataset": "example",
        "batch": "b",
        "plate": "plate",
        "well": position.rsplit("/", 1)[0],
        "position": position,
        "field": position.rsplit("/", 1)[1],
        "T": t,
        "Z": None,
        "C": CHANNELS.index(channel),
        "channel_name": channel,
        "channel_type": "labelfree",
        "metric_name": f"intensity_stats_{metric}" if metric != "other" else "focus_score",
        "value": value,
        "granularity": "volume",
        "stage": 5,
        "source": "imaging-qc",
        "timestamp": "2026-09-25T00:00:00",
    }


def _write_qc(plate: Path, rows: list[dict], stage: int = 5) -> None:
    qc_dir = plate / "tables" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["T"] = frame["T"].astype("Int64")
    frame["Z"] = frame["Z"].astype("Int64")
    frame["stage"] = frame["stage"].astype("int64")
    frame.to_parquet(qc_dir / f"stage{stage}_position.parquet")


def _full_rows(channel: str, per_t: dict[int, dict[str, float]]) -> list[dict]:
    return [
        _row(channel, t, metric, value)
        for t, stats in per_t.items()
        for metric, value in stats.items()
    ]


def _stats(p1: float = 0.0, mean: float = 0.0, std: float = 1.0, p99: float = 5.0):
    keys = ("p5", "p95", "median", "iqr", "p95_p5", "p99_p1")
    return {"p1": p1, "mean": mean, "std": std, "p99": p99, **{k: 1.0 for k in keys}}


def _qc(tmp_path: Path, rows: list[dict]) -> pd.DataFrame:
    plate = tmp_path / "plate.zarr"
    _write_qc(plate, rows)
    return load_qc_statistics(plate, 5)


def test_position_statistics_shapes(tmp_path):
    rows = _full_rows(
        "Phase3D", {0: _stats(p1=1), 1: _stats(p1=2), 2: _stats(p1=9)}
    ) + _full_rows("nuclei_prediction", {0: _stats(), 1: _stats(), 2: _stats(p99=7.5)})
    rows.append(_row("Phase3D", 0, "p25", 3.0))
    rows.append(_row("Phase3D", 0, "other", 3.0))
    result = position_statistics(_qc(tmp_path, rows), POSITION, CHANNELS)
    assert result["0"]["dataset_statistics"]["p1"] == 2
    assert result["1"]["timepoint_statistics"]["2"]["p99"] == 7.5
    assert "p25" not in result["0"]["dataset_statistics"]
    assert "p25" not in result["0"]["timepoint_statistics"]["0"]
    assert set(result["0"]["timepoint_statistics"]) == {"0", "1", "2"}


def test_pooled_mean_and_std_exact(tmp_path):
    per_t = {0: _stats(mean=0, std=1), 1: _stats(mean=2, std=1)}
    rows = _full_rows("Phase3D", per_t) + _full_rows("nuclei_prediction", per_t)
    result = position_statistics(_qc(tmp_path, rows), POSITION, CHANNELS)
    assert result["0"]["dataset_statistics"]["mean"] == 1.0
    assert result["0"]["dataset_statistics"]["std"] == pytest.approx(2**0.5)


def test_channel_index_follows_omero_order(tmp_path):
    rows = _full_rows("nuclei_prediction", {0: _stats(p1=4)}) + _full_rows(
        "Phase3D", {0: _stats(p1=8)}
    )
    result = position_statistics(
        _qc(tmp_path, rows), POSITION, ["nuclei_prediction", "Phase3D"]
    )
    assert result["0"]["dataset_statistics"]["p1"] == 4
    assert result["1"]["dataset_statistics"]["p1"] == 8


def test_rows_with_null_t_ignored(tmp_path):
    per_t = {0: _stats(p1=1), 1: _stats(p1=3)}
    rows = _full_rows("Phase3D", per_t) + _full_rows("nuclei_prediction", per_t)
    baseline = position_statistics(_qc(tmp_path, rows), POSITION, CHANNELS)
    rows.append(_row("Phase3D", None, "p1", 1000.0))
    assert position_statistics(_qc(tmp_path, rows), POSITION, CHANNELS) == baseline


def test_other_positions_ignored(tmp_path):
    per_t = {0: _stats(p1=1)}
    rows = _full_rows("Phase3D", per_t) + _full_rows("nuclei_prediction", per_t)
    rows.append(_row("Phase3D", 0, "p1", 500.0, position="A/1/0000"))
    result = position_statistics(_qc(tmp_path, rows), POSITION, CHANNELS)
    assert result["0"]["dataset_statistics"]["p1"] == 1


def test_non_volume_rows_not_loaded(tmp_path):
    rows = _full_rows("Phase3D", {0: _stats(p1=1)}) + _full_rows(
        "nuclei_prediction", {0: _stats()}
    )
    rows.append({**_row("Phase3D", 0, "p1", 900.0), "granularity": "slice"})
    qc = _qc(tmp_path, rows)
    assert set(qc.columns) == {
        "position",
        "T",
        "channel_name",
        "metric_name",
        "value",
        "granularity",
    }
    assert (qc["granularity"] == "volume").all()
    assert set(qc["metric_name"]) <= {f"intensity_stats_{k}" for k in _stats()}


def test_missing_percentile_names_channel(tmp_path):
    rows = _full_rows("Phase3D", {0: _stats()})
    rows += [
        r
        for r in _full_rows("nuclei_prediction", {0: _stats()})
        if r["metric_name"] != "intensity_stats_p1"
    ]
    with pytest.raises(MissingStatisticsError) as error:
        position_statistics(_qc(tmp_path, rows), POSITION, CHANNELS)
    message = str(error.value)
    assert POSITION in message and "nuclei_prediction" in message and "p1" in message


def test_missing_parquet_raises(tmp_path):
    plate = tmp_path / "plate.zarr"
    with pytest.raises(MissingStatisticsError) as error:
        load_qc_statistics(plate, 5)
    assert str(plate / "tables" / "qc" / "stage5_position.parquet") in str(error.value)


@pytest.mark.parametrize(
    ("table", "store", "expected"),
    [("30", 1800.0, None), ("", 1800.0, None), ("", "", None)],
)
def test_time_increment_consistent(table, store, expected):
    row = {"time_interval_min": table}
    assert time_increment_issue(row, {"store_time_increment_s": store}) is expected


def test_time_increment_mismatch_and_mixed():
    facts = {"store_time_increment_s": 600.0}
    message = time_increment_issue({"time_interval_min": "30"}, facts)
    assert "1800" in message and "600" in message
    assert message.startswith("acquisition.time_increment_s:")
    mixed = time_increment_issue({"time_interval_min": "30; 10"}, facts)
    assert "not supported" in mixed


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("2026_09_25_example", "2026-09-25"),
        ("2026_13_40_example", None),
        ("example", None),
    ],
)
def test_dataset_date(stem, expected):
    assert dataset_date(stem) == expected


@pytest.mark.parametrize(("unit", "scale"), [("second", 1800), ("minute", 30), ("hour", 0.5)])
def test_store_facts(tmp_path, unit, scale):
    plate = tmp_path / "2026_09_25_example.zarr"
    with open_ome_zarr(plate, layout="hcs", mode="w", channel_names=["a"]) as dataset:
        position = dataset.create_position("C", "2", "0001")
        position["0"] = np.zeros((2, 1, 1, 2, 2), dtype=np.uint16)
        position.set_scale("0", "t", scale)
        position.metadata.multiscales[0].axes[0].unit = unit
        position.dump_meta()
    with open_ome_zarr(plate / POSITION, mode="r") as position:
        facts = store_facts(plate, POSITION, position)
    assert set(facts) == STORE_COLUMNS
    assert facts["store_name_date"] == "2026-09-25"
    assert facts["store_time_increment_s"] == 1800.0
    assert facts["store_dataset_name"] == "2026_09_25_example"
    assert facts["store_group_id"] == "2026_09_25_example/C/2"


def test_store_facts_without_time_unit(tmp_path):
    plate = tmp_path / "example.zarr"
    with open_ome_zarr(plate, layout="hcs", mode="w", channel_names=["a"]) as dataset:
        position = dataset.create_position("C", "2", "0001")
        position["0"] = np.zeros((2, 1, 1, 2, 2), dtype=np.uint16)
        position.metadata.multiscales[0].axes[0].unit = None
        position.dump_meta()
    with open_ome_zarr(plate / POSITION, mode="r") as position:
        facts = store_facts(plate, POSITION, position)
    assert facts["store_time_increment_s"] == ""
    assert facts["store_name_date"] == ""
