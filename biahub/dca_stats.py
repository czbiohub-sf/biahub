"""Store facts and normalization statistics for ``dca-finalize``."""

import math

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as pads

from dca_helpers.metadata import STORE_COLUMNS
from iohub.ngff import Position

DCA_STAT_KEYS = ("p1", "p5", "p95", "p99", "mean", "std", "median", "iqr", "p95_p5", "p99_p1")
PERCENTILE_KEYS = ("p1", "p5", "p95", "p99", "median", "iqr", "p95_p5", "p99_p1")
STATS_METADATA = {
    "source": "imaging-qc intensity_stats",
    "dataset_statistics": "median across T; mean and std pooled",
}

QC_COLUMNS = ["position", "T", "channel_name", "metric_name", "value", "granularity"]
METRIC_PREFIX = "intensity_stats_"
SECONDS_PER_UNIT = {"second": 1.0, "minute": 60.0, "hour": 3600.0}
TIME_INCREMENT_TOLERANCE_S = 1.0
DATE_PREFIX_LENGTH = len("YYYY_MM_DD")
DERIVED_COLUMNS: tuple[str, ...] = ("dataset_year",)
YEAR_LENGTH = len("YYYY")


class MissingStatisticsError(ValueError):
    """Raised when the QC table cannot supply the statistics a position needs."""


def dataset_date(stem: str) -> str | None:
    """Return ``YYYY-MM-DD`` from a ``YYYY_MM_DD_`` name prefix, or None when absent or invalid."""
    prefix = stem[:DATE_PREFIX_LENGTH]
    if stem[DATE_PREFIX_LENGTH : DATE_PREFIX_LENGTH + 1] != "_":
        return None
    try:
        return datetime.strptime(prefix, "%Y_%m_%d").date().isoformat()
    except ValueError:
        return None


def derived_columns(plate: Path) -> dict[str, str]:
    """Return the ``DERIVED_COLUMNS`` values for ``plate``, blank when its name has no date."""
    return {"dataset_year": (dataset_date(plate.stem) or "")[:YEAR_LENGTH]}


def _time_increment_s(position: Position) -> float | str:
    for axis, scale in zip(position.axes, position.scale, strict=True):
        if axis.type == "time" or axis.name.lower() == "t":
            factor = SECONDS_PER_UNIT.get(axis.unit or "")
            return "" if factor is None else float(scale) * factor
    return ""


def store_facts(plate: Path, position_key: str, position: Position) -> dict[str, Any]:
    """Return the ``STORE_COLUMNS`` facts for one FOV of ``plate``."""
    row, col = position_key.split("/")[:2]
    facts = {
        "store_dataset_name": plate.stem,
        "store_group_id": f"{plate.stem}/{row}/{col}",
        "store_name_date": dataset_date(plate.stem) or "",
        "store_time_increment_s": _time_increment_s(position),
    }
    if set(facts) != STORE_COLUMNS:
        raise RuntimeError(f"store facts keys {sorted(facts)} differ from STORE_COLUMNS")
    return facts


def qc_path(plate: Path, stage: int) -> Path:
    return plate / "tables" / "qc" / f"stage{stage}_position.parquet"


def load_qc_statistics(plate: Path, stage: int) -> pd.DataFrame:
    """Read the DCA intensity statistics rows of a plate QC table, once per plate."""
    path = qc_path(plate, stage)
    if not path.is_file():
        raise MissingStatisticsError(f"QC statistics table not found: {path}")
    metrics = [f"{METRIC_PREFIX}{key}" for key in DCA_STAT_KEYS]
    table = pads.dataset(path, format="parquet").to_table(
        columns=QC_COLUMNS,
        filter=pads.field("metric_name").isin(metrics)
        & (pads.field("granularity") == "volume"),
    )
    return table.to_pandas()


def _channel_statistics(frame: pd.DataFrame) -> dict[str, Any] | None:
    wide = frame.pivot_table(index="T", columns="stat", values="value", aggfunc="first")
    missing = [key for key in DCA_STAT_KEYS if key not in wide.columns]
    if missing:
        return None
    mean = float(wide["mean"].mean())
    second_moment = float((wide["std"] ** 2 + wide["mean"] ** 2).mean())
    dataset = {key: float(wide[key].median()) for key in PERCENTILE_KEYS}
    dataset["mean"] = mean
    dataset["std"] = math.sqrt(max(second_moment - mean**2, 0.0))
    timepoints = {
        str(int(t)): {key: float(wide.loc[t, key]) for key in DCA_STAT_KEYS}
        for t in wide.index
    }
    return {"dataset_statistics": dataset, "timepoint_statistics": timepoints}


def position_statistics(
    qc: pd.DataFrame, position_key: str, channel_names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Return per-channel DCA statistics for one position, keyed by omero channel index."""
    rows = qc[(qc["position"] == position_key) & qc["T"].notna()].copy()
    rows["stat"] = rows["metric_name"].str.removeprefix(METRIC_PREFIX)
    result: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(channel_names):
        channel_rows = rows[rows["channel_name"] == name]
        statistics = _channel_statistics(channel_rows) if len(channel_rows) else None
        if statistics is None:
            present = set(channel_rows["stat"])
            missing = [key for key in DCA_STAT_KEYS if key not in present]
            raise MissingStatisticsError(
                f"{position_key}: channel {name!r} has no intensity_stats rows for "
                f"{', '.join(missing)}"
            )
        result[str(index)] = statistics
    return result


def _format_seconds(value: float) -> str:
    return f"{value:g}"


def time_increment_issue(row: Mapping[str, str], facts: Mapping[str, Any]) -> str | None:
    """Compare the table's ``time_interval_min`` with the store T scale.

    Returns None when consistent or when either side is empty.
    """
    raw = (row.get("time_interval_min") or "").strip()
    store = facts.get("store_time_increment_s", "")
    if not raw:
        return None
    values = [part.strip() for part in raw.split(";") if part.strip()]
    if len(values) > 1:
        return (
            "acquisition.time_increment_s: channels acquired at different intervals "
            f"({', '.join(values)}); not supported"
        )
    if store == "":
        return None
    table_s = float(values[0]) * 60.0
    if abs(table_s - float(store)) > TIME_INCREMENT_TOLERANCE_S:
        return (
            f"acquisition.time_increment_s: table says {_format_seconds(table_s)} s, "
            f"store T scale says {_format_seconds(float(store))} s"
        )
    return None
