"""Synthetic assembled plates with a sample sheet, QC table and confirmed file for dca-finalize."""

import hashlib
import shutil

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from click.testing import CliRunner, Result
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli
from biahub.dca_metadata import TABLE_COLUMNS

BASE_CONFIG = Path(__file__).parents[1] / "nextflow" / "configs" / "a549" / "dca_base.toml"
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
RELEASE_DATE = "2026-12-01"
CONFIRMED = (
    "[dca.study]\n"
    'experimenter = [{ first_name = "Ada", last_name = "Example", institution = "Example Lab" }]\n'
    f'release_date = "{RELEASE_DATE}"\n'
    "[dca.provenance.study.experimenter]\n"
    'source_kind = "human"\n'
    "[dca.provenance.study.release_date]\n"
    'source_kind = "human"\n'
)


def p1_value(position_key: str, channel_index: int) -> float:
    return float(10 * WELLS.index(position_key.rsplit("/", 1)[0]) + channel_index)


def table_row(well: str, dataset: str = STEM, **overrides: str) -> dict[str, str]:
    row = {
        "dataset": dataset,
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


def sheet_path(plate: Path) -> Path:
    return plate.parent / f"{plate.stem}.dca-metadata.csv"


def confirmed_path(plate: Path) -> Path:
    return plate.parent / f"{plate.stem}.dca-confirmed.toml"


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows, columns=list(TABLE_COLUMNS)).to_csv(path, index=False)


def _qc_row(stem: str, position_key: str, t: int, c: int, key: str, value: float) -> dict:
    return {
        "store_path": f"{stem}.zarr",
        "dataset": stem,
        "batch": "b",
        "plate": stem,
        "well": position_key.rsplit("/", 1)[0],
        "position": position_key,
        "field": position_key.rsplit("/", 1)[1],
        "T": t,
        "Z": None,
        "C": c,
        "channel_name": CHANNELS[c],
        "channel_type": "labelfree",
        "metric_name": f"intensity_stats_{key}",
        "value": value,
        "granularity": "volume",
        "stage": 5,
        "source": "imaging-qc",
        "timestamp": "2026-09-25T00:00:00",
    }


def write_qc(
    plate: Path,
    skip: dict[str, str] | None = None,
    p1: dict[str, float] | None = None,
) -> None:
    """Write ``tables/qc/stage5_position.parquet``.

    ``skip`` maps a position key to a metric left out for every channel. ``p1`` maps a
    position key to the p1 value written for every channel and T.
    """
    skip = skip or {}
    p1 = p1 or {}
    rows = []
    for well in WELLS:
        key = f"{well}/0"
        for c in range(len(CHANNELS)):
            for t in range(TIME_POINTS):
                for stat in STAT_KEYS:
                    if stat == skip.get(key):
                        continue
                    value = 1.0 + STAT_KEYS.index(stat)
                    if stat == "p1":
                        value = p1.get(key, p1_value(key, c))
                    rows.append(_qc_row(plate.stem, key, t, c, stat, value))
    frame = pd.DataFrame(rows)
    frame["T"] = frame["T"].astype("Int64")
    frame["Z"] = frame["Z"].astype("Int64")
    frame["stage"] = frame["stage"].astype("int64")
    qc_dir = plate / "tables" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(qc_dir / "stage5_position.parquet")


def build_plate(tmp_path: Path, create_custom_plate: Callable, stem: str = STEM) -> Path:
    """Build ``<tmp>/data/<stem>.zarr`` with its own sheet, QC table and confirmed file."""
    build = tmp_path / f"build-{stem}"
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
    plate = tmp_path / "data" / f"{stem}.zarr"
    plate.parent.mkdir(exist_ok=True)
    shutil.move(built_path, plate)
    for well in WELLS:
        with open_ome_zarr(plate / well / "0", mode="r+") as position:
            position.set_scale("0", "t", TIME_INCREMENT_S)
            position.zattrs["biahub-concatenate"] = {"marker": "kept"}
    write_table(sheet_path(plate), [table_row(well, dataset=stem) for well in WELLS])
    write_qc(plate)
    confirmed_path(plate).write_text(CONFIRMED)
    return plate


def write_config(tmp_path: Path) -> Path:
    """Write ``configs/dca_finalize.yml`` next to a copy of the base config it names."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    shutil.copy(BASE_CONFIG, config_dir / "dca_base.toml")
    path = config_dir / "dca_finalize.yml"
    path.write_text("base_config: dca_base.toml\n")
    return path


def positions(plate: Path) -> list[str]:
    return [str(plate / well / "0") for well in WELLS]


def run(plate: Path, config: Path, *more_plates: Path) -> Result:
    dirpaths = [d for p in (plate, *more_plates) for d in positions(p)]
    return CliRunner().invoke(cli, ["dca-finalize", "-i", *dirpaths, "-c", str(config)])


def attributes(plate: Path, well: str) -> dict:
    with open_ome_zarr(plate / well / "0", mode="r") as position:
        return dict(position.zattrs)


def file_record(path: Path) -> dict[str, str]:
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
