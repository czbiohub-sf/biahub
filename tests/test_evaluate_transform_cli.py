"""evaluate-transform: a transforms file (and each block of its run) on positions, in µm."""

import csv

import numpy as np

from click.testing import CliRunner

from biahub.cli.main import cli
from biahub.core.transform import Transform
from biahub.settings import TransformEntry, TransformSettings
from biahub.utils.config import model_to_yaml
from tests.test_benchmark_core import SHIFT, _plate


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_evaluate_transform_writes_one_row_per_candidate_and_position(tmp_path):
    positions = _plate(tmp_path)
    transforms = tmp_path / "transforms.yml"
    exact = Transform.from_translation([-s for s in SHIFT])
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["DAPI"],
            reference_channel="VS",
            transforms=[TransformEntry(matrix=np.asarray(exact.matrix).tolist())],
        ),
        transforms,
    )
    output = tmp_path / "evaluation.csv"
    result = CliRunner().invoke(
        cli,
        [
            "evaluate-transform",
            "-m",
            *map(str, positions),
            "-r",
            *map(str, positions),
            "-c",
            str(transforms),
            "-o",
            str(output),
            "--tile-px",
            "64",
            "--figures",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = _rows(output)
    assert {(r["candidate"], r["position"]) for r in rows} == {
        (c, p) for c in ("unregistered", "transforms") for p in ("A/1/0", "A/1/1")
    }
    by = {(r["candidate"], r["position"]): float(r["tiles_median_um"]) for r in rows}
    assert by["transforms", "A/1/0"] < by["unregistered", "A/1/0"]
    assert sorted(p.name for p in (tmp_path / "evaluation").glob("*.png"))  # figures


def test_evaluate_transform_is_listed_with_its_siblings():
    result = CliRunner().invoke(cli, ["--help"])
    assert "evaluate-transform" in result.output
