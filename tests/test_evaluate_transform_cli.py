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


def test_tags_add_columns_so_tables_can_be_joined(tmp_path):
    positions = _plate(tmp_path)[:1]
    transforms = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["DAPI"],
            reference_channel="VS",
            transforms=[TransformEntry(matrix=np.eye(4).tolist())],
        ),
        transforms,
    )
    output = tmp_path / "evaluation.csv"
    args = ["evaluate-transform", "-m", str(positions[0]), "-r", str(positions[0])]
    args += ["-c", str(transforms), "-o", str(output), "--tile-px", "64"]
    args += ["--tag", "case=dragonfly", "--tag", "pipeline=tiles > nuclei"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    rows = _rows(output)
    assert list(rows[0])[:2] == ["case", "pipeline"]  # first, for reading
    assert {(r["case"], r["pipeline"]) for r in rows} == {("dragonfly", "tiles > nuclei")}


def test_a_per_position_file_is_evaluated_on_held_out_positions(tmp_path):
    # Estimated on several positions, the file holds one list each; held-out positions
    # are not in it: each estimated position's transform is a candidate there.
    positions = _plate(tmp_path)
    exact = Transform.from_translation([-s for s in SHIFT])
    transforms = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["DAPI"],
            reference_channel="VS",
            positions={"A/1/0": [TransformEntry(matrix=np.asarray(exact.matrix).tolist())]},
        ),
        transforms,
    )
    output = tmp_path / "evaluation.csv"
    args = ["evaluate-transform", "-m", str(positions[1]), "-r", str(positions[1])]
    args += ["-c", str(transforms), "-o", str(output), "--tile-px", "64"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert {r["candidate"] for r in _rows(output)} == {"unregistered", "transforms (A/1/0)"}


def test_block_candidates_read_only_the_current_runs_layout(tmp_path):
    import json

    from biahub.registration.evaluate import block_candidates

    run = tmp_path / "run"
    record = {"t": 0, "blocks": [{"block": "1:a", "matrix": np.eye(4).tolist()}]}
    (run / "timepoints").mkdir(parents=True)
    (run / "timepoints" / "0.json").write_text(json.dumps(record))
    stale = run / "positions" / "A" / "1" / "0" / "timepoints"  # an earlier run's layout
    stale.mkdir(parents=True)
    (stale / "0.json").write_text(
        json.dumps({**record, "blocks": [{**record["blocks"][0], "block": "old"}]})
    )
    (run / "run.json").write_text(json.dumps({"positions": ["A/1/0"]}))
    assert set(block_candidates(run)) == {"1:a"}
