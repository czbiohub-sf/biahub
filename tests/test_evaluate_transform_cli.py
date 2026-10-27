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

    from biahub.registration.benchmark import block_candidates

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


def test_bead_landmarks_take_the_runs_detection_threshold(tmp_path):
    # --landmarks beads crashed: bead detection needs a threshold. It comes from the run's
    # own settings (what the estimate detected beads with), or --bead-threshold.
    from scipy.ndimage import shift as ndi_shift

    from tests.test_estimate_transform import (
        APPLIED_SHIFT_ZYX,
        SHAPE,
        _run,
        _synthetic_bead_volume,
        _write_config,
        _write_plate,
    )

    ref = _synthetic_bead_volume(np.random.default_rng(11), SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    plate = _write_plate(tmp_path / "beads.zarr", [(ref, mov)])
    output = tmp_path / "out" / "transforms.yml"
    _run(plate, _write_config(tmp_path), output)

    table = tmp_path / "evaluation.csv"
    args = ["evaluate-transform", "-m", str(plate), "-r", str(plate), "-c", str(output)]
    args += ["-o", str(table), "--tile-px", "32", "--landmarks", "beads"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    final = [r for r in _rows(table) if r["candidate"] == "transforms"][0]
    assert int(final["points_n"]) > 5 and float(final["points_median_um"]) < 1.0

    # no run folder and no threshold: say what is missing
    bare = tmp_path / "bare.yml"
    bare.write_text(output.read_text())
    result = CliRunner().invoke(cli, args[:6] + [str(bare)] + args[7:])
    assert result.exit_code != 0 and "--bead-threshold" in result.output


def test_a_stabilization_is_evaluated_against_its_reference_frame(tmp_path):
    # Stabilization compares timepoint t with the reference frame (t=0), in the same store
    # and channel; comparing t with itself measured nothing.
    from iohub import open_ome_zarr
    from scipy.ndimage import shift as ndi_shift

    from tests.test_registration_estimators import _synthetic_blob_volume

    first = _synthetic_blob_volume(
        np.random.default_rng(3), (24, 192, 192), n_blobs=200, sigma=3.0
    )
    first = first + 100.0
    drifted = ndi_shift(first, SHIFT, order=1, mode="nearest")
    path = tmp_path / "drift.zarr"
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["DAPI"]) as plate:
        plate.create_position("A", "1", "0").create_image(
            "0", np.stack([first, drifted])[:, None].astype(np.float32)
        )
    position = path / "A" / "1" / "0"
    exact = Transform.from_translation([-s for s in SHIFT])
    transforms = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["DAPI"],
            transforms=[
                TransformEntry(t=0, matrix=np.eye(4).tolist()),
                TransformEntry(t=1, matrix=np.asarray(exact.matrix).tolist()),
            ],
        ),
        transforms,
    )
    output = tmp_path / "evaluation.csv"
    args = ["evaluate-transform", "-m", str(position), "-r", str(position)]
    args += ["-c", str(transforms), "-o", str(output), "--tile-px", "64", "--t", "1"]
    args += ["--reference-t", "0"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    tiles = {r["candidate"]: float(r["tiles_median_um"]) for r in _rows(output)}
    assert tiles["transforms"] < 0.05 < tiles["unregistered"]


def test_a_blank_position_is_reported_not_left_as_empty_rows(tmp_path):
    from iohub import open_ome_zarr

    positions = _plate(tmp_path)
    with open_ome_zarr(positions[1], mode="r+") as position:
        position.data[0, 0] = 0
    transforms = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["DAPI"],
            reference_channel="VS",
            transforms=[TransformEntry(matrix=Transform.identity(3).to_list())],
        ),
        transforms,
    )
    output = tmp_path / "evaluation.csv"
    args = ["evaluate-transform", "-m", *map(str, positions), "-r", *map(str, positions)]
    args += ["-c", str(transforms), "-o", str(output), "--tile-px", "64"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "A/1/1: WARNING the moving image is blank, skipped" in result.output
    assert [r["empty"] for r in _rows(output) if r["position"] == "A/1/1"] == ["moving"]
