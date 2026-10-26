"""nextflow/registration_benchmark_wf.nf end to end, local executor (needs Nextflow)."""

import csv
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from iohub import open_ome_zarr
from scipy.ndimage import shift as ndi_shift

from biahub.settings import EstimateTransformSettings
from biahub.utils.config import model_to_yaml
from tests.test_estimate_transform import APPLIED_SHIFT_ZYX, SHAPE, _synthetic_bead_volume
from tests.test_nextflow_registration import NEXTFLOW_DIR, _env
from tests.test_stacked_methods import BEADS_BLOCK, PEAKS

pytestmark = pytest.mark.skipif(shutil.which("nextflow") is None, reason="needs nextflow")


def _plate(tmp_path):
    """Two positions, the same beads shifted the same way: estimate on one, test the other."""
    path = tmp_path / "beads.zarr"
    with open_ome_zarr(
        path, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]
    ) as plate:
        for fov in ("0", "1"):
            ref = _synthetic_bead_volume(np.random.default_rng(int(fov) + 11), SHAPE)
            mov = ndi_shift(ref, APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
            plate.create_position("A", "1", fov)["0"] = np.stack([ref, mov])[None].astype(
                np.float32
            )
    return path


def _config(path, method):
    # channels are placeholders: the benchmark puts each case's in
    settings = EstimateTransformSettings(
        moving={"channel": "placeholder"},
        reference={"frame": "cross", "channel": "placeholder"},
        method=method,
        **({"beads": BEADS_BLOCK["beads"]} if method == "beads" else {}),
    )
    model_to_yaml(settings, path)
    return path.name


def _workflow(tmp_path, output, *extra):
    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(NEXTFLOW_DIR / "registration_benchmark_wf.nf"),
            "-ansi-log",
            "false",
        ]
        + ["-c", str(NEXTFLOW_DIR / "nextflow.config"), "-work-dir", str(tmp_path / "work")]
        + ["--benchmark", str(tmp_path / "benchmark.yml"), "--output", str(output)]
        + list(extra),
        cwd=tmp_path,
        env=_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result.stdout


def _rows(output):
    with open(output / "benchmark.csv") as f:
        return list(csv.DictReader(f))


def test_the_benchmark_runs_every_pipeline_on_every_case_into_one_table(tmp_path):
    plate = _plate(tmp_path)
    spec = {
        "cases": [
            {
                "name": "beads",
                "moving": str(plate),
                "reference": str(plate),
                "moving_channel": "GFP",
                "reference_channel": "Phase3D",
                "estimate_positions": "A/1/0",
                "evaluate_positions": ["A/1/1"],
                "tile_px": 32,
            }
        ],
        "pipelines": [
            {"name": "beads", "config": _config(tmp_path / "beads.yml", "beads")},
            {
                "name": "pcc_beads",
                "config": _config(
                    tmp_path / "stack.yml", [{"phase-cross-corr": {}}, BEADS_BLOCK]
                ),
            },
        ],
    }
    (tmp_path / "benchmark.yml").write_text(yaml.safe_dump(spec))
    output = tmp_path / "out"
    result = subprocess.run(
        ["nextflow", "run", str(NEXTFLOW_DIR / "registration_benchmark_wf.nf")]
        + ["-c", str(NEXTFLOW_DIR / "nextflow.config"), "-work-dir", str(tmp_path / "work")]
        + ["--benchmark", str(tmp_path / "benchmark.yml"), "--output", str(output)],
        cwd=tmp_path,
        env=_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    with open(output / "benchmark.csv") as f:
        rows = list(csv.DictReader(f))
    by = {(r["pipeline"], r["candidate"]): r for r in rows}
    assert {r["position"] for r in rows} == {"A/1/1"}  # the held-out position
    assert {k for k in by if k[0] == "pcc_beads"} == {
        ("pcc_beads", c)
        for c in ("unregistered", "transforms", "1:phase-cross-corr", "2:beads")
    }
    assert ("beads", "transforms") in by and ("beads", "unregistered") in by
    tile = {k: float(r["tiles_median_um"]) for k, r in by.items() if r["tiles_median_um"]}
    assert tile["beads", "transforms"] < tile["beads", "unregistered"]
    assert (
        yaml.safe_load((output / "beads" / "beads" / "estimate.yml").read_text())["moving"][
            "channel"
        ]
        == "GFP"
    )  # the case's channel, not the pipeline's placeholder
    assert PEAKS  # (the bead block's detection settings come from the stacked-methods tests)


def _one_case_spec(tmp_path, plate, pipelines):
    spec = {
        "cases": [
            {
                "name": "beads",
                "moving": str(plate),
                "reference": str(plate),
                "moving_channel": "GFP",
                "reference_channel": "Phase3D",
                "estimate_positions": "A/1/0",
                "evaluate_positions": ["A/1/1"],
                "tile_px": 32,
            }
        ],
        "pipelines": pipelines,
    }
    (tmp_path / "benchmark.yml").write_text(yaml.safe_dump(spec))


def test_resume_reuses_unchanged_runs_and_an_edited_config_reestimates(tmp_path):
    # A plain -resume reran every estimate (its config was rewritten at launch) yet kept
    # the old evaluation (keyed on the path); an edited config failed the whole run.
    plate = _plate(tmp_path)
    beads = _config(tmp_path / "beads.yml", "beads")
    _one_case_spec(tmp_path, plate, [{"name": "beads", "config": beads}])
    output = tmp_path / "out"
    _workflow(tmp_path, output)
    again = _workflow(tmp_path, output, "-resume")
    assert again.count("Cached process") == 2  # estimate and evaluate both reused

    config = yaml.safe_load((tmp_path / "beads.yml").read_text())
    config["transform"]["type"] = (
        "euclidean" if config["transform"]["type"] != "euclidean" else "affine"
    )
    (tmp_path / "beads.yml").write_text(yaml.safe_dump(config))
    edited = _workflow(tmp_path, output, "-resume")
    assert "Cached process" not in edited  # re-estimated, and so re-evaluated
    assert {r["pipeline"] for r in _rows(output)} == {"beads"}


def test_one_failing_pipeline_does_not_stop_the_others(tmp_path):
    plate = _plate(tmp_path)
    beads = _config(tmp_path / "beads.yml", "beads")
    broken = tmp_path / "broken.yml"
    broken.write_text(yaml.safe_dump({"moving": {"channel": "x"}, "method": "no-such-method"}))
    _one_case_spec(
        tmp_path,
        plate,
        [{"name": "beads", "config": beads}, {"name": "broken", "config": broken.name}],
    )
    output = tmp_path / "out"
    _workflow(tmp_path, output)
    assert {r["pipeline"] for r in _rows(output)} == {"beads"}


def test_a_stabilization_case_needs_no_reference(tmp_path):
    from tests.test_nextflow_registration import _drifting_plate

    plate = _drifting_plate(tmp_path)
    with open_ome_zarr(plate, mode="r") as position:
        channel = position.channel_names[0]
        n_t = position.data.shape[0]
    pcc = tmp_path / "pcc.yml"
    pcc.write_text(
        yaml.safe_dump(
            {
                "moving": {"channel": "placeholder"},
                "reference": {"frame": "cross", "channel": "placeholder"},
                "method": "phase-cross-corr",
                "transform": {"type": "euclidean"},
                "fallback": {"repair": None},
            }
        )
    )
    store, position_key = str(plate.parents[2]), "/".join(plate.parts[-3:])
    spec = {
        "cases": [
            {
                "name": "drift",
                "moving": store,
                "moving_channel": channel,
                "estimate_positions": position_key,
                "evaluate_positions": [position_key],
                "evaluate_t": n_t - 1,
                "tile_px": 32,
            }
        ],
        "pipelines": [{"name": "pcc", "config": pcc.name}],
    }
    (tmp_path / "benchmark.yml").write_text(yaml.safe_dump(spec))
    output = tmp_path / "out"
    _workflow(tmp_path, output)
    rows = {r["candidate"]: r for r in _rows(output)}
    assert yaml.safe_load((output / "drift" / "pcc" / "estimate.yml").read_text())[
        "reference"
    ] == {"frame": "first"}
    assert float(rows["transforms"]["tiles_median_um"]) < float(
        rows["unregistered"]["tiles_median_um"]
    )
