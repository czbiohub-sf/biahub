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
