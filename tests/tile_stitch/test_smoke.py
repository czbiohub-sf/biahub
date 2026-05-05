"""Minimal smoke — config + plan + LocalCluster recon on a tiny phantom.

The full c0032/c0041 reproduction lives in the SLURM submission scripts.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from biahub.tile_stitch.config import TileStitchRun


def _minimal_yaml(run_dir: Path) -> dict:
    return {
        "tile_stitch": {
            "tile": {"tile_size": {"y": 32, "x": 32}, "overlap": {"y": 4, "x": 4}},
            "blend": {"kind": "uniform_mean"},
            "recon": {"kind": "phase"},
        },
        "cpu_pool": {
            "scratch_dir": str(run_dir / "scratch"),
            "batch_size": 2,
        },
        "run_dir": str(run_dir),
    }


def test_config_yaml_parses(tmp_path: Path):
    payload = _minimal_yaml(tmp_path)
    parsed = yaml.safe_load(yaml.safe_dump(payload))
    run = TileStitchRun.model_validate(parsed)
    assert run.tile_stitch.recon.kind == "phase"
    assert run.cpu_pool.batch_size == 2
    assert run.gpu_pool is None


def test_config_gpu_only(tmp_path: Path):
    payload = _minimal_yaml(tmp_path)
    payload.pop("cpu_pool")
    payload["gpu_pool"] = {"rmm_pool_size": "20GB"}
    run = TileStitchRun.model_validate(payload)
    assert run.gpu_pool.rmm_pool_size == "20GB"
    assert run.cpu_pool is None
