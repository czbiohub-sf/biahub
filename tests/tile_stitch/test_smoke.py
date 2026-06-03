"""Config smoke — MonarchConfig defaults + cross-field validators.

The full c0032/c0041 reproduction lives in the SLURM submission scripts; the
GPU/Monarch path is exercised by ``test_monarch_smoke.py`` (skips in CI).
"""

from __future__ import annotations

import warnings

from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError

from biahub.tile_stitch.config import (
    CompileMode,
    Device,
    MonarchConfig,
    TileStitchRun,
)


def _minimal_yaml(run_dir: Path) -> dict:
    return {
        "tile_stitch": {
            "tile": {"tile_size": {"y": 32, "x": 32}, "overlap": {"y": 4, "x": 4}},
            "blend": {"kind": "uniform_mean"},
            "recon": {
                "input_channel_names": ["c"],
                "reconstruction_dimension": 3,
                "phase": {},
            },
        },
        "run_dir": str(run_dir),
    }


def test_config_yaml_parses_with_monarch_defaults(tmp_path: Path):
    payload = _minimal_yaml(tmp_path)
    parsed = yaml.safe_load(yaml.safe_dump(payload))
    run = TileStitchRun.model_validate(parsed)
    assert run.tile_stitch.blend.kind == "uniform_mean"
    # monarch block omitted → default_factory supplies bench-best defaults.
    assert run.monarch.recon_batch == 4
    assert run.monarch.prefetch_depth == 6
    assert run.monarch.device is Device.CUDA
    assert run.monarch.compile_mode is CompileMode.REDUCE_OVERHEAD
    # StrEnum members are plain strings on the wire / in comparisons.
    assert run.monarch.device == "cuda"
    assert run.monarch.compile_mode == "reduce-overhead"


def test_config_monarch_overrides(tmp_path: Path):
    payload = _minimal_yaml(tmp_path)
    payload["monarch"] = {"gpus_per_node": 4, "recon_batch": 1, "compile_mode": "none"}
    run = TileStitchRun.model_validate(payload)
    assert run.monarch.gpus_per_node == 4
    assert run.monarch.recon_batch == 1
    assert run.monarch.compile_mode is CompileMode.NONE


def test_default_config_is_valid_and_quiet():
    # resident=False, recon_batch=4, prefetch_depth=6 trips neither validator.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = MonarchConfig()
    assert cfg.recon_batch == 4


def test_resident_volume_with_batch_rejected():
    with pytest.raises(ValueError, match="resident_volume=True is incompatible"):
        MonarchConfig(resident_volume=True, recon_batch=2)


def test_resident_volume_with_batch1_ok():
    cfg = MonarchConfig(resident_volume=True, recon_batch=1)
    assert cfg.resident_volume is True


def test_prefetch_below_batch_warns():
    # warn != raise — the model must still construct with the given values.
    with pytest.warns(UserWarning, match="prefetch_depth .* < recon_batch"):
        cfg = MonarchConfig(prefetch_depth=1, recon_batch=4)
    assert cfg.prefetch_depth == 1
    assert cfg.recon_batch == 4


def test_prefetch_zero_does_not_warn():
    # 0 disables prefetch entirely — not a partial-overlap footgun.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = MonarchConfig(prefetch_depth=0, recon_batch=4)
    assert cfg.prefetch_depth == 0


def test_monarch_config_is_frozen():
    cfg = MonarchConfig()
    with pytest.raises(ValidationError):  # frozen → mutation rejected
        cfg.recon_batch = 8
