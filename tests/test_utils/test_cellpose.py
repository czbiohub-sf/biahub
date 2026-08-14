import getpass
import os
import sys
import types

import pytest
import torch

from biahub.utils.cellpose import (
    cellpose_device,
    stage_cellpose_weights,
    warm_cellpose_weights,
)


def test_cellpose_device_cpu_when_not_requested():
    assert cellpose_device(gpu=False) == torch.device("cpu")


def test_cellpose_device_raises_instead_of_falling_back(monkeypatch):
    """An unusable GPU must fail loudly, not silently segment on the CPU."""

    def unavailable(*args, **kwargs):
        raise RuntimeError("CUDA error: all CUDA-capable devices are busy or unavailable")

    monkeypatch.setattr(torch, "zeros", unavailable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    with pytest.raises(RuntimeError, match="Refusing to fall back to the CPU") as excinfo:
        cellpose_device(gpu=True)

    # The message has to carry enough to debug the node it happened on.
    assert "CUDA_VISIBLE_DEVICES='3'" in str(excinfo.value)
    assert "busy or unavailable" in str(excinfo.value)


@pytest.fixture()
def cellpose_cache(tmp_path, monkeypatch):
    """A shared weights directory and an empty node-local scratch to stage into."""
    # cellpose.models freezes its model directory at import time, so staging is a
    # no-op once it is imported -- and another test may have imported it already.
    monkeypatch.delitem(sys.modules, "cellpose.models", raising=False)

    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "cpsam_v2").write_bytes(b"cpsam_v2 weights")
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(shared))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    return shared


def test_stage_cellpose_weights_copies_to_scratch(cellpose_cache):
    dest = stage_cellpose_weights()

    assert dest is not None
    assert (dest / "cpsam_v2").read_bytes() == b"cpsam_v2 weights"
    # cellpose reads the staged copy, and nothing partial is left behind.
    assert os.environ["CELLPOSE_LOCAL_MODELS_PATH"] == str(dest)
    assert sorted(path.name for path in dest.iterdir()) == ["cpsam_v2"]

    # Staging again is a no-op now that the staging dir is the source.
    assert stage_cellpose_weights() == dest


def test_stage_cellpose_weights_replaces_truncated_copy(cellpose_cache, tmp_path):
    """A partial copy left behind by a killed task is replaced, not trusted."""
    scratch = tmp_path / "scratch" / f"cellpose-models-{getpass.getuser()}"
    scratch.mkdir(parents=True)
    (scratch / "cpsam_v2").write_bytes(b"truncated")

    dest = stage_cellpose_weights()

    assert dest == scratch
    assert (dest / "cpsam_v2").read_bytes() == b"cpsam_v2 weights"


def test_stage_cellpose_weights_skips_when_no_shared_cache(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "cellpose.models", raising=False)
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path / "never_downloaded"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))

    assert stage_cellpose_weights() is None
    # Left alone, so cellpose downloads to the shared cache as it always has.
    assert os.environ["CELLPOSE_LOCAL_MODELS_PATH"] == str(tmp_path / "never_downloaded")


def test_stage_cellpose_weights_skips_once_cellpose_is_imported(cellpose_cache, monkeypatch):
    """Redirecting after the import would not take effect, so don't pretend it did."""
    monkeypatch.setitem(sys.modules, "cellpose.models", types.ModuleType("cellpose.models"))

    assert stage_cellpose_weights() is None
    assert os.environ["CELLPOSE_LOCAL_MODELS_PATH"] == str(cellpose_cache)


def _fake_cellpose(monkeypatch, cellpose_model):
    """Stand in for the real cellpose, which the test extra does not install."""
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = cellpose_model
    package = types.ModuleType("cellpose")
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)


def test_warm_cellpose_weights_returns_the_resolved_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "models" / "cpsam_v2"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"downloaded by warming")

    class FakeModel:
        def __init__(self, gpu=False):
            # --init runs on the head node, which has no GPU to ask for.
            assert gpu is False
            self.pretrained_model = str(checkpoint)

    _fake_cellpose(monkeypatch, FakeModel)

    assert warm_cellpose_weights() == checkpoint


def test_warm_cellpose_weights_survives_an_unusable_cellpose(monkeypatch):
    """Warming is an optimisation: --init must not fail when it cannot happen."""

    class Unavailable:
        def __init__(self, gpu=False):
            raise RuntimeError("checkpoint is corrupt")

    _fake_cellpose(monkeypatch, Unavailable)

    assert warm_cellpose_weights() is None
