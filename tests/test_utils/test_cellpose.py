import getpass
import os
import sys
import types

import pytest
import torch

from biahub.utils.cellpose import (
    cellpose_device,
    load_cellpose_model,
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


def _no_model(*args, **kwargs):
    raise AssertionError("warming must not build the network on the head node")


def test_warm_cellpose_weights_returns_the_resolved_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "models" / "cpdino"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"downloaded by warming")
    checked = []

    def cache_model_path(name):
        assert name == "cpdino"
        return str(checkpoint)

    _fake_cellpose(monkeypatch, _no_model)
    sys.modules["cellpose.models"].cache_model_path = cache_model_path
    sys.modules["cellpose.models"].get_backbone = checked.append

    assert warm_cellpose_weights("cpdino") == checkpoint
    assert checked == [str(checkpoint)]


def test_warm_cellpose_weights_uses_a_path_as_is(tmp_path, monkeypatch):
    checkpoint = tmp_path / "my_model"
    checkpoint.write_bytes(b"user weights")

    def cache_model_path(name):
        raise AssertionError("a path must not be resolved into the weights directory")

    _fake_cellpose(monkeypatch, _no_model)
    sys.modules["cellpose.models"].cache_model_path = cache_model_path
    sys.modules["cellpose.models"].get_backbone = lambda path: "sam_vitl"

    assert warm_cellpose_weights(str(checkpoint)) == checkpoint


def test_warm_cellpose_weights_survives_an_unusable_cellpose(tmp_path, monkeypatch):
    """Warming is an optimisation: --init must not fail when it cannot happen."""

    def corrupt(path):
        raise RuntimeError("PytorchStreamReader failed reading zip archive")

    _fake_cellpose(monkeypatch, _no_model)
    sys.modules["cellpose.models"].cache_model_path = lambda name: str(tmp_path / name)
    sys.modules["cellpose.models"].get_backbone = corrupt

    assert warm_cellpose_weights() is None


def test_load_cellpose_model_rejects_unknown_names(monkeypatch):
    """Cellpose 4 would silently run cpsam_v2 for a cellpose 3 name such as nuclei."""

    def unreachable(**kwargs):
        raise AssertionError("must not build a model for an unknown name")

    _fake_cellpose(monkeypatch, unreachable)
    sys.modules["cellpose.models"].MODEL_NAMES = ["cpsam_v2", "cpdino"]
    sys.modules["cellpose.models"].get_user_models = lambda: []

    with pytest.raises(ValueError, match="Unknown cellpose model 'nuclei'"):
        load_cellpose_model("nuclei", torch.device("cpu"))


def test_load_cellpose_model_passes_the_requested_model(monkeypatch):
    built = {}

    class FakeModel:
        def __init__(self, pretrained_model, gpu, device):
            built.update(pretrained_model=pretrained_model, gpu=gpu)
            self.pretrained_model, self.device = pretrained_model, device

    _fake_cellpose(monkeypatch, FakeModel)
    sys.modules["cellpose.models"].MODEL_NAMES = ["cpsam_v2", "cpdino"]
    sys.modules["cellpose.models"].get_user_models = lambda: []

    load_cellpose_model("cpdino", torch.device("cpu"))
    assert built == {"pretrained_model": "cpdino", "gpu": False}
