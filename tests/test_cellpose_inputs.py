"""What segment and track actually hand to cellpose 4.

Cellpose 4 ignores ``model_type`` and ``channels`` and keeps the first 3 channels of
whatever it is given, so these tests record the model and the image cellpose receives.
The test extra does not install cellpose, so a recording stand-in replaces it.
"""

import sys
import types
import warnings

import numpy as np
import pytest


class RecordingModel:
    """Stand-in for cellpose.models.CellposeModel that records how it is used."""

    built = []
    evaluated = []

    def __init__(self, gpu=False, pretrained_model="cpsam_v2", model_type=None, device=None):
        # Like cellpose 4: model_type is accepted and ignored.
        self.pretrained_model, self.device = pretrained_model, device
        RecordingModel.built.append(pretrained_model)

    def eval(
        self,
        x,
        channels=None,
        channel_axis=None,
        z_axis=None,
        diameter=None,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        do_3D=False,
        anisotropy=None,
        min_size=15,
    ):
        RecordingModel.evaluated.append(np.array(x))
        shape = x.shape[1:] if channel_axis == 0 else x.shape
        return np.ones(shape, dtype=np.uint16), None, None


@pytest.fixture
def fake_cellpose(monkeypatch, tmp_path):
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = RecordingModel
    models.MODEL_NAMES = ["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb"]
    models.get_user_models = lambda: []
    package = types.ModuleType("cellpose")
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path / "no-weights"))
    RecordingModel.built, RecordingModel.evaluated = [], []
    return RecordingModel


def test_track_cellpose_uses_the_requested_model(fake_cellpose):
    from biahub.track import run_cellpose_per_frame

    labels = run_cellpose_per_frame(
        np.zeros((2, 8, 8), dtype=np.float32), pretrained_model="cpdino", gpu=False
    )

    assert fake_cellpose.built == ["cpdino"]
    assert labels.shape == (2, 8, 8)


def test_model_type_in_a_track_config_warns():
    from biahub.settings import CellposeConfig

    with pytest.warns(UserWarning, match="model_type"):
        config = CellposeConfig(model_type="nuclei")
    assert config.pretrained_model == "cpsam_v2"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CellposeConfig()
