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
        stitch_threshold=0.0,
    ):
        # Cellpose 4 refuses a z axis for 2D processing (cellpose/transforms.py:491).
        if z_axis is not None and not do_3D and not stitch_threshold:
            raise ValueError("2D image processing selected, but z_axis is not None.")
        RecordingModel.evaluated.append(np.array(x))
        # Like cellpose: masks drop the channel axis and come back squeezed.
        shape = x.shape[1:] if channel_axis == 0 else x.shape
        return np.ones(shape, dtype=np.uint16).squeeze(), None, None


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
    from biahub import segment

    segment._MODEL_CACHE.clear()  # models are cached per process
    return RecordingModel


def _segment(czyx, channels, **model):
    """Run segment_data on ``czyx`` whose channels are named c0, c1, ..."""
    from biahub.segment import resolve_models, segment_data
    from biahub.settings import SegmentationSettings

    names = [f"c{i}" for i in range(czyx.shape[0])]
    settings = SegmentationSettings(
        models={"nuc": {"channels": [names[c] for c in channels], **model}}
    )
    models = resolve_models(settings, names, scale=(1,) * 5, z_size=czyx.shape[1])
    return segment_data(czyx, models, gpu=False)


def test_segment_uses_the_requested_model(fake_cellpose):
    _segment(
        np.zeros((1, 1, 8, 8), dtype=np.float32),
        channels=[0],
        pretrained_model="cpdino",
        eval_args={"diameter": None, "do_3D": False},
        z_slice_2D=0,
    )

    assert fake_cellpose.built == ["cpdino"]


def test_segment_passes_only_the_configured_channel(fake_cellpose):
    # Three channels with distinct constant values; the config asks for channel 2 only
    # (segment_cli has already turned the channel name into this index).
    czyx = np.stack([np.full((1, 8, 8), c, dtype=np.float32) for c in (10, 20, 30)])

    _segment(
        czyx,
        channels=[2],
        pretrained_model="cpsam_v2",
        eval_args={"diameter": None, "do_3D": False},
        z_slice_2D=0,
    )

    (seen,) = fake_cellpose.evaluated
    assert seen.shape[0] == 1
    assert np.all(seen == 30)


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


def test_segment_slices_the_configured_plane(fake_cellpose):
    # Channel 0 holds the plane index (x10) so the plane cellpose receives is visible.
    czyx = np.stack([np.full((1, 8, 8), 10 * z, dtype=np.float32) for z in range(6)], axis=1)

    _segment(
        czyx,
        channels=[0],
        pretrained_model="cpsam_v2",
        eval_args={"diameter": None, "do_3D": False},
        z_slice_2D=3,
    )

    (seen,) = fake_cellpose.evaluated
    assert seen.shape == (1, 8, 8)  # (C, Y, X): 2D input carries no z axis
    assert np.all(seen == 30)


def test_mixing_2d_and_3d_models_is_refused(fake_cellpose):
    from biahub.settings import SegmentationSettings

    with pytest.raises(ValueError, match="2D and 3D"):
        SegmentationSettings(
            models={
                "flat": {
                    "pretrained_model": "cpsam_v2",
                    "channels": ["GFP"],
                    "z_slice_2D": 1,
                },
                "volume": {
                    "pretrained_model": "cpsam_v2",
                    "channels": ["RFP"],
                    "eval_args": {"do_3D": True},
                },
            }
        )


def test_segment_cli_rejects_a_plane_outside_the_stack(fake_cellpose, example_plate, tmp_path):
    import yaml

    from click.testing import CliRunner

    from biahub.cli.main import cli

    plate_path, _ = example_plate  # Z = 4
    config = tmp_path / "segment.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "nuc": {
                        "pretrained_model": "cpsam_v2",
                        "channels": ["GFP"],
                        "eval_args": {"do_3D": False},
                        "z_slice_2D": 10,
                    }
                }
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        # --local: never submit to SLURM from a test, even if the check is missing.
        [
            "segment",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config),
            "--local",
        ],
    )

    assert result.exit_code != 0
    assert "z_slice_2D" in str(result.exception) + result.output
    assert not (tmp_path / "out.zarr").exists()


def test_segment_3d_passes_the_z_stack(fake_cellpose):
    czyx = np.zeros((2, 5, 8, 8), dtype=np.float32)

    _segment(czyx, channels=[0], pretrained_model="cpsam_v2", eval_args={"do_3D": True})

    (seen,) = fake_cellpose.evaluated
    assert seen.shape == (1, 5, 8, 8)


def test_3d_model_without_do_3d_is_refused(fake_cellpose):
    """Cellpose 4 would raise on a z stack with do_3D False and no stitching."""
    from biahub.settings import SegmentationModel

    with pytest.raises(ValueError, match="do_3D"):
        SegmentationModel(pretrained_model="cpsam_v2", channels=["GFP"])


def test_segment_loads_each_model_once_across_timepoints(fake_cellpose):
    """process_single_position calls segment_data once per timepoint; reloading the
    1.2 GB model every frame is what made segment slow and GPU-hungry."""
    from biahub.segment import resolve_models, segment_data
    from biahub.settings import SegmentationSettings

    settings = SegmentationSettings(
        models={"nuc": {"pretrained_model": "cpdino", "channels": ["c0"], "z_slice_2D": 0}}
    )
    models = resolve_models(settings, ["c0"], scale=(1,) * 5, z_size=1)
    for _ in range(3):
        segment_data(np.zeros((1, 1, 8, 8), dtype=np.float32), models, gpu=False)

    assert fake_cellpose.built == ["cpdino"]
    assert len(fake_cellpose.evaluated) == 3
