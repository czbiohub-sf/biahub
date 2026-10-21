"""What segment and track actually hand to cellpose 4.

Cellpose 4 ignores ``model_type`` and ``channels`` and keeps the first 3 channels of
whatever it is given, so these tests record the model and the image cellpose receives.
The test extra does not install cellpose, so a recording stand-in replaces it.
"""

import numpy as np
import pytest


def _segment(czyx, channels, **model):
    """Run _segment_czyx on ``czyx`` whose channels are named c0, c1, ..."""
    from biahub.segment import _segment_czyx, resolve_models
    from biahub.settings import SegmentationSettings

    names = [f"c{i}" for i in range(czyx.shape[0])]
    settings = SegmentationSettings(
        models={"nuc": {"channels": [names[c] for c in channels], **model}}
    )
    models = resolve_models(settings, names, scale=(1,) * 5, z_size=czyx.shape[1])
    return _segment_czyx(czyx, models, gpu=False)


def test_segment_uses_the_requested_model(fake_cellpose):
    _segment(
        np.zeros((1, 1, 8, 8), dtype=np.float32),
        channels=[0],
        pretrained_model="cpdino",
        eval_args={"diameter": None, "do_3D": False},
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
    )

    (seen,) = fake_cellpose.evaluated
    assert seen.shape[0] == 1
    assert np.all(seen == 30)


def test_segment_2d_passes_the_single_plane_without_a_z_axis(fake_cellpose):
    czyx = np.full((1, 1, 8, 8), 30, dtype=np.float32)

    _segment(
        czyx,
        channels=[0],
        pretrained_model="cpsam_v2",
        eval_args={"diameter": None, "do_3D": False},
    )

    (seen,) = fake_cellpose.evaluated
    assert seen.shape == (1, 8, 8)  # (C, Y, X): 2D input carries no z axis
    assert np.all(seen == 30)


def test_mixing_2d_and_3d_models_is_refused(fake_cellpose):
    from biahub.settings import SegmentationSettings

    with pytest.raises(ValueError, match="2D and 3D"):
        SegmentationSettings(
            models={
                "flat": {"pretrained_model": "cpsam_v2", "channels": ["GFP"]},
                "volume": {
                    "pretrained_model": "cpsam_v2",
                    "channels": ["RFP"],
                    "eval_args": {"do_3D": True},
                },
            }
        )


def test_segment_cli_rejects_a_stack_for_a_2d_model(fake_cellpose, example_plate, tmp_path):
    """Segment does not project: a 2D model needs the Z=1 plate `biahub project` writes."""
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
                    }
                }
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        # --init: the dimensionality check runs while creating the plate, before any job.
        [
            "segment",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config),
            "--init",
        ],
    )

    assert result.exit_code != 0
    assert "biahub project" in str(result.exception) + result.output
    assert not (tmp_path / "out.zarr").exists()


def test_segment_3d_passes_the_z_stack(fake_cellpose):
    czyx = np.zeros((2, 5, 8, 8), dtype=np.float32)

    _segment(czyx, channels=[0], pretrained_model="cpsam_v2", eval_args={"do_3D": True})

    (seen,) = fake_cellpose.evaluated
    assert seen.shape == (1, 5, 8, 8)


def test_a_2d_model_is_refused_a_z_stack(fake_cellpose):
    """Cellpose 4 would raise on a z stack with do_3D False and no stitching, so
    resolve_models refuses it first (segment does not pick a plane; project does)."""
    with pytest.raises(ValueError, match="input is 3D.*biahub project"):
        _segment(
            np.zeros((1, 5, 8, 8), dtype=np.float32),
            channels=[0],
            pretrained_model="cpsam_v2",
            eval_args={"do_3D": False},
        )


def test_segment_loads_each_model_once_across_timepoints(fake_cellpose):
    """process_single_position calls _segment_czyx once per timepoint; reloading the
    1.2 GB model every frame is what made segment slow and GPU-hungry."""
    from biahub.segment import _segment_czyx, resolve_models
    from biahub.settings import SegmentationSettings

    settings = SegmentationSettings(
        models={"nuc": {"pretrained_model": "cpdino", "channels": ["c0"]}}
    )
    models = resolve_models(settings, ["c0"], scale=(1,) * 5, z_size=1)
    for _ in range(3):
        _segment_czyx(np.zeros((1, 1, 8, 8), dtype=np.float32), models, gpu=False)

    assert fake_cellpose.built == ["cpdino"]
    assert len(fake_cellpose.evaluated) == 3


def test_segment_2d_restores_the_singleton_z(fake_cellpose):
    """Cellpose returns (Y, X) for a 2D model; the label plate expects (1, Y, X)."""
    out = _segment(
        np.zeros((1, 1, 8, 8), dtype=np.float32),
        channels=[0],
        pretrained_model="cpsam_v2",
        eval_args={"do_3D": False},
    )
    assert out.shape == (1, 1, 8, 8)


def test_a_3d_model_is_refused_a_single_plane(fake_cellpose):
    with pytest.raises(ValueError, match="one Z-plane but the models segment in 3D"):
        _segment(
            np.zeros((1, 1, 8, 8), dtype=np.float32),
            channels=[0],
            pretrained_model="cpsam_v2",
            eval_args={"do_3D": True},
        )
