"""Segmentation config schema and resolving models against an input plate."""

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def fake_cellpose_signature(monkeypatch):
    """eval_args are checked against CellposeModel.eval; the test extra lacks cellpose."""

    class Model:
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
            pass

    models = types.ModuleType("cellpose.models")
    models.CellposeModel = Model
    package = types.ModuleType("cellpose")
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)


def _settings(**model):
    from biahub.settings import SegmentationSettings

    base = {
        "pretrained_model": "cpsam_v2",
        "channels": ["nuc"],
        "z_slice_2D": 2,
        "eval_args": {"diameter": None},
    }
    return SegmentationSettings(models={"nucleus": {**base, **model}})


def test_new_schema_is_accepted():
    s = _settings()
    m = s.models["nucleus"]
    assert m.pretrained_model == "cpsam_v2"
    assert m.channels == ["nuc"]


def test_old_path_to_model_key_says_what_to_rename():
    from biahub.settings import SegmentationSettings

    with pytest.raises(ValueError, match="path_to_model.*pretrained_model"):
        SegmentationSettings(
            models={
                "nucleus": {
                    "path_to_model": "cpsam_v2",
                    "channels": ["nuc"],
                    "z_slice_2D": 0,
                    "eval_args": {},
                }
            }
        )


def test_channels_inside_eval_args_says_where_they_moved():
    with pytest.raises(ValueError, match="eval_args.channels"):
        _settings(eval_args={"channels": ["nuc"]})


@pytest.mark.parametrize("key", ["channel_axis", "z_axis"])
def test_axes_are_set_by_biahub_not_the_config(key):
    with pytest.raises(ValueError, match=f"{key} is set by biahub"):
        _settings(eval_args={key: 0})


def test_at_most_three_channels():
    with pytest.raises(ValueError, match="at most 3"):
        _settings(channels=["a", "b", "c", "d"])


def test_unknown_model_keys_are_refused():
    with pytest.raises(ValueError, match="(?s)typo_key.*Extra inputs are not permitted"):
        _settings(typo_key=1)


def test_resolve_models_maps_names_without_mutating_the_settings():
    from biahub.segment import resolve_models

    s = _settings(
        channels=["mem", "nuc"], preprocessing=[{"function": "numpy.sqrt", "channel": "mem"}]
    )
    before = s.model_dump(mode="json")

    (m,) = resolve_models(
        s, channel_names=["BF", "nuc", "mem"], scale=(1, 1, 0.5, 0.1, 0.1), z_size=5
    )

    assert m.name == "nucleus"
    assert m.channel_indices == [2, 1]
    assert m.preprocessing[0][1] == 2
    assert m.z_slice_2D == 2
    assert s.model_dump(mode="json") == before


def test_resolve_models_fills_anisotropy_for_3d_only():
    from biahub.segment import resolve_models

    s3d = _settings(z_slice_2D=None, eval_args={"do_3D": True})
    (m,) = resolve_models(s3d, channel_names=["nuc"], scale=(1, 1, 0.5, 0.1, 0.1), z_size=5)
    assert m.eval_args["anisotropy"] == pytest.approx(5.0)

    (m,) = resolve_models(
        _settings(), channel_names=["nuc"], scale=(1, 1, 0.5, 0.1, 0.1), z_size=5
    )
    assert "anisotropy" not in m.eval_args


def test_resolve_models_rejects_unknown_channels_and_planes():
    from biahub.segment import resolve_models

    with pytest.raises(ValueError, match="not in the input"):
        resolve_models(
            _settings(channels=["missing"]), channel_names=["nuc"], scale=(1,) * 5, z_size=5
        )
    with pytest.raises(ValueError, match="z_slice_2D"):
        resolve_models(
            _settings(z_slice_2D=9), channel_names=["nuc"], scale=(1,) * 5, z_size=5
        )
