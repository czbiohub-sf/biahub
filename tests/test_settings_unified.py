import numpy as np
import pytest

from biahub.settings import (
    ChannelSettings,
    EstimateTransformSettings,
    TransformSettings,
    load_estimate_transform_settings,
    load_transform_settings,
)
from biahub.utils.config import model_to_yaml


def test_cross_reference_needs_a_target_and_self_reference_forbids_one():
    with pytest.raises(ValueError, match="needs a target"):
        EstimateTransformSettings(source=ChannelSettings(channel="GFP"), method="beads")
    with pytest.raises(ValueError, match="drop target"):
        EstimateTransformSettings(
            source=ChannelSettings(channel="GFP"),
            target=ChannelSettings(channel="GFP"),
            reference="first",
            method="beads",
        )
    stabilization = EstimateTransformSettings(
        source=ChannelSettings(channel="GFP"), reference="previous", method="phase-cross-corr"
    )
    assert stabilization.target_channel == "GFP"
    assert stabilization.phase_cross_corr is not None, "the chosen method's block is filled in"
    assert stabilization.beads is None
    assert stabilization.effective_score_metric == "correlation"


def test_loader_reads_unified_configs_and_points_legacy_ones_at_convert_settings(tmp_path):
    unified = EstimateTransformSettings(
        source=ChannelSettings(channel="GFP"),
        target=ChannelSettings(channel="Phase3D"),
        method="beads",
        score_metric="residual",
    )
    model_to_yaml(unified, tmp_path / "unified.yml")
    assert load_estimate_transform_settings(tmp_path / "unified.yml") == unified

    (tmp_path / "legacy.yml").write_text(
        "source_channel_name: GFP\ntarget_channel_name: Phase3D\nestimation_method: beads\n"
    )
    with pytest.raises(ValueError, match="convert-settings"):
        load_estimate_transform_settings(tmp_path / "legacy.yml")
    (tmp_path / "stabilize.yml").write_text(
        "stabilization_estimation_channel: GFP\naffine_transform_zyx_list: []\n"
    )
    with pytest.raises(ValueError, match="convert-settings"):
        load_transform_settings(tmp_path / "stabilize.yml")


def test_transform_settings_as_direction_inverts_when_asked():
    forward = np.eye(4)
    forward[:3, 3] = [-2.0, 3.0, -4.0]
    series = TransformSettings(
        direction="forward", matrices=[forward.tolist()] * 3, source_channels=["GFP"]
    )
    assert series.as_direction("forward") == [forward.tolist()] * 3
    np.testing.assert_allclose(
        np.asarray(series.as_direction("pull"))[:, :3, 3], [[2, -3, 4]] * 3
    )
    with pytest.raises(ValueError, match="4x4"):
        TransformSettings(direction="forward", matrices=[], source_channels=["GFP"])
