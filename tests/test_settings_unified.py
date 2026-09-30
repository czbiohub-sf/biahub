import numpy as np
import pytest

from biahub.settings import (
    ChannelSettings,
    EstimateTransformSettings,
    ReferenceSettings,
    TransformEntry,
    TransformSettings,
    load_estimate_transform_settings,
    load_transform_settings,
)
from biahub.utils.config import model_to_yaml


def test_cross_frame_needs_a_channel_and_self_frames_forbid_one():
    with pytest.raises(ValueError, match="needs a reference channel"):
        ReferenceSettings(frame="cross")
    with pytest.raises(ValueError, match="drop reference.channel"):
        ReferenceSettings(frame="first", channel="GFP")
    stabilization = EstimateTransformSettings(
        moving=ChannelSettings(channel="GFP"),
        reference=ReferenceSettings(frame="previous"),
        method="phase-cross-corr",
    )
    assert stabilization.reference_channel == "GFP"
    assert stabilization.phase_cross_corr is not None, "the chosen method's block is filled in"
    assert stabilization.beads is None
    assert stabilization.effective_score_metric == "correlation"


def test_loader_reads_unified_configs_and_points_old_ones_at_convert_settings(tmp_path):
    unified = EstimateTransformSettings(
        moving=ChannelSettings(channel="GFP"),
        reference=ReferenceSettings(frame="cross", channel="Phase3D"),
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
    (tmp_path / "first_unified.yml").write_text(
        "direction: forward\nmatrices: []\nsource_channels: [GFP]\n"
    )
    with pytest.raises(ValueError, match="convert-settings"):
        load_transform_settings(tmp_path / "first_unified.yml")


def test_transform_settings_series_wide_entry_and_direction():
    forward = np.eye(4)
    forward[:3, 3] = [-2.0, 3.0, -4.0]
    series = TransformSettings(
        direction="forward",
        moving_channels=["GFP"],
        transforms=[TransformEntry(matrix=forward.tolist())],
    )
    assert series.series_wide and series.timepoints() is None
    np.testing.assert_allclose(series.matrix_for(17, "forward"), forward)
    np.testing.assert_allclose(series.matrix_for(17, "pull")[:3, 3], [2, -3, 4])
    with pytest.raises(ValueError, match="at least one entry"):
        TransformSettings(direction="forward", moving_channels=["GFP"], transforms=[])
