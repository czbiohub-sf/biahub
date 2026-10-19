"""Projection config schema and Z-plane resolution (`biahub project`)."""

import numpy as np
import pytest

from biahub.project import resolve_z_indices
from biahub.settings import ChannelProjection, ProjectionSettings


def _channel(method="max", **z):
    return ChannelProjection(method=method, z=z or None)


# ---------------------------------------------------------------------------
# resolve_z_indices
# ---------------------------------------------------------------------------


def test_unset_z_is_the_whole_stack():
    assert np.array_equal(resolve_z_indices(_channel(), 5), np.arange(5))


def test_absolute_start_stop_with_step():
    indices = resolve_z_indices(_channel(start=10, stop=50, step=2), 60)
    assert indices.tolist() == list(range(10, 50, 2))


def test_window_is_below_and_above_a_center():
    indices = resolve_z_indices(_channel(center=30, below=20, above=40), 100)
    assert indices.tolist() == list(range(10, 70))  # 60 planes, 20 below the center


def test_focus_center_uses_the_given_plane():
    indices = resolve_z_indices(_channel(center="focus", below=2, above=3), 20, focus_z=5)
    assert indices.tolist() == [3, 4, 5, 6, 7]
    with pytest.raises(ValueError, match="no focus plane"):
        resolve_z_indices(_channel(center="focus", below=2, above=3), 20)


def test_slice_is_one_plane():
    assert resolve_z_indices(_channel("slice", center=42), 60).tolist() == [42]
    assert resolve_z_indices(_channel("slice", center="focus"), 60, focus_z=7).tolist() == [7]


def test_selections_past_the_stack_are_clipped():
    assert resolve_z_indices(_channel(center=2, below=5, above=5), 10).tolist() == list(
        range(7)
    )
    assert resolve_z_indices(_channel(start=8, stop=50), 10).tolist() == [8, 9]


def test_a_selection_outside_the_stack_is_an_error():
    with pytest.raises(ValueError, match="select nothing"):
        resolve_z_indices(_channel("slice", center=42), 10)
    with pytest.raises(ValueError, match="select nothing"):
        resolve_z_indices(_channel(start=10, stop=20), 10)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_slice_needs_a_center_and_takes_no_window():
    with pytest.raises(ValueError, match="slice needs z.center"):
        _channel("slice")
    with pytest.raises(ValueError, match="takes no below/above"):
        _channel("slice", center=3, below=1, above=1)


@pytest.mark.parametrize("method", ["max", "mean", "sum"])
def test_a_center_needs_a_window_for_reductions(method):
    with pytest.raises(ValueError, match="needs z.below/above"):
        _channel(method, center=3)


def test_start_and_stop_go_together_and_are_not_mixed_with_a_center():
    with pytest.raises(ValueError, match="start and stop go together"):
        _channel(start=5)
    with pytest.raises(ValueError, match="is empty"):
        _channel(start=5, stop=5)
    with pytest.raises(ValueError, match="absolute planes"):
        _channel(center=3, start=0, stop=5)
    with pytest.raises(ValueError, match="absolute planes"):
        _channel(start=0, stop=5, below=1, above=1)


def test_below_and_above_go_together_and_need_a_center():
    with pytest.raises(ValueError, match="below and above go together"):
        _channel(center=3, below=2)
    with pytest.raises(ValueError, match="need a center"):
        _channel(below=2, above=2)
    with pytest.raises(ValueError, match="select no plane"):
        _channel(center=3, below=0, above=0)


def test_step_needs_a_selection():
    with pytest.raises(ValueError, match="step needs"):
        _channel(step=2)


def test_focus_block_is_required_by_focus_users():
    with pytest.raises(ValueError, match="use z.center: focus"):
        ProjectionSettings(channels={"a": {"method": "slice", "z": {"center": "focus"}}})
    s = ProjectionSettings(
        channels={"a": {"method": "slice", "z": {"center": "focus"}}},
        focus={"channel": "Phase3D"},
    )
    assert s.uses_focus
    assert s.focus.per == "fov"
    assert s.focus.center_crop_xy == [800, 800]  # FocusSettings' default crop


def test_an_unused_focus_block_warns():
    with pytest.warns(UserWarning, match="ignored"):
        s = ProjectionSettings(channels={"a": {"method": "max"}}, focus={"channel": "x"})
    assert not s.uses_focus


def test_focus_has_no_axes():
    with pytest.raises(ValueError, match="axes"):
        ProjectionSettings(
            channels={"a": {"method": "slice", "z": {"center": "focus"}}},
            focus={"channel": "x", "axes": "xyz"},
        )


def test_unknown_keys_are_refused():
    with pytest.raises(ValueError, match="Extra inputs"):
        ProjectionSettings(channels={"a": {"method": "max", "mode": "max"}})
    with pytest.raises(ValueError, match="Extra inputs"):
        ProjectionSettings(channels={"a": {"method": "max", "z": {"range": [0, 3]}}})
