"""Every setting a config accepts is read: one the run would ignore is refused, saying why."""

import pytest

from pydantic import ValidationError

from biahub.settings import EstimateTransformSettings


def _settings(method="phase_cross_corr", **kwargs):
    kwargs.setdefault("reference", {"frame": "cross", "channel": "Phase3D"})
    return EstimateTransformSettings(moving={"channel": "GFP"}, method=method, **kwargs)


def test_another_methods_block_is_refused():
    with pytest.raises(ValidationError, match="ants block.*method phase_cross_corr"):
        _settings(ants={})
    # a stack's blocks carry their own settings: no top-level block is read
    with pytest.raises(ValidationError, match="belong in the blocks"):
        _settings(method=[{"phase_cross_corr": {}}, {"ants": {}}], phase_cross_corr={})
    _settings(phase_cross_corr={})  # its own block is fine


def test_the_neighbour_window_is_given_with_neighbours_and_only_there():
    # window defaulted to 6 (a hidden choice: how many frames on each side), and was
    # accepted by every frame though only neighbours reads it
    with pytest.raises(ValidationError, match="'neighbours' needs its window"):
        _settings(reference={"frame": "neighbours"})
    with pytest.raises(ValidationError, match="window is read only by frame neighbours"):
        _settings(reference={"frame": "first", "window": 4})
    assert _settings(reference={"frame": "neighbours", "window": 4}).reference.window == 4
