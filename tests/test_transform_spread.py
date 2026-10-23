"""How many transforms a dataset needs: one, per position, per timepoint, or both."""

import pytest

from biahub.core.transform import Transform

SHAPE, UM = (20, 512, 512), (0.2, 0.1, 0.1)  # half a voxel in yx: 0.05 µm
POSITIONS, TIMES = ["A/1/0", "A/2/0", "B/1/0"], [0, 10, 20]


def _estimates(per_position=0.0, per_time=0.0):
    """Translations in x that differ by `per_position` voxels between positions and
    `per_time` voxels between timepoints (drift)."""
    return {
        (p, t): Transform.from_translation([0.0, 0.0, 5.0 + i * per_position + j * per_time])
        for i, p in enumerate(POSITIONS)
        for j, t in enumerate(TIMES)
    }


@pytest.mark.parametrize(
    "per_position, per_time, choice",
    [
        (0.1, 0.1, "one"),
        (3.0, 0.1, "per position"),
        (0.1, 3.0, "per timepoint"),
        (3.0, 3.0, "per position and timepoint"),
    ],
)
def test_the_spread_says_how_many_transforms(per_position, per_time, choice):
    from biahub.registration.evaluate import transform_spread

    spread = transform_spread(_estimates(per_position, per_time), SHAPE, UM)
    assert spread["choice"] == choice
    # the largest gap between positions at one timepoint, and between timepoints at one
    # position, in µm (two steps apart at most: 2 x step x 0.1 µm)
    assert spread["across_positions_um"] == pytest.approx(2 * per_position * 0.1)
    assert spread["across_timepoints_um"] == pytest.approx(2 * per_time * 0.1)


def test_the_tolerance_can_be_set_and_the_worst_pair_is_named():
    from biahub.registration.evaluate import transform_spread

    estimates = _estimates(per_position=3.0)
    spread = transform_spread(estimates, SHAPE, UM, tolerance_um=1.0)
    assert spread["choice"] == "one"  # 0.6 µm apart, under a 1 µm tolerance
    assert set(spread["worst_positions"]) == {"A/1/0", "B/1/0"}


def test_one_sample_on_an_axis_cannot_show_spread_there():
    from biahub.registration.evaluate import transform_spread

    one_time = {k: v for k, v in _estimates(3.0, 3.0).items() if k[1] == 0}
    spread = transform_spread(one_time, SHAPE, UM)
    assert spread["across_timepoints_um"] is None
    assert spread["choice"] == "per position"
