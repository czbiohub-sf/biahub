"""Pooling per-position estimates into one shared transform (the `pool:` setting)."""

import numpy as np
import pytest

from biahub.core.transform import Transform

SHAPE, UM = (20, 256, 256), (0.2, 0.1, 0.1)


def _estimate(z, y, x=4.0):
    return Transform.from_translation([z, y, x])


# yx agree to a few tenths of a voxel, z differs by FOV (DragonFly): A/1/0, at the
# median depth, is closest to the others
ESTIMATES = {
    "A/1/0": _estimate(8.0, 3.0),
    "A/2/0": _estimate(6.0, 3.1),
    "B/1/0": _estimate(12.0, 2.9),
    "B/2/0": _estimate(7.0, 3.0),
    "C/1/0": _estimate(9.0, 3.4),
}
# without A/1/0: B/2/0 and C/1/0 are as close in z; C/1/0's y sets it further off
WITHOUT_A = {k: v for k, v in ESTIMATES.items() if k != "A/1/0"}


def test_the_medoid_is_the_estimate_closest_to_all_the_others():
    from biahub.registration.evaluate import pool_transforms

    pooled = pool_transforms(ESTIMATES, SHAPE, UM)
    assert pooled["medoid"] == "A/1/0"
    assert pooled["transform"].to_list() == ESTIMATES["A/1/0"].to_list()
    assert pooled["spread"]["choice_z"] == "per position"
    # median over the others of their gap to A/1/0: z 1, 1, 2, 4 slices -> 1.5 x 0.2 µm
    assert pooled["gap_to_medoid_um"]["z"] == pytest.approx(0.3)
    assert pooled["gap_to_medoid_um"]["yx"] == pytest.approx(0.01, abs=0.005)


def test_median_z_keeps_the_medoids_yx_and_takes_the_median_depth():
    from biahub.registration.evaluate import pool_transforms

    pooled = pool_transforms(WITHOUT_A, SHAPE, UM, z="median")
    assert pooled["medoid"] == "B/2/0"
    matrix = np.asarray(pooled["transform"].matrix)
    assert matrix[0, 3] == pytest.approx(8.0)  # median of 6, 7, 9, 12 slices
    assert matrix[1:3, 3] == pytest.approx([3.0, 4.0])  # B/2/0's yx
    assert pooled["z_shift_um"] == pytest.approx(1.0 * 0.2)
