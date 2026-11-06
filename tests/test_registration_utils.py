"""Registration helpers: shape matching."""

import numpy as np

from biahub.registration.utils import match_shape, match_shape_origin


def test_match_shape_pads_and_crops_each_axis_on_its_own():
    # one axis smaller, two larger: tuples compare lexicographically, so the per-axis
    # decision once read (8, 20, 20) < (10, 10, 10) as "smaller" and failed
    img = np.zeros((10, 10, 10))
    img[3, 4, 5] = 1.0
    out = match_shape(img, (8, 20, 20))
    assert out.shape == (8, 20, 20)
    # the input voxel lands where match_shape_origin says (reflect padding copies it too)
    landed = np.array([3, 4, 5]) + match_shape_origin((10, 10, 10), (8, 20, 20))
    assert out[tuple(landed)] == 1.0
    assert match_shape(img, (10, 10, 10)) is img  # nothing to do, no copy


def test_the_centred_window_takes_whole_axes_when_the_crop_is_larger_and_keeps_odd_sizes():
    from biahub.registration.utils import center_window_yx

    # crop_xy is (X, Y), the legacy center_crop_xy convention; slices are (y, x)
    assert center_window_yx((100, 80), (40, 21)) == (slice(40, 61), slice(20, 60))
    assert center_window_yx((100, 80), (150, 150)) == (slice(0, 100), slice(0, 80))
