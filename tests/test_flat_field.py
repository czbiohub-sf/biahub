import numpy as np
import pytest

from biahub.flat_field import (
    _MEDIAN_TILE_BYTES,
    _flat_field_czyx,
    _median_tile_width,
    _median_tiled,
    flat_field_zyx,
)

# One mantis-v2 position, as flat-field sees it per (t, c) unit.
MANTIS_ZYX = (1068, 256, 1664)


def _data(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    # +1 keeps the median pattern away from zero so the division stays finite and
    # exact-equality comparisons are not comparing NaNs.
    return (rng.random(shape) * 4095 + 1).astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize(
    "shape",
    [
        (8, 6, 4),  # even reduction axis -> median averages the two middle values
        (9, 6, 4),  # odd reduction axis  -> median picks a single element
        (2, 3, 5),
        (1, 4, 4),  # degenerate reduction axis
    ],
)
def test_median_tiled_matches_numpy(shape, dtype):
    data = _data(shape, dtype)
    expected = np.median(data, axis=0)
    # A 1-byte budget forces the narrowest possible tile, so the tiling path runs
    # even for these small arrays.
    result = _median_tiled(data, axis=0, tile_bytes=1)

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("tile_bytes", [1, 64, 4096, _MEDIAN_TILE_BYTES])
def test_median_tiled_is_invariant_to_tile_size(tile_bytes):
    """Tile width is a performance knob only; it must not move the result."""
    data = _data((16, 12, 9), np.uint16)
    np.testing.assert_array_equal(
        _median_tiled(data, axis=0, tile_bytes=tile_bytes), np.median(data, axis=0)
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_median_tiled_matches_numpy_on_every_axis(axis):
    data = _data((7, 8, 9), np.uint16)
    np.testing.assert_array_equal(
        _median_tiled(data, axis=axis, tile_bytes=1), np.median(data, axis=axis)
    )


def test_median_tiled_handles_noncontiguous_input():
    """Production hands in ``czyx_data[c]``, and callers may pass strided views."""
    view = _data((8, 6, 10), np.uint16)[:, :, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(
        _median_tiled(view, axis=0, tile_bytes=1), np.median(view, axis=0)
    )


def test_median_tiled_handles_1d_input():
    data = _data((32,), np.uint16)
    assert _median_tiled(data, axis=0) == np.median(data, axis=0)


def test_flat_field_zyx_matches_reference_expression():
    data = _data((16, 12, 9), np.uint16)
    pattern = np.median(data, axis=0)
    expected = data / pattern * pattern.mean()

    np.testing.assert_array_equal(flat_field_zyx(data), expected)


def test_flat_field_czyx_corrects_only_target_channels():
    czyx = _data((3, 8, 6, 4), np.uint16)
    out = _flat_field_czyx(czyx, target_indices=[1])

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0], czyx[0].astype(np.float32))
    np.testing.assert_array_equal(out[1], flat_field_zyx(czyx[1]).astype(np.float32))
    np.testing.assert_array_equal(out[2], czyx[2].astype(np.float32))


def test_mantis_position_is_tiled_under_the_default_budget():
    """Guards the optimisation itself: a real position must not become one tile.

    Without this, a change to the budget or the width maths could silently fall
    back to a single whole-array median -- correct, but 11.5x slower on Zen 2.
    """
    # Only shape and itemsize are consulted, so a zero-stride view stands in for
    # the real 0.9 GB volume.
    data = np.lib.stride_tricks.as_strided(
        np.zeros(1, dtype=np.uint16), shape=MANTIS_ZYX, strides=(0, 0, 0)
    )
    width = _median_tile_width(data, axis=0, tile_bytes=_MEDIAN_TILE_BYTES)

    assert 1 <= width < MANTIS_ZYX[1]
    tile_nbytes = width * MANTIS_ZYX[0] * MANTIS_ZYX[2] * data.itemsize
    assert tile_nbytes <= _MEDIAN_TILE_BYTES


def test_median_tile_width_is_clamped_to_the_axis():
    data = _data((4, 3, 2), np.uint16)
    assert _median_tile_width(data, axis=0, tile_bytes=1) == 1
    assert _median_tile_width(data, axis=0, tile_bytes=10**9) == 3
