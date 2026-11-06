"""estimate-crop on one position: where both channels have data, inside the phase circle."""

import numpy as np

from iohub import open_ome_zarr

from biahub.estimate_crop import estimate_crop_one_position


def _position(path, volume_tczyx):
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["c"]) as plate:
        plate.create_position("A", "1", "0").create_image("0", volume_tczyx)
    return path / "A" / "1" / "0"


def test_the_phase_circle_is_centred_on_a_non_square_field(tmp_path):
    shape = (2, 1, 4, 40, 80)  # y 40, x 80
    phase = _position(tmp_path / "lf.zarr", np.ones(shape, np.float32))
    fluor = _position(tmp_path / "ls.zarr", np.ones(shape, np.float32))
    z, y, x = estimate_crop_one_position(phase, fluor, lf_mask_radius=1.0)
    # the circle (radius 20, the shorter half-width) sits at the field's centre (y 20, x 40)
    assert abs((y[0] + y[1]) / 2 - 20) <= 1 and abs((x[0] + x[1]) / 2 - 40) <= 1
    assert x[1] - x[0] <= 41 and z == [0, 4]


def test_no_valid_data_keeps_the_whole_common_volume(tmp_path):
    phase = _position(tmp_path / "lf.zarr", np.zeros((2, 1, 4, 30, 30), np.float32))
    fluor = _position(tmp_path / "ls.zarr", np.zeros((2, 1, 4, 30, 30), np.float32))
    assert estimate_crop_one_position(phase, fluor) == ([0, 4], [0, 30], [0, 30])
