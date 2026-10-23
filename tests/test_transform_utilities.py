"""Small transform helpers the registration agents kept rewriting."""

import numpy as np
import pytest

from biahub.core.transform import Transform
from biahub.settings import TransformEntry, TransformSettings

UM = (0.2, 0.1, 0.1)


def test_how_far_apart_two_transforms_put_the_volume():
    shape = (10, 101, 101)
    a = Transform.from_translation([0.0, 0.0, 0.0])
    b = Transform.from_translation([0.0, 3.0, 4.0])  # 5 voxels in yx, 0.5 µm
    d = a.displacement_to(b, shape, UM)
    assert d["max_um"] == pytest.approx(0.5) and d["mean_um"] == pytest.approx(0.5)

    # a rotation about the centre: nothing at the centre, most at the corners
    angle = np.deg2rad(1.0)
    m = np.eye(4)
    m[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    c = np.array([4.5, 50.0, 50.0])
    m[:3, 3] = c - m[:3, :3] @ c
    d = a.displacement_to(Transform(m), shape, UM)
    corner = np.hypot(50, 50) * np.deg2rad(1.0) * 0.1  # radius x angle, in µm
    assert d["max_um"] == pytest.approx(corner, rel=1e-3)
    assert 0 < d["mean_um"] < d["max_um"]


def _file(direction="forward", matrix=None):
    matrix = Transform.from_translation([1.0, 2.0, 3.0]).matrix if matrix is None else matrix
    return TransformSettings(
        direction=direction,
        moving_channels=["GFP"],
        transforms=[TransformEntry(t=0, matrix=np.asarray(matrix).tolist())],
    )


def test_a_transforms_file_gives_its_forward_transform_whichever_way_it_is_stored():
    forward = Transform.from_translation([1.0, 2.0, 3.0])
    stored_inverse = _file("inverse", np.linalg.inv(forward.matrix))
    for settings in (_file("forward"), stored_inverse):
        np.testing.assert_allclose(settings.transform_for(0).matrix, forward.matrix)


def test_an_entry_status_and_note_change_with_a_reason():
    settings = _file()
    settings.set_entry(0, status="rejected", note="corners off by 2 µm in the visual check")
    entry = settings.entry_for(0)
    assert (entry.status, entry.note) == (
        "rejected",
        "corners off by 2 µm in the visual check",
    )
    with pytest.raises(ValueError, match="no entry for t=5"):
        settings.set_entry(5, status="accepted", note="x")
    with pytest.raises(ValueError, match="note"):
        settings.set_entry(0, status="accepted")  # a change of status says why
