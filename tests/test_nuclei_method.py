"""The nuclei block: match nucleus centres between the channels and fit yx plus a z offset."""

import numpy as np
import pytest

from scipy.ndimage import affine_transform, gaussian_filter

from biahub.core.transform import Transform

UM = (0.2, 0.103, 0.103)


def _nuclei_layer(shape=(24, 512, 512), seed=0, layout="grid"):
    """Nucleus-sized disks (3 µm radius) in one cell layer, on a jittered grid or scattered."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, np.float32)
    yy, xx = np.ogrid[: shape[1], : shape[2]]
    depth = np.exp(-0.5 * ((np.arange(shape[0]) - 12) / 3.0) ** 2)[:, None]
    if layout == "grid":  # ~75 px apart: no merged nuclei
        steps = np.arange(55, shape[1] - 45, 75)
        centres = [
            np.array([gy, gx]) + rng.uniform(-12, 12, 2) for gy in steps for gx in steps
        ]
    else:  # scattered, as cells: no period for a wrong match to lock onto
        centres = []
        for _ in range(2000):
            c = rng.uniform(45, shape[1] - 45, 2)
            if all(np.hypot(*(c - o)) > 70 for o in centres):
                centres.append(c)
    for cy, cx in centres:
        vol[:, ((yy - cy) ** 2 + (xx - cx) ** 2) <= 29**2] += 100.0 * depth
    return gaussian_filter(vol, (1, 2, 2)) + rng.normal(0, 1, shape).astype(np.float32) + 10


def _forward(
    scale_yx=0.95, angle_deg=0.6, shift_vox=(2.0, 15.0, -10.0), centre=(12, 256, 256)
):
    """A forward map moving -> reference: yx scale and rotation about the centre, a shift."""
    a = np.deg2rad(angle_deg)
    linear = np.eye(3)
    linear[1:, 1:] = scale_yx * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.asarray(centre, float)
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = c - linear @ c + np.asarray(shift_vox)
    return Transform(matrix)


def _pair(truth, layout="grid"):
    ref = _nuclei_layer(layout=layout)
    # content at moving p sits at reference truth(p): sample the reference there
    m = np.asarray(truth.matrix)
    mov = affine_transform(ref, m[:3, :3], offset=m[:3, 3], order=1, mode="nearest")
    return mov.astype(np.float32), ref


def _estimator(**settings):
    from biahub.registration.methods.nuclei import NucleiEstimator
    from biahub.settings import NucleiSettings

    return NucleiEstimator(NucleiSettings(**settings), "similarity", UM, UM)


def test_nuclei_recover_a_camera_scale_rotation_and_shift():
    truth = _forward()
    mov, ref = _pair(truth)
    found = _estimator().estimate(mov, ref, seed=Transform.identity(3))
    probe = np.array(
        [[12, 60, 60], [12, 60, 450], [12, 450, 60], [12, 450, 450], [12, 256, 256]]
    )
    err_um = (found.apply_points(probe) - truth.apply_points(probe)) * np.asarray(UM)
    assert np.abs(err_um).max() < 0.15  # a fraction of a µm across the field, z included


def test_the_fit_holds_yx_and_a_z_offset_only():
    # a cell layer constrains no z scale or tilt: the fit leaves them alone
    found = _estimator().estimate(*_pair(_forward()), seed=Transform.identity(3))
    m = np.asarray(found.matrix)
    assert (
        m[0, 0] == pytest.approx(1.0)
        and np.allclose(m[0, 1:3], 0)
        and np.allclose(m[1:3, 0], 0)
    )


def test_a_seed_far_past_the_radius_fails_and_says_why():
    # 20 µm off with 5 µm radii: too few pairs, and the message points to a coarse block.
    # (Somewhat past the first radius, nuclei can pair with neighbours instead: the seed
    # must be within it -- see NucleiEstimator.)
    from biahub.registration.estimators import EstimationError

    mov, ref = _pair(_forward(shift_vox=(0.0, 200.0, 0.0)), layout="scattered")
    with pytest.raises(EstimationError, match=r"pairs within .* µm .*coarse block"):
        _estimator().estimate(mov, ref, seed=Transform.identity(3))


def test_different_voxel_sizes_on_the_two_sides():
    # the moving grid is twice as coarse in yx: the fit works in µm, not voxels
    from biahub.registration.methods.nuclei import NucleiEstimator
    from biahub.settings import NucleiSettings

    ref = _nuclei_layer()
    mov = ref[:, ::2, ::2].copy()  # same field, half the pixels
    found = NucleiEstimator(NucleiSettings(), "similarity", (0.2, 0.206, 0.206), UM).estimate(
        mov, ref, seed=Transform(np.diag([1.0, 2.0, 2.0, 1.0]))
    )
    np.testing.assert_allclose(np.diag(np.asarray(found.matrix))[1:3], [2.0, 2.0], atol=0.01)


def test_nuclei_is_a_method_and_a_block():
    from biahub.settings import EstimateTransformSettings

    def settings(method):
        return EstimateTransformSettings(
            moving={"channel": "DAPI"},
            reference={"frame": "cross", "channel": "nuclei_prediction"},
            method=method,
        )

    single = settings("nuclei")
    assert single.nuclei.match_radii_um == [5.0, 2.0]
    assert single.effective_score_metric == "correlation"
    stack = settings([{"phase-cross-corr": {}}, {"nuclei": {"match_radii_um": [4.0, 1.5]}}])
    assert stack.stack_settings()[1].nuclei.match_radii_um == [4.0, 1.5]
