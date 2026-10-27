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
    from biahub.registration.methods.landmark.nuclei_matching import nuclei_estimator
    from biahub.settings import NucleiSettings

    return nuclei_estimator(NucleiSettings(**settings), "similarity", UM)


PROBE = np.array([[12, 60, 60], [12, 60, 450], [12, 450, 60], [12, 450, 450], [12, 256, 256]])


def test_nuclei_is_a_landmark_based_graph_matching_estimator():
    # The same parts as beads: a detector per channel, a graph matcher, a fit.
    from biahub.registration.methods.landmark.bead_matching import NodeGraphEstimator
    from biahub.registration.methods.landmark.nuclei_matching import NucleiNodeDetector

    estimator = _estimator()
    assert isinstance(estimator, NodeGraphEstimator)
    assert isinstance(estimator.mov_detector, NucleiNodeDetector)
    assert isinstance(estimator.ref_detector, NucleiNodeDetector)
    assert estimator.beads_match_settings.algorithm == "hungarian"


def test_nuclei_recover_a_camera_scale_rotation_and_shift():
    truth = _forward()
    mov, ref = _pair(truth)
    found = _estimator().estimate(mov, ref, seed=Transform.identity(3))
    err_um = (found.apply_points(PROBE) - truth.apply_points(PROBE)) * np.asarray(UM)
    assert np.abs(err_um).max() < 0.15


def test_the_fit_holds_yx_and_a_z_offset_only():
    # a cell layer constrains no z scale or tilt: the fit leaves them alone
    found = _estimator().estimate(*_pair(_forward()), seed=Transform.identity(3))
    m = np.asarray(found.matrix)
    assert (
        m[0, 0] == pytest.approx(1.0)
        and np.allclose(m[0, 1:3], 0)
        and np.allclose(m[1:3, 0], 0)
    )


def test_different_voxel_sizes_on_the_two_sides():
    # the moving grid is twice as coarse in yx: nuclei are detected on the warped moving
    # volume, on the reference grid, so the fit is in one grid
    from biahub.registration.methods.landmark.nuclei_matching import nuclei_estimator
    from biahub.settings import NucleiSettings

    ref = _nuclei_layer()
    mov = ref[:, ::2, ::2].copy()
    found = nuclei_estimator(NucleiSettings(), "similarity", UM).estimate(
        mov,
        ref,
        seed=Transform(np.diag([1.0, 2.0, 2.0, 1.0])),  # forward: moving -> reference
    )
    np.testing.assert_allclose(np.diag(np.asarray(found.matrix))[1:3], [2.0, 2.0], atol=0.01)


def test_nuclei_is_a_method_and_a_block_with_the_matchers_of_beads():
    from biahub.settings import EstimateTransformSettings

    def settings(method):
        return EstimateTransformSettings(
            moving={"channel": "DAPI"},
            reference={"frame": "cross", "channel": "nuclei_prediction"},
            method=method,
        )

    single = settings("nuclei_matching")
    assert single.nuclei_matching.algorithm == "hungarian"
    assert single.effective_score_metric == "correlation"
    stack = settings(
        [{"phase_cross_corr": {"local": {}}}, {"nuclei_matching": {"algorithm": "spectral"}}]
    )
    assert stack.stack_settings()[1].nuclei_matching.algorithm == "spectral"


def test_nuclei_on_a_single_slice():
    truth = _forward(shift_vox=(0.0, 15.0, -10.0))
    mov, ref = _pair(truth)
    mov, ref = mov[12:13], ref[12:13]
    found = _estimator().estimate(mov, ref, seed=Transform.identity(3))
    probe = np.array([[0, 60, 60], [0, 450, 450], [0, 256, 256]])
    err = (found.apply_points(probe) - truth.apply_points(probe))[:, 1:] * np.asarray(UM[1:])
    assert np.abs(err).max() < 0.3
