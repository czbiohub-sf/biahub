"""Local phase cross-correlation: tile shifts on max projections, fitted in yx; z stays the seed's."""

import numpy as np
import pytest

from scipy.ndimage import affine_transform

from biahub.core.transform import Transform

UM = (0.2, 0.103, 0.103)


def _cells(seed=0, shape=(24, 384, 384)):
    from tests.test_registration_estimators import _synthetic_blob_volume

    rng = np.random.default_rng(seed)
    return _synthetic_blob_volume(rng, shape, n_blobs=400, sigma=3.0) + 100.0


def _yx_similarity(scale=0.97, angle_deg=0.5, shift_yx=(6.0, -4.0), centre=(192, 192)):
    """A forward map moving -> reference, acting in yx only (z unchanged)."""
    a = np.deg2rad(angle_deg)
    linear = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.asarray(centre, float)
    matrix = np.eye(4)
    matrix[1:3, 1:3] = linear
    matrix[1:3, 3] = c - linear @ c + np.asarray(shift_yx)
    return Transform(matrix)


def _pair(truth):
    ref = _cells()
    m = np.asarray(truth.matrix)
    mov = affine_transform(ref, m[:3, :3], offset=m[:3, 3], order=1, mode="nearest")
    return mov.astype(np.float32), ref


def _estimator(**settings):
    from biahub.registration.methods.intensity.phase_cross_corr import LocalPCCEstimator
    from biahub.settings import LocalPhaseCrossCorrSettings

    return LocalPCCEstimator(
        LocalPhaseCrossCorrSettings(tile_px=96, grid=5, **settings), "similarity", UM
    )


PROBE = np.array([[12, 50, 50], [12, 50, 330], [12, 330, 50], [12, 330, 330], [12, 192, 192]])


def test_tiles_recover_a_yx_scale_rotation_and_shift():
    truth = _yx_similarity()
    mov, ref = _pair(truth)
    found = _estimator().estimate(mov, ref, seed=Transform.identity(3))
    err_um = (found.apply_points(PROBE) - truth.apply_points(PROBE))[:, 1:] * np.asarray(
        UM[1:]
    )
    assert np.abs(err_um).max() < 0.1


def test_z_stays_the_seeds():
    # the block measures yx only: whatever z the seed brings, it keeps
    mov, ref = _pair(_yx_similarity())
    seed = Transform.from_translation([1.5, 0.0, 0.0])
    found = _estimator().estimate(mov, ref, seed=seed)
    np.testing.assert_allclose(np.asarray(found.matrix)[0], np.asarray(seed.matrix)[0])


def test_too_few_tiles_fail_and_say_why():
    from biahub.registration.estimators import EstimationError

    rng = np.random.default_rng(3)
    # independent noise in each channel: no structure in common
    mov, ref = (rng.normal(100, 1, (24, 384, 384)).astype(np.float32) for _ in range(2))
    with pytest.raises(EstimationError, match="tiles correlated"):
        _estimator().estimate(mov, ref, seed=Transform.identity(3))


def test_local_is_an_option_of_phase_cross_corr():
    from biahub.settings import EstimateTransformSettings

    def settings(method):
        return EstimateTransformSettings(
            moving={"channel": "DAPI"},
            reference={"frame": "cross", "channel": "nuclei_prediction"},
            method=method,
        )

    whole = settings("phase_cross_corr")
    assert whole.phase_cross_corr.local is None  # one translation, as before
    stack = settings([{"phase_cross_corr": {"local": {"compare": "intensity"}}}, {"ants": {}}])
    assert stack.stack_settings()[0].phase_cross_corr.local.compare == "intensity"
    assert stack.stack_settings()[0].phase_cross_corr.local.rounds == 2
