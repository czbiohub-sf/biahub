import numpy as np
import pytest

from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform


def _interior(shape, margin):
    return tuple(slice(margin, s - margin) for s in shape)


def test_apply_scipy_and_ants_backends_agree_on_same_transform():
    """A single Transform must warp identically regardless of backend.

    Regression test: `_apply_ants` used to feed `apply_to_image` the forward
    (moving -> reference) matrix directly, but ants' `apply_to_image` does 'inverse'
    (backward) resampling like scipy's `affine_transform` -- it needs the inverse.
    `_apply_scipy` already inverted; `_apply_ants` did not, so the two backends silently
    disagreed for the same Transform object.
    """
    rng = np.random.default_rng(0)
    shape = (30, 40, 50)
    ref = rng.random(shape).astype(np.float32)

    applied_zyx = (4, -6, 9)  # distinct per-axis values so an axis mixup would show
    mov = ndi_shift(ref, shift=applied_zyx, order=0, mode="constant", cval=0.0)

    # The forward (moving -> reference) transform, exactly as an estimator would
    # produce it: mov is `applied_zyx` away from ref, so undoing that lands mov on ref.
    forward_transform = Transform.from_translation([-a for a in applied_zyx])

    # order=1: other orders resample with scipy whatever the backend, which would compare
    # scipy with itself (the integer translation keeps linear interpolation exact).
    warped_scipy = forward_transform.apply(mov, reference=ref, order=1, backend="scipy")
    warped_ants = forward_transform.apply(mov, reference=ref, order=1, backend="ants")

    interior = _interior(shape, margin=int(max(abs(a) for a in applied_zyx)) + 1)
    np.testing.assert_allclose(warped_scipy[interior], warped_ants[interior], atol=1e-3)
    np.testing.assert_allclose(warped_scipy[interior], ref[interior], atol=1e-3)
    np.testing.assert_allclose(warped_ants[interior], ref[interior], atol=1e-3)


def test_ants_point_conversion_is_direction_preserving():
    """to_ants()/apply_to_point() is a faithful, non-inverted numeric conversion.

    Only image resampling (`_apply_ants`) needs the inverse -- the point-level
    conversion itself does not silently flip direction.
    """
    transform = Transform.from_translation([1.0, 2.0, 3.0])
    point = np.array([10.0, 20.0, 30.0])

    via_apply_points = transform.apply_points(point[None, :])[0]
    via_ants_point = np.array(transform.to_ants().apply_to_point(tuple(point)))

    np.testing.assert_allclose(via_apply_points, via_ants_point)


def test_non_default_resampling_options_are_honoured_by_the_default_backend():
    # The ANTs backend resamples linearly only; asking for nearest-neighbour (labels) or a
    # fill value must not be ignored silently.
    labels = np.zeros((8, 16, 16), dtype=np.float32)
    labels[2:6, 4:12, 4:12] = 3.0
    labels[3:5, 6:10, 6:10] = 7.0
    shift = Transform.from_translation([0.0, 0.5, 0.5])
    out = shift.apply(labels, order=0)
    assert set(np.unique(out)) <= {0.0, 3.0, 7.0}  # no blended label values
    filled = shift.apply(labels, cval=-1.0)
    assert filled.min() == -1.0  # the fill value outside the moved volume


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("transform_type", ["affine", "similarity", "euclidean"])
def test_fit_recovers_a_transform_from_matched_points(ndim, transform_type):
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (20, ndim))
    angle = np.deg2rad(5.0)
    linear = np.eye(ndim)
    linear[-2:, -2:] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    if transform_type == "similarity":
        linear = 0.95 * linear
    elif transform_type == "affine":
        linear = linear @ np.diag(np.linspace(0.9, 1.05, ndim))
    truth = np.eye(ndim + 1)
    truth[:ndim, :ndim] = linear
    truth[:ndim, ndim] = rng.uniform(-5, 5, ndim)
    dst = Transform(truth).apply_points(src)
    fitted = Transform.fit(src, dst, transform_type)
    np.testing.assert_allclose(fitted.matrix, truth, atol=1e-8)
    assert fitted.ndim == ndim


def test_fit_says_when_it_cannot():
    with pytest.raises(ValueError, match="similarity fit failed on 1 pairs"):
        Transform.fit(np.zeros((1, 3)), np.ones((1, 3)), "similarity")


def test_fit_can_use_scikit_images_affine_estimator():
    # exact pairs: both solvers agree; noisy pairs: "skimage" is scikit-image's estimator
    from skimage.transform import AffineTransform

    rng = np.random.default_rng(1)
    src = rng.uniform(0, 100, (30, 3))
    truth = np.eye(4)
    truth[:3, :3] = np.diag([0.95, 1.02, 0.98])
    truth[:3, 3] = [2.0, -3.0, 1.0]
    exact = Transform(truth).apply_points(src)
    np.testing.assert_allclose(
        Transform.fit(src, exact, solver="skimage").matrix,
        Transform.fit(src, exact).matrix,
        atol=1e-8,
    )
    noisy = exact + rng.normal(0, 0.5, exact.shape)
    np.testing.assert_allclose(
        Transform.fit(src, noisy, solver="skimage").matrix,
        AffineTransform.from_estimate(src, noisy).params,
    )
    with pytest.raises(ValueError, match="solver"):
        Transform.fit(src, noisy, solver="ransac")
