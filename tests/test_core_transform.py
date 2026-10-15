import numpy as np

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

    warped_scipy = forward_transform.apply(mov, reference=ref, order=0, backend="scipy")
    warped_ants = forward_transform.apply(mov, reference=ref, order=0, backend="ants")

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
