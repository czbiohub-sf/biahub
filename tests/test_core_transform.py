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


def test_a_transform_copies_its_matrix_and_leaves_the_callers_writable():
    # np.asarray kept a float64 caller's array itself and then froze it: the caller could
    # no longer write to its own matrix.
    matrix = np.eye(4)
    transform = Transform(matrix)
    matrix[0, 3] = 5.0  # the caller's array is still its own
    assert matrix.flags.writeable
    np.testing.assert_array_equal(
        transform.matrix, np.eye(4)
    )  # and the transform is not changed
    with pytest.raises(ValueError, match="read-only"):
        transform._matrix[0, 3] = 1.0  # the transform's own copy stays immutable


def test_an_inverse_matrix_moves_the_volume_the_other_way():
    # apply-transform's files store the inverse (output -> input) matrix: a translation
    # of (-3, 1, 4) in it moves the data by (+3, -1, -4)
    inverse = np.eye(4)
    inverse[:3, -1] = (-3, 1, 4)
    out = Transform.from_inverse(inverse).apply(
        np.ones((10, 10, 10), dtype=np.float32), reference=np.zeros((10, 10, 10), np.float32)
    )
    assert out.shape == (10, 10, 10)
    assert np.all(out[3:10, 0:9, 0:6] == 1)
    assert not out[:3].any() and not out[:, 9:].any() and not out[:, :, 6:].any()


def test_the_output_voxel_size_is_one_output_step_in_the_input_um():
    # apply-transform writes it as the output's scale when the file gives none. An output
    # step along axis j is the input step column j of the inverse matrix, in the input's
    # µm; rescale_voxel_size took the rows, which swaps sizes when a rotation mixes axes
    # of different voxel sizes.
    um = (0.5, 0.1, 0.2)
    two_planes = np.diag([2.0, 1.0, 1.0, 1.0])  # an output plane every two input planes
    np.testing.assert_allclose(
        Transform.from_inverse(two_planes).output_voxel_size(um), (1.0, 0.1, 0.2)
    )
    swap_yx = np.eye(4)
    swap_yx[1:3, 1:3] = [[0.0, 1.0], [1.0, 0.0]]  # an output y step is an input x step
    np.testing.assert_allclose(
        Transform.from_inverse(swap_yx).output_voxel_size(um), (0.5, 0.2, 0.1)
    )
    # a rotation within an isotropic yx plane keeps its voxel size (mantis: 90 degrees)
    theta = np.deg2rad(90)
    rotate = np.eye(4)
    rotate[1:3, 1:3] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    np.testing.assert_allclose(
        Transform.from_inverse(rotate).output_voxel_size((0.5, 0.15, 0.15)), (0.5, 0.15, 0.15)
    )


def test_scaling_rotation_and_flip_constructors_keep_the_centre_where_it_belongs():
    # from_scaling / from_rotation_yx (forward, start grid -> end grid): the start grid's
    # yx centre lands on the end grid's yx centre; z is scaled about plane 0. from_flip_x
    # flips within one grid, about its own x centre
    start, end = (10, 40, 60), (20, 80, 100)
    centre = lambda shape: np.array([0.0, shape[1] / 2, shape[2] / 2])  # noqa: E731
    scaled = Transform.from_scaling((2.0, 0.5, 0.5), start, end)
    np.testing.assert_allclose(scaled.apply_points(centre(start)[None])[0], centre(end))
    np.testing.assert_allclose(scaled.apply_points(np.array([[3.0, 20, 30]]))[0, 0], 6.0)
    assert scaled.transform_type == "affine"
    turned = Transform.from_rotation_yx(90, start, end)
    np.testing.assert_allclose(turned.apply_points(centre(start)[None])[0], centre(end))
    # a quarter turn: one step along x becomes one step along y
    step = turned.apply_points(centre(start)[None] + [0.0, 0.0, 1.0])[0] - centre(end)
    np.testing.assert_allclose(step, (0.0, -1.0, 0.0), atol=1e-12)
    flipped = Transform.from_flip_x(start)
    np.testing.assert_allclose(
        flipped.apply_points(np.array([[1.0, 5.0, 10.0]]))[0], (1, 5, 50)
    )
    # the same shape twice: the end grid defaults to the start
    np.testing.assert_array_equal(
        Transform.from_rotation_yx(90, start).matrix,
        Transform.from_rotation_yx(90, start, start).matrix,
    )


def test_on_grid_is_the_same_transform_on_another_grid():
    # a point of the new grid, mapped there, lands where the transform sends it in the
    # old coordinates: index i of the new grid sits at scale * i + offset
    rng = np.random.default_rng(3)
    matrix = np.eye(4)
    matrix[:3, :3] += 0.1 * rng.standard_normal((3, 3))
    matrix[:3, 3] = 10 * rng.standard_normal(3)
    transform = Transform(matrix, transform_type="affine")
    scale, offset = np.array([1.0, 4.0, 4.0]), np.array([0.0, 1.5, 1.5])
    on_grid = transform.on_grid(scale, offset)
    points = rng.uniform(0, 50, size=(20, 3))
    old = points * scale + offset
    mapped_old = old @ matrix[:3, :3].T + matrix[:3, 3]
    mapped_new = points @ on_grid.matrix[:3, :3].T + on_grid.matrix[:3, 3]
    np.testing.assert_allclose(mapped_new * scale + offset, mapped_old, atol=1e-9)
    assert on_grid.transform_type == "affine"
    # one scale for every axis, no offset: a 2-voxel shift is 1 voxel on a 2x coarser grid
    shift = Transform.from_translation([2.0, 4.0, -6.0]).on_grid(2)
    np.testing.assert_allclose(shift.translation, [1.0, 2.0, -3.0])


def test_from_yx_applies_a_2d_transform_in_every_plane_and_shifts_z():
    rng = np.random.default_rng(4)
    yx = np.eye(3)
    yx[:2, :2] += 0.1 * rng.standard_normal((2, 2))
    yx[:2, 2] = [5.0, -3.0]
    lifted = Transform.from_yx(yx, z_shift=2.5, transform_type="similarity")
    points = rng.uniform(0, 50, size=(10, 3))
    moved = lifted.apply_points(points)
    np.testing.assert_allclose(moved[:, 0], points[:, 0] + 2.5)
    np.testing.assert_allclose(moved[:, 1:], points[:, 1:] @ yx[:2, :2].T + yx[:2, 2])
    assert lifted.transform_type == "similarity" and lifted.ndim == 3
