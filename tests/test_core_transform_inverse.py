import numpy as np

from biahub.core.transform import Transform


def test_forward_from_inverse_inverts_the_direction():
    # A inverse matrix that shifts sampling coordinates by +5 corresponds to content that
    # must move by -5 to land on the reference.
    inverse = Transform.from_translation([5.0, 0.0, 0.0]).matrix
    forward = Transform.from_inverse(inverse)
    np.testing.assert_allclose(forward.translation, [-5.0, 0.0, 0.0])

    mov_point = np.array([[10.0, 10.0, 10.0]])
    np.testing.assert_allclose(forward.apply_points(mov_point), [[5.0, 10.0, 10.0]])


def test_round_trip_preserves_the_legacy_matrix():
    rng = np.random.default_rng(0)
    linear = np.eye(3) + 0.05 * rng.standard_normal((3, 3))
    inverse = np.eye(4)
    inverse[:3, :3] = linear
    inverse[:3, 3] = [1.5, -2.0, 30.0]

    round_tripped = Transform.from_inverse(inverse).to_inverse()
    np.testing.assert_allclose(round_tripped, inverse, atol=1e-12)


def test_transform_type_is_carried_through():
    forward = Transform.from_inverse(np.eye(4), transform_type="euclidean")
    assert forward.transform_type == "euclidean"
