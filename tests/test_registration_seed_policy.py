import numpy as np

from biahub.core.transform import Transform
from biahub.registration.seed_policy import FixedSeed, PreviousSeed, SeedPolicy


def test_all_variants_satisfy_protocol():
    identity = Transform.from_translation([0.0, 0.0, 0.0])
    assert isinstance(FixedSeed(identity), SeedPolicy)
    assert isinstance(PreviousSeed(history={}, fallback=FixedSeed(identity)), SeedPolicy)


def test_fixed_seed_returns_the_same_transform_for_every_t():
    transform = Transform.from_translation([1.0, 2.0, 3.0])
    policy = FixedSeed(transform)
    for t in range(5):
        assert policy.seed_for(t) is transform


def test_fixed_seed_is_how_optimize_registration_becomes_a_seed_not_a_command():
    # "Refine an existing transform" is just estimate() with this as the seed policy.
    existing_transform = Transform.from_translation([7.0, -3.0, 0.5])
    policy = FixedSeed(existing_transform)
    assert policy.seed_for(0) is existing_transform


def test_previous_seed_uses_history_and_falls_back_when_missing():
    fallback_transform = Transform.from_translation([0.0, 0.0, 0.0])
    history: dict[int, Transform] = {}
    policy = PreviousSeed(history=history, fallback=FixedSeed(fallback_transform))

    # t=0: nothing in history yet, falls back.
    assert policy.seed_for(0) is fallback_transform

    # Caller accepts a result for t=0 and records it; t=1 should now seed from it.
    accepted_t0 = Transform.from_translation([1.0, 1.0, 1.0])
    history[0] = accepted_t0
    assert policy.seed_for(1) is accepted_t0

    # A gap (t=2 never recorded) falls back again rather than reading stale history.
    assert policy.seed_for(3) is fallback_transform


def test_previous_seed_never_mutates_the_fallback_or_history():
    fallback_transform = Transform.from_translation([0.0, 0.0, 0.0])
    fallback_matrix_before = fallback_transform.matrix.copy()
    history: dict[int, Transform] = {0: Transform.from_translation([5.0, 5.0, 5.0])}
    history_before = dict(history)

    policy = PreviousSeed(history=history, fallback=FixedSeed(fallback_transform))
    for t in range(4):
        policy.seed_for(t)

    np.testing.assert_array_equal(fallback_transform.matrix, fallback_matrix_before)
    assert history.keys() == history_before.keys()
    for t in history:
        np.testing.assert_array_equal(history[t].matrix, history_before[t].matrix)
