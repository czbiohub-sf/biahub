import numpy as np
import pytest

from biahub.core.transform import Transform
from biahub.registration.seed_policy import (
    ConsensusSeed,
    FixedSeed,
    PreviousSeed,
    SeedPolicy,
)


def test_all_variants_satisfy_protocol():
    identity = Transform.from_translation([0.0, 0.0, 0.0])
    assert isinstance(FixedSeed(identity), SeedPolicy)
    assert isinstance(PreviousSeed(history={}, fallback=FixedSeed(identity)), SeedPolicy)
    assert isinstance(ConsensusSeed(history={}, scores={}), SeedPolicy)


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


def test_consensus_seed_raises_when_too_few_timepoints_score_well():
    history = {0: Transform.from_translation([1.0, 1.0, 1.0])}
    scores = {0: 0.9}
    policy = ConsensusSeed(history=history, scores=scores, score_threshold=0.75)
    with pytest.raises(ValueError, match="only 1 timepoints"):
        policy.seed_for(3)


def test_consensus_seed_is_the_median_transform_over_good_timepoints():
    history = {
        0: Transform.from_translation([1.0, 0.0, 0.0]),
        1: Transform.from_translation([2.0, 0.0, 0.0]),
        2: Transform.from_translation([3.0, 0.0, 0.0]),
        3: Transform.from_translation([100.0, 0.0, 0.0]),  # poorly scored, excluded
        4: Transform.from_translation([4.0, 0.0, 0.0]),
        5: Transform.from_translation([5.0, 0.0, 0.0]),
    }
    scores = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.1, 4: 0.9, 5: 0.9}
    policy = ConsensusSeed(history=history, scores=scores, score_threshold=0.75, min_good=5)

    seed = policy.seed_for(t=3)

    np.testing.assert_allclose(seed.matrix[:3, 3], [3.0, 0.0, 0.0])


def test_consensus_seed_excludes_t_itself_even_if_it_scores_well():
    history = {
        0: Transform.from_translation([1.0, 0.0, 0.0]),
        1: Transform.from_translation([2.0, 0.0, 0.0]),
        2: Transform.from_translation([3.0, 0.0, 0.0]),
        3: Transform.from_translation([-1000.0, 0.0, 0.0]),
        4: Transform.from_translation([4.0, 0.0, 0.0]),
    }
    # t=3 has a (spuriously) good score but must never contaminate its own consensus seed.
    scores = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.99, 4: 0.9}
    policy = ConsensusSeed(history=history, scores=scores, score_threshold=0.75, min_good=4)

    seed = policy.seed_for(t=3)

    np.testing.assert_allclose(seed.matrix[:3, 3], [2.5, 0.0, 0.0])
