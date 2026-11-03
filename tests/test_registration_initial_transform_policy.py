import numpy as np
import pytest

from biahub.core.transform import Transform
from biahub.registration.policies import (
    ConsensusInitialTransform,
    FixedInitialTransform,
    InitialTransformPolicy,
)


def test_all_variants_satisfy_protocol():
    identity = Transform.from_translation([0.0, 0.0, 0.0])
    assert isinstance(FixedInitialTransform(identity), InitialTransformPolicy)
    assert isinstance(ConsensusInitialTransform(history={}, scores={}), InitialTransformPolicy)


def test_fixed_seed_returns_the_same_transform_for_every_t():
    transform = Transform.from_translation([1.0, 2.0, 3.0])
    policy = FixedInitialTransform(transform)
    for t in range(5):
        assert policy.initial_for(t) is transform


def test_fixed_seed_is_how_optimize_registration_becomes_a_seed_not_a_command():
    # "Refine an existing transform" is just estimate() with this as the seed policy.
    existing_transform = Transform.from_translation([7.0, -3.0, 0.5])
    policy = FixedInitialTransform(existing_transform)
    assert policy.initial_for(0) is existing_transform


def test_consensus_seed_raises_when_too_few_timepoints_score_well():
    history = {0: Transform.from_translation([1.0, 1.0, 1.0])}
    scores = {0: 0.9}
    policy = ConsensusInitialTransform(history=history, scores=scores, score_threshold=0.75)
    with pytest.raises(ValueError, match="only 1 timepoints"):
        policy.initial_for(3)


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
    policy = ConsensusInitialTransform(
        history=history, scores=scores, score_threshold=0.75, min_good=5
    )

    seed = policy.initial_for(t=3)

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
    policy = ConsensusInitialTransform(
        history=history, scores=scores, score_threshold=0.75, min_good=4
    )

    seed = policy.initial_for(t=3)

    np.testing.assert_allclose(seed.matrix[:3, 3], [2.5, 0.0, 0.0])


def test_a_shapes_initial_transform_is_the_approximate_transform_from_the_two_grids():
    # what init_run computes today: get_aprox_transform's matrix, read as inverse
    from biahub.registration.policies import ShapesInitialTransform
    from biahub.registration.seed import get_aprox_transform

    grids = dict(
        mov_shape=(30, 100, 100),
        ref_shape=(20, 120, 120),
        mov_voxel_size=(0.5, 0.2, 0.2),
        ref_voxel_size=(0.6, 0.15, 0.15),
    )
    policy = ShapesInitialTransform(
        **grids, rotate90=-1, fliplr=False, transform_type="affine"
    )
    expected = get_aprox_transform(
        **grids, pre_affine_90degree_rotation=-1, pre_affine_fliplr=False
    )
    for t in (0, 7):
        np.testing.assert_array_equal(
            policy.initial_for(t).matrix, Transform.from_inverse(expected.matrix).matrix
        )
    assert policy.initial_for(0).transform_type == "affine"
    assert isinstance(policy, InitialTransformPolicy)


def test_a_file_initial_transform_gives_each_timepoints_and_asks_to_be_kept_unless_beaten():
    from biahub.registration.policies import FileInitialTransform

    by_t = {
        0: Transform.from_translation([1.0, 2.0, 3.0]),
        4: Transform.from_translation([0.0, -1.0, 0.0]),
    }
    policy = FileInitialTransform(by_t)
    assert policy.initial_for(4) is by_t[4]
    assert (
        policy.keep_unless_beaten is True
    )  # an earlier result: the engine keeps it unless an estimate scores better
    with pytest.raises(KeyError, match="no initial transform for t=2"):
        policy.initial_for(2)
    assert isinstance(policy, InitialTransformPolicy)
