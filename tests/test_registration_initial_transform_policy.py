import numpy as np
import pytest

from biahub.core.transform import Transform
from biahub.registration.engine.policies import (
    ConsensusInitialTransform,
    FixedInitialTransform,
    InitialTransformPolicy,
)


def test_all_variants_satisfy_protocol():
    identity = Transform.from_translation([0.0, 0.0, 0.0])
    assert isinstance(FixedInitialTransform(identity), InitialTransformPolicy)
    assert isinstance(ConsensusInitialTransform(history={}, scores={}), InitialTransformPolicy)


def test_fixed_initial_transform_returns_the_same_transform_for_every_t():
    transform = Transform.from_translation([1.0, 2.0, 3.0])
    policy = FixedInitialTransform(transform)
    for t in range(5):
        assert policy.initial_for(t) is transform


def test_a_fixed_initial_transform_is_how_optimize_registration_needs_no_command():
    # "Refine an existing transform" is just estimate() with this as the initial
    # transform policy.
    existing_transform = Transform.from_translation([7.0, -3.0, 0.5])
    policy = FixedInitialTransform(existing_transform)
    assert policy.initial_for(0) is existing_transform


def test_consensus_initial_transform_raises_when_too_few_timepoints_score_well():
    from biahub.registration.estimators import CandidateUnavailableError

    history = {0: Transform.from_translation([1.0, 1.0, 1.0])}
    scores = {0: 0.9}
    policy = ConsensusInitialTransform(history=history, scores=scores, score_threshold=0.75)
    with pytest.raises(CandidateUnavailableError, match="only 1 timepoints"):
        policy.initial_for(3)


def test_consensus_initial_transform_is_the_median_transform_over_good_timepoints():
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

    initial_transform = policy.initial_for(t=3)

    np.testing.assert_allclose(initial_transform.matrix[:3, 3], [3.0, 0.0, 0.0])


def test_consensus_initial_transform_excludes_t_itself_even_if_it_scores_well():
    history = {
        0: Transform.from_translation([1.0, 0.0, 0.0]),
        1: Transform.from_translation([2.0, 0.0, 0.0]),
        2: Transform.from_translation([3.0, 0.0, 0.0]),
        3: Transform.from_translation([-1000.0, 0.0, 0.0]),
        4: Transform.from_translation([4.0, 0.0, 0.0]),
    }
    # t=3 has a (spuriously) good score but must never contaminate its own consensus
    # initial transform.
    scores = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.99, 4: 0.9}
    policy = ConsensusInitialTransform(
        history=history, scores=scores, score_threshold=0.75, min_good=4
    )

    initial_transform = policy.initial_for(t=3)

    np.testing.assert_allclose(initial_transform.matrix[:3, 3], [2.5, 0.0, 0.0])


def test_a_shapes_initial_transform_takes_the_voxel_sizes_it_is_given():
    # No built-in voxel sizes: the caller reads them from each store's metadata. Same
    # grids and no rotation or flip leave the moving volume where it is.
    from biahub.registration.engine.policies import ShapesInitialTransform

    shape, um = (30, 100, 100), (1.0, 0.5, 0.5)
    same = ShapesInitialTransform(shape, shape, um, um, rotate90=0, fliplr=False)
    np.testing.assert_allclose(same.initial_for(0).matrix, np.eye(4), atol=1e-12)
    # the moving voxels twice the reference's in z: one moving plane spans two reference
    # ones (the inverse matrix, reference -> moving, halves z)
    coarse = ShapesInitialTransform(
        shape, shape, (2.0, 0.5, 0.5), um, rotate90=0, fliplr=False
    )
    np.testing.assert_allclose(
        np.diag(coarse.initial_for(0).invert().matrix)[:3], (0.5, 1.0, 1.0)
    )


def test_a_shapes_initial_transform_is_the_same_for_every_t_and_composes_flip_rotation_scale():
    from biahub.registration.engine.policies import ShapesInitialTransform

    grids = dict(
        mov_shape=(30, 100, 100),
        ref_shape=(20, 120, 120),
        mov_voxel_size=(0.5, 0.2, 0.2),
        ref_voxel_size=(0.6, 0.15, 0.15),
    )
    policy = ShapesInitialTransform(**grids, rotate90=-1, fliplr=True, transform_type="affine")
    mov, ref = grids["mov_shape"], grids["ref_shape"]
    forward = (
        Transform.from_scaling((0.5 / 0.6, 0.2 / 0.15, 0.2 / 0.15), ref, ref).matrix
        @ Transform.from_rotation_yx(-90, mov, ref).matrix
        @ Transform.from_flip_x(mov, ref).matrix
    )  # flip first, then rotate, then scale (moving -> reference)
    for t in (0, 7):
        np.testing.assert_allclose(policy.initial_for(t).matrix, forward, atol=1e-12)
    np.testing.assert_allclose(
        policy.initial_for(0).invert().matrix, np.linalg.inv(forward), atol=1e-12
    )
    assert policy.initial_for(0).transform_type == "affine"
    assert isinstance(policy, InitialTransformPolicy)


def test_a_file_initial_transform_gives_each_timepoints_and_asks_to_be_kept_unless_beaten():
    from biahub.registration.engine.policies import FileInitialTransform

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


def test_repair_tries_the_initial_transform_as_the_candidate_initial():
    from biahub.registration.engine import neighbour_consensus_initial_candidates
    from biahub.settings import RepairSettings

    assert RepairSettings().candidates == ["t-1", "t+1", "consensus", "initial"]
    initial = Transform.from_translation([1.0, 0.0, 0.0])
    candidates = neighbour_consensus_initial_candidates(initial)(5, {}, {}, set())
    assert list(candidates) == ["consensus_full", "initial"]
    assert candidates["initial"].initial_for(5) is initial


def test_only_an_earlier_result_asks_to_be_kept_unless_beaten():
    from biahub.registration.engine.policies import (
        ConsensusInitialTransform,
        FixedInitialTransform,
        ShapesInitialTransform,
    )

    for policy in (FixedInitialTransform, ShapesInitialTransform, ConsensusInitialTransform):
        assert policy.keep_unless_beaten is False
    assert "keep_unless_beaten" in InitialTransformPolicy.__annotations__
