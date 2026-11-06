import numpy as np

from biahub.registration.engine.policies import (
    CrossChannel,
    FixedFrame,
    PreviousFrame,
    ReferencePolicy,
)


def test_all_variants_satisfy_protocol():
    mov = np.arange(3 * 4 * 4).reshape(3, 4, 4)
    assert isinstance(CrossChannel(mov), ReferencePolicy)
    assert isinstance(FixedFrame(), ReferencePolicy)
    assert isinstance(PreviousFrame(), ReferencePolicy)


def test_cross_channel_uses_the_external_reference_series():
    mov = np.zeros((3, 4, 4))
    ref = np.arange(3 * 4 * 4).reshape(3, 4, 4)
    policy = CrossChannel(ref)
    for t in range(3):
        np.testing.assert_array_equal(policy.reference_for(mov, t), ref[t])


def test_fixed_frame_always_returns_the_same_frame():
    mov = np.arange(5 * 4 * 4).reshape(5, 4, 4)
    policy = FixedFrame(t_ref=0)
    for t in range(5):
        np.testing.assert_array_equal(policy.reference_for(mov, t), mov[0])

    policy_t2 = FixedFrame(t_ref=2)
    np.testing.assert_array_equal(policy_t2.reference_for(mov, 4), mov[2])


def test_previous_frame_uses_t_minus_1_and_itself_at_t0():
    mov = np.arange(5 * 4 * 4).reshape(5, 4, 4)
    policy = PreviousFrame()
    np.testing.assert_array_equal(policy.reference_for(mov, 0), mov[0])
    for t in range(1, 5):
        np.testing.assert_array_equal(policy.reference_for(mov, t), mov[t - 1])


class _LazySeries:
    """Stands in for a dask/zarr-backed (T, ...) series: frames are cheap to index, but
    materializing the whole series is an error."""

    def __init__(self, frames: np.ndarray):
        self._frames = frames
        self.materialized = []

    def __getitem__(self, t):
        self.materialized.append(t)
        return self._frames[t]

    def __array__(self, *args, **kwargs):
        raise AssertionError("whole series materialized")


def test_policies_index_one_frame_and_never_materialize_the_series():
    frames = np.arange(4 * 2 * 2).reshape(4, 2, 2)
    mov, ref = _LazySeries(frames), _LazySeries(frames * 10)

    np.testing.assert_array_equal(CrossChannel(ref).reference_for(mov, 2), frames[2] * 10)
    np.testing.assert_array_equal(FixedFrame(t_ref=1).reference_for(mov, 3), frames[1])
    np.testing.assert_array_equal(PreviousFrame().reference_for(mov, 3), frames[2])

    assert ref.materialized == [2]
    assert mov.materialized == [1, 2]


def test_a_neighbour_median_reference_leaves_out_t_and_keeps_inside_the_run():
    from biahub.registration.engine.policies import NeighbourMedian

    frames = np.stack([np.full((2, 2), float(v)) for v in [1, 2, 0, 4, 5, 60, 7, 8]])
    mov = _LazySeries(frames)  # frame 2 is empty; frame 5 is a jump
    policy = NeighbourMedian(window=2, first=1, last=6)
    # t=4: frames 2..6 without 4 -> 2 is empty, so 3, 5, 6 -> median(4, 60, 7) = 7
    np.testing.assert_array_equal(policy.reference_for(mov, 4), np.full((2, 2), 7.0))
    assert sorted(set(mov.materialized)) == [2, 3, 5, 6]  # each read, nothing outside 1-6
    # t=1 (the run's first): frames 2, 3 (frame 0 is outside the run) -> 2 is empty -> 4
    np.testing.assert_array_equal(policy.reference_for(mov, 1), np.full((2, 2), 4.0))
    assert policy.neighbour_indices(mov, 1) == [3]
    # a timepoint with no neighbour with data compares against itself
    lonely = NeighbourMedian(window=1, first=0, last=7)
    alone = np.stack([np.zeros((2, 2)), np.ones((2, 2)), np.zeros((2, 2))])
    assert lonely.neighbour_indices(alone, 1) == []
    np.testing.assert_array_equal(lonely.reference_for(alone, 1), np.ones((2, 2)))


def test_a_neighbour_median_holds_its_frames_once():
    # the real run ran out of memory: frames kept in a list, stacked into a copy, and the
    # median sorting another; now the peak is 2 x window frames, the one read and the median
    import tracemalloc

    from biahub.registration.engine.policies import NeighbourMedian

    frames = np.random.default_rng(0).random((13, 16, 64, 64)).astype(np.float32)
    frame_bytes = frames[0].nbytes
    policy = NeighbourMedian(window=6)
    tracemalloc.start()
    policy.reference_for(frames, 6)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= (2 * 6 + 2) * frame_bytes * 1.05
    policy.neighbour_indices(frames, 6)  # the same timepoint: no second read


def test_each_reference_policy_says_whether_t_is_its_own_reference_and_spans_a_gap():
    from biahub.registration.engine.policies import (
        CrossChannel,
        FixedFrame,
        NeighbourMedian,
        PreviousFrame,
        ReferencePolicy,
    )

    mov = np.ones((5, 2, 4, 4), np.float32)
    mov[0] = 0  # leading empty frame
    mov[2] = 0  # a gap
    fixed, previous = FixedFrame(t_ref=0), PreviousFrame(first=0)
    neighbours, cross = NeighbourMedian(window=1), CrossChannel(mov)
    for policy in (fixed, previous, neighbours, cross):
        assert isinstance(policy, ReferencePolicy)

    # the frame that starts the chain is compared with itself
    assert [fixed.compares_with_itself(mov, t) for t in range(5)] == [0, 1, 0, 0, 0]
    assert [previous.compares_with_itself(mov, t) for t in range(5)] == [1, 1, 0, 0, 0]
    # t=1's neighbours (t=0, t=2) are both empty; t=3 has t=4
    assert [neighbours.compares_with_itself(mov, t) for t in (1, 3)] == [True, False]
    assert not any(cross.compares_with_itself(mov, t) for t in range(5))

    # only "previous" reaches back over empty frames
    assert [previous.spans_gap(mov, t) for t in range(5)] == [0, 0, 0, 1, 0]
    assert not any(p.spans_gap(mov, t) for p in (fixed, neighbours, cross) for t in range(5))
