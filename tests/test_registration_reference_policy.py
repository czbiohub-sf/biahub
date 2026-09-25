import numpy as np

from biahub.registration.policies import (
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
