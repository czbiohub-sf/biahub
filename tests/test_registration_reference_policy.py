import numpy as np

from biahub.registration.reference_policy import (
    CrossChannel,
    FixedFrame,
    ReferencePolicy,
    RollingPrevious,
)


def test_all_variants_satisfy_protocol():
    mov = np.arange(3 * 4 * 4).reshape(3, 4, 4)
    assert isinstance(CrossChannel(mov), ReferencePolicy)
    assert isinstance(FixedFrame(), ReferencePolicy)
    assert isinstance(RollingPrevious(), ReferencePolicy)


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


def test_rolling_previous_uses_t_minus_1_and_itself_at_t0():
    mov = np.arange(5 * 4 * 4).reshape(5, 4, 4)
    policy = RollingPrevious()
    np.testing.assert_array_equal(policy.reference_for(mov, 0), mov[0])
    for t in range(1, 5):
        np.testing.assert_array_equal(policy.reference_for(mov, t), mov[t - 1])
