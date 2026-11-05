import numpy as np
import pytest

from biahub.core.transform import Transform
from biahub.registration.engine import repair
from biahub.registration.engine.policies import FixedInitialTransform
from biahub.registration.estimators import CandidateUnavailableError, EstimationError


class _EchoEstimator:
    """Test double: estimate() just returns whatever seed it was given (or identity
    if none), so repair's candidate-selection logic can be tested without a real
    estimator or real image data."""

    def estimate(self, mov, ref, seed=None):
        return seed if seed is not None else Transform.from_translation([0.0, 0.0, 0.0])


def _score_by_closeness(transform: Transform, target: np.ndarray) -> float:
    """Higher score = translation closer to `target` -- stands in for a real quality
    score (e.g. bead-overlap) so tests can control which candidate should win."""
    return -float(np.abs(transform.matrix[:3, 3] - target).sum())


def test_repair_picks_the_best_scoring_candidate_not_the_first_or_last():
    target = np.array([10.0, 10.0, 10.0])
    candidates = {
        "far": FixedInitialTransform(Transform.from_translation([0.0, 0.0, 0.0])),
        "close": FixedInitialTransform(Transform.from_translation([9.0, 9.0, 9.0])),  # best
        "medium": FixedInitialTransform(Transform.from_translation([5.0, 5.0, 5.0])),
    }

    result = repair(
        t=3,
        mov=None,
        ref=None,
        estimator=_EchoEstimator(),
        current_transform=Transform.from_translation([0.0, 0.0, 0.0]),
        current_score=_score_by_closeness(Transform.from_translation([0.0, 0.0, 0.0]), target),
        candidates=candidates,
        score_fn=lambda t: _score_by_closeness(t, target),
    )

    assert result.accepted is True
    assert result.source == "close"
    np.testing.assert_allclose(result.transform.matrix[:3, 3], [9.0, 9.0, 9.0])


def test_repair_keeps_current_when_no_candidate_beats_it():
    target = np.array([10.0, 10.0, 10.0])
    current = Transform.from_translation([9.9, 9.9, 9.9])  # already very close
    candidates = {
        "far": FixedInitialTransform(Transform.from_translation([0.0, 0.0, 0.0])),
        "medium": FixedInitialTransform(Transform.from_translation([5.0, 5.0, 5.0])),
    }

    result = repair(
        t=3,
        mov=None,
        ref=None,
        estimator=_EchoEstimator(),
        current_transform=current,
        current_score=_score_by_closeness(current, target),
        candidates=candidates,
        score_fn=lambda t: _score_by_closeness(t, target),
    )

    assert result.accepted is False
    assert result.source == "unchanged"
    assert result.transform is current


def test_repair_skips_a_candidate_that_raises_without_failing_the_whole_pass():
    target = np.array([10.0, 10.0, 10.0])

    class _RaisingEstimator:
        def estimate(self, mov, ref, seed=None):
            if seed is not None and seed.matrix[0, 3] == 999.0:
                raise EstimationError("simulated candidate failure")
            return seed

    candidates = {
        "broken": FixedInitialTransform(Transform.from_translation([999.0, 0.0, 0.0])),
        "good": FixedInitialTransform(Transform.from_translation([9.0, 9.0, 9.0])),
    }

    result = repair(
        t=3,
        mov=None,
        ref=None,
        estimator=_RaisingEstimator(),
        current_transform=Transform.from_translation([0.0, 0.0, 0.0]),
        current_score=_score_by_closeness(Transform.from_translation([0.0, 0.0, 0.0]), target),
        candidates=candidates,
        score_fn=lambda t: _score_by_closeness(t, target),
    )

    assert result.accepted is True
    assert result.source == "good"
    assert result.failures == {"broken": "EstimationError: simulated candidate failure"}
    assert set(result.scores) == {"good"}


def test_repair_names_each_failed_candidate():
    target = np.array([10.0, 10.0, 10.0])

    class _RaisingSeedPolicy:
        def initial_for(self, t):
            raise CandidateUnavailableError("not enough good timepoints yet")

    result = repair(
        t=3,
        mov=None,
        ref=None,
        estimator=_EchoEstimator(),
        current_transform=Transform.from_translation([0.0, 0.0, 0.0]),
        current_score=_score_by_closeness(Transform.from_translation([0.0, 0.0, 0.0]), target),
        candidates={"consensus": _RaisingSeedPolicy()},
        score_fn=lambda t: _score_by_closeness(t, target),
    )

    assert result.accepted is False
    assert result.failures == {
        "consensus": "CandidateUnavailableError: not enough good timepoints yet"
    }


def test_repair_skips_a_candidate_whose_seed_for_raises_without_failing_the_whole_pass():
    """E.g. `ConsensusInitialTransform` raises `CandidateUnavailableError` when too few timepoints
    score well enough yet -- skipped like any other candidate failure, not crash `repair`."""
    target = np.array([10.0, 10.0, 10.0])

    class _RaisingSeedPolicy:
        def initial_for(self, t):
            raise CandidateUnavailableError("not enough good timepoints yet")

    candidates = {
        "not_ready": _RaisingSeedPolicy(),
        "good": FixedInitialTransform(Transform.from_translation([9.0, 9.0, 9.0])),
    }

    result = repair(
        t=3,
        mov=None,
        ref=None,
        estimator=_EchoEstimator(),
        current_transform=Transform.from_translation([0.0, 0.0, 0.0]),
        current_score=_score_by_closeness(Transform.from_translation([0.0, 0.0, 0.0]), target),
        candidates=candidates,
        score_fn=lambda t: _score_by_closeness(t, target),
    )

    assert result.accepted is True
    assert result.source == "good"


def test_a_repair_candidate_with_a_bug_is_not_hidden_as_a_failed_candidate():
    # only EstimationError and CandidateUnavailableError mean "this candidate failed"; any other
    # exception is a bug and propagates instead of being recorded as a lost candidate
    target = np.array([10.0, 10.0, 10.0])

    class _BuggyPolicy:
        def initial_for(self, t):
            raise TypeError("a bug")

    with pytest.raises(TypeError, match="a bug"):
        repair(
            t=3,
            mov=None,
            ref=None,
            estimator=_EchoEstimator(),
            current_transform=Transform.from_translation([0.0, 0.0, 0.0]),
            current_score=0.0,
            candidates={"buggy": _BuggyPolicy()},
            score_fn=lambda t: _score_by_closeness(t, target),
        )


def test_a_sweep_outcome_scoring_nan_is_not_folded_over_a_finite_estimate():
    # `not score <= current` let a NaN outcome replace a finite estimate
    from biahub.registration.engine.fallback import PassResult, _fold_sweep
    from biahub.registration.engine.series import SeriesResult

    kept = Transform.from_translation([1.0, 0.0, 0.0])
    result = SeriesResult(transforms={0: kept}, scores={0: 0.8})
    nan_outcome = PassResult(
        transform=Transform.from_translation([5.0, 0.0, 0.0]),
        score=float("nan"),
        accepted=True,
        source="trial_0",
        scores={},
        failures={},
    )
    _fold_sweep(result, 0, nan_outcome)
    assert result.transforms[0] is kept and result.scores[0] == 0.8
