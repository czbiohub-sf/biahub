import numpy as np

from biahub.core.transform import Transform
from biahub.registration.fallback import repair
from biahub.registration.run_journal import RunJournal
from biahub.registration.seed_policy import FixedSeed


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
        "far": FixedSeed(Transform.from_translation([0.0, 0.0, 0.0])),
        "close": FixedSeed(Transform.from_translation([9.0, 9.0, 9.0])),  # best
        "medium": FixedSeed(Transform.from_translation([5.0, 5.0, 5.0])),
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
        "far": FixedSeed(Transform.from_translation([0.0, 0.0, 0.0])),
        "medium": FixedSeed(Transform.from_translation([5.0, 5.0, 5.0])),
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
                raise RuntimeError("simulated candidate failure")
            return seed

    candidates = {
        "broken": FixedSeed(Transform.from_translation([999.0, 0.0, 0.0])),
        "good": FixedSeed(Transform.from_translation([9.0, 9.0, 9.0])),
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


def test_repair_skips_a_candidate_whose_seed_for_raises_without_failing_the_whole_pass():
    """E.g. `ConsensusSeed` raises when too few timepoints score well enough yet --
    that must be skipped like any other candidate failure, not crash `repair`."""
    target = np.array([10.0, 10.0, 10.0])

    class _RaisingSeedPolicy:
        def seed_for(self, t):
            raise ValueError("not enough good timepoints yet")

    candidates = {
        "not_ready": _RaisingSeedPolicy(),
        "good": FixedSeed(Transform.from_translation([9.0, 9.0, 9.0])),
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


def test_repair_records_to_the_journal_when_given_one():
    journal = RunJournal()
    target = np.array([10.0, 10.0, 10.0])
    candidates = {"close": FixedSeed(Transform.from_translation([9.0, 9.0, 9.0]))}

    repair(
        t=3,
        mov=None,
        ref=None,
        estimator=_EchoEstimator(),
        current_transform=Transform.from_translation([0.0, 0.0, 0.0]),
        current_score=_score_by_closeness(Transform.from_translation([0.0, 0.0, 0.0]), target),
        candidates=candidates,
        score_fn=lambda t: _score_by_closeness(t, target),
        journal=journal,
    )

    assert journal.attempted_this_run(3, pass_name="repair") is True
    assert journal.accepted_this_run(3, pass_name="repair") is True
