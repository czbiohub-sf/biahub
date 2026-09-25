import numpy as np

from biahub.core.transform import Transform
from biahub.registration.engine import (
    estimate_series,
    neighbour_consensus_config_candidates,
    repair_series,
)
from biahub.registration.estimators import EstimationError
from biahub.registration.policies import (
    ConsensusSeed,
    CrossChannel,
    FixedFrame,
    FixedSeed,
    PreviousFrame,
    PreviousSeed,
)

IDENTITY = Transform.identity(3)


def _constant_frames(n_t: int, scale: float = 1.0) -> np.ndarray:
    """Frame t is filled with the value scale*t, so a frame's mean identifies it."""
    return np.stack([np.full((2, 2, 2), scale * t, dtype=float) for t in range(n_t)])


class _MeanShiftEstimator:
    """Toy estimator whose answer depends on which reference it was handed: the shift
    from mov's mean to ref's mean. Records every call."""

    def __init__(self, fail_for_mov_means=()):
        self.fail_for_mov_means = set(fail_for_mov_means)
        self.calls = []

    def estimate(self, mov, ref, seed=None):
        self.calls.append({"mov": float(mov.mean()), "ref": float(ref.mean()), "seed": seed})
        if round(float(mov.mean())) in self.fail_for_mov_means:
            raise EstimationError("too few nodes")
        return Transform.from_translation([float(ref.mean() - mov.mean()), 0.0, 0.0])


class _EchoEstimator:
    """Returns the seed it is given (identity when none)."""

    def estimate(self, mov, ref, seed=None):
        return seed if seed is not None else IDENTITY


def _finite_score(transform, mov, ref):
    return 1.0


def test_reference_policy_decides_what_each_timepoint_registers_against():
    mov = _constant_frames(4)
    x_shift = lambda result: [result.transforms[t].translation[0] for t in range(4)]  # noqa: E731

    fixed = estimate_series(
        mov, FixedFrame(0), _MeanShiftEstimator(), FixedSeed(IDENTITY), _finite_score, range(4)
    )
    previous = estimate_series(
        mov,
        PreviousFrame(),
        _MeanShiftEstimator(),
        FixedSeed(IDENTITY),
        _finite_score,
        range(4),
    )
    cross = estimate_series(
        mov,
        CrossChannel(_constant_frames(4, scale=10.0)),
        _MeanShiftEstimator(),
        FixedSeed(IDENTITY),
        _finite_score,
        range(4),
    )

    assert x_shift(fixed) == [0.0, -1.0, -2.0, -3.0]
    assert x_shift(previous) == [0.0, -1.0, -1.0, -1.0]
    assert x_shift(cross) == [0.0, 9.0, 18.0, 27.0]


def test_failed_timepoint_is_recorded_and_the_series_continues():
    result = estimate_series(
        _constant_frames(4),
        FixedFrame(0),
        _MeanShiftEstimator(fail_for_mov_means={1}),
        FixedSeed(IDENTITY),
        _finite_score,
        range(4),
    )

    assert sorted(result.transforms) == [0, 2, 3]
    assert np.isnan(result.scores[1])
    assert result.errors == {1: "EstimationError: too few nodes"}


def test_previous_seed_propagates_through_the_shared_history_and_resets_after_a_failure():
    estimator = _MeanShiftEstimator(fail_for_mov_means={2})
    history = {}
    config_seed = Transform.from_translation([100.0, 0.0, 0.0])

    result = estimate_series(
        _constant_frames(5),
        FixedFrame(0),
        estimator,
        PreviousSeed(history, fallback=FixedSeed(config_seed)),
        _finite_score,
        range(5),
        history=history,
    )

    assert result.transforms is history
    seeds = [call["seed"] for call in estimator.calls]
    assert seeds[0] is config_seed
    assert seeds[1] is history[0]
    assert seeds[2] is history[1]
    assert seeds[3] is config_seed, "t=2 failed, so t=3 must restart from the fallback"
    assert seeds[4] is history[3]


def test_series_is_indexed_one_frame_at_a_time():
    class _LazySeries:
        def __init__(self, frames):
            self._frames = frames
            self.materialized = []

        def __getitem__(self, t):
            self.materialized.append(t)
            return self._frames[t]

        def __array__(self, *args, **kwargs):
            raise AssertionError("whole series materialized")

    mov = _LazySeries(_constant_frames(3))
    ref = _LazySeries(_constant_frames(3))
    estimate_series(
        mov,
        CrossChannel(ref),
        _MeanShiftEstimator(),
        FixedSeed(IDENTITY),
        _finite_score,
        [0, 2],
    )
    assert mov.materialized == [0, 2]
    assert ref.materialized == [0, 2]


def _score_penalising_frame_5(transform, mov, ref):
    if transform.translation[0] == 1.0:
        return 0.95
    return 0.2 if round(float(mov.mean())) == 5 else 0.9


def test_repair_series_touches_only_flagged_timepoints_and_journals_them():
    mov = _constant_frames(8)
    result = estimate_series(
        mov,
        FixedFrame(0),
        _EchoEstimator(),
        FixedSeed(IDENTITY),
        _score_penalising_frame_5,
        range(8),
    )
    assert result.scores[5] == 0.2

    fix = Transform.from_translation([1.0, 0.0, 0.0])
    result = repair_series(
        mov,
        FixedFrame(0),
        _EchoEstimator(),
        _score_penalising_frame_5,
        result,
        candidates=lambda t, history, scores, flagged: {"fix": FixedSeed(fix)},
    )

    assert result.flagged == [5]
    assert list(result.repairs) == [5]
    assert result.repairs[5].accepted and result.repairs[5].source == "fix"
    assert result.transforms[5] is fix and result.scores[5] == 0.95
    assert [result.transforms[t] for t in range(8) if t != 5] == [IDENTITY] * 7
    assert result.journal.accepted_this_run(5, pass_name="repair")
    assert not result.journal.attempted_this_run(4)


def test_repair_series_rescues_a_timepoint_whose_estimate_failed():
    class _FailsUnseeded:
        def estimate(self, mov, ref, seed=None):
            if round(float(mov.mean())) == 2 and seed.is_identity:
                raise EstimationError("too few matches")
            return seed

    mov = _constant_frames(4)
    result = estimate_series(
        mov, FixedFrame(0), _FailsUnseeded(), FixedSeed(IDENTITY), _finite_score, range(4)
    )
    assert 2 in result.errors

    fix = Transform.from_translation([1.0, 0.0, 0.0])
    result = repair_series(
        mov,
        FixedFrame(0),
        _FailsUnseeded(),
        _finite_score,
        result,
        candidates=lambda t, history, scores, flagged: {"fix": FixedSeed(fix)},
    )

    assert result.flagged == [2]
    assert result.transforms[2] is fix
    assert result.scores[2] == 1.0
    assert 2 not in result.errors


def test_neighbour_consensus_config_candidates_skips_flagged_neighbours():
    history = {t: Transform.from_translation([float(t), 0.0, 0.0]) for t in range(7)}
    scores = {t: 0.9 for t in range(7)}
    config_seed = Transform.from_translation([-1.0, 0.0, 0.0])
    factory = neighbour_consensus_config_candidates(config_seed)

    middle = factory(4, history, scores, flagged={3, 4})
    assert list(middle) == ["t+1", "consensus_full", "config_seed"]
    assert middle["t+1"].seed_for(4) is history[5]
    assert isinstance(middle["consensus_full"], ConsensusSeed)
    assert middle["config_seed"].seed_for(4) is config_seed

    first = factory(0, history, scores, flagged={0})
    assert list(first) == ["t+1", "consensus_full", "config_seed"]
