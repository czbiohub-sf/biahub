import numpy as np
import pytest

from biahub.core.transform import Transform
from biahub.registration.engine import (
    RunJournal,
    estimate_series,
    neighbour_consensus_config_candidates,
    polish,
    repair_series,
    sweep_timepoint,
    transforms_for_file,
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


def test_repair_series_replaces_a_transform_whose_score_is_nan():
    # Estimation succeeded but scoring found nothing to score (e.g. no beads): the
    # timepoint holds a transform with a NaN score, and any finite candidate must win.
    fix = Transform.from_translation([1.0, 0.0, 0.0])

    def score(transform, mov, ref):  # t=1 (mean 1) finds nothing to score unless fixed
        if transform is fix:
            return 0.7
        return float("nan") if round(float(mov.mean())) == 1 else 0.9

    class _EchoSeed:
        def estimate(self, mov, ref, seed=None):
            return seed

    mov = _constant_frames(3)
    result = estimate_series(mov, FixedFrame(0), _EchoSeed(), FixedSeed(IDENTITY), score, range(3))
    assert np.isnan(result.scores[1]) and 1 in result.transforms

    result = repair_series(
        mov,
        FixedFrame(0),
        _EchoSeed(),
        score,
        result,
        candidates=lambda t, history, scores, flagged: {"fix": FixedSeed(fix)},
    )

    assert result.transforms[1] is fix
    assert result.scores[1] == 0.7


def test_polish_improves_on_a_nan_score():
    better = Transform.from_translation([2.0, 0.0, 0.0])

    class _Refines:
        def estimate(self, mov, ref, seed=None):
            return better

    transform, score, rounds = polish(
        t=0,
        mov=np.zeros((2, 2, 2)),
        ref=np.zeros((2, 2, 2)),
        estimator=_Refines(),
        transform=IDENTITY,
        score=float("nan"),
        score_fn=lambda transform: 0.5 if transform is better else float("nan"),
        rounds=1,
    )
    assert transform is better and score == 0.5 and rounds == 1


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


class _StepEstimator:
    """Moves the seed's x translation one step towards `target` per call."""

    def __init__(self, target: float):
        self.target = target
        self.calls = 0

    def estimate(self, mov, ref, seed=None):
        self.calls += 1
        x = seed.translation[0]
        return Transform.from_translation([min(x + 1.0, self.target), 0.0, 0.0])


def _score_by_x(transform):
    return float(transform.translation[0])


def test_polish_keeps_improving_rounds_and_stops_at_the_first_flat_one():
    estimator = _StepEstimator(target=2.0)
    start = Transform.from_translation([0.0, 0.0, 0.0])
    journal = RunJournal()

    transform, score, rounds = polish(
        3, None, None, estimator, start, 0.0, _score_by_x, rounds=5, journal=journal
    )

    assert (score, rounds) == (2.0, 2)
    assert transform.translation[0] == 2.0
    assert estimator.calls == 3, "the third round did not improve, so polish stops there"
    assert journal.accepted_this_run(3, pass_name="polish")


def test_polish_is_capped_and_a_raising_round_keeps_the_accepted_transform():
    start = Transform.from_translation([0.0, 0.0, 0.0])
    capped = polish(0, None, None, _StepEstimator(9.0), start, 0.0, _score_by_x, rounds=1)
    assert capped[1:] == (1.0, 1)

    class _Raises:
        def estimate(self, mov, ref, seed=None):
            raise EstimationError("too few matches")

    transform, score, rounds = polish(0, None, None, _Raises(), start, 0.5, _score_by_x, 3)
    assert transform is start and (score, rounds) == (0.5, 0)


def test_repair_series_polishes_only_accepted_repairs():
    mov = _constant_frames(8)

    def score(transform, mov_t, ref_t):
        if round(float(mov_t.mean())) != 5:
            return 0.9
        return 0.2 + 0.1 * transform.translation[0]

    result = estimate_series(
        mov, FixedFrame(0), _StepEstimator(3.0), FixedSeed(IDENTITY), score, range(8)
    )
    result.transforms = {t: IDENTITY for t in result.transforms}
    result.scores[5] = 0.2
    result = repair_series(
        mov,
        FixedFrame(0),
        _StepEstimator(3.0),
        score,
        result,
        candidates=lambda t, history, scores, flagged: {"fix": FixedSeed(IDENTITY)},
        polish_rounds=3,
    )

    repair = result.repairs[5]
    assert repair.accepted and repair.reseed_score == pytest.approx(0.3)
    assert repair.polish_rounds == 2 and repair.source == "fix+polish2"
    assert result.scores[5] == pytest.approx(0.5)
    assert result.transforms[5].translation[0] == 3.0
    assert list(result.repairs) == [5]


class _FixedEstimator:
    def __init__(self, x: float):
        self.x = x

    def estimate(self, mov, ref, seed=None):
        return Transform.from_translation([self.x, 0.0, 0.0])


def _score_by_x_on_frame(transform, mov, ref):
    return 0.1 * float(transform.translation[0])


def _series_with_one_poor_timepoint():
    mov = _constant_frames(4)
    result = estimate_series(
        mov,
        FixedFrame(0),
        _FixedEstimator(1.0),
        FixedSeed(IDENTITY),
        _score_by_x_on_frame,
        range(4),
    )
    return mov, result


def test_sweep_timepoint_keeps_the_best_trial_that_beats_the_estimate():
    mov, result = _series_with_one_poor_timepoint()
    trials = {"a": _FixedEstimator(0.5), "b": _FixedEstimator(4.0), "c": _FixedEstimator(3.0)}

    outcome = sweep_timepoint(
        2, mov, FixedFrame(0), trials, IDENTITY, _score_by_x_on_frame, result
    )

    assert outcome.accepted and outcome.source == "b"
    assert outcome.scores == pytest.approx({"a": 0.05, "b": 0.4, "c": 0.3})
    assert result.scores[2] == pytest.approx(0.4)
    assert result.provenance[2] == "sweep:b"
    assert result.journal.accepted_this_run(2, pass_name="sweep")


def test_sweep_competes_with_repair_from_the_pre_fallback_score():
    mov, result = _series_with_one_poor_timepoint()
    baseline = result.scores[2]
    result.transforms[2] = Transform.from_translation([6.0, 0.0, 0.0])  # a repair landed
    result.scores[2], result.provenance[2] = 0.6, "t-1"

    outcome = sweep_timepoint(
        2,
        mov,
        FixedFrame(0),
        {"b": _FixedEstimator(4.0)},
        IDENTITY,
        _score_by_x_on_frame,
        result,
        baseline_score=baseline,
    )

    assert outcome.accepted, "the sweep beat the estimate it started from"
    assert result.provenance[2] == "t-1" and result.scores[2] == 0.6, "but not the repair"
    assert result.sweeps[2] is outcome


def test_sweep_skips_a_raising_trial_and_names_it():
    class _Raises:
        def estimate(self, mov, ref, seed=None):
            raise EstimationError("too few matches")

    mov, result = _series_with_one_poor_timepoint()
    outcome = sweep_timepoint(
        2, mov, FixedFrame(0), {"bad": _Raises()}, IDENTITY, _score_by_x_on_frame, result
    )
    assert not outcome.accepted and outcome.source == "unchanged"
    assert outcome.failures == {"bad": "EstimationError: too few matches"}
    assert 2 not in result.provenance


def _shift_x(dx):
    return Transform.from_translation([0.0, 0.0, float(dx)])


def test_transforms_for_file_previous_treats_a_missing_step_as_identity():
    # Steps of +1 in x per timepoint; t=2's step is missing (estimate failed, not repaired).
    from biahub.registration.engine import SeriesResult

    result = SeriesResult(transforms={0: IDENTITY, 1: _shift_x(1), 3: _shift_x(1)})
    chained = transforms_for_file(result, [0, 1, 2, 3], IDENTITY, "previous")
    # t=2 adds no drift; t=3 adds its own single step: 0, 1, 1, 2 -- not 0, 1, 2, 3.
    assert [t.translation[2] for t in chained] == [0.0, 1.0, 1.0, 2.0]


def test_transforms_for_file_absolute_frames_fill_from_the_nearest_earlier():
    from biahub.registration.engine import SeriesResult

    result = SeriesResult(transforms={0: _shift_x(5), 2: _shift_x(7)})
    filled = transforms_for_file(result, [0, 1, 2], IDENTITY, "first")
    assert [t.translation[2] for t in filled] == [5.0, 5.0, 7.0]
