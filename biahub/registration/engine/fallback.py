"""What to do when a timepoint looks wrong, in memory: flag, repair, sweep, polish.

Flagging picks timepoints scoring below the run's adaptive line; repair re-estimates
them from other starts (a non-flagged neighbour, the run's consensus, the initial
transform); sweep from other method settings; polish refines an accepted candidate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields

import click
import numpy as np
import pandas as pd

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.engine.policies import (
    ConsensusInitialTransform,
    FixedInitialTransform,
    InitialTransformPolicy,
    ReferencePolicy,
)
from biahub.registration.engine.series import (
    SeriesResult,
)
from biahub.registration.estimators import (
    CANDIDATE_FAILURES,
    ScoreFn,
    TransformEstimator,
    is_higher,
    select_best,
)
from biahub.settings import (
    FlagSettings,
)

# The flagging defaults are FlagSettings' (what a config sets); one source for both.
_FLAG_DEFAULTS = FlagSettings()


HARD_FAIL_SCORE = _FLAG_DEFAULTS.hard_fail


FLAG_FLOOR_SCORE = _FLAG_DEFAULTS.floor


FLAG_K_MAD = _FLAG_DEFAULTS.k_mad


def flag_timepoints(
    scores: ArrayLike,
    k_mad: float = FLAG_K_MAD,
    floor: float = FLAG_FLOOR_SCORE,
    hard_fail: float = HARD_FAIL_SCORE,
) -> pd.DataFrame:
    """One row per timepoint, flagged against the run's own median - k*MAD line.

    Flagged when below the adaptive line AND below the absolute floor, or below
    `hard_fail`, or missing a score entirely.
    """
    s = np.asarray(scores, dtype=float)
    finite = s[np.isfinite(s)]
    if not len(finite):
        raise ValueError("no finite scores to flag against")
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    line = median - k_mad * mad

    rows = []
    for t, score in enumerate(s):
        reasons = []
        if not np.isfinite(score):
            reasons.append("no_score")
        else:
            if score < line and score < floor:
                reasons.append("below_adaptive_line")
            if score < hard_fail:
                reasons.append("below_hard_fail")
        rows.append(
            {
                "t": t,
                "quality_score": score,
                "flagged": bool(reasons),
                "reasons": ";".join(reasons),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update(
        {
            "median": median,
            "mad": mad,
            "adaptive_line": line,
            "floor": floor,
            "hard_fail": hard_fail,
        }
    )
    return out


def select_flagged(
    score_col: ArrayLike,
    label: str,
    k_mad: float = FLAG_K_MAD,
    floor: float = FLAG_FLOOR_SCORE,
    hard_fail: float = HARD_FAIL_SCORE,
) -> tuple[list[int], dict]:
    """Timepoints for a fallback pass to act on, from the adaptive median-2*MAD line.

    Shared by every fallback pass so they can't drift apart in how they choose work, and
    so no pass carries its own fixed score threshold.

    Returns (flagged timepoints, run statistics) -- the statistics are returned rather
    than left for callers to recompute, so whatever gets logged is provably the same
    numbers the selection used. Each pass caps its own share (`cap_worst`).
    """
    score_col = np.asarray(score_col, dtype=float)
    n_t = len(score_col)
    try:
        flags = flag_timepoints(score_col, k_mad=k_mad, floor=floor, hard_fail=hard_fail)
    except ValueError:
        click.echo(f"{label}: no finite scores to flag against; skipping.")
        return [], {}
    flagged = [int(t) for t in flags.loc[flags["flagged"], "t"]]
    if not flagged:
        click.echo(f"{label}: nothing flagged, nothing to do.")
        return [], dict(flags.attrs)

    click.echo(
        f"{label}: {len(flagged)} of {n_t} timepoints flagged "
        f"(adaptive line {flags.attrs['adaptive_line']:.3f}, median "
        f"{flags.attrs['median']:.3f}) -> {flagged}"
    )
    return flagged, dict(flags.attrs)


def cap_worst(
    timepoints: list[int], scores: dict[int, float], max_timepoints: int | None
) -> list[int]:
    """Keep the worst `max_timepoints` of `timepoints` by score, naming the ones left out.

    It always says which ones it dropped: a capped pass that logs nothing about the drop
    would read as full coverage.
    """
    if max_timepoints is None or len(timepoints) <= max_timepoints:
        return list(timepoints)
    worst = sorted(timepoints, key=lambda t: np.nan_to_num(scores[t], nan=-1.0))
    dropped = sorted(worst[max_timepoints:])
    kept = sorted(worst[:max_timepoints])
    click.echo(
        f"  capped at max_timepoints={max_timepoints}; taking the worst {len(kept)} "
        f"and LEAVING {len(dropped)} untouched: {dropped}"
    )
    return kept


RepairCandidates = Callable[
    [int, dict[int, Transform], dict[int, float], set[int]], dict[str, InitialTransformPolicy]
]


def neighbour_consensus_initial_candidates(
    initial: Transform,
    consensus_score_threshold: float = 0.75,
    consensus_min_good: int = 5,
    order: tuple[str, ...] = ("t-1", "t+1", "consensus", "initial"),
) -> RepairCandidates:
    """Build the repair candidate set in the given order.

    "t-1" / "t+1": non-flagged neighbours; "consensus": the run's consensus geometry
    (named `consensus_full` in results); "initial": the timepoint's initial transform.
    """

    def candidates(
        t: int, history: dict[int, Transform], scores: dict[int, float], flagged: set[int]
    ) -> dict[str, InitialTransformPolicy]:
        out: dict[str, InitialTransformPolicy] = {}
        for name in order:
            if name in ("t-1", "t+1"):
                idx = t - 1 if name == "t-1" else t + 1
                if idx in history and idx not in flagged:
                    out[name] = FixedInitialTransform(history[idx])
            elif name == "consensus":
                out["consensus_full"] = ConsensusInitialTransform(
                    history=history,
                    scores=scores,
                    score_threshold=consensus_score_threshold,
                    min_good=consensus_min_good,
                )
            elif name == "initial":
                out["initial"] = FixedInitialTransform(initial)
            else:
                raise ValueError(f"unknown repair candidate {name!r}")
        return out

    return candidates


@dataclass
class PassResult:
    """Outcome of one fallback pass (repair or sweep) at one timepoint."""

    transform: Transform
    score: float
    accepted: bool
    source: str  # name of the winning candidate, or "unchanged"
    scores: dict[str, float] = field(default_factory=dict)  # every candidate that ran
    failures: dict[str, str] = field(default_factory=dict)  # candidate -> "Type: message"
    polish_rounds: int = 0  # polish rounds that improved the accepted candidate
    # the accepted repair candidate's score, before polish refined it
    repair_transform_score: float | None = None
    # score.report metrics of the accepted transform
    report: dict[str, float | None] | None = None


@dataclass
class PassRecord:
    """One flagged timepoint's repair or sweep, as a job writes it (`repairs/<t>.json`).

    Read back by finalize as a `PassResult` (`to_result`); one type for the writers and
    the reader. `matrix` is the accepted candidate's transform, else None.
    """

    t: int
    matrix: list | None
    score: float | None
    accepted: bool
    source: str
    candidate_scores: dict[str, float | None]
    candidate_failures: dict[str, str]
    polish_rounds: int = 0
    repair_transform_score: float | None = None
    report: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> PassRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def to_result(self, transform_type: str, fallback: Transform) -> PassResult:
        """Return this pass as a `PassResult`; `fallback` stands in when none was accepted."""
        return PassResult(
            transform=(
                Transform(self.matrix, transform_type=transform_type)
                if self.matrix is not None
                else fallback
            ),
            score=-np.inf if self.score is None else self.score,
            accepted=self.accepted,
            source=self.source,
            scores={
                k: float("nan") if v is None else v for k, v in self.candidate_scores.items()
            },
            failures=self.candidate_failures,
            polish_rounds=self.polish_rounds,
            repair_transform_score=self.repair_transform_score,
            report=self.report,
        )


def _best_of(
    attempts: dict[str, Callable[[], Transform]],
    current_transform: Transform,
    current_score: float,
    score_fn: Callable[[Transform], float],
) -> PassResult:
    """Run each attempt in order; keep the best-scoring one if it strictly beats current.

    An attempt that raises is skipped, not fatal, and its exception is kept in
    `PassResult.failures` so a broken candidate is distinguishable
    from one that merely lost. Ties go to the earlier attempt.
    """

    def scored(attempt):
        def run():
            candidate = attempt()
            return candidate, score_fn(candidate)

        return run

    chosen = select_best(
        {name: scored(attempt) for name, attempt in attempts.items()},
        current=("unchanged", current_transform, current_score),
    )
    return PassResult(
        transform=chosen.transform,
        score=chosen.score,
        accepted=chosen.name != "unchanged",
        source=chosen.name,
        scores=chosen.scores,
        failures=chosen.failures,
    )


def repair(
    t: int,
    mov: ArrayLike,
    ref: ArrayLike,
    estimator: TransformEstimator,
    current_transform: Transform,
    current_score: float,
    candidates: dict[str, InitialTransformPolicy],
    score_fn: Callable[[Transform], float],
) -> PassResult:
    """Re-estimate timepoint `t` from each candidate initial transform; keep the best score.

    A candidate whose `initial_for()` raises -- e.g. `ConsensusInitialTransform` when
    too few timepoints score well enough yet -- is skipped like one whose `estimate()`
    raises.
    """
    return _best_of(
        {
            name: (
                lambda policy=policy: estimator.estimate(
                    mov, ref, initial_transform=policy.initial_for(t)
                )
            )
            for name, policy in candidates.items()
        },
        current_transform,
        current_score,
        score_fn,
    )


def sweep(
    mov: ArrayLike,
    ref: ArrayLike,
    trials: dict[str, TransformEstimator],
    initial_transform: Transform,
    current_transform: Transform,
    current_score: float,
    score_fn: Callable[[Transform], float],
) -> PassResult:
    """Re-estimate with each trial's estimator from `initial_transform`; keep the best.

    Repair changes the initial transform and keeps the method; the sweep keeps the
    initial transform and changes the method's settings, for the timepoints the run's
    one parameter set does not serve. `score_fn` is the run's own, so every trial is
    judged on the same scale.
    """
    return _best_of(
        {
            name: (
                lambda estimator=estimator: estimator.estimate(
                    mov, ref, initial_transform=initial_transform
                )
            )
            for name, estimator in trials.items()
        },
        current_transform,
        current_score,
        score_fn,
    )


def polish(
    mov: ArrayLike,
    ref: ArrayLike,
    estimator: TransformEstimator,
    transform: Transform,
    score: float,
    score_fn: Callable[[Transform], float],
    rounds: int,
) -> tuple[Transform, float, int]:
    """Restart the estimator from `transform` and refine again, up to `rounds` times.

    Repair candidates are coarse initial transforms and detection runs in the
    initial-transform-warped space, so an estimate started from the accepted transform
    sees better nodes than the one that produced it. A round is kept only on a strict
    score gain; the first round that fails to improve (or raises) ends the pass. Returns
    (transform, score, improving rounds).
    """
    improved = 0
    for _ in range(rounds):
        try:
            candidate = estimator.estimate(mov, ref, initial_transform=transform)
            candidate_score = score_fn(candidate)
        except CANDIDATE_FAILURES:  # a failed round keeps the accepted transform
            break
        if not is_higher(candidate_score, score):
            break
        transform, score, improved = candidate, candidate_score, improved + 1
    return transform, score, improved


def flag_series(
    result: SeriesResult,
    k_mad: float = FLAG_K_MAD,
    floor: float = FLAG_FLOOR_SCORE,
    hard_fail: float = HARD_FAIL_SCORE,
) -> list[int]:
    """Flag attempted timepoints against the run's own score distribution.

    Only timepoints that were attempted can be flagged; gaps in `time_indices` are not
    "missing scores". Sets and returns `result.flagged`.
    """
    attempted = sorted(result.scores)
    if not attempted:
        result.flagged = []
        return result.flagged
    flagged_positions, _stats = select_flagged(
        np.array([result.scores[t] for t in attempted]),
        label="flag",  # flagging decides `unreliable` whether or not repair runs
        k_mad=k_mad,
        floor=floor,
        hard_fail=hard_fail,
    )
    result.flagged = [attempted[i] for i in flagged_positions]
    return result.flagged


def repair_timepoint(
    t: int,
    mov,
    reference_policy: ReferencePolicy,
    estimator: TransformEstimator,
    score_fn: ScoreFn,
    result: SeriesResult,
    candidates: RepairCandidates,
    polish_rounds: int = 0,
) -> PassResult:
    """Offer one flagged timepoint to `repair`, `polish` an accepted result, fold it in.

    A timepoint whose estimate failed has no current transform, so any candidate with a
    finite score is an improvement for it.
    """
    mov_t = np.asarray(mov[t])
    ref_t = np.asarray(reference_policy.reference_for(mov, t))
    current = result.transforms.get(t)
    outcome = repair(
        t=t,
        mov=mov_t,
        ref=ref_t,
        estimator=estimator,
        current_transform=current if current is not None else Transform.identity(mov_t.ndim),
        current_score=result.scores[t] if current is not None else -np.inf,
        candidates=candidates(t, result.transforms, result.scores, set(result.flagged)),
        score_fn=lambda transform, m=mov_t, r=ref_t: score_fn(transform, m, r),
    )
    if outcome.accepted and polish_rounds:
        outcome.repair_transform_score = outcome.score
        outcome.transform, outcome.score, outcome.polish_rounds = polish(
            mov_t,
            ref_t,
            estimator,
            outcome.transform,
            outcome.score,
            lambda transform, m=mov_t, r=ref_t: score_fn(transform, m, r),
            polish_rounds,
        )
        if outcome.polish_rounds:
            outcome.source += f"+polish{outcome.polish_rounds}"
    result.repairs[t] = outcome
    if outcome.accepted:
        _accept(result, t, outcome.transform, outcome.score, outcome.source, outcome.report)
    return outcome


def _accept(
    result: SeriesResult,
    t: int,
    transform: Transform,
    score: float,
    source: str,
    report: dict | None = None,
):
    result.transforms[t] = transform
    result.scores[t] = score
    result.errors.pop(t, None)
    result.provenance[t] = source
    if report is None:
        result.reports.pop(t, None)
    else:
        result.reports[t] = report


def sweep_timepoint(
    t: int,
    mov,
    reference_policy: ReferencePolicy,
    trials: dict[str, TransformEstimator],
    initial_transform: Transform,
    score_fn: ScoreFn,
    result: SeriesResult,
    baseline_score: float | None = None,
) -> PassResult:
    """Offer one flagged timepoint to `sweep`; fold the result in if it beats the current one.

    The sweep is judged against `baseline_score` (the timepoint's score before any
    fallback, so it competes with repair rather than following it; default: its current
    score), and folded in only if it also beats whatever the timepoint holds now.
    """
    mov_t = np.asarray(mov[t])
    ref_t = np.asarray(reference_policy.reference_for(mov, t))
    current = result.transforms.get(t)
    if baseline_score is None:
        baseline_score = result.scores[t] if current is not None else -np.inf
    outcome = sweep(
        mov=mov_t,
        ref=ref_t,
        trials=trials,
        initial_transform=initial_transform,
        current_transform=current if current is not None else Transform.identity(mov_t.ndim),
        current_score=-np.inf if not np.isfinite(baseline_score) else baseline_score,
        score_fn=lambda transform, m=mov_t, r=ref_t: score_fn(transform, m, r),
    )
    _fold_sweep(result, t, outcome)
    return outcome


def _fold_sweep(result: SeriesResult, t: int, outcome: PassResult) -> None:
    result.sweeps[t] = outcome
    current_score = result.scores.get(t, float("nan"))
    if outcome.accepted and is_higher(outcome.score, current_score):
        _accept(
            result,
            t,
            outcome.transform,
            outcome.score,
            f"sweep:{outcome.source}",
            outcome.report,
        )
