"""Drive estimation over a timepoint series.

Per timepoint: a `ReferencePolicy` picks what to compare against, a `SeedPolicy` picks
where to start, a `TransformEstimator` computes the transform and `score_fn` grades it.
A second pass flags timepoints against the run's own score distribution and offers each
to `fallback.repair`. The passes are separate functions because the first is
independent per timepoint (fan-out friendly) while the second needs every score in.

Series are indexed lazily (`series[t]` then `np.asarray`) so dask/zarr-backed inputs only
load the frames requested.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimationError, ScoreFn, TransformEstimator
from biahub.registration.fallback import RepairCandidates, RepairResult, repair
from biahub.registration.qc import select_flagged
from biahub.registration.reference_policy import ReferencePolicy
from biahub.registration.run_journal import RunJournal
from biahub.registration.seed_policy import SeedPolicy


@dataclass
class SeriesResult:
    # Accepted transforms by t. Doubles as the `history` a PreviousSeed reads from.
    transforms: dict[int, Transform] = field(default_factory=dict)
    # Every attempted t; nan where estimation failed and nothing repaired it.
    scores: dict[int, float] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    flagged: list[int] = field(default_factory=list)
    repairs: dict[int, RepairResult] = field(default_factory=dict)
    journal: RunJournal = field(default_factory=RunJournal)


OnTimepoint = Callable[[int, SeriesResult], None]


def estimate_series(
    mov,
    reference_policy: ReferencePolicy,
    estimator: TransformEstimator,
    seed_policy: SeedPolicy,
    score_fn: ScoreFn,
    time_indices: Iterable[int],
    history: dict[int, Transform] | None = None,
    journal: RunJournal | None = None,
    on_timepoint: OnTimepoint | None = None,
) -> SeriesResult:
    """One estimate per timepoint, in the given order.

    `history` is the caller-owned mapping a `PreviousSeed` reads from. It is filled here
    as timepoints are accepted, so handing the same dict to the seed policy is what makes
    propagation work; the result's `transforms` IS that dict. An `EstimationError` records
    nan plus the message for that timepoint and moves on.
    """
    result = SeriesResult(
        transforms=history if history is not None else {},
        journal=journal if journal is not None else RunJournal(),
    )
    for t in time_indices:
        mov_t = np.asarray(mov[t])
        ref_t = np.asarray(reference_policy.reference_for(mov, t))
        seed = seed_policy.seed_for(t)
        try:
            transform = estimator.estimate(mov_t, ref_t, seed=seed)
        except EstimationError as e:
            result.scores[t] = float("nan")
            result.errors[t] = f"{type(e).__name__}: {e}"
        else:
            result.transforms[t] = transform
            result.scores[t] = float(score_fn(transform, mov_t, ref_t))
        if on_timepoint is not None:
            on_timepoint(t, result)
    return result


def flag_series(
    result: SeriesResult,
    max_timepoints: int | None = None,
    k_mad: float = 2.0,
    floor: float = 0.80,
    hard_fail: float = 0.40,
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
        label="repair",
        max_timepoints=max_timepoints,
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
) -> RepairResult:
    """Offer one flagged timepoint to `repair` and fold an accepted result back in.

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
        journal=result.journal,
    )
    result.repairs[t] = outcome
    if outcome.accepted:
        result.transforms[t] = outcome.transform
        result.scores[t] = outcome.score
        result.errors.pop(t, None)
    return outcome


def repair_series(
    mov,
    reference_policy: ReferencePolicy,
    estimator: TransformEstimator,
    score_fn: ScoreFn,
    result: SeriesResult,
    candidates: RepairCandidates,
    max_timepoints: int | None = None,
    on_timepoint: OnTimepoint | None = None,
) -> SeriesResult:
    """Flag attempted timepoints and offer each to `repair`, in-process and in order.

    Unflagged timepoints are never touched.
    """
    for t in flag_series(result, max_timepoints):
        repair_timepoint(t, mov, reference_policy, estimator, score_fn, result, candidates)
        if on_timepoint is not None:
            on_timepoint(t, result)
    return result
