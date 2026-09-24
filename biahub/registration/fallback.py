"""Fallback passes: repair (and later sweep/polish) for flagged timepoints.

Generalizes PR #339's repair pass (reseed from t-1/t+1/consensus/config, keep whichever
scores best) into a list of named `SeedPolicy` candidates -- any `SeedPolicy` works as a
candidate source, not just the ones `beads.py` hardcodes. This is also how repair stays
estimator-agnostic: it only calls `TransformEstimator.estimate(mov, ref, seed=...)`, so
it works the same way for beads, ants, or any future seeded estimator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.estimators import TransformEstimator
from biahub.registration.run_journal import RunJournal
from biahub.registration.seed_policy import SeedPolicy


@dataclass
class RepairResult:
    transform: Transform
    score: float
    accepted: bool
    source: str  # name of the winning candidate, or "unchanged"
    scores: dict[str, float] = field(default_factory=dict)  # every candidate that ran
    failures: dict[str, str] = field(default_factory=dict)  # candidate -> "Type: message"


def repair(
    t: int,
    mov: ArrayLike,
    ref: ArrayLike,
    estimator: TransformEstimator,
    current_transform: Transform,
    current_score: float,
    candidates: dict[str, SeedPolicy],
    score_fn: Callable[[Transform], float],
    journal: RunJournal | None = None,
) -> RepairResult:
    """Try each candidate seed for timepoint `t`; keep whichever scores best.

    Every candidate is tried in the given order. A candidate whose `seed_for()` or
    `estimate()` raises -- e.g. `ConsensusSeed` when too few timepoints score well enough
    yet -- is skipped rather than aborting the repair, and the exception is kept in
    `RepairResult.failures` (and the journal) so a silently-broken candidate is visible
    rather than indistinguishable from one that merely lost. The best-scoring result
    wins, ties broken by candidate order; `accepted=True` only if it strictly beats
    `current_score`, otherwise the original transform/score are returned unchanged.
    """
    best_name = "unchanged"
    best_transform = current_transform
    best_score = current_score
    scores: dict[str, float] = {}
    failures: dict[str, str] = {}

    for name, seed_policy in candidates.items():
        try:
            seed = seed_policy.seed_for(t)
            candidate_transform = estimator.estimate(mov, ref, seed=seed)
            candidate_score = score_fn(candidate_transform)
        except Exception as e:  # noqa: BLE001
            failures[name] = f"{type(e).__name__}: {e}"
            continue
        scores[name] = candidate_score
        if candidate_score is not None and candidate_score > best_score:
            best_name, best_transform, best_score = name, candidate_transform, candidate_score

    accepted = best_name != "unchanged"
    if journal is not None:
        journal.record(
            t=t,
            pass_name="repair",
            before_score=current_score,
            after_score=best_score,
            accepted=accepted,
            failures=failures,
        )
    return RepairResult(
        transform=best_transform,
        score=best_score,
        accepted=accepted,
        source=best_name,
        scores=scores,
        failures=failures,
    )
