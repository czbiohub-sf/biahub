"""Estimate every timepoint of a series, in memory: no files, no cluster.

`estimate_series` estimates each timepoint from its own initial transform (independent);
`estimate_propagated` estimates them in order, each from the previous one's result. For
each timepoint the reference policy gives what it is compared against and the
initial-transform policy where it starts; the estimator registers that pair.
"""

from __future__ import annotations

import inspect
import json
import uuid

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from biahub.core.transform import Transform
from biahub.registration.engine.policies import (
    FixedFrame,
    InitialTransformPolicy,
    NeighbourMedian,
    PreviousFrame,
    ReferencePolicy,
    is_empty,
)
from biahub.registration.estimators import (
    EstimationError,
    ScoreFn,
    TransformEstimator,
)

if TYPE_CHECKING:  # an annotation only
    from biahub.registration.engine.fallback import PassResult


@dataclass
class Attempt:
    run_id: str
    t: int
    pass_name: str
    before_score: float
    after_score: float
    accepted: bool
    # candidate name -> "ExceptionType: message" for candidates that raised
    failures: dict[str, str] = field(default_factory=dict)


class RunJournal:
    """Records fallback-pass attempts for one timepoint series, tagged with a run id.

    A fresh `RunJournal()` generates a new run id. Loading from disk
    (`RunJournal.load(path)`) with no `run_id` given ALSO starts a fresh run id while
    preserving prior attempts in history -- so `attempted_this_run(t)` correctly
    returns False for anything recorded under a different (stale) run, forcing
    re-attempt on an ordinary restart. Explicitly continuing the *same* run (e.g.
    resubmitting a job that was killed mid-run, without wanting to redo already-accepted
    work) requires the caller to pass that run's id back in -- resume semantics are
    something the caller decides, never an implicit default.
    """

    def __init__(self, run_id: str | None = None):
        self.current_run_id = run_id or uuid.uuid4().hex
        self.attempts: list[Attempt] = []

    def record(
        self,
        t: int,
        pass_name: str,
        before_score: float,
        after_score: float,
        accepted: bool,
        failures: dict[str, str] | None = None,
    ) -> None:
        self.attempts.append(
            Attempt(
                self.current_run_id,
                t,
                pass_name,
                before_score,
                after_score,
                accepted,
                failures=dict(failures or {}),
            )
        )

    def attempted_this_run(self, t: int, pass_name: str | None = None) -> bool:
        return any(
            a.t == t
            and a.run_id == self.current_run_id
            and (pass_name is None or a.pass_name == pass_name)
            for a in self.attempts
        )

    def accepted_this_run(self, t: int, pass_name: str | None = None) -> bool:
        return any(
            a.t == t
            and a.run_id == self.current_run_id
            and a.accepted
            and (pass_name is None or a.pass_name == pass_name)
            for a in self.attempts
        )

    def to_dict(self) -> dict:
        return {"attempts": [asdict(a) for a in self.attempts]}

    @classmethod
    def from_dict(cls, data: dict, run_id: str | None = None) -> RunJournal:
        journal = cls(run_id=run_id)
        journal.attempts = [Attempt(**a) for a in data.get("attempts", [])]
        return journal

    def save(self, path: Path) -> None:
        from biahub.registration.engine.core import _write_json  # files: core's (no cycle)

        _write_json(path, self.to_dict(), indent=2)

    @classmethod
    def load(cls, path: Path, run_id: str | None = None) -> RunJournal:
        if not path.exists():
            return cls(run_id=run_id)
        return cls.from_dict(json.loads(path.read_text()), run_id=run_id)


def _beats(candidate: float | None, current: float | None) -> bool:
    """Return whether a finite candidate beats a missing / NaN or strictly lower score."""
    if candidate is None or not np.isfinite(candidate):
        return False
    return current is None or not np.isfinite(current) or candidate > current


@dataclass
class SeriesResult:
    # Accepted transforms by t.
    transforms: dict[int, Transform] = field(default_factory=dict)
    # Every attempted t; nan where estimation failed and nothing repaired it.
    scores: dict[int, float] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    # score.report metrics of each accepted transform, from the job that produced it.
    reports: dict[int, dict[str, float | None]] = field(default_factory=dict)
    flagged: list[int] = field(default_factory=list)
    repairs: dict[int, PassResult] = field(default_factory=dict)
    sweeps: dict[int, PassResult] = field(default_factory=dict)
    # How an accepted fallback transform was reached, e.g. "consensus_full+polish1".
    provenance: dict[int, str] = field(default_factory=dict)
    # Timepoints with no transform of their own, and where the stand-in written for them
    # came from ("t=81", "initial", "identity"); set by `transforms_for_file`.
    filled_from: dict[int, str] = field(default_factory=dict)
    # A stand-in decided while estimating (propagation: a failed timepoint returns the
    # seed it started from), with its source; used instead of the input seed.
    stand_ins: dict[int, tuple[Transform, str]] = field(default_factory=dict)
    # A stack's blocks at each estimated t, from its record (see ChainedEstimator).
    blocks: dict[int, list[dict]] = field(default_factory=dict)
    # From a file: "file" (its transform kept as it is) or "file+refined".
    initialized_from: dict[int, str] = field(default_factory=dict)
    # With reference.anchor: each timepoint's centroid from its record (None: none found),
    # and the anchor `finalize_run` added (centroid, smoothed, translation).
    centroids: dict[int, list[float] | None] = field(default_factory=dict)
    anchors: dict[int, dict[str, list[float] | None]] = field(default_factory=dict)
    # With reference.leftover_motion: each timepoint's raw shift from the run's previous
    # timepoint, from its record ({"from": t, "shift": voxels or None}), and the motion
    # `finalize_run` derived from them.
    raw_shifts: dict[int, dict] = field(default_factory=dict)
    leftover_motion: dict | None = None
    journal: RunJournal = field(default_factory=RunJournal)


OnTimepoint = Callable[[int, SeriesResult], None]


def estimate_series(
    mov,
    reference_policy: ReferencePolicy,
    estimator: TransformEstimator,
    initial_policy: InitialTransformPolicy,
    score_fn: ScoreFn,
    time_indices: Iterable[int],
    journal: RunJournal | None = None,
    on_timepoint: OnTimepoint | None = None,
    keep_initial: bool = False,
) -> SeriesResult:
    """One estimate per timepoint, in the given order (propagation: `estimate_propagated`).

    An `EstimationError` records nan plus the message for that timepoint and moves on.
    With `keep_initial` (an initial transform from a file, an earlier result) it competes
    as it is (`_keep_initial`).
    """
    result = SeriesResult(journal=journal if journal is not None else RunJournal())
    for t in time_indices:
        _forget_blocks(estimator)
        mov_t = np.asarray(mov[t])
        ref_t = np.asarray(reference_policy.reference_for(mov, t))
        if is_empty(mov_t) or is_empty(ref_t):
            # Nothing to estimate from; reported like propagation does, not as a failed fit.
            result.scores[t] = float("nan")
            result.errors[t] = "empty frame (no data)"
            if on_timepoint is not None:
                on_timepoint(t, result)
            continue
        seed = initial_policy.initial_for(t)
        estimate, error = None, None
        try:
            transform = estimator.estimate(mov_t, ref_t, seed=seed)
        except EstimationError as e:
            error = f"{type(e).__name__}: {e}"
        else:
            transform = _gap_rule(estimator, reference_policy, mov, t, transform)
            estimate = (transform, float(score_fn(transform, mov_t, ref_t)))
        if keep_initial:
            estimate, result.initialized_from[t] = _keep_initial(
                estimator, estimate, seed, score_fn, mov_t, ref_t
            )
        if estimate is None:
            result.scores[t] = float("nan")
            result.errors[t] = error
        else:
            result.transforms[t], result.scores[t] = estimate
        if on_timepoint is not None:
            on_timepoint(t, result)
    return result


def _keep_initial(estimator, estimate, seed, score_fn, mov, ref) -> tuple[tuple, str]:
    """Return the estimate if it scores strictly better than the seed as it is, else the seed.

    The seed is an earlier result (a seed file), so a run from it never ends worse than
    it: a failed or no better estimate keeps it. Returns ((transform, score), source),
    source "file+refined" or "file". A kept seed was not produced by the stack,
    so its blocks are forgotten.
    """
    kept = (seed, float(score_fn(seed, mov, ref)))
    if estimate is not None and _beats(estimate[1], kept[1]):
        return estimate, "file+refined"
    _forget_blocks(estimator)
    return kept, "file"


def _across_gap(reference_policy: ReferencePolicy, mov, t: int) -> bool:
    """Whether t's 'previous' reference reaches back over empty frames (t-1 has no data)."""
    if not isinstance(reference_policy, PreviousFrame):
        return False
    index = reference_policy.reference_index(mov, t)
    return index is not None and index != t - 1


def _hold_xy(transform: Transform) -> Transform:
    """Keep the z part of a step; the yx part becomes identity."""
    matrix = np.array(transform.matrix)
    matrix[1:3, :] = np.eye(4)[1:3, :]
    return Transform(matrix, transform_type=transform.transform_type)


def _gap_rule(estimator, reference_policy, mov, t: int, transform: Transform) -> Transform:
    """Legacy focus-finding across a gap: z measured, yx held (see FocusEstimator)."""
    if getattr(estimator, "holds_xy_across_gaps", False) and _across_gap(
        reference_policy, mov, t
    ):
        return _hold_xy(transform)
    return transform


def _compares_with_itself(reference_policy: ReferencePolicy, mov, t: int) -> bool:
    """Whether timepoint t is its own reference (it starts the stabilization chain)."""
    if isinstance(reference_policy, FixedFrame):
        return t == reference_policy.reference_index(mov)
    if isinstance(reference_policy, PreviousFrame):
        return reference_policy.reference_index(mov, t) is None
    if isinstance(reference_policy, NeighbourMedian):
        return not reference_policy.neighbour_indices(mov, t)
    return False


def _estimate_competing(estimator, mov, ref, seed, competitor, score_fn) -> Transform:
    """Estimate from `seed`, with `competitor` competing (ties to the seed).

    An estimator that takes a competitor refines both on its first pass (the legacy
    rule); any other estimator is run from both seeds and the better score is kept.
    """
    if competitor is None:
        return estimator.estimate(mov, ref, seed=seed)
    if "competitor" in inspect.signature(estimator.estimate).parameters:
        return estimator.estimate(mov, ref, seed=seed, competitor=competitor)
    results, errors = [], []
    for start in (seed, competitor):
        try:
            transform = estimator.estimate(mov, ref, seed=start)
        except EstimationError as e:
            errors.append(str(e))
            continue
        score = score_fn(transform, mov, ref)
        # a stack's record of its blocks belongs to the run that is kept
        stages = getattr(estimator, "last_stages", None)
        results.append((transform, float(score) if np.isfinite(score) else -np.inf, stages))
    if not results:
        raise EstimationError(
            f"failed from both the previous result and the input seed: {'; '.join(errors)}"
        )
    best = results[0]
    for candidate in results[1:]:
        if candidate[1] > best[1]:
            best = candidate
    if hasattr(estimator, "last_stages"):
        estimator.last_stages = best[2]
    return best[0]


def _forget_blocks(estimator) -> None:
    """Clear a stack's record of its blocks: a timepoint not estimated has none."""
    if getattr(estimator, "last_stages", None) is not None:
        estimator.last_stages = None


def estimate_propagated(
    mov,
    reference_policy: ReferencePolicy,
    estimator: TransformEstimator,
    input_seed: Transform,
    score_fn: ScoreFn,
    time_indices: Iterable[int],
    done: dict[int, dict] | None = None,
    on_timepoint: OnTimepoint | None = None,
) -> SeriesResult:
    """Estimate timepoints in order, each starting from the previous one's result.

    The legacy `use_prev_t_transform` rules: timepoint t starts from what t-1 returned,
    with `input_seed` competing on the first pass; a timepoint that fails returns the
    seed it started from (recorded as a stand-in, the error kept) and that seed is
    passed on; an empty frame (no data) is skipped and the chain carries on from the last
    result before it; the
    stabilization reference frame is not estimated against itself -- it is identity, and
    the chain starts after it from `input_seed`. `done` holds records of timepoints
    already estimated (resume): they are not redone, only used to continue the chain.
    """
    done = done or {}
    result = SeriesResult()
    previous: Transform | None = None  # what the next timepoint starts from
    previous_t: int | None = None  # the timepoint it came from (a stand-in's source)
    for t in time_indices:
        _forget_blocks(estimator)
        if t in done:
            # Carry the chain on as the fresh run did: the reference frame restarts it, an
            # empty frame (no transform, no stand-in) leaves it alone.
            carried = _record_into(result, t, done[t])
            if _compares_with_itself(reference_policy, mov, t):
                previous = None
            elif carried is not None:
                previous, previous_t = carried, t
        else:
            mov_t = np.asarray(mov[t])
            ref_t = np.asarray(reference_policy.reference_for(mov, t))
            if is_empty(mov_t) or is_empty(ref_t):
                # Legacy skipped empty frames without touching the propagated transform.
                result.scores[t] = float("nan")
                result.errors[t] = "empty frame (no data)"
            elif _compares_with_itself(reference_policy, mov, t):
                identity = Transform.identity(mov_t.ndim)
                result.transforms[t] = identity
                result.scores[t] = float(score_fn(identity, mov_t, ref_t))
                previous = None
            else:
                seed = previous if previous is not None else input_seed
                competitor = input_seed if previous is not None else None
                try:
                    transform = _estimate_competing(
                        estimator, mov_t, ref_t, seed, competitor, score_fn
                    )
                except EstimationError as e:
                    result.scores[t] = float("nan")
                    result.errors[t] = f"{type(e).__name__}: {e}"
                    source = f"t={previous_t}" if previous is not None else "initial"
                    result.stand_ins[t] = (seed, source)
                    previous = seed
                else:
                    transform = _gap_rule(estimator, reference_policy, mov, t, transform)
                    result.transforms[t] = transform
                    result.scores[t] = float(score_fn(transform, mov_t, ref_t))
                    previous, previous_t = transform, t
        if on_timepoint is not None:
            on_timepoint(t, result)
    return result


def _record_into(result: SeriesResult, t: int, record: dict) -> Transform | None:
    """Fold a timepoint record into `result`; return what the next timepoint starts from."""
    result.scores[t] = float("nan") if record.get("score") is None else record["score"]
    if record.get("error"):
        result.errors[t] = record["error"]
    if record.get("matrix") is not None:
        result.transforms[t] = Transform(np.asarray(record["matrix"], dtype=float))
        return result.transforms[t]
    if record.get("stand_in") is not None:
        stand_in = Transform(np.asarray(record["stand_in"], dtype=float))
        result.stand_ins[t] = (stand_in, record.get("stand_in_from", "initial"))
        return stand_in
    return None
