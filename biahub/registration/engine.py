"""The registration engine's run loop.

Estimate every timepoint, flag the poor ones against the run's own score distribution,
repair them from seeds the run itself provides, and journal every attempt under an
explicit run identity.

Layout of this module, top to bottom: the run journal, adaptive flagging, the repair
pass, and the series drivers (`estimate_series`, `flag_series`, `repair_timepoint`,
`repair_series`). The two passes are separate so estimation can be fanned out per
timepoint while repair waits for every score.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import uuid

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import click
import numpy as np
import pandas as pd
import submitit

from iohub import open_ome_zarr
from numpy.typing import ArrayLike

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import sbatch_to_submitit
from biahub.core.transform import Transform
from biahub.registration.estimators import (
    ChainedEstimator,
    CompetingEstimator,
    EstimationError,
    ScoreFn,
    TransformEstimator,
)
from biahub.registration.methods.ants import AntsEstimator
from biahub.registration.methods.beads import NodeGraphEstimator
from biahub.registration.methods.focus import FocusEstimator
from biahub.registration.methods.manual import ManualEstimator
from biahub.registration.methods.pcc import PCCEstimator
from biahub.registration.methods.vote_icp import VoteIcpEstimator, VoteSeedCorrection
from biahub.registration.metrics import (
    bead_alignment_metrics,
    beads_score_fn,
    correlation_score,
    gradient_correlation,
    normalized_mutual_information,
    residual_score,
    score_transform,
)
from biahub.registration.policies import (
    ConsensusSeed,
    CrossChannel,
    FixedFrame,
    FixedSeed,
    PreviousFrame,
    ReferencePolicy,
    SeedPolicy,
)
from biahub.registration.utils import get_aprox_transform
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    EstimateTransformSettings,
)
from biahub.utils.cluster import estimate_resources, get_submitit_cluster
from biahub.utils.config import model_to_yaml, yaml_to_model

ENGINE_SETTINGS_FILENAME = "estimate_transform_settings.yml"
RUN_MANIFEST_FILENAME = "run_manifest.json"


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
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path, run_id: str | None = None) -> RunJournal:
        if not path.exists():
            return cls(run_id=run_id)
        return cls.from_dict(json.loads(path.read_text()), run_id=run_id)


HARD_FAIL_SCORE = 0.40


FLAG_FLOOR_SCORE = 0.80


FLAG_K_MAD = 2.0


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
    max_timepoints: int | None = None,
    k_mad: float = FLAG_K_MAD,
    floor: float = FLAG_FLOOR_SCORE,
    hard_fail: float = HARD_FAIL_SCORE,
) -> tuple[list[int], dict]:
    """Timepoints for a fallback pass to act on, from the adaptive median-2*MAD line.

    Shared by every fallback pass so they can't drift apart in how they choose work, and
    so no pass carries its own fixed score threshold.

    Returns (flagged timepoints, run statistics) -- the statistics are returned rather
    than left for callers to recompute, so whatever gets logged is provably the same
    numbers the selection used.

    Returns the worst `max_timepoints` when a cap is set, and always says which ones it
    dropped: a capped pass that logs nothing about the drop would read as full coverage.
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
    flagged = cap_worst(flagged, dict(enumerate(score_col)), max_timepoints)
    return flagged, dict(flags.attrs)


def cap_worst(
    timepoints: list[int], scores: dict[int, float], max_timepoints: int | None
) -> list[int]:
    """Keep the worst `max_timepoints` of `timepoints` by score, naming the ones left out."""
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
    [int, dict[int, Transform], dict[int, float], set[int]], dict[str, SeedPolicy]
]


def neighbour_consensus_config_candidates(
    config_seed: Transform,
    consensus_score_threshold: float = 0.75,
    consensus_min_good: int = 5,
    order: tuple[str, ...] = ("t-1", "t+1", "consensus", "seed"),
) -> RepairCandidates:
    """Build the repair candidate set in the given order.

    "t-1" / "t+1": non-flagged neighbours; "consensus": the run's consensus geometry
    (named `consensus_full` in results); "seed": the config seed (`config_seed`).
    """

    def candidates(
        t: int, history: dict[int, Transform], scores: dict[int, float], flagged: set[int]
    ) -> dict[str, SeedPolicy]:
        out: dict[str, SeedPolicy] = {}
        for name in order:
            if name in ("t-1", "t+1"):
                idx = t - 1 if name == "t-1" else t + 1
                if idx in history and idx not in flagged:
                    out[name] = FixedSeed(history[idx])
            elif name == "consensus":
                out["consensus_full"] = ConsensusSeed(
                    history=history,
                    scores=scores,
                    score_threshold=consensus_score_threshold,
                    min_good=consensus_min_good,
                )
            elif name == "seed":
                out["config_seed"] = FixedSeed(config_seed)
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
    reseed_score: float | None = None  # the accepted candidate's score before polish


def _beats(candidate: float | None, current: float | None) -> bool:
    """A finite candidate beats a missing / NaN current score, or a strictly lower one."""
    if candidate is None or not np.isfinite(candidate):
        return False
    return current is None or not np.isfinite(current) or candidate > current


def _best_of(
    t: int,
    pass_name: str,
    attempts: dict[str, Callable[[], Transform]],
    current_transform: Transform,
    current_score: float,
    score_fn: Callable[[Transform], float],
    journal: RunJournal | None,
) -> PassResult:
    """Run each attempt in order; keep the best-scoring one if it strictly beats current.

    An attempt that raises is skipped, not fatal, and its exception is kept in
    `PassResult.failures` (and the journal) so a broken candidate is distinguishable
    from one that merely lost. Ties go to the earlier attempt.
    """
    best_name = "unchanged"
    best_transform = current_transform
    best_score = current_score
    scores: dict[str, float] = {}
    failures: dict[str, str] = {}

    for name, attempt in attempts.items():
        try:
            candidate_transform = attempt()
            candidate_score = score_fn(candidate_transform)
        except Exception as e:  # noqa: BLE001
            failures[name] = f"{type(e).__name__}: {e}"
            continue
        scores[name] = candidate_score
        if _beats(candidate_score, best_score):
            best_name, best_transform, best_score = name, candidate_transform, candidate_score

    accepted = best_name != "unchanged"
    if journal is not None:
        journal.record(
            t=t,
            pass_name=pass_name,
            before_score=current_score,
            after_score=best_score,
            accepted=accepted,
            failures=failures,
        )
    return PassResult(
        transform=best_transform,
        score=best_score,
        accepted=accepted,
        source=best_name,
        scores=scores,
        failures=failures,
    )


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
) -> PassResult:
    """Re-estimate timepoint `t` from each candidate seed; keep whichever scores best.

    A candidate whose `seed_for()` raises -- e.g. `ConsensusSeed` when too few timepoints
    score well enough yet -- is skipped like one whose `estimate()` raises.
    """
    return _best_of(
        t,
        "repair",
        {
            name: (lambda policy=policy: estimator.estimate(mov, ref, seed=policy.seed_for(t)))
            for name, policy in candidates.items()
        },
        current_transform,
        current_score,
        score_fn,
        journal,
    )


def sweep(
    t: int,
    mov: ArrayLike,
    ref: ArrayLike,
    trials: dict[str, TransformEstimator],
    seed: Transform,
    current_transform: Transform,
    current_score: float,
    score_fn: Callable[[Transform], float],
    journal: RunJournal | None = None,
) -> PassResult:
    """Re-estimate timepoint `t` with each trial's estimator from `seed`; keep the best.

    Repair changes the seed and keeps the method; the sweep keeps the seed and changes the
    method's settings, for the timepoints the run's one parameter set does not serve.
    `score_fn` is the run's own, so every trial is judged on the same scale.
    """
    return _best_of(
        t,
        "sweep",
        {
            name: (lambda estimator=estimator: estimator.estimate(mov, ref, seed=seed))
            for name, estimator in trials.items()
        },
        current_transform,
        current_score,
        score_fn,
        journal,
    )


def polish(
    t: int,
    mov: ArrayLike,
    ref: ArrayLike,
    estimator: TransformEstimator,
    transform: Transform,
    score: float,
    score_fn: Callable[[Transform], float],
    rounds: int,
    journal: RunJournal | None = None,
) -> tuple[Transform, float, int]:
    """Re-seed the estimator from `transform` and refine again, up to `rounds` times.

    Repair candidates are coarse seeds and detection runs in the seed-warped space, so an
    estimate seeded from the accepted transform sees better nodes than the one that
    produced it. A round is kept only on a strict score gain; the first round that fails
    to improve (or raises) ends the pass. Returns (transform, score, improving rounds).
    """
    before, improved = score, 0
    for _ in range(rounds):
        try:
            candidate = estimator.estimate(mov, ref, seed=transform)
            candidate_score = score_fn(candidate)
        except Exception:  # noqa: BLE001 -- a failed round keeps the accepted transform
            break
        if not _beats(candidate_score, score):
            break
        transform, score, improved = candidate, candidate_score, improved + 1
    if journal is not None and rounds:
        journal.record(
            t=t,
            pass_name="polish",
            before_score=before,
            after_score=score,
            accepted=improved > 0,
        )
    return transform, score, improved


@dataclass
class SeriesResult:
    # Accepted transforms by t. Doubles as the `history` a PreviousSeed reads from.
    transforms: dict[int, Transform] = field(default_factory=dict)
    # Every attempted t; nan where estimation failed and nothing repaired it.
    scores: dict[int, float] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    flagged: list[int] = field(default_factory=list)
    repairs: dict[int, PassResult] = field(default_factory=dict)
    sweeps: dict[int, PassResult] = field(default_factory=dict)
    # How an accepted fallback transform was reached, e.g. "consensus_full+polish1".
    provenance: dict[int, str] = field(default_factory=dict)
    # Timepoints with no transform of their own, and where the stand-in written for them
    # came from ("t=81", "seed", "identity"); set by `transforms_for_file`.
    filled_from: dict[int, str] = field(default_factory=dict)
    # A stand-in decided while estimating (propagation: a failed timepoint returns the
    # seed it started from), with its source; used instead of the input seed.
    stand_ins: dict[int, tuple[Transform, str]] = field(default_factory=dict)
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


def _empty(volume: np.ndarray) -> bool:
    """A frame with no data: all zeros or all NaN."""
    return not np.any(np.nan_to_num(volume))


def _compares_with_itself(reference_policy: ReferencePolicy, t: int) -> bool:
    """Whether timepoint t is its own reference (the stabilization reference frame)."""
    if isinstance(reference_policy, FixedFrame):
        return t == reference_policy.t_ref
    if isinstance(reference_policy, PreviousFrame):
        return t == 0
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
        results.append((transform, float(score) if np.isfinite(score) else -np.inf))
    if not results:
        raise EstimationError(
            f"failed from both the previous result and the input seed: {'; '.join(errors)}"
        )
    best = results[0]
    for candidate in results[1:]:
        if candidate[1] > best[1]:
            best = candidate
    return best[0]


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
    for t in time_indices:
        if t in done:
            record = done[t]
            previous = _record_into(result, t, record)
        else:
            mov_t = np.asarray(mov[t])
            ref_t = np.asarray(reference_policy.reference_for(mov, t))
            if _empty(mov_t) or _empty(ref_t):
                # Legacy skipped empty frames without touching the propagated transform.
                result.scores[t] = float("nan")
                result.errors[t] = "empty frame (no data)"
            elif _compares_with_itself(reference_policy, t):
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
                    source = f"t={t - 1}" if previous is not None else "seed"
                    result.stand_ins[t] = (seed, source)
                    previous = seed
                else:
                    result.transforms[t] = transform
                    result.scores[t] = float(score_fn(transform, mov_t, ref_t))
                    previous = transform
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
        result.stand_ins[t] = (stand_in, record.get("stand_in_from", "seed"))
        return stand_in
    return None


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
        journal=result.journal,
    )
    if outcome.accepted and polish_rounds:
        outcome.reseed_score = outcome.score
        outcome.transform, outcome.score, outcome.polish_rounds = polish(
            t,
            mov_t,
            ref_t,
            estimator,
            outcome.transform,
            outcome.score,
            lambda transform, m=mov_t, r=ref_t: score_fn(transform, m, r),
            polish_rounds,
            journal=result.journal,
        )
        if outcome.polish_rounds:
            outcome.source += f"+polish{outcome.polish_rounds}"
    result.repairs[t] = outcome
    if outcome.accepted:
        _accept(result, t, outcome.transform, outcome.score, outcome.source)
    return outcome


def _accept(result: SeriesResult, t: int, transform: Transform, score: float, source: str):
    result.transforms[t] = transform
    result.scores[t] = score
    result.errors.pop(t, None)
    result.provenance[t] = source


def sweep_timepoint(
    t: int,
    mov,
    reference_policy: ReferencePolicy,
    trials: dict[str, TransformEstimator],
    seed: Transform,
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
        t=t,
        mov=mov_t,
        ref=ref_t,
        trials=trials,
        seed=seed,
        current_transform=current if current is not None else Transform.identity(mov_t.ndim),
        current_score=-np.inf if not np.isfinite(baseline_score) else baseline_score,
        score_fn=lambda transform, m=mov_t, r=ref_t: score_fn(transform, m, r),
        journal=result.journal,
    )
    _fold_sweep(result, t, outcome)
    return outcome


def _fold_sweep(result: SeriesResult, t: int, outcome: PassResult) -> None:
    result.sweeps[t] = outcome
    current_score = result.scores.get(t, float("nan"))
    if outcome.accepted and not outcome.score <= current_score:
        _accept(result, t, outcome.transform, outcome.score, f"sweep:{outcome.source}")


def repair_series(
    mov,
    reference_policy: ReferencePolicy,
    estimator: TransformEstimator,
    score_fn: ScoreFn,
    result: SeriesResult,
    candidates: RepairCandidates,
    max_timepoints: int | None = None,
    on_timepoint: OnTimepoint | None = None,
    polish_rounds: int = 0,
) -> SeriesResult:
    """Flag attempted timepoints and offer each to `repair`, in-process and in order.

    Unflagged timepoints are never touched.
    """
    for t in flag_series(result, max_timepoints):
        repair_timepoint(
            t, mov, reference_policy, estimator, score_fn, result, candidates, polish_rounds
        )
        if on_timepoint is not None:
            on_timepoint(t, result)
    return result


def resolve_time_indices(time_indices, n_t: int) -> list[int]:
    if time_indices == "all":
        return list(range(n_t))
    if isinstance(time_indices, int):
        return [time_indices]
    return list(time_indices)


def _open_series(position_dirpath: Path, channel_name: str):
    """Open one channel as a (T, Z, Y, X) dask series, with its ZYX voxel size."""
    with open_ome_zarr(position_dirpath, mode="r") as position:
        series = position.data.dask_array()[:, position.channel_names.index(channel_name)]
        return series, tuple(position.scale[-3:])


def _reference_policy(settings: EstimateTransformSettings, ref) -> ReferencePolicy:
    if settings.reference.frame == "cross":
        return CrossChannel(ref)
    if settings.reference.frame == "first":
        return FixedFrame(0)
    return PreviousFrame()


def _seed(settings: EstimateTransformSettings) -> Transform:
    fit = settings.transform
    if fit.seed_direction == "forward":
        return Transform(np.asarray(fit.seed, dtype=float), transform_type=fit.type)
    return Transform.from_legacy_pull(fit.seed, fit.type)


def _score_fn(settings: EstimateTransformSettings) -> ScoreFn:
    metric = settings.effective_score_metric
    if metric == "overlap":
        return lambda transform, mov, ref: score_transform(transform, mov, ref, settings.beads)
    if metric == "residual":
        return lambda transform, mov, ref: residual_score(transform, mov, ref, settings.beads)
    if metric == "mutual_information":
        return normalized_mutual_information
    if metric == "correlation":
        sobel = settings.method == "ants" and settings.ants.sobel_filter
        return lambda transform, mov, ref: correlation_score(
            transform, mov, ref, sobel_filter=sobel
        )
    if metric == "gradient_correlation":
        return gradient_correlation
    raise ValueError(f"unknown score_metric {metric!r}")


def _affine_settings(settings: EstimateTransformSettings) -> AffineTransformSettings:
    """Build the legacy `AffineTransformSettings` the estimators' constructors still take."""
    fit = settings.transform
    seed_pull = (
        fit.seed
        if fit.seed_direction == "pull"
        else np.linalg.inv(np.asarray(fit.seed, dtype=float)).tolist()
    )
    return AffineTransformSettings(
        transform_type=fit.type,
        approx_transform=seed_pull,
        use_prev_t_transform=False,
        compute_approx_transform=fit.seed_from_shapes,
        t_reference="first"
        if settings.reference.frame == "cross"
        else settings.reference.frame,
    )


def build_beads_estimator(
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    score_fn: ScoreFn | None = None,
    iterations: int | None = None,
) -> TransformEstimator:
    """Build the beads method as configured: matcher, estimation mode, seed correction, arms.

    `estimation_mode: vote_icp` swaps the graph matcher for `VoteIcpEstimator`; a
    `seed_correction_settings.mode` other than `none` chains a `VoteSeedCorrection` in
    front; `spectral_arm` on returns a `CompetingEstimator` of the configured matcher
    and a spectral-acquire-then-refine cascade, the higher-scoring arm wins.
    """
    score_fn = score_fn or beads_score_fn(beads_match_settings)

    def node_graph(settings: BeadsMatchSettings) -> NodeGraphEstimator:
        return NodeGraphEstimator.from_beads_settings(
            settings, affine_transform_settings, iterations=iterations, score_fn=score_fn
        )

    if beads_match_settings.estimation_mode == "vote_icp":
        configured: TransformEstimator = VoteIcpEstimator.from_beads_settings(
            beads_match_settings,
            score_fn=score_fn,
            transform_type=affine_transform_settings.transform_type,
        )
    else:
        configured = node_graph(beads_match_settings)
    if beads_match_settings.seed_correction_settings.mode != "none":
        configured = ChainedEstimator(
            [VoteSeedCorrection.from_beads_settings(beads_match_settings), configured],
            score_fn=score_fn,
        )
    if beads_match_settings.spectral_arm == "off":
        return configured
    spectral_settings = beads_match_settings.model_copy(deep=True)
    spectral_settings.algorithm = "spectral"
    cascade = ChainedEstimator(
        [node_graph(spectral_settings), node_graph(beads_match_settings)], score_fn=score_fn
    )
    return CompetingEstimator(
        {beads_match_settings.algorithm: configured, "spectral+hungarian": cascade},
        score_fn=score_fn,
        escalate_below=(
            beads_match_settings.qc_settings.score_threshold
            if beads_match_settings.spectral_arm == "on_low_score"
            else None
        ),
    )


def build_estimator(
    settings: EstimateTransformSettings,
    shape_zyx: tuple[int, int, int] | None = None,
    mov_voxel_size: tuple[float, float, float] | None = None,
    ref_voxel_size: tuple[float, float, float] | None = None,
) -> tuple[TransformEstimator, ScoreFn, Transform]:
    """Estimator, score function and forward seed for the settings' method."""
    score_fn = _score_fn(settings)
    seed = _seed(settings)
    affine = _affine_settings(settings)
    if settings.method == "beads":
        estimator: TransformEstimator = build_beads_estimator(
            settings.beads, affine, score_fn=score_fn
        )
    elif settings.method == "ants":
        estimator = AntsEstimator.from_settings(
            settings.ants, affine, verbose=settings.verbose
        )
    elif settings.method == "phase-cross-corr":
        estimator = PCCEstimator.from_settings(settings.phase_cross_corr, shape_zyx)
    elif settings.method == "focus-finding":
        estimator = FocusEstimator.from_settings(
            settings.focus_finding, pixel_size=(mov_voxel_size or (1.0, 1.0, 1.0))[-1]
        )
    elif settings.method == "manual":
        manual = settings.manual
        estimator = ManualEstimator(
            source_channel_name=settings.moving.channel,
            target_channel_name=settings.reference_channel,
            source_channel_voxel_size=mov_voxel_size or (1.0, 1.0, 1.0),
            target_channel_voxel_size=ref_voxel_size or (1.0, 1.0, 1.0),
            similarity=settings.transform.type == "similarity",
            pre_affine_90degree_rotation=manual.affine_90degree_rotation,
            pre_affine_fliplr=manual.affine_fliplr,
        )
    else:
        raise click.UsageError(f"unknown method '{settings.method}'")
    return estimator, score_fn, seed


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _bead_metrics(settings, result, t, mov, ref) -> dict | None:
    """Continuous bead residuals next to the score, for the beads method."""
    if settings.method != "beads" or t not in result.transforms:
        return None
    ref_t = np.asarray(_reference_policy(settings, ref).reference_for(mov, t))
    metrics = bead_alignment_metrics(
        result.transforms[t], np.asarray(mov[t]), ref_t, settings.beads
    )
    return None if metrics is None else metrics.to_dict()


def _estimate_timepoint_job(
    moving_position_dirpath: Path,
    reference_position_dirpath: Path,
    settings_path: Path,
    t: int,
    record_path: Path,
) -> dict:
    """One independent estimate, from the config seed, written as a JSON record."""
    settings = yaml_to_model(settings_path, EstimateTransformSettings)
    mov, mov_voxel_size = _open_series(moving_position_dirpath, settings.moving.channel)
    ref, ref_voxel_size = _open_series(reference_position_dirpath, settings.reference_channel)
    estimator, score_fn, seed = build_estimator(
        settings, tuple(mov.shape[-3:]), mov_voxel_size, ref_voxel_size
    )

    result = estimate_series(
        mov, _reference_policy(settings, ref), estimator, FixedSeed(seed), score_fn, [t]
    )
    record = {
        "t": t,
        "matrix": result.transforms[t].to_list() if t in result.transforms else None,
        "score": _finite_or_none(result.scores.get(t)),
        "error": result.errors.get(t),
        "arm": getattr(estimator, "last_winner", None),
        "metrics": _bead_metrics(settings, result, t, mov, ref),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    return record


def _estimate_propagated_job(
    moving_position_dirpath: Path,
    reference_position_dirpath: Path,
    settings_path: Path,
    time_indices: list[int],
    records_dir: Path,
    resume: bool,
) -> dict[int, dict]:
    """All timepoints in order (`seed_from: previous_timepoint`), one record per timepoint.

    Each record is written as soon as its timepoint is done, so an interrupted job
    resumes the chain where it stopped.
    """
    settings = yaml_to_model(settings_path, EstimateTransformSettings)
    mov, mov_voxel_size = _open_series(moving_position_dirpath, settings.moving.channel)
    ref, ref_voxel_size = _open_series(reference_position_dirpath, settings.reference_channel)
    estimator, score_fn, seed = build_estimator(
        settings, tuple(mov.shape[-3:]), mov_voxel_size, ref_voxel_size
    )
    done = {}
    if resume:
        for t in time_indices:
            path = records_dir / f"{t}.json"
            if path.exists():
                done[t] = json.loads(path.read_text())
    records = dict(done)
    records_dir.mkdir(parents=True, exist_ok=True)

    def write_record(t: int, result: SeriesResult) -> None:
        if t in done:
            return
        stand_in = result.stand_ins.get(t)
        record = {
            "t": t,
            "matrix": result.transforms[t].to_list() if t in result.transforms else None,
            "score": _finite_or_none(result.scores.get(t)),
            "error": result.errors.get(t),
            "stand_in": stand_in[0].to_list() if stand_in else None,
            "stand_in_from": stand_in[1] if stand_in else None,
            "arm": getattr(estimator, "last_winner", None),
            "metrics": _bead_metrics(settings, result, t, mov, ref),
        }
        (records_dir / f"{t}.json").write_text(json.dumps(record))
        records[t] = record

    estimate_propagated(
        mov,
        _reference_policy(settings, ref),
        estimator,
        seed,
        score_fn,
        time_indices,
        done=done,
        on_timepoint=write_record,
    )
    return records


def _run_fingerprint(settings: EstimateTransformSettings, source: Path, target: Path) -> dict:
    """What a resumed run must share with the run it resumes."""
    digest = hashlib.sha256(settings.model_dump_json().encode()).hexdigest()
    return {"settings_sha256": digest, "moving": str(source), "reference": str(target)}


def _start_run(
    output_dir: Path,
    settings: EstimateTransformSettings,
    source: Path,
    target: Path,
    resume: bool,
) -> None:
    """Begin a run in `output_dir`: clear earlier records, or check a resume is compatible.

    Without `resume`, the record dirs of any earlier run are removed so none of their
    files can stand in for a job of this run. With `resume`, the settings and inputs must
    match the manifest of the run being resumed.
    """
    manifest_path = output_dir / RUN_MANIFEST_FILENAME
    fingerprint = _run_fingerprint(settings, source, target)
    if resume:
        if not manifest_path.exists():
            click.echo(f"resume: no {RUN_MANIFEST_FILENAME} in {output_dir}; cannot verify the "
                       "earlier run used the same settings and inputs")
        else:
            previous = json.loads(manifest_path.read_text())
            changed = sorted(k for k in fingerprint if previous.get(k) != fingerprint[k])
            if changed:
                raise click.UsageError(
                    f"--resume: {', '.join(changed)} changed since the run in {output_dir}; "
                    "rerun without --resume"
                )
        return
    for name in ("timepoints", "repairs", "sweeps"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    manifest_path.write_text(json.dumps(fingerprint, indent=2))


def _load_series(
    records_dir: Path,
    time_indices: list[int],
    transform_type: str,
    records: dict[int, dict] | None = None,
    resumed: Iterable[int] = (),
) -> SeriesResult:
    """Rebuild the series from per-timepoint records.

    In-memory records (this run's jobs) first; the files on disk only for `resumed`
    timepoints, which this run skipped. A timepoint with neither is an error -- a job
    that failed this run is never filled from an earlier run's record.
    """
    records = dict(records or {})
    resumed = set(resumed)
    result = SeriesResult()
    for t in time_indices:
        record = records.get(t)
        if record is None:
            path = records_dir / f"{t}.json"
            if t not in resumed or not path.exists():
                result.scores[t] = float("nan")
                result.errors[t] = "no record: job did not finish"
                continue
            record = json.loads(path.read_text())
        if record["matrix"] is not None:
            result.transforms[t] = Transform(
                np.asarray(record["matrix"], dtype=float), transform_type=transform_type
            )
        elif record.get("stand_in") is not None:
            result.stand_ins[t] = (
                Transform(np.asarray(record["stand_in"], dtype=float), transform_type=transform_type),
                record.get("stand_in_from") or "seed",
            )
        result.scores[t] = float("nan") if record["score"] is None else record["score"]
        if record["error"]:
            result.errors[t] = record["error"]
    return result


def _repair_timepoint_job(
    moving_position_dirpath: Path,
    reference_position_dirpath: Path,
    settings_path: Path,
    t: int,
    time_indices: list[int],
    flagged: list[int],
    records_dir: Path,
    record_path: Path,
) -> dict:
    """Repair one flagged timepoint against the frozen whole-run history."""
    settings = yaml_to_model(settings_path, EstimateTransformSettings)
    mov, mov_voxel_size = _open_series(moving_position_dirpath, settings.moving.channel)
    ref, ref_voxel_size = _open_series(reference_position_dirpath, settings.reference_channel)
    estimator, score_fn, seed = build_estimator(
        settings, tuple(mov.shape[-3:]), mov_voxel_size, ref_voxel_size
    )
    repair_settings = settings.fallback.repair

    # This run's estimate records (the driver cleared any earlier run's).
    series = _load_series(records_dir, time_indices, settings.transform.type, resumed=time_indices)
    series.flagged = list(flagged)
    outcome = repair_timepoint(
        t,
        mov,
        _reference_policy(settings, ref),
        estimator,
        score_fn,
        series,
        neighbour_consensus_config_candidates(
            seed,
            consensus_score_threshold=repair_settings.consensus_threshold,
            consensus_min_good=repair_settings.consensus_min_good,
            order=tuple(repair_settings.candidates),
        ),
        polish_rounds=repair_settings.polish_rounds,
    )
    record = {
        "t": t,
        "accepted": outcome.accepted,
        "source": outcome.source,
        "score": _finite_or_none(outcome.score),
        "matrix": outcome.transform.to_list() if outcome.accepted else None,
        "candidate_scores": {k: _finite_or_none(v) for k, v in outcome.scores.items()},
        "candidate_failures": outcome.failures,
        "polish_rounds": outcome.polish_rounds,
        "reseed_score": _finite_or_none(outcome.reseed_score),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    return record


def _sweep_timepoint_job(
    moving_position_dirpath: Path,
    reference_position_dirpath: Path,
    settings_path: Path,
    t: int,
    records_dir: Path,
    record_path: Path,
) -> dict:
    """Sweep one flagged timepoint against its own pre-fallback estimate."""
    settings = yaml_to_model(settings_path, EstimateTransformSettings)
    mov, mov_voxel_size = _open_series(moving_position_dirpath, settings.moving.channel)
    ref, ref_voxel_size = _open_series(reference_position_dirpath, settings.reference_channel)
    shape_zyx = tuple(mov.shape[-3:])
    _estimator, score_fn, seed = build_estimator(
        settings, shape_zyx, mov_voxel_size, ref_voxel_size
    )
    trials = {
        name: build_estimator(trial, shape_zyx, mov_voxel_size, ref_voxel_size)[0]
        for name, trial in settings.sweep_trials().items()
    }

    series = _load_series(records_dir, [t], settings.transform.type, resumed=[t])
    outcome = sweep_timepoint(
        t, mov, _reference_policy(settings, ref), trials, seed, score_fn, series
    )
    record = {
        "t": t,
        "accepted": outcome.accepted,
        "source": outcome.source,
        "score": _finite_or_none(outcome.score),
        "matrix": outcome.transform.to_list() if outcome.accepted else None,
        "candidate_scores": {k: _finite_or_none(v) for k, v in outcome.scores.items()},
        "candidate_failures": outcome.failures,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    return record


def _load_pass(record: dict, transform_type: str, fallback: Transform) -> PassResult:
    return PassResult(
        transform=(
            Transform(np.asarray(record["matrix"], dtype=float), transform_type=transform_type)
            if record["matrix"] is not None
            else fallback
        ),
        score=-np.inf if record["score"] is None else record["score"],
        accepted=record["accepted"],
        source=record["source"],
        scores={
            k: (float("nan") if v is None else v)
            for k, v in record["candidate_scores"].items()
        },
        failures=record["candidate_failures"],
        polish_rounds=record.get("polish_rounds", 0),
        reseed_score=record.get("reseed_score"),
    )


def _run_jobs(
    executor: submitit.AutoExecutor,
    resolved_cluster: str,
    monitor_flag: bool,
    label: str,
    submissions: list[tuple[int, Callable, tuple]],
) -> tuple[dict[int, dict], dict[int, str]]:
    """Submit one job per (t, fn, args); return ({t: record}, {t: error}).

    Records come back through submitit's own result channel rather than being re-read
    from the job's JSON file: on a shared filesystem the driver can observe a job as
    finished before the file it wrote is visible.
    """
    if not submissions:
        return {}, {}
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for _t, fn, args in submissions:
            jobs.append(executor.submit(fn, *args))
    log_path = Path(executor.folder) / f"{label}_job_ids.log"
    log_path.write_text("\n".join(str(job.job_id) for job in jobs))
    click.echo(f"{label}: submitted {len(jobs)} job(s) on cluster='{resolved_cluster}'")

    if monitor_flag and resolved_cluster == "slurm":
        monitor_jobs(jobs, [Path(f"t={t}") for t, _fn, _args in submissions])

    records: dict[int, dict] = {}
    failures: dict[int, str] = {}
    for (t, _fn, _args), job in zip(submissions, jobs, strict=True):
        try:
            records[t] = job.result()
        except Exception as e:  # noqa: BLE001 -- one job's infrastructure failure must not abort the run
            failures[t] = f"job failed: {type(e).__name__}: {str(e).splitlines()[-1][:200]}"
            click.echo(f"{label} t={t}: {failures[t]}")
    return records, failures


# Wall-clock budget per timepoint for the sequential (propagation) job, in minutes.
PROPAGATION_MINUTES_PER_TIMEPOINT = 5


def _run_propagated(
    executor, resolved_cluster, monitor_flag, job_args, to_estimate, records_dir, user_set_time
) -> tuple[dict[int, dict], dict[int, str]]:
    """Run the sequential estimate as one job; return ({t: record}, {t: error}).

    If the job dies part-way, the records it already wrote (this run's -- the record dirs
    were cleared at the start) are kept; only the timepoints it never reached fail.
    """
    if not to_estimate:
        return {}, {}
    if not user_set_time:
        minutes = max(30, PROPAGATION_MINUTES_PER_TIMEPOINT * len(to_estimate))
        executor.update_parameters(slurm_time=minutes)
    executor.update_parameters(slurm_job_name="estimate_transform_propagated")
    by_job, failures = _run_jobs(
        executor, resolved_cluster, monitor_flag, "estimate", [(-1, _estimate_propagated_job, job_args)]
    )
    if -1 in by_job:
        return {int(t): r for t, r in by_job[-1].items()}, {}
    records = {}
    for t in to_estimate:
        path = records_dir / f"{t}.json"
        if path.exists():
            records[t] = json.loads(path.read_text())
    error = failures.get(-1, "sequential job failed")
    return records, {t: error for t in to_estimate if t not in records}


def _one_transform_per_timepoint(
    result: SeriesResult, time_indices: list[int], fallback: Transform
) -> list[Transform]:
    """Return one transform per requested timepoint.

    The accepted transform, else `fallback` -- the input seed (e.g. the approximate
    transform), as the legacy pipeline returned when refinement failed. Each stand-in is
    recorded in `result.filled_from`.
    """
    out = []
    for t in time_indices:
        if t in result.transforms:
            out.append(result.transforms[t])
        elif t in result.stand_ins:
            transform, source = result.stand_ins[t]
            result.filled_from[t] = source
            out.append(transform)
        else:
            result.filled_from[t] = "seed"
            out.append(fallback)
    return out


def _pass_report(outcome: PassResult) -> dict:
    return {
        "accepted": outcome.accepted,
        "source": outcome.source,
        "score": _finite_or_none(outcome.score),
        "candidate_scores": {k: _finite_or_none(v) for k, v in outcome.scores.items()},
        "candidate_failures": outcome.failures,
    }


def _report(result: SeriesResult, time_indices: list[int]) -> dict:
    return {
        "run_id": result.journal.current_run_id,
        "time_indices": time_indices,
        "scores": {
            str(t): _finite_or_none(result.scores[t])
            for t in time_indices
            if t in result.scores
        },
        "errors": {str(t): e for t, e in result.errors.items()},
        "flagged": result.flagged,
        "repairs": {
            str(t): {
                **_pass_report(r),
                "polish_rounds": r.polish_rounds,
                "reseed_score": _finite_or_none(r.reseed_score),
            }
            for t, r in result.repairs.items()
        },
        "sweeps": {str(t): _pass_report(r) for t, r in result.sweeps.items()},
        "provenance": {str(t): source for t, source in sorted(result.provenance.items())},
        "stand_ins": {str(t): source for t, source in sorted(result.filled_from.items())},
    }


def estimate_transform_series(
    moving_position_dirpath: Path,
    reference_position_dirpath: Path,
    settings: EstimateTransformSettings,
    output_dir: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    resume: bool = False,
) -> tuple[SeriesResult, list[int], list[Transform]]:
    """Run both fan-out phases; return the series, its timepoints and one forward transform per timepoint.

    Writes the settings the jobs read (`estimate_transform_settings.yml`), the per-timepoint
    records (`timepoints/`, `repairs/`), `run_journal.json` and
    `estimate_transform_report.json` under `output_dir`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timepoints_dir = output_dir / "timepoints"
    repairs_dir = output_dir / "repairs"
    sweeps_dir = output_dir / "sweeps"
    slurm_out_path = output_dir / "slurm_output"
    slurm_out_path.mkdir(exist_ok=True)
    source, target = Path(moving_position_dirpath), Path(reference_position_dirpath)

    with open_ome_zarr(source, mode="r") as position:
        T, _C, Z, Y, X = position.data.shape
        mov_voxel_size = tuple(position.scale[-3:])
    with open_ome_zarr(target, mode="r") as position:
        ref_shape = position.data.shape[-3:]
        ref_voxel_size = tuple(position.scale[-3:])

    settings = settings.model_copy(deep=True)
    if settings.transform.seed_from_shapes:
        approx = get_aprox_transform(
            mov_shape=(Z, Y, X),
            ref_shape=tuple(ref_shape),
            pre_affine_90degree_rotation=-1,
            pre_affine_fliplr=False,
            verbose=settings.verbose,
            ref_voxel_size=ref_voxel_size,
            mov_voxel_size=mov_voxel_size,
        )
        settings.transform.seed = approx.to_list()
        settings.transform.seed_direction = "pull"
        click.echo(f"Computed seed from the store shapes:\n{approx.matrix}")
    _estimator, _score_fn, seed = build_estimator(
        settings, (Z, Y, X), mov_voxel_size, ref_voxel_size
    )
    if settings.method == "manual":
        # Interactive (napari); one timepoint, in this process.
        settings.time_indices = settings.manual.time_index
        cluster = "debug"
    settings_path = output_dir / ENGINE_SETTINGS_FILENAME
    _start_run(output_dir, settings, source, target, resume)
    model_to_yaml(settings, settings_path)
    time_indices = resolve_time_indices(settings.time_indices, T)
    transform_type = settings.transform.type

    # Bead matching is single-threaded; ANTs' optimizer uses ITK threads.
    _, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(1, 2, Z, Y, X),
        ram_multiplier=5,
        max_num_cpus=8 if settings.method == "ants" else 4,
    )
    slurm_args = {
        "slurm_job_name": "estimate_transform",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to N timepoints at a time
        "slurm_time": 30,
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))
    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    to_estimate = [
        t for t in time_indices if not (resume and (timepoints_dir / f"{t}.json").exists())
    ]
    if resume:
        click.echo(
            f"resume: {len(time_indices) - len(to_estimate)} timepoint(s) already estimated"
        )
    # An sbatch file that sets a time limit wins over every phase's default below.
    user_set_time = bool(sbatch_filepath and "slurm_time" in sbatch_to_submitit(sbatch_filepath))
    if settings.transform.seed_from == "previous_timepoint":
        estimate_records, job_failures = _run_propagated(
            executor,
            resolved_cluster,
            monitor,
            (source, target, settings_path, time_indices, timepoints_dir, resume),
            to_estimate,
            timepoints_dir,
            user_set_time=user_set_time,
        )
    else:
        estimate_records, job_failures = _run_jobs(
            executor,
            resolved_cluster,
            monitor,
            "estimate",
            [
                (
                    t,
                    _estimate_timepoint_job,
                    (source, target, settings_path, t, timepoints_dir / f"{t}.json"),
                )
                for t in to_estimate
            ],
        )

    result = _load_series(
        timepoints_dir,
        time_indices,
        transform_type,
        records=estimate_records,
        resumed=set(time_indices) - set(to_estimate),
    )
    for t, error in job_failures.items():
        result.errors[t] = error
    for t in time_indices:
        line = f"t={t}: score={result.scores[t]:.4f}"
        if t in result.errors:
            line += f"  estimate failed: {result.errors[t]}"
        click.echo(line)

    repair_settings = settings.fallback.repair
    flag = settings.fallback.flag
    flagged = flag_series(result, k_mad=flag.k_mad, floor=flag.floor, hard_fail=flag.hard_fail)
    base_scores = dict(result.scores)
    repair_ts = (
        cap_worst(flagged, base_scores, repair_settings.max_timepoints)
        if repair_settings is not None
        else []
    )
    if repair_ts and not getattr(_estimator, "uses_seed", True):
        click.echo(
            f"repair skipped: method {settings.method!r} ignores seeds, so re-seeding "
            f"would return the same transforms ({len(repair_ts)} flagged timepoint(s) stay flagged)"
        )
        repair_ts = []
    to_repair = [t for t in repair_ts if not (resume and (repairs_dir / f"{t}.json").exists())]
    executor.update_parameters(slurm_job_name="estimate_transform_repair")
    if not user_set_time:
        executor.update_parameters(slurm_time=60)
    repair_records, _repair_failures = _run_jobs(
        executor,
        resolved_cluster,
        monitor,
        "repair",
        [
            (
                t,
                _repair_timepoint_job,
                (
                    source,
                    target,
                    settings_path,
                    t,
                    time_indices,
                    flagged,
                    timepoints_dir,
                    repairs_dir / f"{t}.json",
                ),
            )
            for t in to_repair
        ],
    )
    for t in repair_ts:
        record = repair_records.get(t)
        if record is None:
            record_path = repairs_dir / f"{t}.json"
            if t in to_repair or not record_path.exists():  # failed this run, or never ran
                continue
            record = json.loads(record_path.read_text())
        outcome = _load_pass(record, transform_type, fallback=result.transforms.get(t, seed))
        result.repairs[t] = outcome
        reseed_score = outcome.score if outcome.reseed_score is None else outcome.reseed_score
        result.journal.record(
            t=t,
            pass_name="repair",
            before_score=result.scores[t],
            after_score=reseed_score,
            accepted=outcome.accepted,
            failures=outcome.failures,
        )
        if outcome.reseed_score is not None:
            result.journal.record(
                t=t,
                pass_name="polish",
                before_score=reseed_score,
                after_score=outcome.score,
                accepted=outcome.polish_rounds > 0,
            )
        if outcome.accepted:
            _accept(result, t, outcome.transform, outcome.score, outcome.source)
        click.echo(f"repair t={t}: {outcome.source} -> {outcome.score:.4f}")

    # The sweep competes with repair rather than following it: it starts from each
    # timepoint's pre-fallback estimate, and the better of the two is kept.
    sweep_settings = settings.fallback.sweep
    sweep_ts = (
        cap_worst(flagged, base_scores, sweep_settings.max_timepoints)
        if sweep_settings is not None
        else []
    )
    to_sweep = [t for t in sweep_ts if not (resume and (sweeps_dir / f"{t}.json").exists())]
    n_trials = len(settings.sweep_trials()) if sweep_settings is not None else 0
    executor.update_parameters(slurm_job_name="estimate_transform_sweep")
    if not user_set_time:
        executor.update_parameters(slurm_time=30 + 3 * n_trials)
    sweep_records, _sweep_failures = _run_jobs(
        executor,
        resolved_cluster,
        monitor,
        "sweep",
        [
            (
                t,
                _sweep_timepoint_job,
                (source, target, settings_path, t, timepoints_dir, sweeps_dir / f"{t}.json"),
            )
            for t in to_sweep
        ],
    )
    for t in sweep_ts:
        record = sweep_records.get(t)
        if record is None:
            record_path = sweeps_dir / f"{t}.json"
            if t in to_sweep or not record_path.exists():  # failed this run, or never ran
                continue
            record = json.loads(record_path.read_text())
        outcome = _load_pass(record, transform_type, fallback=result.transforms.get(t, seed))
        result.journal.record(
            t=t,
            pass_name="sweep",
            before_score=base_scores[t],
            after_score=outcome.score,
            accepted=outcome.accepted,
            failures=outcome.failures,
        )
        _fold_sweep(result, t, outcome)
        click.echo(
            f"sweep t={t}: {outcome.source} -> {outcome.score:.4f}"
            + ("" if result.provenance.get(t, "").startswith("sweep:") else " (not kept)")
        )

    # Before the report: this records which timepoints got a stand-in.
    transforms = transforms_for_file(result, time_indices, seed, settings.reference.frame)
    result.journal.save(output_dir / "run_journal.json")
    (output_dir / "estimate_transform_report.json").write_text(
        json.dumps(_report(result, time_indices), indent=2)
    )
    return result, time_indices, transforms


def transforms_for_file(
    result: SeriesResult, time_indices: list[int], seed: Transform, frame: str
) -> list[Transform]:
    """One transform per timepoint, onto the reference grid, for the transforms file.

    `cross` / `first` transforms are absolute, so a timepoint with none takes the input
    seed. `previous` transforms are relative steps (t -> t-1): a
    missing step is identity -- reusing a neighbour's step would add its drift again to
    every later timepoint -- and the steps are chained onto the first frame.
    """
    if frame != "previous":
        return _one_transform_per_timepoint(result, time_indices, seed)
    for t in time_indices:
        if t not in result.transforms:
            result.filled_from[t] = "identity"
    steps = [result.transforms.get(t, Transform.identity()) for t in time_indices]
    return chain_to_first_frame(steps, time_indices)


def chain_to_first_frame(
    transforms: list[Transform], time_indices: list[int]
) -> list[Transform]:
    """Compose t -> t-1 transforms into t -> first-frame transforms.

    `reference: previous` estimates each timepoint against the one before it, which is
    what makes the estimate robust on slowly drifting data, but apply-transform puts
    every timepoint on one grid, so the file must hold the cumulative transform:
    F_t = F_{t-1} @ (t -> t-1). That needs every link, hence contiguous timepoints.
    """
    if list(time_indices) != list(range(time_indices[0], time_indices[-1] + 1)):
        raise click.UsageError(
            "reference 'previous' needs contiguous time_indices (each timepoint is chained "
            f"through the one before it); got {time_indices}"
        )
    chained, cumulative = [], Transform.identity()
    for transform in transforms:
        cumulative = cumulative @ transform
        chained.append(cumulative)
    return chained
