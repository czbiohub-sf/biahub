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

import json
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
    if max_timepoints is not None and len(flagged) > max_timepoints:
        worst = sorted(flagged, key=lambda t: np.nan_to_num(score_col[t], nan=-1.0))
        dropped = sorted(worst[max_timepoints:])
        flagged = sorted(worst[:max_timepoints])
        click.echo(
            f"  capped at max_timepoints={max_timepoints}; taking the worst {len(flagged)} "
            f"and LEAVING {len(dropped)} untouched: {dropped}"
        )
    return flagged, dict(flags.attrs)


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
    if settings.reference == "cross":
        return CrossChannel(ref)
    if settings.reference == "first":
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
        t_reference="first" if settings.reference == "cross" else settings.reference,
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
            beads_match_settings, score_fn=score_fn
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
    elif settings.method == "manual":
        manual = settings.manual
        estimator = ManualEstimator(
            source_channel_name=settings.source.channel,
            target_channel_name=settings.target_channel,
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
    source_position_dirpath: Path,
    target_position_dirpath: Path,
    settings_path: Path,
    t: int,
    record_path: Path,
) -> dict:
    """One independent estimate, from the config seed, written as a JSON record."""
    settings = yaml_to_model(settings_path, EstimateTransformSettings)
    mov, mov_voxel_size = _open_series(source_position_dirpath, settings.source.channel)
    ref, ref_voxel_size = _open_series(target_position_dirpath, settings.target_channel)
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


def _load_series(
    records_dir: Path,
    time_indices: list[int],
    transform_type: str,
    records: dict[int, dict] | None = None,
) -> SeriesResult:
    """Rebuild the series from per-timepoint records.

    In-memory records first, then the files on disk (resumed timepoints); a timepoint
    with neither is an error.
    """
    records = dict(records or {})
    result = SeriesResult()
    for t in time_indices:
        record = records.get(t)
        if record is None:
            path = records_dir / f"{t}.json"
            if not path.exists():
                result.scores[t] = float("nan")
                result.errors[t] = "no record: job did not finish"
                continue
            record = json.loads(path.read_text())
        if record["matrix"] is not None:
            result.transforms[t] = Transform(
                np.asarray(record["matrix"], dtype=float), transform_type=transform_type
            )
        result.scores[t] = float("nan") if record["score"] is None else record["score"]
        if record["error"]:
            result.errors[t] = record["error"]
    return result


def _repair_timepoint_job(
    source_position_dirpath: Path,
    target_position_dirpath: Path,
    settings_path: Path,
    t: int,
    time_indices: list[int],
    flagged: list[int],
    records_dir: Path,
    record_path: Path,
) -> dict:
    """Repair one flagged timepoint against the frozen whole-run history."""
    settings = yaml_to_model(settings_path, EstimateTransformSettings)
    mov, mov_voxel_size = _open_series(source_position_dirpath, settings.source.channel)
    ref, ref_voxel_size = _open_series(target_position_dirpath, settings.target_channel)
    estimator, score_fn, seed = build_estimator(
        settings, tuple(mov.shape[-3:]), mov_voxel_size, ref_voxel_size
    )
    repair_settings = settings.fallback.repair

    series = _load_series(records_dir, time_indices, settings.transform.type)
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


def _load_repair(record: dict, transform_type: str, fallback: Transform) -> RepairResult:
    return RepairResult(
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


def _one_transform_per_timepoint(
    result: SeriesResult, time_indices: list[int], fallback: Transform
) -> list[Transform]:
    """Return one transform per requested timepoint.

    The accepted transform, else the nearest earlier accepted one, else `fallback`.
    """
    out, last = [], fallback
    for t in time_indices:
        last = result.transforms.get(t, last)
        out.append(last)
    return out


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
                "accepted": r.accepted,
                "source": r.source,
                "score": _finite_or_none(r.score),
                "candidate_scores": {k: _finite_or_none(v) for k, v in r.scores.items()},
                "candidate_failures": r.failures,
            }
            for t, r in result.repairs.items()
        },
        "filled_from_neighbour": [t for t in time_indices if t not in result.transforms],
    }


def estimate_transform_series(
    source_position_dirpath: Path,
    target_position_dirpath: Path,
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
    slurm_out_path = output_dir / "slurm_output"
    slurm_out_path.mkdir(exist_ok=True)
    source, target = Path(source_position_dirpath), Path(target_position_dirpath)

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
        timepoints_dir, time_indices, transform_type, records=estimate_records
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
    flagged = flag_series(
        result,
        max_timepoints=repair_settings.max_timepoints if repair_settings else None,
        k_mad=flag.k_mad,
        floor=flag.floor,
        hard_fail=flag.hard_fail,
    )
    to_repair = (
        [t for t in flagged if not (resume and (repairs_dir / f"{t}.json").exists())]
        if repair_settings is not None
        else []
    )
    executor.update_parameters(slurm_time=60, slurm_job_name="estimate_transform_repair")
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
    for t in flagged if repair_settings is not None else []:
        record = repair_records.get(t)
        if record is None:
            record_path = repairs_dir / f"{t}.json"
            if not record_path.exists():
                continue
            record = json.loads(record_path.read_text())
        outcome = _load_repair(record, transform_type, fallback=result.transforms.get(t, seed))
        result.repairs[t] = outcome
        result.journal.record(
            t=t,
            pass_name="repair",
            before_score=result.scores[t],
            after_score=outcome.score,
            accepted=outcome.accepted,
            failures=outcome.failures,
        )
        if outcome.accepted:
            result.transforms[t] = outcome.transform
            result.scores[t] = outcome.score
            result.errors.pop(t, None)
        click.echo(f"repair t={t}: {outcome.source} -> {outcome.score:.4f}")

    result.journal.save(output_dir / "run_journal.json")
    (output_dir / "estimate_transform_report.json").write_text(
        json.dumps(_report(result, time_indices), indent=2)
    )
    return result, time_indices, _one_transform_per_timepoint(result, time_indices, seed)
