"""Where a run's work happens: what one job does, and submitting jobs and waiting for them.

A job is a fresh process (a SLURM task, or this one with --cluster debug / --step): it
rebuilds the run's objects from the saved settings, estimates (or repairs, sweeps) its
timepoints, computes the per-timepoint extras and writes its record. The driver side
submits those jobs through submitit, waits, and reports what failed.
"""

from __future__ import annotations

import json

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import submitit

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import sbatch_to_submitit
from biahub.core.transform import Transform
from biahub.registration.engine.core import (
    ESTIMATE_MINUTES,
    _final,
    _finite_or_none,
    _initial_policy,
    _metric_fn,
    _propagation_minutes,
    _reference_policy,
    _score_fn,
    _static_initial,
    build_estimator,
    finalize_run,
    flag_run,
    init_run,
)
from biahub.registration.engine.disk import (
    ENGINE_SETTINGS_FILENAME,
    JOB_FAILED_PREFIX,
    _failed_record,
    _finished,
    _flags,
    _job_failure,
    _load_series,
    _open_series,
    _read_record,
    _record_dirs,
    _run_plan,
    _write_json,
)
from biahub.registration.engine.fallback import (
    PassRecord,
    neighbour_consensus_initial_candidates,
    repair_timepoint,
    sweep_timepoint,
)
from biahub.registration.engine.policies import (
    FixedInitialTransform,
    InitialTransformPolicy,
    ReferencePolicy,
)
from biahub.registration.engine.series import (
    SeriesResult,
    TimepointRecord,
    estimate_propagated,
    estimate_series,
)
from biahub.registration.estimators import (
    ScoreFn,
    TransformEstimator,
)
from biahub.registration.methods.intensity.ants import use_task_threads
from biahub.registration.metrics import (
    PREWARPED,
    bead_alignment_metrics,
    intensity_centroid,
    raw_shift,
)
from biahub.registration.utils import RunError, resolve_time_indices
from biahub.settings import (
    EstimateTransformSettings,
)
from biahub.utils.cluster import (
    apply_sbatch_overrides,
    get_submitit_cluster,
)
from biahub.utils.config import yaml_to_model


def _bead_metrics(settings, result, t, mov, ref, reference_policy=None) -> dict | None:
    """Continuous bead residuals next to the score, for the beads method."""
    final = _final(settings)
    if final.method != "bead_matching" or t not in result.transforms:
        return None
    policy = reference_policy or _reference_policy(settings, ref)
    ref_t = np.asarray(policy.reference_for(mov, t))
    metrics = bead_alignment_metrics(
        result.transforms[t], np.asarray(mov[t]), ref_t, final.bead_matching
    )
    return None if metrics is None else metrics.to_dict()


def _report_metrics(
    settings, transform, t, mov, ref, score: float | None = None, reference_policy=None
) -> dict | None:
    """Return score.report metrics of `transform` at `t`, on the raw warped pair.

    Also the unregistered pair's (identity): its score and each report metric, as
    `unregistered_<name>`, so a result can be read against doing nothing; a warning when
    the unregistered pair scores at least as high as the result's `score` (not for an
    identity result, e.g. a stabilization reference frame, which is that pair).
    """
    if not settings.score.report or transform is None:
        return None
    policy = reference_policy or _reference_policy(settings, ref)
    ref_t = np.asarray(policy.reference_for(mov, t))
    mov_t = np.asarray(mov[t])
    click.echo(f"t={t}: report metrics {settings.score.report}, and the unregistered pair's")
    report = {}
    # one warp for every metric, for the result and for the unregistered pair
    for prefix, candidate in (("", transform), ("unregistered_", Transform.identity(3))):
        warped = candidate.apply(mov_t, reference=ref_t)
        for metric in settings.score.report:
            fn = _metric_fn(settings, metric)
            report[prefix + metric] = _finite_or_none(fn(PREWARPED, warped, ref_t))
        if prefix:
            report["unregistered_score"] = _finite_or_none(
                _score_fn(settings)(PREWARPED, warped, ref_t)
            )
    unregistered = report["unregistered_score"]
    if score is not None and unregistered is not None and np.isfinite(score):
        if unregistered >= score and not transform.is_identity:
            click.echo(
                f"t={t}: the unregistered pair scores {unregistered:.4g}, at least the "
                f"result's {score:.4g}: check that the score metric suits this pair",
                err=False,
            )
    return report


def _anchor_centroid(settings: EstimateTransformSettings, mov, t: int) -> list[float] | None:
    """Return frame t's centroid for `reference.anchor` (None: no anchor, or no signal)."""
    anchor = settings.reference.anchor
    if anchor is None:
        return None
    centroid = intensity_centroid(
        mov[t],
        bg_percentile=anchor.bg_percentile,
        downsample=anchor.downsample,
        blur_sigma=anchor.blur_sigma,
    )
    return centroid.tolist() if np.isfinite(centroid).all() else None


def _previous_in_run(settings: EstimateTransformSettings, mov, t: int) -> int | None:
    """Return the run's timepoint before `t` (None at its first)."""
    earlier = [k for k in resolve_time_indices(settings.time_indices, len(mov)) if k < t]
    return earlier[-1] if earlier else None


def _leftover_raw_shift(settings: EstimateTransformSettings, mov, t: int) -> dict | None:
    """Return frame t's raw shift from the run's previous timepoint (`reference.leftover_motion`).

    None without the setting, or at the run's first timepoint; `shift` is None when
    either frame has no data.
    """
    if settings.reference.leftover_motion is None:
        return None
    previous = _previous_in_run(settings, mov, t)
    if previous is None:
        return None
    shift = raw_shift(mov[previous], mov[t])
    return {"from": previous, "shift": None if shift is None else shift.tolist()}


class _JobFrames:
    """A frame series whose frames a job uses more than once (`keep`) are read once.

    Other frames (e.g. a neighbour window, read once by the reference policy) pass
    straight through, so they are not all held at once. Kept frames are read-only: a
    step that changed one in place would corrupt the next one's input.
    """

    def __init__(self, series):
        self.series = series
        self.shape = series.shape
        self.keep: set[int] = set()
        self._kept: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, t):
        if not isinstance(t, (int, np.integer)) or int(t) not in self.keep:
            return self.series[t]
        t = int(t)
        if t not in self._kept:
            frame = np.asarray(self.series[t])
            frame.flags.writeable = False
            self._kept[t] = frame
        return self._kept[t]

    def keep_only(self, timepoints) -> None:
        """Keep these frames (read on first use); forget the others."""
        self.keep = {int(t) for t in timepoints if t is not None}
        self._kept = {t: f for t, f in self._kept.items() if t in self.keep}


class _JobInputs(NamedTuple):
    settings: EstimateTransformSettings
    mov: object
    ref: object
    mov_voxel_size: tuple
    ref_voxel_size: tuple
    estimator: TransformEstimator
    score_fn: ScoreFn
    # where each timepoint starts (`transform.initial`)
    initial: InitialTransformPolicy
    # what each timepoint is compared against, built once for the job (its caches, e.g.
    # the neighbour median, serve the estimate and the per-timepoint extras alike)
    reference_policy: ReferencePolicy


def _job_inputs(
    moving_position_dirpath: Path, reference_position_dirpath: Path, settings_path: Path
) -> _JobInputs:
    """Return what every job starts from: settings, series, estimator, initial transform."""
    use_task_threads()  # a fresh job process: before any ITK operation
    settings = yaml_to_model(settings_path, EstimateTransformSettings)
    mov, mov_voxel_size = _open_series(moving_position_dirpath, settings.moving.channel)
    ref, ref_voxel_size = _open_series(reference_position_dirpath, settings.reference_channel)
    mov, ref = _JobFrames(mov), _JobFrames(ref)
    estimator, score_fn, _ = build_estimator(
        settings, tuple(mov.shape[-3:]), mov_voxel_size, ref_voxel_size
    )
    initial = _initial_policy(settings, Path(settings_path).parent)
    return _JobInputs(
        settings,
        mov,
        ref,
        mov_voxel_size,
        ref_voxel_size,
        estimator,
        score_fn,
        initial,
        _reference_policy(settings, ref),
    )


def _estimate_timepoint_job(
    moving_position_dirpath: Path,
    reference_position_dirpath: Path,
    settings_path: Path,
    t: int,
    record_path: Path,
) -> dict:
    """One independent estimate, from the run's initial transform, written as a JSON record."""
    job = _job_inputs(moving_position_dirpath, reference_position_dirpath, settings_path)
    settings, mov, ref = job.settings, job.mov, job.ref
    # t is read by the estimate and every extra; the previous timepoint only by leftover
    # motion (kept otherwise, it would be one more volume than the job's memory plan)
    previous = (
        _previous_in_run(settings, mov, t) if settings.reference.leftover_motion else None
    )
    mov.keep_only([t, previous])
    ref.keep_only([t])
    estimator, score_fn = job.estimator, job.score_fn

    result = estimate_series(
        mov,
        job.reference_policy,
        estimator,
        FixedInitialTransform(job.initial.initial_for(t)),
        score_fn,
        [t],
        keep_initial=job.initial.keep_unless_beaten,
    )
    record = TimepointRecord(
        t=t,
        matrix=result.transforms[t].to_list() if t in result.transforms else None,
        score=_finite_or_none(result.scores.get(t)),
        error=result.errors.get(t),
        initialized_from=result.initialized_from.get(t),
        best_candidate=result.best_candidate.get(t),
        # a stack: what each block did (its transform, score, error) and which won
        blocks=result.blocks.get(t),
        metrics=_bead_metrics(settings, result, t, mov, ref, job.reference_policy),
        report=_report_metrics(
            settings,
            result.transforms.get(t),
            t,
            mov,
            ref,
            score=result.scores.get(t),
            reference_policy=job.reference_policy,
        ),
        centroid=_anchor_centroid(settings, mov, t),
        raw_shift=_leftover_raw_shift(settings, mov, t),
    ).to_dict()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(record_path, record)
    return record


def _estimate_propagated_job(
    moving_position_dirpath: Path,
    reference_position_dirpath: Path,
    settings_path: Path,
    time_indices: list[int],
    records_dir: Path,
    resume: bool,
) -> dict[int, dict]:
    """All timepoints in order (`transform.initial.propagate`), one record per timepoint.

    Each record is written as soon as its timepoint is done, so an interrupted job
    resumes the chain where it stopped.
    """
    job = _job_inputs(moving_position_dirpath, reference_position_dirpath, settings_path)
    settings, mov, ref = job.settings, job.mov, job.ref
    estimator, score_fn = job.estimator, job.score_fn
    initial_transform = _static_initial(settings)
    done = {}
    if resume:
        for t in time_indices:
            path = records_dir / f"{t}.json"
            if _finished(path):  # a failed record is redone
                done[t] = json.loads(path.read_text())
    records = dict(done)
    records_dir.mkdir(parents=True, exist_ok=True)

    order = list(time_indices)

    def keep_for(t: int) -> None:
        # t (its extras) and the next timepoint (estimated next, then its extras)
        following = order[order.index(t) + 1] if order.index(t) + 1 < len(order) else None
        mov.keep_only([t, following])
        ref.keep_only([following])

    if order:
        mov.keep_only([order[0]])
        ref.keep_only([order[0]])

    def write_record(t: int, result: SeriesResult) -> None:
        if t in done:
            keep_for(t)
            return
        stand_in = result.stand_ins.get(t)
        record = TimepointRecord(
            t=t,
            matrix=result.transforms[t].to_list() if t in result.transforms else None,
            score=_finite_or_none(result.scores.get(t)),
            error=result.errors.get(t),
            stand_in=stand_in[0].to_list() if stand_in else None,
            stand_in_from=stand_in[1] if stand_in else None,
            best_candidate=result.best_candidate.get(t),
            blocks=result.blocks.get(t),
            metrics=_bead_metrics(settings, result, t, mov, ref, job.reference_policy),
            report=_report_metrics(
                settings,
                result.transforms.get(t),
                t,
                mov,
                ref,
                score=result.scores.get(t),
                reference_policy=job.reference_policy,
            ),
            centroid=_anchor_centroid(settings, mov, t),
            raw_shift=_leftover_raw_shift(settings, mov, t),
        ).to_dict()
        _write_json(records_dir / f"{t}.json", record)
        records[t] = record
        keep_for(t)

    estimate_propagated(
        mov,
        job.reference_policy,
        estimator,
        initial_transform,
        score_fn,
        time_indices,
        done=done,
        on_timepoint=write_record,
        transform_type=settings.transform.type,
    )
    return records


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
    job = _job_inputs(moving_position_dirpath, reference_position_dirpath, settings_path)
    settings, mov, ref = job.settings, job.mov, job.ref
    mov.keep_only([t])
    ref.keep_only([t])
    estimator, score_fn, initial_transform = (
        job.estimator,
        job.score_fn,
        job.initial.initial_for(t),
    )
    repair_settings = settings.fallback.repair

    # This run's estimate records (the driver cleared any earlier run's).
    series = _load_series(
        records_dir, time_indices, settings.transform.type, resumed=time_indices
    )
    series.flagged = list(flagged)
    outcome = repair_timepoint(
        t,
        mov,
        job.reference_policy,
        estimator,
        score_fn,
        series,
        neighbour_consensus_initial_candidates(
            initial_transform,
            consensus_score_threshold=repair_settings.consensus_threshold,
            consensus_min_good=repair_settings.consensus_min_good,
            order=tuple(repair_settings.candidates),
        ),
        polish_rounds=repair_settings.polish_rounds,
    )
    record = PassRecord(
        t=t,
        accepted=outcome.accepted,
        source=outcome.source,
        score=_finite_or_none(outcome.score),
        matrix=outcome.transform.to_list() if outcome.accepted else None,
        candidate_scores={k: _finite_or_none(v) for k, v in outcome.scores.items()},
        candidate_failures=outcome.failures,
        polish_rounds=outcome.polish_rounds,
        repair_transform_score=_finite_or_none(outcome.repair_transform_score),
        report=_report_metrics(
            settings,
            outcome.transform if outcome.accepted else None,
            t,
            mov,
            ref,
            score=outcome.score,
            reference_policy=job.reference_policy,
        ),
    ).to_dict()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(record_path, record)
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
    job = _job_inputs(moving_position_dirpath, reference_position_dirpath, settings_path)
    settings, mov, ref = job.settings, job.mov, job.ref
    mov.keep_only([t])
    ref.keep_only([t])
    score_fn, initial_transform = job.score_fn, job.initial.initial_for(t)
    trials = {
        name: build_estimator(
            trial, tuple(mov.shape[-3:]), job.mov_voxel_size, job.ref_voxel_size
        )[0]
        for name, trial in settings.sweep_trials().items()
    }

    series = _load_series(records_dir, [t], settings.transform.type, resumed=[t])
    outcome = sweep_timepoint(
        t, mov, job.reference_policy, trials, initial_transform, score_fn, series
    )
    record = PassRecord(
        t=t,
        accepted=outcome.accepted,
        source=outcome.source,
        score=_finite_or_none(outcome.score),
        matrix=outcome.transform.to_list() if outcome.accepted else None,
        candidate_scores={k: _finite_or_none(v) for k, v in outcome.scores.items()},
        candidate_failures=outcome.failures,
        report=_report_metrics(
            settings,
            outcome.transform if outcome.accepted else None,
            t,
            mov,
            ref,
            score=outcome.score,
            reference_policy=job.reference_policy,
        ),
    ).to_dict()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(record_path, record)
    return record


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
            failures[t] = _job_failure(e)
            click.echo(f"{label} t={t}: {failures[t]}")
    return records, failures


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
        minutes = _propagation_minutes(len(to_estimate))
        executor.update_parameters(slurm_time=minutes)
    executor.update_parameters(slurm_job_name="estimate_transform_propagated")
    by_job, failures = _run_jobs(
        executor,
        resolved_cluster,
        monitor_flag,
        "estimate",
        [(-1, _estimate_propagated_job, job_args)],
    )
    if -1 in by_job:
        return {int(t): r for t, r in by_job[-1].items()}, {}
    records = {}
    for t in to_estimate:
        record = _read_record(records_dir / f"{t}.json")
        if record is not None:
            records[t] = record
    error = failures.get(-1, f"{JOB_FAILED_PREFIX}: the sequential job did not finish")
    return records, {t: error for t in to_estimate if t not in records}


def run_timepoint_jobs(
    step: str,
    moving_position_dirpath: Path,
    reference_position_dirpath: Path,
    output_dir: Path,
    timepoints: list[int] | None = None,
    resume: bool = False,
) -> tuple[dict[int, dict], list[int]]:
    """Run one phase's jobs for the given timepoints in this process.

    Returns their records and the timepoints whose job raised (recorded as the driver
    records a failed job, so the run can carry on; the caller decides how to report them).

    `step` is `estimate`, `repair` or `sweep`; `timepoints` defaults to every timepoint
    the phase has (all for estimate, the flagged ones for repair / sweep). A propagated
    estimate (`transform.initial.propagate`) is one job over the whole series. With
    `resume`, timepoints that already have a record are skipped.
    """
    output_dir = Path(output_dir)
    plan = _run_plan(output_dir)
    settings_path = output_dir / ENGINE_SETTINGS_FILENAME
    source, target = Path(moving_position_dirpath), Path(reference_position_dirpath)
    time_indices = plan["time_indices"]
    timepoints_dir, repairs_dir, sweeps_dir = _record_dirs(output_dir)

    if step == "estimate" and plan["propagated"]:
        if timepoints is not None:
            raise RunError(
                "transform.initial.propagate estimates the whole series in one job, in "
                "order; drop --timepoints"
            )
        try:
            return (
                _estimate_propagated_job(
                    source, target, settings_path, time_indices, timepoints_dir, resume
                ),
                [],
            )
        except Exception as e:  # noqa: BLE001 -- recorded per timepoint, as the driver does
            # The timepoints it reached keep their records; the rest failed with it.
            click.echo(f"estimate: {_job_failure(e)}")
            timepoints_dir.mkdir(parents=True, exist_ok=True)
            records, failed = {}, []
            for t in time_indices:
                path = timepoints_dir / f"{t}.json"
                if not _finished(path):
                    _write_json(path, _failed_record(t, e))
                    failed.append(t)
                records[t] = json.loads(path.read_text())
            return records, failed

    if step == "estimate":
        available, records_dir = time_indices, timepoints_dir
    elif step in ("repair", "sweep"):
        available = _flags(output_dir)[step]
        records_dir = repairs_dir if step == "repair" else sweeps_dir
    else:
        raise ValueError(f"unknown step {step!r}")
    wanted = available if timepoints is None else list(timepoints)
    unknown = sorted(set(wanted) - set(available))
    if unknown:
        raise RunError(
            f"{step}: timepoints {unknown} are not in this run's {step} list {available}"
        )
    flagged = _flags(output_dir)["flagged"] if step == "repair" else None
    records, failed = {}, []
    for t in wanted:
        record_path = records_dir / f"{t}.json"
        if resume and _finished(record_path):
            continue
        try:
            records[t] = _run_timepoint_job(
                step,
                source,
                target,
                settings_path,
                t,
                record_path,
                time_indices,
                timepoints_dir,
                flagged,
            )
        except Exception as e:  # noqa: BLE001 -- one timepoint's failure must not end the run
            # As the driver records a job that raised: an estimate becomes a failed record
            # (the timepoint gets a stand-in, the error in its note); a repair / sweep
            # leaves no record, so the timepoint keeps its estimate.
            click.echo(f"{step} t={t}: {_job_failure(e)}")
            if step == "estimate":
                record_path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(record_path, _failed_record(t, e))
            failed.append(t)
            continue
        click.echo(f"{step} t={t}: score={records[t].get('score')}")
    return records, failed


def _run_timepoint_job(
    step, source, target, settings_path, t, record_path, time_indices, timepoints_dir, flagged
) -> dict:
    """Run one timepoint's job of `step` in this process; return its record."""
    if step == "estimate":
        return _estimate_timepoint_job(source, target, settings_path, t, record_path)
    if step == "repair":
        return _repair_timepoint_job(
            source,
            target,
            settings_path,
            t,
            time_indices,
            flagged,
            timepoints_dir,
            record_path,
        )
    return _sweep_timepoint_job(source, target, settings_path, t, timepoints_dir, record_path)


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
    """Run every phase through submitit; return the series, its timepoints and one forward transform per timepoint.

    `init_run`, then one job per timepoint (or one sequential job), `flag_run`, one job
    per timepoint to repair and to sweep, and `finalize_run` -- the steps
    `estimate-transform --init / --step` runs one at a time. Writes the resolved settings
    (`estimate_transform_settings.yml`), the per-timepoint records (`timepoints/`,
    `repairs/`, `sweeps/`), `run_journal.json` and `estimate_transform_report.json` under
    `output_dir`.
    """
    output_dir = Path(output_dir)
    source, target = Path(moving_position_dirpath), Path(reference_position_dirpath)
    plan = init_run(source, target, settings, output_dir, resume)
    settings_path = output_dir / ENGINE_SETTINGS_FILENAME
    time_indices = plan["time_indices"]
    timepoints_dir, repairs_dir, sweeps_dir = _record_dirs(output_dir)
    slurm_out_path = output_dir / "slurm_output"
    slurm_out_path.mkdir(exist_ok=True)

    estimate = plan["resources"]["estimate"]
    slurm_args = {
        "slurm_job_name": "estimate_transform",
        "slurm_mem_per_cpu": f"{estimate['gb_ram_per_cpu']}G",
        "slurm_cpus_per_task": estimate["cpus"],
        "slurm_array_parallelism": 100,  # process up to N timepoints at a time
        "slurm_time": ESTIMATE_MINUTES,
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }
    if sbatch_filepath:
        slurm_args = apply_sbatch_overrides(slurm_args, sbatch_to_submitit(sbatch_filepath))
    # Manual registration is interactive (napari): in this process.
    resolved_cluster = (
        "debug" if plan["interactive"] else get_submitit_cluster(cluster=cluster)
    )
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    to_estimate = [
        t for t in time_indices if not (resume and _finished(timepoints_dir / f"{t}.json"))
    ]
    if resume:
        click.echo(
            f"resume: {len(time_indices) - len(to_estimate)} timepoint(s) already estimated"
        )
    # An sbatch file that sets a time limit wins over every phase's default below.
    user_set_time = bool(
        sbatch_filepath and "slurm_time" in sbatch_to_submitit(sbatch_filepath)
    )
    if plan["propagated"]:
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

    result, flags = flag_run(
        output_dir,
        records=estimate_records,
        failures=job_failures,
        from_disk=set(time_indices) - set(to_estimate),
    )

    to_repair = [
        t for t in flags["repair"] if not (resume and (repairs_dir / f"{t}.json").exists())
    ]
    executor.update_parameters(slurm_job_name="estimate_transform_repair")
    if not user_set_time:
        executor.update_parameters(slurm_time=plan["resources"]["repair"]["time_minutes"])
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
                    flags["flagged"],
                    timepoints_dir,
                    repairs_dir / f"{t}.json",
                ),
            )
            for t in to_repair
        ],
    )

    to_sweep = [
        t for t in flags["sweep"] if not (resume and (sweeps_dir / f"{t}.json").exists())
    ]
    executor.update_parameters(slurm_job_name="estimate_transform_sweep")
    if not user_set_time:
        executor.update_parameters(slurm_time=plan["resources"]["sweep"]["time_minutes"])
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

    return finalize_run(
        output_dir,
        result,
        flags,
        repair_records=repair_records,
        sweep_records=sweep_records,
        repaired=to_repair,
        swept=to_sweep,
    )
