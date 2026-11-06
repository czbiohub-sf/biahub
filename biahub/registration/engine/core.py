"""The engine's core: build a run from its settings, and its steps on disk.

`init_run` checks the config against the stores, builds the run's objects (reference and
initial-transform policies, score, estimator), plans each job's resources and starts the
run folder; `flag_run` picks the timepoints to repair; `finalize_run` folds repairs and
sweeps in and turns the series into one transform per timepoint (stand-ins, the
`previous` chain, the anchor, leftover motion) with the run's report. The estimating
itself is `series`, the fallback passes `fallback`, the run folder `disk`, the jobs and
their submission `cluster`.
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterable
from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import position_key
from biahub.core.transform import Transform
from biahub.registration.engine.disk import (
    ENGINE_SETTINGS_FILENAME,
    FLAGS_FILENAME,
    INITIAL_FILENAME,
    RUN_PLAN_FILENAME,
    RunJournal,
    _flags,
    _initial_transforms_from_file,
    _load_series,
    _read_record,
    _record_dirs,
    _run_plan,
    _run_settings,
    _start_run,
    _write_json,
)
from biahub.registration.engine.fallback import (
    PassRecord,
    PassResult,
    _accept,
    _fold_sweep,
    cap_worst,
    flag_series,
)
from biahub.registration.engine.policies import (
    CrossChannel,
    FileInitialTransform,
    FixedFrame,
    FixedInitialTransform,
    InitialTransformPolicy,
    NeighbourMedian,
    PreviousFrame,
    ReferencePolicy,
)
from biahub.registration.engine.series import (
    SeriesResult,
)
from biahub.registration.estimators import (
    ChainedEstimator,
    ScoreFn,
    TransformEstimator,
    is_higher,
)
from biahub.registration.methods.intensity.ants import AntsEstimator
from biahub.registration.methods.intensity.focus_finding import FocusEstimator
from biahub.registration.methods.intensity.local_phase_cross_corr import LocalPCCEstimator
from biahub.registration.methods.intensity.phase_cross_corr import PCCEstimator
from biahub.registration.methods.landmark.bead_matching import (
    build_beads_estimator,
)
from biahub.registration.methods.landmark.nuclei_matching import nuclei_estimator
from biahub.registration.methods.manual.manual import ManualEstimator
from biahub.registration.methods.manual.shapes import ShapesEstimator
from biahub.registration.metrics import (
    LEFTOVER_MOTION_VOLUMES,
    leftover_motion,
    metric_fn,
    smooth_track,
)
from biahub.registration.preprocess import (
    PreprocessedEstimator,
    RawVolumeScore,
    apply_image_blocks,
    blocks_footprint,
)
from biahub.registration.utils import RunError, resolve_time_indices
from biahub.settings import (
    AnchorSettings,
    EstimateTransformSettings,
    InitialTransformSettings,
)
from biahub.utils.cluster import (
    estimate_resources,
)
from biahub.utils.config import model_to_yaml


def _reference_policy(settings: EstimateTransformSettings, ref) -> ReferencePolicy:
    if settings.reference.frame == "cross":
        return CrossChannel(ref)
    if settings.reference.frame == "first":
        return FixedFrame(0)
    run = resolve_time_indices(settings.time_indices, len(ref))
    if settings.reference.frame == "neighbours":
        return NeighbourMedian(settings.reference.window, first=min(run), last=max(run))
    # the chain starts at the run's first timepoint: nothing before it is compared
    return PreviousFrame(first=min(run))


def _static_initial(settings: EstimateTransformSettings) -> Transform:
    """Return the run's one initial transform: `transform.initial.matrix`, else identity.

    `init_run` makes a `from: shapes` initial a matrix; a file's per-timepoint transforms
    are `_initial_policy`'s.
    """
    fit, init = settings.transform, settings.transform.initial
    if init is None or init.from_ != "matrix":
        return Transform(np.eye(4), transform_type=fit.type)
    if init.direction == "forward":
        return Transform(init.matrix, transform_type=fit.type)
    return Transform.from_inverse(init.matrix, fit.type)


def _initial_policy(
    settings: EstimateTransformSettings, run_dir: Path
) -> InitialTransformPolicy:
    """Return where each timepoint starts: the run's file, else the one initial transform.

    `init_run` wrote a file's transform for each timepoint beside the run's settings.
    """
    if settings.transform.initial_file is not None:
        by_t = json.loads((Path(run_dir) / INITIAL_FILENAME).read_text())
        return FileInitialTransform(
            {
                int(t): Transform(m, transform_type=settings.transform.type)
                for t, m in by_t.items()
            }
        )
    return FixedInitialTransform(_static_initial(settings))


def _score_fn(settings: EstimateTransformSettings) -> ScoreFn:
    """Return the score every decision uses: `settings.score`, never the method's blocks."""
    blocks = settings.score.preprocess
    prepare = None
    if blocks:

        def prepare(warped, ref, overlap):
            return (
                apply_image_blocks(warped, blocks, "moving", where=overlap),
                apply_image_blocks(ref, blocks, "reference", where=overlap),
            )

        # how far the blocks reach: the score leaves that margin by the empty padding
        prepare.footprint = blocks_footprint(blocks)
    return _metric_fn(settings, settings.effective_score_metric, prepare)


def _final(settings: EstimateTransformSettings) -> EstimateTransformSettings:
    """Return the config of the method with the last word (a stack's final block)."""
    return settings.stack_settings()[-1] if settings.is_stack else settings


def _metric_fn(settings: EstimateTransformSettings, metric: str, prepare=None) -> ScoreFn:
    """Return one metric as a score function, with the bead settings of a stack's final block."""
    return metric_fn(metric, _final(settings).bead_matching, prepare)


def build_estimator(
    settings: EstimateTransformSettings,
    shape_zyx: tuple[int, int, int] | None = None,
    mov_voxel_size: tuple[float, float, float] | None = None,
    ref_voxel_size: tuple[float, float, float] | None = None,
) -> tuple[TransformEstimator, ScoreFn]:
    """Return the estimator and score function for the settings' method.

    Where it starts is the initial-transform policy's (`_initial_policy`).
    """
    if settings.is_stack:
        # each block is the estimator its own config builds (its preprocess, transform
        # type, settings), started from the block before; the run's score keeps the best
        blocks = [
            build_estimator(block, shape_zyx, mov_voxel_size, ref_voxel_size)[0]
            for block in settings.stack_settings()
        ]
        score_fn = _score_fn(settings)
        names = [f"{k}:{m}" for k, m in enumerate(settings.method_names, start=1)]
        chain = ChainedEstimator(blocks, score_fn=score_fn, names=names)
        return chain, score_fn
    score_fn = _score_fn(settings)
    # the bead estimators' own choices score the raw volumes, not the prepared ones
    method_score = RawVolumeScore(score_fn) if settings.preprocess else score_fn
    if settings.method == "bead_matching":
        estimator: TransformEstimator = build_beads_estimator(
            settings.bead_matching, settings.transform.type, score_fn=method_score
        )
    elif settings.method == "ants":
        estimator = AntsEstimator.from_settings(
            settings.ants, settings.transform.type, verbose=settings.verbose
        )
    elif settings.method == "phase_cross_corr" and settings.phase_cross_corr.local:
        estimator = LocalPCCEstimator(
            settings.phase_cross_corr.local,
            settings.transform.type,
            ref_voxel_size or (1.0, 1.0, 1.0),
        )
    elif settings.method == "phase_cross_corr":
        estimator = PCCEstimator.from_settings(settings.phase_cross_corr, shape_zyx)
    elif settings.method == "focus_finding":
        estimator = FocusEstimator.from_settings(
            settings.focus_finding, pixel_size=(mov_voxel_size or (1.0, 1.0, 1.0))[-1]
        )
    elif settings.method == "nuclei_matching":
        estimator = nuclei_estimator(
            settings.nuclei_matching,
            settings.transform.type,
            ref_voxel_size or (1.0, 1.0, 1.0),
            score_fn=method_score,
        )
    elif settings.method == "shapes":
        estimator = ShapesEstimator.from_settings(
            settings.shapes,
            settings.transform.type,
            mov_voxel_size or (1.0, 1.0, 1.0),
            ref_voxel_size or (1.0, 1.0, 1.0),
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
        raise ValueError(f"unknown method '{settings.method}'")
    if settings.preprocess:
        estimator = PreprocessedEstimator(estimator, settings.preprocess, method_score)
    return estimator, score_fn


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


# Default wall-clock budgets per phase, in minutes (an sbatch file's time wins).
ESTIMATE_MINUTES = 30
REPAIR_MINUTES = 60
# Wall-clock budget per timepoint for the sequential (propagation) job, in minutes:
# 2024_11_07 A549 beads took 7-8 min per timepoint (two passes from two initial
# transforms), so ~1.3x.
PROPAGATION_MINUTES_PER_TIMEPOINT = 10
# The `preempted` partition's limit: sbatch rejects a longer request outright.
MAX_MINUTES = 2880


def _propagation_minutes(n_timepoints: int) -> int:
    """Return the sequential job's budget for this many timepoints, within the partition.

    As every step sizes its time: `estimate_resources` counts the ZYX volumes processed
    (one channel per timepoint) times the step's calibrated minutes per volume.
    """
    time_minutes, _, _ = estimate_resources(
        shape=(n_timepoints, 1, 1, 1, 1),
        time_multiplier=PROPAGATION_MINUTES_PER_TIMEPOINT,
        min_time_minutes=ESTIMATE_MINUTES,
    )
    return min(MAX_MINUTES, time_minutes)


def _one_transform_per_timepoint(
    result: SeriesResult,
    time_indices: list[int],
    initial: Transform | InitialTransformPolicy,
) -> list[Transform]:
    """Return one transform per requested timepoint.

    The accepted transform, else the timepoint's initial transform (e.g. the one
    from the two grids, or a file's own for that timepoint), as the legacy pipeline returned when
    refinement failed. Each stand-in is recorded in `result.filled_from`.
    """
    if isinstance(initial, Transform):
        initial = FixedInitialTransform(initial)
    from_file = initial.keep_unless_beaten
    out = []
    for t in time_indices:
        if t in result.transforms:
            out.append(result.transforms[t])
        elif t in result.stand_ins:
            transform, source = result.stand_ins[t]
            result.filled_from[t] = source
            out.append(transform)
        else:
            result.filled_from[t] = "file" if from_file else "initial"
            out.append(initial.initial_for(t))
    return out


def _pass_report(outcome: PassResult) -> dict:
    return {
        "accepted": outcome.accepted,
        "source": outcome.source,
        "score": _finite_or_none(outcome.score),
        "candidate_scores": {k: _finite_or_none(v) for k, v in outcome.scores.items()},
        "candidate_failures": outcome.failures,
    }


def _report(result: SeriesResult, time_indices: list[int], run_id: str) -> dict:
    return {
        "run_id": run_id,
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
                "repair_transform_score": _finite_or_none(r.repair_transform_score),
            }
            for t, r in result.repairs.items()
        },
        "sweeps": {str(t): _pass_report(r) for t, r in result.sweeps.items()},
        "provenance": {str(t): source for t, source in sorted(result.provenance.items())},
        "stand_ins": {str(t): source for t, source in sorted(result.filled_from.items())},
        **({"blocks": _blocks_summary(result.blocks)} if result.blocks else {}),
        **(
            {"anchor": {str(t): a for t, a in result.anchors.items()}}
            if result.anchors
            else {}
        ),
        **(
            {"leftover_motion": result.leftover_motion}
            if result.leftover_motion is not None
            else {}
        ),
    }


def block_failure(stages: list[dict] | None) -> str | None:
    """Say which block of a stack failed and whose result was kept instead, or None.

    Blocks after a failed one are "not run"; the first failure is the one to report.
    """
    failed = next((s for s in stages or [] if s["matrix"] is None and s["error"]), None)
    if failed is None:
        return None
    kept = next((s["block"] for s in stages if s["won"]), None)
    message = f"block {failed['block']} failed ({failed['error']})"
    return f"{message}; kept block {kept}'s result" if kept else message


def _blocks_summary(blocks: dict[int, list[dict]]) -> dict:
    """Return, per block of a stack, how often it won or failed and its median score."""
    summary = {}
    for stages in blocks.values():
        for stage in stages:
            entry = summary.setdefault(stage["block"], {"won": 0, "failed": 0, "scores": []})
            entry["won"] += bool(stage["won"])
            entry["failed"] += stage["matrix"] is None
            if stage["score"] is not None:
                entry["scores"].append(stage["score"])
    return {
        name: {
            "won": e["won"],
            "failed": e["failed"],
            "median_score": float(np.median(e["scores"])) if e["scores"] else None,
        }
        for name, e in summary.items()
    }


def _sweep_minutes(n_trials: int) -> int:
    return 30 + 3 * n_trials


# Memory of one estimate job, in volumes of the moving store, whatever its CPU count (the
# peak is the job's copies, not per thread). ANTs copies both volumes several times (Sobel,
# crop, its image pyramid): on a 66 x 2048 x 2048 pair it peaked at 14.5-22.6 GB (SLURM
# MaxRSS, 12 jobs), i.e. ~22 volumes; 30 leaves ~30% headroom. The other methods fit in 10.
RAM_VOLUMES_PER_JOB = {"ants": 30}
DEFAULT_RAM_VOLUMES_PER_JOB = 10
# CPUs per estimate job: ANTs' optimizer runs ITK threads, one per CPU (`use_task_threads`);
# the other methods are single-threaded.
CPUS_PER_JOB = {"ants": 8}
DEFAULT_CPUS_PER_JOB = 2
# SLURM's smallest useful memory request per CPU, in GB.
MIN_GB_PER_CPU = 4


def _estimate_job_resources(
    method: str, zyx_shape: tuple[int, int, int], extra_volumes: int = 0, at_least: int = 0
) -> tuple[int, int]:
    """Return (cpus, GB of RAM per CPU) for one estimate job of `method` on this volume.

    `extra_volumes`: volumes the reference holds besides the method's own (a neighbour
    median holds its frames and the median). `at_least`: volumes a later step of the
    job holds on its own, once the method's are freed (leftover motion's raw shift).
    """
    # in CI the suite runs serially (as estimate_resources does)
    in_ci = os.environ.get("CI") == "true"
    num_cpus = 1 if in_ci else CPUS_PER_JOB.get(method, DEFAULT_CPUS_PER_JOB)
    gb_per_volume = np.prod(zyx_shape) * np.dtype(np.float32).itemsize / 2**30
    volumes = max(
        RAM_VOLUMES_PER_JOB.get(method, DEFAULT_RAM_VOLUMES_PER_JOB) + extra_volumes, at_least
    )
    job_gb = gb_per_volume * volumes
    gb_per_cpu = max(MIN_GB_PER_CPU, np.ceil(gb_per_volume), np.ceil(job_gb / num_cpus))
    return num_cpus, int(gb_per_cpu)


def init_run(
    moving_position_dirpath: Path,
    reference_position_dirpath: Path,
    settings: EstimateTransformSettings,
    output_dir: Path,
    resume: bool = False,
) -> dict:
    """Start a run in `output_dir` and return its plan (also written as `run_plan.json`).

    Resolves the settings the jobs read (an initial transform from the store shapes is
    computed once, here), writes them with the run manifest -- clearing an earlier run's
    records unless `resume` -- and plans the run: the timepoints, whether they are
    estimated in one sequential job (`transform.initial.propagate`), whether the method
    uses an initial transform (repair needs one), and each phase's resources. Reads only
    store metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source, target = Path(moving_position_dirpath), Path(reference_position_dirpath)
    with open_ome_zarr(source, mode="r") as position:
        T, _C, Z, Y, X = position.data.shape
        mov_voxel_size = tuple(position.scale[-3:])
    with open_ome_zarr(target, mode="r") as position:
        ref_T = position.data.shape[0]
        ref_shape = position.data.shape[-3:]
        ref_voxel_size = tuple(position.scale[-3:])
    # Checked before anything is written: a refused run must not clear an earlier one's
    # records.
    time_indices = resolve_time_indices(settings.time_indices, T)
    if settings.reference.frame == "cross":
        beyond = [t for t in time_indices if t >= ref_T]
        if beyond:
            raise RunError(
                f"timepoints {beyond} are past the reference store, which has {ref_T} "
                f"timepoints ({target})"
            )

    settings = settings.model_copy(deep=True)
    init = settings.transform.initial
    if init is not None and init.from_ == "shapes":
        # the run's settings hold the matrix it computed, so every job reads the same one
        method = ShapesEstimator(
            mov_voxel_size, ref_voxel_size, init.rotate90, init.fliplr, settings.transform.type
        )
        shapes = method.transform_for((Z, Y, X), tuple(ref_shape))
        settings.transform.initial = InitialTransformSettings(
            from_="matrix",
            matrix=shapes.to_list(),
            direction="forward",
            propagate=init.propagate,
        )
        click.echo(
            f"Computed the initial transform from the store shapes (forward):\n{shapes.matrix}"
        )
    estimator, _score_fn = build_estimator(settings, (Z, Y, X), mov_voxel_size, ref_voxel_size)
    initial_transforms = (
        None
        if settings.transform.initial_file is None
        else _initial_transforms_from_file(
            settings.transform.initial_file, position_key(source), time_indices
        )
    )
    _start_run(output_dir, settings, source, target, resume)
    model_to_yaml(settings, output_dir / ENGINE_SETTINGS_FILENAME)
    if initial_transforms is not None:
        _write_json(output_dir / INITIAL_FILENAME, initial_transforms, indent=2)

    propagated = settings.transform.propagates
    # a stack's jobs run every block: plan for the heaviest
    # a neighbour median holds 2 x window frames, the one being read and the median,
    # besides the method's volumes
    reference_volumes = (
        2 * settings.reference.window + 2 if settings.reference.frame == "neighbours" else 0
    )
    # each volume sized as the larger grid: the warped moving volume, the reference and
    # the scores' arrays live on the reference grid
    volume_zyx = max((Z, Y, X), tuple(ref_shape), key=np.prod)
    num_cpus, gb_ram_per_cpu = max(
        (
            _estimate_job_resources(
                m,
                volume_zyx,
                reference_volumes,
                at_least=LEFTOVER_MOTION_VOLUMES if settings.reference.leftover_motion else 0,
            )
            for m in settings.method_names
        ),
        key=lambda r: r[0] * r[1],
    )
    n_trials = len(settings.sweep_trials()) if settings.fallback.sweep is not None else 0
    # a stack's blocks run one after another in each job
    estimate_minutes = len(settings.method_names) * (
        _propagation_minutes(len(time_indices)) if propagated else ESTIMATE_MINUTES
    )

    def resources(minutes: int) -> dict:
        return {
            "cpus": num_cpus,
            "mem_gb": num_cpus * gb_ram_per_cpu,
            "gb_ram_per_cpu": gb_ram_per_cpu,
            "time_minutes": minutes,
        }

    plan = {
        "time_indices": time_indices,
        "frame_shape": [Z, Y, X],
        "voxel_um": list(mov_voxel_size),
        "propagated": propagated,
        "uses_initial_transform": estimator.uses_initial_transform,
        "interactive": "manual" in settings.method_names,
        "resources": {
            "estimate": resources(estimate_minutes),
            "repair": resources(REPAIR_MINUTES),
            "sweep": resources(_sweep_minutes(n_trials)),
        },
    }
    _write_json(output_dir / RUN_PLAN_FILENAME, plan, indent=2)
    return plan


def flag_run(
    output_dir: Path,
    records: dict[int, dict] | None = None,
    failures: dict[int, str] | None = None,
    from_disk: Iterable[int] | None = None,
) -> tuple[SeriesResult, dict]:
    """Load the estimates and decide which timepoints to repair and sweep.

    Records come from `records` (this call's jobs) and, for the timepoints in
    `from_disk` (all of them when None), from the run's record files. Writes and returns
    the flags (`flags.json`): flagged, repair and sweep timepoints, and the scores they
    were decided on.
    """
    output_dir = Path(output_dir)
    settings, plan = _run_settings(output_dir), _run_plan(output_dir)
    time_indices = plan["time_indices"]
    timepoints_dir, _, _ = _record_dirs(output_dir)
    result = _load_series(
        timepoints_dir,
        time_indices,
        settings.transform.type,
        records=records,
        resumed=time_indices if from_disk is None else from_disk,
    )
    for t, error in (failures or {}).items():
        result.errors[t] = error
    for t in time_indices:
        line = f"t={t}: score={result.scores[t]:.4f}"
        if t in result.errors:
            line += f"  estimate failed: {result.errors[t]}"
        elif failure := block_failure(result.blocks.get(t)):
            line += f"  WARNING {failure}"
        click.echo(line)

    thresholds = settings.fallback_thresholds()
    flagged = flag_series(
        result,
        k_mad=settings.fallback.flag.k_mad,
        floor=thresholds.floor,
        hard_fail=thresholds.hard_fail,
    )
    base_scores = dict(result.scores)
    repair_settings, sweep_settings = settings.fallback.repair, settings.fallback.sweep
    repair_ts = (
        cap_worst(flagged, base_scores, repair_settings.max_timepoints)
        if repair_settings is not None
        else []
    )
    if repair_ts and not plan["uses_initial_transform"]:
        click.echo(
            f"repair skipped: method {settings.method_label!r} ignores the initial "
            "transform, so starting from another would return the same transforms "
            f"({len(repair_ts)} flagged timepoint(s) stay flagged)"
        )
        repair_ts = []
    sweep_ts = (
        cap_worst(flagged, base_scores, sweep_settings.max_timepoints)
        if sweep_settings is not None
        else []
    )
    if flagged and repair_settings is None and sweep_settings is None:
        click.echo(
            f"flag: repair and sweep are off; the {len(flagged)} flagged timepoint(s) are "
            "written as they are, marked unreliable where still below "
            + ("hard_fail" if thresholds.hard_fail is not None else "the run's line")
        )
    flags = {
        "flagged": flagged,
        "repair": repair_ts,
        "sweep": sweep_ts,
        "base_scores": {str(t): _finite_or_none(v) for t, v in base_scores.items()},
        "line": _finite_or_none(result.flag_line),
    }
    _write_json(output_dir / FLAGS_FILENAME, flags, indent=2)
    return result, flags


def finalize_run(
    output_dir: Path,
    result: SeriesResult | None = None,
    flags: dict | None = None,
    repair_records: dict[int, dict] | None = None,
    sweep_records: dict[int, dict] | None = None,
    repaired: Iterable[int] = (),
    swept: Iterable[int] = (),
) -> tuple[SeriesResult, list[int], list[Transform]]:
    """Fold the repairs and sweeps into the series; write the report and journal.

    `result` and `flags` default to what is on disk (the estimates, `flags.json`).
    Repair and sweep outcomes come from the records given, else from the record files --
    except for the timepoints `repaired` / `swept` by this call: a job that failed this
    call is never filled from an earlier record. Returns the series, its
    timepoints and one forward transform per timepoint.
    """
    output_dir = Path(output_dir)
    settings, plan = _run_settings(output_dir), _run_plan(output_dir)
    time_indices = plan["time_indices"]
    transform_type = settings.transform.type
    # a timepoint without a transform of its own falls back to where it started: the
    # initial transform file's transform for it, else the config initial transform
    initial = _initial_policy(settings, output_dir)
    _, repairs_dir, sweeps_dir = _record_dirs(output_dir)
    if flags is None:
        flags = _flags(output_dir)
    if result is None:
        timepoints_dir, _, _ = _record_dirs(output_dir)
        result = _load_series(
            timepoints_dir, time_indices, transform_type, resumed=time_indices
        )
        result.flagged = list(flags["flagged"])
        result.flag_line = flags.get("line")
    base_scores = {
        int(t): float("nan") if v is None else v for t, v in flags["base_scores"].items()
    }

    def outcome_for(t: int, records: dict[int, dict] | None, records_dir: Path, ran: set):
        record = (records or {}).get(t)
        if record is None:
            if t in ran:  # failed this call
                return None
            record = _read_record(records_dir / f"{t}.json")
            if record is None:  # never ran (or unreadable)
                return None
        return PassRecord.from_dict(record).to_result(
            transform_type, fallback=result.transforms.get(t, initial.initial_for(t))
        )

    journal = RunJournal()  # this finalize's fallback attempts, from the records
    for t in flags["repair"]:
        outcome = outcome_for(t, repair_records, repairs_dir, set(repaired))
        if outcome is None:
            continue
        result.repairs[t] = outcome
        repair_transform_score = (
            outcome.score
            if outcome.repair_transform_score is None
            else outcome.repair_transform_score
        )
        journal.record(
            t=t,
            pass_name="repair",
            before_score=result.scores[t],
            after_score=repair_transform_score,
            accepted=outcome.accepted,
            failures=outcome.failures,
        )
        if outcome.repair_transform_score is not None:
            journal.record(
                t=t,
                pass_name="polish",
                before_score=repair_transform_score,
                after_score=outcome.score,
                accepted=outcome.polish_rounds > 0,
            )
        # As a fresh repair is judged (and as the sweep fold does): only if it beats the
        # current estimate -- a repair kept by resume may predate a redone, better one.
        if outcome.accepted and is_higher(outcome.score, result.scores.get(t)):
            _accept(
                result, t, outcome.transform, outcome.score, outcome.source, outcome.report
            )
        click.echo(f"repair t={t}: {outcome.source} -> {outcome.score:.4f}")

    # The sweep competes with repair rather than following it: it starts from each
    # timepoint's pre-fallback estimate, and the better of the two is kept.
    for t in flags["sweep"]:
        outcome = outcome_for(t, sweep_records, sweeps_dir, set(swept))
        if outcome is None:
            continue
        journal.record(
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
    transforms = transforms_for_file(result, time_indices, initial, settings.reference.frame)
    if settings.reference.anchor is not None:
        transforms = anchor_transforms(
            result, time_indices, transforms, settings.reference.anchor, plan["frame_shape"]
        )
    if settings.reference.leftover_motion is not None:
        result.leftover_motion = leftover_motion_report(
            result, time_indices, transforms, plan["voxel_um"]
        )
    journal.save(output_dir / "run_journal.json")
    _write_json(
        output_dir / "estimate_transform_report.json",
        _report(result, time_indices, journal.current_run_id),
        indent=2,
    )
    return result, time_indices, transforms


def transforms_for_file(
    result: SeriesResult,
    time_indices: list[int],
    initial: Transform | InitialTransformPolicy,
    frame: str,
) -> list[Transform]:
    """One transform per timepoint, onto the reference grid, for the transforms file.

    `cross` / `first` transforms are absolute, so a timepoint with none takes the input
    initial transform. `previous` transforms are relative steps (t -> t-1): a
    missing step is identity -- reusing a neighbour's step would add its drift again to
    every later timepoint -- and the steps are chained onto the first frame.
    """
    if frame != "previous":
        return _one_transform_per_timepoint(result, time_indices, initial)
    for t in time_indices:
        if t not in result.transforms:
            result.filled_from[t] = "identity"
    steps = [result.transforms.get(t, Transform.identity()) for t in time_indices]
    return chain_to_first_frame(steps, time_indices)


def anchor_transforms(
    result: SeriesResult,
    time_indices: list[int],
    transforms: list[Transform],
    anchor: AnchorSettings,
    frame_shape: list[int],
) -> list[Transform]:
    """Add the centroid anchor to each timepoint's transform: the sample to the centre.

    The neighbour median follows slow drift, so its transforms hold only the jumps. The
    centroids (a timepoint without one is interpolated) are smoothed over the run's
    timepoints, in order, and the translation (frame centre - smoothed centroid) is
    composed after each transform, stand-ins included -- a forward translation is the
    shift applied to the volume. Records each timepoint's anchor in `result.anchors`.
    """
    track = [result.centroids.get(t) or [np.nan] * 3 for t in time_indices]
    smoothed = smooth_track(track, median=anchor.smooth_median, sigma=anchor.smooth_sigma)
    centre = (np.asarray(frame_shape, dtype=float) - 1) / 2
    out = []
    for t, transform, slow in zip(time_indices, transforms, smoothed, strict=True):
        out.append(Transform.from_translation(centre - slow) @ transform)
        result.anchors[t] = {
            "centroid": result.centroids.get(t),
            "smoothed": slow.tolist(),
            "translation": (centre - slow).tolist(),
        }
    return out


def leftover_motion_report(
    result: SeriesResult,
    time_indices: list[int],
    transforms: list[Transform],
    voxel_um: list[float],
) -> dict:
    """Return the frame-to-frame motion left by `transforms`, and by none, from the raw shifts.

    Pairs are each timepoint's raw shift from the run's previous timepoint (a job that
    failed has none: its pair is skipped and listed). The corrections are the transforms'
    translations; `approximate` when any also rotates or scales, which the derivation
    leaves out.
    """
    pairs = [
        (record["from"], t, record["shift"])
        for t, record in sorted(result.raw_shifts.items())
        if t in time_indices
    ]
    corrections = {t: m.translation for t, m in zip(time_indices, transforms, strict=True)}
    approximate = any(not np.allclose(m.linear, np.eye(3), atol=1e-6) for m in transforms)
    report = {
        "approximate": approximate,
        "transforms": leftover_motion(pairs, corrections, voxel_um),
        "unregistered": leftover_motion(pairs, None, voxel_um),
        "no_raw_shift": [t for t in time_indices[1:] if t not in result.raw_shifts],
    }
    for name in ("transforms", "unregistered"):
        r = report[name]
        if r["mean_um"] is not None:
            click.echo(
                f"leftover motion, {name}: mean {r['mean_um']:.3f} µm, max {r['max_um']:.3f} µm"
                f" over {len(r['pairs'])} pairs" + (" (approximate)" if approximate else "")
            )
    return report


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
        raise RunError(
            "reference 'previous' needs contiguous time_indices (each timepoint is chained "
            f"through the one before it); got {time_indices}"
        )
    chained, cumulative = [], Transform.identity()
    for transform in transforms:
        cumulative = cumulative @ transform
        chained.append(cumulative)
    return chained
