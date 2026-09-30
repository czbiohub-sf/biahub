"""Estimate a transform series with the registration engine.

Maps a moving channel onto a reference per timepoint -- another channel (registration)
or the same channel at a fixed or previous timepoint (stabilization) -- and writes a
`TransformSettings` config that the apply steps consume. Two SLURM fan-out phases: one
job per timepoint to estimate, then one job per flagged timepoint to repair against the
frozen whole-run history. Every job writes a small JSON record, so an interrupted run
resumes.
"""

from __future__ import annotations

import json

from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
import submitit

from iohub import open_ome_zarr

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    monitor,
    output_filepath,
    resume,
    sbatch_filepath,
    sbatch_to_submitit,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.core.transform import Transform
from biahub.registration.ants import correlation_score
from biahub.registration.beads import score_transform
from biahub.registration.estimators import (
    AntsEstimator,
    ManualEstimator,
    NodeGraphEstimator,
    PCCEstimator,
    ScoreFn,
    TransformEstimator,
)
from biahub.registration.fallback import RepairResult, neighbour_consensus_config_candidates
from biahub.registration.legacy import forward_from_legacy_pull
from biahub.registration.metrics import (
    bead_alignment_metrics,
    gradient_correlation,
    normalized_mutual_information,
    residual_score,
)
from biahub.registration.orchestrator import (
    SeriesResult,
    estimate_series,
    flag_series,
    repair_timepoint,
)
from biahub.registration.reference_policy import (
    CrossChannel,
    FixedFrame,
    PreviousFrame,
    ReferencePolicy,
)
from biahub.registration.seed_policy import FixedSeed
from biahub.registration.utils import get_aprox_transform
from biahub.settings import (
    AffineTransformSettings,
    EstimateTransformSettings,
    TransformSettings,
    load_estimate_transform_settings,
)
from biahub.utils.cluster import estimate_resources, get_submitit_cluster
from biahub.utils.config import model_to_yaml, yaml_to_model

ENGINE_SETTINGS_FILENAME = "estimate_transform_settings.yml"


def _resolve_time_indices(time_indices, n_t: int) -> list[int]:
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
    return forward_from_legacy_pull(fit.seed, fit.type)


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


def _engine(
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
        estimator: TransformEstimator = NodeGraphEstimator.from_beads_settings(
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
    estimator, score_fn, seed = _engine(
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
    estimator, score_fn, seed = _engine(
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
    _estimator, _score_fn, seed = _engine(settings, (Z, Y, X), mov_voxel_size, ref_voxel_size)
    if settings.method == "manual":
        # Interactive (napari); one timepoint, in this process.
        settings.time_indices = settings.manual.time_index
        cluster = "debug"
    settings_path = output_dir / ENGINE_SETTINGS_FILENAME
    model_to_yaml(settings, settings_path)
    time_indices = _resolve_time_indices(settings.time_indices, T)
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


def estimate_transform(
    source_position_dirpaths: list[Path],
    target_position_dirpaths: list[Path],
    config_filepath: Path,
    output_filepath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    resume: bool = False,
) -> None:
    """Estimate one transform per timepoint mapping the source channel onto its reference.

    Reads an `EstimateTransformSettings` YAML (legacy `estimate-registration` /
    `estimate-stabilization` configs are accepted and converted). Writes a
    `TransformSettings` YAML -- forward matrices, one per timepoint -- next to the engine's
    records; see `estimate_transform_series`.
    """
    output_filepath = Path(output_filepath)
    output_dir = output_filepath.parent
    settings = load_estimate_transform_settings(config_filepath)
    source = Path(source_position_dirpaths[0])
    target = Path(target_position_dirpaths[0]) if settings.reference == "cross" else source
    with open_ome_zarr(target, mode="r") as position:
        voxel_size = [float(v) for v in position.scale]

    _result, _time_indices, transforms = estimate_transform_series(
        source,
        target,
        settings,
        output_dir,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        resume=resume,
    )

    model = TransformSettings(
        direction="forward",
        matrices=[transform.to_list() for transform in transforms],
        time_indices=settings.time_indices,
        source_channels=[settings.source.channel],
        target_channel=settings.target.channel if settings.target is not None else None,
        method=settings.method,
        voxel_size=voxel_size,
    )
    model_to_yaml(model, output_filepath)
    click.echo(f"Transform settings saved to {output_filepath.resolve()}")


@click.command("estimate-transform")
@source_position_dirpaths()
@target_position_dirpaths()
@config_filepath()
@output_filepath()
@sbatch_filepath()
@cluster()
@monitor()
@resume()
def estimate_transform_cli(
    source_position_dirpaths: list[Path],
    target_position_dirpaths: list[Path],
    config_filepath: Path,
    output_filepath: Path,
    sbatch_filepath: str | None,
    cluster: str,
    monitor: bool,
    resume: bool,
) -> None:
    """Estimate a transform series from source onto its reference with the registration engine.

    Takes an `EstimateTransformSettings` YAML (or a legacy `estimate-registration` /
    `estimate-stabilization` one) and writes a `TransformSettings` config for `register` /
    `stabilize`, plus a run journal and a per-timepoint report. One SLURM job per
    timepoint, then one per flagged timepoint for repair. With `reference: first` or
    `previous` the source channel is stabilized against itself and `-t` is ignored.

    \b
    Registration (source channel onto target channel):
    >>> biahub estimate-transform -s source.zarr/0/0/0 -t target.zarr/0/0/0 \\
        -c estimate-transform-beads.yml -o ./transforms.yml

    \b
    Stabilization (a channel onto its own first / previous timepoint):
    >>> biahub estimate-transform -s data.zarr/0/0/0 -t data.zarr/0/0/0 \\
        -c estimate-transform-stabilize.yml -o ./transforms.yml

    \b
    Retry an interrupted run, keeping finished timepoints:
    >>> biahub estimate-transform --resume -s ... -t ... -c ... -o ./transforms.yml
    """  # noqa: D301
    estimate_transform(
        source_position_dirpaths=source_position_dirpaths,
        target_position_dirpaths=target_position_dirpaths,
        config_filepath=config_filepath,
        output_filepath=output_filepath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        resume=resume,
    )


if __name__ == "__main__":
    estimate_transform_cli()
