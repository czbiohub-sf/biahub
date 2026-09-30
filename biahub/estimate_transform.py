"""Estimate a transform series with the registration engine.

Maps a moving channel onto a reference channel per timepoint and writes a config that
`register` / `stabilize` can apply. Two SLURM fan-out phases: one job per timepoint to
estimate, then one job per flagged timepoint to repair against the frozen whole-run
history. Every job writes a small JSON record, so an interrupted run resumes.
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
    NodeGraphEstimator,
    ScoreFn,
    TransformEstimator,
)
from biahub.registration.fallback import RepairResult, neighbour_consensus_config_candidates
from biahub.registration.legacy import forward_from_legacy_pull, legacy_pull_from_forward
from biahub.registration.orchestrator import (
    SeriesResult,
    estimate_series,
    flag_series,
    repair_timepoint,
)
from biahub.registration.reference_policy import CrossChannel
from biahub.registration.seed_policy import FixedSeed
from biahub.settings import (
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)
from biahub.utils.cluster import estimate_resources, get_submitit_cluster
from biahub.utils.config import model_to_yaml, yaml_to_model


def _resolve_time_indices(time_indices, n_t: int) -> list[int]:
    if time_indices == "all":
        return list(range(n_t))
    if isinstance(time_indices, int):
        return [time_indices]
    return list(time_indices)


def _open_series(position_dirpath: Path, channel_name: str):
    with open_ome_zarr(position_dirpath, mode="r") as position:
        return position.data.dask_array()[:, position.channel_names.index(channel_name)]


def _engine(
    settings: EstimateRegistrationSettings,
) -> tuple[TransformEstimator, ScoreFn, Transform]:
    """Estimator, score function and forward config seed for the settings' method."""
    affine_transform_settings = settings.affine_transform_settings
    config_seed = forward_from_legacy_pull(
        affine_transform_settings.approx_transform, affine_transform_settings.transform_type
    )
    if settings.estimation_method == "beads":
        beads_match_settings = settings.beads_match_settings
        estimator = NodeGraphEstimator.from_beads_settings(
            beads_match_settings, affine_transform_settings
        )

        def score_fn(transform: Transform, mov_t: np.ndarray, ref_t: np.ndarray) -> float:
            return score_transform(transform, mov_t, ref_t, beads_match_settings)

        return estimator, score_fn, config_seed

    if settings.estimation_method == "ants":
        ants_settings = settings.ants_registration_settings
        estimator = AntsEstimator.from_settings(
            ants_settings, affine_transform_settings, verbose=settings.verbose
        )

        def score_fn(transform: Transform, mov_t: np.ndarray, ref_t: np.ndarray) -> float:
            return correlation_score(
                transform, mov_t, ref_t, sobel_filter=ants_settings.sobel_filter
            )

        return estimator, score_fn, config_seed

    raise click.UsageError(
        f"estimate-transform supports estimation_method 'beads' and 'ants'; got "
        f"'{settings.estimation_method}'. Use estimate-registration for the others."
    )


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _estimate_timepoint_job(
    source_position_dirpath: Path,
    target_position_dirpath: Path,
    config_filepath: Path,
    t: int,
    record_path: Path,
) -> dict:
    """One independent estimate, from the config seed, written as a JSON record."""
    settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    mov = _open_series(source_position_dirpath, settings.source_channel_name)
    ref = _open_series(target_position_dirpath, settings.target_channel_name)
    estimator, score_fn, config_seed = _engine(settings)

    result = estimate_series(
        mov, CrossChannel(ref), estimator, FixedSeed(config_seed), score_fn, [t]
    )
    record = {
        "t": t,
        "matrix": result.transforms[t].to_list() if t in result.transforms else None,
        "score": _finite_or_none(result.scores.get(t)),
        "error": result.errors.get(t),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    return record


def _load_series(
    records_dir: Path, time_indices: list[int], transform_type: str
) -> SeriesResult:
    """Rebuild the series from the per-timepoint records (a missing record is an error)."""
    result = SeriesResult()
    for t in time_indices:
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
    config_filepath: Path,
    t: int,
    time_indices: list[int],
    flagged: list[int],
    records_dir: Path,
    record_path: Path,
) -> dict:
    """Repair one flagged timepoint against the frozen whole-run history."""
    settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    mov = _open_series(source_position_dirpath, settings.source_channel_name)
    ref = _open_series(target_position_dirpath, settings.target_channel_name)
    estimator, score_fn, config_seed = _engine(settings)

    series = _load_series(
        records_dir, time_indices, settings.affine_transform_settings.transform_type
    )
    series.flagged = list(flagged)
    outcome = repair_timepoint(
        t,
        mov,
        CrossChannel(ref),
        estimator,
        score_fn,
        series,
        neighbour_consensus_config_candidates(config_seed),
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


def _load_repair(record_path: Path, transform_type: str, fallback: Transform) -> RepairResult:
    record = json.loads(record_path.read_text())
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
) -> dict[int, str]:
    """Submit one job per (t, fn, args); return {t: error} for jobs that raised."""
    if not submissions:
        return {}
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for _t, fn, args in submissions:
            jobs.append(executor.submit(fn, *args))
    log_path = Path(executor.folder) / f"{label}_job_ids.log"
    log_path.write_text("\n".join(str(job.job_id) for job in jobs))
    click.echo(f"{label}: submitted {len(jobs)} job(s) on cluster='{resolved_cluster}'")

    if monitor_flag and resolved_cluster == "slurm":
        monitor_jobs(jobs, [Path(f"t={t}") for t, _fn, _args in submissions])

    failures: dict[int, str] = {}
    for (t, _fn, _args), job in zip(submissions, jobs, strict=True):
        try:
            job.result()
        except Exception as e:  # noqa: BLE001 -- one job's infrastructure failure must not abort the run
            failures[t] = f"job failed: {type(e).__name__}: {str(e).splitlines()[-1][:200]}"
            click.echo(f"{label} t={t}: {failures[t]}")
    return failures


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
    """Estimate one transform per timepoint mapping the source channel onto the target.

    Reads an `EstimateRegistrationSettings` YAML (the same file `estimate-registration`
    takes; `estimation_method: beads` only). Phase 1 estimates every timepoint
    independently from the config seed, one job each. Phase 2 flags timepoints against
    the run's own score distribution and repairs each flagged one in its own job, seeded
    from non-flagged neighbours, the whole-run consensus geometry and the config seed.
    Writes a `RegistrationSettings` (single timepoint) or `StabilizationSettings` (series)
    YAML, plus `run_journal.json` and `estimate_transform_report.json`, next to the
    per-timepoint records in `timepoints/` and `repairs/`. With `resume`, timepoints
    that already have a record are not re-estimated.
    """
    output_filepath = Path(output_filepath)
    output_dir = output_filepath.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    timepoints_dir = output_dir / "timepoints"
    repairs_dir = output_dir / "repairs"
    slurm_out_path = output_dir / "slurm_output"
    slurm_out_path.mkdir(exist_ok=True)

    settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    _estimator, _score_fn, config_seed = _engine(settings)  # validates the method up front
    transform_type = settings.affine_transform_settings.transform_type
    source, target = Path(source_position_dirpaths[0]), Path(target_position_dirpaths[0])

    with open_ome_zarr(source, mode="r") as position:
        T, _C, Z, Y, X = position.data.shape
    with open_ome_zarr(target, mode="r") as position:
        voxel_size = list(position.scale)
    time_indices = _resolve_time_indices(settings.time_indices, T)

    # Bead matching is single-threaded; ANTs' optimizer uses ITK threads.
    _, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(1, 2, Z, Y, X),
        ram_multiplier=5,
        max_num_cpus=8 if settings.estimation_method == "ants" else 4,
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

    if settings.affine_transform_settings.use_prev_t_transform:
        click.echo(
            "use_prev_t_transform is set, but timepoints are estimated independently under "
            "fan-out; neighbour information enters through the repair phase instead."
        )

    to_estimate = [
        t for t in time_indices if not (resume and (timepoints_dir / f"{t}.json").exists())
    ]
    if resume:
        click.echo(
            f"resume: {len(time_indices) - len(to_estimate)} timepoint(s) already estimated"
        )
    job_failures = _run_jobs(
        executor,
        resolved_cluster,
        monitor,
        "estimate",
        [
            (
                t,
                _estimate_timepoint_job,
                (source, target, config_filepath, t, timepoints_dir / f"{t}.json"),
            )
            for t in to_estimate
        ],
    )

    result = _load_series(timepoints_dir, time_indices, transform_type)
    for t, error in job_failures.items():
        result.errors[t] = error
    for t in time_indices:
        line = f"t={t}: score={result.scores[t]:.4f}"
        if t in result.errors:
            line += f"  estimate failed: {result.errors[t]}"
        click.echo(line)

    flagged = flag_series(result)
    executor.update_parameters(slurm_time=60, slurm_job_name="estimate_transform_repair")
    to_repair = [t for t in flagged if not (resume and (repairs_dir / f"{t}.json").exists())]
    _run_jobs(
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
                    config_filepath,
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
    for t in flagged:
        record_path = repairs_dir / f"{t}.json"
        if not record_path.exists():
            continue
        outcome = _load_repair(
            record_path, transform_type, fallback=result.transforms.get(t, config_seed)
        )
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

    transforms = _one_transform_per_timepoint(result, time_indices, config_seed)
    pull_matrices = [legacy_pull_from_forward(transform) for transform in transforms]
    if len(pull_matrices) == 1:
        model = RegistrationSettings(
            source_channel_names=[settings.source_channel_name],
            target_channel_name=settings.target_channel_name,
            affine_transform_zyx=pull_matrices[0],
        )
    else:
        model = StabilizationSettings(
            stabilization_estimation_channel=settings.target_channel_name,
            stabilization_type="affine",
            stabilization_method="beads",
            stabilization_channels=[
                settings.source_channel_name,
                settings.target_channel_name,
            ],
            affine_transform_zyx_list=pull_matrices,
            time_indices=settings.time_indices,
            output_voxel_size=voxel_size,
        )
    model_to_yaml(model, output_filepath)
    result.journal.save(output_dir / "run_journal.json")
    (output_dir / "estimate_transform_report.json").write_text(
        json.dumps(_report(result, time_indices), indent=2)
    )
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
    """Estimate a transform series from source onto target with the registration engine.

    Takes the same YAML as `estimate-registration` (beads method only for now) and writes
    a config for `register` / `stabilize`, plus a run journal and a per-timepoint report.
    One SLURM job per timepoint, then one per flagged timepoint for repair.

    \b
    >>> biahub estimate-transform \\
        -s source.zarr/0/0/0 \\
        -t target.zarr/0/0/0 \\
        -c estimate-registration-beads.yml \\
        -o ./registration_settings.yml

    \b
    Retry an interrupted run, keeping finished timepoints:
    >>> biahub estimate-transform --resume -s ... -t ... -c ... -o ./registration_settings.yml
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
