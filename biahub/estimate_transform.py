"""Estimate a transform series with the registration engine.

Maps a moving channel onto a reference per timepoint -- another channel (registration)
or the same channel at a fixed or previous timepoint (stabilization) -- and writes a
config that `register` / `stabilize` can apply. Two SLURM fan-out phases: one job per
timepoint to estimate, then one job per flagged timepoint to repair against the frozen
whole-run history. Every job writes a small JSON record, so an interrupted run resumes.
"""

from __future__ import annotations

import json

from collections.abc import Callable
from pathlib import Path
from typing import Literal

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
from biahub.registration.estimators import (
    AntsEstimator,
    ManualEstimator,
    NodeGraphEstimator,
    PCCEstimator,
    ScoreFn,
    TransformEstimator,
    beads_score_fn,
)
from biahub.registration.fallback import RepairResult, neighbour_consensus_config_candidates
from biahub.registration.legacy import forward_from_legacy_pull, legacy_pull_from_forward
from biahub.registration.metrics import (
    bead_alignment_metrics,
    gradient_correlation,
    normalized_mutual_information,
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
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)
from biahub.utils.cluster import estimate_resources, get_submitit_cluster
from biahub.utils.config import model_to_yaml, yaml_to_model

# "cross": register onto another channel; "first"/"previous": stabilize a channel against
# its own first / previous timepoint.
ReferenceKind = Literal["cross", "first", "previous"]

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


def _reference_policy(kind: ReferenceKind, ref) -> ReferencePolicy:
    if kind == "cross":
        return CrossChannel(ref)
    if kind == "first":
        return FixedFrame(0)
    if kind == "previous":
        return PreviousFrame()
    raise ValueError(f"unknown reference kind {kind!r}")


def _engine(
    settings: EstimateRegistrationSettings,
    shape_zyx: tuple[int, int, int] | None = None,
    mov_voxel_size: tuple[float, float, float] | None = None,
    ref_voxel_size: tuple[float, float, float] | None = None,
) -> tuple[TransformEstimator, ScoreFn, Transform]:
    """Estimator, score function and forward config seed for the settings' method."""
    affine_transform_settings = settings.affine_transform_settings
    config_seed = forward_from_legacy_pull(
        affine_transform_settings.approx_transform, affine_transform_settings.transform_type
    )
    if settings.estimation_method == "phase-cross-corr":
        estimator = PCCEstimator.from_settings(settings.phase_cross_corr_settings, shape_zyx)
        return estimator, correlation_score, config_seed

    if settings.estimation_method == "manual":
        manual = settings.manual_registration_settings
        estimator = ManualEstimator(
            source_channel_name=settings.source_channel_name,
            target_channel_name=settings.target_channel_name,
            source_channel_voxel_size=mov_voxel_size or (1.0, 1.0, 1.0),
            target_channel_voxel_size=ref_voxel_size or (1.0, 1.0, 1.0),
            similarity=affine_transform_settings.transform_type == "similarity",
            pre_affine_90degree_rotation=manual.affine_90degree_rotation,
            pre_affine_fliplr=manual.affine_fliplr,
        )
        return estimator, gradient_correlation, config_seed
    if settings.estimation_method == "beads":
        beads_match_settings = settings.beads_match_settings
        estimator = NodeGraphEstimator.from_beads_settings(
            beads_match_settings, affine_transform_settings
        )

        return estimator, beads_score_fn(beads_match_settings), config_seed

    if settings.estimation_method == "ants":
        ants_settings = settings.ants_registration_settings
        estimator = AntsEstimator.from_settings(
            ants_settings, affine_transform_settings, verbose=settings.verbose
        )

        if ants_settings.score_metric == "mutual_information":
            return estimator, normalized_mutual_information, config_seed

        def score_fn(transform: Transform, mov_t: np.ndarray, ref_t: np.ndarray) -> float:
            return correlation_score(
                transform, mov_t, ref_t, sobel_filter=ants_settings.sobel_filter
            )

        return estimator, score_fn, config_seed

    raise click.UsageError(f"unknown estimation_method '{settings.estimation_method}'")


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _estimate_timepoint_job(
    source_position_dirpath: Path,
    target_position_dirpath: Path,
    settings_path: Path,
    reference_kind: ReferenceKind,
    t: int,
    record_path: Path,
) -> dict:
    """One independent estimate, from the config seed, written as a JSON record."""
    settings = yaml_to_model(settings_path, EstimateRegistrationSettings)
    mov, mov_voxel_size = _open_series(source_position_dirpath, settings.source_channel_name)
    ref, ref_voxel_size = _open_series(target_position_dirpath, settings.target_channel_name)
    estimator, score_fn, config_seed = _engine(
        settings, tuple(mov.shape[-3:]), mov_voxel_size, ref_voxel_size
    )

    result = estimate_series(
        mov,
        _reference_policy(reference_kind, ref),
        estimator,
        FixedSeed(config_seed),
        score_fn,
        [t],
    )
    record = {
        "t": t,
        "matrix": result.transforms[t].to_list() if t in result.transforms else None,
        "score": _finite_or_none(result.scores.get(t)),
        "error": result.errors.get(t),
        "arm": getattr(estimator, "last_winner", None),
        "metrics": _bead_metrics(settings, result, t, mov, ref, reference_kind),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    return record


def _bead_metrics(settings, result, t, mov, ref, reference_kind) -> dict | None:
    """Continuous bead residuals next to the score, for beads methods with a transform."""
    if settings.estimation_method != "beads" or t not in result.transforms:
        return None
    ref_t = np.asarray(_reference_policy(reference_kind, ref).reference_for(mov, t))
    metrics = bead_alignment_metrics(
        result.transforms[t], np.asarray(mov[t]), ref_t, settings.beads_match_settings
    )
    return None if metrics is None else metrics.to_dict()


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
    reference_kind: ReferenceKind,
    t: int,
    time_indices: list[int],
    flagged: list[int],
    records_dir: Path,
    record_path: Path,
) -> dict:
    """Repair one flagged timepoint against the frozen whole-run history."""
    settings = yaml_to_model(settings_path, EstimateRegistrationSettings)
    mov, mov_voxel_size = _open_series(source_position_dirpath, settings.source_channel_name)
    ref, ref_voxel_size = _open_series(target_position_dirpath, settings.target_channel_name)
    estimator, score_fn, config_seed = _engine(
        settings, tuple(mov.shape[-3:]), mov_voxel_size, ref_voxel_size
    )

    series = _load_series(
        records_dir, time_indices, settings.affine_transform_settings.transform_type
    )
    series.flagged = list(flagged)
    outcome = repair_timepoint(
        t,
        mov,
        _reference_policy(reference_kind, ref),
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
    settings: EstimateRegistrationSettings,
    output_dir: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    resume: bool = False,
    reference_kind: ReferenceKind = "cross",
) -> tuple[SeriesResult, list[int], list[Transform]]:
    """Run both fan-out phases; return the series, its timepoints and one forward transform per timepoint.

    Writes the settings the jobs read (`estimate_transform_settings.yml`), the per-timepoint
    records (`timepoints/`, `repairs/`), `run_journal.json` and
    `estimate_transform_report.json` under `output_dir`. This is what `estimate-transform`,
    and the beads branches of `estimate-registration` / `estimate-stabilization`, run.
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
    affine_transform_settings = settings.affine_transform_settings
    if affine_transform_settings.compute_approx_transform:
        approx = get_aprox_transform(
            mov_shape=(Z, Y, X),
            ref_shape=tuple(ref_shape),
            pre_affine_90degree_rotation=-1,
            pre_affine_fliplr=False,
            verbose=settings.verbose,
            ref_voxel_size=ref_voxel_size,
            mov_voxel_size=mov_voxel_size,
        )
        affine_transform_settings.approx_transform = approx.to_list()
        click.echo(f"Computed approx transform:\n{approx.matrix}")
    _estimator, _score_fn, config_seed = _engine(
        settings, (Z, Y, X), mov_voxel_size, ref_voxel_size
    )
    transform_type = affine_transform_settings.transform_type
    if settings.estimation_method == "manual":
        # Interactive (napari); one timepoint, in this process.
        settings.time_indices = settings.manual_registration_settings.time_index
        cluster = "debug"
    settings_path = output_dir / ENGINE_SETTINGS_FILENAME
    model_to_yaml(settings, settings_path)
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

    if affine_transform_settings.use_prev_t_transform:
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
    estimate_records, job_failures = _run_jobs(
        executor,
        resolved_cluster,
        monitor,
        "estimate",
        [
            (
                t,
                _estimate_timepoint_job,
                (
                    source,
                    target,
                    settings_path,
                    reference_kind,
                    t,
                    timepoints_dir / f"{t}.json",
                ),
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

    flagged = flag_series(result)
    executor.update_parameters(slurm_time=60, slurm_job_name="estimate_transform_repair")
    to_repair = [t for t in flagged if not (resume and (repairs_dir / f"{t}.json").exists())]
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
                    reference_kind,
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
        record = repair_records.get(t)
        if record is None:
            record_path = repairs_dir / f"{t}.json"
            if not record_path.exists():
                continue
            record = json.loads(record_path.read_text())
        outcome = _load_repair(
            record, transform_type, fallback=result.transforms.get(t, config_seed)
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

    result.journal.save(output_dir / "run_journal.json")
    (output_dir / "estimate_transform_report.json").write_text(
        json.dumps(_report(result, time_indices), indent=2)
    )
    return (
        result,
        time_indices,
        _one_transform_per_timepoint(result, time_indices, config_seed),
    )


def _reference_kind(
    source: Path, target: Path, settings: EstimateRegistrationSettings
) -> ReferenceKind:
    """Pick the reference: the same position and channel on both sides is stabilization."""
    if (
        Path(source) == Path(target)
        and settings.source_channel_name == settings.target_channel_name
    ):
        return settings.affine_transform_settings.t_reference
    return "cross"


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
    takes; `estimation_method` beads or ants). When source and target are the same
    position and channel, this is stabilization: each timepoint is registered onto the
    channel's own first or previous timepoint per `affine_transform_settings.t_reference`.
    Writes a `RegistrationSettings` (single timepoint) or `StabilizationSettings` (series)
    YAML next to the engine's records; see `estimate_transform_series`.
    """
    output_filepath = Path(output_filepath)
    output_dir = output_filepath.parent
    settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    source, target = Path(source_position_dirpaths[0]), Path(target_position_dirpaths[0])
    with open_ome_zarr(target, mode="r") as position:
        voxel_size = list(position.scale)

    _result, time_indices, transforms = estimate_transform_series(
        source,
        target,
        settings,
        output_dir,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        resume=resume,
        reference_kind=_reference_kind(source, target, settings),
    )

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
            stabilization_method=settings.estimation_method,
            stabilization_channels=sorted(
                {settings.source_channel_name, settings.target_channel_name}
            ),
            affine_transform_zyx_list=pull_matrices,
            time_indices=settings.time_indices,
            output_voxel_size=voxel_size,
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
    """Estimate a transform series from source onto target with the registration engine.

    Takes the same YAML as `estimate-registration` (beads or ants) and writes a config
    for `register` / `stabilize`, plus a run journal and a per-timepoint report. One
    SLURM job per timepoint, then one per flagged timepoint for repair. Passing the same
    position and channel as source and target stabilizes that channel against its own
    first or previous timepoint (`affine_transform_settings.t_reference`).

    \b
    Registration (source channel onto target channel):
    >>> biahub estimate-transform -s source.zarr/0/0/0 -t target.zarr/0/0/0 \\
        -c estimate-registration-beads.yml -o ./registration_settings.yml

    \b
    Stabilization (a channel onto itself over time):
    >>> biahub estimate-transform -s data.zarr/0/0/0 -t data.zarr/0/0/0 \\
        -c estimate-stabilization-beads.yml -o ./stabilization_settings.yml

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
