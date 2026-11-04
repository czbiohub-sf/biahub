"""Everything the engine reads and writes: the stores it estimates on, and the run folder.

The run folder (`<output>/transforms/`, or `positions/<key>/` per position) holds the
settings the run started with, its plan, the per-timepoint records (timepoints/, repairs/,
sweeps/), the flags, the initial transforms of a `from: file` run, and the manifest that
decides whether `--resume` may reuse it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

from collections.abc import Iterable
from pathlib import Path

import click

from iohub import open_ome_zarr

from biahub.registration.engine.series import (
    SeriesResult,
    TimepointRecord,
)
from biahub.settings import (
    EstimateTransformSettings,
)
from biahub.utils.config import yaml_to_model

ENGINE_SETTINGS_FILENAME = "estimate_transform_settings.yml"


# A seed file's transform for each timepoint of the run's position, read by every job.
INITIAL_FILENAME = "initial_transforms.json"


RUN_MANIFEST_FILENAME = "run_manifest.json"


def _open_series(position_dirpath: Path, channel_name: str):
    """Open one channel as a (T, Z, Y, X) dask series, with its ZYX voxel size."""
    with open_ome_zarr(position_dirpath, mode="r") as position:
        series = position.data.dask_array()[:, position.channel_names.index(channel_name)]
        return series, tuple(position.scale[-3:])


def _run_fingerprint(settings: EstimateTransformSettings, source: Path, target: Path) -> dict:
    """Return what a resumed run must share with the run it resumes."""
    digest = hashlib.sha256(settings.model_dump_json().encode()).hexdigest()

    def resolved(path):
        return None if path is None else str(Path(path).resolve())

    seed_file = settings.transform.initial_file
    return {
        "settings_sha256": digest,
        "moving": resolved(source),
        "reference": resolved(target),
        # the seed file's content, not only its name: a resumed run must start from the same
        **(
            {"seed_file_sha256": hashlib.sha256(Path(seed_file).read_bytes()).hexdigest()}
            if seed_file is not None
            else {}
        ),
    }


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
    match the manifest of the run being resumed; records with no manifest to check them
    against are refused, and with no records at all the run simply starts fresh.
    """
    manifest_path = output_dir / RUN_MANIFEST_FILENAME
    fingerprint = _run_fingerprint(settings, source, target)
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        changed = sorted(k for k in fingerprint if previous.get(k) != fingerprint[k])
        if changed:
            raise click.UsageError(
                f"--resume: {', '.join(changed)} changed since the run in {output_dir}; "
                "rerun without --resume"
            )
        return
    if resume and any(
        (output_dir / name).is_dir() and any((output_dir / name).iterdir())
        for name in ("timepoints", "repairs", "sweeps")
    ):
        raise click.UsageError(
            f"--resume: no {RUN_MANIFEST_FILENAME} in {output_dir}, so its records cannot be "
            "checked against these settings and inputs; rerun without --resume"
        )
    for name in ("timepoints", "repairs", "sweeps"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    _write_json(manifest_path, fingerprint, indent=2)


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
            record = _read_record(path) if t in resumed else None
            if record is None:  # missing, unreadable, or not this run's to reuse
                result.scores[t] = float("nan")
                result.errors[t] = "no record: job did not finish"
                continue
        TimepointRecord.from_dict(record).fold_into(result, transform_type)
    return result


def _write_json(path: Path, obj, indent: int | None = None) -> None:
    """Write JSON atomically: a job killed mid-write leaves the old file or none, never half."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=indent))
    os.replace(tmp, path)


def _read_record(path: Path) -> dict | None:
    """Return a record, or None if it is missing or unreadable (e.g. half-written before)."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# How a job that raised is recorded, so a caller can tell it from an estimate that failed.
JOB_FAILED_PREFIX = "job failed"


def _job_failure(error: Exception) -> str:
    """Return how a job that raised is recorded: 'job failed: <type>: <last line>'."""
    message = str(error).splitlines()[-1][:200] if str(error) else ""
    return f"{JOB_FAILED_PREFIX}: {type(error).__name__}: {message}"


def _failed_record(t: int, error: Exception) -> dict:
    """Return an estimate record for a timepoint whose job raised (as the driver records it).

    `failed` marks it as not finished, so a retry with resume does it again.
    """
    return TimepointRecord(t=t, error=_job_failure(error), failed=True).to_dict()


def _finished(record_path: Path) -> bool:
    """Whether a record exists for a timepoint that did not fail (what resume skips)."""
    record = _read_record(record_path)
    return record is not None and not record.get("failed")


RUN_PLAN_FILENAME = "run_plan.json"


FLAGS_FILENAME = "flags.json"


def _record_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    """Return the run's per-timepoint record dirs: estimates, repairs, sweeps."""
    return output_dir / "timepoints", output_dir / "repairs", output_dir / "sweeps"


def _seeds_from_file(seed_file, key: str, time_indices: list[int]) -> dict[str, list]:
    """Return a seed file's forward transform at each timepoint of position `key`."""
    from biahub.settings import load_transform_settings

    transforms = load_transform_settings(seed_file)
    if transforms.per_position and key not in transforms.positions:
        raise click.UsageError(
            f"transform.initial.file {seed_file} has no transforms for position {key} (it has "
            f"{sorted(transforms.positions)})"
        )
    return {str(t): transforms.transform_for(t, key).to_list() for t in time_indices}


def _run_settings(output_dir: Path) -> EstimateTransformSettings:
    """Return the settings `init_run` resolved for this run."""
    path = Path(output_dir) / ENGINE_SETTINGS_FILENAME
    if not path.exists():
        raise click.UsageError(f"no run in {output_dir}: run with --init first")
    return yaml_to_model(path, EstimateTransformSettings)


def _run_plan(output_dir: Path) -> dict:
    path = Path(output_dir) / RUN_PLAN_FILENAME
    if not path.exists():
        raise click.UsageError(f"no run in {output_dir}: run with --init first")
    return json.loads(path.read_text())


def _flags(output_dir: Path) -> dict:
    path = Path(output_dir) / FLAGS_FILENAME
    if not path.exists():
        raise click.UsageError(f"no flags in {output_dir}: run --step flag first")
    return json.loads(path.read_text())
