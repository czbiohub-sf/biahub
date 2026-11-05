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
import uuid

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from iohub import open_ome_zarr

from biahub.registration.engine.series import (
    SeriesResult,
    TimepointRecord,
)
from biahub.registration.utils import RunError
from biahub.settings import (
    EstimateTransformSettings,
)
from biahub.utils.config import yaml_to_model

ENGINE_SETTINGS_FILENAME = "estimate_transform_settings.yml"


# `transform.initial: {from: file}`: the file's transform for each timepoint of the
# run's position, read by every job.
INITIAL_FILENAME = "initial_transforms.json"


RUN_MANIFEST_FILENAME = "run_manifest.json"
# At the top of a run folder: its positions and config (see RunFolder).
RUN_FILENAME = "run.json"


def _open_series(position_dirpath: Path, channel_name: str):
    """Open one channel as a (T, Z, Y, X) dask series, with its ZYX voxel size."""
    with open_ome_zarr(position_dirpath, mode="r") as position:
        series = position.data.dask_array()[:, position.channel_names.index(channel_name)]
        return series, tuple(position.scale[-3:])


def config_fingerprint(settings: EstimateTransformSettings) -> dict:
    """Return what identifies a run's config: its settings, and the initial file's content.

    With `transform.initial` from a file, a changed file is a different run. The one
    fingerprint behind both checks: `run.json` (a `--step` must use the config `--init`
    planned) and the resume manifest (records may be reused).
    """
    fingerprint = {
        "settings_sha256": hashlib.sha256(settings.model_dump_json().encode()).hexdigest()
    }
    initial_file = settings.transform.initial_file
    if initial_file is not None:
        content = Path(initial_file).read_bytes()
        fingerprint["initial_file_sha256"] = hashlib.sha256(content).hexdigest()
    return fingerprint


def _run_fingerprint(settings: EstimateTransformSettings, source: Path, target: Path) -> dict:
    """Return what a resumed run must share with the run it resumes: config and inputs."""

    def resolved(path):
        return None if path is None else str(Path(path).resolve())

    return {
        **config_fingerprint(settings),
        "moving": resolved(source),
        "reference": resolved(target),
    }


class RunFolder:
    """A run's folder (`<output>/transforms/` for one estimate-transform call): its layout.

    `run.json` records the run's positions and config fingerprint (written by `--init`).
    With one position its work files (settings, plan, records) sit in the folder itself;
    with several, each position's under `positions/<row>/<col>/<fov>/`.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def run_file(self) -> Path:
        return self.path / RUN_FILENAME

    def start(self, positions: list[str], config: dict) -> None:
        """Record the run's positions and config fingerprint."""
        self.path.mkdir(parents=True, exist_ok=True)
        self.run_file.write_text(
            json.dumps({"positions": positions, "config": config}, indent=2)
        )

    def _run(self) -> dict | None:
        return json.loads(self.run_file.read_text()) if self.run_file.exists() else None

    def positions(self) -> list[str] | None:
        """Return the run's positions, or None when no `run.json` was written."""
        run = self._run()
        return None if run is None else run["positions"]

    def config(self) -> dict | None:
        """Return the config fingerprint `--init` recorded (None: none, or an older run)."""
        run = self._run()
        return None if run is None else run.get("config")

    def work_dir(self, key: str | None, positions: list[str] | None = None) -> Path:
        """One position's work folder: the run folder itself, or `positions/<key>/`."""
        positions = self.positions() if positions is None else positions
        if key is None or positions is None or len(positions) == 1:
            return self.path
        return self.path / "positions" / key

    def work_dirs(self) -> list[tuple[str | None, Path]]:
        """Every position's (key, work folder); a single position's key is None.

        Without a `run.json` (an older run), the folder itself and any `positions/*/*/*/`
        found on disk.
        """
        positions = self.positions()
        if positions is None:
            found = sorted(d for d in (self.path / "positions").glob("*/*/*") if d.is_dir())
            return [(None, self.path)] + [
                ("/".join(d.relative_to(self.path / "positions").parts), d) for d in found
            ]
        if len(positions) == 1:
            return [(None, self.path)]
        return [(key, self.work_dir(key, positions)) for key in positions]


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
            raise RunError(
                f"--resume: {', '.join(changed)} changed since the run in {output_dir}; "
                "rerun without --resume"
            )
        return
    if resume and any(
        (output_dir / name).is_dir() and any((output_dir / name).iterdir())
        for name in ("timepoints", "repairs", "sweeps")
    ):
        raise RunError(
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


def _initial_transforms_from_file(
    initial_transform_file, key: str, time_indices: list[int]
) -> dict[str, list]:
    """Return the `from: file` forward transform at each timepoint of position `key`."""
    from biahub.settings import load_transform_settings

    transforms = load_transform_settings(initial_transform_file)
    if transforms.per_position and key not in transforms.positions:
        raise RunError(
            f"transform.initial.file {initial_transform_file} has no transforms for position "
            f"{key} (it has {sorted(transforms.positions)})"
        )
    return {str(t): transforms.transform_for(t, key).to_list() for t in time_indices}


def _run_settings(output_dir: Path) -> EstimateTransformSettings:
    """Return the settings `init_run` resolved for this run."""
    path = Path(output_dir) / ENGINE_SETTINGS_FILENAME
    if not path.exists():
        raise RunError(f"no run in {output_dir}: run with --init first")
    return yaml_to_model(path, EstimateTransformSettings)


def _run_plan(output_dir: Path) -> dict:
    path = Path(output_dir) / RUN_PLAN_FILENAME
    if not path.exists():
        raise RunError(f"no run in {output_dir}: run with --init first")
    return json.loads(path.read_text())


def _flags(output_dir: Path) -> dict:
    path = Path(output_dir) / FLAGS_FILENAME
    if not path.exists():
        raise RunError(f"no flags in {output_dir}: run --step flag first")
    return json.loads(path.read_text())


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
    """The run's fallback attempts (repair, polish, sweep), written by finalize.

    Built from the pass records when a run is finalized and saved as `run_journal.json`;
    `current_run_id` tags every attempt of this finalize.
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

    def to_dict(self) -> dict:
        return {"attempts": [asdict(a) for a in self.attempts]}

    def save(self, path: Path) -> None:
        _write_json(path, self.to_dict(), indent=2)
