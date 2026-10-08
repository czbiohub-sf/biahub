"""Estimate a transform series with the registration engine.

Maps a moving channel onto a reference per timepoint -- another channel (registration)
or the same channel at its first or previous timepoint (stabilization) -- and writes the
`TransformSettings` file that `apply-transform` consumes. The phases: one job per
timepoint to estimate (or one sequential job), flagging, one job per flagged timepoint to
repair (and sweep) against the frozen whole-run history, and writing the file. Every job
writes a small JSON record, so an interrupted run resumes. Run in one call, the phases go
through submitit; `--init` and `--step` run them one at a time, in process, so a
Nextflow workflow can own the fan-out like it does for every other step.
"""

from __future__ import annotations

import hashlib
import json

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    monitor,
    moving_position_dirpaths,
    output_filepath,
    pair_reference_positions,
    position_key,
    reference_position_dirpaths,
    resume,
    sbatch_filepath,
)
from biahub.registration.engine import (
    SeriesResult,
    estimate_transform_series,
    finalize_run,
    flag_run,
    init_run,
    run_timepoint_jobs,
)
from biahub.settings import (
    TransformEntry,
    TransformSettings,
    load_estimate_transform_settings,
)
from biahub.utils.cluster import echo_resources
from biahub.utils.config import model_to_yaml


def transform_entries(
    result: SeriesResult, time_indices: list[int], transforms, hard_fail: float
) -> list[TransformEntry]:
    """One entry per estimated timepoint with its score, repair provenance and status.

    A timepoint is `unreliable` when the pipeline found no good transform for it: none
    at all (the entry is a stand-in -- the input seed -- and `note` says why), no score,
    or flagged and still below `hard_fail` after repair and sweep. Otherwise `accepted`.

    A single timepoint (manual, `time_indices: 0`) is the series' transform: one entry
    without `t`, which `apply-transform` applies to every timepoint.
    """
    entries = []
    for t, transform in zip(time_indices, transforms, strict=True):
        score = result.scores.get(t)
        finite = score is not None and np.isfinite(score)
        filled_from = result.filled_from.get(t)
        unreliable = (
            filled_from is not None
            or not finite
            or (t in result.flagged and score < hard_fail)
        )
        entries.append(
            TransformEntry(
                t=None if len(time_indices) == 1 else t,
                estimated_at=t if len(time_indices) == 1 else None,
                matrix=transform.to_list(),
                score=float(score) if finite else None,
                repaired_from=result.provenance.get(t),
                status="unreliable" if unreliable else "accepted",
                filled_from=filled_from,
                # from --initial-transforms, unless a repair / sweep replaced it since
                seeded_from=None if t in result.provenance else result.seeded_from.get(t),
                # why there is no transform of its own (estimate error, job cancelled, ...)
                note=result.errors.get(t) if filled_from is not None else None,
            )
        )
    return entries


# Positions estimated at once; each driver fans its own timepoints out to SLURM.
MAX_CONCURRENT_POSITIONS = 8

# At the top of a run folder: the positions and the config `--init` started the run with.
RUN_FILENAME = "run.json"

STEPS = ("estimate", "flag", "repair", "sweep", "finalize")

# A --step whose jobs raised exits with this after recording them (registration.nf
# retries it, then lets the run finish with those timepoints as stand-ins).
JOBS_FAILED_EXIT_CODE = 3


class _Run:
    """One estimate's positions, references and per-position run folders."""

    def __init__(
        self,
        moving_position_dirpaths: list[Path],
        config_filepath: Path,
        output_filepath: Path,
        reference_position_dirpaths: list[Path] | None,
    ):
        self.output_filepath = Path(output_filepath)
        # The run's records live in a folder named after the output (reg/transforms.yml ->
        # reg/transforms/), so estimates written to one folder never share or clear them.
        self.output_dir = self.output_filepath.with_suffix("")
        self.settings = load_estimate_transform_settings(config_filepath)
        self.config_sha256 = hashlib.sha256(
            self.settings.model_dump_json().encode()
        ).hexdigest()
        self.movings = [Path(p) for p in moving_position_dirpaths]
        self.keys = [position_key(p) for p in self.movings]
        if self.settings.reference.frame == "cross":
            if not reference_position_dirpaths:
                raise click.UsageError(
                    "reference frame 'cross' needs the reference positions (-r)"
                )
            self.reference_for = pair_reference_positions(
                self.keys, reference_position_dirpaths
            )
        else:
            self.reference_for = dict(zip(self.keys, self.movings, strict=True))

    def start(self) -> None:
        """Record the run's positions and config (what later steps check against)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / RUN_FILENAME).write_text(
            json.dumps({"positions": self.keys, "config_sha256": self.config_sha256}, indent=2)
        )
        self.run_positions = list(self.keys)

    def load(self) -> None:
        """Read what `--init` recorded; refuse positions or a config it did not start."""
        path = self.output_dir / RUN_FILENAME
        if not path.exists():
            raise click.UsageError(f"no run for {self.output_filepath}: run with --init first")
        run = json.loads(path.read_text())
        if run["config_sha256"] != self.config_sha256:
            raise click.UsageError(
                f"the config changed since --init started {self.output_dir}; run --init again"
            )
        unknown = sorted(set(self.keys) - set(run["positions"]))
        if unknown:
            raise click.UsageError(
                f"positions {unknown} are not in the run --init started (it has "
                f"{run['positions']})"
            )
        self.run_positions = run["positions"]

    def work_dir(self, key: str) -> Path:
        """One position's run folder: the run folder itself, or `positions/<key>/`."""
        if len(self.run_positions) == 1:
            return self.output_dir
        return self.output_dir / "positions" / key

    def moving(self, key: str) -> Path:
        return self.movings[self.keys.index(key)]

    def write(self, entries_by_key: dict[str, list[TransformEntry]]) -> None:
        """Write the transforms file: one shared list, or one list per position."""
        with open_ome_zarr(self.reference_for[self.keys[0]], mode="r") as position:
            voxel_size = [float(v) for v in position.scale]
        common = dict(
            direction="forward",
            moving_channels=[self.settings.moving.channel],
            reference_channel=self.settings.reference.channel,
            method=self.settings.method,
            reference_frame=self.settings.reference.frame,
            voxel_size=voxel_size,
        )
        if len(self.run_positions) == 1:
            model = TransformSettings(**common, transforms=entries_by_key[self.keys[0]])
        else:
            model = TransformSettings(**common, positions=entries_by_key)
        model_to_yaml(model, self.output_filepath)
        click.echo(f"Transform settings saved to {self.output_filepath.resolve()}")

    def entries(self, result, time_indices, transforms) -> list[TransformEntry]:
        return transform_entries(
            result, time_indices, transforms, self.settings.fallback.flag.hard_fail
        )


def estimate_transform(
    moving_position_dirpaths: list[Path],
    config_filepath: Path,
    output_filepath: Path,
    reference_position_dirpaths: list[Path] | None = None,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    resume: bool = False,
) -> None:
    """Estimate one transform per timepoint mapping the moving channel onto its reference.

    Reads an `EstimateTransformSettings` YAML and writes a `TransformSettings` YAML --
    forward matrices with their scores -- and keeps the engine's records in a folder named
    after it (`<output stem>/`); see `estimate_transform_series`. One moving position
    writes a list shared by every position (e.g. beads registration estimated on the bead
    FOV); several moving positions are each estimated on their own and written per
    position (e.g. stabilization, where every FOV drifts differently). The reference
    positions are only read for `reference.frame: cross`: one serves every moving
    position, several are paired with them by row/col/fov. `estimate_transform_init` and
    `estimate_transform_step` run the same phases one at a time (for Nextflow).
    """
    run = _Run(
        moving_position_dirpaths, config_filepath, output_filepath, reference_position_dirpaths
    )
    run.start()

    def estimate_position(key: str) -> list[TransformEntry]:
        return run.entries(
            *estimate_transform_series(
                run.moving(key),
                run.reference_for[key],
                run.settings,
                run.work_dir(key),
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                monitor=monitor,
                resume=resume,
            )
        )

    if len(run.keys) == 1:
        run.write({run.keys[0]: estimate_position(run.keys[0])})
        return
    click.echo(f"Estimating {len(run.keys)} positions, each with its own transforms")
    workers = min(len(run.keys), MAX_CONCURRENT_POSITIONS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {key: pool.submit(estimate_position, key) for key in run.keys}
    failed = {}
    positions = {}
    for key, future in futures.items():
        try:
            positions[key] = future.result()
        except Exception as e:  # noqa: BLE001 -- report every failed position at once
            failed[key] = f"{type(e).__name__}: {e}"
    if failed:
        listing = "\n".join(f"  {k}: {v}" for k, v in failed.items())
        raise click.ClickException(
            f"{len(failed)} of {len(run.keys)} positions failed; no transforms file "
            f"written (rerun with --resume to redo only what is missing):\n{listing}"
        )
    run.write(positions)


def estimate_transform_init(
    moving_position_dirpaths: list[Path],
    config_filepath: Path,
    output_filepath: Path,
    reference_position_dirpaths: list[Path] | None = None,
    resume: bool = False,
) -> dict:
    """Start a run for the steps: check the config, plan every position, print the plan.

    Reads only store metadata. Prints `RESOURCES:` (one estimate task) and
    `PLAN:{positions, time_indices, propagated, interactive, resources}` for Nextflow, and
    returns the plan.
    """
    run = _Run(
        moving_position_dirpaths, config_filepath, output_filepath, reference_position_dirpaths
    )
    run.start()
    plans = {
        key: init_run(
            run.moving(key), run.reference_for[key], run.settings, run.work_dir(key), resume
        )
        for key in run.keys
    }
    first = plans[run.keys[0]]
    plan = {
        "positions": run.keys,
        "time_indices": first["time_indices"],
        "propagated": first["propagated"],
        # manual registration needs a display and a terminal: not a batch task
        "interactive": first["interactive"],
        "resources": first["resources"],
    }
    estimate = first["resources"]["estimate"]
    echo_resources(estimate["cpus"], estimate["mem_gb"], estimate["time_minutes"])
    click.echo("PLAN:" + json.dumps(plan))
    return plan


def estimate_transform_step(
    step: str,
    moving_position_dirpaths: list[Path],
    config_filepath: Path,
    output_filepath: Path,
    reference_position_dirpaths: list[Path] | None = None,
    timepoints: list[int] | None = None,
    resume: bool = False,
) -> dict | None:
    """Run one step of a run `--init` started, in this process, for the given positions.

    `estimate`, `repair` and `sweep` do the given `timepoints` (default: all the step
    has); `estimate` of a `seed_from: previous_timepoint` run does the whole series, in
    order; they return `{"failed": {position: [t, ...]}}` for jobs that raised (recorded,
    so the run can carry on). `flag` prints `PLAN:{position: {repair, sweep}}`.
    `finalize` writes the transforms file and needs every position of the run.
    """
    run = _Run(
        moving_position_dirpaths, config_filepath, output_filepath, reference_position_dirpaths
    )
    run.load()
    if step == "finalize":
        missing = sorted(set(run.run_positions) - set(run.keys))
        if missing:
            raise click.UsageError(f"finalize writes every position; missing {missing}")
        run.write({key: run.entries(*finalize_run(run.work_dir(key))) for key in run.keys})
        return None
    if step == "flag":
        flags = {}
        for key in run.keys:
            _result, position_flags = flag_run(run.work_dir(key))
            flags[key] = {"repair": position_flags["repair"], "sweep": position_flags["sweep"]}
        click.echo("PLAN:" + json.dumps(flags))
        return flags
    failed = {}
    for key in run.keys:
        _records, failed_ts = run_timepoint_jobs(
            step,
            run.moving(key),
            run.reference_for[key],
            run.work_dir(key),
            timepoints=timepoints,
            resume=resume,
        )
        if failed_ts:
            failed[key] = failed_ts
    return {"failed": failed}


@click.command("estimate-transform")
@moving_position_dirpaths()
@reference_position_dirpaths(required=False)
@config_filepath()
@output_filepath()
@sbatch_filepath()
@cluster()
@monitor(short=False)
@resume(
    help="Keep the timepoints an earlier run of this output already finished and estimate "
    "only the rest (e.g. after jobs hit their time limit). Refused if the settings or "
    "inputs changed since that run."
)
@init_only()
@click.option(
    "--step",
    type=click.Choice(STEPS),
    default=None,
    help="Run one step of a run --init started, in this process (for Nextflow): "
    "estimate -> flag -> repair / sweep -> finalize.",
)
@click.option(
    "--timepoints",
    default=None,
    help="Timepoints for --step estimate / repair / sweep, comma-separated (e.g. 5 or "
    "5,6); default: every timepoint the step has.",
)
def estimate_transform_cli(
    moving_position_dirpaths: list[Path],
    reference_position_dirpaths: list[Path] | None,
    config_filepath: Path,
    output_filepath: Path,
    sbatch_filepath: str | None,
    cluster: str,
    monitor: bool,
    resume: bool,
    init_only: bool,
    step: str | None,
    timepoints: str | None,
) -> None:
    """Estimate a transform series mapping a moving channel onto its reference.

    Takes an `EstimateTransformSettings` YAML and writes a `TransformSettings` file for
    `apply-transform` (forward matrices with their scores). The run's report, journal and
    per-timepoint records (what --resume continues from) go in a folder named after the
    output (`-o reg/transforms.yml` -> `reg/transforms/`). One SLURM job per timepoint,
    then one per flagged timepoint for repair. With `reference.frame: first` or `previous`
    the moving channel is stabilized against itself and `-r` is not needed.

    \b
    Registration (moving channel onto the reference channel):
    >>> biahub estimate-transform -m moving.zarr/0/0/0 -r reference.zarr/0/0/0 \\
        -c estimate-transform-beads.yml -o ./transforms.yml

    \b
    Stabilization (a channel onto its own first / previous timepoint):
    >>> biahub estimate-transform -m data.zarr/0/0/0 \\
        -c estimate-transform-stabilize.yml -o ./transforms.yml

    \b
    Retry an interrupted run, keeping finished timepoints:
    >>> biahub estimate-transform --resume -m ... -r ... -c ... -o ./transforms.yml

    \b
    The same phases one step at a time, as the Nextflow module runs them (each step in
    this process; --init prints the plan as PLAN:{...}):
    >>> biahub estimate-transform --init -m ... -r ... -c ... -o ./transforms.yml
    >>> biahub estimate-transform --step estimate --timepoints 5 -m <one position> ...
    >>> biahub estimate-transform --step flag -m <one position> ...
    >>> biahub estimate-transform --step repair --timepoints 12 -m <one position> ...
    >>> biahub estimate-transform --step finalize -m <every position> ...
    """  # noqa: D301
    if init_only and step:
        raise click.UsageError("--init and --step are separate calls")
    if timepoints is not None and step not in ("estimate", "repair", "sweep"):
        raise click.UsageError("--timepoints goes with --step estimate / repair / sweep")
    common = dict(
        moving_position_dirpaths=moving_position_dirpaths,
        reference_position_dirpaths=reference_position_dirpaths,
        config_filepath=config_filepath,
        output_filepath=output_filepath,
    )
    if init_only:
        estimate_transform_init(**common, resume=resume)
        return
    if step:
        result = estimate_transform_step(
            step,
            **common,
            timepoints=None
            if timepoints is None
            else [int(t) for t in timepoints.split(",") if t.strip()],
            resume=resume,
        )
        if result and result.get("failed"):
            # Recorded (the run can carry on), but not a success: a distinct exit code
            # lets a workflow retry it and keep it out of its cache (registration.nf).
            click.echo(f"{step}: jobs failed at {result['failed']}", err=True)
            raise SystemExit(JOBS_FAILED_EXIT_CODE)
        return
    estimate_transform(
        moving_position_dirpaths=moving_position_dirpaths,
        reference_position_dirpaths=reference_position_dirpaths,
        config_filepath=config_filepath,
        output_filepath=output_filepath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        resume=resume,
    )


if __name__ == "__main__":
    estimate_transform_cli()
