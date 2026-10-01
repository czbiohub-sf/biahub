"""Estimate a transform series with the registration engine.

Maps a moving channel onto a reference per timepoint -- another channel (registration)
or the same channel at its first or previous timepoint (stabilization) -- and writes the
`TransformSettings` file that `apply-transform` consumes. Two SLURM fan-out phases: one
job per timepoint to estimate, then one job per flagged timepoint to repair against the
frozen whole-run history. Every job writes a small JSON record, so an interrupted run
resumes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import (
    cluster,
    config_filepath,
    monitor,
    moving_position_dirpaths,
    output_filepath,
    pair_reference_positions,
    position_key,
    reference_position_dirpaths,
    resume,
    sbatch_filepath,
)
from biahub.registration.engine import SeriesResult, estimate_transform_series
from biahub.settings import (
    TransformEntry,
    TransformSettings,
    load_estimate_transform_settings,
)
from biahub.utils.config import model_to_yaml


def transform_entries(
    result: SeriesResult, time_indices: list[int], transforms, hard_fail: float
) -> list[TransformEntry]:
    """One entry per estimated timepoint with its score, repair provenance and status.

    A timepoint is `unreliable` when the pipeline found no good transform for it: none
    at all (the entry is a stand-in, `filled_from` says which), no score, or flagged and
    still below `hard_fail` after repair and sweep. Otherwise `accepted`.

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
            )
        )
    return entries


# Positions estimated at once; each driver fans its own timepoints out to SLURM.
MAX_CONCURRENT_POSITIONS = 8


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
    forward matrices with their scores -- next to the engine's records; see
    `estimate_transform_series`. One moving position writes a list shared by every
    position (e.g. beads registration estimated on the bead FOV); several moving
    positions are each estimated on their own and written per position (e.g.
    stabilization, where every FOV drifts differently). The reference positions are only
    read for `reference.frame: cross`: one serves every moving position, several are
    paired with them by row/col/fov.
    """
    output_filepath = Path(output_filepath)
    output_dir = output_filepath.parent
    settings = load_estimate_transform_settings(config_filepath)
    movings = [Path(p) for p in moving_position_dirpaths]
    keys = [position_key(p) for p in movings]
    if settings.reference.frame == "cross":
        if not reference_position_dirpaths:
            raise click.UsageError(
                "reference frame 'cross' needs the reference positions (-r)"
            )
        reference_for = pair_reference_positions(keys, reference_position_dirpaths)
    else:
        reference_for = dict(zip(keys, movings, strict=True))
    with open_ome_zarr(reference_for[keys[0]], mode="r") as position:
        voxel_size = [float(v) for v in position.scale]

    def estimate_position(key: str, moving: Path, work_dir: Path) -> list[TransformEntry]:
        result, time_indices, transforms = estimate_transform_series(
            moving,
            reference_for[key],
            settings,
            work_dir,
            sbatch_filepath=sbatch_filepath,
            cluster=cluster,
            monitor=monitor,
            resume=resume,
        )
        return transform_entries(
            result, time_indices, transforms, settings.fallback.flag.hard_fail
        )

    common = dict(
        direction="forward",
        moving_channels=[settings.moving.channel],
        reference_channel=settings.reference.channel,
        method=settings.method,
        voxel_size=voxel_size,
    )
    if len(movings) == 1:
        model = TransformSettings(
            **common, transforms=estimate_position(keys[0], movings[0], output_dir)
        )
    else:
        click.echo(f"Estimating {len(movings)} positions, each with its own transforms")
        workers = min(len(movings), MAX_CONCURRENT_POSITIONS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                key: pool.submit(
                    estimate_position, key, moving, output_dir / "positions" / key
                )
                for key, moving in zip(keys, movings, strict=True)
            }
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
                f"{len(failed)} of {len(movings)} positions failed; no transforms file "
                f"written (rerun with --resume to redo only what is missing):\n{listing}"
            )
        model = TransformSettings(**common, positions=positions)
    model_to_yaml(model, output_filepath)
    click.echo(f"Transform settings saved to {output_filepath.resolve()}")


@click.command("estimate-transform")
@moving_position_dirpaths()
@reference_position_dirpaths(required=False)
@config_filepath()
@output_filepath()
@sbatch_filepath()
@cluster()
@monitor(short=False)
@resume()
def estimate_transform_cli(
    moving_position_dirpaths: list[Path],
    reference_position_dirpaths: list[Path] | None,
    config_filepath: Path,
    output_filepath: Path,
    sbatch_filepath: str | None,
    cluster: str,
    monitor: bool,
    resume: bool,
) -> None:
    """Estimate a transform series mapping a moving channel onto its reference.

    Takes an `EstimateTransformSettings` YAML and writes a `TransformSettings` file for
    `apply-transform` (forward matrices with their scores), plus a run journal and a
    per-timepoint report. One SLURM job per timepoint, then one per flagged timepoint
    for repair. With `reference.frame: first` or `previous` the moving channel is
    stabilized against itself and `-r` is not needed.

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
    """  # noqa: D301
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
