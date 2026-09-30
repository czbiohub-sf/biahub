"""Estimate a transform series with the registration engine.

Maps a moving channel onto a reference per timepoint -- another channel (registration)
or the same channel at its first or previous timepoint (stabilization) -- and writes the
`TransformSettings` file that `apply-transform` consumes. Two SLURM fan-out phases: one
job per timepoint to estimate, then one job per flagged timepoint to repair against the
frozen whole-run history. Every job writes a small JSON record, so an interrupted run
resumes.
"""

from __future__ import annotations

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
    result: SeriesResult, time_indices: list[int], transforms
) -> list[TransformEntry]:
    """One entry per estimated timepoint with its score and repair provenance.

    A single timepoint (manual, `time_indices: 0`) is the series' transform: one entry
    without `t`, which `apply-transform` applies to every timepoint.
    """
    entries = []
    for t, transform in zip(time_indices, transforms, strict=True):
        score = result.scores.get(t)
        repair = result.repairs.get(t)
        entries.append(
            TransformEntry(
                t=None if len(time_indices) == 1 else t,
                matrix=transform.to_list(),
                score=None if score is None or not np.isfinite(score) else float(score),
                repaired_from=repair.source
                if repair is not None and repair.accepted
                else None,
            )
        )
    return entries


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
    `estimate_transform_series`. The reference positions are only read for
    `reference.frame: cross`.
    """
    output_filepath = Path(output_filepath)
    output_dir = output_filepath.parent
    settings = load_estimate_transform_settings(config_filepath)
    moving = Path(moving_position_dirpaths[0])
    if settings.reference.frame == "cross":
        if not reference_position_dirpaths:
            raise click.UsageError(
                "reference frame 'cross' needs the reference positions (-r)"
            )
        reference = Path(reference_position_dirpaths[0])
    else:
        reference = moving
    with open_ome_zarr(reference, mode="r") as position:
        voxel_size = [float(v) for v in position.scale]

    result, time_indices, transforms = estimate_transform_series(
        moving,
        reference,
        settings,
        output_dir,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        resume=resume,
    )

    model = TransformSettings(
        direction="forward",
        moving_channels=[settings.moving.channel],
        reference_channel=settings.reference.channel,
        method=settings.method,
        voxel_size=voxel_size,
        transforms=transform_entries(result, time_indices, transforms),
    )
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
