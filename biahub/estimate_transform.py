"""Estimate a transform series with the registration engine.

Maps a moving channel onto a reference per timepoint -- another channel (registration)
or the same channel at a fixed or previous timepoint (stabilization) -- and writes a
`TransformSettings` config that the apply steps consume. Two SLURM fan-out phases: one
job per timepoint to estimate, then one job per flagged timepoint to repair against the
frozen whole-run history. Every job writes a small JSON record, so an interrupted run
resumes.
"""

from __future__ import annotations

from pathlib import Path

import click

from iohub import open_ome_zarr

from biahub.cli.parsing import (
    cluster,
    config_filepath,
    monitor,
    output_filepath,
    resume,
    sbatch_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.registration.engine import estimate_transform_series
from biahub.settings import (
    TransformSettings,
    load_estimate_transform_settings,
)
from biahub.utils.config import model_to_yaml


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

    # One matrix (a single timepoint, e.g. manual or `time_indices: 0`) is the transform
    # for the whole series and applies to every timepoint; several matrices for a subset
    # keep the subset so apply-transform pairs each with its timepoint.
    model = TransformSettings(
        direction="forward",
        matrices=[transform.to_list() for transform in transforms],
        time_indices="all" if len(transforms) == 1 else settings.time_indices,
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
