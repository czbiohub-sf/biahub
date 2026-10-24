"""Compute multiscale pyramids in place, on a plate's images and its label images.

Every position of the input plate gets ``levels`` resolution levels: level 0 is the
existing array and each further level is 2x coarser along the spatial axes longer than
one voxel, so a projected (Z=1) plate is downsampled in Y and X only. Images are pooled
with ``method`` and the label images under ``labels/`` with ``label_method``, one that
never invents a label id. The work is iohub's ``compute_pyramid``, run once per
position; this module fans it out and records what it did.
"""

import math

from pathlib import Path

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    init_only,
    input_position_dirpaths,
    monitor,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.settings import LABEL_PYRAMID_METHODS, PYRAMID_METHODS, PyramidSettings
from biahub.utils.cluster import (
    apply_sbatch_overrides,
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
    local_executor_parameters,
)

PROVENANCE_KEY = "biahub-pyramid"


def pyramid_dims(shape_zyx: tuple[int, int, int]) -> set[str]:
    """Spatial axes a pyramid downsamples: those longer than one voxel.

    A singleton axis cannot be halved, and scaling its metadata would misreport the
    voxel size, so the Z=1 plate ``biahub project`` writes is downsampled in Y and X
    only. Time and channel are never downsampled.
    """
    return {name for name, size in zip("zyx", shape_zyx, strict=True) if size > 1}


def _check_levels(position_dirpath: Path, levels: int) -> None:
    """Refuse a position whose image or labels already have a different pyramid.

    ``compute_pyramid`` recomputes an existing pyramid of the same depth in place, which
    is what a retry wants, but a different depth needs ``delete_pyramid`` first; catch
    that before any job is submitted, on the first position.
    """
    with open_ome_zarr(position_dirpath, mode="r") as position:
        found = {"image": len(position.array_keys())}
        for name, label in position.labels():
            found[f"labels/{name}"] = len(label.array_keys())
    mismatched = {key: n for key, n in found.items() if n not in (1, levels)}
    if mismatched:
        raise ValueError(
            f"{position_dirpath} already has a pyramid of a different depth than the "
            f"{levels} levels requested: {mismatched}. Delete it (iohub's delete_pyramid) "
            "or request that depth."
        )


def _pyramid_position(position_dirpath: Path, settings: PyramidSettings) -> None:
    """Compute the pyramid of one position's image and of each of its label images."""
    click.echo(f"Computing pyramid for {position_dirpath}")
    with open_ome_zarr(position_dirpath, mode="r+") as position:
        dims = pyramid_dims(position.data.shape[-3:])
        position.compute_pyramid(levels=settings.levels, method=settings.method, dims=dims)
        for name, label in position.labels():
            click.echo(f"Computing pyramid for labels/{name} of {position_dirpath}")
            label.compute_pyramid(
                levels=settings.levels, method=settings.label_method, dims=dims
            )
        position.zattrs[PROVENANCE_KEY] = {**settings.model_dump(), "dims": sorted(dims)}


def pyramid(
    input_position_dirpaths: list[Path],
    levels: int = 3,
    method: str = "mean",
    label_method: str = "max",
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
) -> None:
    """Compute multiscale pyramids of the given positions, images and labels alike.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to the position directories; the pyramids are written into them.
    levels : int, optional
        Total number of resolution levels, counting level 0.
    method : str, optional
        Downsampling method for the images, one of ``PYRAMID_METHODS``.
    label_method : str, optional
        Downsampling method for the label images, one of ``LABEL_PYRAMID_METHODS``.
    sbatch_filepath : str, optional
        Path to the SLURM batch file.
    cluster : str, optional
        Execution cluster: 'slurm' submits to a Slurm cluster, 'local' runs jobs as
        subprocesses on this machine, 'debug' runs jobs in-process in the foreground.
    monitor : bool, optional
        If True, monitor the submitted jobs.
    init_only : bool, optional
        Only validate the request against the first position and report the resources
        a position needs; skip per-position processing.
    """
    slurm_out_path = input_position_dirpaths[0].parents[3] / "slurm_output"

    settings = PyramidSettings(levels=levels, method=method, label_method=label_method)
    _check_levels(input_position_dirpaths[0], settings.levels)

    with open_ome_zarr(input_position_dirpaths[0], mode="r") as position:
        T, C, Z, Y, X = position.data.shape
        dtype = position.data.dtype
        shards = position.data.shards or position.data.chunks

    # RAM follows the SHARD, not the volume: iohub writes a pyramid level one target
    # shard at a time and reads the source region feeding it, 8x the target in 3D.
    # Measured on one A549 position (67 x 6 x 86 x 1664 x 1193 float32, level-0
    # shards 5 x 1 x 96 x 1792 x 1280 = 4.1 GiB, levels sized to the same bytes):
    # 44 GB peak RSS and 41.5 min single-threaded, i.e. ~10x the shard and ~0.1 min
    # per (t, c) volume; the projected plate's 0.14 GiB shards peaked at 1.7 GB in 27 s.
    # All 54 positions of that plate in the v9 debug run (shards 6 x 1 x 88 x 1792 x
    # 1280 = 4.51 GiB): 46.5 GB peak RSS on every position (10.3x the shard), and
    # 45.6 / 54.5 / 67.3 min min / median / max, i.e. up to 0.17 min per volume, the
    # slowest on the 120 GB cpu-c nodes. The old 12x / 0.15 asked 56 GB / 70 min,
    # 4% above the slowest position. 14x the shard (~36% over the peak) and
    # 0.3 min/volume (~1.8x the slowest) leave room for slower nodes.
    itemsize = np.dtype(dtype).itemsize
    shard_gb = math.prod(shards) * itemsize / 2**30
    volume_gb = Z * Y * X * itemsize / 2**30
    time_minutes, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, C, Z, Y, X),
        dtype=dtype,
        ram_multiplier=max(4.0, 14 * shard_gb / volume_gb / 4),  # per CPU, 4 CPUs share it
        time_multiplier=0.3,
        max_num_cpus=4,
    )
    mem_gb = num_cpus * gb_ram_per_cpu
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        click.echo(
            f"Validated {settings.levels}-level pyramids for {len(input_position_dirpaths)} "
            "positions"
        )
        return

    slurm_args = {
        "slurm_job_name": "pyramid",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": time_minutes,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args = apply_sbatch_overrides(slurm_args, sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    if resolved_cluster == "local":
        # Explicit --sbatch-filepath values (e.g. '#LOCAL --timeout-min') win over
        # the computed local executor parameters.
        slurm_args = {**local_executor_parameters(time_minutes), **slurm_args}
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path in input_position_dirpaths:
            jobs.append(executor.submit(_pyramid_position, input_position_path, settings))

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a DebugJob but
    # execution only happens on .wait()/.done()/.result(). On the Nextflow path
    # (--cluster debug) run each position in the foreground.
    if resolved_cluster == "debug":
        for job, path in zip(jobs, input_position_dirpaths, strict=True):
            job.wait()
            click.echo(f"Pyramid complete: {path}")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("pyramid")
@input_position_dirpaths()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@click.option(
    "--levels",
    type=int,
    default=3,
    show_default=True,
    help="Total number of resolution levels, counting level 0: 3 adds 2x and 4x levels.",
)
@click.option(
    "--method",
    type=click.Choice(PYRAMID_METHODS),
    default="mean",
    show_default=True,
    help="Downsampling method for the images.",
)
@click.option(
    "--label-method",
    type=click.Choice(LABEL_PYRAMID_METHODS),
    default="max",
    show_default=True,
    help="Downsampling method for the label images under labels/; none invents a label id.",
)
def pyramid_cli(
    input_position_dirpaths: list[Path],
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    levels: int = 3,
    method: str = "mean",
    label_method: str = "max",
) -> None:
    """Compute multiscale pyramids of a plate's images and label images, in place.

    Each level is 2x coarser than the previous along the spatial axes longer than one
    voxel, so a Z=1 plate is downsampled in Y and X only. Images use --method, the label
    images under labels/ use --label-method. Rerunning recomputes the same levels.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub pyramid -i ./plate.zarr/*/*/* --levels 3

    \b
    Validate the request and report the per-position resources only:
    >>> biahub pyramid --init -i ./plate.zarr/*/*/* --levels 3

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub pyramid --cluster debug -i ./plate.zarr/A/1/0 --levels 3
    """  # noqa: D301
    pyramid(
        input_position_dirpaths=input_position_dirpaths,
        levels=levels,
        method=method,
        label_method=label_method,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
    )


if __name__ == "__main__":
    pyramid_cli()
