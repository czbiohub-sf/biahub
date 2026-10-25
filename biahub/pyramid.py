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
    resume,
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
from biahub.utils.config import settings_fingerprint

PROVENANCE_KEY = "biahub-pyramid"
# Downsampling threads per CPU of the job. A thread spends most of a (t, c) volume
# waiting on the read (about 8 s of a 9 s cold Lustre read is I/O), so two per CPU
# keep the CPUs busy while the other half waits.
THREADS_PER_CPU = 2


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


def _pyramid_position(
    position_dirpath: Path,
    settings: PyramidSettings,
    resume: bool = False,
    num_workers: int = 1,
) -> None:
    """Compute the pyramid of one position's image and of each of its label images.

    Every level is written one shard-aligned region at a time, each a resumable unit
    recorded beside the store under the settings' fingerprint; with ``resume`` the
    regions an earlier attempt finished are skipped (see iohub's ``compute_pyramid``).
    ``num_workers`` threads downsample each region, one (t, c) volume each; it does
    not change the result, so it is not part of the fingerprint.
    """
    click.echo(f"Computing pyramid for {position_dirpath}")
    token = settings_fingerprint(settings)
    with open_ome_zarr(position_dirpath, mode="r+") as position:
        dims = pyramid_dims(position.data.shape[-3:])
        position.compute_pyramid(
            levels=settings.levels,
            method=settings.method,
            dims=dims,
            resume=resume,
            resume_token=token,
            num_workers=num_workers,
        )
        for name, label in position.labels():
            click.echo(f"Computing pyramid for labels/{name} of {position_dirpath}")
            label.compute_pyramid(
                levels=settings.levels,
                method=settings.label_method,
                dims=dims,
                resume=resume,
                resume_token=token,
                num_workers=num_workers,
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
    resume: bool = False,
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
    resume : bool, optional
        Skip the pyramid regions (one shard of one level each) a previous attempt
        already finished, rather than recomputing every level; see
        ``iohub.ngff.nodes.MultiscaleNode.compute_pyramid``.
    """
    slurm_out_path = input_position_dirpaths[0].parents[3] / "slurm_output"

    settings = PyramidSettings(levels=levels, method=method, label_method=label_method)
    _check_levels(input_position_dirpaths[0], settings.levels)

    with open_ome_zarr(input_position_dirpaths[0], mode="r") as position:
        T, C, Z, Y, X = position.data.shape
        dtype = position.data.dtype
        shards = position.data.shards or position.data.chunks

    # iohub writes a level one target shard at a time, and downsamples each shard in
    # (t, c) sub-blocks, THREADS_PER_CPU threads per CPU. RAM is the shard twice (the
    # buffer the sub-blocks fill, and its encoded copy on write) plus about two
    # source volumes per thread (the read, and np.pad's copy along an odd axis).
    # Measured on full, uncached A549 v10 assembled positions (67 x 6 x 86 x 1664 x
    # 1193 float32, 4.5 GiB shards) on 8 CPUs: the v10 run itself (iohub main,
    # serial, 4 CPUs) 54 min median and 46.5 GB, with 2 of 54 positions past 130 min
    # on slow gpu-sm nodes; 16 threads 14.6 min and 21 GB; 16 threads with
    # RAYON_NUM_THREADS=64 (set by modules/pyramid.nf) 7.3 min and 23 GB, i.e.
    # 0.018 min/volume. Requested: 40 GB, and 0.08 min/volume (40 min), ~5x the
    # measured position, for slower nodes and 60 jobs reading Lustre at once.
    itemsize = np.dtype(dtype).itemsize
    shard_gb = math.prod(shards) * itemsize / 2**30
    volume_gb = Z * Y * X * itemsize / 2**30
    max_num_cpus = 8
    cpus = min(T * C, max_num_cpus)
    time_minutes, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, C, Z, Y, X),
        dtype=dtype,
        # Per CPU, with a 30% margin: two volumes per thread, plus this CPU's share of
        # two shards.
        ram_multiplier=1.3 * (2 * THREADS_PER_CPU + 2 * shard_gb / volume_gb / cpus),
        time_multiplier=0.08,
        max_num_cpus=max_num_cpus,
        min_ram_per_cpu=2,
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
            jobs.append(
                executor.submit(
                    _pyramid_position,
                    input_position_path,
                    settings,
                    resume=resume,
                    # Per CPU the job holds, including an sbatch override.
                    num_workers=THREADS_PER_CPU * slurm_args["slurm_cpus_per_task"],
                )
            )

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
@resume(
    help="Skip the pyramid regions (one shard of one level each) an earlier attempt finished, "
    "keyed by the levels and methods; for retrying an interrupted run."
)
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
    resume: bool = False,
    levels: int = 3,
    method: str = "mean",
    label_method: str = "max",
) -> None:
    """Compute multiscale pyramids of a plate's images and label images, in place.

    Each level is 2x coarser than the previous along the spatial axes longer than one
    voxel, so a Z=1 plate is downsampled in Y and X only. Images use --method, the label
    images under labels/ use --label-method. Rerunning recomputes the same levels;
    --resume recomputes only the shards an interrupted attempt did not finish.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub pyramid -i ./plate.zarr/*/*/* --levels 3

    \b
    Validate the request and report the per-position resources only:
    >>> biahub pyramid --init -i ./plate.zarr/*/*/* --levels 3

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub pyramid --cluster debug --resume -i ./plate.zarr/A/1/0 --levels 3
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
        resume=resume,
    )


if __name__ == "__main__":
    pyramid_cli()
