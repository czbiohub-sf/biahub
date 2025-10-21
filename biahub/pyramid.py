import datetime

from pathlib import Path
from typing import List, Optional

import click
import submitit
import tensorstore as ts

from iohub.ngff import open_ome_zarr

from biahub.cli.parsing import (
    input_position_dirpaths,
    local,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources


def _write_ts_downsampled(
    source_ts: ts.TensorStore,
    target_ts: ts.TensorStore,
    downsample_factors: list[int],
    method: str,
    level: int,
) -> None:
    """
    Tensorstore downsampling with chunking.

    Parameters
    ----------
    source_ts : ts.TensorStore
        Source tensorstore Zarr to downsample from
    target_ts : ts.TensorStore
        Target tensorstore Zarr to write downsampled data to
    downsample_factors : list[int]
        Downsampling factors for each dimension
    method : str
        Downsampling method (e.g., 'mean', 'max', 'min')
    level : int
        Pyramid level being processed
    """
    downsampled = ts.downsample(
        source_ts, downsample_factors=downsample_factors, method=method
    )

    step = target_ts.chunk_layout.write_chunk.shape[0]

    for start in range(0, downsampled.shape[0], step):
        with ts.Transaction() as txn:
            target_with_txn = target_ts.with_transaction(txn)
            downsampled_with_txn = downsampled.with_transaction(txn)
            stop = min(start + step, downsampled.shape[0])
            target_with_txn[start:stop].write(downsampled_with_txn[start:stop]).result()


def pyramid(fov_path: Path, levels: int, method: str) -> None:
    """
    Create pyramid levels for a single field of view using tensorstore downsampling.

    This function uses cascade downsampling, where each level is downsampled from
    the previous level rather than from level 0. This avoids aliasing artifacts
    and chunk boundary issues that occur with large downsample factors.

    Parameters
    ----------
    fov_path : Path
        Path to the FOV position directory
    levels : int
        Number of downsampling levels to create
    method : str
        Downsampling method (e.g., 'mean', 'max', 'min')
    """

    with open_ome_zarr(fov_path, mode="r+") as dataset:
        dataset.initialize_pyramid(levels=levels + 1)

        for level in range(1, levels + 1):
            previous_level = dataset[str(level - 1)].tensorstore()

            current_scale = dataset.get_effective_scale(str(level))
            previous_scale = dataset.get_effective_scale(str(level - 1))
            downsample_factors = [
                int(round(current_scale[i] / previous_scale[i]))
                for i in range(len(current_scale))
            ]

            target_store = dataset[str(level)].tensorstore()
            _write_ts_downsampled(
                source_ts=previous_level,
                target_ts=target_store,
                downsample_factors=downsample_factors,
                method=method,
                level=level,
            )

    click.echo(f"Completed pyramid for FOV: {fov_path}")


@click.command("pyramid")
@input_position_dirpaths()
@sbatch_filepath()
@local()
@click.option(
    "--levels",
    "-lv",
    type=int,
    default=3,
    show_default=True,
    help="Number of downsampling levels to create.",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(
        [
            "stride",
            "median",
            "mode",
            "mean",
            "min",
            "max",
        ]
    ),
    default="mean",
    show_default=True,
    help="Downsampling method to use.",
)
def pyramid_cli(
    input_position_dirpaths: List[Path],
    levels: int = 3,
    method: str = "mean",
    sbatch_filepath: Optional[Path] = None,
    local: bool = False,
) -> None:
    """
    Creates additional levels of multi-scale pyramids for OME-Zarr datasets.

    Uses tensorstore downsampling to generate progressively downscaled pyramid levels.
    Setting levels=0 skips pyramid creation. For levels > 0, creates n additional pyramid
    levels with 2^i downsampling (e.g., levels=3 creates 2x, 4x, and 8x downsampled versions).

    Example:
        biahub pyramid -i ./data.zarr/*/*/* --levels 5 --local
        biahub pyramid -i ./data.zarr/0/0/0 -lv 3 --method max
    """
    if levels == 0:
        click.echo("No pyramid levels to create.")
        return

    # Estimate resources based on first FOV data shape
    with open_ome_zarr(input_position_dirpaths[0], mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape

    num_cpus, gb_ram = estimate_resources(shape=(T, C, Z, Y, X), ram_multiplier=5)

    cluster = "local" if local else "slurm"

    slurm_args = {
        "slurm_job_name": "pyramid",
        "slurm_partition": "cpu",
        "slurm_cpus_per_task": num_cpus,
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_time": 30,
        "slurm_array_parallelism": 100,
    }

    # Override with sbatch file parameters if provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    slurm_out_path = Path("slurm_output")
    slurm_out_path.mkdir(exist_ok=True)

    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(
        f"Submitting {len(input_position_dirpaths)} pyramid jobs with resources: {slurm_args}"
    )

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for fov_path in input_position_dirpaths:
            job = executor.submit(pyramid, fov_path=fov_path, levels=levels, method=method)
            jobs.append(job)

    job_ids = [job.job_id for job in jobs]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"pyramid-jobs_{timestamp}.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))


if __name__ == "__main__":
    pyramid_cli()
