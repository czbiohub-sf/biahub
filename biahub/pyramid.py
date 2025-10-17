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
        dataset.initialize_pyramid(levels=levels)

        for level in range(1, levels):
            previous_level = dataset[str(level - 1)].tensorstore()

            current_scale = dataset.get_effective_scale(str(level))
            previous_scale = dataset.get_effective_scale(str(level - 1))
            downsample_factors = [
                int(round(current_scale[i] / previous_scale[i]))
                for i in range(len(current_scale))
            ]

            click.echo(f"  Level {level}: factors {downsample_factors} (from level {level-1})")

            downsampled = ts.downsample(
                previous_level, downsample_factors=downsample_factors, method=method
            )

            target_store = dataset[str(level)].tensorstore()
            target_store[:].write(downsampled[:].read().result()).result()

        click.echo(f"Completed pyramid for FOV: {fov_path}")


@click.command("pyramid")
@input_position_dirpaths()
@sbatch_filepath()
@local()
@click.option(
    "--levels",
    "-lv",
    type=int,
    default=4,
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
    levels: int,
    method: str,
    sbatch_filepath: Optional[Path],
    local: bool,
) -> None:
    """
    Creates additional levels of multi-scale pyramids for OME-Zarr datasets.

    Uses efficient downsampling to create pyramid levels
    in parallel. Each field-of-view (FOV) is processed as a separate SLURM job,
    downsampling all timepoints and channels. The pyramids are created in-place
    within the input zarr store using the specified downsampling method (default: 'mean').

    Example:
        biahub pyramid -i ./data.zarr/0/0/0 -lv 4 --method max
        biahub pyramid -i ./data.zarr/*/*/* --levels 5 --local
    """
    cluster = "local" if local else "slurm"

    slurm_args = {
        "slurm_job_name": "pyramid",
        "slurm_partition": "preempted",
        "slurm_cpus_per_task": 16,
        "slurm_mem_per_cpu": "8G",
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

    # wait_for_jobs_to_finish(jobs)


if __name__ == "__main__":
    pyramid_cli()
