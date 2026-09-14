import datetime

from pathlib import Path
from typing import Annotated, Literal

import submitit
import typer

from iohub.ngff import open_ome_zarr

from biahub.cli.parsing import InputPositionDirpaths, SbatchFilepath, local, sbatch_to_submitit
from biahub.utils.cluster import estimate_resources, get_submitit_cluster


def pyramid(fov_path: Path, levels: int, method: str) -> None:
    """
    Create pyramid levels for a single field of view.

    Delegates to iohub's Position.compute_pyramid() which uses cascade downsampling,
    where each level is downsampled from the previous level rather than from level 0.
    This avoids aliasing artifacts and chunk boundary issues.

    Parameters
    ----------
    fov_path : Path
        Path to the FOV position directory
    levels : int
        Total number of resolution levels (including level 0).
        E.g., levels=4 creates arrays "0", "1", "2", "3".
    method : str
        Downsampling method (e.g., 'mean', 'max', 'min')
    """
    typer.echo(f"Computing pyramid for FOV: {fov_path}")

    with open_ome_zarr(fov_path, mode="r+") as dataset:
        dataset.compute_pyramid(levels=levels, method=method)


def pyramid_cli(
    input_position_dirpaths: InputPositionDirpaths,
    sbatch_filepath: SbatchFilepath = None,
    local: local = False,
    levels: Annotated[
        int,
        typer.Option(
            "--levels",
            "-lv",
            show_default=True,
            help=(
                "Total number of resolution levels including level 0. "
                "E.g., levels=4 creates 0, 1, 2, 3."
            ),
        ),
    ] = 4,
    method: Annotated[
        Literal["stride", "median", "mode", "mean", "min", "max"],
        typer.Option(
            "--method",
            "-m",
            show_default=True,
            help="Downsampling method to use.",
        ),
    ] = "mean",
) -> None:
    """Create multi-scale pyramids for OME-Zarr datasets.

    Uses cascade downsampling to generate progressively downscaled pyramid levels.
    Each level is 2x downsampled from the previous (e.g., levels=4 creates the original
    plus 2x, 4x, and 8x downsampled versions as arrays "0", "1", "2", "3").

    >>> biahub pyramid -i ./data.zarr/*/*/* --levels 4 --local
    """
    if levels <= 1:
        typer.echo("No pyramid levels to create (levels must be > 1).")
        return

    # Estimate resources based on first FOV data shape
    with open_ome_zarr(input_position_dirpaths[0], mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape

    _, num_cpus, gb_ram = estimate_resources(shape=(T, C, Z, Y, X), ram_multiplier=5)

    cluster = get_submitit_cluster(local)

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

    typer.echo(
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
