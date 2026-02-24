import shutil

from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr
from numpy.typing import ArrayLike
from pystackreg import StackReg

from biahub.cli.parsing import (
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import estimate_resources
from biahub.settings import (
    StackRegSettings,
)


def estimate_xy_stabilization_per_position(
    input_position_dirpath: Path,
    output_folder_path: Path,
    df_z_focus_path: Path,
    channel_index: int,
    center_crop_xy: list[int, int],
    t_reference: str = "previous",
    verbose: bool = False,
) -> ArrayLike:
    """
    Estimate the xy stabilization for a single position.

    Parameters
    ----------
    input_position_dirpath : Path
        Path to the input position directory.
    output_folder_path : Path
        Path to the output folder.
    df_z_focus_path : Path
        Path to the input focus CSV file.
    channel_index : int
        Index of the channel to process.
    center_crop_xy : list[int, int]
        Size of the crop in the XY plane.
    t_reference : str
        Reference timepoint.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Transformation matrix.
    """
    with open_ome_zarr(input_position_dirpath) as input_position:
        T, _, _, Y, X = input_position.data.shape
        x_idx = slice(X // 2 - center_crop_xy[0] // 2, X // 2 + center_crop_xy[0] // 2)
        y_idx = slice(Y // 2 - center_crop_xy[1] // 2, Y // 2 + center_crop_xy[1] // 2)

        if verbose:
            click.echo(f"Reading focus index from {df_z_focus_path}")
        df = pd.read_csv(df_z_focus_path)
        pos_idx = str(Path(*input_position_dirpath.parts[-3:]))
        focus_idx = df[df["position"] == pos_idx]["focus_idx"]
        focus_idx = focus_idx.replace(0, np.nan).ffill().fillna(focus_idx.mean())

        z_idx = focus_idx.astype(int).to_list()

        if verbose:
            click.echo("Calculating xy stabilization...")
        # Get the data for the specified channel and crop
        tyx_data = np.stack(
            [
                input_position[0][t, channel_index, z, y_idx, x_idx]
                for t, z in zip(range(T), z_idx)
            ]
        )
        tyx_data = np.clip(tyx_data, a_min=0, a_max=None)

        sr = StackReg(StackReg.TRANSLATION)
        T_stackreg = sr.register_stack(tyx_data, reference=t_reference, axis=0)

        # Swap translation directions: (x, y) -> (y, x)
        for tform in T_stackreg:
            tform[0, 2], tform[1, 2] = tform[1, 2], tform[0, 2]

        transform = np.zeros((T_stackreg.shape[0], 4, 4))
        transform[:, 1:4, 1:4] = T_stackreg
        transform[:, 0, 0] = 1
        # save the transforms as
        position_filename = str(Path(*input_position_dirpath.parts[-3:]))
        position_filename = position_filename.replace("/", "_")

        np.save(output_folder_path / f"{position_filename}.npy", transform.astype(np.float32))

    return transform


def estimate_xy_stabilization(
    input_position_dirpaths: list[Path],
    output_folder_path: Path,
    stack_reg_settings: StackRegSettings,
    channel_index: int = 0,
    sbatch_filepath: Optional[Path] = None,
    cluster: str = "local",
    verbose: bool = False,
) -> dict[str, list[ArrayLike]]:
    """
    Estimate XY stabilization using StackReg.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to the input position directories.
    output_folder_path : Path
        Path to the output folder.
    stack_reg_settings : StackRegSettings
        Settings for the stack registration.
    channel_index : int
        Index of the channel to process.
    sbatch_filepath : Path
        Path to the sbatch file.
    cluster : str
        Cluster to use.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    dict[str, list[ArrayLike]]
        Dictionary of the xy stabilization for each position.
    """

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    # Estimate resources from a sample dataset
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        shape = dataset.data.shape  # (T, C, Z, Y, X)

    df_focus_path = output_folder_path / "positions_focus.csv"

    if df_focus_path.exists():
        click.echo("Using existing Z focus index file.")
    else:
        click.echo("Estimating Z focus positions...")

        from biahub.registration.z_focus_finding import estimate_z_stabilization

        estimate_z_stabilization(
            input_position_dirpaths=input_position_dirpaths,
            output_folder_path=output_folder_path,
            channel_index=channel_index,
            sbatch_filepath=sbatch_filepath,
            cluster=cluster,
            verbose=verbose,
            estimate_z_index=True,
            focus_finding_settings=stack_reg_settings.focus_finding_settings,
        )

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=16, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_focus_z",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 10,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")
    output_transforms_path = output_folder_path / "xy_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_dirpath in input_position_dirpaths:
            job = executor.submit(
                estimate_xy_stabilization_per_position,
                input_position_dirpath=input_position_dirpath,
                output_folder_path=output_transforms_path,
                df_z_focus_path=df_focus_path,
                channel_index=channel_index,
                center_crop_xy=stack_reg_settings.center_crop_xy,
                t_reference=stack_reg_settings.t_reference,
                verbose=verbose,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    wait_for_jobs_to_finish(jobs)

    transforms_paths = list(output_transforms_path.glob("*.npy"))
    fov_transforms = {}

    for file_path in transforms_paths:
        fov_filename = file_path.stem
        fov_transforms[fov_filename] = np.load(file_path).tolist()

    shutil.rmtree(output_transforms_path)

    return fov_transforms
