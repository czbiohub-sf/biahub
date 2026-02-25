import itertools
import shutil

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Tuple

import click
import numpy as np
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr
from numpy.typing import ArrayLike
from waveorder.focus import focus_from_transverse_band

from biahub.cli.parsing import (
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import estimate_resources
from biahub.settings import (
    FocusFindingSettings,
)

from biahub.core.transform import Transform


NA_DET = 1.35
LAMBDA_ILL = 0.500

def estimate(
    ref_zyx: np.ndarray,
    mov_zyx: np.ndarray,
    pixel_size: tuple[float, float, float],
    ref_z_index: int = None,
    mov_z_index: int = None,
    verbose: bool = False,
    output_path: Path = None,
) -> Transform:

    # TODO: reaad from metadata eds pr

    if ref_z_index is None:
        ref_z_index = focus_from_transverse_band(
            ref_zyx,
            NA_det=NA_DET,
            lambda_ill=LAMBDA_ILL,
            pixel_size=pixel_size,
        )

    if mov_z_index is None:
        mov_z_index = focus_from_transverse_band(
            mov_zyx,
            NA_det=NA_DET,
            lambda_ill=LAMBDA_ILL,
            pixel_size=pixel_size,
        )
 

    transform = np.eye(4)
    transform[0, 3] = mov_z_index - ref_z_index

    return Transform(matrix=transform)



def estimate_tzyx(
    fov: str,
    t: int,
    ref_zyx: np.ndarray,
    mov_zyx: np.ndarray,
    pixel_size: tuple[float, float, float],
    verbose: bool = False,
    output_dirpath: Path = None,
) -> Transform:
    """
    Estimate the z-focus for each timepoint and channel.

    Parameters
    ----------
    input_position_dirpath : Path
        Path to the input position directory.
    input_channel_indices : Tuple[int, ...]
        Indices of the channels to process.
    center_crop_xy : list[int, int]
        Size of the crop in the XY plane.
    output_path_focus_csv : Path
        Path to the output focus CSV file.
    output_path_transform : Path
        Path to the output transform file.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    None
    """
    output_dirpath_fov_t = output_dirpath / fov 
    output_dirpath_fov_t.mkdir(parents=True, exist_ok=True)

    transform = estimate(
        ref_zyx=ref_zyx,
        mov_zyx=mov_zyx,
        pixel_size=pixel_size,
    )
    np.save(output_dirpath_fov_t / f"{t}.npy", transform.matrix)
  


# def get_mean_z_positions(
#     dataframe_path: Path,
#     verbose: bool = False,
#     method: Literal["mean", "median"] = "mean",
# ) -> None:
#     """
#     Get the mean or median z-focus for each timepoint.

#     Parameters
#     ----------
#     dataframe_path : Path
#         Path to the input focus CSV file.
#     verbose : bool
#         If True, print verbose output.
#     method : Literal["mean", "median"]
#         Method to use for averaging the z-focus.

#     Returns
#     -------
#     np.ndarray
#         Array of the mean or median z-focus for each timepoint.
#     """
#     df = pd.read_csv(dataframe_path)

#     df = df.sort_values("time_idx")

#     # When focus finding fails, it may return 0, which here is replaced with NaN
#     # before calculating the mean focus index per position
#     df["focus_idx"] = df["focus_idx"].replace(0, np.nan)

#     # Get the mean of positions for each time point
#     if method == "mean":
#         average_focus_idx = df.groupby("time_idx")["focus_idx"].mean().reset_index()
#     elif method == "median":
#         average_focus_idx = df.groupby("time_idx")["focus_idx"].median().reset_index()

#     if verbose:
#         import matplotlib.pyplot as plt

#         plt.plot(average_focus_idx["focus_idx"], linestyle="--", label="mean of all positions")
#         plt.xlabel("Time index")
#         plt.ylabel("Focus index")
#         plt.ylim(0, 100)
#         plt.legend()
#         plt.savefig(dataframe_path.parent / "z_drift.png")

#     return average_focus_idx["focus_idx"].values


# def estimate_z_stabilization(
#     input_position_dirpaths: list[Path],
#     output_folder_path: Path,
#     focus_finding_settings: FocusFindingSettings,
#     channel_index: int,
#     sbatch_filepath: Optional[Path] = None,
#     cluster: str = "local",
#     verbose: bool = False,
#     estimate_z_index: bool = False,
# ) -> dict[str, list[ArrayLike]]:
#     """
#     Estimate the z stabilization for a list of positions.
#     Parameters
#     ----------
#     input_position_dirpaths : list[Path]
#         Paths to the input position directories.
#     output_folder_path : Path
#         Path to the output folder.
#     focus_finding_settings : FocusFindingSettings
#         Settings for the focus finding.
#     channel_index : int
#         Index of the channel to process.
#     sbatch_filepath : Path
#         Path to the sbatch file.
#     cluster : str
#         Cluster to use.
#     verbose : bool
#         If True, print verbose output.
#     estimate_z_index : bool
#         If True, estimate the z index and save the focus csv without saving the transforms (for xy stabilization).

#     Returns
#     -------
#     dict[str, list[ArrayLike]]
#         Dictionary of the z stabilization for each position.
#     """

#     output_folder_path.mkdir(parents=True, exist_ok=True)
#     slurm_out_path = output_folder_path / "slurm_output"
#     slurm_out_path.mkdir(parents=True, exist_ok=True)

#     # Estimate resources from a sample dataset
#     with open_ome_zarr(input_position_dirpaths[0]) as dataset:
#         shape = dataset.data.shape  # (T, C, Z, Y, X)

#     num_cpus, gb_ram_per_cpu = estimate_resources(
#         shape=shape, ram_multiplier=16, max_num_cpus=16
#     )

#     # Prepare SLURM arguments
#     slurm_args = {
#         "slurm_job_name": "estimate_focus_z",
#         "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
#         "slurm_cpus_per_task": num_cpus,
#         "slurm_array_parallelism": 100,
#         "slurm_time": 30,
#         "slurm_partition": "preempted",
#     }

#     if sbatch_filepath:
#         slurm_args.update(sbatch_to_submitit(sbatch_filepath))

#     # Submitit executor
#     executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
#     executor.update_parameters(**slurm_args)

#     click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")
#     output_folder_focus_path = output_folder_path / "z_focus_positions"
#     output_folder_focus_path.mkdir(parents=True, exist_ok=True)

#     output_transforms_path = output_folder_path / "z_transforms"
#     output_transforms_path.mkdir(parents=True, exist_ok=True)

#     # Submit jobs
#     jobs = []

#     with submitit.helpers.clean_env(), executor.batch():
#         for input_position_dirpath in input_position_dirpaths:
#             job = executor.submit(
#                 estimate_tzyx,
#                 input_position_dirpath=input_position_dirpath,
#                 input_channel_indices=(channel_index,),
#                 center_crop_xy=focus_finding_settings.center_crop_xy,
#                 output_path_focus_csv=output_folder_focus_path,
#                 output_path_transform=output_transforms_path,
#                 verbose=verbose,
#             )
#             jobs.append(job)

#     # Save job IDs
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     log_path = slurm_out_path / f"job_ids_{timestamp}.log"
#     with open(log_path, "w") as log_file:
#         for job in jobs:
#             log_file.write(f"{job.job_id}\n")

#     wait_for_jobs_to_finish(jobs)

#     # Load the focus CSV files and concatenate them
#     focus_csvs_path = list(output_folder_focus_path.glob("*.csv"))
#     if len(focus_csvs_path) != len(input_position_dirpaths):
#         click.echo(
#             f"Warning: {len(focus_csvs_path)} focus CSV files found for {len(input_position_dirpaths)} input data paths."
#         )
#     df = pd.concat([pd.read_csv(f) for f in focus_csvs_path])

#     # Check if the existing focus CSV file exists
#     if Path(output_folder_path / "positions_focus.csv").exists():
#         click.echo("Using existing focus CSV file.")
#         df_old = pd.read_csv(output_folder_path / "positions_focus.csv")
#         df = pd.concat([df, df_old])
#         df = df.drop_duplicates(subset=["position", "time_idx"])
#     df = df.sort_values(["position", "time_idx"])
#     df.to_csv(output_folder_path / "positions_focus.csv", index=False)

#     # Remove the output temporary folder
#     shutil.rmtree(output_folder_focus_path)

#     if estimate_z_index:
#         shutil.rmtree(output_transforms_path)
#         return

#     if focus_finding_settings.average_across_wells:
#         z_drift_offsets = get_mean_z_positions(
#             dataframe_path=output_folder_path / "positions_focus.csv",
#             method=focus_finding_settings.average_across_wells_method,
#             verbose=verbose,
#         )

#         # Initialize the z-focus shift
#         z_focus_shift = [np.eye(4)]
#         # get first non-zero focus index
#         z_val = next((v for v in z_drift_offsets if v != 0), None)
#         if z_val is None:
#             raise ValueError(
#                 "Z index of focus reference is None, z_drift_offsets contains only zeros"
#             )

#         transform = {}

#         # Compute the z-focus shift for each timepoint
#         for z_val_next in z_drift_offsets[1:]:
#             # Set the translation components of the transform
#             shift = np.eye(4)
#             shift[0, 3] = z_val_next - z_val
#             z_focus_shift.append(shift)
#         transform["average"] = np.array(z_focus_shift).tolist()

#         if verbose:
#             click.echo(f"Saving z focus shift matrices to {output_folder_path}")
#             np.save(output_folder_path / "z_focus_shift.npy", transform["average"])

#         return transform
#     else:
#         # Load the transforms
#         transforms_paths = list(output_transforms_path.glob("*.npy"))
#         fov_transforms = {}

#         for file_path in transforms_paths:
#             transform = np.load(file_path).tolist()
#             fov_filename = file_path.stem
#             fov_transforms[fov_filename] = transform

#         # Remove the output temporary folder
#         shutil.rmtree(output_transforms_path)

#     return fov_transforms
