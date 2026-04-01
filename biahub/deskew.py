from pathlib import Path
from typing import List

import click
import numpy as np
import submitit
import torch
from scipy.ndimage import binary_dilation

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from monai.transforms.spatial.array import Affine

from biahub.cli import utils
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.settings import DeskewSettings

# Needed for multiprocessing with GPUs
# https://github.com/pytorch/pytorch/issues/40403#issuecomment-1422625325
torch.multiprocessing.set_start_method('spawn', force=True)


def _average_n_slices(data, average_window_width=1):
    """Average an array over its first axis

    Parameters
    ----------
    data : np.array

    average_window_width : int, optional
        Averaging window applied to the first axis.

    Returns
    -------
    data_averaged : np.array

    """
    # If first dimension isn't divisible by average_window_width, pad it
    remainder = data.shape[0] % average_window_width
    if remainder > 0:
        padding_width = average_window_width - remainder
        data = np.pad(data, [(0, padding_width)] + [(0, 0)] * (data.ndim - 1), mode='edge')

    # Reshape then average over the first dimension
    new_shape = (data.shape[0] // average_window_width, average_window_width) + data.shape[1:]
    data_averaged = np.mean(data.reshape(new_shape), axis=1)

    return data_averaged


def _get_averaged_shape(deskewed_data_shape: tuple, average_window_width: int) -> tuple:
    """
    Compute the shape of the data returned from `_average_n_slices` function.

    Parameters
    ----------
    deskewed_data_shape : tuple
        Shape of the original data before averaging.

    average_window_width : int
        Averaging window applied to the first axis.

    Returns
    -------
    averaged_shape : tuple
        Shape of the data returned from `_average_n_slices`.
    """
    averaged_shape = (
        int(np.ceil(deskewed_data_shape[0] / average_window_width)),
    ) + deskewed_data_shape[1:]
    return averaged_shape


def _get_transform_matrix(ls_angle_deg: float, px_to_scan_ratio: float):
    """
    Compute the 4x4 affine matrix that maps oblique light-sheet coordinates
    to a deskewed (coverslip-aligned) coordinate system.

    Parameters
    ----------
    ls_angle_deg : float
        Light-sheet angle relative to the optical axis, in degrees.
    px_to_scan_ratio : float
        Ratio of camera pixel spacing to scan step size in object space.

    Returns
    -------
    matrix : np.ndarray, shape (4, 4)
        Affine transformation matrix for use with MONAI's Affine transform.
    """
    ct = np.cos(ls_angle_deg * np.pi / 180)

    matrix = np.array(
        [
            [
                -px_to_scan_ratio * ct,
                0,
                px_to_scan_ratio,
                0,
            ],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    return matrix


def _get_deskewed_data_shape(
    raw_data_shape: tuple,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    keep_overhang: bool,
    average_n_slices: int = 1,
    pixel_size_um: float = 1,
):
    """Get the shape of the deskewed data set and its voxel size
    Parameters
    ----------
    raw_data_shape : tuple
        Shape of the raw data, must be len = 3
    ls_angle_deg : float
        Angle of the light sheet relative to the optical axis, in degrees
    px_to_scan_ratio : float
        Ratio of the pixel size to light sheet scan step
    keep_overhang : bool, optional
        If true, the shape of the whole volume within the tilted parallelepiped
        will be returned
        If false, the shape of the deskewed volume within a cuboid region will
        be returned
    average_n_slices : int, optional
        after deskewing, averages every n slices (default = 1 applies no averaging)
    pixel_size_um : float, optional
        Pixel size in micrometers. If not provided, a default value of 1 will be
        used and the returned voxel size will represent a voxel scale
    Returns
    -------
    output_shape : tuple
        Output shape of the deskewed data in ZYX order
    voxel_size : tuple
        Size of the deskewed voxels in micrometers. If the default
        pixel_size_um = 1 is used this parameter will represent the voxel scale
    """

    # Trig
    theta = ls_angle_deg * np.pi / 180
    st = np.sin(theta)
    ct = np.cos(theta)

    # Prepare transforms
    Z, Y, X = raw_data_shape

    if keep_overhang:
        Xp = int(np.ceil((Z / px_to_scan_ratio) + (Y * ct)))
    else:
        Xp = int(np.ceil((Z / px_to_scan_ratio) - (Y * ct)))
        if Xp <= 0:
            raise ValueError(
                f"Dataset contains only overhang when keep_overhang=False. "
                f"Computed Xp={Xp} <= 0. Either set keep_overhang=True or use a dataset "
                f"with non-overhang content."
            )

    output_shape = (Y, X, Xp)
    voxel_size = (average_n_slices * st * pixel_size_um, pixel_size_um, pixel_size_um)

    averaged_output_shape = _get_averaged_shape(output_shape, average_n_slices)

    return averaged_output_shape, voxel_size


def _fill_overhang_with_mean(
    data: np.ndarray,
    dilation_iterations: int = 3,
    debug_plot_path: Path = None,
) -> np.ndarray:
    """Replace zero-padded overhang regions with the mean of the valid signal.

    After deskewing with padding_mode="zeros", overhang voxels are exactly 0.
    Bilinear interpolation at the boundary produces a gradient from signal to 0,
    so the mask is dilated inward to also cover those blended voxels.

    Parameters
    ----------
    data : np.ndarray
        Deskewed 3D volume with zero-padded overhangs.
    dilation_iterations : int
        Number of binary dilation iterations to grow the zero-mask inward,
        capturing interpolation artifacts at the overhang boundary.
    debug_plot_path : Path, optional
        If provided, saves a diagnostic figure showing the masks and result.

    Returns
    -------
    filled : np.ndarray
        Volume with overhang regions replaced by the mean of the valid signal.
    """
    zero_mask = data == 0
    dilated_mask = binary_dilation(zero_mask, iterations=dilation_iterations)
    valid_mean = data[~dilated_mask].mean()
    filled = data.copy()
    filled[dilated_mask] = valid_mean

    if debug_plot_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mid_z = data.shape[0] // 2
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].imshow(data[mid_z], cmap="gray")
        axes[0, 0].set_title(f"Deskewed (z={mid_z})")

        axes[0, 1].imshow(zero_mask[mid_z], cmap="gray")
        axes[0, 1].set_title("Zero mask")

        axes[1, 0].imshow(dilated_mask[mid_z], cmap="gray")
        axes[1, 0].set_title(f"Dilated mask (iterations={dilation_iterations})")

        im = axes[1, 1].imshow(filled[mid_z], cmap="gray")
        axes[1, 1].set_title(f"Filled (mean={valid_mean:.1f})")

        fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
        fig.tight_layout()
        fig.savefig(debug_plot_path, dpi=150)
        plt.close(fig)
        print(f"Overhang mask debug plot saved to {debug_plot_path}")

    return filled


def deskew_zyx(
    raw_data: np.ndarray,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    keep_overhang: bool,
    average_n_slices: int = 1,
    overhang_fill: str = "zero",
    device='cpu',
    debug_plot_path: Path = None,
) -> np.ndarray:
    """Deskew a single ZYX volume from oblique light-sheet coordinates to
    coverslip-aligned coordinates using an affine transform.

    Parameters
    ----------
    raw_data : np.ndarray, shape (Z, Y, X)
        Raw 3D volume where Z is the scanning axis, Y is the tilted axis,
        and X is in the plane of the coverslip.
    ls_angle_deg : float
        Light-sheet angle relative to the optical axis, in degrees.
    px_to_scan_ratio : float
        Ratio of camera pixel spacing to scan step size in object space.
        E.g. pixel_size=6.5um, mag=1.4x40 -> spacing=0.116um, scan=0.3um
        -> px_to_scan_ratio = 0.116/0.3 = 0.386.
    keep_overhang : bool
        If True, output the full volume within the tilted parallelepiped.
        If False, crop to the cuboid region without overhang.
    average_n_slices : int, optional
        Number of Z slices to average after deskewing (default=1, no averaging).
    overhang_fill : str, optional
        How to fill overhang regions: "zero" keeps them as zeros (default),
        "mean" replaces them with the mean of the valid signal.
    device : str, optional
        Torch device for computation ('cpu' or 'cuda'). Default is 'cpu'.
    debug_plot_path : Path, optional
        If provided, saves an overhang mask diagnostic plot to this path.

    Returns
    -------
    deskewed_data : np.ndarray, shape (Z', Y', X')
        Deskewed volume where Z' is normal to the coverslip, Y' and X' are
        in the coverslip plane.
    """
    # Prepare transforms
    matrix = _get_transform_matrix(
        ls_angle_deg,
        px_to_scan_ratio,
    )

    output_shape, _ = _get_deskewed_data_shape(
        raw_data.shape, ls_angle_deg, px_to_scan_ratio, keep_overhang
    )

    # convert to tensor on GPU
    # convert raw_data to int32 if it is uint16
    raw_data_tensor = torch.from_numpy(raw_data.astype(np.float32)).to(device)

    # Returns callable
    affine_func = Affine(affine=matrix, padding_mode="zeros", image_only=True)

    # affine_func accepts CZYX array, so for ZYX input we need [None] and for ZYX output we need [0]
    deskewed_data = affine_func(
        raw_data_tensor[None], mode="bilinear", spatial_size=output_shape
    )[0]

    # to numpy array on CPU
    deskewed_data = deskewed_data.cpu().numpy()

    # Apply averaging
    averaged_deskewed_data = _average_n_slices(
        deskewed_data, average_window_width=average_n_slices
    )

    # Fill overhang regions after averaging
    if keep_overhang and overhang_fill == "mean":
        averaged_deskewed_data = _fill_overhang_with_mean(
            averaged_deskewed_data, debug_plot_path=debug_plot_path
        )

    return averaged_deskewed_data


def deskew_czyx(
    input_position_path: Path,
    output_dirpath: Path,
    t_idx: int,
    settings: DeskewSettings,
) -> None:
    """Deskew all channels for a single timepoint and write to the output zarr.

    This is a top-level function so it can be pickled by submitit for SLURM jobs.
    Each SLURM job processes one (position, timepoint) pair.

    Parameters
    ----------
    input_position_path : Path
        Path to the input OME-Zarr position (e.g. input.zarr/A/1/fov0).
    output_dirpath : Path
        Root path of the output OME-Zarr store.
    t_idx : int
        Timepoint index to process.
    settings : DeskewSettings
        Deskew parameters (angles, ratios, averaging, etc.).
    """
    position_key = input_position_path.parts[-3:]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Deskewing position={'/'.join(position_key)} t={t_idx} on {device}")

    with open_ome_zarr(input_position_path, mode="r") as input_dataset:
        _, C, _, _, _ = input_dataset.data.shape


    plot_dirpath = output_dirpath.parent / "deskew_debug_plots"
    plot_dirpath.mkdir(parents=True, exist_ok=True)

    output_path = output_dirpath / Path(*position_key)
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        for c in range(C):
            data = np.asarray(input_dataset.data[t_idx, c])
            deskewed_data = deskew_zyx(
                data,
                settings.ls_angle_deg,
                settings.px_to_scan_ratio,
                settings.keep_overhang,
                settings.average_n_slices,
                overhang_fill=settings.overhang_fill,
                device=device,
                debug_plot_path=plot_dirpath / f"deskew_debug_plot_tfov_{t_idx}_c_{c}.png",
            )
            output_dataset[0][t_idx, c] = deskewed_data
    print(f"Done position={'/'.join(position_key)} t={t_idx}")

def deskew(
    input_position_dirpaths: List[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True,
):
    """Deskew all positions across T and C axes, submitting one SLURM job
    per (position, timepoint) pair.

    Reads deskew parameters from a YAML config file, creates the output
    OME-Zarr store, and submits parallel jobs via submitit.

    Parameters
    ----------
    input_position_dirpaths : list of str
        Paths to input OME-Zarr positions (e.g. input.zarr/A/1/fov0).
    config_filepath : Path
        Path to the YAML file with DeskewSettings.
    output_dirpath : str
        Root path for the output OME-Zarr store.
    sbatch_filepath : str, optional
        Path to an sbatch file to override default SLURM parameters.
    local : bool, optional
        If True, run locally instead of submitting to SLURM.
    monitor : bool, optional
        If True, monitor job progress after submission.
    """
    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)

    slurm_out_path = output_dirpath.parent / "slurm_output"

    # Get the deskewing parameters
    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        channel_names = input_dataset.channel_names
        T, C, Z, Y, X = input_dataset.data.shape

    settings = yaml_to_model(config_filepath, DeskewSettings)
    deskewed_shape, voxel_size = _get_deskewed_data_shape(
        (Z, Y, X),
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        settings.keep_overhang,
        settings.average_n_slices,
        settings.pixel_size_um,
    )

    # Create a zarr store output to mirror the input
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        channel_names=channel_names,
        shape=(T, C) + deskewed_shape,
        scale=(1, 1) + voxel_size,
        version="0.5",
        dtype=np.float32,
    )

    # Estimate resources
    num_cpus, gb_ram = estimate_resources(
        shape=(T, C, Z, Y, X), ram_multiplier=16, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "deskew",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 60,
        "slurm_partition": "preempted",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    n_positions = len(input_position_dirpaths)
    n_jobs = n_positions * T
    click.echo(
        f"Deskew: {n_positions} position(s) x {T} timepoints = {n_jobs} jobs "
        f"({cluster} mode)"
    )
    click.echo(f"SLURM params: {slurm_args}")
    click.echo(f"Output: {output_dirpath}")
    click.echo(f"Deskewed shape per position: (T={T}, C={C}) + {deskewed_shape}")

    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path in input_position_dirpaths:
            for t in range(T):
                jobs.append(
                    executor.submit(
                        deskew_czyx,
                        input_position_path,
                        output_dirpath,
                        t,
                        settings,
                    )
            )

    job_ids = [job.job_id for job in jobs]
    click.echo(f"Submitted {len(job_ids)} jobs: {job_ids[0]} .. {job_ids[-1]}")

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))
    click.echo(f"Job IDs saved to {log_path}")

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("deskew")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
@monitor()
def deskew_cli(
    input_position_dirpaths: List[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True,
):
    """
    Deskew a single position across T and C axes using a configuration file
    generated by estimate_deskew.py

    >> biahub deskew \
        -i ./input.zarr/*/*/* \
        -c ./deskew_params.yml \
        -o ./output.zarr
    """
    deskew(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        local=local,
        monitor=monitor,
    )


if __name__ == "__main__":
    deskew_cli()
