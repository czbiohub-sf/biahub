from pathlib import Path
from typing import List, Literal

import click
import numpy as np
import submitit
import torch

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position
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
    Compute affine transformation matrix used to deskew data.

    Parameters
    ----------
    ls_angle_deg : float
    px_to_scan_ratio : float
    keep_overhang : bool

    Returns
    -------
    matrix : np.array
        Affine transformation matrix.
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


def get_deskewed_data_shape(
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
    device: str = 'cpu',
    average_n_slices: int = 1,
    overhang_fill: Literal["zero", "mean"] = "zero",
    debug_plot_path: Path = None,
) -> np.ndarray:
    """Deskews fluorescence data from the mantis microscope
    Parameters
    ----------
    raw_data : NDArray with ndim == 3
        raw data from the mantis microscope
        - axis 0 corresponds to the scanning axis
        - axis 1 corresponds to the "tilted" axis
        - axis 2 corresponds to the axis in the plane of the coverslip
    ls_angle_deg : float
        angle of light sheet with respect to the optical axis in degrees
    px_to_scan_ratio : float
        (pixel spacing / scan spacing) in object space
        e.g. if camera pixels = 6.5 um and mag = 1.4*40, then the pixel spacing
        is 6.5/(1.4*40) = 0.116 um. If the scan spacing is 0.3 um, then
        px_to_scan_ratio = 0.116 / 0.3 = 0.386
    keep_overhang : bool
        If true, compute the whole volume within the tilted parallelepiped.
        If false, only compute the deskewed volume within a cuboid region.
    average_n_slices : int, optional
        after deskewing, averages every n slices (default = 1 applies no averaging)
    device : str, optional
        torch device to use for computation. Default is 'cpu'.
    Returns
    -------
    deskewed_data : NDArray with ndim == 3
        axis 0 is the Z axis, normal to the coverslip
        axis 1 is the Y axis, input axis 2 in the plane of the coverslip
        axis 2 is the X axis, the scanning axis
    """
    # Prepare transforms
    matrix = _get_transform_matrix(
        ls_angle_deg,
        px_to_scan_ratio,
    )

    output_shape, _ = get_deskewed_data_shape(
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


# Adapt ZYX function to CZYX
# Needs to be a top-level function for multiprocessing pickling
def _czyx_deskew_data(data, **kwargs):
    return deskew_zyx(data[0], **kwargs)[None]

def deskew(
    input_position_dirpaths: List[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True,
):
    """
    Deskew a dataset across T and C axes using a configuration file

    Parameters
    ----------
    input_position_dirpaths : List[str]
        List of input position directory paths
    config_filepath : Path
        Path to the configuration file
    output_dirpath : str
        Path to the output directory
    sbatch_filepath : str, optional
        Path to the SLURM batch file
    local : bool, optional
        Whether to run locally or submit to SLURM
    monitor : bool, optional
        Whether to monitor the jobs

    Returns
    -------
    None
    """
        # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)

    slurm_out_path = output_dirpath.parent / "slurm_output"

    # Handle single position or wildcard filepath
    output_position_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)

    # Get the deskewing parameters
    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        channel_names = input_dataset.channel_names
        T, C, Z, Y, X = input_dataset.data.shape

    settings = yaml_to_model(config_filepath, DeskewSettings)
    deskewed_shape, voxel_size = get_deskewed_data_shape(
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
    )

    deskew_args = {
        'ls_angle_deg': settings.ls_angle_deg,
        'px_to_scan_ratio': settings.px_to_scan_ratio,
        'keep_overhang': settings.keep_overhang,
        'average_n_slices': settings.average_n_slices,
        'overhang_fill': settings.overhang_fill,
        'extra_metadata': {'deskew': settings.model_dump()},
    }

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

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo('Submitting SLURM jobs...')

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    _czyx_deskew_data,
                    input_position_path,
                    output_position_path,
                    num_processes=slurm_args["slurm_cpus_per_task"],
                    **deskew_args,
                )
            )

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

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
