from pathlib import Path

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


def _average_n_slices(data: np.ndarray, average_window_width: int = 1) -> np.ndarray:
    """
    Average an array over its first axis.

    Parameters
    ----------
    data : np.ndarray
        Input array to average.
    average_window_width : int, optional
        Averaging window applied to the first axis, by default 1

    Returns
    -------
    np.ndarray
        Averaged array with shape (data.shape[0] // average_window_width, ...) + data.shape[1:]
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


def _get_averaged_shape(
    deskewed_data_shape: tuple[int, ...], average_window_width: int
) -> tuple[int, ...]:
    """
    Compute the shape of the data returned from `_average_n_slices` function.

    Parameters
    ----------
    deskewed_data_shape : tuple[int, ...]
        Shape of the original data before averaging.
    average_window_width : int
        Averaging window applied to the first axis.

    Returns
    -------
    tuple[int, ...]
        Shape of the data returned from `_average_n_slices`.
    """
    averaged_shape = (
        int(np.ceil(deskewed_data_shape[0] / average_window_width)),
    ) + deskewed_data_shape[1:]
    return averaged_shape


def _get_transform_matrix(ls_angle_deg: float, px_to_scan_ratio: float) -> np.ndarray:
    """
    Compute affine transformation matrix used to deskew data.

    Parameters
    ----------
    ls_angle_deg : float
        Light sheet angle in degrees.
    px_to_scan_ratio : float
        Ratio of pixel size to scan step size.

    Returns
    -------
    np.ndarray
        4x4 affine transformation matrix.
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
    raw_data_shape: tuple[int, int, int],
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    keep_overhang: bool,
    average_n_slices: int = 1,
    pixel_size_um: float = 1,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """
    Get the shape of the deskewed data set and its voxel size.

    Parameters
    ----------
    raw_data_shape : tuple[int, int, int]
        Shape of the raw data in ZYX order.
    ls_angle_deg : float
        Angle of the light sheet relative to the optical axis, in degrees.
    px_to_scan_ratio : float
        Ratio of the pixel size to light sheet scan step.
    keep_overhang : bool
        If True, the shape of the whole volume within the tilted parallelepiped
        will be returned. If False, the shape of the deskewed volume within a
        cuboid region will be returned.
    average_n_slices : int, optional
        After deskewing, averages every n slices, by default 1 (applies no averaging).
    pixel_size_um : float, optional
        Pixel size in micrometers, by default 1. If the default value is used,
        the returned voxel size will represent a voxel scale.

    Returns
    -------
    tuple[tuple[int, int, int], tuple[float, float, float]]
        Tuple containing:
        - output_shape : Output shape of the deskewed data in ZYX order
        - voxel_size : Size of the deskewed voxels in micrometers. If the default
          pixel_size_um = 1 is used, this parameter will represent the voxel scale.

    Raises
    ------
    ValueError
        If keep_overhang=False and the computed Xp <= 0, indicating the dataset
        contains only overhang.
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


def deskew(
    raw_data: np.ndarray,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    keep_overhang: bool,
    average_n_slices: int = 1,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Deskew fluorescence data from the mantis microscope.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw data from the mantis microscope with ndim == 3:
        - axis 0 corresponds to the scanning axis
        - axis 1 corresponds to the "tilted" axis
        - axis 2 corresponds to the axis in the plane of the coverslip
    ls_angle_deg : float
        Angle of light sheet with respect to the optical axis in degrees.
    px_to_scan_ratio : float
        Ratio of (pixel spacing / scan spacing) in object space.
        For example, if camera pixels = 6.5 um and mag = 1.4*40, then the pixel
        spacing is 6.5/(1.4*40) = 0.116 um. If the scan spacing is 0.3 um, then
        px_to_scan_ratio = 0.116 / 0.3 = 0.386.
    keep_overhang : bool
        If True, compute the whole volume within the tilted parallelepiped.
        If False, only compute the deskewed volume within a cuboid region.
    average_n_slices : int, optional
        After deskewing, averages every n slices, by default 1 (applies no averaging).
    device : str, optional
        Torch device to use for computation, by default 'cpu'.

    Returns
    -------
    np.ndarray
        Deskewed data with ndim == 3:
        - axis 0 is the Z axis, normal to the coverslip
        - axis 1 is the Y axis, input axis 2 in the plane of the coverslip
        - axis 2 is the X axis, the scanning axis
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
    return averaged_deskewed_data


# Adapt ZYX function to CZYX
# Needs to be a top-level function for multiprocessing pickling
def _czyx_deskew_data(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Apply deskewing to CZYX data by processing each channel separately.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (C, Z, Y, X) where C is the channel dimension.
    **kwargs
        Additional keyword arguments passed to `deskew` function.

    Returns
    -------
    np.ndarray
        Deskewed data with shape (C, Z', Y', X') where the spatial dimensions
        may differ from input due to deskewing transformation.
    """
    return deskew(data[0], **kwargs)[None]


@click.command("deskew")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
@monitor()
def deskew_cli(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str | None = None,
    local: bool = False,
    monitor: bool = True,
) -> None:
    """
    Deskew a single position across T and C axes using a configuration file.

    Uses a configuration file generated by estimate_deskew.py to deskew
    fluorescence data from the mantis microscope.

    Parameters
    ----------
    input_position_dirpaths : list[str]
        List of input position directory paths to process.
    config_filepath : Path
        Path to the deskew configuration YAML file.
    output_dirpath : str
        Path to the output zarr directory.
    sbatch_filepath : str | None, optional
        Path to the SLURM batch file for job configuration, by default None
    local : bool, optional
        Whether to run locally instead of on a SLURM cluster, by default False
    monitor : bool, optional
        Whether to monitor job progress, by default True

    Returns
    -------
    None
        Results are written directly to disk in the `output_dirpath`.

    Examples
    --------
    >> biahub deskew \\
        -i ./input.zarr/*/*/* \\
        -c ./deskew_params.yml \\
        -o ./output.zarr
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


if __name__ == "__main__":
    deskew_cli()
