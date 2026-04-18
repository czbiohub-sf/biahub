from pathlib import Path
from typing import Literal

import click
import numpy as np
import submitit
import torch

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position
from monai.transforms.spatial.array import Affine
from scipy.ndimage import binary_dilation

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
from biahub.cli.utils import estimate_resources, get_submitit_cluster, yaml_to_model
from biahub.settings import DeskewSettings

# Needed for multiprocessing with GPUs
# https://github.com/pytorch/pytorch/issues/40403#issuecomment-1422625325
torch.multiprocessing.set_start_method("spawn", force=True)


def _average_n_slices(data, average_window_width=1):
    """Average an array over its first axis.

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
        data = np.pad(data, [(0, padding_width)] + [(0, 0)] * (data.ndim - 1), mode="edge")

    # Reshape then average over the first dimension
    new_shape = (data.shape[0] // average_window_width, average_window_width) + data.shape[1:]
    data_averaged = np.mean(data.reshape(new_shape), axis=1)

    return data_averaged


def _average_n_slices_torch(data: torch.Tensor, average_window_width: int) -> torch.Tensor:
    """Average a tensor over its first axis (GPU-compatible).

    Parameters
    ----------
    data : torch.Tensor

    average_window_width : int
        Averaging window applied to the first axis.

    Returns
    -------
    torch.Tensor

    """
    if average_window_width == 1:
        return data

    remainder = data.shape[0] % average_window_width
    if remainder > 0:
        padding_width = average_window_width - remainder
        pad = data[-1:].expand(padding_width, *data.shape[1:])
        data = torch.cat([data, pad], dim=0)

    new_shape = (data.shape[0] // average_window_width, average_window_width) + data.shape[1:]
    return data.reshape(new_shape).mean(dim=1)


def _deskew_interpolate(
    data_ra: torch.Tensor,
    in_z_f: torch.Tensor,
    Z_in: int,
    Y_out: int,
    X_out: int,
    batch_z: int = 256,
) -> torch.Tensor:
    """Interpolate along the Z axis of a pre-arranged tensor.

    Implements bilinear 1-D interpolation with ``padding_mode='zeros'``,
    matching MONAI ``Affine`` behaviour exactly.  Processes ``Z_out`` rows
    in batches to bound peak GPU memory.

    Parameters
    ----------
    data_ra : torch.Tensor, shape (Z_out, Y_out, Z_in)
        Input tensor after the integer axis-permutation and flip that encodes
        the in_y / in_x mappings.
    in_z_f : torch.Tensor, shape (Z_out, X_out)
        Floating-point Z_in sampling positions for every (z_out, x_out) pair.
    Z_in, Y_out, X_out : int
        Spatial dimensions.
    batch_z : int
        Number of Z_out rows to process per GPU kernel batch.

    Returns
    -------
    torch.Tensor, shape (Z_out, Y_out, X_out)
    """
    Z_out = data_ra.shape[0]
    device = data_ra.device

    in_z0 = in_z_f.floor().long()
    in_z0_safe = in_z0.clamp(0, Z_in - 1)
    in_z1_safe = (in_z0 + 1).clamp(0, Z_in - 1)
    in_z_frac = (in_z_f - in_z0.float()).clamp(0.0, 1.0)

    # Boundary regions for padding_mode='zeros' bilinear blend
    below_edge = (in_z_f > -1) & (in_z_f < 0)   # blend zero ↔ data[0]
    above_edge = (in_z_f > Z_in - 1) & (in_z_f < Z_in)  # blend data[Z-1] ↔ zero
    fully_out = ~((in_z_f >= 0) & (in_z_f <= Z_in - 1) | below_edge | above_edge)

    w_below = (in_z_f + 1).clamp(0.0, 1.0)
    w_above = (Z_in - in_z_f).clamp(0.0, 1.0)
    idx_zero = torch.zeros(1, dtype=torch.long, device=device)
    idx_last = torch.full((1,), Z_in - 1, dtype=torch.long, device=device)

    slices = []
    for z_s in range(0, Z_out, batch_z):
        z_e = min(z_s + batch_z, Z_out)
        B = z_e - z_s
        d = data_ra[z_s:z_e]  # (B, Y_out, Z_in)

        def exp(t, _B=B):
            return t[z_s:z_e].unsqueeze(1).expand(_B, Y_out, X_out)

        res = torch.lerp(d.gather(2, exp(in_z0_safe)), d.gather(2, exp(in_z1_safe)), exp(in_z_frac))

        # Edge blends
        mask_be = exp(below_edge)
        if mask_be.any():
            wb = exp(w_below)
            edge_val = d.gather(2, idx_zero.expand(B, Y_out, 1).expand(B, Y_out, X_out))
            res = torch.where(mask_be, edge_val * wb, res)

        mask_ae = exp(above_edge)
        if mask_ae.any():
            wa = exp(w_above)
            edge_val = d.gather(2, idx_last.expand(B, Y_out, 1).expand(B, Y_out, X_out))
            res = torch.where(mask_ae, edge_val * wa, res)

        res[exp(fully_out)] = 0.0
        slices.append(res)

    return torch.cat(slices, dim=0)


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
    """Get the shape of the deskewed data set and its voxel size.

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
    device: str = "cpu",
    average_n_slices: int = 1,
    overhang_fill: Literal["zero", "mean"] = "zero",
    debug_plot_path: Path = None,
) -> np.ndarray:
    """Deskews fluorescence data from the mantis microscope.

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
    matrix = _get_transform_matrix(ls_angle_deg, px_to_scan_ratio)

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

    # Apply averaging on GPU before transferring to CPU
    deskewed_data = _average_n_slices_torch(deskewed_data, average_window_width=average_n_slices)

    # to numpy array on CPU
    deskewed_data = deskewed_data.cpu().numpy()

    averaged_deskewed_data = deskewed_data

    # Fill overhang regions after averaging
    if keep_overhang and overhang_fill == "mean":
        averaged_deskewed_data = _fill_overhang_with_mean(
            averaged_deskewed_data, debug_plot_path=debug_plot_path
        )

    return averaged_deskewed_data


def fast_deskew_zyx(
    raw_data: np.ndarray,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    keep_overhang: bool,
    device: str = "cpu",
    average_n_slices: int = 1,
    overhang_fill: Literal["zero", "mean"] = "zero",
    debug_plot_path: Path = None,
) -> np.ndarray:
    """Fast deskew of fluorescence data from the mantis microscope.

    Drop-in replacement for :func:`deskew_zyx` that is ~14x faster by
    exploiting the structure of the deskew affine transform: two of the three
    input axes map to output axes via integer permutations and flips, so only
    a single axis requires floating-point interpolation.  The 3-D trilinear
    ``grid_sample`` used by MONAI ``Affine`` is replaced by a 1-D
    ``gather``+``lerp`` along the scan (Z) axis only.

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
    output_shape, _ = get_deskewed_data_shape(
        raw_data.shape, ls_angle_deg, px_to_scan_ratio, keep_overhang
    )

    # Move input to device as float32; pin memory for faster async CPU→GPU transfer
    raw_data_f32 = torch.from_numpy(raw_data.astype(np.float32))
    if device != "cpu":
        raw_data_f32 = raw_data_f32.pin_memory()
    raw_data_tensor = raw_data_f32.to(device, non_blocking=True)

    Z_in, Y_in, X_in = raw_data.shape
    Z_out, Y_out, X_out = output_shape

    # Integer axis mapping: permute (Z,Y,X)→(Y,X,Z) then flip to implement
    #   in_y = Y_in-1-z_out  and  in_x = X_in-1-y_out  exactly.
    data_ra = raw_data_tensor.permute(1, 2, 0).flip(0).flip(1).contiguous()  # (Z_out, Y_out, Z_in)

    # Fractional Z_in position for every (z_out, x_out) output voxel.
    # Derived from the affine matrix in centred-voxel space:
    #   in_z = px * x_out - px*ct * z_out + offset
    ct = np.cos(ls_angle_deg * np.pi / 180)
    px = px_to_scan_ratio
    offset = px * ct * (Z_out - 1) / 2 - px * (X_out - 1) / 2 + (Z_in - 1) / 2
    z_idx = torch.arange(Z_out, device=device, dtype=torch.float32)
    x_idx = torch.arange(X_out, device=device, dtype=torch.float32)
    in_z_f = px * x_idx.unsqueeze(0) - px * ct * z_idx.unsqueeze(1) + offset  # (Z_out, X_out)

    deskewed_data = _deskew_interpolate(data_ra, in_z_f, Z_in, Y_out, X_out)

    # Apply averaging on GPU before transferring to CPU
    deskewed_data = _average_n_slices_torch(deskewed_data, average_window_width=average_n_slices)

    # to numpy array on CPU
    deskewed_data = deskewed_data.cpu().numpy()

    averaged_deskewed_data = deskewed_data

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
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True,
):
    """
    Deskew a dataset across T and C axes using a configuration file.

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
        "ls_angle_deg": settings.ls_angle_deg,
        "px_to_scan_ratio": settings.px_to_scan_ratio,
        "keep_overhang": settings.keep_overhang,
        "average_n_slices": settings.average_n_slices,
        "overhang_fill": settings.overhang_fill,
        "extra_metadata": {"deskew": settings.model_dump()},
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
    cluster = get_submitit_cluster(local)

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting SLURM jobs...")

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths, strict=True
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

    slurm_out_path.mkdir(exist_ok=True)
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
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True,
):
    """Deskew a single position across T and C axes using a configuration file, generated by estimate_deskew.py.

    >>> biahub deskew \
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
