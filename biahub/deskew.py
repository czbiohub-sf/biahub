import warnings

from pathlib import Path
from typing import Literal

import click
import numpy as np
import submitit
import torch
import torch.nn.functional as F  # noqa: N812

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position
from monai.transforms.spatial.array import Affine
from scipy.ndimage import binary_dilation

from biahub.cli import utils
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
    resolve_ome_zarr_version,
    yaml_to_model,
)
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


def _rearrange_axes(data: torch.Tensor) -> torch.Tensor:
    """Apply the integer part of the deskew affine: axis permutation and flips.

    Maps input (Z_scan, Y_tilt, X_coverslip) to output (Z_out, Y_out, Z_in)
    where Z_out indexes the output Z axis (normal to coverslip) and the last
    axis is the scan axis that still requires fractional interpolation.

    The mapping implemented is:
        in_y = Y_in - 1 - z_out   (flip along axis 0 after permute)
        in_x = X_in - 1 - y_out   (flip along axis 1 after permute)
    """
    return data.permute(1, 2, 0).flip([0, 1]).contiguous()


def _build_deskew_grid(
    Z_out_full: int,
    X_out: int,
    Z_in: int,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    average_n_slices: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the 2-D sampling grid for `F.grid_sample`.

    Returns a grid of shape `(Z_avg, N, X_out, 2)` where `N` is
    `average_n_slices` and `Z_avg = ceil(Z_out_full / N)`.  The two
    coordinates per sample are:

    * **W** (`grid[..., 0]`): the normalised scan-axis (Z_in) position,
      derived from `in_z = px * x_out - px * cos(θ) * z_out + offset`.
    * **H** (`grid[..., 1]`): indexes the N grouped sub-slices that will
      be averaged after interpolation.
    """
    N = average_n_slices
    Z_avg = int(np.ceil(Z_out_full / N))

    ct = np.cos(ls_angle_deg * np.pi / 180)
    px = px_to_scan_ratio
    offset = px * ct * (Z_out_full - 1) / 2 - px * (X_out - 1) / 2 + (Z_in - 1) / 2

    # z_out index for each (avg_slice a, sub-slice k): z_out = a*N + k
    a_idx = torch.arange(Z_avg, device=device, dtype=torch.float32)
    k_idx = torch.arange(N, device=device, dtype=torch.float32)
    x_idx = torch.arange(X_out, device=device, dtype=torch.float32)
    z_out_all = a_idx.unsqueeze(1) * N + k_idx.unsqueeze(0)  # (Z_avg, N)

    # W coordinate: in_z normalised to [-1, 1] for align_corners=True
    in_z_f = px * x_idx - px * ct * z_out_all.unsqueeze(2) + offset  # (Z_avg, N, X_out)
    in_z_norm = 2.0 * in_z_f / (Z_in - 1) - 1.0

    # H coordinate: point to each of the N grouped sub-slice positions
    h_norm = (2.0 * k_idx / max(N - 1, 1) - 1.0) if N > 1 else torch.zeros(1, device=device)
    h_grid = h_norm.view(1, N, 1).expand(Z_avg, N, X_out)

    return torch.stack([in_z_norm, h_grid], dim=-1)  # (Z_avg, N, X_out, 2)


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
    debug_plot_path: Path | None = None,
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


def _fill_overhang_torch(
    data: torch.Tensor,
    fill_value: float | None = None,
    dilation_iterations: int = 3,
) -> torch.Tensor:
    """Replace zero-padded overhang regions on GPU.

    Uses `F.max_pool3d` for binary dilation instead of `scipy.ndimage`.
    The structuring element is a 3x3x3 cube (26-connectivity), which is
    slightly more aggressive than scipy's default cross (6-connectivity).

    Parameters
    ----------
    data : torch.Tensor
        Deskewed 3D volume with zero-padded overhangs.
    fill_value : float or None
        Value to fill overhang regions with.  If None, the mean of the valid
        (non-overhang) signal is used.
    dilation_iterations : int
        Number of binary dilation iterations to grow the zero-mask inward,
        capturing interpolation artifacts at the overhang boundary.
    """
    mask = (data == 0).float().unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
    for _ in range(dilation_iterations):
        mask = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    dilated_mask = mask.squeeze(0).squeeze(0) > 0.5  # back to bool (Z, Y, X)

    if fill_value is None:
        fill_value = data[~dilated_mask].mean()
    return torch.where(dilated_mask, fill_value, data)


def deskew_zyx(
    raw_data: np.ndarray,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    keep_overhang: bool,
    device: str = "cpu",
    average_n_slices: int = 1,
    overhang_fill: Literal["zero", "mean"] = "zero",
    debug_plot_path: Path | None = None,
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
    overhang_fill : "zero" or "mean", optional
        How to fill overhang regions (only used when keep_overhang=True).
        "mean" replaces with the mean of the valid signal, "zero" leaves the
        zero-padded overhang as-is. Default is "zero".
    debug_plot_path : Path, optional
        If provided, saves a diagnostic figure showing the masks and result.

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
    deskewed_data = _average_n_slices_torch(
        deskewed_data, average_window_width=average_n_slices
    )

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
    raw_data: torch.Tensor,
    ls_angle_deg: float,
    px_to_scan_ratio: float,
    keep_overhang: bool,
    average_n_slices: int = 1,
    overhang_fill: Literal["mean"] | float = 0,
) -> torch.Tensor:
    """Fast deskew of fluorescence data from the mantis microscope.

    Exploits the structure of the deskew affine: two of the three input axes
    map to output axes via integer permutations/flips (no interpolation
    needed), and only the scan axis (Z) requires fractional resampling.
    A 2-D `grid_sample` with Y_out as the channel dimension replaces the
    full 3-D trilinear `grid_sample` used by MONAI `Affine`.

    Parameters
    ----------
    raw_data : torch.Tensor with ndim == 3, dtype float32
        raw data from the mantis microscope, already on the target device
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
    overhang_fill : "mean" or float, optional
        How to fill overhang regions (only used when keep_overhang=True).
        "mean" replaces with the mean of the valid signal, or pass a numeric
        value (e.g. 0 to leave as-is, or 100) to fill with that constant.
        Default is 0.

    Returns
    -------
    deskewed_data : torch.Tensor with ndim == 3
        axis 0 is the Z axis, normal to the coverslip
        axis 1 is the Y axis, input axis 2 in the plane of the coverslip
        axis 2 is the X axis, the scanning axis
    """
    device = raw_data.device
    Z_in = raw_data.shape[0]

    # Un-averaged output shape (average_n_slices defaults to 1)
    output_shape, _ = get_deskewed_data_shape(
        raw_data.shape, ls_angle_deg, px_to_scan_ratio, keep_overhang
    )
    Z_out_full, _, X_out = output_shape  # Z_out_full = Y_in

    N = average_n_slices
    Z_avg = int(np.ceil(Z_out_full / N))

    # Integer axis mapping: (Z_scan, Y_tilt, X_coverslip) → (Z_out, Y_out, Z_in)
    data_ra = _rearrange_axes(raw_data)

    # Pad z_out dim to be divisible by N (edge replication)
    pad_n = Z_avg * N - Z_out_full
    if pad_n > 0:
        data_ra = torch.cat([data_ra, data_ra[-1:].expand(pad_n, -1, -1)], dim=0)

    # Reshape to (Batch=Z_avg, C=Y_out, H=N, W=Z_in) — consecutive z_out
    # slices are grouped into H so averaging becomes a mean over dim 2.
    Y_out = data_ra.shape[1]
    data_ra = data_ra.reshape(Z_avg, N, Y_out, Z_in).permute(0, 2, 1, 3)

    # Fractional scan-axis interpolation via 2-D grid_sample
    grid = _build_deskew_grid(
        Z_out_full, X_out, Z_in, ls_angle_deg, px_to_scan_ratio, N, device
    )

    deskewed = F.grid_sample(
        data_ra, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    del data_ra  # release ~1V; only deskewed is needed for the mean
    # (Z_avg, Y_out, N, X_out) → average over the N grouped slices
    result = deskewed.mean(dim=2)

    if keep_overhang and (overhang_fill == "mean" or overhang_fill != 0):
        fill_value = None if overhang_fill == "mean" else float(overhang_fill)
        result = _fill_overhang_torch(result, fill_value=fill_value)

    return result


# Adapt ZYX function to CZYX
# Needs to be a top-level function for multiprocessing pickling
def _czyx_deskew_data(data, **kwargs):
    return deskew_zyx(data[0], **kwargs)[None]


def _czyx_fast_deskew_data(data, device="cuda", num_splits=1, **kwargs):
    """CZYX wrapper for `fast_deskew_zyx`. Handles numpy↔torch conversion.

    When `num_splits` > 1 the volume is split along input axis 2
    (X_coverslip → output Y) before transfer to GPU, so each chunk fits in
    device memory.  The Y axis is independent in the deskew transform so the
    result is exact.
    """
    zyx = data[0]  # (Z, Y, X)
    if num_splits > 1:
        # Split along X (axis 2), deskew each chunk, concatenate along output Y (axis 1).
        # The deskew flips X → higher input X maps to lower output Y, so reverse.
        chunks = np.array_split(zyx, num_splits, axis=2)
        results = [
            fast_deskew_zyx(
                torch.from_numpy(c).to(device=device, dtype=torch.float32),
                **kwargs,
            )
            .cpu()
            .numpy()
            for c in reversed(chunks)
        ]
        return np.concatenate(results, axis=1)[None]

    # Cast in torch (one allocation in torch's allocator) instead of via
    # numpy's astype (which would create an intermediate numpy buffer that
    # the torch tensor then shares memory with).
    tensor = torch.from_numpy(zyx).to(device=device, dtype=torch.float32)
    return fast_deskew_zyx(tensor, **kwargs).cpu().numpy()[None]


def _warn_pixel_size_mismatch(
    settings: DeskewSettings, reference_position_path: str | Path
) -> None:
    """Warn if the config's lateral pixel size disagrees with the input zarr scale.

    The config is the source of truth for ``pixel_size_um``; this only surfaces a
    likely misconfiguration when it differs (>5%) from the input plate's XY scale
    metadata.
    """
    with open_ome_zarr(str(reference_position_path), mode="r") as ds:
        zarr_pixel_size = float(ds.scale[-1])
    if not np.isclose(settings.pixel_size_um, zarr_pixel_size, rtol=0.05):
        warnings.warn(
            f"Config pixel_size_um={settings.pixel_size_um} differs from the input "
            f"zarr metadata XY scale ({zarr_pixel_size:.4f}).",
            stacklevel=2,
        )


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    settings: DeskewSettings,
) -> tuple[tuple[int, int, int, int, int], list[str]]:
    """Create (or extend) the empty deskewed output plate.

    create_empty_plate is idempotent: re-running with the same positions is a
    no-op, and new positions get appended. Safe to call from both the SLURM
    orchestrator and from per-position runs.

    Returns the input (T, C, Z, Y, X) shape and channel names.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        channel_names = input_dataset.channel_names
        input_shape = input_dataset.data.shape
    T, C, Z, Y, X = input_shape

    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        settings.keep_overhang,
        settings.average_n_slices,
        settings.pixel_size_um,
    )

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(p).parts[-3:] for p in input_position_dirpaths],
        channel_names=channel_names,
        shape=(T, C) + deskewed_shape,
        scale=(1, 1) + voxel_size,
        version=resolve_ome_zarr_version(
            input_position_dirpaths[0], settings.output_ome_zarr_version
        ),
        metadata_sources=input_plate,
    )

    return (T, C, Z, Y, X), channel_names


def deskew(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
):
    """Deskew oblique plane light-sheet dataset, processing time point and channels in parallel.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to input positions, for example: "input.zarr/0/0/0", "input.zarr/0/0/[0-9]",
        or "input.zarr/*/*/*".
    config_filepath : Path
        Path to YAML configuration file.
    output_dirpath : Path
        Path to "output.zarr" directory.
    sbatch_filepath : str, optional
        SBATCH filepath that contains slurm parameters to overwrite defaults.
        For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU.
    cluster : str, optional
        Execution cluster: 'slurm' submits to a Slurm cluster, 'local' runs jobs as
        subprocesses on this machine, 'debug' runs jobs in-process in the foreground.
    monitor : bool, optional
        Monitor of submitted SLURM jobs.
    init_only : bool, optional
        Only initialize the output store and exit; skip per-position processing.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    settings = yaml_to_model(config_filepath, DeskewSettings)
    _warn_pixel_size_mismatch(settings, input_position_dirpaths[0])
    input_shape, _ = _init_output_plate(input_position_dirpaths, output_dirpath, settings)

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=input_shape, ram_multiplier=8, max_num_cpus=16
    )
    mem_gb = num_cpus * gb_ram_per_cpu
    time_minutes = 60
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        click.echo(f"Initialized {output_dirpath} ({len(input_position_dirpaths)} positions)")
        return

    output_position_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)

    deskew_args = {
        "ls_angle_deg": settings.ls_angle_deg,
        "px_to_scan_ratio": settings.px_to_scan_ratio,
        "keep_overhang": settings.keep_overhang,
        "average_n_slices": settings.average_n_slices,
        "overhang_fill": settings.overhang_fill,
        "device": settings.device,
        "extra_metadata": {"biahub-deskew": settings.model_dump()},
    }

    slurm_args = {
        "slurm_job_name": "deskew",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": time_minutes,
        "slurm_partition": "preempted",
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths, strict=True
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    _czyx_fast_deskew_data,
                    input_position_path,
                    output_position_path,
                    num_workers=slurm_args["slurm_cpus_per_task"],
                    **deskew_args,
                )
            )

    job_ids = [job.job_id for job in jobs]

    slurm_out_path.mkdir(exist_ok=True)
    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a
    # DebugJob but execution only happens when .wait()/.done()/.result() is
    # called. Run each one in the foreground and stream progress; monitor's
    # async polling UI is pointless against synchronous in-process jobs.
    if resolved_cluster == "debug":
        for job, path in zip(jobs, input_position_dirpaths, strict=True):
            job.wait()
            click.echo(f"Deskew complete: {path}")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("deskew")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
def deskew_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
):
    """Deskew oblique plane light-sheet dataset. Deskew parameters can be estimated with estimate-deskew.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub deskew -i ./input.zarr/*/*/* -c ./deskew_params.yml -o ./output.zarr

    \b
    Initialize the output plate only (e.g. before running per-position Nextflow workers):
    >>> biahub deskew --init -i ./input.zarr/*/*/* -c ./deskew_params.yml -o ./output.zarr

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub deskew --cluster debug -i ./input.zarr/A/1/0 -c ./deskew_params.yml -o ./output.zarr


    """  # noqa: D301
    deskew(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
    )


if __name__ == "__main__":
    deskew_cli()
