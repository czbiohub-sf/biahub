import shutil

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

import click
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr
from numpy.typing import ArrayLike
from pystackreg import StackReg
from scipy.fftpack import next_fast_len
from tqdm import tqdm

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.registration.utils import (
    evaluate_transforms,
    save_transforms,
)
from biahub.settings import (
    EstimateStabilizationSettings,
    PhaseCrossCorrSettings,
    StabilizationSettings,
    StackRegSettings,
)

NA_DET = 1.35
LAMBDA_ILL = 0.500




def pad_to_shape(
    arr: ArrayLike,
    shape: Tuple[int, ...],
    mode: str,
    verbose: bool = False,
    **kwargs,
) -> ArrayLike:
    """
    Pad or crop array to match provided shape.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    shape : Tuple[int]
        Output shape.
    mode : str
        Padding mode (see np.pad).
    verbose : bool
        If True, print verbose output.
    kwargs : dict
        Additional keyword arguments for np.pad.

    Returns
    -------
    ArrayLike
        Padded array.
    """
    assert arr.ndim == len(shape)

    dif = tuple(s - a for s, a in zip(shape, arr.shape))
    assert all(d >= 0 for d in dif)

    pad_width = [[s // 2, s - s // 2] for s in dif]

    if verbose:
        click.echo(
            f"padding: input shape {arr.shape}, output shape {shape}, padding {pad_width}"
        )

    return np.pad(arr, pad_width=pad_width, mode=mode, **kwargs)


def center_crop(
    arr: ArrayLike,
    shape: Tuple[int, ...],
    verbose: bool = False,
) -> ArrayLike:
    """
    Crop the center of `arr` to match provided shape.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    shape : Tuple[int, ...]
    """
    assert arr.ndim == len(shape)

    starts = tuple((cur_s - s) // 2 for cur_s, s in zip(arr.shape, shape))

    assert all(s >= 0 for s in starts)

    slicing = tuple(slice(s, s + d) for s, d in zip(starts, shape))
    if verbose:
        click.echo(
            f"center crop: input shape {arr.shape}, output shape {shape}, slicing {slicing}"
        )
    return arr[slicing]


def match_shape(
    img: ArrayLike,
    shape: Tuple[int, ...],
    verbose: bool = False,
) -> ArrayLike:
    """
    Pad or crop array to match provided shape.

    Parameters
    ----------
    img : ArrayLike
        Input array.
    shape : Tuple[int, ...]
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Padded or cropped array.
    """

    if np.any(shape > img.shape):
        padded_shape = np.maximum(img.shape, shape)
        img = pad_to_shape(img, padded_shape, mode="reflect")

    if np.any(shape < img.shape):
        img = center_crop(img, shape)

    if verbose:
        click.echo(f"matched shape: input shape {img.shape}, output shape {shape}")

    return img


def plot_cross_correlation(
    corr,
    title="Cross-Correlation",
    output_path=None,
    xlabel="X shift (pixels)",
    ylabel="Y shift (pixels)",
) -> None:
    """
    Plot the cross-correlation.

    Parameters
    ----------
    corr : ArrayLike
        Cross-correlation array.
    title : str
        Title for the plot.
    output_path : Path
        Path to the output directory.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.

    Returns
    -------
    None
        Saves the plot to the output directory.
    """
    # Convert to 2D if necessary
    if corr.ndim == 3:
        corr_to_plot = np.max(corr, axis=0)  # Or a center slice
    else:
        corr_to_plot = corr

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_to_plot, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation strength")

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)  # This prevents overlap in future plots


def phase_cross_corr_padding(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    maximum_shift: float = 1.2,
    normalization: Optional[Literal["magnitude", "classic"]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[int, ...]:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2

    Computes translation shift using arg. maximum of phase cross correlation.
    Input are padded or cropped for fast FFT computation assuming a maximum translation shift.

    Parameters
    ----------
    ref_img : ArrayLike
        Reference image.
    mov_img : ArrayLike
        Moved image.
    maximum_shift : float, optional
        Maximum location shift normalized by axis size, by default 1.0

    Returns
    -------
    Tuple[int, ...]
        Shift between reference and moved image.
    """
    shape = tuple(
        cast(int, next_fast_len(int(max(s1, s2) * maximum_shift)))
        for s1, s2 in zip(ref_img.shape, mov_img.shape)
    )

    if verbose:
        click.echo(
            f"phase cross corr. fft shape of {shape} for arrays of shape {ref_img.shape} and {mov_img.shape} "
            f"with maximum shift of {maximum_shift}"
        )

    ref_img = match_shape(ref_img, shape)
    mov_img = match_shape(mov_img, shape)
    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps
    del ref_img, mov_img

    prod = Fimg1 * Fimg2.conj()

    if normalization == "magnitude":
        prod /= np.fmax(np.abs(prod), eps)
    elif normalization == "classic":
        prod /= np.abs(Fimg1) * np.abs(Fimg2)

    corr = np.fft.irfftn(prod)
    del prod, Fimg1, Fimg2

    corr = np.fft.fftshift(np.abs(corr))

    argmax = np.argmax(corr)
    peak = np.unravel_index(argmax, corr.shape)
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak))

    if verbose:
        click.echo(f"phase cross corr. peak at {peak}")
    if output_path:
        plot_cross_correlation(corr, title="Cross-Correlation", output_path=output_path)

    return peak, corr


def phase_cross_corr(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    normalization: Optional[Literal["magnitude", "classic"]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[int, ...]:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2

    Computes translation shift using arg. maximum of phase cross correlation.
    Input are padded or cropped for fast FFT computation assuming a maximum translation shift.

    Parameters
    ----------
    ref_img : ArrayLike
        Reference image.
    mov_img : ArrayLike
        Moved image.
    normalization : Literal["magnitude", "classic"]
        Normalization method.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    Tuple[int, ...]
        Shift between reference and moved image.
    """

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps

    prod = Fimg1 * Fimg2.conj()

    if normalization == "magnitude":
        norm = np.fmax(np.abs(prod), eps)
    elif normalization == "classic":
        norm = np.abs(Fimg1) * np.abs(Fimg2)
    else:
        norm = 1.0

    corr = np.fft.irfftn(prod / norm)
    corr_shifted = np.fft.fftshift(np.abs(corr))
    if output_path:
        plot_cross_correlation(
            corr_shifted, title="Cross-Correlation", output_path=output_path
        )
    maxima = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    midpoint = np.array([np.fix(axis_size / 2) for axis_size in corr.shape])

    float_dtype = prod.real.dtype
    del Fimg1, Fimg2, prod, norm

    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(corr.shape)[shift > midpoint]

    return shift, corr_shifted


def get_tform_from_pcc(
    t: int,
    source_channel_tzyx: da.Array,
    target_channel_tzyx: da.Array,
    function_type: Literal["custom_padding", "custom"] = "custom",
    normalization: Optional[Literal["magnitude", "classic"]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[ArrayLike, Tuple[int, int, int]]:
    """
    Get the transformation matrix from phase cross correlation.

    Parameters
    ----------
    t : int
        Time index.
    source_channel_tzyx : da.Array
        Source channel data.
    target_channel_tzyx : da.Array
        Target channel data.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Transformation matrix.
    """

    target = np.asarray(source_channel_tzyx[t]).astype(np.float32)
    source = np.asarray(target_channel_tzyx[t]).astype(np.float32)

    if function_type == "custom_padding":
        shift, corr = phase_cross_corr_padding(
            target, source, normalization=normalization, output_path=output_path
        )
    elif function_type == "custom":
        shift, corr = phase_cross_corr(
            target, source, normalization=normalization, output_path=output_path
        )
    if verbose:
        click.echo(f"Time {t}: shift (dz,dy,dx) = {shift[0]}, {shift[1]}, {shift[2]}")

    dz, dy, dx = shift

    transform = np.eye(4)
    transform[0, 3] = dx
    transform[1, 3] = dy
    transform[2, 3] = dz
    if verbose:
        click.echo(f"transform: {transform}")

    return transform, shift, corr


def plot_pcc_drifts(
    df: pd.DataFrame,
    output_dir: Path,
    label='sample',
    title='PCC Drift Analysis',
    unit: Literal['µm', 'px'] = 'µm',
    voxel_size: Tuple[float, float, float] = (0.174, 0.1494, 0.1494),  # (Z, Y, X) in microns
) -> None:
    """
    Plot the drifts from PCC per timepoint for a single position

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cross-correlation data.
    output_dir : Path
        Path to the output directory.
    label : str
        Label for the plot.
    title : str
        Title for the plot.
    voxel_size : Tuple[float, float, float]
        Voxel size in microns.
    Returns
    -------
    None
        Saves the plot to the output directory.
    """
    if unit == 'µm':
        # Unpack voxel sizes
        z_scale, y_scale, x_scale = voxel_size

        # Convert to microns
        df['ShiftX'] = df['ShiftX'] * x_scale
        df['ShiftY'] = df['ShiftY'] * y_scale
        df['ShiftZ'] = df['ShiftZ'] * z_scale

    # Cumulative and magnitude drift
    df['CumulativeShiftX'] = df['ShiftX'].cumsum()
    df['CumulativeShiftY'] = df['ShiftY'].cumsum()
    df['CumulativeShiftZ'] = df['ShiftZ'].cumsum()
    df['DriftMagnitude'] = np.sqrt(df['ShiftX'] ** 2 + df['ShiftY'] ** 2 + df['ShiftZ'] ** 2)
    df['CumulativeDrift'] = np.sqrt(
        df['CumulativeShiftX'] ** 2 + df['CumulativeShiftY'] ** 2 + df['CumulativeShiftZ'] ** 2
    )

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(df["TimepointID"], df["ShiftX"], label="ShiftX")
    axs[0].plot(df["TimepointID"], df["ShiftY"], label="ShiftY")
    axs[0].plot(df["TimepointID"], df["ShiftZ"], label="ShiftZ")
    axs[0].set_ylabel(f"Shift ({unit})")
    axs[0].legend()
    axs[0].set_title("Raw Drift per Axis")
    axs[0].grid(True)

    axs[1].plot(df["TimepointID"], df["CumulativeShiftX"], label="Cumulative X")
    axs[1].plot(df["TimepointID"], df["CumulativeShiftY"], label="Cumulative Y")
    axs[1].plot(df["TimepointID"], df["CumulativeShiftZ"], label="Cumulative Z")
    axs[1].set_ylabel(f"Cumulative Shift ({unit})")
    axs[1].legend()
    axs[1].set_title("Cumulative Shift")
    axs[1].grid(True)

    axs[2].plot(df["TimepointID"], df["DriftMagnitude"], label="Instantaneous", color="gray")
    axs[2].plot(
        df["TimepointID"],
        df["CumulativeDrift"],
        label="Cumulative",
        color="black",
        linestyle="--",
    )
    axs[2].set_ylabel(f"Drift Magnitude ({unit})")
    axs[2].legend()
    axs[2].set_title("Drift Magnitude")
    axs[2].grid(True)
    axs[2].set_xlabel("Timepoints")

    fig.suptitle(f"{title} - {label}", fontsize=16, y=1.02)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{label}.png", bbox_inches="tight")
    plt.close(fig)


def plot_corr_max_min_sum(
    corr_df: pd.DataFrame, output_path: Path, label='sample', title='Cross-Correlation Summary'
) -> None:
    """
    Plot the max, min, and sum of the cross-correlation from PCC per timepoint for a single position

    Parameters
    ----------
    corr_df : pd.DataFrame
        DataFrame containing the cross-correlation data.
    output_path : Path
        Path to the output directory.
    label : str
        Label for the plot.

    Returns
    -------
    None
        Saves the plot to the output directory.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(corr_df["TimepointID"], corr_df["max"], label="Corr Max")
    axs[1].plot(corr_df["TimepointID"], corr_df["min"], label="Corr Min")
    axs[2].plot(corr_df["TimepointID"], corr_df["sum"], label="Corr Sum")
    axs[0].set_ylabel("Max")
    axs[1].set_ylabel("Min")
    axs[2].set_ylabel("Sum")
    axs[0].set_title("Corr Max")
    axs[1].set_title("Corr Min")
    axs[2].set_title("Corr Sum")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    axs[2].set_xlabel("Timepoint")
    fig.suptitle(f"{title} - {label}", y=1.02, fontsize=16)
    fig.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / f"{label}.png", bbox_inches="tight")
    plt.close(fig)


def estimate_xyz_stabilization_pcc_per_position(
    input_position_dirpath: Path,
    output_folder_path: Path,
    output_shifts_path: Path,
    channel_index: int,
    phase_cross_corr_settings: PhaseCrossCorrSettings,
    verbose: bool = False,
) -> list[ArrayLike]:
    """
    Estimate the xyz stabilization for a single position.

    Parameters
    ----------
    input_position_dirpath : Path
        Path to the input position directory.
    output_folder_path : Path
        Path to the output folder.
    channel_index : int
        Index of the channel to process.
    center_crop_xy : list[int]
        Size of the crop in the XY plane.
    t_reference : str
        Reference timepoint.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    list[ArrayLike]
        List of the xyz stabilization for each timepoint.
    """
    with open_ome_zarr(input_position_dirpath) as input_position:
        channel_tzyx = input_position.data.dask_array()[:, channel_index]
        T, Z, Y, X = channel_tzyx.shape
        X_slice = phase_cross_corr_settings.X_slice
        Y_slice = phase_cross_corr_settings.Y_slice
        Z_slice = phase_cross_corr_settings.Z_slice

        if phase_cross_corr_settings.center_crop_xy:
            x_idx = slice(
                X // 2 - phase_cross_corr_settings.center_crop_xy[0] // 2,
                X // 2 + phase_cross_corr_settings.center_crop_xy[0] // 2,
            )
            y_idx = slice(
                Y // 2 - phase_cross_corr_settings.center_crop_xy[1] // 2,
                Y // 2 + phase_cross_corr_settings.center_crop_xy[1] // 2,
            )
        else:
            x_idx = slice(0, X)
            y_idx = slice(0, Y)
        if X_slice == "all":
            x_idx = slice(0, X)
        else:
            x_idx = slice(X_slice[0], X_slice[1])
        if Y_slice == "all":
            y_idx = slice(0, Y)
        else:
            y_idx = slice(Y_slice[0], Y_slice[1])
        if Z_slice == "all":
            z_idx = slice(0, Z)
        else:
            z_idx = slice(Z_slice[0], Z_slice[1])

        print(f"x_idx: {x_idx}, y_idx: {y_idx}, z_idx: {z_idx}")

        channel_tzyx_cropped = channel_tzyx[:, z_idx, y_idx, x_idx]

        if phase_cross_corr_settings.t_reference == "first":
            target_channel_tzyx = np.broadcast_to(
                channel_tzyx_cropped[0], channel_tzyx_cropped.shape
            ).copy()
        elif phase_cross_corr_settings.t_reference == "previous":
            target_channel_tzyx = np.roll(channel_tzyx_cropped, shift=1, axis=0)
            target_channel_tzyx[0] = channel_tzyx_cropped[0]
        source_channel_tzyx = channel_tzyx_cropped

        position_filename = str(Path(*input_position_dirpath.parts[-3:])).replace("/", "_")

        transforms = []
        shifts = []
        corr_list = []
        output_path_corr = output_folder_path.parent / "corr_plots" / position_filename
        output_path_corr.mkdir(parents=True, exist_ok=True)

        for t in range(T):
            click.echo(f"Estimating PCC for timepoint {t}")
            if t == 0:
                transforms.append(np.eye(4).tolist())
                corr_list.append((t, 0, 0, 0))
                shifts.append((t, 0, 0, 0))
            else:
                transform, shift, corr = get_tform_from_pcc(
                    t=t,
                    source_channel_tzyx=source_channel_tzyx,
                    target_channel_tzyx=target_channel_tzyx,
                    verbose=verbose,
                    function_type=phase_cross_corr_settings.function_type,
                    normalization=phase_cross_corr_settings.normalization,
                    output_path=output_path_corr / f"{t}.png",
                )
                transforms.append(transform)
                shifts.append((t, *shift))
                if corr is not None:
                    corr_list.append((t, corr.max(), corr.min(), corr.sum()))
                else:
                    corr_list.append((t, None, None, None))
            click.echo(f"Transform for timepoint {t}: {transforms[-1]}")

        np.save(
            output_folder_path / f"{position_filename}.npy",
            np.array(transforms, dtype=np.float32),
        )
        # save the shifts as a csv
        if verbose:
            shifts_df = pd.DataFrame(
                shifts, columns=["TimepointID", "ShiftZ", "ShiftY", "ShiftX"]
            )
            shifts_df["TimepointID"] = shifts_df["TimepointID"].astype(int)
            shifts_df["ShiftZ"] = shifts_df["ShiftZ"].astype(float)
            shifts_df["ShiftY"] = shifts_df["ShiftY"].astype(float)
            shifts_df["ShiftX"] = shifts_df["ShiftX"].astype(float)
            shifts_df.to_csv(output_shifts_path / f"{position_filename}.csv", index=False)

            output_path_shift_plots = output_shifts_path / "plots"
            output_path_shift_plots.mkdir(parents=True, exist_ok=True)
            plot_pcc_drifts(shifts_df, output_path_shift_plots, label=position_filename)

            output_path_corr_csv = output_folder_path.parent / "corr_max_min_sum"
            output_path_corr_csv.mkdir(parents=True, exist_ok=True)

            corr_df = pd.DataFrame(corr_list, columns=["TimepointID", "max", "min", "sum"])
            corr_df["TimepointID"] = corr_df["TimepointID"].astype(int)
            corr_df["max"] = corr_df["max"].astype(float)
            corr_df["min"] = corr_df["min"].astype(float)
            corr_df["sum"] = corr_df["sum"].astype(float)
            corr_df.to_csv(output_path_corr_csv / f"{position_filename}.csv", index=False)

            output_path_corr_plots = output_path_corr_csv / "plots"
            output_path_corr_plots.mkdir(parents=True, exist_ok=True)
            plot_corr_max_min_sum(corr_df, output_path_corr_plots, label=position_filename)

        click.echo(f"Saved transforms for {position_filename}.")

    return transforms


def estimate_xyz_stabilization_pcc(
    input_position_dirpaths: list[Path],
    output_folder_path: Path,
    phase_cross_corr_settings: PhaseCrossCorrSettings,
    channel_index: int = 0,
    sbatch_filepath: Path = None,
    cluster: str = "local",
    verbose: bool = False,
) -> dict[str, list[ArrayLike]]:
    """
    Estimate the xyz stabilization for a list of positions.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to the input position directories.
    output_folder_path : Path
        Path to the output folder.
    phase_cross_corr_settings : PhaseCrossCorrSettings
        Settings for the phase cross correlation.
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
        Dictionary of the xyz stabilization for each position.
    """

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        shape = dataset.data.shape
        T, C, Z, Y, X = shape

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, C, Z, Y, X), ram_multiplier=16, max_num_cpus=16
    )

    slurm_args = {
        "slurm_job_name": "estimate_xyz_pcc",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 60,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM xyz PCC jobs with resources: {slurm_args}")
    transforms_out_path = output_folder_path / "transforms_per_position"
    transforms_out_path.mkdir(parents=True, exist_ok=True)
    shifts_out_path = output_folder_path / "shifts_per_position"
    shifts_out_path.mkdir(parents=True, exist_ok=True)

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_dirpath in input_position_dirpaths:
            job = executor.submit(
                estimate_xyz_stabilization_pcc_per_position,
                input_position_dirpath=input_position_dirpath,
                output_folder_path=transforms_out_path,
                output_shifts_path=shifts_out_path,
                channel_index=channel_index,
                phase_cross_corr_settings=phase_cross_corr_settings,
                verbose=verbose,
            )
            jobs.append(job)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    wait_for_jobs_to_finish(jobs)

    transform_files = list(transforms_out_path.glob("*.npy"))

    fov_transforms = {}
    for file_path in transform_files:
        fov_filename = file_path.stem
        fov_transforms[fov_filename] = np.load(file_path).tolist()

    # Remove the output folder
    shutil.rmtree(transforms_out_path)

    return fov_transforms




























# import itertools
# import shutil

# from datetime import datetime
# from pathlib import Path
# from typing import Literal, Optional, Tuple, cast

# import click
# import dask.array as da
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import submitit

# from iohub.ngff import open_ome_zarr
# from numpy.typing import ArrayLike
# from pystackreg import StackReg
# from scipy.fftpack import next_fast_len
# from skimage.registration import phase_cross_correlation as sk_phase_cross_correlation
# from waveorder.focus import focus_from_transverse_band

# from biahub.cli.parsing import (
#     sbatch_to_submitit,
# )
# from biahub.cli.slurm import wait_for_jobs_to_finish
# from biahub.cli.utils import estimate_resources
# from biahub.core.transform import Transform
# from biahub.registration.utils import match_shape
# from biahub.settings import (
#     FocusFindingSettings,
#     PhaseCrossCorrSettings,
#     StackRegSettings,
# )

# NA_DET = 1.35
# LAMBDA_ILL = 0.500


# def phase_cross_correlation(
#     ref_img: ArrayLike,
#     mov_img: ArrayLike,
#     maximum_shift: float = 1.2,
#     normalization: Optional[Literal["magnitude", "classic"]] = None,
#     output_path: Optional[Path] = None,
#     verbose: bool = False,
# ) -> Tuple[int, ...]:
#     """
#     Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2

#     Computes translation shift using arg. maximum of phase cross correlation.
#     Input are padded or cropped for fast FFT computation assuming a maximum translation shift.
#     moving -> reference
#     Parameters
#     ----------
#     ref_img : ArrayLike
#         Reference image.
#     mov_img : ArrayLike
#         Moved image.
#     maximum_shift : float, optional
#         Maximum location shift normalized by axis size, by default 1.0

#     Returns
#     -------
#     Tuple[int, ...]
#         Shift between reference and moved image.
#     """
#     shape = tuple(
#         cast(int, next_fast_len(int(max(s1, s2) * maximum_shift)))
#         for s1, s2 in zip(ref_img.shape, mov_img.shape)
#     )

#     if verbose:
#         click.echo(
#             f"phase cross corr. fft shape of {shape} for arrays of shape {ref_img.shape} and {mov_img.shape} "
#             f"with maximum shift of {maximum_shift}"
#         )

#     ref_img = match_shape(ref_img, shape)
#     mov_img = match_shape(mov_img, shape)
#     Fimg1 = np.fft.rfftn(ref_img)
#     Fimg2 = np.fft.rfftn(mov_img)
#     eps = np.finfo(Fimg1.dtype).eps
#     del ref_img, mov_img

#     prod = Fimg1 * Fimg2.conj()

#     if normalization == "magnitude":
#         prod /= np.fmax(np.abs(prod), eps)
#     elif normalization == "classic":
#         prod /= np.abs(Fimg1) * np.abs(Fimg2)

#     corr = np.fft.irfftn(prod)
#     del prod, Fimg1, Fimg2

#     corr = np.fft.fftshift(np.abs(corr))

#     argmax = np.argmax(corr)
#     peak = np.unravel_index(argmax, corr.shape)
#     peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak))

#     if verbose:
#         click.echo(f"phase cross corr. peak at {peak}")

#     return peak


# def estimate(
#     mov: da.Array,
#     ref: da.Array,
#     function_type: Literal["skimage", "custom"] = "custom",
#     normalization: Optional[Literal["magnitude", "classic"]] = None,
#     output_path: Optional[Path] = None,
#     verbose: bool = False,
# ) -> Transform:
#     """
#     Estimate the transformation matrix from the shift.

#     Parameters
#     ----------
#     mov : da.Array
#         Moving image.
#     ref : da.Array
#         Reference image.
#     function_type : Literal["skimage", "custom"]
#         Function type to use for the phase cross correlation.
#     normalization : Optional[Literal["magnitude", "classic"]]
#         Normalization to use for the phase cross correlation.
#     output_path : Optional[Path]
#         Output path to save the plot.
#     verbose : bool
#         If True, print verbose output.
#     """
#     shift = shift_from_pcc(
#         mov=mov,
#         ref=ref,
#         function_type=function_type,
#         normalization=normalization,
#         output_path=output_path,
#         verbose=verbose,
#     )

#     if shift.ndim == 2:
#         dy, dx = shift

#         transform = np.eye(3)
#         transform[1, 2] = dx
#         transform[0, 2] = dy

#     elif shift.ndim == 3:

#         dz, dy, dx = shift

#         transform = np.eye(4)
#         transform[2, 3] = dx
#         transform[1, 3] = dy
#         transform[0, 3] = dz

#     if verbose:
#         click.echo(f"transform: {transform}")

#     return Transform(matrix=transform), shift


# def shift_from_pcc(
#     mov: da.Array,
#     ref: da.Array,
#     function_type: Literal["skimage", "custom"] = "custom",
#     normalization: Optional[Literal["magnitude", "classic"]] = None,
#     output_path: Optional[Path] = None,
#     verbose: bool = False,
# ) -> Tuple[ArrayLike, Tuple[int, int, int]]:
#     """
#     Get the transformation matrix from phase cross correlation.

#     Parameters
#     ----------
#     t : int
#         Time index.
#     source_channel_tzyx : da.Array
#         Source channel data.
#     target_channel_tzyx : da.Array
#         Target channel data.
#     verbose : bool
#         If True, print verbose output.

#     Returns
#     -------
#     ArrayLike
#         Transformation matrix.
#     """

#     if function_type == "skimage":
#         if normalization is not None:
#             normalization = 'phase'
#         shift, _, _ = sk_phase_cross_correlation(
#             reference_image=ref,
#             moving_image=mov,
#             normalization=normalization,
#         )
#     elif function_type == "custom":
#         shift = phase_cross_correlation(mov, ref, normalization=normalization)
#     if verbose:
#         click.echo(f"shift: {shift}")
#     return shift


# def get_crop_idx(X, Y, Z, phase_cross_corr_settings):
#     """
#     Get the crop indices for the phase cross correlation.

#     Parameters
#     ----------
#     X : int
#         X dimension.
#     Y : int
#     Z : int
#         Z dimension.
#     phase_cross_corr_settings : PhaseCrossCorrSettings
#         Phase cross correlation settings.

#     Returns
#     -------
#     slice
#         Crop indices.
#     """
#     X_slice = phase_cross_corr_settings.X_slice
#     Y_slice = phase_cross_corr_settings.Y_slice
#     Z_slice = phase_cross_corr_settings.Z_slice

#     if phase_cross_corr_settings.center_crop_xy:
#         x_idx = slice(
#             X // 2 - phase_cross_corr_settings.center_crop_xy[0] // 2,
#             X // 2 + phase_cross_corr_settings.center_crop_xy[0] // 2,
#         )
#         y_idx = slice(
#             Y // 2 - phase_cross_corr_settings.center_crop_xy[1] // 2,
#             Y // 2 + phase_cross_corr_settings.center_crop_xy[1] // 2,
#         )
#     else:
#         x_idx = slice(0, X)
#         y_idx = slice(0, Y)
#     if X_slice == "all":
#         x_idx = slice(0, X)
#     else:
#         x_idx = slice(X_slice[0], X_slice[1])
#     if Y_slice == "all":
#         y_idx = slice(0, Y)
#     else:
#         y_idx = slice(Y_slice[0], Y_slice[1])
#     if Z_slice == "all":
#         z_idx = slice(0, Z)
#     else:
#         z_idx = slice(Z_slice[0], Z_slice[1])

#     print(f"x_idx: {x_idx}, y_idx: {y_idx}, z_idx: {z_idx}")

#     return x_idx, y_idx, z_idx


# def estimate_tczyx(
#     input_position_dirpath: Path,
#     output_folder_path: Path,
#     output_shifts_path: Path,
#     channel_index: int,
#     phase_cross_corr_settings: PhaseCrossCorrSettings,
#     verbose: bool = False,
#     mode: Literal["registration", "stabilization"] = "stabilization",
# ) -> list[ArrayLike]:
#     """
#     Estimate the xyz stabilization for a single position.

#     Parameters
#     ----------
#     input_position_dirpath : Path
#         Path to the input position directory.
#     output_folder_path : Path
#         Path to the output folder.
#     channel_index : int
#         Index of the channel to process.
#     center_crop_xy : list[int]
#         Size of the crop in the XY plane.
#     t_reference : str
#         Reference timepoint.
#     verbose : bool
#         If True, print verbose output.

#     Returns
#     -------
#     list[ArrayLike]
#         List of the xyz stabilization for each timepoint.
#     """
#     with open_ome_zarr(input_position_dirpath) as input_position:
#         data_tzyx = input_position.data.dask_array()[:, channel_index]
#         T, Z, Y, X = data_tzyx.shape

#     x_idx, y_idx, z_idx = get_crop_idx(X, Y, Z, phase_cross_corr_settings)
#     data_tzyx_cropped = data_tzyx[:, z_idx, y_idx, x_idx]

#     if mode == "stabilization":
#         if phase_cross_corr_settings.t_reference == "first":
#             ref_tzyx = np.broadcast_to(data_tzyx_cropped[0], data_tzyx_cropped.shape).copy()
#         elif phase_cross_corr_settings.t_reference == "previous":
#             ref_tzyx = np.roll(data_tzyx_cropped, shift=1, axis=0)
#             ref_tzyx[0] = data_tzyx_cropped[0]
#     elif mode == "registration":
#         raise ValueError("Registration mode not implemented yet")

#     mov_tzyx = data_tzyx_cropped

#     position_filename = str(Path(*input_position_dirpath.parts[-3:])).replace("/", "_")

#     transforms = []
#     shifts = []
#     output_path_corr = output_folder_path.parent / "corr_plots" / position_filename
#     output_path_corr.mkdir(parents=True, exist_ok=True)

#     for t in range(T):
#         click.echo(f"Estimating PCC for timepoint {t}")
#         if t == 0:
#             transforms.append(np.eye(4).tolist())
#             shifts.append((t, 0, 0, 0))
#         else:
#             transform, shift = estimate(
#                 mov=mov_tzyx[t],
#                 ref=ref_tzyx[t],
#                 function_type=phase_cross_corr_settings.function_type,
#                 normalization=phase_cross_corr_settings.normalization,
#                 output_path=output_path_corr / f"{t}.png",
#                 verbose=verbose,
#             )
#             transforms.append(transform)
#             shifts.append((t, *shift))

#         click.echo(f"Transform for timepoint {t}: {transforms[-1]}")

#     np.save(
#         output_folder_path / f"{position_filename}.npy",
#         np.array(transforms, dtype=np.float32),
#     )
#     # save the shifts as a csv
#     if verbose:
#         shifts_df = pd.DataFrame(shifts, columns=["TimepointID", "ShiftZ", "ShiftY", "ShiftX"])
#         shifts_df["TimepointID"] = shifts_df["TimepointID"].astype(int)
#         shifts_df["ShiftZ"] = shifts_df["ShiftZ"].astype(float)
#         shifts_df["ShiftY"] = shifts_df["ShiftY"].astype(float)
#         shifts_df["ShiftX"] = shifts_df["ShiftX"].astype(float)
#         shifts_df.to_csv(output_shifts_path / f"{position_filename}.csv", index=False)

#         output_path_shift_plots = output_shifts_path / "plots"
#         output_path_shift_plots.mkdir(parents=True, exist_ok=True)
#         # plot_pcc_drifts(shifts_df, output_path_shift_plots, label=position_filename)

#     click.echo(f"Saved transforms for {position_filename}.")

#     return transforms


# def estimate_xyz_stabilization_pcc(
#     input_position_dirpaths: list[Path],
#     output_folder_path: Path,
#     phase_cross_corr_settings: PhaseCrossCorrSettings,
#     channel_index: int = 0,
#     sbatch_filepath: Path = None,
#     cluster: str = "local",
#     verbose: bool = False,
# ) -> dict[str, list[ArrayLike]]:
#     """
#     Estimate the xyz stabilization for a list of positions.

#     Parameters
#     ----------
#     input_position_dirpaths : list[Path]
#         Paths to the input position directories.
#     output_folder_path : Path
#         Path to the output folder.
#     phase_cross_corr_settings : PhaseCrossCorrSettings
#         Settings for the phase cross correlation.
#     channel_index : int
#         Index of the channel to process.
#     sbatch_filepath : Path
#         Path to the sbatch file.
#     cluster : str
#         Cluster to use.
#     verbose : bool
#         If True, print verbose output.

#     Returns
#     -------
#     dict[str, list[ArrayLike]]
#         Dictionary of the xyz stabilization for each position.
#     """

#     output_folder_path.mkdir(parents=True, exist_ok=True)
#     slurm_out_path = output_folder_path / "slurm_output"
#     slurm_out_path.mkdir(parents=True, exist_ok=True)

#     with open_ome_zarr(input_position_dirpaths[0]) as dataset:
#         shape = dataset.data.shape
#         T, C, Z, Y, X = shape

#     num_cpus, gb_ram_per_cpu = estimate_resources(
#         shape=(T, C, Z, Y, X), ram_multiplier=16, max_num_cpus=16
#     )

#     slurm_args = {
#         "slurm_job_name": "estimate_xyz_pcc",
#         "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
#         "slurm_cpus_per_task": num_cpus,
#         "slurm_array_parallelism": 100,
#         "slurm_time": 60,
#         "slurm_partition": "preempted",
#     }

#     if sbatch_filepath:
#         slurm_args.update(sbatch_to_submitit(sbatch_filepath))

#     executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
#     executor.update_parameters(**slurm_args)

#     click.echo(f"Submitting SLURM xyz PCC jobs with resources: {slurm_args}")
#     transforms_out_path = output_folder_path / "transforms_per_position"
#     transforms_out_path.mkdir(parents=True, exist_ok=True)
#     shifts_out_path = output_folder_path / "shifts_per_position"
#     shifts_out_path.mkdir(parents=True, exist_ok=True)

#     jobs = []
#     with submitit.helpers.clean_env(), executor.batch():
#         for input_position_dirpath in input_position_dirpaths:
#             job = executor.submit(
#                 estimate_tczyx,
#                 input_position_dirpath=input_position_dirpath,
#                 output_folder_path=transforms_out_path,
#                 output_shifts_path=shifts_out_path,
#                 channel_index=channel_index,
#                 phase_cross_corr_settings=phase_cross_corr_settings,
#                 verbose=verbose,
#             )
#             jobs.append(job)

#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     log_path = slurm_out_path / f"job_ids_{timestamp}.log"
#     with open(log_path, "w") as log_file:
#         for job in jobs:
#             log_file.write(f"{job.job_id}\n")

#     wait_for_jobs_to_finish(jobs)

#     transform_files = list(transforms_out_path.glob("*.npy"))

#     fov_transforms = {}
#     for file_path in transform_files:
#         fov_filename = file_path.stem
#         fov_transforms[fov_filename] = np.load(file_path).tolist()

#     # Remove the output folder
#     shutil.rmtree(transforms_out_path)

#     return fov_transforms
