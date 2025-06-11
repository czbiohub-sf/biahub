import itertools
import os
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
from waveorder.focus import focus_from_transverse_band

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_filepath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, model_to_yaml, yaml_to_model
from biahub.estimate_registration import (
    _get_tform_from_beads,
    _interpolate_transforms,
    _validate_transforms,
    wait_for_jobs_to_finish,
)
from biahub.settings import EstimateStabilizationSettings, StabilizationSettings

NA_DET = 1.35
LAMBDA_ILL = 0.500


def estimate_position_focus(
    input_data_path: Path,
    input_channel_indices: Tuple[int, ...],
    crop_size_xy: list[int, int],
    output_path_focus_csv: Path,
    output_path_transform: Path,
    verbose: bool = False,
) -> None:
    position, time_idx, channel, focus_idx = [], [], [], []

    with open_ome_zarr(input_data_path) as dataset:
        channel_names = dataset.channel_names
        T, _, Z, Y, X = dataset[0].shape
        _, _, _, _, pixel_size = dataset.scale

        for tc_idx in itertools.product(range(T), input_channel_indices):
            data_zyx = dataset.data[tc_idx][
                :,
                Y // 2 - crop_size_xy[1] // 2 : Y // 2 + crop_size_xy[1] // 2,
                X // 2 - crop_size_xy[0] // 2 : X // 2 + crop_size_xy[0] // 2,
            ]

            # if the FOV is empty, set the focal plane to 0
            if np.sum(data_zyx) == 0:
                z_idx = 0
            else:
                z_idx = focus_from_transverse_band(
                    data_zyx,
                    NA_det=NA_DET,
                    lambda_ill=LAMBDA_ILL,
                    pixel_size=pixel_size,
                )
                click.echo(
                    f"Estimating focus for timepoint {tc_idx[0]} and channel {tc_idx[1]}: {z_idx}"
                )

            position.append(str(Path(*input_data_path.parts[-3:])))
            time_idx.append(tc_idx[0])
            channel.append(channel_names[tc_idx[1]])
            focus_idx.append(z_idx)

    df = pd.DataFrame(
        {
            "position": position,
            "time_idx": time_idx,
            "channel": channel,
            "focus_idx": focus_idx,
        }
    )
    output_path_focus_csv.mkdir(parents=True, exist_ok=True)
    if verbose:
        click.echo(f"Saving focus finding results to {output_path_focus_csv}")

    position_filename = str(Path(*input_data_path.parts[-3:])).replace("/", "_")
    output_csv = output_path_focus_csv / f"{position_filename}.csv"
    df.to_csv(output_csv, index=False)

    # ---- Generate and save Z transformation matrix per timepoint

    # Compute Z drifts

    z_focus_shift = [np.eye(4)]
    z_val = focus_idx[0]
    for z_val_next in focus_idx[1:]:
        shift = np.eye(4)
        shift[0, 3] = z_val_next - z_val
        z_focus_shift.append(shift)
    T_z_drift_mats = np.array(z_focus_shift)

    output_path_transform.mkdir(parents=True, exist_ok=True)
    np.save(output_path_transform / f"{position_filename}.npy", T_z_drift_mats)

    if verbose:
        click.echo(f"Saved Z transform matrices to {output_path_transform}")


def get_mean_z_positions(
    dataframe_path: Path, verbose: bool = False, method: Literal["mean", "median"] = "mean"
) -> None:
    df = pd.read_csv(dataframe_path)

    # Sort the DataFrame based on 'time_idx'
    df = df.sort_values("time_idx")

    # TODO: this is a hack to deal with the fact that the focus finding function returns 0 if it fails
    df["focus_idx"] = df["focus_idx"].replace(0, np.nan).ffill()

    # Get the mean of positions for each time point
    if method == "mean":
        average_focus_idx = df.groupby("time_idx")["focus_idx"].mean().reset_index()
    elif method == "median":
        average_focus_idx = df.groupby("time_idx")["focus_idx"].median().reset_index()
    else:
        raise ValueError("Unknown averaging method.")

    if verbose:
        import matplotlib.pyplot as plt

        # # Get the moving average of the focus_idx
        plt.plot(average_focus_idx["focus_idx"], linestyle="--", label="mean of all positions")
        plt.xlabel('Time index')
        plt.ylabel('Focus index')
        plt.ylim(0, 100)
        plt.legend()
        plt.savefig(dataframe_path.parent / "z_drift.png")

    return average_focus_idx["focus_idx"].values


def pad_to_shape(
    arr: ArrayLike, shape: Tuple[int, ...], mode: str, verbose: str = False, **kwargs
) -> ArrayLike:
    """Pads array to shape.
    from shimPy

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    shape : Tuple[int]
        Output shape.
    mode : str
        Padding mode (see np.pad).

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


def center_crop(arr: ArrayLike, shape: Tuple[int, ...], verbose: str = False) -> ArrayLike:
    """Crops the center of `arr`
    from shimPy
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


def _match_shape(img: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
    """Pad or crop array to match provided shape.
    from shimPy
    """

    if np.any(shape > img.shape):
        padded_shape = np.maximum(img.shape, shape)
        img = pad_to_shape(img, padded_shape, mode="reflect")

    if np.any(shape < img.shape):
        img = center_crop(img, shape)

    return img


def phase_cross_corr(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    maximum_shift: float = 1.2,
    normalization: bool = False,
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

    ref_img = _match_shape(ref_img, shape)
    mov_img = _match_shape(mov_img, shape)

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps
    del ref_img, mov_img

    prod = Fimg1 * Fimg2.conj()
    del Fimg1, Fimg2

    if normalization:
        norm = np.fmax(np.abs(prod), eps)
    else:
        norm = 1.0
    corr = np.fft.irfftn(prod / norm)
    del prod, norm

    corr = np.fft.fftshift(np.abs(corr))

    argmax = np.argmax(corr)
    peak = np.unravel_index(argmax, corr.shape)
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak))

    if verbose:
        click.echo(f"phase cross corr. peak at {peak}")

    return peak


def get_tform_from_pcc(
    t: int,
    source_channel_tzyx: da.Array,
    target_channel_tzyx: da.Array,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    try:
        # Get the target and source images
        target = np.asarray(source_channel_tzyx[t]).astype(np.float32)
        source = np.asarray(target_channel_tzyx[t]).astype(np.float32)

        shift = phase_cross_corr(target, source)
        if verbose:
            click.echo(f"Time {t}: shift = {shift}")

    except Exception as e:
        click.echo(f"Failed PCC at time {t}: {e}")
        return None

    dz, dy, dx = shift
    mat = np.eye(4)
    mat[0, 3] = dx
    mat[1, 3] = dy
    mat[2, 3] = dz
    return mat


def estimate_xyz_stabilization_pcc_per_position(
    input_data_path: Path,
    output_folder_path: Path,
    c_idx: int,
    crop_size_xy: list[int],
    t_reference: str = "first",
    verbose: bool = False,
) -> None:
    with open_ome_zarr(input_data_path) as input_position:
        channel_tzyx = input_position.data.dask_array()[:, c_idx]
        T, _, Y, X = channel_tzyx.shape

        x_idx = slice(X // 2 - crop_size_xy[0] // 2, X // 2 + crop_size_xy[0] // 2)
        y_idx = slice(Y // 2 - crop_size_xy[1] // 2, Y // 2 + crop_size_xy[1] // 2)

        channel_tzyx_cropped = channel_tzyx[:, :, y_idx, x_idx]

        if t_reference == "first":
            target_channel_tzyx = np.broadcast_to(
                channel_tzyx_cropped[0], channel_tzyx_cropped.shape
            ).copy()
        elif t_reference == "previous":
            target_channel_tzyx = np.roll(channel_tzyx_cropped, shift=1, axis=0)
            target_channel_tzyx[0] = channel_tzyx_cropped[0]
        else:
            raise ValueError("Invalid reference. Use 'first' or 'previous'.")

        source_channel_tzyx = channel_tzyx_cropped

        transforms = []

        for t in range(T):
            click.echo(f"Estimating PCC for timepoint {t}")
            if t == 0:
                transforms.append(np.eye(4).tolist())
            else:
                transforms.append(
                    get_tform_from_pcc(
                        t,
                        source_channel_tzyx,
                        target_channel_tzyx,
                        verbose=verbose,
                    )
                )
            click.echo(f"Transform for timepoint {t}: {transforms[-1]}")

        position_filename = str(Path(*input_data_path.parts[-3:])).replace("/", "_")
        np.save(
            output_folder_path / f"{position_filename}.npy",
            np.array(transforms, dtype=np.float32),
        )
        click.echo(f"Saved transforms for {position_filename}.")
    return transforms


def estimate_xyz_stabilization_pcc(
    input_data_paths: list[Path],
    output_folder_path: Path,
    c_idx: int = 0,
    crop_size_xy: list[int] = (800, 800),
    t_reference: str = "first",
    sbatch_filepath: Path = None,
    cluster: str = "local",
    verbose: bool = False,
) -> np.ndarray:

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    with open_ome_zarr(input_data_paths[0]) as dataset:
        shape = dataset.data.shape
        T, C, Y, X = shape

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, C, Y, X), ram_multiplier=16, max_num_cpus=16
    )

    slurm_args = {
        "slurm_job_name": "estimate_xyz_pcc",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 10,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM xyz PCC jobs with resources: {slurm_args}")
    transforms_out_path = output_folder_path / "transforms_per_position"
    transforms_out_path.mkdir(parents=True, exist_ok=True)

    jobs = []
    with executor.batch():
        for input_data_path in input_data_paths:
            job = executor.submit(
                estimate_xyz_stabilization_pcc_per_position,
                input_data_path=input_data_path,
                output_folder_path=transforms_out_path,
                c_idx=c_idx,
                crop_size_xy=crop_size_xy,
                t_reference=t_reference,
                verbose=verbose,
            )
            jobs.append(job)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)

    transform_files = list(transforms_out_path.glob("*.npy"))

    fov_transforms = {}
    for file_path in transform_files:
        fov_filename = file_path.stem
        fov_transforms[fov_filename] = np.load(file_path).tolist()

    shutil.rmtree(transforms_out_path)

    return fov_transforms


def estimate_xyz_stabilization_with_beads(
    channel_tzyx: da.Array,
    t_reference: str = "first",
    match_algorithm: str = 'hungarian',
    match_filter_angle_threshold: float = 0,
    transform_type: str = 'euclidean',
    xy: bool = False,
    verbose: bool = False,
    cluster: str = "local",
    sbatch_filepath: Optional[Path] = None,
    output_folder_path: Path = None,
):
    """
    Perform beads-based temporal registration of 4D data using affine transformations.

    This function calculates timepoint-specific affine transformations to align a source channel
    to a target channel in 4D (T, Z, Y, X) data. It validates, smooths, and interpolates transformations
    across timepoints for consistent registration.

    Parameters:
    - channel_tzyx (da.Array): 4D array (T, Z, Y, X) of the source channel (Dask array).
    - approx_tform (list): Initial approximate affine transform (4x4 matrix) for guiding registration.
    - num_processes (int): Number of parallel processes for transformation computation.
    - window_size (int): Size of the moving window for smoothing transformations.
    - tolerance (float): Maximum allowed difference between consecutive transformations for validation.
    - angle_threshold (int): Threshold for filtering outliers in detected bead matches (in degrees).
    - verbose (bool): If True, prints detailed logs of the registration process.

    Returns:
    - transforms (list): List of affine transformation matrices (4x4), one for each timepoint.
                         Invalid or missing transformations are interpolated.

    Notes:
    - Each timepoint is processed in parallel using a multiprocessing pool.
    - Transformations are smoothed with a moving average window and validated against a reference.
    - Missing transformations are interpolated linearly across timepoints.
    - Use verbose=True for detailed logging during registration.
    """

    (T, Z, Y, X) = channel_tzyx.shape

    approx_tform = np.eye(4)

    if t_reference == "first":
        target_channel_tzyx = np.broadcast_to(channel_tzyx[0], (T, Z, Y, X)).copy()
    elif t_reference == "previous":
        target_channel_tzyx = np.roll(channel_tzyx, shift=-1, axis=0)
        target_channel_tzyx[0] = channel_tzyx[0]

    else:
        raise ValueError("Invalid reference. Please use 'first' or 'previous as reference")

    # Compute transformations in parallel

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, 1, Z, Y, X), ram_multiplier=5, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_focus_z",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 5,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")
    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    jobs = []
    with executor.batch():
        for t in range(1, T, 1):
            job = executor.submit(
                _get_tform_from_beads,
                approx_tform=approx_tform,
                source_channel_tzyx=channel_tzyx,
                target_channel_tzyx=target_channel_tzyx,
                verbose=verbose,
                source_block_size=[8, 8, 8],
                source_threshold_abs=0.8,
                source_nms_distance=16,
                source_min_distance=0,
                target_block_size=[8, 8, 8],
                target_threshold_abs=0.8,
                target_nms_distance=16,
                target_min_distance=0,
                match_filter_angle_threshold=match_filter_angle_threshold,
                match_algorithm=match_algorithm,
                transform_type=transform_type,
                slurm=True,
                output_folder_path=output_transforms_path,
                t_idx=t,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")
    # Wait for all jobs to finish
    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)
    # Get list of .npy transform files
    # Load and collect all transform arrays
    transforms = [np.eye(4).tolist()]
    for t in range(1, T):
        file_path = output_transforms_path / f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
            click.echo(f"Transform for timepoint {t} not found. Using None.")
        else:
            T_zyx_shift = np.load(file_path).tolist()
            transforms.append(T_zyx_shift)

    # check if len(transforms) == T
    if len(transforms) != T:
        raise ValueError(
            f"Number of transforms {len(transforms)} does not match number of timepoints {T}"
        )

    return transforms


def estimate_xy_stabilization_per_position(
    input_data_path: Path,
    output_folder_path: Path,
    df_z_focus_path: Path,
    channel_index: int,
    crop_size_xy: list[int, int],
    t_reference: str = "previous",
    verbose: bool = False,
) -> np.ndarray:

    with open_ome_zarr(input_data_path) as input_position:
        T, _, _, Y, X = input_position.data.shape
        x_idx = slice(X // 2 - crop_size_xy[0] // 2, X // 2 + crop_size_xy[0] // 2)
        y_idx = slice(Y // 2 - crop_size_xy[1] // 2, Y // 2 + crop_size_xy[1] // 2)

        if verbose:
            click.echo(f"Reading focus index from {df_z_focus_path}")
        df = pd.read_csv(df_z_focus_path)
        pos_idx = str(Path(*input_data_path.parts[-3:]))
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

        T_zyx_shift = np.zeros((T_stackreg.shape[0], 4, 4))
        T_zyx_shift[:, 1:4, 1:4] = T_stackreg
        T_zyx_shift[:, 0, 0] = 1
        # save the transforms as
        position_filename = str(Path(*input_data_path.parts[-3:]))
        position_filename = position_filename.replace("/", "_")

        np.save(
            output_folder_path / f"{position_filename}.npy", T_zyx_shift.astype(np.float32)
        )

    return T_zyx_shift


def estimate_xy_stabilization(
    input_data_paths: list[Path],
    output_folder_path: Path,
    channel_index: int = 0,
    crop_size_xy: list[int] = (400, 400),
    t_reference: str = "previous",
    sbatch_filepath: Optional[Path] = None,
    cluster: str = "local",
    verbose: bool = False,
) -> np.ndarray:

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    # Estimate resources from a sample dataset
    with open_ome_zarr(input_data_paths[0]) as dataset:
        shape = dataset.data.shape  # (T, C, Z, Y, X)

    df_focus_path = output_folder_path / "positions_focus.csv"

    if df_focus_path.exists():
        click.echo("Using existing Z focus index file.")
    else:
        click.echo("Estimating Z focus positions...")

        estimate_z_stabilization(
            input_data_paths=input_data_paths,
            output_folder_path=output_folder_path,
            channel_index=channel_index,
            crop_size_xy=crop_size_xy,
            sbatch_filepath=sbatch_filepath,
            cluster=cluster,
            verbose=verbose,
            estimate_z_index=True,
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
    with executor.batch():
        for input_data_path in input_data_paths:
            job = executor.submit(
                estimate_xy_stabilization_per_position,
                input_data_path=input_data_path,
                output_folder_path=output_transforms_path,
                df_z_focus_path=df_focus_path,
                channel_index=channel_index,
                crop_size_xy=crop_size_xy,
                t_reference=t_reference,
                verbose=verbose,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    # Wait for all jobs to finish
    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)

    # Get list of .npy transform files
    transforms_paths = list(output_transforms_path.glob("*.npy"))
    # Load and collect all transform arrays
    fov_transforms = {}

    for file_path in transforms_paths:
        fov_filename = file_path.stem
        fov_transforms[fov_filename] = np.load(file_path).tolist()

    shutil.rmtree(output_transforms_path)

    return fov_transforms


def estimate_z_stabilization(
    input_data_paths: list[Path],
    output_folder_path: Path,
    channel_index: int,
    crop_size_xy: list[int],
    sbatch_filepath: Optional[Path] = None,
    cluster: str = "local",
    verbose: bool = False,
    estimate_z_index: bool = False,
    average_index: bool = False,
) -> np.ndarray:
    """
    Submit SLURM jobs to estimate per-position Z focus and return averaged drift matrices.
    """

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    # Estimate resources from a sample dataset
    with open_ome_zarr(input_data_paths[0]) as dataset:
        shape = dataset.data.shape  # (T, C, Z, Y, X)

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=16, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_focus_z",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 30,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")
    output_folder_focus_path = output_folder_path / "z_focus_positions"
    output_folder_focus_path.mkdir(parents=True, exist_ok=True)

    output_transforms_path = output_folder_path / "z_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    jobs = []

    with executor.batch():
        for input_data_path in input_data_paths:
            job = executor.submit(
                estimate_position_focus,
                input_data_path=input_data_path,
                input_channel_indices=(channel_index,),
                crop_size_xy=crop_size_xy,
                output_path_focus_csv=output_folder_focus_path,
                output_path_transform=output_transforms_path,
                verbose=verbose,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    # Wait for all jobs to finish
    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)

    # Aggregate results
    focus_csvs_path = list(output_folder_focus_path.glob("*.csv"))

    # Check length of focus_csvs_path
    if len(focus_csvs_path) == 0:
        click.echo("No focus CSV files found. Exiting.")
        return
    elif len(focus_csvs_path) != len(input_data_paths):
        click.echo(
            f"Warning: {len(focus_csvs_path)} focus CSV files found for {len(input_data_paths)} input data paths."
        )

    df = pd.concat([pd.read_csv(f) for f in focus_csvs_path])
    # sort by position and time_idx
    if Path(output_folder_path / "positions_focus.csv").exists():
        click.echo("Using existing focus CSV file.")
        # read the existing CSV file
        df_old = pd.read_csv(output_folder_path / "positions_focus.csv")
        # concatenate the new and old dataframes
        df = pd.concat([df, df_old])
        # drop duplicates
        df = df.drop_duplicates(subset=["position", "time_idx"])
        # sort by position and time_idx
    df = df.sort_values(["position", "time_idx"])
    # Save the results to a CSV file
    df.to_csv(output_folder_path / "positions_focus.csv", index=False)

    shutil.rmtree(output_folder_focus_path)

    if estimate_z_index:
        shutil.rmtree(output_transforms_path)
        return

    if average_index:
        # # Compute Z drifts
        z_drift_offsets = get_mean_z_positions(
            dataframe_path=output_folder_path / "positions_focus.csv",
            method="median",
            verbose=verbose,
        )

        z_focus_shift = [np.eye(4)]
        z_val = z_drift_offsets[0]
        transform = {}
        for z_val_next in z_drift_offsets[1:]:
            shift = np.eye(4)
            shift[0, 3] = z_val_next - z_val
            z_focus_shift.append(shift)
        transform["average"] = np.array(z_focus_shift).tolist()

        if verbose:
            click.echo(f"Saving z focus shift matrices to {output_folder_path}")
            np.save(output_folder_path / "z_focus_shift.npy", transform["average"])

        return transform
    else:
        # Get list of .npy transform files
        transforms_paths = list(output_transforms_path.glob("*.npy"))

        # Load and collect all transform arrays
        fov_transforms = {}

        for file_path in transforms_paths:
            T_zyx_shift = np.load(file_path).tolist()
            fov_filename = file_path.stem
            #
            fov_transforms[fov_filename] = T_zyx_shift

        shutil.rmtree(output_transforms_path)

    return fov_transforms

def plot_translations(transforms_zyx: np.ndarray, output_filepath: Path):
    z_transforms = transforms_zyx[:, 0, 3]  
    y_transforms = transforms_zyx[:, 1, 3] 
    x_transforms = transforms_zyx[:, 2, 3]
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(z_transforms)
    axs[0].set_title("Z-Translation")
    axs[1].plot(x_transforms)
    axs[1].set_title("X-Translation")
    axs[2].plot(y_transforms)
    axs[2].set_title("Y-Translation")
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()



def save_stabilization_settings(
    stabilization_type: str,
    stabilization_method: str,
    stabilization_estimation_channel: str,
    stabilization_channels: list[str],
    transforms: np.ndarray,
    voxel_size: list[float],
    output_filepath: Path):
    model = StabilizationSettings(
        stabilization_type=stabilization_type,
        stabilization_method=stabilization_method,
        stabilization_estimation_channel=stabilization_estimation_channel,
        stabilization_channels=stabilization_channels,
        affine_transform_zyx_list=transforms,
        time_indices="all",
        output_voxel_size=voxel_size,
    )
    model_to_yaml(model, output_filepath)


@click.command("estimate-stabilization")
@input_position_dirpaths()
@output_filepath()
@config_filepath()
@sbatch_filepath()
@local()
def estimate_stabilization_cli(
    input_position_dirpaths: List[str],
    output_filepath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate the Z and/or XY timelapse stabilization matrices.

    This function estimates xy and z drifts and returns the affine matrices per timepoint taking t=0 as reference saved as a yaml file.
    The level of verbosity can be controlled with the verbose flag.
    The size of the crop in xy can be specified with the crop-size-xy option.

    Example usage:
    biahub stabilization -i ./timelapse.zarr/0/0/0 -o ./stabilization.yml -y -z -b -v --crop-size-xy 300 300

    Note: the verbose output will be saved at the same level as the output zarr.
    """

    # Load the settings
    config_filepath = Path(config_filepath)

    settings = yaml_to_model(config_filepath, EstimateStabilizationSettings)
    click.echo(f"Settings: {settings}")

    verbose = settings.verbose
    crop_size_xy = settings.crop_size_xy
    estimate_stabilization_channel = settings.estimate_stabilization_channel
    stabilization_type = settings.stabilization_type
    stabilization_method = settings.stabilization_method
    skip_beads_fov = settings.skip_beads_fov
    average_across_wells = settings.average_across_wells

    if skip_beads_fov != '0':
        # Remove the beads FOV from the input data paths
        click.echo(f"Removing beads FOV {skip_beads_fov} from input data paths")
        input_position_dirpaths = [
            path for path in input_position_dirpaths if skip_beads_fov not in str(path)
        ]

    output_dirpath = output_filepath.parent
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Channel names to process
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        channel_names = dataset.channel_names
        voxel_size = dataset.scale
        channel_index = channel_names.index(estimate_stabilization_channel)
        T, C, Z, Y, X = dataset.data.shape

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    if "xyz" == stabilization_type:
        if stabilization_method == "focus-finding":
            click.echo("Estimating z stabilization parameters")
            z_transforms_dict = estimate_z_stabilization(
                input_data_paths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                crop_size_xy=crop_size_xy,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            click.echo("Estimating xy stabilization parameters")
            xy_transforms_dict = estimate_xy_stabilization(
                input_data_paths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                crop_size_xy=crop_size_xy,
                t_reference=settings.t_reference,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )
            os.makedirs(output_dirpath / "xyz_stabilization_settings", exist_ok=True)
            os.makedirs(output_dirpath / "z_stabilization_settings", exist_ok=True)
            os.makedirs(output_dirpath / "xy_stabilization_settings", exist_ok=True)
        
            # save each FOV separately
            for fov, xy_transforms in xy_transforms_dict.items():
                click.echo(f"Processing FOV {fov}")
                # Get the z drift matrices for the current FOV
                z_transforms = np.asarray(z_transforms_dict[fov])
                xy_transforms = np.asarray(xy_transforms)

                if xy_transforms.shape[0] != z_transforms.shape[0]:
                    raise ValueError(
                        "The number of translation matrices and z drift matrices must be the same"
                    )

                xyz_transforms = np.asarray(
                    [a @ b for a, b in zip(xy_transforms, z_transforms)]
                ).tolist()
                # Validate and filter transforms
                xyz_transforms = _validate_transforms(
                    transforms=xyz_transforms,
                    window_size=settings.affine_transform_validation_window_size,
                    tolerance=settings.affine_transform_validation_tolerance,
                    Z=Z,
                    Y=Y,
                    X=X,
                    verbose=verbose,
                )
                # Interpolate missing transforms
                xyz_transforms = _interpolate_transforms(
                    transforms=xyz_transforms,
                    window_size=settings.affine_transform_interpolation_window_size,
                    interpolation_type=settings.affine_transform_interpolation_type,
                    verbose=verbose,
                )
                output_filepath_fov = (
                    output_dirpath / "xyz_stabilization_settings" / f"{fov}.yml"
                )
                # Save the combined matrices
                model = StabilizationSettings(
                    stabilization_type=stabilization_type,
                    stabilization_method=stabilization_method,
                    stabilization_estimation_channel=estimate_stabilization_channel,
                    stabilization_channels=settings.stabilization_channels,
                    affine_transform_zyx_list=xyz_transforms,
                    time_indices="all",
                    output_voxel_size=voxel_size,
                )
                model_to_yaml(model, output_filepath_fov)

                model = StabilizationSettings(
                    stabilization_type="z",
                    stabilization_method=stabilization_method,
                    stabilization_estimation_channel=estimate_stabilization_channel,
                    stabilization_channels=settings.stabilization_channels,
                    affine_transform_zyx_list=z_transforms.tolist(),
                    time_indices="all",
                    output_voxel_size=voxel_size,
                )
                model_to_yaml(
                    model, output_dirpath / "z_stabilization_settings" / f"{fov}.yml"
                )
                model = StabilizationSettings(
                    stabilization_type="xy",
                    stabilization_method=stabilization_method,
                    stabilization_estimation_channel=estimate_stabilization_channel,
                    stabilization_channels=settings.stabilization_channels,
                    affine_transform_zyx_list=xy_transforms.tolist(),
                    time_indices="all",
                    output_voxel_size=voxel_size,
                )
                model_to_yaml(
                    model, output_dirpath / "xy_stabilization_settings" / f"{fov}.yml"
                )
                if verbose:
                    os.makedirs(output_dirpath / "translation_plots", exist_ok=True)
                    plot_translations(np.array(xyz_transforms) , output_dirpath / "translation_plots" / f"{fov}.png")

        elif stabilization_method == "beads":

            click.echo("Estimating xyz stabilization parameters with beads")
            with open_ome_zarr(input_position_dirpaths[0], mode="r") as beads_position:
                source_channels = beads_position.channel_names
                source_channel_index = source_channels.index(estimate_stabilization_channel)
                channel_tzyx = beads_position.data.dask_array()[:, source_channel_index]

            xyz_transforms = estimate_xyz_stabilization_with_beads(
                channel_tzyx=channel_tzyx,
                t_reference=settings.t_reference,
                match_algorithm=settings.match_algorithm,
                match_filter_angle_threshold=settings.match_filter_angle_threshold,
                transform_type=settings.affine_transform_type,
                verbose=verbose,
                output_folder_path=output_dirpath,
                cluster=cluster,
                sbatch_filepath=sbatch_filepath,
            )

            # Validate and filter transforms
            xyz_transforms = _validate_transforms(
                transforms=xyz_transforms,
                window_size=settings.affine_transform_validation_window_size,
                tolerance=settings.affine_transform_validation_tolerance,
                Z=Z,
                Y=Y,
                X=X,
                verbose=verbose,
            )
            # Interpolate missing transforms
            xyz_transforms = _interpolate_transforms(
                transforms=xyz_transforms,
                window_size=settings.affine_transform_interpolation_window_size,
                interpolation_type=settings.affine_transform_interpolation_type,
                verbose=verbose,
            )
            # Save the combined matrices
            model = StabilizationSettings(
                stabilization_type=stabilization_type,
                stabilization_method=stabilization_method,
                stabilization_estimation_channel=estimate_stabilization_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=xyz_transforms,
                time_indices="all",
                output_voxel_size=voxel_size,
            )
            model_to_yaml(model, output_dirpath / "xyz_stabilization_settings.yml")

            if verbose:
                os.makedirs(output_dirpath / "translation_plots", exist_ok=True)
                plot_translations(np.array(xyz_transforms) , output_dirpath / "translation_plots" / "beads.png")

        elif stabilization_method == "phase-cross-corr":
            click.echo("Estimating xyz stabilization parameters with phase cross correlation")
            xyz_transforms_dict = estimate_xyz_stabilization_pcc(
                input_data_paths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                c_idx=channel_index,
                crop_size_xy=crop_size_xy,
                t_reference=settings.t_reference,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )
            
            os.makedirs(output_dirpath / "xyz_stabilization_settings", exist_ok=True)
            for fov, xyz_transforms in xyz_transforms_dict.items():
                click.echo(f"Processing FOV {fov}")
                # Validate and filter transforms
                xyz_transforms = _validate_transforms(
                    transforms=xyz_transforms,
                    window_size=settings.affine_transform_validation_window_size,
                    tolerance=settings.affine_transform_validation_tolerance,
                    Z=Z,
                    Y=Y,
                    X=X,
                    verbose=verbose,
                )
                # Interpolate missing transforms
                xyz_transforms = _interpolate_transforms(
                    transforms=xyz_transforms,
                    window_size=settings.affine_transform_interpolation_window_size,
                    interpolation_type=settings.affine_transform_interpolation_type,
                    verbose=verbose,
                )
                output_filepath_fov = (
                    output_dirpath / "xyz_stabilization_settings" / f"{fov}.yml"
                )
                # Save the combined matrices
                model = StabilizationSettings(
                    stabilization_type=stabilization_type,
                    stabilization_method=stabilization_method,
                    stabilization_estimation_channel=estimate_stabilization_channel,
                    stabilization_channels=channel_names,
                    affine_transform_zyx_list=xyz_transforms,
                    time_indices="all",
                    output_voxel_size=voxel_size,
                )
                model_to_yaml(model, output_filepath_fov)
                if verbose:
                    os.makedirs(output_dirpath / "translation_plots", exist_ok=True)
                    plot_translations(np.array(xyz_transforms) , output_dirpath / "translation_plots" / f"{fov}.png")

    # Estimate z drift
    if "z" == stabilization_type and stabilization_method == "focus-finding":
        click.echo("Estimating z stabilization parameters")

        z_transforms_dict = estimate_z_stabilization(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            channel_index=channel_index,
            crop_size_xy=crop_size_xy,
            sbatch_filepath=sbatch_filepath,
            average_index=average_across_wells,
            cluster=cluster,
            verbose=verbose,
        )

        os.makedirs(output_dirpath / "z_stabilization_settings", exist_ok=True)
        # save each FOV separately
        try:
            for fov, z_transforms in z_transforms_dict.items():
                # Validate and filter transforms
                z_transforms = _validate_transforms(
                    transforms=z_transforms,
                    window_size=settings.affine_transform_validation_window_size,
                    tolerance=settings.affine_transform_validation_tolerance,
                    Z=Z,
                    Y=Y,
                    X=X,
                    verbose=verbose,
                )
                # Interpolate missing transforms
                z_transforms = _interpolate_transforms(
                    transforms=z_transforms,
                    window_size=settings.affine_transform_interpolation_window_size,
                    interpolation_type=settings.affine_transform_interpolation_type,
                    verbose=verbose,
                )
                output_filepath_fov = (
                    output_dirpath / "z_stabilization_settings" / f"{fov}.yml"
                )
                # Save the combined matrices
                model = StabilizationSettings(
                    stabilization_type=stabilization_type,
                    stabilization_method=stabilization_method,
                    stabilization_estimation_channel=estimate_stabilization_channel,
                    stabilization_channels=settings.stabilization_channels,
                    affine_transform_zyx_list=z_transforms,
                    time_indices="all",
                    output_voxel_size=voxel_size,
                )
                model_to_yaml(model, output_filepath_fov)

                if verbose:
                    os.makedirs(output_dirpath / "translation_plots", exist_ok=True)
                    plot_translations(np.array(z_transforms) , output_dirpath / "translation_plots" / f"{fov}.png")

        except Exception as e:
            click.echo(f"Error estimating {stabilization_type} stabilization parameters: {e}")

    # Estimate yx drift
    if "xy" == stabilization_type:
        if stabilization_method == "focus-finding":

            click.echo("Estimating xy stabilization parameters")
            xy_transforms_dict = estimate_xy_stabilization(
                input_data_paths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                crop_size_xy=crop_size_xy,
                t_reference=settings.t_reference,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )
            os.makedirs(output_dirpath / "xy_stabilization_settings", exist_ok=True)

            # save each FOV separately
            for fov, xy_transforms in xy_transforms_dict.items():
                # Validate and filter transforms
                xy_transforms = _validate_transforms(
                    transforms=xy_transforms,
                    window_size=settings.affine_transform_validation_window_size,
                    tolerance=settings.affine_transform_validation_tolerance,
                    Z=Z,
                    Y=Y,
                    X=X,
                    verbose=verbose,
                )
                # Interpolate missing transforms
                xy_transforms = _interpolate_transforms(
                    transforms=xy_transforms,
                    window_size=settings.affine_transform_interpolation_window_size,
                    interpolation_type=settings.affine_transform_interpolation_type,
                    verbose=verbose,
                )
                output_filepath_fov = (
                    output_dirpath / "xy_stabilization_settings" / f"{fov}.yml"
                )
                # Save the combined matrices
                model = StabilizationSettings(
                    stabilization_type=stabilization_type,
                    stabilization_method=stabilization_method,
                    stabilization_estimation_channel=estimate_stabilization_channel,
                    stabilization_channels=settings.stabilization_channels,
                    affine_transform_zyx_list=xy_transforms,
                    time_indices="all",
                    output_voxel_size=voxel_size,
                )
                model_to_yaml(model, output_filepath_fov)

                if verbose:
                    os.makedirs(output_dirpath / "translation_plots", exist_ok=True)
                    plot_translations(np.array(xy_transforms) , output_dirpath / "translation_plots" / f"{fov}.png")



if __name__ == "__main__":
    estimate_stabilization_cli()
