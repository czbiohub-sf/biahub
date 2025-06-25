import itertools
import os
import shutil

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

import click
import dask.array as da
import numpy as np
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr
from numpy.typing import ArrayLike
from pystackreg import StackReg
from scipy.fftpack import next_fast_len
from tqdm import tqdm
from waveorder.focus import focus_from_transverse_band

from biahub.cli.parsing import (
    config_filepaths,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.estimate_registration import (
    estimate_transform_from_beads,
    evaluate_transforms,
    save_transforms,
)
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    EstimateStabilizationSettings,
    FocusFindingSettings,
    PhaseCrossCorrSettings,
    StabilizationSettings,
    StackRegSettings,
)

NA_DET = 1.35
LAMBDA_ILL = 0.500


def remove_beads_fov_from_path_list(
    position_dirpaths: list[Path],
    skip_beads_fov: str,
) -> list[Path]:
    """
    Remove the beads FOV from the input data paths.

    Parameters
    ----------
    position_dirpaths : list[Path]
        Paths to the input position directories.
    skip_beads_fov : str
        Beads FOV to skip.

    Returns
    -------
    list[Path]
        Paths to the input position directories without the beads FOV.
    """
    if skip_beads_fov != '0':
        click.echo(f"Removing beads FOV {skip_beads_fov} from input data paths")
        position_dirpaths = [
            path for path in position_dirpaths if skip_beads_fov not in str(path)
        ]
    return position_dirpaths


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

    ref_img = match_shape(ref_img, shape)
    mov_img = match_shape(mov_img, shape)

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
) -> ArrayLike:
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
    try:
        target = np.asarray(source_channel_tzyx[t]).astype(np.float32)
        source = np.asarray(target_channel_tzyx[t]).astype(np.float32)

        shift = phase_cross_corr(target, source)
        if verbose:
            click.echo(f"Time {t}: shift = {shift}")

    except Exception as e:
        click.echo(f"Failed PCC at time {t}: {e}")
        return None

    dz, dy, dx = shift

    transform = np.eye(4)
    # Set the translation components of the transform

    transform[0, 3] = dx
    transform[1, 3] = dy
    transform[2, 3] = dz

    return transform


def estimate_xyz_stabilization_pcc_per_position(
    input_position_dirpath: Path,
    output_folder_path: Path,
    channel_index: int,
    crop_size_xy: list[int],
    t_reference: str = "first",
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
    crop_size_xy : list[int]
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

        position_filename = str(Path(*input_position_dirpath.parts[-3:])).replace("/", "_")
        np.save(
            output_folder_path / f"{position_filename}.npy",
            np.array(transforms, dtype=np.float32),
        )
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

    input_position_dirpaths = remove_beads_fov_from_path_list(
        input_position_dirpaths, phase_cross_corr_settings.skip_beads_fov
    )

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
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
        for input_position_dirpath in input_position_dirpaths:
            job = executor.submit(
                estimate_xyz_stabilization_pcc_per_position,
                input_position_dirpath=input_position_dirpath,
                output_folder_path=transforms_out_path,
                channel_index=channel_index,
                crop_size_xy=phase_cross_corr_settings.crop_size_xy,
                t_reference=phase_cross_corr_settings.t_reference,
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

    # Load the transforms
    transform_files = list(transforms_out_path.glob("*.npy"))

    fov_transforms = {}
    for file_path in transform_files:
        fov_filename = file_path.stem
        fov_transforms[fov_filename] = np.load(file_path).tolist()

    # Remove the output folder
    shutil.rmtree(transforms_out_path)

    return fov_transforms


def estimate_xyz_stabilization_with_beads(
    channel_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    cluster: str = "local",
    sbatch_filepath: Optional[Path] = None,
    output_folder_path: Path = None,
) -> list[ArrayLike]:
    """
    Estimate the xyz stabilization for a single position.

    Parameters
    ----------
    channel_tzyx : da.Array
        Source channel data.
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, print verbose output.
    cluster : str
        Cluster to use.
    sbatch_filepath : Path
        Path to the sbatch file.
    output_folder_path : Path
        Path to the output folder.

    Returns
    -------
    list[ArrayLike]
        List of the xyz stabilization for each timepoint.
    """

    (T, Z, Y, X) = channel_tzyx.shape

    if beads_match_settings.t_reference == "first":
        target_channel_tzyx = np.broadcast_to(channel_tzyx[0], (T, Z, Y, X)).copy()
    elif beads_match_settings.t_reference == "previous":
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
        "slurm_time": 30,
        "slurm_partition": "gpu",
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
                estimate_transform_from_beads,
                source_channel_tzyx=channel_tzyx,
                target_channel_tzyx=target_channel_tzyx,
                verbose=verbose,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
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

    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)

    # Load the transforms
    transforms = [np.eye(4).tolist()]
    for t in range(1, T):
        file_path = output_transforms_path / f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
            click.echo(f"Transform for timepoint {t} not found.")
        else:
            T_zyx_shift = np.load(file_path).tolist()
            transforms.append(T_zyx_shift)

    # Check if the number of transforms matches the number of timepoints
    if len(transforms) != T:
        raise ValueError(
            f"Number of transforms {len(transforms)} does not match number of timepoints {T}"
        )

    # Remove the output folder
    shutil.rmtree(output_transforms_path)

    return transforms


def estimate_xy_stabilization_per_position(
    input_position_dirpath: Path,
    output_folder_path: Path,
    df_z_focus_path: Path,
    channel_index: int,
    crop_size_xy: list[int, int],
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
    crop_size_xy : list[int, int]
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
        x_idx = slice(X // 2 - crop_size_xy[0] // 2, X // 2 + crop_size_xy[0] // 2)
        y_idx = slice(Y // 2 - crop_size_xy[1] // 2, Y // 2 + crop_size_xy[1] // 2)

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

    input_position_dirpaths = remove_beads_fov_from_path_list(
        input_position_dirpaths, stack_reg_settings.skip_beads_fov
    )

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
        "slurm_partition": "cpu",
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
        for input_position_dirpath in input_position_dirpaths:
            job = executor.submit(
                estimate_xy_stabilization_per_position,
                input_position_dirpath=input_position_dirpath,
                output_folder_path=output_transforms_path,
                df_z_focus_path=df_focus_path,
                channel_index=channel_index,
                crop_size_xy=stack_reg_settings.crop_size_xy,
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

    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)

    transforms_paths = list(output_transforms_path.glob("*.npy"))
    fov_transforms = {}

    for file_path in transforms_paths:
        fov_filename = file_path.stem
        fov_transforms[fov_filename] = np.load(file_path).tolist()

    shutil.rmtree(output_transforms_path)

    return fov_transforms


def estimate_z_focus_per_position(
    input_position_dirpath: Path,
    input_channel_indices: Tuple[int, ...],
    crop_size_xy: list[int, int],
    output_path_focus_csv: Path,
    output_path_transform: Path,
    verbose: bool = False,
) -> None:
    """
    Estimate the z-focus for each timepoint and channel.

    Parameters
    ----------
    input_position_dirpath : Path
        Path to the input position directory.
    input_channel_indices : Tuple[int, ...]
        Indices of the channels to process.
    crop_size_xy : list[int, int]
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
    position, time_idx, channel, focus_idx = [], [], [], []

    with open_ome_zarr(input_position_dirpath) as dataset:
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

            position.append(str(Path(*input_position_dirpath.parts[-3:])))
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

    position_filename = str(Path(*input_position_dirpath.parts[-3:])).replace("/", "_")
    output_csv = output_path_focus_csv / f"{position_filename}.csv"
    df.to_csv(output_csv, index=False)

    # Compute Z drifts
    z_focus_shift = [np.eye(4)]

    # Initialize the z-value
    z_val = focus_idx[0]

    for z_val_next in focus_idx[1:]:
        shift = np.eye(4)
        # Set the translation components of the transform
        shift[0, 3] = z_val_next - z_val
        z_focus_shift.append(shift)

    transform = np.array(z_focus_shift)

    # Save the transform
    output_path_transform.mkdir(parents=True, exist_ok=True)
    np.save(output_path_transform / f"{position_filename}.npy", transform)

    if verbose:
        click.echo(f"Saved Z transform matrices to {output_path_transform}")


def get_mean_z_positions(
    dataframe_path: Path,
    verbose: bool = False,
    method: Literal["mean", "median"] = "mean",
) -> None:
    """
    Get the mean or median z-focus for each timepoint.

    Parameters
    ----------
    dataframe_path : Path
        Path to the input focus CSV file.
    verbose : bool
        If True, print verbose output.
    method : Literal["mean", "median"]
        Method to use for averaging the z-focus.

    Returns
    -------
    np.ndarray
        Array of the mean or median z-focus for each timepoint.
    """
    df = pd.read_csv(dataframe_path)

    df = df.sort_values("time_idx")

    # TODO: this is a hack to deal with the fact that the focus finding function returns 0 if it fails
    df["focus_idx"] = df["focus_idx"].replace(0, np.nan).ffill()

    # Get the mean of positions for each time point
    if method == "mean":
        average_focus_idx = df.groupby("time_idx")["focus_idx"].mean().reset_index()
    elif method == "median":
        average_focus_idx = df.groupby("time_idx")["focus_idx"].median().reset_index()

    if verbose:
        import matplotlib.pyplot as plt

        plt.plot(average_focus_idx["focus_idx"], linestyle="--", label="mean of all positions")
        plt.xlabel("Time index")
        plt.ylabel("Focus index")
        plt.ylim(0, 100)
        plt.legend()
        plt.savefig(dataframe_path.parent / "z_drift.png")

    return average_focus_idx["focus_idx"].values


def estimate_z_stabilization(
    input_position_dirpaths: list[Path],
    output_folder_path: Path,
    focus_finding_settings: FocusFindingSettings,
    channel_index: int,
    sbatch_filepath: Optional[Path] = None,
    cluster: str = "local",
    verbose: bool = False,
    estimate_z_index: bool = False,
) -> dict[str, list[ArrayLike]]:
    """
    Estimate the z stabilization for a list of positions.
    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to the input position directories.
    output_folder_path : Path
        Path to the output folder.
    focus_finding_settings : FocusFindingSettings
        Settings for the focus finding.
    channel_index : int
        Index of the channel to process.
    sbatch_filepath : Path
        Path to the sbatch file.
    cluster : str
        Cluster to use.
    verbose : bool
        If True, print verbose output.
    estimate_z_index : bool
        If True, estimate the z index and save the focus csv without saving the transforms (for xy stabilization).

    Returns
    -------
    dict[str, list[ArrayLike]]
        Dictionary of the z stabilization for each position.
    """
    input_position_dirpaths = remove_beads_fov_from_path_list(
        input_position_dirpaths, focus_finding_settings.skip_beads_fov
    )

    output_folder_path.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    # Estimate resources from a sample dataset
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
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
        "slurm_partition": "cpu",
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
        for input_position_dirpath in input_position_dirpaths:
            job = executor.submit(
                estimate_z_focus_per_position,
                input_position_dirpath=input_position_dirpath,
                input_channel_indices=(channel_index,),
                crop_size_xy=focus_finding_settings.crop_size_xy,
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

    job_ids = [str(j.job_id) for j in jobs]
    wait_for_jobs_to_finish(job_ids)

    # Load the focus CSV files and concatenate them
    focus_csvs_path = list(output_folder_focus_path.glob("*.csv"))
    if len(focus_csvs_path) != len(input_position_dirpaths):
        click.echo(
            f"Warning: {len(focus_csvs_path)} focus CSV files found for {len(input_position_dirpaths)} input data paths."
        )
    df = pd.concat([pd.read_csv(f) for f in focus_csvs_path])

    # Check if the existing focus CSV file exists
    if Path(output_folder_path / "positions_focus.csv").exists():
        click.echo("Using existing focus CSV file.")
        df_old = pd.read_csv(output_folder_path / "positions_focus.csv")
        df = pd.concat([df, df_old])
        df = df.drop_duplicates(subset=["position", "time_idx"])
    df = df.sort_values(["position", "time_idx"])
    df.to_csv(output_folder_path / "positions_focus.csv", index=False)

    # Remove the output temporary folder
    shutil.rmtree(output_folder_focus_path)

    if estimate_z_index:
        shutil.rmtree(output_transforms_path)
        return

    if focus_finding_settings.average_across_wells:
        z_drift_offsets = get_mean_z_positions(
            dataframe_path=output_folder_path / "positions_focus.csv",
            method=focus_finding_settings.average_across_wells_method,
            verbose=verbose,
        )

        # Initialize the z-focus shift
        z_focus_shift = [np.eye(4)]
        z_val = z_drift_offsets[0]
        transform = {}

        # Compute the z-focus shift for each timepoint
        for z_val_next in z_drift_offsets[1:]:
            # Set the translation components of the transform
            shift = np.eye(4)
            shift[0, 3] = z_val_next - z_val
            z_focus_shift.append(shift)
        transform["average"] = np.array(z_focus_shift).tolist()

        if verbose:
            click.echo(f"Saving z focus shift matrices to {output_folder_path}")
            np.save(output_folder_path / "z_focus_shift.npy", transform["average"])

        return transform
    else:
        # Load the transforms
        transforms_paths = list(output_transforms_path.glob("*.npy"))
        fov_transforms = {}

        for file_path in transforms_paths:
            transform = np.load(file_path).tolist()
            fov_filename = file_path.stem
            fov_transforms[fov_filename] = transform

        # Remove the output temporary folder
        shutil.rmtree(output_transforms_path)

    return fov_transforms


def estimate_stabilization(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = False,
) -> None:
    """
    Estimate the stabilization matrices for a list of positions.

    Parameters
    ----------
    input_position_dirpaths : list[str]
        Paths to the input position directories.
    output_filepath : str
        Path to the output file.
    config_filepath : str
        Path to the configuration file.
    sbatch_filepath : str
        Path to the sbatch file.
    local : bool
        If True, run locally.

    Returns
    -------
    None

    Notes
    -----
    The verbose output will be saved at the same level as the output zarr.
    """

    # Load the settings
    config_filepath = Path(config_filepath)

    settings = yaml_to_model(config_filepath, EstimateStabilizationSettings)
    click.echo(f"Settings: {settings}")

    verbose = settings.verbose
    stabilization_estimation_channel = settings.stabilization_estimation_channel
    stabilization_type = settings.stabilization_type
    stabilization_method = settings.stabilization_method

    output_dirpath = Path(output_dirpath)
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Channel names to process
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        channel_names = dataset.channel_names
        voxel_size = dataset.scale
        channel_index = channel_names.index(stabilization_estimation_channel)
        T, C, Z, Y, X = dataset.data.shape

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Load the evaluation settings
    eval_transform_settings = settings.eval_transform_settings

    if "xyz" == stabilization_type:
        if stabilization_method == "focus-finding":
            click.echo(
                "Estimating xyz stabilization parameters with focus finding and stack registration"
            )

            z_transforms_dict = estimate_z_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                focus_finding_settings=settings.focus_finding_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            xy_transforms_dict = estimate_xy_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                stack_reg_settings=settings.stack_reg_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )

            try:

                for fov, xy_transforms in tqdm(
                    xy_transforms_dict.items(), desc="Processing FOVs"
                ):

                    z_transforms = np.asarray(z_transforms_dict[fov])
                    xy_transforms = np.asarray(xy_transforms)

                    if xy_transforms.shape[0] != z_transforms.shape[0]:
                        raise ValueError(
                            "The number of translation matrices and z drift matrices must be the same"
                        )

                    xyz_transforms = np.asarray(
                        [a @ b for a, b in zip(xy_transforms, z_transforms)]
                    ).tolist()

                    if eval_transform_settings:
                        xyz_transforms = evaluate_transforms(
                            transforms=xyz_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=False,
                        )
                        z_transforms = evaluate_transforms(
                            transforms=z_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=False,
                        )
                        xy_transforms = evaluate_transforms(
                            transforms=xy_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=False,
                        )

                    save_transforms(
                        model=model,
                        transforms=xyz_transforms,
                        output_filepath_settings=output_dirpath
                        / "xyz_stabilization_settings"
                        / f"{fov}.yml",
                        output_filepath_plot=output_dirpath
                        / "translation_plots"
                        / f"{fov}.png",
                        verbose=verbose,
                    )
                    save_transforms(
                        model=model,
                        transforms=z_transforms,
                        output_filepath_settings=output_dirpath
                        / "z_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                    )
                    save_transforms(
                        model=model,
                        transforms=xy_transforms,
                        output_filepath_settings=output_dirpath
                        / "xy_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                    )

            except Exception as e:
                click.echo(
                    f"Error estimating {stabilization_type} stabilization parameters: {e}"
                )

        elif stabilization_method == "beads":

            click.echo("Estimating xyz stabilization parameters with beads")
            with open_ome_zarr(input_position_dirpaths[0], mode="r") as beads_position:
                source_channels = beads_position.channel_names
                source_channel_index = source_channels.index(stabilization_estimation_channel)
                channel_tzyx = beads_position.data.dask_array()[:, source_channel_index]

            xyz_transforms = estimate_xyz_stabilization_with_beads(
                channel_tzyx=channel_tzyx,
                beads_match_settings=settings.beads_match_settings,
                affine_transform_settings=settings.affine_transform_settings,
                verbose=verbose,
                output_folder_path=output_dirpath,
                cluster=cluster,
                sbatch_filepath=sbatch_filepath,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )

            if eval_transform_settings:
                xyz_transforms = evaluate_transforms(
                    transforms=xyz_transforms,
                    shape_zyx=(Z, Y, X),
                    validation_window_size=eval_transform_settings.validation_window_size,
                    validation_tolerance=eval_transform_settings.validation_tolerance,
                    interpolation_window_size=eval_transform_settings.interpolation_window_size,
                    interpolation_type=eval_transform_settings.interpolation_type,
                    verbose=False,
                )

            save_transforms(
                model=model,
                transforms=xyz_transforms,
                output_filepath_settings=output_dirpath / "xyz_stabilization_settings.yml",
                verbose=verbose,
                output_filepath_plot=output_dirpath / "translation_plots" / "beads.png",
            )

        elif stabilization_method == "phase-cross-corr":
            click.echo("Estimating xyz stabilization parameters with phase cross correlation")

            xyz_transforms_dict = estimate_xyz_stabilization_pcc(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                phase_cross_corr_settings=settings.phase_cross_corr_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )

            try:
                for fov, xyz_transforms in tqdm(
                    xyz_transforms_dict.items(), desc="Processing FOVs"
                ):
                    if eval_transform_settings:
                        xyz_transforms = evaluate_transforms(
                            transforms=xyz_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=False,
                        )

                    save_transforms(
                        model=model,
                        transforms=xyz_transforms,
                        output_filepath_settings=output_dirpath
                        / "xyz_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                        output_filepath_plot=output_dirpath
                        / "translation_plots"
                        / f"{fov}.png",
                    )
            except Exception as e:
                click.echo(
                    f"Error estimating {stabilization_type} stabilization parameters: {e}"
                )

    # Estimate z drift
    if "z" == stabilization_type and stabilization_method == "focus-finding":
        click.echo("Estimating z stabilization parameters with focus finding")

        z_transforms_dict = estimate_z_stabilization(
            input_position_dirpaths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            channel_index=channel_index,
            focus_finding_settings=settings.focus_finding_settings,
            sbatch_filepath=sbatch_filepath,
            cluster=cluster,
            verbose=verbose,
        )

        model = StabilizationSettings(
            stabilization_type=settings.stabilization_type,
            stabilization_method=settings.stabilization_method,
            stabilization_estimation_channel=settings.stabilization_estimation_channel,
            stabilization_channels=settings.stabilization_channels,
            affine_transform_zyx_list=[],
            time_indices="all",
            output_voxel_size=voxel_size,
        )

        try:
            for fov, z_transforms in tqdm(z_transforms_dict.items(), desc="Processing FOVs"):
                if eval_transform_settings:
                    z_transforms = evaluate_transforms(
                        transforms=z_transforms,
                        shape_zyx=(Z, Y, X),
                        validation_window_size=eval_transform_settings.validation_window_size,
                        validation_tolerance=eval_transform_settings.validation_tolerance,
                        interpolation_window_size=eval_transform_settings.interpolation_window_size,
                        interpolation_type=eval_transform_settings.interpolation_type,
                        verbose=False,
                    )

                save_transforms(
                    model=model,
                    transforms=z_transforms,
                    output_filepath_settings=output_dirpath
                    / "z_stabilization_settings"
                    / f"{fov}.yml",
                    verbose=verbose,
                    output_filepath_plot=output_dirpath / "translation_plots" / f"{fov}.png",
                )
        except Exception as e:
            click.echo(f"Error estimating {stabilization_type} stabilization parameters: {e}")

    # Estimate yx drift
    if "xy" == stabilization_type:
        if stabilization_method == "focus-finding":
            click.echo(
                "Estimating xy stabilization parameters with focus finding and stack registration"
            )

            xy_transforms_dict = estimate_xy_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                stack_reg_settings=settings.stack_reg_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )
            try:
                for fov, xy_transforms in tqdm(
                    xy_transforms_dict.items(), desc="Processing FOVs"
                ):
                    if eval_transform_settings:
                        xy_transforms = evaluate_transforms(
                            transforms=xy_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=False,
                        )

                    save_transforms(
                        model=model,
                        transforms=xy_transforms,
                        output_filepath_settings=output_dirpath
                        / "xy_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                        output_filepath_plot=output_dirpath
                        / "translation_plots"
                        / f"{fov}.png",
                    )
            except Exception as e:
                click.echo(
                    f"Error estimating {stabilization_type} stabilization parameters: {e}"
                )


@click.command("estimate-stabilization")
@input_position_dirpaths()
@output_dirpath()
@config_filepaths()
@sbatch_filepath()
@local()
def estimate_stabilization_cli(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepaths: list[str],
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate the Z and/or XY timelapse stabilization matrices.

    This function estimates xy and z drifts and returns the affine matrices computed with focus finding, beads or phase cross correlation.

    Parameters
    ----------
    input_position_dirpaths : list[str]
        Paths to the input position directories.
    output_filepath : str
        Path to the output file.
    config_filepaths : list[str]
        Path to the configuration file.
    sbatch_filepath : str
        Path to the sbatch file.
    local : bool
        If True, run locally.

    Returns
    -------
    None

    Notes
    -----
    The verbose output will be saved at the same level as the output zarr.
    The stabilization matrices are saved as a yaml file.
    The translation plots are saved as a png file.
    The focus csv is saved as a csv file.

    ---------

    Example usage:
    biahub estimate-stabilization -i ./timelapse.zarr/0/0/0 -o ./stabilization.yml  -c ./config.yml -s ./sbatch.sh --local --verbose

    """
    if len(config_filepaths) == 1:
        config_filepath = Path(config_filepaths[0])
    else:
        raise ValueError(
            "Only one configuration file is supported for estimate_stabilization. Please provide a single configuration file."
        )

    estimate_stabilization(
        input_position_dirpaths=input_position_dirpaths,
        output_dirpath=output_dirpath,
        config_filepath=config_filepath,
        sbatch_filepath=sbatch_filepath,
        local=local,
    )


if __name__ == "__main__":
    estimate_stabilization_cli()
