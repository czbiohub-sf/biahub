import shutil

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Tuple, cast

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
from skimage.registration import phase_cross_correlation as sk_phase_cross_correlation
from waveorder.focus import focus_from_transverse_band

from biahub.cli.parsing import (
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import estimate_resources
from biahub.core.transform import Transform
from biahub.registration.utils import match_shape
from biahub.settings import (
    FocusFindingSettings,
    PhaseCrossCorrSettings,
    StackRegSettings,
)

NA_DET = 1.35
LAMBDA_ILL = 0.500


def phase_cross_correlation(
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
    moving -> reference
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

    return peak


def estimate(
    mov: da.Array,
    ref: da.Array,
    function_type: Literal["skimage", "custom"] = "custom",
    normalization: Optional[Literal["magnitude", "classic"]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Transform:
    """
    Estimate the transformation matrix from the shift.

    Parameters
    ----------
    mov : da.Array
        Moving image.
    ref : da.Array
        Reference image.
    function_type : Literal["skimage", "custom"]
        Function type to use for the phase cross correlation.
    normalization : Optional[Literal["magnitude", "classic"]]
        Normalization to use for the phase cross correlation.
    output_path : Optional[Path]
        Output path to save the plot.
    verbose : bool
        If True, print verbose output.
    """
    shift = shift_from_pcc(
        mov=mov,
        ref=ref,
        function_type=function_type,
        normalization=normalization,
        output_path=output_path,
        verbose=verbose,
    )

    if shift.ndim == 2:
        dy, dx = shift

        transform = np.eye(3)
        transform[1, 2] = dx
        transform[0, 2] = dy

    elif shift.ndim == 3:

        dz, dy, dx = shift

        transform = np.eye(4)
        transform[2, 3] = dx
        transform[1, 3] = dy
        transform[0, 3] = dz

    if verbose:
        click.echo(f"transform: {transform}")

    return Transform(matrix=transform), shift


def shift_from_pcc(
    mov: da.Array,
    ref: da.Array,
    function_type: Literal["skimage", "custom"] = "custom",
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

    if function_type == "skimage":
        if normalization is not None:
            normalization = "phase"
        shift, _, _ = sk_phase_cross_correlation(
            reference_image=ref,
            moving_image=mov,
            normalization=normalization,
        )
    elif function_type == "custom":
        shift = phase_cross_correlation(mov, ref, normalization=normalization)
    if verbose:
        click.echo(f"shift: {shift}")
    return shift


def get_crop_idx(X, Y, Z, phase_cross_corr_settings):
    """
    Get the crop indices for the phase cross correlation.

    Parameters
    ----------
    X : int
        X dimension.
    Y : int
    Z : int
        Z dimension.
    phase_cross_corr_settings : PhaseCrossCorrSettings
        Phase cross correlation settings.

    Returns
    -------
    slice
        Crop indices.
    """
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

    return x_idx, y_idx, z_idx


def estimate_tczyx(
    input_position_dirpath: Path,
    output_folder_path: Path,
    output_shifts_path: Path,
    channel_index: int,
    phase_cross_corr_settings: PhaseCrossCorrSettings,
    verbose: bool = False,
    mode: Literal["registration", "stabilization"] = "stabilization",
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
        data_tzyx = input_position.data.dask_array()[:, channel_index]
        T, Z, Y, X = data_tzyx.shape

    x_idx, y_idx, z_idx = get_crop_idx(X, Y, Z, phase_cross_corr_settings)
    data_tzyx_cropped = data_tzyx[:, z_idx, y_idx, x_idx]

    if mode == "stabilization":
        if phase_cross_corr_settings.t_reference == "first":
            ref_tzyx = np.broadcast_to(data_tzyx_cropped[0], data_tzyx_cropped.shape).copy()
        elif phase_cross_corr_settings.t_reference == "previous":
            ref_tzyx = np.roll(data_tzyx_cropped, shift=1, axis=0)
            ref_tzyx[0] = data_tzyx_cropped[0]
    elif mode == "registration":
        raise ValueError("Registration mode not implemented yet")

    mov_tzyx = data_tzyx_cropped

    position_filename = str(Path(*input_position_dirpath.parts[-3:])).replace("/", "_")

    transforms = []
    shifts = []
    output_path_corr = output_folder_path.parent / "corr_plots" / position_filename
    output_path_corr.mkdir(parents=True, exist_ok=True)

    for t in range(T):
        click.echo(f"Estimating PCC for timepoint {t}")
        if t == 0:
            transforms.append(np.eye(4).tolist())
            shifts.append((t, 0, 0, 0))
        else:
            transform, shift = estimate(
                mov=mov_tzyx[t],
                ref=ref_tzyx[t],
                function_type=phase_cross_corr_settings.function_type,
                normalization=phase_cross_corr_settings.normalization,
                output_path=output_path_corr / f"{t}.png",
                verbose=verbose,
            )
            transforms.append(transform)
            shifts.append((t, *shift))

        click.echo(f"Transform for timepoint {t}: {transforms[-1]}")

    np.save(
        output_folder_path / f"{position_filename}.npy",
        np.array(transforms, dtype=np.float32),
    )
    # save the shifts as a csv
    if verbose:
        shifts_df = pd.DataFrame(shifts, columns=["TimepointID", "ShiftZ", "ShiftY", "ShiftX"])
        shifts_df["TimepointID"] = shifts_df["TimepointID"].astype(int)
        shifts_df["ShiftZ"] = shifts_df["ShiftZ"].astype(float)
        shifts_df["ShiftY"] = shifts_df["ShiftY"].astype(float)
        shifts_df["ShiftX"] = shifts_df["ShiftX"].astype(float)
        shifts_df.to_csv(output_shifts_path / f"{position_filename}.csv", index=False)

        output_path_shift_plots = output_shifts_path / "plots"
        output_path_shift_plots.mkdir(parents=True, exist_ok=True)
        # plot_pcc_drifts(shifts_df, output_path_shift_plots, label=position_filename)

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
                estimate_tczyx,
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
