import os
import shutil

from datetime import datetime
from pathlib import Path

import ants
import click
import dask.array as da
import numpy as np
import submitit

from numpy.typing import ArrayLike
from skimage import filters

from biahub.cli.parsing import (
    sbatch_to_submitit,
)

from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.registration.transform import Transform
from biahub.cli.utils import _check_nan_n_zeros, estimate_resources
from biahub.registration.utils import (
    convert_transform_to_ants,
    convert_transform_to_numpy,
    find_lir,
)
from biahub.settings import (
    AffineTransformSettings,
    AntsRegistrationSettings,
)


def _optimize_registration(
    source_czyx: np.ndarray,
    target_czyx: np.ndarray,
    initial_tform: np.ndarray,
    source_channel_index: int | list = 0,
    target_channel_index: int = 0,
    crop: bool = False,
    target_mask_radius: float | None = None,
    clip: bool = False,
    sobel_fitler: bool = False,
    verbose: bool = False,
    slurm: bool = False,
    t_idx: int = 0,
    output_folder_path: str | None = None,
) -> Transform | None:
    """
    Optimize the affine transform between source and target channels using ANTs library.

    Parameters
    ----------
    source_czyx : np.ndarray
        Source channel data in CZYX format.
    target_czyx : np.ndarray
        Target channel data in CZYX format.
    initial_tform : np.ndarray
        Approximate estimate of the affine transform matrix, often obtained through manual registration, see `estimate-registration`.
    source_channel_index : int | list, optional
        Index or list of indices of source channels to be used for registration, by default 0.
    target_channel_index : int, optional
        Index of the target channel to be used for registration, by default 0.
    crop : bool, optional
        Whether to crop the source and target channels to the overlapping region as determined by the LIR algorithm, by default False.
    target_mask_radius : float | None, optional
        Radius of the circular mask which will be applied to the phase channel. By default None in which case no masking will be applied.
    clip : bool, optional
        Whether to clip the source and target channels to reasonable (hardcoded) values, by default False.
    sobel_fitler : bool, optional
        Whether to apply Sobel filter to the source and target channels, by default False.
    verbose : bool, optional
        Whether to print verbose output during registration, by default False.
    slurm : bool, optional
        Whether the function is running in a SLURM job, which will save the output to a file, by default False.
    t_idx : int, optional
        Time index for the registration, by default 0. Only used if `slurm` is True.
    output_folder_path : str | None, optional
        Path to the folder where the output transform will be saved if `slurm` is True, by default None.

    Returns
    -------
    Transform | None
        Optimized affine transform matrix or None if the input data contains NaN or zeros.

    Notes
    -----
    This function applies an initial affine transform to the source channels, crops them to the overlapping region with the target channel,
    clips the values, applies Sobel filtering if specified, and then optimizes the registration parameters using ANTs library.

    This function currently assumes that target channel is phase and source channels are fluorescence.
    If multiple source channels are provided, they will be summed, after clipping, filtering, and cropping, if enabled.
    """

    source_czyx = np.asarray(source_czyx).astype(np.float32)
    target_czyx = np.asarray(target_czyx).astype(np.float32)
    print(f"initial_tform: {initial_tform}")

    if target_mask_radius is not None and not (0 < target_mask_radius <= 1):
        raise ValueError(
            "target_mask_radius must be given as a fraction of image width, i.e. (0, 1]."
        )

    if _check_nan_n_zeros(source_czyx) or _check_nan_n_zeros(target_czyx):
        return None
    initial_tform = Transform(matrix=initial_tform)
    t_form_ants = initial_tform.to_ants()

    target_zyx = target_czyx[target_channel_index]
    if target_zyx.ndim != 3:
        raise ValueError(f"Expected 3D target channel, got shape {target_zyx.shape}")
    target_ants_pre_crop = ants.from_numpy(target_zyx)

    if not isinstance(source_channel_index, list):
        source_channel_index = [source_channel_index]
    source_channels = []
    for idx in source_channel_index:
        # Cropping, clipping, and filtering are applied after registration with initial_tform
        _source_channel = np.asarray(source_czyx[idx]).astype(np.float32)
        if _source_channel.ndim != 3:
            raise ValueError(f"Expected 3D source channel, got shape {_source_channel.shape}")
        source_channel = t_form_ants.apply_to_image(
            ants.from_numpy(_source_channel), reference=target_ants_pre_crop
        ).numpy()
        if source_channel.ndim != 3:
            raise ValueError(
                f"apply_to_image returned non-3D array: {source_channel.shape}.\n"
                "This is likely caused by mismatched input shape or invalid transform/reference."
            )
        source_channels.append(source_channel)

    _offset = np.zeros(3, dtype=np.float32)
    if crop:
        click.echo("Estimating crop for source and target channels to overlapping region...")
        mask = (target_zyx != 0) & (source_channels[0] != 0)

        # Can be refactored with code in cropping PR #88
        if target_mask_radius is not None:
            target_mask = np.zeros(target_zyx.shape[-2:], dtype=bool)

            y, x = np.ogrid[: target_mask.shape[-2], : target_mask.shape[-1]]
            center = (target_mask.shape[-2] // 2, target_mask.shape[-1] // 2)
            radius = int(target_mask_radius * min(center))

            target_mask[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2] = True
            mask *= target_mask

        z_slice, y_slice, x_slice = find_lir(mask.astype(np.uint8))
        click.echo(
            f"Cropping to region z={z_slice.start}:{z_slice.stop}, "
            f"y={y_slice.start}:{y_slice.stop}, "
            f"x={x_slice.start}:{x_slice.stop}"
        )

        _offset = np.asarray(
            [_s.start for _s in (z_slice, y_slice, x_slice)], dtype=np.float32
        )
        target_zyx = target_zyx[z_slice, y_slice, x_slice]
        source_channels = [_channel[z_slice, y_slice, x_slice] for _channel in source_channels]

    # TODO: hardcoded clipping limits
    if clip:
        click.echo("Clipping source and target channels to reasonable values...")
        target_zyx = np.clip(target_zyx, 0, 0.5)
        source_channels = [
            np.clip(_channel, 110, np.quantile(_channel, 0.99)) for _channel in source_channels
        ]

    if sobel_fitler:
        click.echo("Applying Sobel filter to source and target channels...")
        target_zyx = filters.sobel(target_zyx)
        source_channels = [filters.sobel(_channel) for _channel in source_channels]

    source_zyx = np.sum(source_channels, axis=0)
    target_ants = ants.from_numpy(target_zyx)
    source_ants = ants.from_numpy(source_zyx)

    click.echo("Optimizing registration parameters using ANTs...")
    reg = ants.registration(
        fixed=target_ants,
        moving=source_ants,
        type_of_transform="Similarity",
        aff_shrink_factors=(6, 3, 1),
        aff_iterations=(2100, 1200, 50),
        aff_smoothing_sigmas=(2, 1, 0),
        verbose=verbose,
    )

    tx_opt_mat = ants.read_transform(reg["fwdtransforms"][0])
    tx_opt_numpy = convert_transform_to_numpy(tx_opt_mat)
    print(f"tx_opt_numpy: {tx_opt_numpy}")
    # Account for tx_opt being estimated at a crop rather than starting at the origin,
    # i.e. (0, 0, 0) of the image.
    shift_to_roi_np = np.eye(4)
    shift_to_roi_np[:3, -1] = _offset
    shift_back_np = np.eye(4)
    shift_back_np[:3, -1] = -_offset
    composed_matrix = initial_tform.matrix @ shift_to_roi_np @ tx_opt_numpy @ shift_back_np

    if slurm:
        output_folder_path.mkdir(parents=True, exist_ok=True)
        np.save(output_folder_path / f"{t_idx}.npy", composed_matrix)
    print("composed_matrix", composed_matrix)
    composed_transform = Transform(matrix=composed_matrix)
    print("composed_transform", composed_transform)
    return composed_transform


def shrink_slice(s: slice, shrink_fraction: float = 0.1, min_width: int = 5) -> slice:
    """
    Shrink a slice by a fraction of its length.

    Parameters
    ----------
    s : slice
        The slice to shrink.
    shrink_fraction : float
        The fraction of the slice to shrink.
    min_width : int
        The minimum width of the slice.

    Returns
    -------
    slice
        The shrunk slice.
    Notes
    -----
    If the slice is too small, return the original slice.

    """
    start = s.start or 0
    stop = s.stop or 0
    length = stop - start
    if length <= min_width:
        return slice(start, stop)

    shrink = int(length * shrink_fraction)
    new_start = start + shrink
    new_stop = stop - shrink
    if new_stop <= new_start:
        return slice(start, stop)
    return slice(new_start, new_stop)


def ants_registration(
    source_data_tczyx: da.Array,
    target_data_tczyx: da.Array,
    source_channel_index: int | list[int],
    target_channel_index: int,
    ants_registration_settings: AntsRegistrationSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    cluster: str = 'local',
    sbatch_filepath: Path = None,
) -> list[Transform]:
    """
    Perform ants registration of two volumetric image channels.

    This function calculates timepoint-specific affine transformations to align a source channel
    to a target channel in 4D (T, Z, Y, X) data. It validates, smooths, and interpolates transformations
    across timepoints for consistent registration.

    Parameters
    ----------
    source_data_tczyx : da.Array
       4D array (T, C, Z, Y, X) of the source channel (Dask array).
    target_data_tczyx : da.Array
       4D array (T, C, Z, Y, X) of the target channel (Dask array).
    source_channel_index : int | list[int]
        Index of the source channel.
    target_channel_index : int
        Index of the target channel.
    ants_registration_settings : AntsRegistrationSettings
        Settings for the ants registration.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, prints detailed logs of the registration process.
    output_folder_path : Path
        Path to the output folder.
    cluster : str
        Cluster to use.
    sbatch_filepath : Path
        Path to the sbatch file.

    Returns
    -------
    list[Transform]
        List of affine transformation matrices (4x4), one for each timepoint.
        Invalid or missing transformations are interpolated.

    Notes
    -----
    Each timepoint is processed in parallel using submitit executor.
    Use verbose=True for detailed logging during registration. The verbose output will be saved at the same level as the output zarr.
    """
    T, C, Z, Y, X = source_data_tczyx.shape
    initial_tform = np.asarray(affine_transform_settings.approx_transform)

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, 2, Z, Y, X), ram_multiplier=16, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_registration",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 30,
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
 
    click.echo(f"Submitting SLURM estimate regstration jobs with resources: {slurm_args}")
    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    click.echo('Computing registration transforms...')
    # NOTE: ants is mulitthreaded so no need for multiprocessing here
    # Submit jobs
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for t in range(T):
            job = executor.submit(
                _optimize_registration,
                source_data_tczyx[t],
                target_data_tczyx[t],
                initial_tform=initial_tform,
                source_channel_index=source_channel_index,
                target_channel_index=target_channel_index,
                crop=True,
                target_mask_radius=0.8,
                clip=True,
                sobel_fitler=ants_registration_settings.sobel_filter,
                verbose=verbose,
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

    wait_for_jobs_to_finish(jobs)

    # Load the transforms
    transforms = []
    for t in range(T):
        file_path = output_transforms_path / f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
            click.echo(f"Transform for timepoint {t} not found.")
        else:
            T_zyx_shift = np.load(file_path)
            print("T_zyx_shift", T_zyx_shift)
            transforms.append(T_zyx_shift.tolist())

    if len(transforms) != T:
        raise ValueError(
            f"Number of transforms {len(transforms)} does not match number of timepoints {T}"
        )

    # Remove the output temporary folder
    shutil.rmtree(output_transforms_path)

    return transforms


