from pathlib import Path
from typing import List, Tuple

import ants
import click
import largestinteriorrectangle as lir
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import submitit

from iohub import open_ome_zarr

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.utils import (
    copy_n_paste_czyx,
    create_empty_hcs_zarr,
    estimate_resources,
    process_single_position_v2,
    yaml_to_model,
)
from biahub.settings import RegistrationSettings


def get_3D_rescaling_matrix(start_shape_zyx, scaling_factor_zyx=(1, 1, 1), end_shape_zyx=None):
    center_Y_start, center_X_start = np.array(start_shape_zyx)[-2:] / 2
    if end_shape_zyx is None:
        center_Y_end, center_X_end = (center_Y_start, center_X_start)
    else:
        center_Y_end, center_X_end = np.array(end_shape_zyx)[-2:] / 2

    scaling_matrix = np.array(
        [
            [scaling_factor_zyx[-3], 0, 0, 0],
            [
                0,
                scaling_factor_zyx[-2],
                0,
                -center_Y_start * scaling_factor_zyx[-2] + center_Y_end,
            ],
            [
                0,
                0,
                scaling_factor_zyx[-1],
                -center_X_start * scaling_factor_zyx[-1] + center_X_end,
            ],
            [0, 0, 0, 1],
        ]
    )
    return scaling_matrix


def get_3D_rotation_matrix(
    start_shape_zyx: Tuple, angle: float = 0.0, end_shape_zyx: Tuple = None
) -> np.ndarray:
    """
    Rotate Transformation Matrix

    Parameters
    ----------
    start_shape_zyx : Tuple
        Shape of the input
    angle : float, optional
        Angles of rotation in degrees
    end_shape_zyx : Tuple, optional
       Shape of output space

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    # TODO: make this 3D?
    center_Y_start, center_X_start = np.array(start_shape_zyx)[-2:] / 2
    if end_shape_zyx is None:
        center_Y_end, center_X_end = (center_Y_start, center_X_start)
    else:
        center_Y_end, center_X_end = np.array(end_shape_zyx)[-2:] / 2

    theta = np.radians(angle)

    rotation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [
                0,
                np.cos(theta),
                -np.sin(theta),
                -center_Y_start * np.cos(theta)
                + np.sin(theta) * center_X_start
                + center_Y_end,
            ],
            [
                0,
                np.sin(theta),
                np.cos(theta),
                -center_Y_start * np.sin(theta)
                - center_X_start * np.cos(theta)
                + center_X_end,
            ],
            [0, 0, 0, 1],
        ]
    )
    return rotation_matrix


def convert_transform_to_ants(T_numpy: np.ndarray):
    """Homogeneous 3D transformation matrix from numpy to ants

    Parameters
    ----------
    numpy_transform :4x4 homogenous matrix

    Returns
    -------
    Ants transformation matrix object
    """
    assert T_numpy.shape == (4, 4)

    T_ants_style = T_numpy[:, :-1].ravel()
    T_ants_style[-3:] = T_numpy[:3, -1]
    T_ants = ants.new_ants_transform(
        transform_type="AffineTransform",
    )
    T_ants.set_parameters(T_ants_style)

    return T_ants


def convert_transform_to_numpy(T_ants):
    """
    Convert the ants transformation matrix to numpy 3D homogenous transform

    Modified from Jordao's dexp code

    Parameters
    ----------
    T_ants : Ants transfromation matrix object

    Returns
    -------
    np.array
        Converted Ants to numpy array

    """

    T_numpy = T_ants.parameters.reshape((3, 4), order="F")
    T_numpy[:, :3] = T_numpy[:, :3].transpose()
    T_numpy = np.vstack((T_numpy, np.array([0, 0, 0, 1])))

    # Reference:
    # https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/
    # https://github.com/netstim/leaddbs/blob/a2bb3e663cf7fceb2067ac887866124be54aca7d/helpers/ea_antsmat2mat.m
    # T = original translation offset from A
    # T = T + (I - A) @ centering

    T_numpy[:3, -1] += (np.eye(3) - T_numpy[:3, :3]) @ T_ants.fixed_parameters

    return T_numpy


def apply_affine_transform(
    zyx_data: np.ndarray,
    matrix: np.ndarray,
    output_shape_zyx: Tuple,
    method="ants",
    interpolation: str = "linear",
    crop_output_slicing: bool = None,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    zyx_data : np.ndarray
        3D input array to be transformed
    matrix : np.ndarray
        3D Homogenous transformation matrix
    output_shape_zyx : Tuple
        output target zyx shape
    method : str, optional
        method to use for transformation, by default 'ants'
    interpolation: str, optional
        interpolation mode for ants, by default "linear"
    crop_output : bool, optional
        crop the output to the largest interior rectangle, by default False

    Returns
    -------
    np.ndarray
        registered zyx data
    """

    Z, Y, X = output_shape_zyx
    if crop_output_slicing is not None:
        Z_slice, Y_slice, X_slice = crop_output_slicing
        Z = Z_slice.stop - Z_slice.start
        Y = Y_slice.stop - Y_slice.start
        X = X_slice.stop - X_slice.start

    # TODO: based on the signature of this function, it should not be called on 4D array
    if zyx_data.ndim == 4:
        registered_czyx = np.zeros((zyx_data.shape[0], Z, Y, X), dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            registered_czyx[c] = apply_affine_transform(
                zyx_data[c],
                matrix,
                output_shape_zyx,
                method=method,
                interpolation=interpolation,
                crop_output_slicing=crop_output_slicing,
            )
        return registered_czyx
    else:
        # Convert nans to 0
        zyx_data = np.nan_to_num(zyx_data, nan=0)

        # NOTE: default set to ANTS apply_affine method until we decide we get a benefit from using cupy
        # The ants method on CPU is 10x faster than scipy on CPU. Cupy method has not been bencharked vs ANTs

        if method == "ants":
            # The output has to be a ANTImage Object
            empty_target_array = np.zeros((output_shape_zyx), dtype=np.float32)
            target_zyx_ants = ants.from_numpy(empty_target_array)

            T_ants = convert_transform_to_ants(matrix)

            zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
            registered_zyx = T_ants.apply_to_image(
                zyx_data_ants, reference=target_zyx_ants, interpolation=interpolation
            ).numpy()

        elif method == "scipy":
            registered_zyx = scipy.ndimage.affine_transform(zyx_data, matrix, output_shape_zyx)

        else:
            raise ValueError(f"Unknown method {method}")

        # Crop the output to the largest interior rectangle
        if crop_output_slicing is not None:
            registered_zyx = registered_zyx[Z_slice, Y_slice, X_slice]

    return registered_zyx


def find_lir(registered_zyx: np.ndarray, plot: bool = False) -> Tuple:
    registered_zyx = np.asarray(registered_zyx, dtype=np.bool)

    # Find the lir in YX at Z//2
    registered_yx = registered_zyx[registered_zyx.shape[0] // 2].copy()
    coords_yx = lir.lir(registered_yx)

    x, y, width, height = coords_yx
    x_start, x_stop = x, x + width
    y_start, y_stop = y, y + height
    x_slice = slice(x_start, x_stop)
    y_slice = slice(y_start, y_stop)

    # Iterate over ZX and ZY slices to find optimal Z cropping params
    _coords = []
    for _x in (x_start, x_start + (x_stop - x_start) // 2, x_stop):
        registered_zy = registered_zyx[:, y_slice, _x].copy()
        coords_zy = lir.lir(registered_zy)
        _, z, _, height = coords_zy
        z_start, z_stop = z, z + height
        _coords.append((z_start, z_stop))
    for _y in (y_start, y_start + (y_stop - y_start) // 2, y_stop):
        registered_zx = registered_zyx[:, _y, x_slice].copy()
        coords_zx = lir.lir(registered_zx)
        _, z, _, depth = coords_zx
        z_start, z_stop = z, z + depth
        _coords.append((z_start, z_stop))

    _coords = np.asarray(_coords)
    z_start = _coords.max(axis=0)[0]
    z_stop = _coords.min(axis=0)[1]
    z_slice = slice(z_start, z_stop)

    if plot:
        xy_corners = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
        rectangle_yx = plt.Polygon(
            xy_corners,
            closed=True,
            fill=None,
            edgecolor="r",
        )
        # Add the rectangle to the plot
        _, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(registered_yx)
        ax[0].add_patch(rectangle_yx)

        zx_corners = ((x, z), (x + width, z), (x + width, z + depth), (x, z + depth))
        rectangle_zx = plt.Polygon(
            zx_corners,
            closed=True,
            fill=None,
            edgecolor="r",
        )
        ax[1].imshow(registered_zx)
        ax[1].add_patch(rectangle_zx)
        plt.savefig("./lir.png")

    return (z_slice, y_slice, x_slice)


def find_overlapping_volume(
    input_zyx_shape: Tuple,
    target_zyx_shape: Tuple,
    transformation_matrix: np.ndarray,
    method: str = "LIR",
    plot: bool = False,
) -> Tuple:
    """
    Find the overlapping rectangular volume after registration of two 3D datasets

    Parameters
    ----------
    input_zyx_shape : Tuple
        shape of input array
    target_zyx_shape : Tuple
        shape of target array
    transformation_matrix : np.ndarray
        affine transformation matrix
    method : str, optional
        method of finding the overlapping volume, by default 'LIR'

    Returns
    -------
    Tuple
        ZYX slices of the overlapping volume after registration

    """

    # Make dummy volumes
    img1 = np.ones(tuple(input_zyx_shape), dtype=np.float32)
    img2 = np.ones(tuple(target_zyx_shape), dtype=np.float32)

    # Conver to ants objects
    target_zyx_ants = ants.from_numpy(img2.astype(np.float32))
    zyx_data_ants = ants.from_numpy(img1.astype(np.float32))

    ants_composed_matrix = convert_transform_to_ants(transformation_matrix)

    # Now apply the transform using this grid
    registered_zyx = ants_composed_matrix.apply_to_image(
        zyx_data_ants, reference=target_zyx_ants
    )
    if method == "LIR":
        click.echo("Starting Largest interior rectangle (LIR) search")
        # This is the *real* overlap mask
        mask = (registered_zyx.numpy() > 0) & (target_zyx_ants.numpy() > 0)

        # Now pass the mask to LIR (or call find_largest_valid_box)
        Z_slice, Y_slice, X_slice = find_lir(mask.astype(np.uint8), plot=plot)

    else:
        raise ValueError(f"Unknown method {method}")

    return (Z_slice, Y_slice, X_slice)


def rescale_voxel_size(affine_matrix, input_scale):
    return np.linalg.norm(affine_matrix, axis=1) * input_scale


@click.command("register")
@source_position_dirpaths()
@target_position_dirpaths()
@config_filepath()
@output_dirpath()
@local()
@sbatch_filepath()
@monitor()
def register_cli(
    source_position_dirpaths: List[str],
    target_position_dirpaths: List[str],
    config_filepath: Path,
    output_dirpath: str,
    local: bool,
    sbatch_filepath: Path,
    monitor: bool = True,
):
    """
    Apply an affine transformation to a single position across T and C axes based on a registration config file.

    Start by generating an initial affine transform with `estimate-register`. Optionally, refine this transform with `optimize-register`. Finally, use `register`.

    >> biahub register -s source.zarr/*/*/* -t target.zarr/*/*/* -c config.yaml -o ./acq_name_registerred.zarr
    """

    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)

    # Parse from the yaml file
    settings = yaml_to_model(config_filepath, RegistrationSettings)
    matrix = np.array(settings.affine_transform_zyx)
    keep_overhang = settings.keep_overhang

    # Calculate the output voxel size from the input scale and affine transform
    with open_ome_zarr(source_position_dirpaths[0]) as source_dataset:
        T, C, Z, Y, X = source_dataset.data.shape
        source_channel_names = source_dataset.channel_names
        source_shape_zyx = source_dataset.data.shape[-3:]
        source_voxel_size = source_dataset.scale[-3:]
        output_voxel_size = rescale_voxel_size(matrix[:3, :3], source_voxel_size)

    with open_ome_zarr(target_position_dirpaths[0]) as target_dataset:
        target_channel_names = target_dataset.channel_names
        target_shape_zyx = target_dataset.data.shape[-3:]

    click.echo("\nREGISTRATION PARAMETERS:")
    click.echo(f"Transformation matrix:\n{matrix}")
    click.echo(f"Voxel size: {output_voxel_size}")

    # Logic to parse time indices
    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]
    else:
        raise ValueError(f"Invalid time_indices type {type(settings.time_indices)}")

    output_channel_names = target_channel_names
    if target_position_dirpaths != source_position_dirpaths:
        output_channel_names += source_channel_names

    if not keep_overhang:
        # Find the largest interior rectangle
        click.echo("\nFinding largest overlapping volume between source and target datasets")
        Z_slice, Y_slice, X_slice = find_overlapping_volume(
            source_shape_zyx, target_shape_zyx, matrix
        )
        # TODO: start or stop may be None
        # Overwrite the previous target shape
        cropped_shape_zyx = (
            Z_slice.stop - Z_slice.start,
            Y_slice.stop - Y_slice.start,
            X_slice.stop - X_slice.start,
        )
        click.echo(f"Shape of cropped output dataset: {cropped_shape_zyx}\n")
    else:
        cropped_shape_zyx = target_shape_zyx
        Z_slice, Y_slice, X_slice = (
            slice(0, cropped_shape_zyx[-3]),
            slice(0, cropped_shape_zyx[-2]),
            slice(0, cropped_shape_zyx[-1]),
        )

    output_metadata = {
        "shape": (len(time_indices), len(output_channel_names)) + tuple(cropped_shape_zyx),
        "chunks": None,
        "scale": (1,) * 2 + tuple(output_voxel_size),
        "channel_names": output_channel_names,
        "dtype": np.float32,
    }

    # Create the output zarr mirroring source_position_dirpaths
    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in source_position_dirpaths],
        **output_metadata,
    )

    # Get the affine transformation matrix
    # NOTE: add any extra metadata if needed:
    extra_metadata = {
        "affine_transformation": {
            "transform_matrix": matrix.tolist(),
        }
    }

    affine_transform_args = {
        "matrix": matrix,
        "output_shape_zyx": target_shape_zyx,  # NOTE: this should be the shape of the original target dataset
        "crop_output_slicing": ([Z_slice, Y_slice, X_slice] if not keep_overhang else None),
        "interpolation": settings.interpolation,
        "extra_metadata": extra_metadata,
    }

    copy_n_paste_kwargs = {"czyx_slicing_params": ([Z_slice, Y_slice, X_slice])}

    # Estimate resources
    num_cpus, gb_ram = estimate_resources(shape=(T, C, Z, Y, X), ram_multiplier=5)

    # Prepare SLURM arguments
    slurm_out_path = Path(output_dirpath).parent / "slurm_output"
    slurm_args = {
        "slurm_job_name": "register",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
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
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    # NOTE: channels will not be processed in parallel
    # NOTE: the the source and target datastores may be the same (e.g. Hummingbird datasets)

    # apply affine transform to channels in the source datastore that should be registered
    # as given in the config file (i.e. settings.source_channel_names)
    affine_jobs = []
    affine_names = []
    with executor.batch():
        for input_position_path in source_position_dirpaths:
            for channel_name in source_channel_names:
                if channel_name not in settings.source_channel_names:
                    continue
                affine_job = executor.submit(
                    process_single_position_v2,
                    apply_affine_transform,
                    input_data_path=input_position_path,  # source store
                    output_path=output_dirpath,
                    time_indices=time_indices,
                    input_channel_idx=[source_channel_names.index(channel_name)],
                    output_channel_idx=[output_channel_names.index(channel_name)],
                    num_processes=int(slurm_args["slurm_cpus_per_task"]),
                    **affine_transform_args,
                )
                affine_jobs.append(affine_job)
                affine_names.append(input_position_path)

    # crop all channels that are not being registered and save them in the output zarr store
    # Note: when target and source datastores are the same we don't process channels which
    # were already registered in the previous step
    copy_jobs = []
    copy_names = []
    with executor.batch():
        for input_position_path in target_position_dirpaths:
            for channel_name in target_channel_names:
                if channel_name in settings.source_channel_names:
                    continue
                copy_job = executor.submit(
                    process_single_position_v2,
                    copy_n_paste_czyx,
                    input_data_path=input_position_path,  # target store
                    output_path=output_dirpath,
                    time_indices=time_indices,
                    input_channel_idx=[target_channel_names.index(channel_name)],
                    output_channel_idx=[output_channel_names.index(channel_name)],
                    num_processes=int(slurm_args["slurm_cpus_per_task"]),
                    **copy_n_paste_kwargs,
                )
                copy_jobs.append(copy_job)
                copy_names.append(input_position_path)

    # concatenate affine_jobs and copy_jobs
    job_ids = [job.job_id for job in affine_jobs + copy_jobs]

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    if monitor:
        monitor_jobs(affine_jobs + copy_jobs, affine_names + copy_names)


if __name__ == "__main__":
    register_cli()
