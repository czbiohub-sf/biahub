from pathlib import Path

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position
from scipy.linalg import svd
from scipy.spatial.transform import Rotation

from biahub.cli import utils
from biahub.cli.disk import check_disk_space_with_du
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepaths,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    copy_n_paste_czyx,
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
    resolve_ome_zarr_version,
    yaml_to_model,
)
from biahub.register import convert_transform_to_ants
from biahub.settings import StabilizationSettings


def apply_stabilization_transform(
    zyx_data: np.ndarray,
    list_of_shifts: list[np.ndarray],
    input_time_index: int,
    output_shape: tuple[int, int, int] = None,
):
    """Apply stabilization transformations to 3D or 4D volumetric data.

    Parameters
    ----------
    zyx_data : np.ndarray
        Input 3D (Z, Y, X) or 4D (C, Z, Y, X) volumetric data.
    list_of_shifts : list[np.ndarray]
        List of transformation matrices (one per time index).
    input_time_index : int
        Time index corresponding to the transformation to apply.
    output_shape : tuple[int, int, int], optional
        Desired shape of the output stabilized volume.

    Returns
    -------
    np.ndarray
        The stabilized 3D or 4D volume.
    """
    import ants

    if output_shape is None:
        output_shape = zyx_data.shape[-3:]

    tx_shifts = convert_transform_to_ants(list_of_shifts[input_time_index])

    if zyx_data.ndim == 4:
        stabilized_czyx = np.zeros((zyx_data.shape[0],) + output_shape, dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            stabilized_czyx[c] = apply_stabilization_transform(
                zyx_data[c], list_of_shifts, input_time_index, output_shape
            )
        return stabilized_czyx
    else:
        click.echo(
            f"shifting matrix with input_time_index:{input_time_index} \n"
            f"{list_of_shifts[input_time_index]}"
        )
        target_zyx_ants = ants.from_numpy(np.zeros((output_shape), dtype=np.float32))

        zyx_data = np.nan_to_num(zyx_data, nan=0)
        zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
        stabilized_zyx = tx_shifts.apply_to_image(
            zyx_data_ants, reference=target_zyx_ants
        ).numpy()

    return stabilized_zyx


def _load_settings(config_filepath: Path) -> StabilizationSettings:
    return yaml_to_model(config_filepath, StabilizationSettings)


def _compute_output_shape(
    settings: StabilizationSettings,
    input_shape: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    """Compute stabilization output shape, handling rotation-based Y/X swap."""
    T, C, Z, Y, X = input_shape

    combined_mats = np.array(settings.affine_transform_zyx_list)
    R_matrix = combined_mats[0][:3, :3]
    U, _, Vt = svd(R_matrix)
    R_pure = U @ Vt
    euler_angles = Rotation.from_matrix(R_pure).as_euler("xyz", degrees=True)

    if np.isclose(euler_angles[0], 90, atol=10):
        out_Y, out_X = X, Y
    else:
        out_Y, out_X = Y, X

    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    return (len(time_indices), C, Z, out_Y, out_X)


def _resolve_time_indices(settings: StabilizationSettings, T: int) -> list[int]:
    if settings.time_indices == "all":
        return list(range(T))
    elif isinstance(settings.time_indices, list):
        return settings.time_indices
    elif isinstance(settings.time_indices, int):
        return [settings.time_indices]


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    settings: StabilizationSettings,
) -> tuple[tuple[int, int, int, int, int], list[str]]:
    """Create the empty stabilized output plate.

    Returns the input (T, C, Z, Y, X) shape and channel names.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as dataset:
        channel_names = dataset.channel_names
        input_shape = dataset.data.shape

    output_shape = _compute_output_shape(settings, input_shape)

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(p).parts[-3:] for p in input_position_dirpaths],
        channel_names=channel_names,
        shape=output_shape,
        chunks=None,
        scale=settings.output_voxel_size,
        dtype=np.float32,
        version=resolve_ome_zarr_version(
            input_position_dirpaths[0], settings.output_ome_zarr_version
        ),
        metadata_sources=input_plate,
    )

    return input_shape, channel_names


def stabilize(
    input_position_dirpaths: list[str],
    output_dirpath: str,
    config_filepaths: list[str],
    sbatch_filepath: str = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
):
    """Stabilize a timelapse dataset by applying spatial transformations.

    Parameters
    ----------
    input_position_dirpaths : list[str]
        Paths to input positions.
    output_dirpath : str
        Path to output zarr directory.
    config_filepaths : list[str]
        Paths to YAML configuration files with transformation settings.
    sbatch_filepath : str, optional
        SBATCH filepath to override default SLURM settings.
    cluster : str, optional
        Execution cluster: 'slurm', 'local', or 'debug'.
    monitor : bool, optional
        Monitor submitted SLURM jobs.
    init_only : bool, optional
        Only initialize the output store and exit.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    settings = _load_settings(config_filepaths[0])
    input_shape, channel_names = _init_output_plate(
        input_position_dirpaths, output_dirpath, settings
    )
    T, C, Z, Y, X = input_shape

    output_shape = _compute_output_shape(settings, input_shape)
    num_cpus, gb_ram = estimate_resources(
        shape=output_shape, ram_multiplier=16, max_num_cpus=16
    )
    echo_resources(num_cpus, num_cpus * gb_ram, 20)

    if init_only:
        click.echo(f"Initialized {output_dirpath} ({len(input_position_dirpaths)} positions)")
        return

    time_indices = _resolve_time_indices(settings, T)
    _, _, out_Z, out_Y, out_X = output_shape

    output_position_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)

    if not check_disk_space_with_du(
        input_path=input_position_dirpaths[0],
        output_path=output_dirpath,
        margin=1.1,
        verbose=True,
    ):
        raise RuntimeError(f"Not enough disk space to store the output at {output_dirpath}")

    slurm_args = {
        "slurm_job_name": "stabilize",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 20,
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
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
            if len(config_filepaths) > 1:
                fov = "_".join(Path(input_position_path).parts[-3:])
                config_filepath = [p for p in config_filepaths if fov in str(p)][0]
            else:
                config_filepath = config_filepaths[0]
            pos_settings = _load_settings(Path(config_filepath))
            combined_mats = np.array(pos_settings.affine_transform_zyx_list)
            stabilize_zyx_args = {"list_of_shifts": combined_mats}

            copy_n_paste_kwargs = {
                "czyx_slicing_params": [
                    slice(0, out_Z),
                    slice(0, out_Y),
                    slice(0, out_X),
                ]
            }

            for channel_name in channel_names:
                ch_idx = channel_names.index(channel_name)
                if channel_name in pos_settings.stabilization_channels:
                    job = executor.submit(
                        process_single_position,
                        apply_stabilization_transform,
                        input_position_path=input_position_path,
                        output_position_path=output_position_path,
                        input_time_indices=time_indices,
                        output_shape=(out_Z, out_Y, out_X),
                        input_channel_indices=[[ch_idx]],
                        output_channel_indices=[[ch_idx]],
                        num_workers=int(slurm_args["slurm_cpus_per_task"]),
                        **stabilize_zyx_args,
                    )
                else:
                    job = executor.submit(
                        process_single_position,
                        copy_n_paste_czyx,
                        input_position_path=input_position_path,
                        output_position_path=output_position_path,
                        input_time_indices=time_indices,
                        input_channel_indices=[[ch_idx]],
                        output_channel_indices=[[ch_idx]],
                        num_workers=int(slurm_args["slurm_cpus_per_task"]),
                        **copy_n_paste_kwargs,
                    )
                jobs.append(job)

    job_ids = [job.job_id for job in jobs]

    slurm_out_path.mkdir(exist_ok=True)
    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # DebugExecutor is lazy: run each job in the foreground.
    if resolved_cluster == "debug":
        for job, _path in zip(jobs, input_position_dirpaths, strict=False):
            job.wait()
        click.echo("Stabilization complete")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("stabilize")
@input_position_dirpaths()
@output_dirpath()
@config_filepaths()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
def stabilize_cli(
    input_position_dirpaths: list[str],
    output_dirpath: str,
    config_filepaths: list[str],
    sbatch_filepath: str,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
):
    r"""Apply stabilization transforms to a timelapse dataset.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub stabilize -i ./input.zarr/*/*/* -c ./transforms.yml -o ./output.zarr

    \b
    Initialize the output plate only (e.g. before Nextflow fan-out):
    >>> biahub stabilize --init -i ./input.zarr/*/*/* -c ./transforms.yml -o ./output.zarr

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub stabilize --cluster debug -i ./input.zarr/A/1/0 -c ./transforms.yml -o ./output.zarr


    """  # noqa: D301
    stabilize(
        input_position_dirpaths=input_position_dirpaths,
        output_dirpath=output_dirpath,
        config_filepaths=config_filepaths,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
    )


if __name__ == "__main__":
    stabilize_cli()
