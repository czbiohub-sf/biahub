from pathlib import Path
from typing import List

import click
import numpy as np
import submitit

from iohub import open_ome_zarr

from biahub.analysis.AnalysisSettings import RegistrationSettings
from biahub.analysis.register import apply_affine_transform, find_overlapping_volume
from biahub.cli.parsing import (
    config_filepath,
    local,
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


def rescale_voxel_size(affine_matrix, input_scale):
    return np.linalg.norm(affine_matrix, axis=1) * input_scale


@click.command()
@source_position_dirpaths()
@target_position_dirpaths()
@config_filepath()
@output_dirpath()
@local()
@sbatch_filepath()
def register(
    source_position_dirpaths: List[str],
    target_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
    local: bool,
    sbatch_filepath: Path,
):
    """
    Apply an affine transformation to a single position across T and C axes based on a registration config file.

    Start by generating an initial affine transform with `estimate-register`. Optionally, refine this transform with `optimize-register`. Finally, use `register`.

    >> biahub register -s source.zarr/*/*/* -t target.zarr/*/*/* -c config.yaml -o ./acq_name_registerred.zarr
    """

    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

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

    # if not local:
    #     monitor_jobs(affine_jobs + copy_jobs, affine_names + copy_names)
    # concatenate affine_jobs and copy_jobs
    job_ids = [job.job_id for job in affine_jobs + copy_jobs]

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))


if __name__ == "__main__":
    register()
