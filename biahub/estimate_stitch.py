from pathlib import Path

import click
import pandas as pd
import submitit

from iohub import open_ome_zarr

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import input_position_dirpaths, local, monitor, output_filepath
from biahub.cli.utils import model_to_yaml
from biahub.settings import ProcessingSettings, StitchSettings
from biahub.stitch import (
    cleanup_shifts,
    compute_total_translation,
    consolidate_zarr_fov_shifts,
    estimate_zarr_fov_shifts,
    get_grid_rows_cols,
)


def write_config_file(
    shifts: pd.DataFrame,
    output_filepath: str,
    channel: str,
    fliplr: bool,
    flipud: bool,
    rot90: int,
):
    total_translation_dict = shifts.apply(
        lambda row: [float(row['shift-y'].round(2)), float(row['shift-x'].round(2))], axis=1
    ).to_dict()

    settings = StitchSettings(
        channels=[channel],
        preprocessing=ProcessingSettings(fliplr=fliplr, flipud=flipud, rot90=rot90),
        postprocessing=ProcessingSettings(),
        total_translation=total_translation_dict,
    )
    model_to_yaml(settings, output_filepath)


def cleanup_and_write_shifts(
    output_filepath, channel, fliplr, flipud, rot90, csv_filepath, pixel_size_um
):
    cleanup_shifts(csv_filepath, pixel_size_um)
    shifts = compute_total_translation(csv_filepath)
    write_config_file(shifts, output_filepath, channel, fliplr, flipud, rot90)


@click.command("estimate-stitch")
@input_position_dirpaths()
@output_filepath()
@click.option(
    "--channel",
    required=True,
    type=str,
    help="Channel to use for estimating stitch parameters",
)
@click.option(
    "--percent-overlap", "-p", required=True, type=float, help="Percent overlap between images"
)
@click.option("--fliplr", is_flag=True, help="Flip images left-right before stitching")
@click.option("--flipud", is_flag=True, help="Flip images up-down before stitching")
@click.option(
    "--rot90",
    default=0,
    type=int,
    help="rotate the images 90 counterclockwise n times before stitching",
)
@click.option(
    "--add_offset",
    is_flag=True,
    help="add the offset to estimated shifts, needed for OPS experiments",
)
@local()
@monitor()
def estimate_stitch_cli(
    input_position_dirpaths: list[Path],
    output_filepath: str,
    channel: str,
    percent_overlap: float,
    fliplr: bool,
    flipud: bool,
    rot90: int,
    add_offset: bool,
    local: bool,
    monitor: bool,
):
    """
    Estimate stitching parameters for positions in wells of a zarr store.
    Position names must follow the naming format XXXYYY, e.g. 000000, 000001, 001000, etc.
    as created by the Micro-manager Tile Creator: https://micro-manager.org/Micro-Manager_User's_Guide#positioning
    Assumes all wells have the save FOV grid layout.

    >>> biahub estimate-stitch -i ./input.zarr/*/*/* -o ./stitch_params.yml --channel DAPI --percent-overlap 0.05
    """
    if not (0 <= percent_overlap <= 1):
        raise ValueError("Percent overlap must be between 0 and 1")

    input_zarr_path = Path(*input_position_dirpaths[0].parts[:-3])
    output_filepath = Path(output_filepath)
    csv_filepath = (
        output_filepath.parent
        / f"stitch_shifts_{input_zarr_path.name.replace('.zarr', '.csv')}"
    )

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        if channel not in dataset.channel_names:
            raise ValueError(f"Channel {channel} not found in input zarr store")
        tcz_idx = (0, dataset.channel_names.index(channel), dataset.data.shape[-3] // 2)
        pixel_size_um = dataset.scale[-1]
    if pixel_size_um == 1.0:
        response = input(
            'The pixel size is equal to the default value of 1.0 um. ',
            'Inaccurate pixel size will affect stitching outlier removal. ',
            'Continue? [y/N]: ',
        )
        if response.lower() != 'y':
            return

    # here we assume that all wells have the same fov grid
    click.echo('Indexing input zarr store')
    wells = list(set([Path(*p.parts[-3:-1]) for p in input_position_dirpaths]))
    fov_names = set([p.name for p in input_position_dirpaths])
    grid_rows, grid_cols = get_grid_rows_cols(fov_names)

    # account for non-square grids
    row_fov_pairs, col_fov_pairs = [], []
    for col in grid_cols:
        for row0, row1 in zip(grid_rows[:-1], grid_rows[1:]):
            fov0 = col + row0
            fov1 = col + row1
            if fov0 in fov_names and fov1 in fov_names:
                row_fov_pairs.append((fov0, fov1))
    for row in grid_rows:
        for col0, col1 in zip(grid_cols[:-1], grid_cols[1:]):
            fov0 = col0 + row
            fov1 = col1 + row
            if fov0 in fov_names and fov1 in fov_names:
                col_fov_pairs.append((fov0, fov1))

    slurm_out_path = output_filepath.parent / "slurm_output"
    csv_dirpath = (
        output_filepath.parent / 'raw_shifts' / input_zarr_path.name.replace('.zarr', '')
    )
    csv_dirpath.mkdir(parents=True, exist_ok=False)

    estimate_shift_params = {
        "tcz_index": tcz_idx,
        "percent_overlap": percent_overlap,
        "fliplr": fliplr,
        "flipud": flipud,
        "rot90": rot90,
        "add_offset": add_offset,
    }

    slurm_args = {
        "slurm_job_name": "estimate-shift",
        "slurm_mem_per_cpu": "8G",
        "slurm_cpus_per_task": 1,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 10,
        "slurm_partition": "preempted",
    }

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    click.echo('Estimating FOV shifts...')
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)
    estimate_jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for well_name in wells:
            for direction, fovs in zip(("row", "col"), (row_fov_pairs, col_fov_pairs)):
                for fov0, fov1 in fovs:
                    fov0_zarr_path = Path(input_zarr_path, well_name, fov0)
                    fov1_zarr_path = Path(input_zarr_path, well_name, fov1)
                    estimate_jobs.append(
                        executor.submit(
                            estimate_zarr_fov_shifts,
                            direction=direction,
                            output_dirname=csv_dirpath,
                            **estimate_shift_params,
                            fov0_zarr_path=fov0_zarr_path,
                            fov1_zarr_path=fov1_zarr_path,
                        )
                    )
    estimate_job_ids = [job.job_id for job in estimate_jobs]

    slurm_args = {
        "slurm_job_name": "consolidate-shifts",
        "slurm_mem_per_cpu": "8G",
        "slurm_cpus_per_task": 1,
        "slurm_time": 10,
        "slurm_partition": "preempted",
        "slurm_dependency": f"afterok:{estimate_job_ids[0]}:{estimate_job_ids[-1]}",
    }

    click.echo('Consolidating FOV shifts...')
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)
    with submitit.helpers.clean_env():
        consolidate_job_id = executor.submit(
            consolidate_zarr_fov_shifts,
            input_dirname=csv_dirpath,
            output_filepath=csv_filepath,
        ).job_id

    slurm_args = {
        "slurm_job_name": "cleanup-shifts",
        "slurm_mem_per_cpu": "8G",
        "slurm_cpus_per_task": 1,
        "slurm_time": 10,
        "slurm_partition": "preempted",
        "slurm_dependency": f"afterok:{consolidate_job_id}",
    }

    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)
    with submitit.helpers.clean_env():
        cleanup_job_id = executor.submit(
            cleanup_and_write_shifts,
            output_filepath,
            channel,
            fliplr,
            flipud,
            rot90,
            csv_filepath,
            pixel_size_um,
        )
    job_ids = [job.job_id for job in estimate_jobs + consolidate_job_id + cleanup_job_id]

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    if monitor:
        monitor_jobs(estimate_jobs + consolidate_job_id + cleanup_job_id, wells)


if __name__ == "__main__":
    estimate_stitch_cli()
