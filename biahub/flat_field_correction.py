from pathlib import Path

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    estimate_resources,
    get_output_paths,
    get_submitit_cluster,
    resolve_ome_zarr_version,
    yaml_to_model,
)
from biahub.settings import FlatFieldCorrectionSettings


def flat_field_correction(zyx_data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Apply flat field correction by dividing out the median pattern along an axis.

    Parameters
    ----------
    zyx_data : np.ndarray
        The data to apply flat field correction to.
    axis : int
        The axis to compute the median along.

    Returns
    -------
    np.ndarray
        Flat-field corrected data, normalised so the mean of the static pattern
        is preserved.
    """
    static_pattern = np.median(zyx_data, axis=axis)
    return zyx_data / static_pattern * static_pattern.mean()


def _czyx_flat_field(czyx_data: np.ndarray, target_indices: list[int]) -> np.ndarray:
    """Apply flat-field correction to selected channels of a CZYX volume.

    Channels listed in ``target_indices`` are corrected; the rest are
    passed through unchanged (cast to float32 to match the output dtype).
    """
    out = np.empty_like(czyx_data, dtype=np.float32)
    target = set(target_indices)
    for c in range(czyx_data.shape[0]):
        if c in target:
            out[c] = flat_field_correction(czyx_data[c])
        else:
            out[c] = czyx_data[c].astype(np.float32)
    return out


def flat_field(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True,
):
    """Apply flat field correction across T and selected C axes.

    Parameters
    ----------
    input_position_dirpaths : List[str]
        Paths to the input position directories.
    config_filepath : Path
        Path to the configuration file.
    output_dirpath : str
        Path to the output directory.
    sbatch_filepath : str, optional
        Path to the SLURM batch file.
    local : bool, optional
        If True, run locally instead of submitting to SLURM.
    monitor : bool, optional
        If True, monitor the submitted jobs.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    output_position_paths = get_output_paths(input_position_dirpaths, output_dirpath)

    # Load settings
    settings = yaml_to_model(config_filepath, FlatFieldCorrectionSettings)

    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        all_channel_names = input_dataset.channel_names
        T, C, Z, Y, X = input_dataset.data.shape
        scale = input_dataset.scale

    # Determine which channels to flat field correct
    if settings.channel_names is None:
        target_channel_names = all_channel_names
        click.echo(f"Flat fielding ALL channels: {all_channel_names}")
    elif settings.channel_names:
        for name in settings.channel_names:
            if name not in all_channel_names:
                raise click.ClickException(
                    f"Channel '{name}' not found in input dataset. "
                    f"Available channels: {all_channel_names}"
                )
        target_channel_names = settings.channel_names
        click.echo(f"Input channels: {all_channel_names}")
        click.echo(f"Flat field channels: {target_channel_names}")
        click.echo("Other channels will be copied as-is")
    else:
        raise click.ClickException(
            "Must specify either 'channel_names' or set channel_names to null in config."
        )

    target_indices = [all_channel_names.index(name) for name in target_channel_names]

    # Create output zarr with all channels
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        channel_names=all_channel_names,
        shape=(T, C, Z, Y, X),
        chunks=None,
        scale=scale,
        version=resolve_ome_zarr_version(
            input_position_dirpaths[0], settings.output_ome_zarr_version
        ),
        dtype=np.float32,
    )

    flat_field_args = {
        "target_indices": target_indices,
        "extra_metadata": {"flat_field_correction": settings.model_dump()},
    }

    # Estimate resources
    num_cpus, gb_ram = estimate_resources(
        shape=(T, C, Z, Y, X),
        ram_multiplier=5,
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "flat-field",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 360,
        "slurm_partition": "cpu",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    cluster = get_submitit_cluster(local)

    # Prepare and submit jobs (one per position)
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting SLURM jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths, strict=True
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    _czyx_flat_field,
                    input_position_path,
                    output_position_path,
                    num_processes=int(slurm_args["slurm_cpus_per_task"]),
                    **flat_field_args,
                )
            )

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("flat-field")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
@monitor()
def flat_field_correction_cli(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True,
):
    """Apply flat field correction across T and selected C axes.

    >>> biahub flat-field \
        -i ./input.zarr/*/*/* \
        -c ./flat_field_params.yml \
        -o ./output.zarr
    """
    flat_field(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        local=local,
        monitor=monitor,
    )


if __name__ == "__main__":
    flat_field_correction_cli()
