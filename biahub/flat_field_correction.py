from pathlib import Path

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    estimate_resources,
    get_submitit_cluster,
    yaml_to_model,
)
from biahub.settings import FlatFieldCorrectionSettings


def flat_field_correction(
    zyx_data: np.ndarray,
    axis: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Apply flat field correction by dividing out the median pattern along an axis.

    Parameters
    ----------
    zyx_data : np.ndarray
        The data to apply flat field correction to.
    axis : int
        The axis to compute the median along.

    Returns
    -------
    Tuple[np.ndarray, dict]
        A tuple containing the flat field corrected data and the static pattern statistics.
    """
    static_pattern = np.median(zyx_data, axis=axis)
    mean_val = static_pattern.mean()
    static_dict = {
        "mean": float(mean_val),
        "min": float(static_pattern.min()),
        "max": float(static_pattern.max()),
        "std": float(static_pattern.std()),
    }
    return zyx_data / static_pattern * mean_val, static_dict


def czyx_flat_field_correction(
    czyx_data: np.ndarray,
    channel_names: list[str] = None,
    all_channel_names: list[str] = None,
) -> np.ndarray:
    """Apply flat-field correction to a CZYX array.

    Parameters
    ----------
    czyx_data : np.ndarray
        Input CZYX array.
    channel_names : list[str], optional
        Channels to flat-field correct. If None, correct all.
    all_channel_names : list[str], optional
        All channel names in the dataset (for index lookup).

    Returns
    -------
    np.ndarray
        Flat-field corrected CZYX array.
    """
    output = np.empty_like(czyx_data, dtype=np.float32)
    for c_idx in range(czyx_data.shape[0]):
        zyx_data = czyx_data[c_idx]
        if channel_names is None or all_channel_names is None:
            output[c_idx], _ = flat_field_correction(zyx_data)
        elif all_channel_names[c_idx] in channel_names:
            output[c_idx], _ = flat_field_correction(zyx_data)
        else:
            output[c_idx] = zyx_data.astype(np.float32)
    return output


def flat_field(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Apply flat field correction across T and selected C axes.

    Parameters
    ----------
    input_position_dirpaths : List[str]
        Paths to the input position directories.
    config_filepath : Path
        Path to the configuration file.
    output_dirpath : str
        Path to the output directory.
    sbatch_filepath : str
        Path to the sbatch file.
    local : bool
        If True, run locally.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    # Load settings
    settings = yaml_to_model(config_filepath, FlatFieldCorrectionSettings)

    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        all_channel_names = input_dataset.channel_names
        T, C, Z, Y, X = input_dataset.data.shape
        scale = input_dataset.scale

    # Determine which channels to flat field correct
    if settings.channel_names is None:
        channel_names = all_channel_names
        click.echo(f"Flat fielding ALL channels: {all_channel_names}")
    elif settings.channel_names:
        for name in settings.channel_names:
            if name not in all_channel_names:
                raise click.ClickException(
                    f"Channel '{name}' not found in input dataset. "
                    f"Available channels: {all_channel_names}"
                )
        channel_names = settings.channel_names
        click.echo(f"Input channels: {all_channel_names}")
        click.echo(f"Flat field channels: {channel_names}")
        click.echo("Other channels will be copied as-is")
    else:
        raise click.ClickException(
            "Must specify either 'channel_names' or set channel_names to null in config."
        )

    # Create output zarr with all channels
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        channel_names=all_channel_names,
        shape=(T, C, Z, Y, X),
        chunks=None,
        scale=scale,
        version="0.5",
        dtype=np.float32,
    )

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

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path in input_position_dirpaths:
            output_position_path = output_dirpath / Path(*input_position_path.parts[-3:])
            jobs.append(
                executor.submit(
                    process_single_position,
                    czyx_flat_field_correction,
                    input_position_path,
                    output_position_path,
                    num_threads=slurm_args["slurm_cpus_per_task"],
                    channel_names=channel_names,
                    all_channel_names=all_channel_names,
                )
            )

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    click.echo(f"Submitted {len(jobs)} jobs ({len(input_position_dirpaths)} positions)")


@click.command("flat-field")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
def flat_field_correction_cli(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
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
    )


if __name__ == "__main__":
    flat_field_correction_cli()
