from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    _check_nan_n_zeros,
    estimate_resources,
    yaml_to_model,
)
from biahub.settings import FlatFieldCorrectionSettings


def flat_field_correction(
    zyx_data: np.ndarray,
    axis: int = 0,
    keepdims: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Apply flat field correction to the data.

    Parameters
    ----------
    zyx_data : np.ndarray
        The data to apply flat field correction to.
    axis : int
        The axis to compute the median on.
    keepdims : bool
        Whether to keep the dimensions of the static pattern. Default is True.

    Returns
    -------
    Tuple[np.ndarray, dict]
        A tuple containing the flat field corrected data and the static pattern statistics.
    """

    static_pattern = np.median(zyx_data, axis=axis, keepdims=keepdims)
    static_dict = {
        "shape": static_pattern.shape,
        "mean": static_pattern.mean(),
        "min": static_pattern.min(),
        "max": static_pattern.max(),
        "std": static_pattern.std(),
        "var": static_pattern.var(),
        "sum": static_pattern.sum(),
    }
    return zyx_data / static_pattern * static_pattern.mean(), static_dict


def flat_field_single_timepoint(
    input_data_path: Path,
    output_path: Path,
    channel_names: List[str],
    t_idx: int,
):
    """Apply flat field correction to selected channels and copy the rest at a single timepoint.

    Parameters
    ----------
    input_data_path : Path
        Path to the input position.
    output_path : Path
        Path to the output position.
    flat_field_channel_indices : list[int]
        Indices of channels to apply flat field correction to.
    """

    click.echo(f"Starting: input={input_data_path}, output={output_path}")
    position_key = input_data_path.parts[-3:]
    fov_name = "_".join(position_key)
    output_metadata_path = output_path.parent / "static_metadata" / fov_name
    output_metadata_path.mkdir(parents=True, exist_ok=True)

    with open_ome_zarr(input_data_path, mode="r") as input_dataset:
        _, C, _, _, _ = input_dataset.data.shape

        for c_idx in range(C):
            click.echo(f"[t={t_idx}, c={c_idx}] Reading data...")
            channel_name = input_dataset.channel_names[c_idx]
            zyx_data = np.asarray(input_dataset.data[t_idx, c_idx])

            if _check_nan_n_zeros(zyx_data):
                click.echo(f"[t={t_idx}, c={c_idx}] Skipped (all zeros or nans)")
                continue

            if channel_name in channel_names:
                click.echo(f"[t={t_idx}, c={c_idx}] Applying flat field correction...")
                zyx_data, static_dict = flat_field_correction(zyx_data)
                click.echo(f"[t={t_idx}, c={c_idx}] Flat field correction done")
                static_dict_path = (
                    output_metadata_path / f"static_dict_t_{t_idx}_{channel_name}.csv"
                )
                pd.DataFrame(static_dict).to_csv(static_dict_path, index=False)
                click.echo(
                    f"[t={t_idx}, c={c_idx}] Static dictionary saved to {static_dict_path}"
                )
            else:
                click.echo(f"[t={t_idx}, c={c_idx}] Copying channel as-is")
                zyx_data = np.asarray(zyx_data, dtype=np.float32)

            click.echo(f"[t={t_idx}, c={c_idx}] Writing to output...")
            with open_ome_zarr(output_path / Path(*position_key), mode="r+") as output_dataset:
                output_dataset[0][t_idx, c_idx] = zyx_data

            click.echo(f"[t={t_idx}, c={c_idx}] Done ({c_idx + 1}/{C} channels)")

        click.echo(f"[t={t_idx}] Completed all {C} channels")


def flat_field(
    input_position_dirpaths: List[str],
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

    Returns
    -------
    None
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
            "Must specify either 'channel_names' or 'flat_field_all: true' in config."
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
        shape=(1, C, Z, Y, X),
        ram_multiplier=15,
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "flat-field",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": 1,
        "slurm_array_parallelism": 100,
        "slurm_time": 360,
        "slurm_partition": "cpu",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    cluster = "local" if local else "slurm"

    # Prepare and submit jobs (one per FOV and T)
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path in input_position_dirpaths:
            for t_idx in range(T):
                jobs.append(
                    executor.submit(
                        flat_field_single_timepoint,
                        input_position_path,
                        output_dirpath,
                        channel_names,
                        t_idx,
                    )
                )

    job_ids = [job.job_id for job in jobs]
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    click.echo(
        f"Submitted {len(jobs)} jobs ({len(input_position_dirpaths)} FOVs x {T} timepoints)"
    )


@click.command("flat-field")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
def flat_field_correction_cli(
    input_position_dirpaths: List[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Apply flat field correction across T and selected C axes.

    >> biahub flat-field \\
        -i ./input.zarr/*/*/* \\
        -c ./flat_field_params.yml \\
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
