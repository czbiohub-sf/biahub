from pathlib import Path

import click
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    monitor,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.segment_organelles import extract_organelle_features_data
from biahub.settings import OrganelleFeatureExtractionSettings


def process_position_features(
    input_position_path: Path,
    labels_channel_idx: int,
    intensity_channel_idx: int,
    frangi_channel_idx: int | None,
    spacing: tuple,
    properties: list | None,
    extra_properties: list | None,
) -> pd.DataFrame:
    """
    Extract features from a single position across all timepoints.

    Parameters
    ----------
    input_position_path : Path
        Path to input zarr position
    labels_channel_idx : int
        Channel index for labels
    intensity_channel_idx : int
        Channel index for intensity
    frangi_channel_idx : int | None
        Channel index for frangi (optional)
    spacing : tuple
        Pixel spacing (Z, Y, X)
    properties : list | None
        Base properties to extract
    extra_properties : list | None
        Extra properties to extract

    Returns
    -------
    position_df : pd.DataFrame
        DataFrame with all features for this position
    """
    click.echo(f"Processing position: {input_position_path}")

    with open_ome_zarr(str(input_position_path), mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape
        fov_name = "/".join(input_position_path.parts[-3:])

        all_features = []

        for t_idx in range(T):
            click.echo(f"  Processing timepoint {t_idx}/{T}")

            # Load data
            labels_zyx = dataset.data[t_idx, labels_channel_idx]
            intensity_zyx = dataset.data[t_idx, intensity_channel_idx]
            frangi_zyx = None
            if frangi_channel_idx is not None:
                frangi_zyx = dataset.data[t_idx, frangi_channel_idx]

            # Extract features
            features_df = extract_organelle_features_data(
                labels_zyx=labels_zyx,
                intensity_zyx=intensity_zyx,
                frangi_zyx=frangi_zyx,
                spacing=spacing,
                properties=properties,
                extra_properties=extra_properties,
                fov_name=fov_name,
                t_idx=t_idx,
            )

            if not features_df.empty:
                all_features.append(features_df)

        if all_features:
            position_df = pd.concat(all_features, ignore_index=True)
            click.echo(f"  Extracted {len(position_df)} features total")
            return position_df
        else:
            click.echo("  No features extracted")
            return pd.DataFrame()


@click.command("extract-organelle-features")
@input_position_dirpaths()
@config_filepath()
@sbatch_filepath()
@local()
@monitor()
def extract_organelle_features_cli(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    sbatch_filepath: str | None = None,
    local: bool = False,
    monitor: bool = True,
):
    """
    Extract morphological features from organelle labels.

    >> biahub extract-organelle-features \\
        -i ./labels.zarr/*/*/* \\
        -c ./feature_extraction_params.yml
    """

    # Convert string paths to Path objects
    config_filepath = Path(config_filepath)
    settings = yaml_to_model(config_filepath, OrganelleFeatureExtractionSettings)

    output_csv_path = Path(settings.output_csv_path)
    slurm_out_path = output_csv_path.parent / "slurm_output"

    if sbatch_filepath is not None:
        sbatch_filepath = Path(sbatch_filepath)

    # Load the first position to get channel info
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        channel_names = input_dataset.channel_names

    # Map channel names to indices
    if settings.labels_channel not in channel_names:
        raise ValueError(
            f"Labels channel {settings.labels_channel} not found in dataset {channel_names}"
        )
    if settings.intensity_channel not in channel_names:
        raise ValueError(
            f"Intensity channel {settings.intensity_channel} not found in dataset {channel_names}"
        )

    labels_channel_idx = channel_names.index(settings.labels_channel)
    intensity_channel_idx = channel_names.index(settings.intensity_channel)
    frangi_channel_idx = None
    if settings.frangi_channel is not None:
        if settings.frangi_channel not in channel_names:
            raise ValueError(
                f"Frangi channel {settings.frangi_channel} not found in dataset {channel_names}"
            )
        frangi_channel_idx = channel_names.index(settings.frangi_channel)

    click.echo(f"Labels channel: {settings.labels_channel} (index {labels_channel_idx})")
    click.echo(
        f"Intensity channel: {settings.intensity_channel} (index {intensity_channel_idx})"
    )
    if frangi_channel_idx is not None:
        click.echo(f"Frangi channel: {settings.frangi_channel} (index {frangi_channel_idx})")

    # Estimate resources
    num_cpus, gb_ram_request = estimate_resources(shape=(T, C, Z, Y, X), ram_multiplier=10)
    slurm_time = 120

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "extract-organelle-features",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": slurm_time,
        "slurm_partition": "preempted",
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path in input_position_dirpaths:
            jobs.append(
                executor.submit(
                    process_position_features,
                    Path(input_position_path),
                    labels_channel_idx,
                    intensity_channel_idx,
                    frangi_channel_idx,
                    settings.spacing,
                    settings.properties,
                    settings.extra_properties,
                )
            )

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)

    # Aggregate results from all positions
    click.echo("Aggregating features from all positions...")
    all_features = []
    for job in jobs:
        result = job.result()
        if not result.empty:
            all_features.append(result)

    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_csv_path, index=False)
        click.echo(f"Saved {len(final_df)} features to {output_csv_path}")
    else:
        click.echo("No features extracted from any position")


if __name__ == "__main__":
    extract_organelle_features_cli()
