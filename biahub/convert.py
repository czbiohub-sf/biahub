from pathlib import Path
from typing import List, Optional

import click

from biahub.cli.parsing import (
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
)
from biahub.cli.utils import yaml_to_model
from biahub.concatenate import concatenate
from biahub.settings import ConcatenateSettings, ConvertSettings


@click.command("convert")
@input_position_dirpaths()
@output_dirpath()
@click.option(
    "--config-filepath",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    default=None,
    help="Optional YAML config for chunks/shards. If not provided, uses input chunks and no sharding.",
)
@sbatch_filepath()
@local()
@monitor()
def convert_cli(
    input_position_dirpaths: List[Path],
    output_dirpath: Path,
    config_filepath: Optional[Path] = None,
    sbatch_filepath: Optional[str] = None,
    local: bool = False,
    monitor: bool = False,
) -> None:
    """
    Convert Zarr V2 OME-NGFF v0.4 to V3 OME-NGFF v0.5.

    This command converts HCS plate stores from Zarr V2 format (OME-NGFF v0.4)
    to Zarr V3 format (OME-NGFF v0.5). Conversion is parallelized per position.

    Examples
    --------
    Basic conversion with defaults:

    >> biahub convert -i ./input_v2.zarr/*/*/* -o ./output_v3.zarr

    With custom chunks and sharding:

    >> biahub convert -i ./input_v2.zarr/*/*/* -o ./output_v3.zarr -c config.yaml

    Local execution (no SLURM):

    >> biahub convert -i ./input_v2.zarr/*/*/* -o ./output_v3.zarr --local

    With SLURM config and monitoring:

    >> biahub convert -i ./input_v2.zarr/*/*/* -o ./output_v3.zarr \\
        -c config.yaml -sb slurm_config.sh -m
    """
    click.echo("Starting Zarr V2 to V3 Conversion")

    if config_filepath:
        click.echo(f"Loading config from: {config_filepath}")
        convert_settings = yaml_to_model(config_filepath, ConvertSettings)
        chunks_czyx = list(convert_settings.chunks[1:]) if convert_settings.chunks else None
        shards_ratio = (
            list(convert_settings.shards_ratio) if convert_settings.shards_ratio else None
        )
    else:
        click.echo("Using default settings (input chunks, no sharding)")
        chunks_czyx = None
        shards_ratio = None

    concat_settings = ConcatenateSettings(
        concat_data_paths=[str(p) for p in input_position_dirpaths],
        channel_names=["all"] * len(input_position_dirpaths),
        time_indices="all",
        X_slice="all",
        Y_slice="all",
        Z_slice="all",
        chunks_czyx=chunks_czyx,
        shards_ratio=shards_ratio,
        ensure_unique_positions=False,
        output_ome_zarr_version="0.5",
    )

    concatenate(
        settings=concat_settings,
        output_dirpath=Path(output_dirpath),
        sbatch_filepath=sbatch_filepath,
        local=local,
        block=False,
        monitor=monitor,
    )

    click.echo("Conversion complete!")


if __name__ == "__main__":
    convert_cli()
