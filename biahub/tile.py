import os

from pathlib import Path
from typing import Dict, List, Tuple, Union

import click
import numpy as np
import submitit
import yaml

from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta
from iohub.ngff.nodes import Plate
from patchly import GridSampler

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
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.settings import StitchSettings, TileSettings


def resolve_sizes(size_list: List[Union[int, str]], data_shape: Tuple[int, ...]) -> List[int]:
    """Convert wildcard (*) values to actual dimension sizes from data shape."""
    resolved = []
    shape_zyx = data_shape[-3:]  # Get ZYX from TCZYX

    for i, size in enumerate(size_list):
        if size == "*":
            resolved.append(shape_zyx[i])
        else:
            resolved.append(int(size))

    return resolved


def resolve_chunks(
    chunk_list: List[Union[int, str]], data_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Convert wildcard (*) values to actual dimension sizes for rechunking."""
    resolved = []

    for i, size in enumerate(chunk_list):
        if size == "*":
            resolved.append(data_shape[i])
        else:
            resolved.append(int(size))

    return tuple(resolved)


def count_inodes(path: Path) -> int:
    """Count total inodes (files + directories) in a directory tree.

    Args:
        path: Path to directory to count inodes in

    Returns:
        Total count of inodes (directories + files)
    """
    count = 0
    for _, _, files in os.walk(path):
        count += 1  # directory itself
        count += len(files)  # all files in directory
    return count


def process_position_to_patches(
    input_position_path: Path,
    output_plate: Plate,
    settings: TileSettings,
    verbose: bool = False,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, List[float]]]:
    """Process a single position and return the list of position keys and translations.

    Each FOV is processed independently with its own GridSampler to handle variable FOV sizes.
    Output positions are created dynamically based on the actual patch count for this FOV.

    Returns:
        Tuple of (position_keys, translations) where translations maps position names to [z, y, x] offsets.
    """
    # Extract row, col, fov from the input path
    row, col, fov_idx = input_position_path.parts[-3:]

    if verbose:
        click.echo(f"Processing position {row}/{col}/{fov_idx}")

    position_keys = []
    translations = {}

    with open_ome_zarr(input_position_path) as input_position:
        data = input_position.data.dask_array()

        if settings.channels:
            channel_names = settings.channels
            channel_indices = [input_position.channel_names.index(ch) for ch in channel_names]
            data = data[:, channel_indices, :, :, :]
        else:
            channel_names = input_position.channel_names

        num_channels = len(channel_names)

        if settings.rechunk is not None:
            resolved_chunks = resolve_chunks(settings.rechunk, data.shape)
            data = data.rechunk(chunks=resolved_chunks)
            chunk_size = list(resolved_chunks[-3:])
            if verbose:
                click.echo(f"Rechunked to: {data.chunks}")
        else:
            # Default chunk_size when no rechunking
            chunk_size = ["*", 1024, 1024]

        if verbose:
            click.echo(f"Input data shape: {data.shape}")
            click.echo(f"Input data chunks: {data.chunks}")

        # Resolve wildcards for patch/step/chunk sizes
        patch_size = resolve_sizes(settings.patch_size, data.shape)
        step_size = resolve_sizes(settings.step_size, data.shape)
        chunk_size = resolve_sizes(chunk_size, data.shape)

        if verbose:
            click.echo(f"Patch size (ZYX): {patch_size}")
            click.echo(f"Step size (ZYX): {step_size}")
            click.echo(f"Chunk size (ZYX): {chunk_size}")

        # Create GridSampler for THIS specific FOV
        gs = GridSampler(
            image=data,
            spatial_size=data.shape[-3:],
            patch_size=patch_size,
            step_size=step_size,
            chunk_size=chunk_size,
            spatial_first=False,
        )

        true_patch_size = gs.patch_size_s
        patch_shape = (data.shape[0], num_channels, *true_patch_size)

        # Calculate padding width based on total number of patches
        total_patches = len(gs)
        padding_width = len(str(total_patches - 1)) if total_patches > 0 else 1

        if settings.output_chunks:
            output_chunks = tuple(settings.output_chunks)
        else:
            output_chunks = (1, 1, *true_patch_size)

        if verbose:
            click.echo(f"  Creating {len(gs)} patches for this FOV")

        for patch_idx, (patch, patch_bbox) in enumerate(gs):
            # Generate position key
            position_key = (row, col, f"{fov_idx}P{patch_idx:0{padding_width}d}")
            position_keys.append(position_key)

            position_name = "/".join(position_key)
            translation_zyx = [float(patch_bbox[i, 0]) for i in range(3)]  # [z, y, x]
            translations[position_name] = translation_zyx

            output_position = output_plate.create_position(
                row, col, f"{fov_idx}P{patch_idx:0{padding_width}d}"
            )
            _ = output_position.create_zeros(
                "0",
                shape=patch_shape,
                chunks=output_chunks,
                dtype=np.float32,
                transform=[
                    TransformationMeta(type="scale", scale=(1, 1, 1, 1, 1)),
                    TransformationMeta(
                        type="translation", translation=[0, 0, *translation_zyx]
                    ),
                ],
            )

            output_position["0"][:] = patch.compute()

            if verbose and (patch_idx + 1) % 10 == 0:
                click.echo(f"  Processed {patch_idx + 1} patches")

        if verbose:
            click.echo(
                f"Completed processing {len(position_keys)} patches from {row}/{col}/{fov_idx}"
            )

    return position_keys, translations


@click.command("tile")
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    type=bool,
    help="Verbose output. Default is False.",
)
@monitor()
def tile_cli(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    config_filepath: Path,
    verbose: bool = False,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = False,
) -> None:
    """
    Create tiles/patches from positions in a zarr store.
    Uses GridSampler to create consistent-sized patches with squeeze sampling.

    Each FOV is processed independently with its own GridSampler, allowing for
    variable FOV sizes within the same plate. Patches are created dynamically
    based on each FOV's actual dimensions.

    Examples:

    Process a single FOV locally:
    >> biahub tile -i ./input.zarr/0/0/0 -o ./output.zarr -c ./tile_config.yaml -l

    Process multiple FOVs and submit to SLURM:
    >> biahub tile -i ./input.zarr/*/*/* -o ./output.zarr -c ./tile_config.yaml

    Submit to SLURM with a custom sbatch file:
    >> biahub tile -i ./input.zarr/*/*/* -o ./output.zarr -c ./tile_config.yaml --sbatch slurm.sh
    """

    click.echo("Starting tiling...")
    settings = yaml_to_model(config_filepath, TileSettings)

    # Count input inodes
    click.echo("Counting input inodes...")
    input_inodes = sum(count_inodes(path) for path in input_position_dirpaths)
    click.echo(f"Input: {input_inodes:,} inodes")

    # Initialize storage for stitch config
    all_translations = {}
    channel_names = None

    # Get channel names from first position for plate creation
    with open_ome_zarr(input_position_dirpaths[0]) as first_position:
        if settings.channels:
            channel_names = settings.channels
        else:
            channel_names = first_position.channel_names

    output_plate = open_ome_zarr(
        output_dirpath, layout='hcs', mode="w", channel_names=channel_names
    )

    if not local:
        estimated_patch_shape = (1, len(channel_names), 51, 512, 512)
        num_cpus, gb_ram = estimate_resources(
            shape=estimated_patch_shape, ram_multiplier=1.2, max_num_cpus=8
        )

        # Prepare SLURM arguments
        slurm_args = {
            "slurm_job_name": "tile",
            "slurm_mem_per_cpu": f"{gb_ram}G",
            "slurm_cpus_per_task": num_cpus,
            "slurm_array_parallelism": 50,  # process up to 50 positions at a time
            "slurm_time": 60,
            "slurm_partition": "cpu",
        }

        if sbatch_filepath:
            slurm_args.update(sbatch_to_submitit(sbatch_filepath))

        click.echo(f"Preparing jobs: {slurm_args}")
        slurm_out_path = Path(output_dirpath).parent / "slurm_output"
        executor = submitit.AutoExecutor(folder=slurm_out_path, cluster="slurm")
        executor.update_parameters(**slurm_args)

        jobs = []
        with executor.batch():
            for input_path in input_position_dirpaths:
                jobs.append(
                    executor.submit(
                        process_position_to_patches,
                        input_path,
                        output_plate,
                        settings,
                        verbose,
                    )
                )

        job_ids = [job.job_id for job in jobs]

        log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
        with log_path.open("w") as log_file:
            log_file.write("\n".join(job_ids))

        if monitor:
            monitor_jobs(jobs, [])

        click.echo("Waiting for all jobs to complete...")
        for i, (job, input_path) in enumerate(zip(jobs, input_position_dirpaths)):
            try:
                position_keys, translations = job.result()
                all_translations.update(translations)
                if verbose:
                    click.echo(f"Collected results from job {i + 1}/{len(jobs)}")
            except Exception as e:
                click.echo(f"Error processing {input_path}: {e}", err=True)
    else:
        # Process locally
        for i, input_path in enumerate(input_position_dirpaths):
            click.echo(
                f"Processing position {i + 1}/{len(input_position_dirpaths)}: {input_path}"
            )
            position_keys, translations = process_position_to_patches(
                input_path, output_plate, settings, verbose
            )
            all_translations.update(translations)

    # Save stitch configuration to parent directory of output zarr store
    stitch_config_path = Path(output_dirpath).parent / "tile_stitch_config.yaml"
    stitch_settings = StitchSettings(
        channels=channel_names, total_translation=all_translations
    )

    with open(stitch_config_path, "w") as f:
        yaml.dump(
            stitch_settings.model_dump(exclude_none=True),
            f,
            default_flow_style=False,
            sort_keys=False,
            width=1000,  # Avoid line wrapping
        )

    click.echo(f"Saved stitch configuration to: {stitch_config_path}")

    # Count output inodes and display comparison
    click.echo("\nCounting output inodes...")
    output_inodes = count_inodes(output_dirpath)
    increase = output_inodes - input_inodes
    multiplier = output_inodes / input_inodes if input_inodes > 0 else 0

    click.echo("\nInode usage:")
    click.echo(f"  Input:    {input_inodes:,} inodes")
    click.echo(f"  Output:   {output_inodes:,} inodes")
    click.echo(f"  Increase: {multiplier:.1f}x ({increase:,} additional inodes)")

    click.echo("\nTiling complete!")
    click.echo(
        f"Remember to delete the tiled Zarr Store at {output_dirpath} after you are done with it"
    )


if __name__ == "__main__":
    tile_cli()
