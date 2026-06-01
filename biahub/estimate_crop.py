import glob as globmod
import shutil

from ast import literal_eval
from pathlib import Path

import click
import dask.array as da
import numpy as np
import pandas as pd
import submitit

from iohub import open_ome_zarr
from natsort import natsorted

from biahub.cli.parsing import (
    init_only,
    local,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import (
    estimate_resources,
    get_submitit_cluster,
    model_to_yaml,
    yaml_to_model,
)
from biahub.register import find_lir
from biahub.settings import ConcatenateSettings


def estimate_crop_one_position(
    lf_dir: np.ndarray,
    ls_dir: np.ndarray,
    lf_mask_radius: float = None,
    output_dir: Path = None,
):
    """
    Estimate a crop region where both phase and fluorescence volumes are non-zero.

    Parameters
    ----------
    lf_dir : Path
        Path to the phase channel.
    ls_dir : Path
        Path to the fluorescence channel.
    lf_mask_radius : float
        Radius of the circular mask which will be applied to the phase channel. If None, no masking will be applied
    output_dir : Path
        Path to save the output CSV file.

    Returns
    -------
    tuple
        Tuple of slices for Z, Y, and X dimensions.

    """
    fov = "/".join(lf_dir.parts[-3:])

    click.echo(f"Processing FOV: {fov}")
    with open_ome_zarr(lf_dir) as lf_dataset:
        lf_data = lf_dataset.data.dask_array()[:, :1]  # Pick only first channel
        lf_mask = ((lf_data != 0) & (~da.isnan(lf_data))).compute()

    with open_ome_zarr(ls_dir) as ls_dataset:
        ls_data = ls_dataset.data.dask_array()[:, :1]
        ls_mask = ((ls_data != 0) & (~da.isnan(ls_data))).compute()

    if lf_mask.ndim != 5 or ls_mask.ndim != 5:
        raise ValueError("Both phase_data and fluor_data must be 5D arrays.")

    # Ensure data dimensions are the same
    lf_shape = lf_mask.shape[-3:]
    ls_shape = ls_mask.shape[-3:]
    _max_zyx_dims = np.asarray([lf_shape, ls_shape]).min(axis=0)
    if lf_shape != ls_shape:
        click.echo(
            "WARNING: Phase and fluorescence datasets should have the same shape, got"
            f" phase shape: {lf_shape}, fluorescence shape: {ls_shape}"
        )
        lf_mask = lf_mask[..., : _max_zyx_dims[0], : _max_zyx_dims[1], : _max_zyx_dims[2]]
        ls_mask = ls_mask[..., : _max_zyx_dims[0], : _max_zyx_dims[1], : _max_zyx_dims[2]]

    # Concatenate the data arrays along the channel axis
    data = np.concatenate([lf_mask, ls_mask], axis=1)

    # Create a mask to find time points and channels where any data is non-zero
    volume = np.sum(data, axis=(2, 3, 4))  # more robust selection
    median_volume = np.median(volume)
    valid_T, valid_C = np.where(
        (volume > 0.8 * median_volume) & (volume < 1.2 * median_volume)
    )

    if len(valid_T) == 0:
        click.echo("No valid data found for current position, will not crop.")
        return tuple(zip((0, 0, 0), _max_zyx_dims, strict=True))
    valid_data = data[valid_T, valid_C]

    # Compute a mask where all voxels are non-zero along time and channel dimensions
    combined_mask = np.all(valid_data, axis=0)

    # Create a circular boolean mask of radius phase_mask_radius to apply to the phase channel
    if lf_mask_radius is not None:
        click.echo(f"Applying circular mask of radius {lf_mask_radius} to phase channel.")
        if not (0 < lf_mask_radius <= 1):
            raise ValueError(
                "lf_mask_radius must be a fraction of image width (0 < lf_mask_radius <= 1)."
            )

        lf_mask = np.zeros(lf_mask.shape[-2:], dtype=bool)

        y, x = np.ogrid[: lf_mask.shape[-2], : lf_mask.shape[-1]]
        center = (lf_mask.shape[-2] // 2, lf_mask.shape[-1] // 2)
        radius = int(lf_mask_radius * min(center))

        lf_mask[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2] = True
        lf_mask_cropped = lf_mask[: _max_zyx_dims[1], : _max_zyx_dims[2]]
        combined_mask = combined_mask * lf_mask_cropped

    # Compute overlapping region
    z_slice, y_slice, x_slice = find_lir(combined_mask)

    click.echo(
        f"Estimated crop for FOV {fov}:\n"
        f"Z: {z_slice.start} - {z_slice.stop}\n"
        f"Y: {y_slice.start} - {y_slice.stop}\n"
        f"X: {x_slice.start} - {x_slice.stop}"
    )

    if output_dir:
        df = pd.DataFrame(
            [
                {
                    "fov": fov,
                    "Z": [z_slice.start, z_slice.stop],
                    "Y": [y_slice.start, y_slice.stop],
                    "X": [x_slice.start, x_slice.stop],
                }
            ]
        )
        df.to_csv(output_dir / f"{fov.replace('/', '_')}.csv", index=False)

    return (
        [z_slice.start, z_slice.stop],
        [y_slice.start, y_slice.stop],
        [x_slice.start, x_slice.stop],
    )


def _init_estimate_crop(lf_data_path: str, ls_data_path: str):
    """Emit RESOURCES and paired position lines for per-FOV estimate-crop fan-out."""
    lf_positions = natsorted(
        [Path(p) for p in globmod.glob(lf_data_path) if Path(p).is_dir()]
    )
    ls_positions = natsorted(
        [Path(p) for p in globmod.glob(ls_data_path) if Path(p).is_dir()]
    )
    if not lf_positions:
        raise click.ClickException(f"No positions found matching '{lf_data_path}'")
    if not ls_positions:
        raise click.ClickException(f"No positions found matching '{ls_data_path}'")
    if len(lf_positions) != len(ls_positions):
        raise click.ClickException(
            f"Mismatched position counts: {len(lf_positions)} label-free "
            f"vs {len(ls_positions)} light-sheet."
        )

    with open_ome_zarr(lf_positions[0]) as ds:
        T, C, Z, Y, X = ds.data.shape
        dtype = ds.data.dtype

    volume_gb = T * Z * Y * X * np.dtype(dtype).itemsize / 2**30
    total_gb = max(8, int(np.ceil(volume_gb * 4)))
    click.echo(f"RESOURCES:1 {total_gb}")

    for lf_pos, ls_pos in zip(lf_positions, ls_positions, strict=True):
        click.echo(f"POSITION:{lf_pos}\t{ls_pos}")


def _reduce_crop_ranges(
    config_path: Path,
    output_config: Path,
    ranges_file: Path,
    concat_data_paths: tuple[str, ...] | None,
):
    """Reduce per-FOV crop ranges into a global standardized crop config."""
    settings = yaml_to_model(config_path, ConcatenateSettings)

    all_ranges = []
    with open(ranges_file) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("RANGES:"):
                continue
            parts = line.replace("RANGES:", "").strip().split()
            z_start, z_end = (int(v) for v in parts[0].split(","))
            y_start, y_end = (int(v) for v in parts[1].split(","))
            x_start, x_end = (int(v) for v in parts[2].split(","))
            all_ranges.append([[z_start, z_end], [y_start, y_end], [x_start, x_end]])

    if not all_ranges:
        raise click.ClickException("No RANGES: lines found in ranges file.")

    all_ranges = np.array(all_ranges)
    standardized_ranges = np.concatenate(
        [
            all_ranges[..., 0].max(axis=0, keepdims=True),
            all_ranges[..., 1].min(axis=0, keepdims=True),
        ]
    )

    click.echo(
        f"Standardized ranges:\nZ: {standardized_ranges[:, 0].tolist()}\n"
        f"Y: {standardized_ranges[:, 1].tolist()}\n"
        f"X: {standardized_ranges[:, 2].tolist()}"
    )

    output_model = settings.model_copy()
    if concat_data_paths:
        output_model.concat_data_paths = list(concat_data_paths)
    output_model.Z_slice = standardized_ranges[:, 0].tolist()
    output_model.Y_slice = standardized_ranges[:, 1].tolist()
    output_model.X_slice = standardized_ranges[:, 2].tolist()
    model_to_yaml(output_model, output_config)

    click.echo(f"Updated config written to {output_config}")


def estimate_crop(
    config_filepath: str,
    output_filepath: str,
    lf_mask_radius: float = 0.95,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate a crop region where both phase and fluorescene volumes are non-zero.

    Parameters
    ----------
    config_filepath : str
        Path to a yaml ConcatenateSettings file.
        This file will be replicated in the output with modified XYZ slicing parametrs.
    output_filepath : str
        Path to save the output config file.
    lf_mask_radius : float
        Radius of the circular mask given as fraction of image width to apply to the phase channel.
        A good value if 0.95.
    sbatch_filepath : str
        Path to a SLURM submission script.
    local : bool
        If True, run the jobs locally.

    """
    if config_filepath.suffix not in [".yml", ".yaml"]:
        raise ValueError("Config file must be a yaml file")

    config_filepath = Path(config_filepath)
    settings = yaml_to_model(config_filepath, ConcatenateSettings)
    output_dir = Path(output_filepath).parent
    slurm_out_path = output_dir / "slurm_output"
    output_path_csv = output_dir / "crop_estimates"
    output_path_csv.mkdir(exist_ok=True, parents=True)

    # Assume phase dataset is first and fluor dataset is second in input_model.concat_data_paths
    lf_paths = config_filepath.parent.glob(settings.concat_data_paths[0])
    lf_position_dirpaths = [p for p in lf_paths if p.is_dir()]
    click.echo(f"Found {len(lf_position_dirpaths)} phase channels.")
    ls_paths = config_filepath.parent.glob(settings.concat_data_paths[1])
    ls_position_dirpaths = [p for p in ls_paths if p.is_dir()]
    click.echo(f"Found {len(ls_position_dirpaths)} fluorescence channels.")

    if len(lf_position_dirpaths) != len(ls_position_dirpaths):
        raise ValueError("Number of phase and fluorescence channels must be the same.")

    # Estimate resources from a sample
    with open_ome_zarr(lf_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
    )
    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_crop",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 30,
        "slurm_partition": "preempted",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    cluster = get_submitit_cluster(local)

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting SLURM jobs...")
    jobs = []

    with submitit.helpers.clean_env(), executor.batch():
        for ls_dir, lf_dir in zip(ls_position_dirpaths, lf_position_dirpaths, strict=True):
            job = executor.submit(
                estimate_crop_one_position,
                lf_mask_radius=lf_mask_radius,
                lf_dir=lf_dir,
                ls_dir=ls_dir,
                output_dir=output_path_csv,
            )
            jobs.append(job)

    # Wait for jobs to finish
    wait_for_jobs_to_finish(jobs)

    # Read and merge results. The same ROI crop will be applied to all positions.
    # Here we estimate the smallest common crop region across all positions.
    estimate_crop_csvs = list(output_path_csv.glob("*.csv"))
    if not estimate_crop_csvs:
        click.echo("No crop CSV files found. Exiting.")
        return

    df = pd.concat(
        [pd.read_csv(f, dtype={"fov": str}) for f in estimate_crop_csvs], ignore_index=True
    )
    df = df.drop_duplicates(subset=["fov", "Z", "Y", "X"])
    df = df.sort_values("fov")
    for col in ["X", "Y", "Z"]:
        df[col] = df[col].apply(literal_eval)
    df.to_csv(output_dir / "crop_slices.csv", index=False)

    # Compute standardized crop
    all_ranges = []
    for _, row in df.iterrows():
        all_ranges.append([row["Z"], row["Y"], row["X"]])

    all_ranges = np.array(all_ranges)
    standardized_ranges = np.concatenate(
        [
            all_ranges[..., 0].max(axis=0, keepdims=True),
            all_ranges[..., 1].min(axis=0, keepdims=True),
        ]
    )

    click.echo(
        f"Standardized ranges:\nZ: {standardized_ranges[:, 0].tolist()}\n"
        f"Y: {standardized_ranges[:, 1].tolist()}\n"
        f"X: {standardized_ranges[:, 2].tolist()}"
    )

    # Save updated YAML config
    output_model = settings.model_copy()
    output_model.Z_slice = standardized_ranges[:, 0].tolist()
    output_model.Y_slice = standardized_ranges[:, 1].tolist()
    output_model.X_slice = standardized_ranges[:, 2].tolist()
    model_to_yaml(output_model, output_filepath)

    shutil.rmtree(output_path_csv)
    click.echo("Done.")


@click.command("estimate-crop")
@click.option(
    "--config-filepath",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    help="Path to YAML configuration file.",
)
@click.option(
    "--output-filepath",
    "-o",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default=None,
    help="Path to output file.",
)
@sbatch_filepath()
@local()
@init_only()
@click.option(
    "--lf-mask-radius",
    type=float,
    default=0.95,
    help="Radius of the circular mask given as fraction of image width to apply to the phase channel.",
)
@click.option(
    "--lf-data-path",
    type=str,
    default=None,
    help="Glob pattern for label-free source zarr (--init mode).",
)
@click.option(
    "--ls-data-path",
    type=str,
    default=None,
    help="Glob pattern for light-sheet source zarr (--init mode).",
)
@click.option(
    "--lf-position",
    type=click.Path(exists=True),
    default=None,
    help="Path to a single label-free position zarr (per-FOV mode).",
)
@click.option(
    "--ls-position",
    type=click.Path(exists=True),
    default=None,
    help="Path to a single light-sheet position zarr (per-FOV mode).",
)
@click.option(
    "--reduce",
    "reduce_mode",
    is_flag=True,
    default=False,
    help="Reduce per-FOV crop ranges into a global standardized crop config.",
)
@click.option(
    "--ranges-file",
    type=click.Path(exists=True),
    default=None,
    help="File with one RANGES: line per FOV (--reduce mode).",
)
@click.option(
    "--concat-data-paths",
    multiple=True,
    type=str,
    help="Override concat_data_paths from config (--reduce mode, repeat flag).",
)
def estimate_crop_cli(
    config_filepath: str | None,
    output_filepath: str | None,
    sbatch_filepath: str | None,
    local: bool,
    init_only: bool,
    lf_mask_radius: float,
    lf_data_path: str | None,
    ls_data_path: str | None,
    lf_position: str | None,
    ls_position: str | None,
    reduce_mode: bool,
    ranges_file: str | None,
    concat_data_paths: tuple[str, ...],
):
    r"""Estimate a crop region where both phase and fluorescence volumes are non-zero.

    \b
    Full end-to-end (SLURM fan-out + reduce):
    >>> biahub estimate-crop -c ./concat.yml -o ./cropped_concat.yml --local

    \b
    Nextflow init (list positions, emit RESOURCES):
    >>> biahub estimate-crop --init \
        --lf-data-path "deskew.zarr/*/*/*" \
        --ls-data-path "reconstruct.zarr/*/*/*"

    \b
    Nextflow per-FOV (estimate one position, emit RANGES):
    >>> biahub estimate-crop \
        --lf-position deskew.zarr/B/3/000000 \
        --ls-position reconstruct.zarr/B/3/000000

    \b
    Nextflow reduce (aggregate ranges into config):
    >>> biahub estimate-crop --reduce \
        -c concat.yml -o cropped.yml --ranges-file ranges.txt
    """
    if init_only:
        if not lf_data_path or not ls_data_path:
            raise click.UsageError("--init requires --lf-data-path and --ls-data-path")
        _init_estimate_crop(lf_data_path, ls_data_path)
    elif lf_position and ls_position:
        z_range, y_range, x_range = estimate_crop_one_position(
            lf_dir=Path(lf_position),
            ls_dir=Path(ls_position),
            lf_mask_radius=lf_mask_radius,
        )
        click.echo(
            f"RANGES:{z_range[0]},{z_range[1]} "
            f"{y_range[0]},{y_range[1]} "
            f"{x_range[0]},{x_range[1]}"
        )
    elif reduce_mode:
        if not config_filepath or not output_filepath or not ranges_file:
            raise click.UsageError("--reduce requires -c, -o, and --ranges-file")
        _reduce_crop_ranges(
            config_path=Path(config_filepath),
            output_config=Path(output_filepath),
            ranges_file=Path(ranges_file),
            concat_data_paths=concat_data_paths or None,
        )
    else:
        if not config_filepath or not output_filepath:
            raise click.UsageError("Default mode requires -c and -o")
        estimate_crop(
            config_filepath=Path(config_filepath),
            output_filepath=Path(output_filepath),
            lf_mask_radius=lf_mask_radius,
            sbatch_filepath=sbatch_filepath,
            local=local,
        )


if __name__ == "__main__":
    estimate_crop_cli()
