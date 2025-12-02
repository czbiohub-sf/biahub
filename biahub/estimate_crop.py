import shutil

from ast import literal_eval
from pathlib import Path

import click
import dask.array as da
import numpy as np
import pandas as pd
import submitit

from iohub import open_ome_zarr

from biahub.cli.parsing import (
    config_filepath,
    local,
    output_filepath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import estimate_resources, model_to_yaml, yaml_to_model
from biahub.registration.utils import find_lir
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
    if lf_shape != ls_shape:
        click.echo(
            "WARNING: Phase and fluorescence datasets should have the same shape, got"
            f" phase shape: {lf_shape}, fluorescence shape: {ls_shape}"
        )
        _max_zyx_dims = np.asarray([lf_shape, ls_shape]).min(axis=0)
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
        return tuple(zip((0, 0, 0), _max_zyx_dims))
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


def estimate_crop(
    config_filepath: str,
    output_filepath: str,
    lf_mask_radius: float = 0.95,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate a crop region where both phase and fluorescene volumes are non-zero.

    Parameters:
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
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo('Submitting SLURM jobs...')
    jobs = []

    with executor.batch():
        for ls_dir, lf_dir in zip(ls_position_dirpaths, lf_position_dirpaths):

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
@config_filepath()
@output_filepath()
@sbatch_filepath()
@local()
@click.option(
    "--lf-mask-radius",
    type=float,
    help="(Optional) Radius of the circular mask given as fraction of image width to apply to the phase channel.",
    required=False,
)
def estimate_crop_cli(
    config_filepath: str,
    output_filepath: str,
    lf_mask_radius: float = 0.95,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate a crop region where both phase and fluorescene volumes are non-zero.

    Parameters:
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
    estimate_crop(
        config_filepath=config_filepath,
        output_filepath=output_filepath,
        lf_mask_radius=lf_mask_radius,
        sbatch_filepath=sbatch_filepath,
        local=local,
    )


if __name__ == "__main__":
    estimate_crop_cli()
