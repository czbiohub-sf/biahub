from pathlib import Path

import click
import numpy as np
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import process_single_position

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    local,
    monitor,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.settings import ComputeStatsSettings


def extract_patch_around_centroid(
    data: np.ndarray, centroid: tuple, patch_size: list, z_range: list = None
) -> np.ndarray:
    """
    Extract a 3D patch around a centroid from CZYX data.

    Parameters
    ----------
    data : np.ndarray
        4D array with shape (C, Z, Y, X)
    centroid : tuple
        (z, y, x) coordinates of centroid (no time dimension)
    patch_size : list
        [z_size, y_size, x_size] for patch dimensions
    z_range : list, optional
        [z_min, z_max] to constrain Z dimension

    Returns
    -------
    np.ndarray
        4D patch with shape (C, Z, Y, X)
    """
    z, y, x = int(centroid[0]), int(centroid[1]), int(centroid[2])
    z_size, y_size, x_size = patch_size

    z_half, y_half, x_half = z_size // 2, y_size // 2, x_size // 2

    z_start = max(0, z - z_half)
    z_end = min(data.shape[1], z + z_half + 1)
    y_start = max(0, y - y_half)
    y_end = min(data.shape[2], y + y_half + 1)
    x_start = max(0, x - x_half)
    x_end = min(data.shape[3], x + x_half + 1)

    if z_range is not None:
        z_start = max(z_start, z_range[0])
        z_end = min(z_end, z_range[1])

    return data[:, z_start:z_end, y_start:y_end, x_start:x_end]


def compute_patch_stats(
    patch: np.ndarray,
    channel_indices: list,
    percentile_lower: float = 50.0,
    percentile_upper: float = 99.0,
) -> dict:
    """
    Compute 3D volume statistics for a patch across specified channels.

    Parameters
    ----------
    patch : np.ndarray
        4D patch with shape (C, Z, Y, X)
    channel_indices : list
        List of channel indices to process
    percentile_lower : float
        Lower percentile bound (default 50th percentile)
    percentile_upper : float
        Upper percentile bound (default 99th percentile)

    Returns
    -------
    dict
        Dictionary containing computed volume statistics
    """
    stats = {}

    for c_idx in channel_indices:
        if c_idx >= patch.shape[0]:
            continue

        channel_data = patch[c_idx]
        flattened_data = channel_data.flatten()

        stats[f'channel_{c_idx}_mean'] = np.mean(flattened_data)
        stats[f'channel_{c_idx}_std'] = np.std(flattened_data)
        stats[f'channel_{c_idx}_median'] = np.median(flattened_data)
        stats[f'channel_{c_idx}_max'] = np.max(flattened_data)
        stats[f'channel_{c_idx}_min'] = np.min(flattened_data)
        stats[f'channel_{c_idx}_sum'] = np.sum(flattened_data)

        q25, q75 = np.percentile(flattened_data, [25, 75])
        stats[f'channel_{c_idx}_iqr'] = q75 - q25
        stats[f'channel_{c_idx}_q25'] = q25
        stats[f'channel_{c_idx}_q75'] = q75

        stats[f'channel_{c_idx}_p{int(percentile_lower)}'] = np.percentile(
            flattened_data, percentile_lower
        )
        stats[f'channel_{c_idx}_p{int(percentile_upper)}'] = np.percentile(
            flattened_data, percentile_upper
        )

    return stats


def get_stats(
    position_path: Path,
    channel_names: list,
    patch_size: list,
    z_range: list = None,
    compute_best_focus: bool = False,
    percentile_lower_bound: float = 50.0,
    percentile_upper_bound: float = 99.0,
) -> pd.DataFrame:
    """
    Extract 3D volume statistics from patches around centroids for a single position.

    Parameters
    ----------
    position_path : Path
        Path to position directory containing track CSV and data
    channel_names : list
        List of channel names to process
    patch_size : list
        [z_size, y_size, x_size] for patch dimensions
    z_range : list, optional
        [z_min, z_max] to constrain Z dimension
    compute_best_focus : bool
        Whether to compute best focus statistics
    percentile_lower_bound : float
        Lower percentile bound for statistics
    percentile_upper_bound : float
        Upper percentile bound for statistics

    Returns
    -------
    pd.DataFrame
        DataFrame with original track data plus computed volume statistics
    """
    position_name = position_path.name

    # Find the track CSV file
    track_csv_files = list(position_path.glob(f"tracks_{position_name}.csv"))
    if not track_csv_files:
        track_csv_files = list(position_path.glob("tracks_*.csv"))

    if not track_csv_files:
        raise FileNotFoundError(f"No track CSV found in {position_path}")

    track_csv_path = track_csv_files[0]
    tracks_df = pd.read_csv(track_csv_path)

    with open_ome_zarr(position_path, mode="r") as dataset:
        data = dataset.data
        available_channels = dataset.channel_names

        channel_indices = []
        for ch_name in channel_names:
            if ch_name in available_channels:
                channel_indices.append(available_channels.index(ch_name))

        all_stats = []
        for _, row in tracks_df.iterrows():
            centroid = (row.get('t', 0), row.get('z', 0), row.get('y', 0), row.get('x', 0))

            patch = extract_patch_around_centroid(data, centroid, patch_size, z_range)

            patch_stats = compute_patch_stats(
                patch, channel_indices, percentile_lower_bound, percentile_upper_bound
            )

            combined_stats = dict(row)
            combined_stats.update(patch_stats)
            all_stats.append(combined_stats)

    return pd.DataFrame(all_stats)


def compute_stats_czyx(
    czyx_data: np.ndarray,
    track_csv_path: Path,
    current_timepoint: int,
    patch_size: list,
    z_range: list = None,
    percentile_lower_bound: float = 50.0,
    percentile_upper_bound: float = 99.0,
) -> np.ndarray:
    """
    Compute volume statistics from CZYX data for tracked centroids.
    This function is designed to work with process_single_position.

    Parameters
    ----------
    czyx_data : np.ndarray
        4D array with shape (C, Z, Y, X)
    track_csv_path : Path
        Path to the tracking CSV file
    current_timepoint : int
        Current timepoint being processed
    patch_size : list
        [z_size, y_size, x_size] for patch dimensions
    z_range : list, optional
        [z_min, z_max] to constrain Z dimension
    percentile_lower_bound : float
        Lower percentile bound for statistics
    percentile_upper_bound : float
        Upper percentile bound for statistics

    Returns
    -------
    np.ndarray
        Output array (same shape as input for compatibility)
    """
    tracks_df = pd.read_csv(track_csv_path)
    current_tracks = tracks_df[tracks_df['t'] == current_timepoint]

    timepoint_stats = []

    for _, row in current_tracks.iterrows():
        centroid = (row.get('z', 0), row.get('y', 0), row.get('x', 0))

        patch = extract_patch_around_centroid(czyx_data, centroid, patch_size, z_range)

        channel_indices = list(range(czyx_data.shape[0]))
        patch_stats = compute_patch_stats(
            patch, channel_indices, percentile_lower_bound, percentile_upper_bound
        )

        combined_stats = dict(row)
        combined_stats.update(patch_stats)
        timepoint_stats.append(combined_stats)

    if hasattr(czyx_data, '_stats_data'):
        czyx_data._stats_data.extend(timepoint_stats)
    else:
        czyx_data._stats_data = timepoint_stats

    return czyx_data


def compute_stats(
    input_position_path: str,
    output_position_path: str,
    settings: ComputeStatsSettings,
) -> None:
    """
    Process a single position to compute statistics using process_single_position.

    Parameters
    ----------
    input_position_path : str
        Path to the input position directory
    output_position_path : str
        Path to the output position directory (same as input for stats)
    settings : ComputeStatsSettings
        Configuration settings for stats computation
    """
    input_path = Path(input_position_path)
    output_path = Path(output_position_path)

    position_name = input_path.name
    track_csv_files = list(input_path.glob(f"tracks_{position_name}.csv"))
    if not track_csv_files:
        track_csv_files = list(input_path.glob("tracks_*.csv"))

    if not track_csv_files:
        raise FileNotFoundError(f"No track CSV found in {input_path}")

    track_csv_path = track_csv_files[0]

    with open_ome_zarr(input_path, mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape
        available_channels = dataset.channel_names

        # Map channel names to indices
        channel_indices = []
        for ch_name in settings.channel_names:
            if ch_name in available_channels:
                channel_indices.append(available_channels.index(ch_name))

    # TODO: check we parallelize over T and C. Pseudocoded this.
    result = process_single_position(
        compute_stats_czyx,
        input_position_path,
        output_position_path,
        input_channel_indices=[channel_indices],
        output_channel_indices=[channel_indices],
        num_processes=4,
        track_csv_path=track_csv_path,
        current_timepoint=0,  # FIXME: This will be handled by process_single_position
        patch_size=settings.patch_size,
        z_range=settings.z_range,
        percentile_lower_bound=settings.percentile_lower_bound,
        percentile_upper_bound=settings.percentile_upper_bound,
    )

    click.echo(f"Processed stats for {position_name}")
    # TODO: The actual CSV saving would need to be handled differently
    # since process_single_position processes per timepoint/channel. Probably a scatter gather approach
    result.to_csv(output_path / "stats.csv", index=False)


@click.command("compute-stats")
@config_filepath()
@sbatch_filepath()
@local()
@monitor()
def compute_stats_cli(
    config_filepath: str,
    sbatch_filepath: str | None = None,
    local: bool = False,
    monitor: bool = True,
):
    """
    Compute 3D volume statistics from patches around tracked centroids.
    Adds the stats into a separate CSV at each position-level directory.

    >> biahub compute-stats \
        -c ./compute_stats_params.yml \
    """
    if sbatch_filepath is not None:
        sbatch_filepath = Path(sbatch_filepath)

    settings = yaml_to_model(config_filepath, ComputeStatsSettings)

    # Get position directories from track_zarr_path
    track_zarr_path = Path(settings.track_zarr_path)
    position_dirs = list(track_zarr_path.glob(settings.input_position_dirpaths))

    if not position_dirs:
        raise ValueError(
            f"No positions found at {track_zarr_path}/{settings.input_position_dirpaths}"
        )

    # Get data shape for resource estimation
    with open_ome_zarr(position_dirs[0], mode="r") as track_dataset:
        T, C, Z, Y, X = track_dataset.data.shape
        available_channels = track_dataset.channel_names

        # Map channel names to indices
        channel_indices = []
        for ch_name in settings.channel_names:
            if ch_name in available_channels:
                channel_indices.append(available_channels.index(ch_name))

    # Estimate resources for process_single_position parallelization
    num_cpus, gb_ram_request = estimate_resources(
        shape=[T, len(channel_indices), Z, Y, X], ram_multiplier=12
    )
    slurm_time = np.ceil(np.max([60, T * len(channel_indices) * 0.5])).astype(int)
    slurm_array_parallelism = 20

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "compute-stats",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": slurm_array_parallelism,
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
    slurm_out_path = track_zarr_path.parent / "slurm_output"
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with executor.batch():
        for position_path in position_dirs:
            jobs.append(
                executor.submit(
                    compute_stats,
                    str(position_path),  # input_position_path
                    str(position_path),  # output_position_path (same for stats)
                    settings,
                )
            )

    if monitor:
        monitor_jobs(jobs, [str(p) for p in position_dirs])


if __name__ == "__main__":
    compute_stats_cli()
