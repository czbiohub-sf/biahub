import warnings

from pathlib import Path

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    resume,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.settings import FlatFieldCorrectionSettings
from biahub.utils.cluster import echo_resources, estimate_resources, get_submitit_cluster
from biahub.utils.config import settings_fingerprint, yaml_to_model
from biahub.utils.ngff import (
    PROVENANCE_METADATA_KEYS,
    get_output_paths,
    resolve_ome_zarr_version,
)

# Byte budget for one median tile. The reduction below runs at DRAM latency when
# its working set spills out of cache, so the tile is sized to fit in the smallest
# per-CCX L3 in the cluster (16 MiB on Zen 2), with headroom for the partition's
# own scratch. See :func:`_median_tiled` for why this matters.
_MEDIAN_TILE_BYTES = 8 * 1024**2


def _median_tile_axis(data: np.ndarray, axis: int) -> int | None:
    """Return the first axis that is not the reduction axis, or None for 1-D input."""
    return next((a for a in range(data.ndim) if a != axis), None)


def _median_tile_width(data: np.ndarray, axis: int, tile_bytes: int) -> int:
    """Return how many indices along the tiled axis fit in ``tile_bytes``.

    Never less than 1, never more than the axis length.
    """
    tile_axis = _median_tile_axis(data, axis)
    bytes_per_step = data.itemsize * (data.size // data.shape[tile_axis])
    width = tile_bytes // max(bytes_per_step, 1)
    return int(min(max(width, 1), data.shape[tile_axis]))


def _median_tiled(
    data: np.ndarray, axis: int, tile_bytes: int = _MEDIAN_TILE_BYTES
) -> np.ndarray:
    """``np.median(data, axis=axis)``, evaluated over cache-resident tiles.

    Output is identical to :func:`numpy.median` -- only the memory access pattern
    changes. ``np.median`` partitions along ``axis``, whose stride in a ZYX volume
    is ``Y * X * itemsize`` (852 KB for a mantis-v2 position), so every element
    touched lands on its own cache line and the reduction runs at DRAM latency.
    That cost is borne unevenly by the cluster: measured on one position, it is
    11.5x worse on Zen 2 cores (EPYC 7742, 7302P) than on Zen 3 and newer, which
    made flat-field spend ~139 CPU-s per (t, c) unit on ``gpu-a``/``gpu-sm`` nodes
    against ~12 CPU-s on ``cpu-h``.

    Copying a slim tile into a compact buffer first keeps the partition in cache.
    Per-position medians measured before and after:

    ==================  ===========  =========  ========
    node                CPU          np.median  tiled
    ==================  ===========  =========  ========
    ``gpu-a-3``         EPYC 7742    134.33 s   5.52 s
    ``gpu-sm02-19``     EPYC 7302P    99.26 s   5.72 s
    ``cpu-e-1``         EPYC 7763      9.89 s   5.18 s
    ==================  ===========  =========  ========

    Transposing to make the reduction axis contiguous was also tried and is much
    worse on the affected nodes (78.25 s on ``gpu-a-3``), because the transpose
    copy is itself a strided read.
    """
    tile_axis = _median_tile_axis(data, axis)
    if tile_axis is None:
        return np.median(data, axis=axis)

    tile = _median_tile_width(data, axis, tile_bytes)
    tiles = []
    for start in range(0, data.shape[tile_axis], tile):
        index = [slice(None)] * data.ndim
        index[tile_axis] = slice(start, start + tile)
        # ascontiguousarray is what buys the locality: without the copy the tile
        # is a view carrying the full array's strides, so the partition still
        # walks the whole buffer.
        tiles.append(np.median(np.ascontiguousarray(data[tuple(index)]), axis=axis))

    if len(tiles) == 1:
        return tiles[0]
    # The reduction drops `axis`, shifting the tiled axis down when it sat after it.
    return np.concatenate(tiles, axis=tile_axis - (1 if tile_axis > axis else 0))


def flat_field_zyx(zyx_data: np.ndarray, axis: int = 0) -> np.ndarray:
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
    static_pattern = _median_tiled(zyx_data, axis=axis)
    return zyx_data / static_pattern * static_pattern.mean()


def flat_field_correction(zyx_data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Apply flat field correction (deprecated alias for :func:`flat_field_zyx`).

    .. deprecated::
        Renamed to :func:`flat_field_zyx` to follow the ``<verb>_zyx`` layer-1
        naming convention. This alias will be removed in a future release.

    Parameters
    ----------
    zyx_data : np.ndarray
        The data to apply flat field correction to.
    axis : int
        The axis to compute the median along.

    Returns
    -------
    np.ndarray
        Flat-field corrected data (see :func:`flat_field_zyx`).
    """
    warnings.warn(
        "flat_field_correction is deprecated; use flat_field_zyx instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return flat_field_zyx(zyx_data, axis=axis)


def _flat_field_czyx(czyx_data: np.ndarray, target_indices: list[int]) -> np.ndarray:
    """Apply flat-field correction to selected channels of a CZYX volume.

    Channels listed in ``target_indices`` are corrected; the rest are
    passed through unchanged (cast to float32 to match the output dtype).
    """
    out = np.empty_like(czyx_data, dtype=np.float32)
    target = set(target_indices)
    for c in range(czyx_data.shape[0]):
        if c in target:
            out[c] = flat_field_zyx(czyx_data[c])
        else:
            out[c] = czyx_data[c].astype(np.float32)
    return out


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    settings: FlatFieldCorrectionSettings,
) -> tuple[tuple[int, int, int, int, int], list[str]]:
    """Create the empty flat-field output plate.

    Returns the input (T, C, Z, Y, X) shape and channel names.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        all_channel_names = input_dataset.channel_names
        input_shape = input_dataset.data.shape
        scale = input_dataset.scale

    T, C, Z, Y, X = input_shape

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(p).parts[-3:] for p in input_position_dirpaths],
        channel_names=all_channel_names,
        shape=(T, C, Z, Y, X),
        scale=scale,
        version=resolve_ome_zarr_version(
            input_position_dirpaths[0], settings.output_ome_zarr_version
        ),
        dtype=np.float32,
        metadata_sources=input_plate,
        metadata_keys=PROVENANCE_METADATA_KEYS,
    )

    return (T, C, Z, Y, X), all_channel_names


def _resolve_target_indices(
    settings: FlatFieldCorrectionSettings,
    all_channel_names: list[str],
) -> list[int]:
    """Resolve which channel indices to flat-field correct."""
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
    return [all_channel_names.index(name) for name in target_channel_names]


def flat_field(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
    resume: bool = False,
):
    """Apply flat field correction across T and selected C axes.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to the input position directories.
    config_filepath : Path
        Path to the configuration file.
    output_dirpath : Path
        Path to the output directory.
    sbatch_filepath : str, optional
        Path to the SLURM batch file.
    cluster : str, optional
        Execution cluster: 'slurm' submits to a Slurm cluster, 'local' runs jobs as
        subprocesses on this machine, 'debug' runs jobs in-process in the foreground.
    monitor : bool, optional
        If True, monitor the submitted jobs.
    init_only : bool, optional
        Only initialize the output store and exit; skip per-position processing.
    resume : bool, optional
        Skip the (time, channel) units a previous attempt already finished,
        rather than recomputing the whole position. For retrying an interrupted
        run; see ``iohub.ngff.utils.process_single_position``.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    settings = yaml_to_model(config_filepath, FlatFieldCorrectionSettings)
    input_shape, all_channel_names = _init_output_plate(
        input_position_dirpaths, output_dirpath, settings
    )

    # RAM scales with one ZYX volume (ram_multiplier=8); wall-time scales with
    # the number of volumes (T*C). time_multiplier = 0.7 min/volume: the worst
    # per-volume rate observed over completed runs is 0.34 min/volume (A549
    # 2026_07_14, 68.3 min for 201 volumes; neuromast 2026_06_25 is 0.28), so
    # this carries a ~2x margin. Channel selection only reduces work, so using
    # all C is a safe upper bound.
    time_minutes, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=input_shape, ram_multiplier=8, time_multiplier=0.7, max_num_cpus=16
    )
    mem_gb = num_cpus * gb_ram_per_cpu
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        click.echo(f"Initialized {output_dirpath} ({len(input_position_dirpaths)} positions)")
        return

    output_position_paths = get_output_paths(input_position_dirpaths, output_dirpath)
    target_indices = _resolve_target_indices(settings, all_channel_names)

    flat_field_args = {
        "target_indices": target_indices,
        "extra_metadata": {"biahub-flat_field": settings.model_dump()},
    }

    slurm_args = {
        "slurm_job_name": "flat-field",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": time_minutes,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths, strict=True
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    _flat_field_czyx,
                    input_position_path,
                    output_position_path,
                    num_workers=slurm_args["slurm_cpus_per_task"],
                    resume=resume,
                    resume_token=settings_fingerprint(settings),
                    **flat_field_args,
                )
            )

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a
    # DebugJob but execution only happens when .wait()/.done()/.result() is
    # called. Run each one in the foreground and stream progress; monitor's
    # async polling UI is pointless against synchronous in-process jobs.
    if resolved_cluster == "debug":
        for job, path in zip(jobs, input_position_dirpaths, strict=True):
            job.wait()
            click.echo(f"Flat-field complete: {path}")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("flat-field")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@resume()
def flat_field_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    resume: bool = False,
):
    """Apply flat field correction across T and selected C axes.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub flat-field -i ./input.zarr/*/*/* -c ./flat_field_params.yml -o ./output.zarr

    \b
    Initialize the output plate only (e.g. before running per-position Nextflow workers):
    >>> biahub flat-field --init -i ./input.zarr/*/*/* -c ./flat_field_params.yml -o ./output.zarr

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub flat-field --cluster debug -i ./input.zarr/A/1/0 -c ./flat_field_params.yml -o ./output.zarr
    """  # noqa: D301
    flat_field(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
        resume=resume,
    )


if __name__ == "__main__":
    flat_field_cli()
