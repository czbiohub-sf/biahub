"""Project a 3D plate to a 2D (Z=1) plate, one projection method per output channel.

Each output channel is an input channel reduced over a selection of Z-planes (``max``,
``mean``, ``sum``) or one plane of it (``slice``). The selection is a fixed range, a
window around a plane, or a window around the in-focus plane, which is found per position
with the registration engine's focus criterion (see ``ProjectionSettings``). Segment and
track take the projected plate as their 2D input; neither projects by itself.
"""

import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
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
from biahub.settings import ChannelProjection, ProjectionFocusSettings, ProjectionSettings
from biahub.utils.cluster import (
    apply_sbatch_overrides,
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
    local_executor_parameters,
)
from biahub.utils.config import settings_fingerprint, yaml_to_model
from biahub.utils.ngff import (
    PROVENANCE_METADATA_KEYS,
    get_output_paths,
    resolve_ome_zarr_version,
)

PROVENANCE_KEY = "biahub-project"

# Methods that compute a new value; a plate with any of them is float32 (the common
# case is float32 in and out). Only an all-`slice` plate keeps the input dtype.
_FLOAT_METHODS = ("max", "mean", "sum")


def resolve_z_indices(
    projection: ChannelProjection, z_size: int, focus_z: int | None = None
) -> np.ndarray:
    """Return the plane indices one channel's projection reads from a ``z_size``-plane stack.

    An unset ``z`` is the whole stack. ``start``/``stop`` are absolute planes; with a
    ``center`` (``focus_z`` when it is ``"focus"``) the planes are
    ``[center - below, center + above)``, or the center alone for ``slice``. ``step``
    thins the selection. Indices outside the stack are dropped; a selection that keeps
    none is an error.
    """
    z = projection.z
    if z is None:
        return np.arange(z_size)
    if z.is_absolute:
        start, stop = z.start, z.stop
    else:
        center = focus_z if z.center == "focus" else z.center
        if center is None:
            raise ValueError("z.center is 'focus' but no focus plane was given.")
        if z.is_window:
            start, stop = center - z.below, center + z.above
        else:
            start, stop = center, center + 1
    indices = np.arange(start, stop, z.step)
    indices = indices[(indices >= 0) & (indices < z_size)]
    if indices.size == 0:
        raise ValueError(
            f"z planes {start}..{stop} (step {z.step}) select nothing from a {z_size}-plane "
            "stack."
        )
    return indices


def project_zyx(zyx: np.ndarray, method: str, z_indices: np.ndarray) -> np.ndarray:
    """Reduce a (Z, Y, X) volume to (Y, X) over ``z_indices`` with ``method``."""
    if method == "slice":
        return zyx[int(z_indices[0])]
    planes = zyx[z_indices]
    if method == "max":
        return planes.max(axis=0)
    if method == "mean":
        return planes.mean(axis=0).astype(np.float32, copy=False)
    if method == "sum":
        return planes.sum(axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown projection method {method!r}.")


def _project_czyx(
    czyx_data: np.ndarray,
    plans: list[tuple[int, ChannelProjection]],
    focus_z: int | np.ndarray | None = None,
    input_time_index: int | None = None,
) -> np.ndarray:
    """Project one timepoint's (C, Z, Y, X) read to the (C_out, 1, Y, X) output.

    ``plans`` pairs each output channel with its index in ``czyx_data``. ``focus_z`` is
    the position's in-focus plane, or one per timepoint (indexed with
    ``input_time_index``, which iohub passes because it is in the signature).
    """
    z_size = czyx_data.shape[1]
    if isinstance(focus_z, np.ndarray):
        focus_z = int(focus_z[input_time_index])
    projected = [
        project_zyx(czyx_data[c], plan.method, resolve_z_indices(plan, z_size, focus_z))
        for c, plan in plans
    ]
    return np.stack(projected)[:, np.newaxis]


def _resolve_channels(
    settings: ProjectionSettings, channel_names: list[str], z_size: int
) -> tuple[list[tuple[int, ChannelProjection]], list[int], int | None]:
    """Map the config's channel names onto the input plate.

    Returns the plans (index into the read, projection) in output order, the input
    channel indices to read (the outputs, then the focus channel if it is not one), and
    the focus channel's index in the read. Fixed selections are resolved against the
    stack here, so a range outside it fails before any job is submitted.
    """
    for name in settings.channels:
        if name not in channel_names:
            raise ValueError(f"Channel {name!r} is not in the input channels {channel_names}.")
    read = [channel_names.index(name) for name in settings.channels]
    plans = list(enumerate(settings.channels.values()))

    focus_index = None
    if settings.uses_focus:
        focus_channel = settings.focus.channel
        if focus_channel not in channel_names:
            raise ValueError(
                f"focus.channel {focus_channel!r} is not in the input channels {channel_names}."
            )
        if focus_channel not in settings.channels:
            read.append(channel_names.index(focus_channel))
        focus_index = channel_names.index(focus_channel)

    for name, projection in settings.channels.items():
        if not projection.uses_focus:
            try:
                resolve_z_indices(projection, z_size)
            except ValueError as e:
                raise ValueError(f"Channel {name}: {e}") from e

    dropped = [name for name in channel_names if name not in settings.channels]
    if dropped:
        click.echo(f"Warning: input channels not in the config are dropped: {dropped}")
    return plans, read, focus_index


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    settings: ProjectionSettings,
) -> tuple[tuple[int, int, int, int, int], list[str], np.dtype]:
    """Create the empty Z=1 output plate and record the settings as provenance.

    Returns the input (T, C, Z, Y, X) shape, its channel names and the output dtype.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        channel_names = input_dataset.channel_names
        T, C, Z, Y, X = input_dataset.data.shape
        scale = input_dataset.scale
        input_dtype = input_dataset.data.dtype

    if Z == 1:
        raise ValueError(
            f"{input_position_dirpaths[0]} already has one Z-plane; nothing to project."
        )
    # Resolve the config against the plate before scaffolding, so a bad channel name or
    # range fails here. The dropped-channel warning is printed once, at init.
    _resolve_channels(settings, channel_names, Z)

    methods = {projection.method for projection in settings.channels.values()}
    dtype = np.dtype(np.float32) if methods & set(_FLOAT_METHODS) else np.dtype(input_dtype)

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(p).parts[-3:] for p in input_position_dirpaths],
        channel_names=list(settings.channels),
        shape=(T, len(settings.channels), 1, Y, X),
        scale=scale,
        version=resolve_ome_zarr_version(
            input_position_dirpaths[0], settings.output_ome_zarr_version
        ),
        dtype=dtype,
        chunks=(1, *settings.chunks_czyx) if settings.chunks_czyx else None,
        shards_ratio=tuple(settings.shards_ratio) if settings.shards_ratio else None,
        metadata_sources=input_plate,
        metadata_keys=PROVENANCE_METADATA_KEYS,
        extra_metadata={PROVENANCE_KEY: settings.model_dump(mode="json")},
    )
    return (T, C, Z, Y, X), channel_names, dtype


def _focus_of_timepoint(
    position_path: Path,
    t: int,
    channel_index: int,
    crop_xy: tuple[int, int],
    pixel_size: float,
    na_det: float,
    lambda_ill: float,
) -> int | None:
    """Find the in-focus plane of one timepoint, or None when the frame has no focus.

    Top-level so a process pool can pickle it; each call reads its own (Z, Y, X) volume.
    """
    from biahub.registration.estimators import EstimationError
    from biahub.registration.methods.focus import center_crop_yx, find_focus

    with open_ome_zarr(str(position_path), mode="r") as dataset:
        zyx = np.asarray(dataset.data[t, channel_index])
    try:
        return find_focus(center_crop_yx(zyx, crop_xy), pixel_size, na_det, lambda_ill)
    except EstimationError:
        return None


def _find_focus_per_t(
    position_path: Path,
    channel_index: int,
    focus: ProjectionFocusSettings,
    pixel_size: float,
    num_workers: int = 1,
) -> np.ndarray:
    """Find the in-focus plane of every timepoint of one position, on ``focus.channel``.

    Measured on the centre crop with waveorder's transverse band like the engine's
    FocusEstimator; an empty frame, or one without a focus, takes the middle plane. The
    timepoints are independent, so with ``num_workers > 1`` they run in a process pool
    (spawned, like iohub's, so a worker killed by the cgroup surfaces as an error);
    each worker reads one (Z, Y, X) volume at a time.
    """
    with open_ome_zarr(str(position_path), mode="r") as dataset:
        T, _, Z, _, _ = dataset.data.shape
    crop_xy = (focus.center_crop_xy[0], focus.center_crop_xy[1])
    constants = (channel_index, crop_xy, pixel_size, focus.na_det, focus.lambda_ill)

    if num_workers > 1 and T > 1:
        with ProcessPoolExecutor(
            max_workers=min(num_workers, T), mp_context=mp.get_context("spawn")
        ) as pool:
            planes = list(
                pool.map(
                    _focus_of_timepoint,
                    repeat(position_path),
                    range(T),
                    *(repeat(value) for value in constants),
                )
            )
    else:
        planes = [_focus_of_timepoint(position_path, t, *constants) for t in range(T)]

    focus_z = np.array([Z // 2 if plane is None else plane for plane in planes], dtype=int)
    return np.clip(focus_z, 0, Z - 1)


def _project_position(
    input_position_path: Path,
    output_position_path: Path,
    settings: ProjectionSettings,
    plans: list[tuple[int, ChannelProjection]],
    input_channel_indices: list[int],
    focus_channel_index: int | None,
    num_workers: int,
    resume: bool,
) -> None:
    """Project one position: the focus pre-pass, then every timepoint.

    The focus is found once per position for both ``focus.per`` modes and recorded under
    the output position's ``biahub-project`` metadata, so the plane each (t) was
    projected around can be read back.
    """
    focus_z = None
    if settings.uses_focus:
        with open_ome_zarr(str(input_position_path), mode="r") as dataset:
            pixel_size = dataset.scale[-1]
        focus_per_t = _find_focus_per_t(
            input_position_path,
            focus_channel_index,
            settings.focus,
            pixel_size,
            num_workers=num_workers,
        )
        if settings.focus.per == "fov":
            focus_z = int(round(float(np.median(focus_per_t))))
        else:
            focus_z = focus_per_t
        click.echo(
            f"Focus on {settings.focus.channel}: "
            + (
                f"plane {focus_z} (median over T)"
                if settings.focus.per == "fov"
                else f"{focus_per_t.tolist()}"
            )
        )
        with open_ome_zarr(str(output_position_path), mode="r+") as output_dataset:
            record = dict(output_dataset.zattrs.get(PROVENANCE_KEY, {}))
            record["focus"] = {
                **settings.focus.model_dump(mode="json", exclude={"axes"}),
                "z": focus_z if settings.focus.per == "fov" else None,
                "z_per_t": focus_per_t.tolist(),
            }
            output_dataset.zattrs[PROVENANCE_KEY] = record

    process_single_position(
        _project_czyx,
        input_position_path,
        output_position_path,
        input_channel_indices=[input_channel_indices],
        output_channel_indices=[list(range(len(plans)))],
        num_workers=num_workers,
        resume=resume,
        resume_token=settings_fingerprint(settings),
        plans=plans,
        focus_z=focus_z,
    )


def project(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
    resume: bool = False,
):
    """Project a 3D plate to a 2D (Z=1) plate, per channel.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to the input position directories.
    config_filepath : Path
        Path to the projection configuration file (``ProjectionSettings``).
    output_dirpath : Path
        Path to the output plate.
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
        Skip the (time, channel) units a previous attempt already finished, rather
        than recomputing the whole position; see ``iohub.ngff.utils.process_single_position``.
        The focus pre-pass runs again either way.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    settings = yaml_to_model(config_filepath, ProjectionSettings)
    (T, _, Z, Y, X), channel_names, _ = _init_output_plate(
        input_position_dirpaths, output_dirpath, settings
    )
    plans, read_indices, focus_index = _resolve_channels(settings, channel_names, Z)

    # Not yet calibrated on a completed run. RAM: one timepoint's read (the listed
    # channels plus the focus channel) per worker, with headroom for the float64
    # accumulation. Time per (t, c) volume: a reduction is cheap, so 0.1 min carries a
    # wide margin; the focus pre-pass reads the focus channel once more per position
    # (~40% on top of track's old window selection, #418), hence the higher rate.
    time_minutes, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, len(read_indices), Z, Y, X),
        ram_multiplier=4,
        time_multiplier=0.15 if settings.uses_focus else 0.1,
        max_num_cpus=8,
    )
    mem_gb = num_cpus * gb_ram_per_cpu
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        click.echo(f"Initialized {output_dirpath} ({len(input_position_dirpaths)} positions)")
        return

    output_position_paths = get_output_paths(input_position_dirpaths, output_dirpath)

    slurm_args = {
        "slurm_job_name": "project",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": time_minutes,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args = apply_sbatch_overrides(slurm_args, sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    if resolved_cluster == "local":
        # Explicit --sbatch-filepath values (e.g. '#LOCAL --timeout-min') win over
        # the computed local executor parameters.
        slurm_args = {**local_executor_parameters(time_minutes), **slurm_args}
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths, strict=True
        ):
            jobs.append(
                executor.submit(
                    _project_position,
                    input_position_path,
                    output_position_path,
                    settings=settings,
                    plans=plans,
                    input_channel_indices=read_indices,
                    focus_channel_index=focus_index,
                    num_workers=slurm_args["slurm_cpus_per_task"],
                    resume=resume,
                )
            )

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a DebugJob but
    # execution only happens on .wait()/.done()/.result(). On the Nextflow path
    # (--cluster debug) run each position in the foreground.
    if resolved_cluster == "debug":
        for job, path in zip(jobs, input_position_dirpaths, strict=True):
            job.wait()
            click.echo(f"Projection complete: {path}")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("project")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@resume()
def project_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    resume: bool = False,
):
    """Project a 3D plate to a 2D (Z=1) plate, with a projection method per channel.

    Each output channel is max, mean or sum over a Z range or window, or one plane of the
    stack. The range or plane may be fixed or centred on the in-focus plane; see
    settings/example_project_settings.yml.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub project -i ./input.zarr/*/*/* -c ./project.yml -o ./output.zarr

    \b
    Initialize the output plate only (e.g. before running per-position Nextflow workers):
    >>> biahub project --init -i ./input.zarr/*/*/* -c ./project.yml -o ./output.zarr

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub project --cluster debug -i ./input.zarr/A/1/0 -c ./project.yml -o ./output.zarr
    """  # noqa: D301
    project(
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
    project_cli()
