import glob

from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
import submitit

from iohub import open_ome_zarr
from iohub.ngff import Plate
from iohub.ngff.utils import create_empty_plate, process_single_position
from natsort import natsorted

from biahub.cli.monitor import monitor_jobs
from biahub.cli.option_eat_all import OptionEatAll
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    monitor,
    output_dirpath,
    resume,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.settings import ConcatenateSettings
from biahub.utils.array_ops import copy_n_paste
from biahub.utils.cluster import echo_resources, estimate_resources, get_submitit_cluster
from biahub.utils.config import settings_fingerprint, yaml_to_model
from biahub.utils.ngff import (
    PROVENANCE_METADATA_KEYS,
    get_output_paths,
    resolve_ome_zarr_version,
)


def _unique_source_plates(data_paths: list[Path]) -> list[Path]:
    """Deduplicated source plate paths from position paths, preserving order."""
    seen = set()
    plates = []
    for p in data_paths:
        plate = Path(p).parents[2]
        if plate not in seen:
            seen.add(plate)
            plates.append(plate)
    return plates


def _path_slice_param(slice_param, path_index, total_paths):
    """Select the slice specification that applies to one source.

    Parameters
    ----------
    slice_param : str | list
        The setting for one axis: ``"all"``, a single ``[start, end]`` range that
        applies to every source, or a list with one entry per source.
    path_index : int
        Index of the source among the per-source entries.
    total_paths : int
        Number of sources.

    Returns
    -------
    str | list
        ``"all"`` or the ``[start, end]`` range for this source. A per-source list
        shorter than ``total_paths`` repeats its last entry.
    """
    # Handle 'all' case
    if slice_param == "all":
        return "all"

    # Handle single slice range [start, end]
    if isinstance(slice_param, list):
        if len(slice_param) == 2 and all(isinstance(i, int) for i in slice_param):
            return slice_param
        else:
            return (
                slice_param[path_index] if path_index < len(slice_param) else slice_param[-1]
            )

    # Handle any other case
    return slice_param


def _path_slicing_params(path_z_slice, path_y_slice, path_x_slice, dataset_shape):
    """Build the ZYX slice objects for one source.

    Parameters
    ----------
    path_z_slice, path_y_slice, path_x_slice : str | list
        Per-axis specification for this source, ``"all"`` or ``[start, end]``.
    dataset_shape : tuple[int, ...]
        TCZYX shape of the source, which bounds an ``"all"`` slice.

    Returns
    -------
    list[slice]
        ``[z_slice, y_slice, x_slice]``.
    """
    z_slice = _slice(path_z_slice, dataset_shape[2])
    y_slice = _slice(path_y_slice, dataset_shape[3])
    x_slice = _slice(path_x_slice, dataset_shape[4])
    return [z_slice, y_slice, x_slice]


def _expand_source_globs(concat_data_paths: list[str]) -> list[list[Path]]:
    """Expand the config's per-source globs into per-source position lists.

    Filters to directories so that per-group ``zarr.json`` metadata files
    (OME-Zarr v0.5 / zarr v3) aren't picked up by wildcards like ``*/*/*``.
    """
    groups = []
    for pattern in concat_data_paths:
        group = [Path(p) for p in natsorted(glob.glob(pattern)) if Path(p).is_dir()]
        if not group:
            raise ValueError(f"No positions matched concat_data_paths entry {pattern!r}.")
        groups.append(group)
    return groups


def _validate_per_source_lengths(settings: ConcatenateSettings, num_sources: int) -> None:
    """Per-source lists must have one entry per source.

    ``ConcatenateSettings`` can only check this against ``concat_data_paths``;
    when the sources come from ``-i`` the count is first known here.
    """
    if isinstance(settings.channel_names, list) and len(settings.channel_names) != num_sources:
        raise ValueError(
            f"channel_names has {len(settings.channel_names)} entries for {num_sources} "
            "sources. Use 'all' or one entry per source."
        )
    for name in ("Z_slice", "Y_slice", "X_slice"):
        value = getattr(settings, name)
        is_single_range = (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(i, int) for i in value)
        )
        if isinstance(value, list) and not is_single_range and len(value) != num_sources:
            raise ValueError(
                f"{name} has {len(value)} entries for {num_sources} sources. Use 'all', "
                "a single [start, end] range, or one entry per source."
            )


def _channel_combiner_metadata(
    source_groups: list[list[Path]],
    processing_channel_names: list[str | list[str]] | str,
    slicing_params: list,
):
    """Resolve which channels of which source go where in the output.

    Parameters
    ----------
    source_groups : list[list[Path]]
        One list of position paths per source store.
    processing_channel_names : str | list[str | list[str]]
        ``"all"`` (every channel of every source) or one entry per source, each
        ``"all"`` or the channel names to take from it.
    slicing_params : list
        ``[Z_slice, Y_slice, X_slice]`` settings.

    Returns
    -------
    tuple
        ``(all_data_paths, all_channel_names, input_channel_idx, output_channel_idx,
        all_slicing_params)``, each per source position except the output
        channel names.
    """
    all_data_paths = []
    all_channel_names = []
    input_channel_idx = []
    output_channel_idx = []
    out_chan_idx_counter = 0
    all_slicing_params = []

    # Unpack slicing parameters
    z_slice_param, y_slice_param, x_slice_param = slicing_params

    if processing_channel_names == "all":
        processing_channel_names = ["all"] * len(source_groups)

    # Flatten the expanded paths
    all_data_paths = [path for paths in source_groups for path in paths]

    # For each original path, determine the appropriate slice specifications
    for i, (paths, per_datapath_channels) in enumerate(
        zip(source_groups, processing_channel_names, strict=True)
    ):
        # NOTE: taking first file as sample to get the channel names
        dataset = open_ome_zarr(paths[0])
        channel_names = dataset.channel_names

        # Determine the slice specifications for this path
        path_z_slice = _path_slice_param(z_slice_param, i, len(source_groups))
        path_y_slice = _path_slice_param(y_slice_param, i, len(source_groups))
        path_x_slice = _path_slice_param(x_slice_param, i, len(source_groups))

        # Create slicing parameters for each path in this group
        for _ in range(len(paths)):
            slicing_params = _path_slicing_params(
                path_z_slice, path_y_slice, path_x_slice, dataset.data.shape
            )
            all_slicing_params.append(slicing_params)

        # Parse channels
        output_channel_indices = []
        input_channel_indices = []

        if per_datapath_channels == "all":
            per_datapath_channels = channel_names

        for channel in per_datapath_channels:
            if channel in channel_names:
                # If the channel already exists in the list, we don't want to add it again
                if channel not in all_channel_names:
                    all_channel_names.append(channel)
                    output_channel_indices.append(out_chan_idx_counter)
                    out_chan_idx_counter += 1
                else:
                    click.echo(
                        f"Warning: Channel {channel} already exists. Skipping and using index from the first entry."
                    )
                    # Set the out_chan_idx_counter to the index of the channel in the all_channel_names list
                    out_chan_idx_counter = all_channel_names.index(channel)
                    output_channel_indices.append(out_chan_idx_counter)
                input_channel_indices.append(channel_names.index(channel))

        dataset.close()

        # Create a list of len paths
        input_channel_idx.extend([input_channel_indices for _ in paths])
        output_channel_idx.extend([output_channel_indices for _ in paths])

    # Validate that all slicing parameters produce the same output size
    if len(all_slicing_params) > 1:
        _validate_slicing_params_zyx(all_slicing_params)

    click.echo(f"Channel names: {all_channel_names}")
    click.echo(f"Input channel indices: {input_channel_idx}")
    click.echo(f"Output channel indices: {output_channel_idx}")

    return (
        all_data_paths,
        all_channel_names,
        input_channel_idx,
        output_channel_idx,
        all_slicing_params,
    )


def _slice(slice_param, max_value: int) -> slice:
    """Convert one axis specification to a slice object.

    Parameters
    ----------
    slice_param : str | list
        ``"all"`` or a single ``[start, end]`` range.
    max_value : int
        Extent of the axis, the stop of an ``"all"`` slice.

    Returns
    -------
    slice

    Raises
    ------
    ValueError
        If ``slice_param`` is neither ``"all"`` nor a two-integer list.
    """
    # Handle 'all' case
    if slice_param == "all":
        return slice(0, max_value)

    # Handle single slice range [start, end]
    if (
        isinstance(slice_param, list)
        and len(slice_param) == 2
        and all(isinstance(i, int) for i in slice_param)
    ):
        return slice(*slice_param)

    raise ValueError(f"Invalid slice parameter: {slice_param}")


def _validate_slicing_params_zyx(slicing_params_zyx_list: list[list[slice]]) -> None:
    """Check that every source crops to the same ZYX size.

    Parameters
    ----------
    slicing_params_zyx_list : list[list[slice]]
        One ``[z_slice, y_slice, x_slice]`` per source position.

    Raises
    ------
    ValueError
        If any source's cropped size differs from the first's.
    """
    first_slice_size = _cropped_size(slicing_params_zyx_list[0])
    for i, slice_obj in enumerate(slicing_params_zyx_list[1:], 1):
        slice_size = _cropped_size(slice_obj)
        if slice_size != first_slice_size:
            raise ValueError(
                f"Inconsistent slice sizes detected. Path 0 has size {first_slice_size}, "
                f"but path {i} has size {slice_size}. All paths must have the same slice size."
            )


def _cropped_size(slice_params_zyx: list[slice]) -> tuple[int, int, int]:
    """Size of a volume after cropping.

    Parameters
    ----------
    slice_params_zyx : list[slice]
        ``[z_slice, y_slice, x_slice]``.

    Returns
    -------
    tuple[int, int, int]
        The cropped ZYX shape.
    """
    # Calculate the size of each dimension by taking the absolute difference between stop and start
    z_size = abs(slice_params_zyx[0].stop - slice_params_zyx[0].start)
    y_size = abs(slice_params_zyx[1].stop - slice_params_zyx[1].start)
    x_size = abs(slice_params_zyx[2].stop - slice_params_zyx[2].start)

    cropped_shape_zyx = (z_size, y_size, x_size)
    click.echo(f"Output ZYX shape after cropping: {cropped_shape_zyx}")

    return cropped_shape_zyx


def _resolve_time_indices(settings: ConcatenateSettings, all_shapes: list[tuple]) -> list[int]:
    """Resolve input time indices from settings and shapes."""
    T = all_shapes[0][0]
    if settings.time_indices == "all":
        if not all(s[0] == T for s in all_shapes):
            click.echo(
                "Warning: Datasets have different number of time points. "
                "Taking the smallest number of time points."
            )
        T = min(s[0] for s in all_shapes)
        return list(range(T))
    elif isinstance(settings.time_indices, list):
        return settings.time_indices
    elif isinstance(settings.time_indices, int):
        return [settings.time_indices]
    return list(range(T))


def _validate_source_groups(
    ctx: click.Context, opt: click.Option, value: tuple[tuple[str, ...], ...]
) -> list[list[Path]] | None:
    """Each ``-i`` occurrence is one source store's positions."""
    if not value:
        return None
    groups = []
    for group in value:
        paths = [p for p in map(Path, natsorted(group)) if p.is_dir()]
        if not paths:
            raise click.BadParameter(f"No position directories in {list(group)}")
        with open_ome_zarr(paths[0], mode="r") as dataset:
            if isinstance(dataset, Plate):
                raise click.BadParameter(
                    f"{paths[0]} is an HCS plate; supply positions, e.g. {paths[0]}/*/*/*"
                )
        groups.append(paths)
    return groups


def input_position_dirpaths() -> Callable:
    """``-i``, repeated once per source store.

    Concatenate's own variant of ``biahub.cli.parsing.input_position_dirpaths``:
    that one flattens every ``-i`` into a single list, whereas here the i-th
    ``-i`` is the i-th source and pairs with the i-th per-source entry of the
    config (channel_names, X/Y/Z_slice). Each occurrence eats the paths a shell
    glob expands to, up to the next option, so ``-i a.zarr/*/*/* -i b.zarr/*/*/*``
    and ``-i a.zarr/A/1/0 -i b.zarr/A/1/0`` both parse as two groups.
    """

    def decorator(f: Callable) -> Callable:
        return click.option(
            "--input-position-dirpaths",
            "-i",
            "input_position_dirpaths",
            cls=OptionEatAll,
            type=tuple,
            multiple=True,
            callback=_validate_source_groups,
            help=(
                "Positions of ONE source store; repeat once per store, in the order of "
                'the config\'s per-source entries. For example "-i a.zarr/*/*/* -i b.zarr/*/*/*" '
                'or, for a single position, "-i a.zarr/A/1/0 -i b.zarr/A/1/0". Overrides '
                "concat_data_paths in the config."
            ),
        )(f)

    return decorator


def _resolve_concatenate_inputs(
    settings: ConcatenateSettings,
    output_dirpath: Path,
    source_groups: list[list[Path]],
) -> dict:
    """Resolve the per-source-position work list and the output plate geometry.

    Runs the channel/slice metadata resolution (the expensive
    ``_channel_combiner_metadata`` call) and reads each source position's
    shape, dtype and scale. It reads METADATA only and writes nothing, so every
    mode calls it: the full run and ``--init`` hand the result to
    ``_init_output_plate``; a per-position worker (``-i`` naming one position
    per source) gets a one-position work list from the same code.
    """
    _validate_per_source_lengths(settings, len(source_groups))
    slicing_params = [settings.Z_slice, settings.Y_slice, settings.X_slice]
    (
        all_data_paths,
        all_channel_names,
        input_channel_idx_list,
        output_channel_idx_list,
        all_slicing_params,
    ) = _channel_combiner_metadata(source_groups, settings.channel_names, slicing_params)

    output_position_paths = get_output_paths(
        all_data_paths,
        output_dirpath,
        ensure_unique_positions=settings.ensure_unique_positions,
    )

    all_shapes = []
    all_dtypes = []
    all_voxel_sizes = []
    for path in all_data_paths:
        with open_ome_zarr(path) as dataset:
            if len(dataset.array_keys()) > 1:
                # TODO: https://github.com/czbiohub-sf/biahub/issues/192
                raise ValueError(
                    "Concatenation of datasets with multiple arrays (pyramid levels) is not supported."
                )
            all_shapes.append(dataset.data.shape)
            all_dtypes.append(dataset.data.dtype)
            all_voxel_sizes.append(dataset.scale[-3:])

    # Only check for shape compatibility when using 'all' for slicing
    if (
        settings.Z_slice == "all"
        and settings.Y_slice == "all"
        and settings.X_slice == "all"
        and not all(shape[-3:] == all_shapes[0][-3:] for shape in all_shapes)
    ):
        raise ValueError(
            "Datasets have different shapes. All ZYX shapes must match to concatenate when using 'all' for slicing."
        )

    if not all(voxel_size == all_voxel_sizes[0] for voxel_size in all_voxel_sizes):
        click.echo(
            "Warning: Datasets have different voxel sizes. Taking the first voxel size."
        )

    T, C, Z, Y, X = all_shapes[0]
    output_voxel_size = all_voxel_sizes[0]

    if all(dtype == all_dtypes[0] for dtype in all_dtypes):
        dtype = all_dtypes[0]
    else:
        click.echo("Warning: not all dtypes match. Casting data at float32.")
        dtype = np.float32

    input_time_indices = _resolve_time_indices(settings, all_shapes)

    # A per-position worker only sees its own sources, so with time_indices
    # "all" its T is the minimum over those, not over the plate. The plate that
    # --init created is the authority: never write past its T.
    first_output = output_position_paths[0]
    if settings.time_indices == "all" and first_output.is_dir():
        with open_ome_zarr(first_output, mode="r") as existing:
            plate_T = existing.data.shape[0]
        if len(input_time_indices) > plate_T:
            click.echo(
                f"Warning: sources have {len(input_time_indices)} time points but "
                f"{output_dirpath} was created with {plate_T}. Writing the first {plate_T}."
            )
            input_time_indices = input_time_indices[:plate_T]

    # If input shapes differ but slicing is specified, inform the user
    if not all(shape[-3:] == all_shapes[0][-3:] for shape in all_shapes):
        click.echo(
            "Warning: Datasets have different shapes, but slicing parameters are specified. Will validate output shapes after cropping."
        )

    cropped_shape_zyx = _cropped_size(all_slicing_params[0])
    if cropped_shape_zyx[0] > Z or cropped_shape_zyx[1] > Y or cropped_shape_zyx[2] > X:
        raise ValueError("The cropped shape is larger than the original shape.")

    if settings.chunks_czyx is not None:
        chunk_size = [1] + list(settings.chunks_czyx)
    else:
        chunk_size = settings.chunks_czyx

    output_metadata = {
        "shape": (len(input_time_indices), len(all_channel_names)) + tuple(cropped_shape_zyx),
        "chunks": chunk_size,
        "shards_ratio": settings.shards_ratio,
        "version": resolve_ome_zarr_version(
            all_data_paths[0], settings.output_ome_zarr_version
        ),
        "scale": (1,) * 2 + tuple(output_voxel_size),
        "channel_names": all_channel_names,
        "dtype": dtype,
    }

    return {
        "all_data_paths": all_data_paths,
        "output_position_paths": output_position_paths,
        "input_channel_idx_list": input_channel_idx_list,
        "output_channel_idx_list": output_channel_idx_list,
        "all_slicing_params": all_slicing_params,
        "input_time_indices": input_time_indices,
        "shape": (T, C, Z, Y, X),
        "output_metadata": output_metadata,
    }


def _init_output_plate(
    prep: dict, settings: ConcatenateSettings, output_dirpath: Path
) -> None:
    """Create the output positions that do not exist yet, stamping provenance once.

    Only positions missing from the plate are created, so the
    ``biahub-concatenate`` record is written exactly once per position: by the
    full run or by ``--init``. A per-position Nextflow worker finds its position
    already scaffolded and writes nothing here. (``create_empty_plate`` would
    otherwise re-stamp every position it is handed on every call, and N
    workers re-stamping is N writes per position racing on the same zattrs.)
    """
    missing = {p.parts[-3:] for p in prep["output_position_paths"] if not p.is_dir()}
    if not missing:
        return
    source_plates = _unique_source_plates(prep["all_data_paths"])
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=sorted(missing),
        metadata_sources=list(reversed(source_plates)),
        metadata_keys=PROVENANCE_METADATA_KEYS,
        extra_metadata={"biahub-concatenate": settings.model_dump()},
        **prep["output_metadata"],
    )
    click.echo(f"Created {len(missing)} positions in {output_dirpath}")


def concatenate(
    input_position_dirpaths: list[list[Path]] | None,
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
    resume: bool = False,
):
    """Concatenate datasets channel-wise (with optional cropping).

    Parameters
    ----------
    input_position_dirpaths : list[list[Path]] | None
        Source positions, one list per source store, in the order of the config's
        per-source entries (channel_names, X/Y/Z_slice). Takes precedence over
        ``concat_data_paths`` in the config; None expands the config's globs.
        One position per store is the per-position worker mode Nextflow fans
        out over.
    config_filepath : Path
        Path to YAML configuration file.
    output_dirpath : Path
        Path to "output.zarr" directory.
    sbatch_filepath : str, optional
        SBATCH filepath that contains slurm parameters to overwrite defaults.
        For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU.
    cluster : str, optional
        Execution cluster: 'slurm' submits to a Slurm cluster, 'local' runs jobs as
        subprocesses on this machine, 'debug' runs jobs in-process in the foreground.
    monitor : bool, optional
        Monitor of submitted SLURM jobs.
    init_only : bool, optional
        Only initialize the output store and exit; skip per-position processing.
    resume : bool, optional
        Skip the (time, channel) units a previous attempt already finished,
        rather than recopying the position. For retrying an interrupted run;
        see ``iohub.ngff.utils.process_single_position``.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    settings = yaml_to_model(config_filepath, ConcatenateSettings)
    if input_position_dirpaths is not None:
        source_groups = [list(map(Path, group)) for group in input_position_dirpaths]
    elif settings.concat_data_paths:
        source_groups = _expand_source_globs(settings.concat_data_paths)
    else:
        raise ValueError(
            "No sources: pass one -i per source store, or set concat_data_paths in the config."
        )

    prep = _resolve_concatenate_inputs(settings, output_dirpath, source_groups)
    _init_output_plate(prep, settings, output_dirpath)
    input_time_indices = prep["input_time_indices"]

    # Per-position resources, estimated once. Calibrated on 2026_08_11 A549
    # SEC61B (67 T x 6 C, 5-T shards): RAM tracks the worker count (~16 GB per
    # in-flight shard unit), and the fan-out is bound by shared filesystem
    # bandwidth, not cores (<3 of 16 busy), so 8 workers per task suffice.
    # 16-worker tasks took 7-26 min; 0.15 min/volume budgets 60 min here.
    T_out, C_out, _, _, _ = prep["output_metadata"]["shape"]
    _, _, Z, Y, X = prep["shape"]
    batch_size = settings.shards_ratio[0] if settings.shards_ratio else 1
    time_minutes, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(max(T_out // batch_size, 1), C_out, Z, Y, X),
        ram_multiplier=8 * batch_size,
        time_multiplier=0.15 * batch_size,
        max_num_cpus=8,
    )
    mem_gb = num_cpus * gb_ram_per_cpu
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        num_positions = len({p.parts[-3:] for p in prep["output_position_paths"]})
        click.echo(f"Initialized {output_dirpath} ({num_positions} positions)")
        return

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "concatenate",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": time_minutes,
        "slurm_partition": "preempted",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []

    # One job per SOURCE position: with three source stores an output position
    # is written by three jobs, each owning the disjoint channel range it
    # contributes. A per-position worker runs exactly those jobs.
    with submitit.helpers.clean_env(), executor.batch():
        for (
            input_position_path,
            output_position_path,
            input_channel_idx,
            output_channel_idx,
            zyx_slicing_params,
        ) in zip(
            prep["all_data_paths"],
            prep["output_position_paths"],
            prep["input_channel_idx_list"],
            prep["output_channel_idx_list"],
            prep["all_slicing_params"],
            strict=True,
        ):
            job = executor.submit(
                process_single_position,
                copy_n_paste,
                input_position_path=input_position_path,
                output_position_path=output_position_path,
                input_channel_indices=input_channel_idx,
                output_channel_indices=output_channel_idx,
                input_time_indices=input_time_indices,
                output_time_indices=list(range(len(input_time_indices))),
                num_workers=slurm_args["slurm_cpus_per_task"],
                resume=resume,
                resume_token=settings_fingerprint(settings),
                zyx_slicing_params=zyx_slicing_params,
            )
            jobs.append(job)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a
    # DebugJob but execution only happens when .wait()/.done()/.result() is
    # called. Run each one in the foreground and stream progress; monitor's
    # async polling UI is pointless against synchronous in-process jobs.
    if resolved_cluster == "debug":
        for job, path in zip(jobs, prep["all_data_paths"], strict=True):
            job.wait()
            click.echo(f"Concatenate complete: {path}")
        return

    if monitor:
        monitor_jobs(jobs, prep["all_data_paths"])


@click.command("concatenate")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@resume()
def concatenate_cli(
    input_position_dirpaths: list[list[Path]] | None,
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    resume: bool = False,
):
    """Concatenate datasets channel-wise (with optional cropping).

    Sources come from one -i per store (or from concat_data_paths in the config).

    \b
    SLURM fan-out of positions across whole plates:
    >>> biahub concatenate -i ./deskew.zarr/*/*/* -i ./phase.zarr/*/*/* -c ./concat.yml -o ./output.zarr

    \b
    Initialize the output plate only (e.g. before running per-position Nextflow workers):
    >>> biahub concatenate --init -i ./deskew.zarr/*/*/* -i ./phase.zarr/*/*/* -c ./concat.yml -o ./output.zarr

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub concatenate --cluster debug -i ./deskew.zarr/A/1/0 -i ./phase.zarr/A/1/0 -c ./concat.yml -o ./output.zarr
    """  # noqa: D301
    concatenate(
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
    concatenate_cli()
