import glob

from pathlib import Path

import click
import numpy as np
import submitit
import yaml

from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position
from natsort import natsorted

from biahub.cli.direct_copy import (
    copy_position_files,
    echo_direct_copy_summary,
    plan_position_copy,
    summarize_direct_copy,
)
from biahub.cli.monitor import monitor_jobs
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
from biahub.cli.utils import (
    PROVENANCE_METADATA_KEYS,
    copy_n_paste,
    echo_resources,
    estimate_resources,
    get_output_paths,
    get_submitit_cluster,
    model_to_yaml,
    resolve_ome_zarr_version,
    settings_fingerprint,
    yaml_to_model,
)
from biahub.settings import ConcatenateSettings

#: Memory to request when every position source is a direct chunk copy. Placing
#: files streams through the kernel rather than buffering a shard, so the shard-
#: sized estimate ``estimate_resources`` produces does not apply.
_DIRECT_COPY_MEM_GB = 8


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


def get_path_slice_param(slice_param, path_index, total_paths):
    """
    Determine the slice parameter for a specific path.

    Args:
        slice_param: The slice parameter from settings (can be 'all', a single slice range, or per-path specifications)
        path_index: The index of the current path
        total_paths: The total number of paths

    Returns
    -------
        The slice parameter for the current path
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


def create_path_slicing_params(path_z_slice, path_y_slice, path_x_slice, dataset_shape):
    """
    Create slicing parameters for a specific path.

    Args:
        path_z_slice: The Z slice parameter for the path
        path_y_slice: The Y slice parameter for the path
        path_x_slice: The X slice parameter for the path
        dataset_shape: The shape of the dataset

    Returns
    -------
        A list of slice objects [z_slice, y_slice, x_slice]
    """
    z_slice = get_slice(path_z_slice, dataset_shape[2])
    y_slice = get_slice(path_y_slice, dataset_shape[3])
    x_slice = get_slice(path_x_slice, dataset_shape[4])
    return [z_slice, y_slice, x_slice]


def get_channel_combiner_metadata(
    data_paths_list: list[str],
    processing_channel_names: list[str],
    slicing_params: list,
):
    """
    Get metadata for channel combination.

    Args:
        data_paths_list: List of data paths
        processing_channel_names: List of channel names to process
        slicing_params: List of slicing parameters [Z_slice, Y_slice, X_slice]

    Returns
    -------
        Tuple of (all_data_paths, all_channel_names, input_channel_idx, output_channel_idx, all_slicing_params)
    """
    all_data_paths = []
    all_channel_names = []
    input_channel_idx = []
    output_channel_idx = []
    out_chan_idx_counter = 0
    all_slicing_params = []

    # Unpack slicing parameters
    z_slice_param, y_slice_param, x_slice_param = slicing_params

    # Expand the data paths. Filter to directories so that per-group
    # `zarr.json` metadata files (OME-Zarr v0.5 / zarr v3) aren't picked up
    # by wildcards like "*/*/*".
    expanded_paths = []
    for paths in data_paths_list:
        expanded_paths.append(
            [Path(path) for path in natsorted(glob.glob(paths)) if Path(path).is_dir()]
        )

    # Flatten the expanded paths
    all_data_paths = [path for paths in expanded_paths for path in paths]

    # For each original path, determine the appropriate slice specifications
    for i, (paths, per_datapath_channels) in enumerate(
        zip(expanded_paths, processing_channel_names, strict=True)
    ):
        # NOTE: taking first file as sample to get the channel names
        dataset = open_ome_zarr(paths[0])
        channel_names = dataset.channel_names

        # Determine the slice specifications for this path
        path_z_slice = get_path_slice_param(z_slice_param, i, len(data_paths_list))
        path_y_slice = get_path_slice_param(y_slice_param, i, len(data_paths_list))
        path_x_slice = get_path_slice_param(x_slice_param, i, len(data_paths_list))

        # Create slicing parameters for each path in this group
        for _ in range(len(paths)):
            slicing_params = create_path_slicing_params(
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
        validate_slicing_params_zyx(all_slicing_params)

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


def get_slice(slice_param, max_value: int):
    """
    Convert slice parameters to slice objects.

    Args:
        slice_param: Can be 'all' or a single slice range [start, end]
        max_value: Maximum value for the dimension

    Returns
    -------
        A slice object
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


def validate_slicing_params_zyx(slicing_params_zyx_list: list[list[slice, slice, slice]]):
    """Validate that all slicing parameters are the same for a given dimension."""
    first_slice_size = calculate_cropped_size(slicing_params_zyx_list[0])
    for i, slice_obj in enumerate(slicing_params_zyx_list[1:], 1):
        slice_size = calculate_cropped_size(slice_obj)
        if slice_size != first_slice_size:
            raise ValueError(
                f"Inconsistent slice sizes detected. Path 0 has size {first_slice_size}, "
                f"but path {i} has size {slice_size}. All paths must have the same slice size."
            )


def calculate_cropped_size(
    slice_params_zyx: list[slice, slice, slice],
) -> tuple[int, int, int]:
    """
    Calculate the size of a dimension after cropping.

    Args:
        slice_params_zyx: A list of slice parameters for the Z, Y, and X dimensions

    Returns
    -------
        A tuple of the size of the dimension after cropping for the Z, Y, and X dimensions
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


def _prepare_concatenate(
    settings: ConcatenateSettings,
    output_dirpath: Path,
    direct_copy: bool = True,
) -> dict:
    """Derive metadata and create the output plate.

    Runs the channel/slice metadata resolution (the expensive
    ``get_channel_combiner_metadata`` call), creates the output plate, and
    returns everything the submit loop needs plus the input ``shape`` used by
    ``concatenate`` to estimate SLURM resources. ``create_empty_plate`` is
    idempotent, so calling this from both ``--init`` and the full run is safe.

    Also decides, per input position, whether its stored chunk/shard files can be
    placed into the output store verbatim instead of being read, cropped and
    rewritten. That has to happen here rather than in ``concatenate`` because it
    compares against the *output* array's metadata, which only exists once
    ``create_empty_plate`` has run. Pass ``direct_copy=False`` to skip the check
    and take the read-write path everywhere.
    """
    slicing_params = [settings.Z_slice, settings.Y_slice, settings.X_slice]
    (
        all_data_paths,
        all_channel_names,
        input_channel_idx_list,
        output_channel_idx_list,
        all_slicing_params,
    ) = get_channel_combiner_metadata(
        settings.concat_data_paths, settings.channel_names, slicing_params
    )

    output_position_paths = get_output_paths(
        all_data_paths,
        output_dirpath,
        ensure_unique_positions=settings.ensure_unique_positions,
    )

    all_shapes = []
    all_dtypes = []
    all_voxel_sizes = []
    all_array_names = []
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
            # Kept so the direct-copy planner can address the array directory;
            # the single-array check above makes this unambiguous.
            all_array_names.append(next(iter(dataset.array_keys())))

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

    # If input shapes differ but slicing is specified, inform the user
    if not all(shape[-3:] == all_shapes[0][-3:] for shape in all_shapes):
        click.echo(
            "Warning: Datasets have different shapes, but slicing parameters are specified. Will validate output shapes after cropping."
        )

    cropped_shape_zyx = calculate_cropped_size(all_slicing_params[0])
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

    source_plates = _unique_source_plates(all_data_paths)
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in output_position_paths],
        metadata_sources=list(reversed(source_plates)),
        metadata_keys=PROVENANCE_METADATA_KEYS,
        **output_metadata,
    )
    click.echo(f"Created {output_dirpath} ({len(output_position_paths)} positions)")

    output_time_indices = list(range(len(input_time_indices)))
    direct_copy_plans, direct_copy_reasons = _plan_direct_copies(
        all_data_paths=all_data_paths,
        all_array_names=all_array_names,
        output_position_paths=output_position_paths,
        input_channel_idx_list=input_channel_idx_list,
        output_channel_idx_list=output_channel_idx_list,
        all_slicing_params=all_slicing_params,
        input_time_indices=input_time_indices,
        output_time_indices=output_time_indices,
        direct_copy=direct_copy,
    )

    return {
        "all_data_paths": all_data_paths,
        "all_array_names": all_array_names,
        "output_position_paths": output_position_paths,
        "input_channel_idx_list": input_channel_idx_list,
        "output_channel_idx_list": output_channel_idx_list,
        "all_slicing_params": all_slicing_params,
        "input_time_indices": input_time_indices,
        "output_time_indices": output_time_indices,
        "direct_copy_plans": direct_copy_plans,
        "direct_copy_reasons": direct_copy_reasons,
        "shape": (T, C, Z, Y, X),
    }


def _plan_direct_copies(
    all_data_paths: list[Path],
    all_array_names: list[str],
    output_position_paths: list[Path],
    input_channel_idx_list: list[list[int]],
    output_channel_idx_list: list[list[int]],
    all_slicing_params: list[list[slice]],
    input_time_indices: list[int],
    output_time_indices: list[int],
    direct_copy: bool,
) -> tuple[list, list[list[str]]]:
    """Decide per input position whether its files can be placed verbatim.

    Returns a plan-or-None per position alongside the reasons a plan was not
    possible, both aligned with ``all_data_paths``. Only array metadata is read,
    so this stays cheap even for a plate with hundreds of position sources.
    """
    if not direct_copy:
        reason = ["disabled by --no-direct-copy"]
        return [None] * len(all_data_paths), [reason] * len(all_data_paths)

    plans = []
    reasons = []
    for (
        input_position_path,
        input_array_name,
        output_position_path,
        input_channel_idx,
        output_channel_idx,
        zyx_slicing_params,
    ) in zip(
        all_data_paths,
        all_array_names,
        output_position_paths,
        input_channel_idx_list,
        output_channel_idx_list,
        all_slicing_params,
        strict=True,
    ):
        plan, plan_reasons = plan_position_copy(
            input_position_path=input_position_path,
            output_position_path=output_position_path,
            input_channel_indices=input_channel_idx,
            output_channel_indices=output_channel_idx,
            input_time_indices=input_time_indices,
            output_time_indices=output_time_indices,
            zyx_slicing_params=zyx_slicing_params,
            input_array_name=input_array_name,
        )
        plans.append(plan)
        reasons.append(plan_reasons)
    return plans, reasons


def _resolve_concatenate_config(
    config_path: Path,
    output_config: Path,
    concat_data_paths: tuple[str, ...],
):
    """Fill in concat_data_paths and write the resolved config.

    The source config's ``concat_data_paths`` is a placeholder — typically left
    blank so Nextflow can inject the upstream store paths at runtime — so the
    override is applied to the raw YAML *before* validation. ConcatenateSettings
    requires ``concat_data_paths`` to be a non-empty list, which a blank
    placeholder is not.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["concat_data_paths"] = list(concat_data_paths)
    settings = ConcatenateSettings(**raw)
    model_to_yaml(settings, output_config)
    click.echo(f"Resolved config written to {output_config}")


def concatenate(
    settings: ConcatenateSettings,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    block: bool = False,
    monitor: bool = True,
    init_only: bool = False,
    resume: bool = False,
    direct_copy: bool = True,
    link: bool = False,
):
    """Concatenate datasets (with optional cropping).

    Parameters
    ----------
    settings : ConcatenateSettings
        Configuration settings for concatenation
    output_dirpath : Path
        Path to the output dataset
    sbatch_filepath : str | None, optional
        Path to the SLURM batch file, by default None
    cluster : str, optional
        Execution cluster: 'slurm' submits to a Slurm cluster, 'local' runs jobs
        as subprocesses on this machine, 'debug' runs jobs in-process in the
        foreground. By default 'slurm'.
    block : bool, optional
        Whether to block until all the jobs are complete,
        by default False
    monitor : bool, optional
        Whether to monitor the jobs, by default True
    init_only : bool, optional
        Only create the output store and emit RESOURCES, then exit; skip the
        per-position copy. By default False.
    resume : bool, optional
        Skip the (time, channel) units a previous attempt already finished,
        rather than recopying the whole plate. This step is a single long job
        covering every position, so a preemption or walltime kill near the end
        otherwise discards hours of work. See
        ``iohub.ngff.utils.process_single_position``.
    direct_copy : bool, optional
        Place a position's stored chunk/shard files into the output store
        verbatim when the output geometry matches the input and no ROI crop is
        requested, instead of reading, cropping and rewriting every byte. Decided
        per input position from the two arrays' metadata; anything that does not
        match falls back to the read-write path. Note that a byte copy cannot
        apply ``copy_n_paste``'s ``nan_to_num``, so NaNs are carried through
        rather than zeroed. By default True.
    link : bool, optional
        Hard-link the files the direct copy places instead of copying their
        bytes. Instant and free of disk, but the output store then shares its
        data with the inputs, so writing into either would corrupt the other.
        Only safe for write-once inputs. Ignored where the direct copy does not
        apply, and downgraded to a byte copy across filesystem boundaries. By
        default False.
    """
    slurm_out_path = output_dirpath.parent / "slurm_output"

    prep = _prepare_concatenate(settings, output_dirpath, direct_copy=direct_copy)
    input_time_indices = prep["input_time_indices"]
    direct_copy_plans = prep["direct_copy_plans"]

    summary = summarize_direct_copy(direct_copy_plans, prep["direct_copy_reasons"])
    echo_direct_copy_summary(summary)

    T, C, Z, Y, X = prep["shape"]
    batch_size = settings.shards_ratio[0] if settings.shards_ratio else 1
    _, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T // batch_size, C, Z, Y, X),
        ram_multiplier=8 * batch_size,
        max_num_cpus=16,
    )
    mem_gb = num_cpus * gb_ram_per_cpu
    time_minutes = 360
    if summary.all_direct:
        # Nothing decodes a volume, so the estimate above — which sizes a shard
        # buffer — no longer describes the job. Placing files needs concurrency,
        # not memory, and a request of hundreds of GB would queue for one of the
        # few nodes that large for no reason. (For the mantis-v2 assemble step
        # this is the difference between 96 GB and 8 GB, or 816 GB and 8 GB with
        # 10-timepoint shards.)
        # A cap, not a fixed value, so this can only ever lower the request.
        mem_gb = min(mem_gb, _DIRECT_COPY_MEM_GB)
        click.echo(
            "Every position source is a direct chunk copy; "
            f"requesting {mem_gb} GB instead of {num_cpus * gb_ram_per_cpu} GB."
        )
    echo_resources(num_cpus, mem_gb, time_minutes=time_minutes)

    if init_only:
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
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting {resolved_cluster} jobs...")
    jobs = []

    with submitit.helpers.clean_env(), executor.batch():
        for (
            input_position_path,
            input_array_name,
            output_position_path,
            input_channel_idx,
            output_channel_idx,
            zyx_slicing_params,
            direct_copy_plan,
        ) in zip(
            prep["all_data_paths"],
            prep["all_array_names"],
            prep["output_position_paths"],
            prep["input_channel_idx_list"],
            prep["output_channel_idx_list"],
            prep["all_slicing_params"],
            direct_copy_plans,
            strict=True,
        ):
            # Preserve extra_metadata written by create_empty_plate's
            # metadata_sources — process_single_position overwrites it with None
            # when the kwarg is absent (iohub <= 0.3.7) — and record this step's
            # own provenance alongside it.
            with open_ome_zarr(str(output_position_path), layout="fov", mode="r") as pos:
                existing_extra = pos.zattrs.get("extra_metadata")
            merged_extra = {
                **(existing_extra or {}),
                "biahub-concatenate": settings.model_dump(),
            }

            # A position whose stored files can be reused verbatim skips the
            # decode/encode round trip entirely; anything else takes the
            # read-write path, so the two can be mixed within one plate.
            if direct_copy_plan is not None:
                job = executor.submit(
                    copy_position_files,
                    input_position_path=input_position_path,
                    output_position_path=output_position_path,
                    input_channel_indices=input_channel_idx,
                    output_channel_indices=output_channel_idx,
                    input_time_indices=input_time_indices,
                    output_time_indices=prep["output_time_indices"],
                    zyx_slicing_params=zyx_slicing_params,
                    input_array_name=input_array_name,
                    mode="link" if link else "copy",
                    num_workers=slurm_args["slurm_cpus_per_task"],
                    resume=resume,
                    extra_metadata=merged_extra,
                )
            else:
                job = executor.submit(
                    process_single_position,
                    copy_n_paste,
                    input_position_path=input_position_path,
                    output_position_path=output_position_path,
                    input_channel_indices=input_channel_idx,
                    output_channel_indices=output_channel_idx,
                    input_time_indices=input_time_indices,
                    output_time_indices=prep["output_time_indices"],
                    num_workers=slurm_args["slurm_cpus_per_task"],
                    resume=resume,
                    resume_token=settings_fingerprint(settings),
                    extra_metadata=merged_extra,
                    zyx_slicing_params=zyx_slicing_params,
                )
            jobs.append(job)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    if block:
        _ = [job.result() for job in jobs]

    if monitor:
        monitor_jobs(jobs, prep["all_data_paths"])


@click.command("concatenate")
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@resume()
@click.option(
    "--concat-data-paths",
    multiple=True,
    type=str,
    help=(
        "Resolve mode: inject these concat_data_paths into the config and write "
        "the resolved config to -o (a YAML file), then exit. Repeat the flag once "
        "per source store."
    ),
)
@click.option(
    "--direct-copy/--no-direct-copy",
    "direct_copy",
    default=True,
    show_default=True,
    help=(
        "Place a position's stored chunk/shard files into the output verbatim "
        "when the output geometry matches the input and no ROI crop is requested, "
        "instead of decoding and re-encoding every byte. Decided per position; "
        "anything that does not match falls back to the read-write path. NaNs are "
        "carried through rather than zeroed, since a byte copy cannot inspect the "
        "data."
    ),
)
@click.option(
    "--link",
    is_flag=True,
    default=False,
    help=(
        "Hard-link the files a direct copy places instead of copying their bytes: "
        "instant and free of disk, but the output then shares its data with the "
        "input stores, so writing into either would corrupt the other. Only pass "
        "this for write-once inputs."
    ),
)
def concatenate_cli(
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    resume: bool = False,
    concat_data_paths: tuple[str, ...] = (),
    direct_copy: bool = True,
    link: bool = False,
):
    r"""Concatenate datasets (with optional cropping).

    \b
    Full end-to-end (SLURM fan-out):
    >>> biahub concatenate -c ./concat.yml -o ./output.zarr

    \b
    Resolve placeholder paths (Nextflow config prep, runs on the login node).
    Passing --concat-data-paths selects resolve mode; -o is the resolved YAML:
    >>> biahub concatenate \
        -c concat.yml -o resolved.yml \
        --concat-data-paths "deskew.zarr/*/*/*" \
        --concat-data-paths "reconstruct.zarr/*/*/*"

    \b
    Emit RESOURCES + create the output plate (Nextflow init, login node):
    >>> biahub concatenate --init -c resolved.yml -o output.zarr

    \b
    Single-shot run on a reserved compute node (Nextflow assemble step):
    'debug' iterates every position in-process; the CLI blocks until done.
    >>> biahub concatenate --cluster debug -c resolved.yml -o output.zarr

    \b
    Same, hard-linking chunks instead of copying them. Near-instant, but the
    output shares its data with the inputs, so only use it when they are
    write-once and will not be edited in place:
    >>> biahub concatenate --cluster debug --link -c resolved.yml -o output.zarr
    """
    config_path = config_filepath
    output_path = output_dirpath

    # Passing --concat-data-paths means "resolve the config": inject the paths
    # and write the resolved config to -o (a YAML file), then exit. This is the
    # only use of --concat-data-paths, so its presence selects resolve mode.
    if concat_data_paths:
        _resolve_concatenate_config(config_path, output_path, concat_data_paths)
        return

    settings = yaml_to_model(config_path, ConcatenateSettings)

    # Default: full end-to-end concatenation (or --init to only create the plate
    # and emit RESOURCES).
    # For in-node clusters ('debug' runs in-process, 'local' spawns
    # subprocesses) the jobs execute lazily and only run when their result is
    # awaited, so block here — otherwise the command would return before any
    # data is written. 'slurm' keeps the submit-and-detach behaviour (optionally
    # followed with --monitor), matching the other CLIs.
    block = cluster in ("debug", "local")
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        block=block,
        monitor=monitor,
        init_only=init_only,
        resume=resume,
        direct_copy=direct_copy,
        link=link,
    )


if __name__ == "__main__":
    concatenate_cli()
