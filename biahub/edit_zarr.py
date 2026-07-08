from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import submitit

from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    copy_n_paste,
    echo_resources,
    estimate_resources,
    get_output_paths,
    get_submitit_cluster,
    resolve_ome_zarr_version,
    yaml_to_model,
)
from biahub.concatenate import get_slice
from biahub.settings import EditZarrSettings


@dataclass
class Edition:
    """A single output store to produce from the input, with its work list.

    One ``edit-zarr`` invocation resolves to one Edition without ``divide`` and
    to several with it. ``input_channel_indices`` are flat integer indices into
    the source channels (channel drop/reorder); ``output_channel_names`` are the
    renamed names written to the output store.
    """

    store_path: Path
    input_positions: list[Path]
    output_positions: list[Path]
    input_channel_indices: list[int]
    output_channel_names: list[str]


def _resolve_channels(
    all_channel_names: list[str], settings: EditZarrSettings
) -> tuple[list[str], dict[str, str]]:
    """Return the ordered kept input-channel names and the rename map.

    ``channels == "all"`` keeps every channel with its original name; a list
    selects a subset (dropping the rest) and renames any channel whose ``output``
    is set.
    """
    if settings.channels == "all":
        return list(all_channel_names), {}

    rename = {c.input: (c.output or c.input) for c in settings.channels}
    kept = [c.input for c in settings.channels]
    missing = [name for name in kept if name not in all_channel_names]
    if missing:
        raise click.ClickException(
            f"channels not found in input {all_channel_names}: {missing}"
        )
    return kept, rename


def _position_key(path: Path) -> str:
    """Row/Col/FOV key like 'A/1/0' from a position path."""
    return "/".join(Path(path).parts[-3:])


def _resolve_editions(
    input_positions: list[Path],
    output_dirpath: Path,
    settings: EditZarrSettings,
    all_channel_names: list[str],
) -> list[Edition]:
    """Build the list of output stores (Editions) to produce.

    Without ``divide`` this is a single store at ``output_dirpath``. With
    ``divide`` each group becomes ``<output_dirpath>/<group.name>.zarr`` -- one
    store per channel subset (by="channels") or per position subset
    (by="positions"). Editions with no input positions (e.g. a per-position
    Nextflow run whose position is not in a group) are omitted.
    """
    kept, rename = _resolve_channels(all_channel_names, settings)

    def selection(input_names: list[str]) -> tuple[list[int], list[str]]:
        idxs = [all_channel_names.index(n) for n in input_names]
        names = [rename.get(n, n) for n in input_names]
        return idxs, names

    editions: list[Edition] = []

    if settings.divide is None:
        idxs, names = selection(kept)
        editions.append(
            Edition(
                store_path=output_dirpath,
                input_positions=input_positions,
                output_positions=get_output_paths(input_positions, output_dirpath),
                input_channel_indices=idxs,
                output_channel_names=names,
            )
        )
        return editions

    if settings.divide.by == "channels":
        for group in settings.divide.groups:
            not_kept = [c for c in group.channels if c not in kept]
            if not_kept:
                raise click.ClickException(
                    f"divide group '{group.name}' references channels not kept by "
                    f"'channels' {kept}: {not_kept}"
                )
            idxs, names = selection(group.channels)
            store = output_dirpath / f"{group.name}.zarr"
            editions.append(
                Edition(
                    store_path=store,
                    input_positions=input_positions,
                    output_positions=get_output_paths(input_positions, store),
                    input_channel_indices=idxs,
                    output_channel_names=names,
                )
            )
        return editions

    # by == "positions"
    idxs, names = selection(kept)
    for group in settings.divide.groups:
        wanted = set(group.positions)
        group_inputs = [p for p in input_positions if _position_key(p) in wanted]
        if not group_inputs:
            continue
        store = output_dirpath / f"{group.name}.zarr"
        editions.append(
            Edition(
                store_path=store,
                input_positions=group_inputs,
                output_positions=get_output_paths(group_inputs, store),
                input_channel_indices=idxs,
                output_channel_names=names,
            )
        )
    return editions


def _resolve_time_indices(time_indices: int | list[int] | str, T: int) -> list[int]:
    if time_indices == "all":
        return list(range(T))
    if isinstance(time_indices, int):
        return [time_indices]
    return list(time_indices)


def _write_position(
    output_position_path: Path,
    source_specs: list[tuple[Path, list[int], list[int]]],
    time_indices: list[int],
    output_time_indices: list[int],
    zyx_slicing_params: list,
) -> None:
    """Fill one output position from one or more source stores.

    ``source_specs`` is a list of ``(source_position_path, input_channel_indices,
    output_channel_indices)``. Each source is read and cropped (same ZYX/T crop)
    and written into the given output channels. Sources are processed
    sequentially in a single job so that channels sharing an output position are
    never written by two concurrent jobs -- the input copy and the substituted
    channels land in one pass.
    """
    z_slice, y_slice, x_slice = zyx_slicing_params
    with open_ome_zarr(str(output_position_path), mode="r+") as out_ds:
        out_arr = out_ds["0"]
        for source_position_path, in_channel_idx, out_channel_idx in source_specs:
            with open_ome_zarr(str(source_position_path), mode="r") as in_ds:
                in_arr = in_ds["0"]
                for t_in, t_out in zip(time_indices, output_time_indices, strict=True):
                    for c_in, c_out in zip(in_channel_idx, out_channel_idx, strict=True):
                        zyx = in_arr[t_in, c_in, z_slice, y_slice, x_slice]
                        out_arr[t_out, c_out] = np.nan_to_num(zyx, nan=0)


def _resolve_substitutions(settings: EditZarrSettings) -> dict[str, Path]:
    """Map each substituted output-channel name to its source store path."""
    source_by_channel: dict[str, Path] = {}
    for sub in settings.substitute_channels:
        for name in sub.channels:
            if name in source_by_channel:
                raise click.ClickException(
                    f"channel '{name}' is substituted from more than one source."
                )
            source_by_channel[name] = Path(sub.source)
    return source_by_channel


def edit_zarr(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
):
    """Edit an OME-Zarr store: crop (T/ZYX), drop/rename channels, and/or divide.

    Mirrors the ``deskew`` execution model so it plugs into both the SLURM
    fan-out and the Nextflow pipeline: ``--init`` creates the output store(s)
    and emits the ``RESOURCES:`` line, and per-position work runs under
    ``--cluster {slurm,local,debug}``.

    Parameters
    ----------
    input_position_dirpaths : list[str]
        Paths to input positions, e.g. "input.zarr/*/*/*" or "input.zarr/A/1/0".
    config_filepath : Path
        Path to the EditZarrSettings YAML.
    output_dirpath : str
        Output store path (without ``divide``) or the parent directory that
        holds ``<group>.zarr`` stores (with ``divide``).
    sbatch_filepath : str, optional
        SBATCH file overriding the default SLURM parameters.
    cluster : str, optional
        'slurm' (default), 'local', or 'debug' (in-process, for Nextflow).
    monitor : bool, optional
        Monitor submitted SLURM jobs.
    init_only : bool, optional
        Only create the output store(s) and emit resources, then exit.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    settings = yaml_to_model(config_filepath, EditZarrSettings)

    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as sample:
        all_channel_names = sample.channel_names
        T, C, Z, Y, X = sample.data.shape
        voxel_size = tuple(sample.scale[-3:])

    input_positions = [Path(p) for p in input_position_dirpaths]
    editions = _resolve_editions(input_positions, output_dirpath, settings, all_channel_names)

    # Substitution: output-channel-name -> source store, validated against the
    # resolved output channels and each source's channel list.
    sub_source_by_channel = _resolve_substitutions(settings)
    source_channel_names: dict[Path, list[str]] = {}
    if sub_source_by_channel:
        output_names = {name for e in editions for name in e.output_channel_names}
        unknown = sorted(n for n in sub_source_by_channel if n not in output_names)
        if unknown:
            raise click.ClickException(
                f"substitute_channels not present in output channels {sorted(output_names)}: "
                f"{unknown}"
            )
        first_key = _position_key(input_positions[0])
        for name, source in sub_source_by_channel.items():
            if source not in source_channel_names:
                with open_ome_zarr(str(source / first_key), mode="r") as src:
                    source_channel_names[source] = src.channel_names
            if name not in source_channel_names[source]:
                raise click.ClickException(
                    f"channel '{name}' not found in substitute source {source} "
                    f"({source_channel_names[source]})"
                )

    time_indices = _resolve_time_indices(settings.time_indices, T)
    z_slice = get_slice(settings.Z_slice, Z)
    y_slice = get_slice(settings.Y_slice, Y)
    x_slice = get_slice(settings.X_slice, X)
    zyx_slicing_params = [z_slice, y_slice, x_slice]
    cropped_zyx = (
        z_slice.stop - z_slice.start,
        y_slice.stop - y_slice.start,
        x_slice.stop - x_slice.start,
    )

    version = resolve_ome_zarr_version(
        input_position_dirpaths[0], settings.output_ome_zarr_version
    )
    input_plate = Path(input_position_dirpaths[0]).parents[2]
    chunks = ([1] + list(settings.chunks_czyx)) if settings.chunks_czyx else None

    for edition in editions:
        create_empty_plate(
            store_path=edition.store_path,
            position_keys=[p.parts[-3:] for p in edition.output_positions],
            channel_names=edition.output_channel_names,
            shape=(len(time_indices), len(edition.output_channel_names)) + cropped_zyx,
            chunks=chunks,
            shards_ratio=settings.shards_ratio,
            scale=(1, 1) + voxel_size,
            version=version,
            metadata_sources=input_plate,
        )

    # Resources scale with a single output volume; the largest kept channel
    # count across editions bounds the per-position working set.
    max_out_channels = max(len(e.output_channel_names) for e in editions)
    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(len(time_indices), max_out_channels) + cropped_zyx,
        ram_multiplier=4,
        max_num_cpus=16,
    )
    mem_gb = num_cpus * gb_ram_per_cpu
    time_minutes = 60
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        click.echo(
            f"Initialized {len(editions)} store(s) under {output_dirpath} "
            f"({sum(len(e.input_positions) for e in editions)} position jobs)"
        )
        return

    slurm_args = {
        "slurm_job_name": "edit-zarr",
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

    output_time_indices = list(range(len(time_indices)))
    click.echo("Submitting jobs...")
    jobs = []
    job_labels = []
    with submitit.helpers.clean_env(), executor.batch():
        for edition in editions:
            output_channel_indices = list(range(len(edition.output_channel_names)))
            for input_position_path, output_position_path in zip(
                edition.input_positions, edition.output_positions, strict=True
            ):
                if not sub_source_by_channel:
                    # Fast path (no substitution): the proven concatenate/deskew
                    # per-position machinery, unchanged.
                    jobs.append(
                        executor.submit(
                            process_single_position,
                            copy_n_paste,
                            input_position_path=input_position_path,
                            output_position_path=output_position_path,
                            input_channel_indices=edition.input_channel_indices,
                            output_channel_indices=output_channel_indices,
                            input_time_indices=time_indices,
                            output_time_indices=output_time_indices,
                            num_workers=int(slurm_args["slurm_cpus_per_task"]),
                            zyx_slicing_params=zyx_slicing_params,
                        )
                    )
                    job_labels.append(output_position_path)
                    continue

                # With substitution, route each output channel to its source
                # (input store, or a substitute store) and write them all in one
                # job per position so no two jobs touch the same position.
                position_key = _position_key(output_position_path)
                input_in_idx, input_out_idx = [], []
                sub_groups: dict[Path, list[int]] = {}
                for out_idx, name in enumerate(edition.output_channel_names):
                    if name in sub_source_by_channel:
                        sub_groups.setdefault(sub_source_by_channel[name], []).append(out_idx)
                    else:
                        input_in_idx.append(edition.input_channel_indices[out_idx])
                        input_out_idx.append(out_idx)

                source_specs: list[tuple[Path, list[int], list[int]]] = []
                if input_out_idx:
                    source_specs.append((input_position_path, input_in_idx, input_out_idx))
                for source, out_idxs in sub_groups.items():
                    src_names = source_channel_names[source]
                    src_in_idx = [
                        src_names.index(edition.output_channel_names[j]) for j in out_idxs
                    ]
                    source_specs.append((source / position_key, src_in_idx, out_idxs))

                jobs.append(
                    executor.submit(
                        _write_position,
                        output_position_path=output_position_path,
                        source_specs=source_specs,
                        time_indices=time_indices,
                        output_time_indices=output_time_indices,
                        zyx_slicing_params=zyx_slicing_params,
                    )
                )
                job_labels.append(output_position_path)

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # DebugExecutor runs lazily; drive each job in the foreground (see deskew).
    if resolved_cluster == "debug":
        for job, label in zip(jobs, job_labels, strict=True):
            job.wait()
            click.echo(f"Edit complete: {label}")
        return

    if monitor:
        monitor_jobs(jobs, job_labels)


@click.command("edit-zarr")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
def edit_zarr_cli(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
):
    """Edit an OME-Zarr: crop (T/ZYX), drop/rename channels, and/or divide into stores.

    \b
    SLURM fan-out over a whole plate:
    >>> biahub edit-zarr -i ./input.zarr/*/*/* -c ./edit.yml -o ./output.zarr

    \b
    Initialize the output store(s) only (e.g. before per-position Nextflow workers):
    >>> biahub edit-zarr --init -i ./input.zarr/*/*/* -c ./edit.yml -o ./output.zarr

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub edit-zarr --cluster debug -i ./input.zarr/A/1/0 -c ./edit.yml -o ./output.zarr
    """  # noqa: D301
    edit_zarr(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
    )


if __name__ == "__main__":
    edit_zarr_cli()
