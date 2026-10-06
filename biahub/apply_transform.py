"""Apply a transform series to positions: one matrix for every timepoint or one per timepoint.

Replaces `register` (one transform, source onto a target store) and `stabilize` (one
transform per timepoint, a store onto itself). The output canvas is decided once, before
the output store is allocated: the largest box inside the overlap of the warped source
and the reference, intersected over every timepoint's transform -- or, with
`keep_overhang`, the reference grid as is.
"""

from __future__ import annotations

import hashlib
import json

from pathlib import Path

import ants
import click
import largestinteriorrectangle as lir
import numpy as np
import submitit

from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    monitor,
    moving_position_dirpaths,
    output_dirpath,
    pair_reference_positions,
    position_key,
    reference_position_dirpaths,
    resume,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.registration.utils import (
    apply_affine_transform,
    convert_transform_to_ants,
    rescale_voxel_size,
)
from biahub.settings import TransformSettings, load_transform_settings
from biahub.utils.array_ops import copy_n_paste_czyx
from biahub.utils.cluster import echo_resources, estimate_resources, get_submitit_cluster
from biahub.utils.ngff import resolve_ome_zarr_version

Slices = tuple[slice, slice, slice]


def _resolve_time_indices(time_indices, n_t: int) -> list[int]:
    if time_indices == "all":
        return list(range(n_t))
    if isinstance(time_indices, int):
        return [time_indices]
    return list(time_indices)


def _coarse(shape_zyx: tuple[int, int, int], f: int) -> tuple[int, int, int]:
    return tuple(max(1, int(np.ceil(s / f))) for s in shape_zyx)


def overlap_mask(
    source_shape_zyx: tuple[int, int, int],
    target_shape_zyx: tuple[int, int, int],
    inverse_matrix: np.ndarray,
    downsample: int = 4,
) -> np.ndarray:
    """Target-grid voxels the warped source covers, on a `downsample`-times coarser grid.

    The warp of an all-ones source is the same geometry at any resolution, so a
    240-timepoint series stays cheap.
    """
    f = max(1, int(downsample))
    scale = np.diag([1.0 / f, 1.0 / f, 1.0 / f, 1.0])
    coarse = scale @ np.asarray(inverse_matrix, dtype=float) @ np.linalg.inv(scale)
    ones_source = ants.from_numpy(np.ones(_coarse(source_shape_zyx, f), dtype=np.float32))
    ones_target = ants.from_numpy(np.ones(_coarse(target_shape_zyx, f), dtype=np.float32))
    warped = convert_transform_to_ants(coarse).apply_to_image(
        ones_source, reference=ones_target
    )
    return warped.numpy() > 0


def largest_box(mask: np.ndarray) -> Slices:
    """Exact largest axis-aligned box of True voxels in a (Z, Y, X) mask.

    For every z-range the slices are ANDed and the exact largest 2D rectangle found
    (`largestinteriorrectangle`); the largest volume wins. `registration.utils.find_lir`
    is a heuristic (rectangle at the middle slice, z clipped to where that rectangle
    still fits) and on a sheared overlap gives up most of z; this is O(Z^2) 2D searches,
    which on the canvas's coarse grid is a few seconds.
    """
    mask = np.asarray(mask, dtype=bool)
    best, best_volume = None, 0
    for z0 in range(mask.shape[0]):
        combined = mask[z0].copy()
        for z1 in range(z0, mask.shape[0]):
            combined &= mask[z1]
            if not combined.any():
                break
            depth = z1 - z0 + 1
            if depth * combined.sum() <= best_volume:
                continue  # even the full remaining area cannot beat the best box
            x, y, width, height = map(int, lir.lir(combined))
            volume = depth * width * height
            if volume > best_volume:
                best, best_volume = (
                    (slice(z0, z1 + 1), slice(y, y + height), slice(x, x + width)),
                    volume,
                )
    if best is None:
        raise ValueError("empty mask")
    return best


def _mask_box(
    mask: np.ndarray, target_shape_zyx: tuple[int, int, int], downsample: int
) -> Slices:
    """Largest box inside a coarse mask, rounded inward onto the full-resolution grid."""
    f = max(1, int(downsample))
    if not mask.any():
        raise click.UsageError(
            "the transform leaves no overlapping region between source and target; "
            "use keep_overhang: true or check the transform"
        )
    z, y, x = largest_box(mask)
    return tuple(
        slice(min(int(np.ceil(s.start * f)), dim), min(int(np.floor(s.stop * f)), dim))
        for s, dim in zip((z, y, x), target_shape_zyx, strict=True)
    )


def overlap_slices(
    source_shape_zyx: tuple[int, int, int],
    target_shape_zyx: tuple[int, int, int],
    inverse_matrix: np.ndarray,
    downsample: int = 4,
) -> Slices:
    """Largest box inside the overlap of the warped source and the target grid."""
    mask = overlap_mask(source_shape_zyx, target_shape_zyx, inverse_matrix, downsample)
    return _mask_box(mask, target_shape_zyx, downsample)


def canvas(
    source_shape_zyx: tuple[int, int, int],
    target_shape_zyx: tuple[int, int, int],
    inverse_matrices: list[np.ndarray],
    keep_overhang: bool,
    downsample: int = 4,
) -> Slices:
    """Find the one crop every timepoint shares: the LIR of the intersected overlap masks.

    The masks are intersected first and the exact largest box found once
    (`largest_box`). Finding a box per transform and intersecting the boxes is much
    smaller: on a rotated overlap each timepoint's box trades the axes differently.

    With `keep_overhang` the full target grid is kept (content the transform pushes
    outside it is lost, content it pulls in from outside is blank).
    """
    if keep_overhang:
        return tuple(slice(0, dim) for dim in target_shape_zyx)
    unique = {
        np.asarray(m, dtype=float).round(9).tobytes(): np.asarray(m, dtype=float)
        for m in inverse_matrices
    }
    mask = None
    for matrix in unique.values():
        current = overlap_mask(source_shape_zyx, target_shape_zyx, matrix, downsample)
        mask = current if mask is None else mask & current
    if mask is None or not mask.any():
        raise click.UsageError(
            "the transforms share no overlapping region across timepoints; "
            "use keep_overhang: true or check the transforms"
        )
    return _mask_box(mask, target_shape_zyx, downsample)


def _apply_transform_czyx(
    czyx_data: np.ndarray,
    matrices: list,
    input_time_index: int,
    output_shape_zyx: tuple[int, int, int],
    crop_output_slicing: list[slice] | None,
    interpolation: str,
) -> np.ndarray:
    """Warp one (C, Z, Y, X) block with its timepoint's matrix (or the single matrix)."""
    matrix = np.asarray(matrices[input_time_index if len(matrices) > 1 else 0], dtype=float)
    return apply_affine_transform(
        czyx_data,
        matrix,
        output_shape_zyx,
        interpolation=interpolation,
        crop_output_slicing=crop_output_slicing,
    )


def parse_time_indices(value: str) -> int | list[int] | str:
    """'all', one index, or a comma-separated list, as typed on the command line."""
    value = value.strip()
    if value == "all":
        return "all"
    indices = [int(v) for v in value.split(",") if v.strip()]
    return indices[0] if len(indices) == 1 else indices


def apply_transform(
    moving_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    reference_position_dirpaths: list[Path] | None = None,
    time_indices: int | list[int] | str = "all",
    keep_overhang: bool = False,
    interpolation: str = "linear",
    output_ome_zarr_version: str | None = None,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    channels: list[str] | None = None,
    init_only: bool = False,
    resume: bool = False,
) -> None:
    """Apply a `TransformSettings` series to positions.

    `channels` picks which moving-store channels are transformed. By default all of them (a
    registration or stabilization comes from the shared optics and stage, so it holds for
    every channel), except that with reference positions a channel the reference store also
    has is copied from it unless it is one of the file's `moving_channels` (so with moving
    and reference in one store only the estimated channels move).

    With reference positions: the output lives on the reference grid and holds the
    reference channels copied plus the transformed moving channels (registration). Without:
    the moving channels are transformed onto their own grid (stabilization). Each timepoint
    takes its own entry's matrix (`TransformSettings.matrix_for`); a series-wide entry
    applies to all. The canvas is the largest box inside the overlap shared by every
    transform in the file at the chosen timepoints -- all positions, not only the ones
    applied, so each position can be written by its own task -- or the full reference grid
    with `keep_overhang`.

    `init_only` creates the output plate and prints `RESOURCES:` (one position's task)
    without writing data; a later call for some positions writes into it. `resume` skips
    the (time, channel) units a previous attempt of the same apply already wrote.
    """
    output_dirpath = Path(output_dirpath)
    settings: TransformSettings = load_transform_settings(config_filepath)

    with open_ome_zarr(moving_position_dirpaths[0], mode="r") as moving:
        T, _C, *moving_shape = moving.data.shape
        moving_channel_names = list(moving.channel_names)
        moving_voxel_size = tuple(moving.scale[-3:])
    unknown = sorted(set(channels or []) - set(moving_channel_names))
    if unknown:
        raise click.UsageError(
            f"channels not in the moving store: {unknown} (it has {moving_channel_names})"
        )
    to_transform = [c for c in moving_channel_names if channels is None or c in channels]
    time_indices = _resolve_time_indices(time_indices, T)
    # Each moving position's matrices by timepoint: its own list, or the shared one.
    position_keys = [position_key(p) for p in moving_position_dirpaths]
    if settings.per_position:
        missing = sorted(set(position_keys) - set(settings.positions))
        if missing:
            raise click.UsageError(
                f"the transforms file has no list for positions {missing}; it has "
                f"{sorted(settings.positions)}"
            )
    inverses = {
        key: {t: settings.matrix_for(t, "inverse", key) for t in time_indices}
        for key in position_keys
    }
    reference_for = pair_reference_positions(position_keys, reference_position_dirpaths)
    # Timepoints written with a transform that is not `accepted`, per position.
    not_accepted = {}
    for key in position_keys:
        by_status = {}
        for t in time_indices:
            status = settings.entry_for(t, key).status
            if status != "accepted":
                by_status.setdefault(status, []).append(t)
        if by_status:
            not_accepted[key] = by_status
            click.echo(
                f"{key}: written with transforms that are not accepted -- "
                + "; ".join(f"{s} t={ts}" for s, ts in by_status.items())
            )
    file_first_key = next(iter(settings.positions)) if settings.per_position else None
    applied_first = settings.matrix_for(time_indices[0], "inverse", file_first_key)

    if reference_position_dirpaths:
        with open_ome_zarr(reference_position_dirpaths[0], mode="r") as reference:
            reference_shape = tuple(reference.data.shape[-3:])
            reference_channel_names = list(reference.channel_names)
            reference_voxel_size = list(reference.scale)
        if channels is None:
            transformed = [
                c
                for c in to_transform
                if c not in reference_channel_names or c in settings.moving_channels
            ]
            if not transformed:
                raise click.UsageError(
                    f"every moving channel is also in the reference store and none is the "
                    f"file's moving_channels {settings.moving_channels}; choose them with "
                    f"--channels"
                )
        else:
            transformed = to_transform
            replaced = [
                c
                for c in transformed
                if c in reference_channel_names and c not in settings.moving_channels
            ]
            if replaced:
                click.echo(f"transforming {replaced} replaces the reference store's copy")
        # A reference channel that is also transformed is written once, by its transform job.
        copied = [c for c in reference_channel_names if c not in transformed]
        output_channel_names = reference_channel_names + [
            c for c in transformed if c not in reference_channel_names
        ]
        output_voxel_size = tuple(reference_voxel_size[-3:])
    else:
        reference_shape = tuple(moving_shape)
        transformed, copied = to_transform, []
        output_channel_names = to_transform
        output_voxel_size = (
            tuple(settings.voxel_size[-3:])
            if settings.voxel_size
            else tuple(
                # the file's first transform, whichever positions are applied
                rescale_voxel_size(applied_first[:3, :3], moving_voxel_size)
            )
        )

    # Every transform in the file at these timepoints, whichever positions are applied.
    file_keys = list(settings.positions) if settings.per_position else [position_keys[0]]
    applied = [
        settings.matrix_for(t, "inverse", key) for key in file_keys for t in time_indices
    ]
    crop = canvas(tuple(moving_shape), reference_shape, applied, keep_overhang)
    cropped_shape = tuple(s.stop - s.start for s in crop)
    click.echo(
        f"Output canvas {cropped_shape} (reference grid {reference_shape}, "
        f"{'kept overhang' if keep_overhang else 'overlap shared by the applied transforms'})"
    )

    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in moving_position_dirpaths],
        shape=(len(time_indices), len(output_channel_names)) + cropped_shape,
        chunks=None,
        scale=(1, 1) + tuple(output_voxel_size),
        channel_names=output_channel_names,
        dtype=np.float32,
        version=resolve_ome_zarr_version(moving_position_dirpaths[0], output_ome_zarr_version),
    )

    # Wall time scales with the volumes one position writes (0.5 min each, deskew's
    # margin); an sbatch file's time overrides it.
    time_minutes, num_cpus, gb_ram = estimate_resources(
        shape=(len(time_indices), len(output_channel_names), *moving_shape),
        ram_multiplier=5,
        time_multiplier=0.5,
    )
    echo_resources(num_cpus, num_cpus * gb_ram, time_minutes)
    if init_only:
        click.echo(f"Initialized {output_dirpath} ({len(moving_position_dirpaths)} positions)")
        return
    slurm_out_path = output_dirpath.parent / "slurm_output"
    slurm_args = {
        "slurm_job_name": "apply_transform",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to N positions at a time
        "slurm_time": time_minutes,
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))
    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    run_metadata = {
        "biahub-apply-transform": {
            "transforms": settings.model_dump(),
            "time_indices": time_indices,
            "keep_overhang": keep_overhang,
            "interpolation": interpolation,
        }
    }
    output_time_indices = list(range(len(time_indices)))
    # The units a resumed attempt may skip must come from this same apply.
    resume_token = hashlib.sha256(
        json.dumps(
            {
                "transforms": settings.model_dump(mode="json"),
                "time_indices": time_indices,
                "channels": output_channel_names,
                "keep_overhang": keep_overhang,
                "interpolation": interpolation,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    jobs, labels = [], []
    with submitit.helpers.clean_env(), executor.batch():
        for key, moving_path in zip(position_keys, moving_position_dirpaths, strict=True):
            output_position_path = output_dirpath / Path(*moving_path.parts[-3:])
            extra_metadata = {
                "biahub-apply-transform": {
                    **run_metadata["biahub-apply-transform"],
                    "timepoints_not_accepted": not_accepted.get(key, {}),
                }
            }
            # Jobs look a matrix up by input timepoint: a T-long list, filled for the selected t.
            matrices_for_jobs = [None] * T
            for t, matrix in inverses[key].items():
                matrices_for_jobs[t] = matrix.tolist()
            for channel_name in transformed:
                jobs.append(
                    executor.submit(
                        process_single_position,
                        _apply_transform_czyx,
                        input_position_path=moving_path,
                        output_position_path=output_position_path,
                        input_time_indices=time_indices,
                        output_time_indices=output_time_indices,
                        input_channel_indices=[[moving_channel_names.index(channel_name)]],
                        output_channel_indices=[[output_channel_names.index(channel_name)]],
                        num_workers=int(slurm_args["slurm_cpus_per_task"]),
                        matrices=matrices_for_jobs,
                        output_shape_zyx=reference_shape,
                        crop_output_slicing=list(crop),
                        interpolation=interpolation,
                        extra_metadata=extra_metadata,
                        resume=resume,
                        resume_token=resume_token,
                    )
                )
                labels.append(Path(f"{moving_path.parts[-3:]} {channel_name}"))
            if copied and reference_position_dirpaths:
                reference_path = reference_for[key]
                for channel_name in copied:
                    jobs.append(
                        executor.submit(
                            process_single_position,
                            copy_n_paste_czyx,
                            input_position_path=reference_path,
                            output_position_path=output_position_path,
                            input_time_indices=time_indices,
                            output_time_indices=output_time_indices,
                            input_channel_indices=[
                                [reference_channel_names.index(channel_name)]
                            ],
                            output_channel_indices=[
                                [output_channel_names.index(channel_name)]
                            ],
                            num_workers=int(slurm_args["slurm_cpus_per_task"]),
                            czyx_slicing_params=list(crop),
                            resume=resume,
                            resume_token=resume_token,
                        )
                    )
                    labels.append(Path(f"{reference_path.parts[-3:]} {channel_name} (copy)"))

    slurm_out_path.mkdir(exist_ok=True)
    (slurm_out_path / "submitit_jobs_ids.log").write_text(
        "\n".join(str(job.job_id) for job in jobs)
    )
    if resolved_cluster == "debug":
        for job in jobs:
            job.wait()
        return
    if monitor:
        monitor_jobs(jobs, labels)


@click.command("apply-transform")
@moving_position_dirpaths()
@reference_position_dirpaths(required=False)
@config_filepath()
@output_dirpath()
@click.option(
    "--time-indices",
    default="all",
    show_default=True,
    help="Timepoints to write: 'all', one index, or a comma-separated list (e.g. 0,82,239).",
)
@click.option(
    "--keep-overhang",
    is_flag=True,
    default=False,
    help="Keep the full reference grid instead of cropping to the overlap shared by the applied transforms.",
)
@click.option(
    "--interpolation",
    default="linear",
    show_default=True,
    type=click.Choice(["linear", "nearest"]),
    help="Resampling interpolation.",
)
@click.option(
    "--ome-zarr-version",
    default=None,
    type=click.Choice(["0.4", "0.5"]),
    help="OME-Zarr version of the output store (default: the moving store's).",
)
@click.option(
    "--channels",
    multiple=True,
    help="Moving-store channel to transform (repeat for several); default: every channel, "
    "but one the reference store also has is copied unless it is in the file's moving_channels.",
)
@sbatch_filepath()
@cluster()
@monitor(short=False)
@init_only()
@resume()
def apply_transform_cli(
    moving_position_dirpaths: list[Path],
    reference_position_dirpaths: list[Path] | None,
    config_filepath: Path,
    output_dirpath: Path,
    time_indices: str,
    keep_overhang: bool,
    interpolation: str,
    ome_zarr_version: str | None,
    sbatch_filepath: str | None,
    cluster: str,
    monitor: bool,
    channels: tuple[str, ...],
    init_only: bool,
    resume: bool,
) -> None:
    """Apply a transform series to positions -- one matrix for all timepoints or one per timepoint.

    Takes the `TransformSettings` file written by `estimate-transform`. How it is applied
    is decided here, not in the file: which timepoints, the canvas (overlap shared by the
    applied transforms, or `--keep-overhang` for the full reference grid), the
    interpolation, the output OME-Zarr version, and which moving channels are
    transformed (`--channels`; default all of them, except that a channel the reference
    store also has is copied unless the file names it in `moving_channels`).

    \b
    Registration (moving channels onto the reference store's grid and channels):
    >>> biahub apply-transform -m moving.zarr/*/*/* -r reference.zarr/*/*/* \\
        -c transforms.yml -o registered.zarr

    \b
    Stabilization (every channel of a store onto its own grid, per-timepoint matrices):
    >>> biahub apply-transform -m data.zarr/*/*/* -c transforms.yml -o stabilized.zarr

    \b
    As the Nextflow module runs it: create the plate once, then one task per position:
    >>> biahub apply-transform --init -m data.zarr/*/*/* -c transforms.yml -o out.zarr
    >>> biahub apply-transform --cluster debug --resume -m data.zarr/A/1/0 -c ... -o out.zarr
    """  # noqa: D301
    apply_transform(
        moving_position_dirpaths=moving_position_dirpaths,
        reference_position_dirpaths=reference_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        time_indices=parse_time_indices(time_indices),
        keep_overhang=keep_overhang,
        interpolation=interpolation,
        output_ome_zarr_version=ome_zarr_version,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        channels=list(channels) or None,
        monitor=monitor,
        init_only=init_only,
        resume=resume,
    )


if __name__ == "__main__":
    apply_transform_cli()
