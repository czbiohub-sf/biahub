"""Apply a transform series to positions: one matrix for every timepoint or one per timepoint.

Replaces `register` (one transform, source onto a target store) and `stabilize` (one
transform per timepoint, a store onto itself). The output canvas is decided once, before
the output store is allocated: the largest box inside the overlap of the warped source
and the reference, intersected over every timepoint's transform -- or, with
`keep_overhang`, the reference grid as is.
"""

from __future__ import annotations

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
    OptionEatAll,
    _validate_and_process_paths,
    cluster,
    config_filepath,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
    source_position_dirpaths,
)
from biahub.registration.utils import (
    apply_affine_transform,
    convert_transform_to_ants,
    rescale_voxel_size,
)
from biahub.settings import TransformSettings, load_transform_settings
from biahub.utils.array_ops import copy_n_paste_czyx
from biahub.utils.cluster import estimate_resources, get_submitit_cluster
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
    pull_matrix: np.ndarray,
    downsample: int = 4,
) -> np.ndarray:
    """Target-grid voxels the warped source covers, on a `downsample`-times coarser grid.

    The warp of an all-ones source is the same geometry at any resolution, so a
    240-timepoint series stays cheap.
    """
    f = max(1, int(downsample))
    scale = np.diag([1.0 / f, 1.0 / f, 1.0 / f, 1.0])
    coarse = scale @ np.asarray(pull_matrix, dtype=float) @ np.linalg.inv(scale)
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
    pull_matrix: np.ndarray,
    downsample: int = 4,
) -> Slices:
    """Largest box inside the overlap of the warped source and the target grid."""
    mask = overlap_mask(source_shape_zyx, target_shape_zyx, pull_matrix, downsample)
    return _mask_box(mask, target_shape_zyx, downsample)


def canvas(
    source_shape_zyx: tuple[int, int, int],
    target_shape_zyx: tuple[int, int, int],
    pull_matrices: list[np.ndarray],
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
        for m in pull_matrices
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


def apply_transform(
    source_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    target_position_dirpaths: list[Path] | None = None,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
) -> None:
    """Apply a `TransformSettings` series (or a legacy register / stabilize config).

    With target positions: the output lives on the target grid and holds every target
    channel copied plus the config's `source_channels` transformed from the source store
    (registration). Without: every channel of the source store is transformed onto its
    own grid (stabilization). One matrix is applied to every timepoint; a list is applied
    per timepoint.
    """
    output_dirpath = Path(output_dirpath)
    settings: TransformSettings = load_transform_settings(config_filepath)
    pull_matrices = [np.asarray(m, dtype=float) for m in settings.as_direction("pull")]

    with open_ome_zarr(source_position_dirpaths[0], mode="r") as source:
        T, _C, *source_shape = source.data.shape
        source_channel_names = list(source.channel_names)
        source_voxel_size = tuple(source.scale[-3:])
    time_indices = _resolve_time_indices(settings.time_indices, T)
    if len(pull_matrices) not in (1, T) and len(pull_matrices) != len(time_indices):
        raise click.UsageError(
            f"{len(pull_matrices)} matrices for {T} timepoints ({len(time_indices)} selected): "
            "give one matrix, or one per timepoint"
        )

    if target_position_dirpaths:
        with open_ome_zarr(target_position_dirpaths[0], mode="r") as target:
            target_shape = tuple(target.data.shape[-3:])
            target_channel_names = list(target.channel_names)
            target_voxel_size = list(target.scale)
        transformed = [c for c in settings.source_channels if c in source_channel_names]
        missing = set(settings.source_channels) - set(transformed)
        if missing:
            raise click.UsageError(
                f"source channels not in the source store: {sorted(missing)}"
            )
        copied = target_channel_names
        output_channel_names = copied + [c for c in transformed if c not in copied]
        output_voxel_size = tuple(target_voxel_size[-3:])
    else:
        target_shape = tuple(source_shape)
        transformed, copied = source_channel_names, []
        output_channel_names = source_channel_names
        output_voxel_size = (
            tuple(settings.voxel_size[-3:])
            if settings.voxel_size
            else tuple(rescale_voxel_size(pull_matrices[0][:3, :3], source_voxel_size))
        )

    crop = canvas(tuple(source_shape), target_shape, pull_matrices, settings.keep_overhang)
    cropped_shape = tuple(s.stop - s.start for s in crop)
    click.echo(
        f"Output canvas {cropped_shape} (target grid {target_shape}, "
        f"{'kept overhang' if settings.keep_overhang else 'overlap intersected over ' + str(len(pull_matrices)) + ' transform(s)'})"
    )

    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in source_position_dirpaths],
        shape=(len(time_indices), len(output_channel_names)) + cropped_shape,
        chunks=None,
        scale=(1, 1) + tuple(output_voxel_size),
        channel_names=output_channel_names,
        dtype=np.float32,
        version=resolve_ome_zarr_version(
            source_position_dirpaths[0], settings.output_ome_zarr_version
        ),
    )

    _, num_cpus, gb_ram = estimate_resources(
        shape=(T, len(output_channel_names), *source_shape), ram_multiplier=5
    )
    slurm_out_path = output_dirpath.parent / "slurm_output"
    slurm_args = {
        "slurm_job_name": "apply_transform",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to N positions at a time
        "slurm_time": 60,
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))
    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    extra_metadata = {"biahub-apply-transform": settings.model_dump()}
    output_time_indices = list(range(len(time_indices)))
    # Jobs index the matrices by the input timepoint, so hand them a T-long list whether
    # the config gave one matrix per timepoint or one per selected timepoint.
    if len(pull_matrices) == 1:
        matrices_for_jobs = [pull_matrices[0].tolist()]
    else:
        matrices_for_jobs = [None] * T
        for i, t in enumerate(time_indices):
            matrices_for_jobs[t] = pull_matrices[t if len(pull_matrices) == T else i].tolist()
    jobs, labels = [], []
    with submitit.helpers.clean_env(), executor.batch():
        for index, source_path in enumerate(source_position_dirpaths):
            output_position_path = output_dirpath / Path(*source_path.parts[-3:])
            for channel_name in transformed:
                jobs.append(
                    executor.submit(
                        process_single_position,
                        _apply_transform_czyx,
                        input_position_path=source_path,
                        output_position_path=output_position_path,
                        input_time_indices=time_indices,
                        output_time_indices=output_time_indices,
                        input_channel_indices=[[source_channel_names.index(channel_name)]],
                        output_channel_indices=[[output_channel_names.index(channel_name)]],
                        num_workers=int(slurm_args["slurm_cpus_per_task"]),
                        matrices=matrices_for_jobs,
                        output_shape_zyx=target_shape,
                        crop_output_slicing=list(crop),
                        interpolation=settings.interpolation,
                        extra_metadata=extra_metadata,
                    )
                )
                labels.append(Path(f"{source_path.parts[-3:]} {channel_name}"))
            if copied and target_position_dirpaths:
                target_path = target_position_dirpaths[
                    min(index, len(target_position_dirpaths) - 1)
                ]
                for channel_name in copied:
                    jobs.append(
                        executor.submit(
                            process_single_position,
                            copy_n_paste_czyx,
                            input_position_path=target_path,
                            output_position_path=output_position_path,
                            input_time_indices=time_indices,
                            output_time_indices=output_time_indices,
                            input_channel_indices=[[target_channel_names.index(channel_name)]],
                            output_channel_indices=[
                                [output_channel_names.index(channel_name)]
                            ],
                            num_workers=int(slurm_args["slurm_cpus_per_task"]),
                            czyx_slicing_params=list(crop),
                        )
                    )
                    labels.append(Path(f"{target_path.parts[-3:]} {channel_name} (copy)"))

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


def _optional_target_position_dirpaths():
    return click.option(
        "--target-position-dirpaths",
        "-t",
        required=False,
        cls=OptionEatAll,
        type=tuple,
        callback=_validate_and_process_paths,
        help='Optional reference positions, e.g. "target.zarr/*/*/*": the output takes their '
        "grid and channels (registration). Omit to transform the source onto its own grid "
        "(stabilization).",
    )


@click.command("apply-transform")
@source_position_dirpaths()
@_optional_target_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
def apply_transform_cli(
    source_position_dirpaths: list[Path],
    target_position_dirpaths: list[Path] | None,
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None,
    cluster: str,
    monitor: bool,
) -> None:
    """Apply a transform series to positions -- one matrix for all timepoints or one per timepoint.

    Takes the `TransformSettings` written by `estimate-transform` (legacy `register` /
    `stabilize` configs are accepted). The output canvas is the overlap of the warped
    source and the reference, intersected over every timepoint's transform, unless the
    config sets `keep_overhang: true`.

    \b
    Registration (source channels onto the target store's grid and channels):
    >>> biahub apply-transform -s source.zarr/*/*/* -t target.zarr/*/*/* \\
        -c transforms.yml -o registered.zarr

    \b
    Stabilization (every channel of a store onto its own grid, per-timepoint matrices):
    >>> biahub apply-transform -s data.zarr/*/*/* -c transforms.yml -o stabilized.zarr
    """  # noqa: D301
    apply_transform(
        source_position_dirpaths=source_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        target_position_dirpaths=target_position_dirpaths or None,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
    )


if __name__ == "__main__":
    apply_transform_cli()
