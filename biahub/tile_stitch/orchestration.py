"""Preparation, output creation, and execution for distributed tile stitching."""

import json
import logging
import os
import time

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biahub.settings import TileStitchReconSettings
from biahub.tile_stitch.plan import (
    StitchProgram,
    StitchWorkUnit,
    program_from_engine_plan,
    write_program,
)

logger = logging.getLogger(__name__)

_CHUNK_BYTES_CAP = 64_000_000


class PreparationError(ValueError):
    """Input metadata cannot form one static stitch program."""


@dataclass(frozen=True, slots=True)
class PreparedWorkUnit:
    """One position label and its immutable actor binding."""

    position: Path
    work: StitchWorkUnit


@dataclass(frozen=True, slots=True)
class PreparedStitchRun:
    """Validated static program, work units, and output metadata."""

    config: TileStitchReconSettings
    final_output: Path
    run_dir: Path
    program: StitchProgram
    work_units: tuple[PreparedWorkUnit, ...]
    channel: str
    output_channel_name: str
    full_shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    scale: tuple[float, ...]
    plate_mode: bool
    position_keys: tuple[tuple[str, ...], ...]


def _is_plate_position(path: Path) -> bool:
    """Return whether a path belongs to an HCS plate hierarchy."""
    from iohub.ngff import Plate, open_ome_zarr

    try:
        with open_ome_zarr(str(path.parents[2]), mode="r") as ancestor:
            return isinstance(ancestor, Plate)
    except Exception:
        return False


def _chunk_shape(tile_spatial: tuple[int, ...]) -> tuple[int, ...]:
    """Keep float32 TCZYX chunks near 64 MB for parallel zarrs writes."""
    import math

    lead, *rest = tile_spatial
    rest_bytes = math.prod(rest) * 4
    cap = max(1, _CHUNK_BYTES_CAP // max(1, rest_bytes))
    return (1, 1, min(lead, cap), *rest)


def prepare_stitch_run(
    positions: Sequence[Path],
    config: TileStitchReconSettings,
    output: Path,
) -> PreparedStitchRun:
    """Validate inputs and build one static program with bound work units.

    Parameters
    ----------
    positions : sequence[pathlib.Path]
        Input OME-Zarr positions.
    config : TileStitchReconSettings
        Validated tile-stitch configuration.
    output : pathlib.Path
        Requested output path.

    Returns
    -------
    PreparedStitchRun
        Complete execution description without a created output store.

    Raises
    ------
    PreparationError
        If inputs are empty, incompatible, or mix unsupported layouts.
    """
    from iohub.ngff import open_ome_zarr
    from waveorder.api.tile_stitch import build_plan

    from biahub.cli.utils import get_output_paths
    from biahub.tile_stitch.cli import _resolve_time_indices

    position_paths = tuple(Path(path) for path in positions)
    if not position_paths:
        raise PreparationError("at least one input position is required")

    channel = config.tile_stitch.recon.input_channel_names[0]
    with open_ome_zarr(str(position_paths[0]), mode="r") as first:
        input_shape = tuple(first.data.shape)
        try:
            channel_idx = first.channel_names.index(channel)
        except ValueError as exc:
            raise PreparationError(
                f"input channel {channel!r} is absent from {position_paths[0]}"
            ) from exc
        input_scale = tuple(first.scale)

    for path in position_paths[1:]:
        with open_ome_zarr(str(path), mode="r") as position:
            shape = tuple(position.data.shape)
            if shape != input_shape:
                raise PreparationError(
                    f"all positions must share TCZYX shape; {path} is {shape} "
                    f"!= {input_shape} ({position_paths[0]})"
                )

    timepoints = tuple(_resolve_time_indices(config.time_indices, input_shape[0]))
    if not timepoints:
        raise PreparationError("time_indices must select at least one timepoint")
    plate_mode = all(_is_plate_position(path) for path in position_paths)
    if not plate_mode and len(position_paths) > 1:
        raise PreparationError(
            "multiple standalone FOVs are not supported; pass HCS plate positions "
            "or a single FOV"
        )

    with open_ome_zarr(str(position_paths[0]), mode="r") as first:
        source = first.to_xarray().isel(t=timepoints[0]).sel(c=[channel])
    engine_plan = build_plan(source, config.tile_stitch, batch_size=None)
    program = program_from_engine_plan(
        engine_plan,
        settings=config.tile_stitch,
        monarch=config.monarch,
    )

    final_output = output if output.suffix == ".zarr" else output.with_suffix(".zarr")
    run_dir = final_output.parent
    spatial_shape = tuple(engine_plan.full_shape[dim] for dim in engine_plan.tile_dims)
    tile_spatial = tuple(
        config.tile_stitch.tile.tile_size[dim] for dim in engine_plan.tile_dims
    )
    full_shape = (max(timepoints) + 1, 1, *spatial_shape)
    scale = input_scale if len(input_scale) == 5 else (1.0, 1.0, *input_scale[-3:])
    output_targets = (
        tuple(Path(path) for path in get_output_paths(list(position_paths), final_output))
        if plate_mode
        else (final_output,)
    )
    work_units = tuple(
        PreparedWorkUnit(
            position=position,
            work=StitchWorkUnit(
                input_path=str(position),
                output_path=str(target),
                channel_idx=channel_idx,
                timepoint=timepoint,
            ),
        )
        for position, target in zip(position_paths, output_targets, strict=True)
        for timepoint in timepoints
    )
    return PreparedStitchRun(
        config=config,
        final_output=final_output,
        run_dir=run_dir,
        program=program,
        work_units=work_units,
        channel=channel,
        output_channel_name=f"{channel}_recon",
        full_shape=full_shape,
        chunk_shape=_chunk_shape(tile_spatial),
        scale=scale,
        plate_mode=plate_mode,
        position_keys=tuple(tuple(path.parts[-3:]) for path in position_paths),
    )


def create_stitch_output(prepared: PreparedStitchRun) -> None:
    """Create the validated run's destination OME-Zarr hierarchy."""
    import numpy as np

    from iohub.ngff import open_ome_zarr
    from iohub.ngff.models import TransformationMeta

    prepared.run_dir.mkdir(parents=True, exist_ok=True)
    if prepared.plate_mode:
        from iohub.ngff.utils import create_empty_plate

        create_empty_plate(
            store_path=prepared.final_output,
            position_keys=list(prepared.position_keys),
            channel_names=[prepared.output_channel_name],
            shape=prepared.full_shape,
            chunks=prepared.chunk_shape,
            scale=prepared.scale,
            dtype=np.float32,
        )
        return

    with open_ome_zarr(
        prepared.final_output,
        layout="fov",
        mode="w",
        channel_names=[prepared.output_channel_name],
    ) as output:
        output.create_zeros(
            "0",
            shape=prepared.full_shape,
            dtype=np.float32,
            chunks=prepared.chunk_shape,
            transform=[TransformationMeta(type="scale", scale=list(prepared.scale))],
        )


def execute_stitch_run(prepared: PreparedStitchRun) -> dict[str, Any]:
    """Execute all bound work units and publish wall-time metrics."""
    from biahub.tile_stitch.monarch.backend import MonarchBackend

    program_path = write_program(prepared.program, prepared.run_dir)
    config = prepared.program.monarch
    gpus_per_node = (
        config.gpus_per_node or int(os.environ.get("SLURM_GPUS_ON_NODE") or 0) or None
    )
    units: list[dict[str, Any]] = []
    started = time.monotonic()
    first = prepared.work_units[0]
    with MonarchBackend(
        gpus_per_node=gpus_per_node,
        window_per_actor=config.window_per_actor,
        device=config.device.value,
    ) as backend:
        backend.setup(program_path, first.work)
        for index, item in enumerate(prepared.work_units):
            if index:
                backend.bind_work_unit(item.work)
            unit_started = time.monotonic()
            summary = backend.drive_tp(prepared.program)
            wall = time.monotonic() - unit_started
            logger.info(
                "%s tp%d: wall=%.1fs (A=%.1fs), tiles=%d/%d",
                item.position.name,
                item.work.timepoint,
                wall,
                summary["stage_a_s"],
                summary["n_outputs"],
                len(prepared.program.output_tiles),
            )
            try:
                actor_stats = backend.collect_recon_stats()
            except Exception as exc:
                logger.warning("recon_stats collection failed: %s", exc)
                actor_stats = []
            units.append(
                {
                    "position": item.position.name,
                    "timepoint": item.work.timepoint,
                    "wall_s": wall,
                    "stage_a_s": summary["stage_a_s"],
                    "n_outputs": summary["n_outputs"],
                    "dispatch": summary["dispatch"],
                    "actors": actor_stats,
                }
            )

    result = {"total_s": time.monotonic() - started, "units": units}
    (prepared.run_dir / "walls.json").write_text(json.dumps(result, indent=2))
    return result
