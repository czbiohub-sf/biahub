"""Run plan — pickled to disk, cached per-worker.

Composes the waveorder TileStitchPlan (recon-side: tiles, output_tiles,
output_to_inputs, input_batches, output_to_batches) with biahub-side
I/O metadata (input/output paths, channel, timepoint, settings).

The driver writes one ``plan.pkl`` to ``run_dir`` at start; workers
``_load_plan(path)`` and cache by path so subsequent tasks within the
same worker process don't re-deserialize.
"""

import pickle

from dataclasses import dataclass
from pathlib import Path

from waveorder.api.tile_stitch import TileStitchSettings
from waveorder.tile_stitch._engine import TileStitchPlan
from waveorder.tile_stitch.partition import InputTile, OutputTile

from biahub.tile_stitch.config import MonarchConfig


@dataclass
class RunPlan:
    """Everything a worker needs to execute one Stage A or Stage B task."""

    # I/O metadata
    input_path: str
    output_path: str
    channel: str
    channel_idx: int
    timepoint: int

    # Engine config (settings carried verbatim for worker-side TF + recon)
    settings: TileStitchSettings

    # Recon plan (from waveorder build_plan)
    input_tiles: list[InputTile]
    output_tiles: list[OutputTile]
    output_to_inputs: dict[int, list[int]]
    input_order: list[int]
    tile_dims: tuple[str, ...]
    full_shape: dict[str, int]
    input_batches: list[list[int]] | None
    output_to_batches: dict[int, list[int]] | None

    # Output zarr leading dims (T, C) — written-by driver, read-by stitch.
    leading_shape: tuple[int, ...] = (1, 1)

    # C slot this channel writes in a (possibly multi-channel) shared output
    # zarr. 0 for single-channel runs; the stitch writes ``[t, c_idx, zyx]``.
    output_channel_index: int = 0

    # Monarch engine knobs, carried so each actor reads the same config across
    # ``setup`` and every per-TP ``swap_to``. Optional: the actor falls back to
    # ``MonarchConfig()`` defaults when this is ``None``.
    monarch: MonarchConfig | None = None


def write_plan(plan: RunPlan, run_dir: str | Path, filename: str = "plan.pkl") -> str:
    """Pickle plan to ``<run_dir>/<filename>``. Returns the path."""
    p = Path(run_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(plan, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(p)


# Per-worker process cache. Keyed by path so multiple plans coexist.
_PLAN_CACHE: dict[str, RunPlan] = {}


def load_plan(plan_path: str) -> RunPlan:
    if plan_path not in _PLAN_CACHE:
        with open(plan_path, "rb") as f:
            _PLAN_CACHE[plan_path] = pickle.load(f)
    return _PLAN_CACHE[plan_path]


def from_engine_plan(
    engine_plan: TileStitchPlan,
    *,
    settings: TileStitchSettings,
    input_path: str,
    output_path: str,
    channel: str,
    channel_idx: int,
    timepoint: int,
    leading_shape: tuple[int, ...] = (1, 1),
    output_channel_index: int = 0,
    monarch: MonarchConfig | None = None,
) -> RunPlan:
    """Compose a RunPlan from a waveorder TileStitchPlan + biahub I/O config."""
    return RunPlan(
        input_path=input_path,
        output_path=output_path,
        channel=channel,
        channel_idx=channel_idx,
        timepoint=timepoint,
        settings=settings,
        input_tiles=engine_plan.input_tiles,
        output_tiles=engine_plan.output_tiles,
        output_to_inputs=engine_plan.output_to_inputs,
        input_order=engine_plan.input_order,
        tile_dims=engine_plan.tile_dims,
        full_shape=engine_plan.full_shape,
        input_batches=engine_plan.input_batches,
        output_to_batches=engine_plan.output_to_batches,
        leading_shape=leading_shape,
        output_channel_index=output_channel_index,
        monarch=monarch,
    )
