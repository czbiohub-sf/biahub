"""Adapter: a ``RunPlan`` → ``(out_to_in, traversal order)`` for the recon gate.

Duck-typed — needs only ``plan.output_to_inputs``, ``plan.output_tiles`` (each
with ``.tile_id`` + ``.slices: dict[str, slice]``), and ``plan.tile_dims`` — so it
tests without a full waveorder plan, and the real ``RunPlan`` satisfies it as-is.
Output tiles are non-overlapping, so their start coords are unique → a clean
Morton/raster ordering.
"""

from __future__ import annotations

import logging

from typing import Literal, Protocol

from biahub.tile_stitch.tile_cache import morton_order

logger = logging.getLogger(__name__)


class _Tile(Protocol):
    tile_id: int
    slices: dict[str, slice]


class _PlanLike(Protocol):
    output_to_inputs: dict[int, list[int]]
    output_tiles: list[_Tile]
    tile_dims: tuple[str, ...]


def output_to_inputs_and_order(
    plan: _PlanLike, order: Literal["morton", "raster"] = "morton"
) -> tuple[dict[int, list[int]], list[int]]:
    """Return ``(out_to_in, ordered_output_ids)`` for the recon-dispatch sweep.

    ``order='morton'`` Z-orders output tiles by their start coords (locality →
    tighter resident band); ``'raster'`` is lexicographic by coords. Coords are
    taken in ``plan.tile_dims`` order.
    """
    out_to_in = {oid: list(ins) for oid, ins in plan.output_to_inputs.items()}
    dims = tuple(plan.tile_dims)
    coord_of = {ot.tile_id: tuple(ot.slices[d].start for d in dims) for ot in plan.output_tiles}
    ids = [oid for oid in out_to_in if oid in coord_of]
    if len(ids) != len(out_to_in):
        # An output with no coord would be silently dropped from the traversal and
        # never stitched. The plan should always carry a coord for every output.
        missing = [oid for oid in out_to_in if oid not in coord_of]
        logger.warning(
            "tile_cache order: %d/%d output tiles have no coord and are excluded "
            "from the sweep (first few: %s)",
            len(missing), len(out_to_in), missing[:5],
        )
    if order == "raster":
        return out_to_in, sorted(ids, key=lambda oid: coord_of[oid])
    # Morton: output tiles are disjoint → start coords are unique → bijective.
    coord_to_id = {coord_of[oid]: oid for oid in ids}
    ordered = [coord_to_id[c] for c in morton_order(coord_of[oid] for oid in ids)]
    return out_to_in, ordered
