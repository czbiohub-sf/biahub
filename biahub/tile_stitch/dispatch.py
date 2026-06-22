"""Traversal order + memory gate for the bounded recon-dispatch path.

The live engine needs: ``morton_output_order`` (Morton sweep over a plan's output
tiles) and ``peak_resident_tiles`` (the plan-time budget — exact peak live input
tiles for a batch=1 sweep).
"""

from __future__ import annotations

import heapq
import logging

from collections.abc import Hashable, Iterable, Sequence

logger = logging.getLogger(__name__)


def morton_order(cells: Iterable[Sequence[int]]) -> list:
    """k-D Z-order (Morton) sort of integer-coordinate cells."""
    cells = list(cells)
    if not cells:
        return cells
    ndim = len(cells[0])
    bits = max(1, max(max(c) for c in cells).bit_length())

    def code(c: Sequence[int]) -> int:
        r = 0
        for i in range(bits):
            for d in range(ndim):
                r |= ((c[d] >> i) & 1) << (i * ndim + d)
        return r

    return sorted(cells, key=code)


def morton_output_order(plan) -> list[int]:
    """Z-order a plan's output tiles by their start coords (the recon sweep order).

    Duck-typed on ``plan``: needs ``output_to_inputs``, ``output_tiles`` (each with
    ``.tile_id`` + ``.slices: dict[str, slice]``), and ``tile_dims``. Output tiles
    are disjoint, so start coords are unique → the coord↔id map is bijective.
    """
    dims = tuple(plan.tile_dims)
    coord_of = {
        ot.tile_id: tuple(ot.slices[d].start for d in dims) for ot in plan.output_tiles
    }
    ids = [oid for oid in plan.output_to_inputs if oid in coord_of]
    missing = [oid for oid in plan.output_to_inputs if oid not in coord_of]
    if missing:
        # An output with no coord would be silently dropped from the sweep and
        # never stitched — the plan should carry a coord for every output.
        logger.warning(
            "morton order: %d/%d output tiles have no coord, excluded from the sweep "
            "(first few: %s)",
            len(missing),
            len(plan.output_to_inputs),
            missing[:5],
        )
    coord_to_id = {coord_of[oid]: oid for oid in ids}
    return [coord_to_id[c] for c in morton_order(coord_of[oid] for oid in ids)]


def peak_resident_tiles(out_to_in: dict[Hashable, Sequence], order: Sequence) -> int:
    """Exact peak resident input tiles for a batch=1 windowed sweep over ``order``.

    Interval-overlap (register-allocation liveness) of each input tile's
    ``[first use, last use]`` span — the minimum budget that can stitch without
    deadlock, used as the recon-dispatch gate's auto floor.
    """
    pos = {o: i for i, o in enumerate(order)}
    uses: dict = {}
    for o in order:
        for t in out_to_in[o]:
            uses.setdefault(t, []).append(pos[o])
    cur = peak = 0
    ends: list[int] = []
    for s, e in sorted((min(ps), max(ps)) for ps in uses.values()):
        while ends and ends[0] < s:
            heapq.heappop(ends)
            cur -= 1
        heapq.heappush(ends, e)
        cur += 1
        peak = max(peak, cur)
    return peak
