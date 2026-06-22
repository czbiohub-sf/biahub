"""Traversal order + memory gate for the bounded recon-dispatch path.

The two pure functions the live engine needs: ``morton_order`` (locality sweep,
keeps a cell's contributors co-resident) and ``peak_resident_tiles`` (the
plan-time budget — exact peak live input tiles for a batch=1 sweep).
"""

from __future__ import annotations

import heapq

from collections.abc import Hashable, Iterable, Sequence


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
