"""CPU unit tests for the P3b driver-side pieces in ``monarch.backend`` — the
bounded recon-dispatch gate and the Morton schedule/budget. No Monarch, no GPU:
``backend``'s Monarch imports are function-local, so the module imports clean and
``_ResidentGate`` (pure asyncio) + ``_tile_cache_schedule`` (pure logic over a
duck-typed plan) test in isolation."""

from __future__ import annotations

import asyncio

from dataclasses import dataclass

from biahub.tile_stitch.config import MonarchConfig, TileCacheOrder
from biahub.tile_stitch.monarch.backend import _ResidentGate, _tile_cache_schedule

TILE, OVERLAP = 8, 2
STRIDE = TILE - OVERLAP
DIMS = ("z", "y", "x")


@dataclass
class _Tile:
    tile_id: int
    slices: dict


@dataclass
class _Plan:
    output_to_inputs: dict
    output_tiles: list
    tile_dims: tuple
    input_order: list


def _fake_plan(n: int) -> _Plan:
    def gid(z, y, x):
        return z * n * n + y * n + x

    def ov(a, b):
        return a * STRIDE < b * STRIDE + STRIDE and a * STRIDE + TILE > b * STRIDE

    out_to_in, out_tiles, all_in = {}, [], set()
    for oz in range(n):
        for oy in range(n):
            for ox in range(n):
                ins = [
                    gid(iz, iy, ix)
                    for iz in range(n) if ov(iz, oz)
                    for iy in range(n) if ov(iy, oy)
                    for ix in range(n) if ov(ix, ox)
                ]
                out_to_in[gid(oz, oy, ox)] = ins
                all_in.update(ins)
                out_tiles.append(_Tile(gid(oz, oy, ox), {
                    "z": slice(oz * STRIDE, oz * STRIDE + STRIDE),
                    "y": slice(oy * STRIDE, oy * STRIDE + STRIDE),
                    "x": slice(ox * STRIDE, ox * STRIDE + STRIDE),
                }))
    return _Plan(out_to_in, out_tiles, DIMS, sorted(all_in))


def test_resident_gate_bounds_concurrency_no_deadlock():
    """Resident set never exceeds the budget; units of n acquire atomically; the
    whole batch of work-units completes (no partial-hold deadlock)."""
    async def main():
        budget = 3
        gate = _ResidentGate(budget)
        cur = peak = 0

        async def unit(n):
            nonlocal cur, peak
            await gate.acquire(n)
            cur += n
            peak = max(peak, cur)
            await asyncio.sleep(0)  # yield so units interleave
            for _ in range(n):  # Stage B frees tiles one at a time
                cur -= 1
                await gate.release(1)

        await asyncio.gather(*[unit(2) for _ in range(12)])
        return peak

    peak = asyncio.run(main())
    assert peak <= 3


def test_tile_cache_schedule_morton_valid_and_budget_safe():
    plan = _fake_plan(6)
    max_fanin = max(len(v) for v in plan.output_to_inputs.values())
    cfg = MonarchConfig(tile_cache=True, tile_cache_order=TileCacheOrder.MORTON)
    in_order, budget = _tile_cache_schedule(plan, recon_batch=4, cfg=cfg)
    assert sorted(in_order) == sorted(plan.input_order)  # valid permutation of inputs
    assert budget >= max_fanin and budget >= 4  # deadlock-safe + >= recon_batch
    assert in_order != plan.input_order  # Morton actually reordered


def test_tile_cache_schedule_plan_order_is_identity():
    plan = _fake_plan(4)
    cfg = MonarchConfig(tile_cache=True, tile_cache_order=TileCacheOrder.PLAN)
    in_order, budget = _tile_cache_schedule(plan, recon_batch=1, cfg=cfg)
    assert in_order == plan.input_order  # PLAN = engine's existing order, unchanged
    assert budget >= 1


def test_explicit_budget_below_floor_is_raised():
    """A resident_budget below the mandatory deadlock-safe floor (auto_peak +
    recon_batch*n_gpus, the stranded-slots fix) is raised, not honored."""
    plan = _fake_plan(4)
    cfg = MonarchConfig(tile_cache=True, resident_budget=1)  # far below the floor
    max_fanin = max(len(v) for v in plan.output_to_inputs.values())
    _, budget = _tile_cache_schedule(plan, recon_batch=2, cfg=cfg)
    assert budget >= max_fanin and budget >= 2   # deadlock-safe minimums
    assert budget > cfg.resident_budget          # the sub-floor request was floored up


def test_explicit_budget_above_floor_is_honored():
    """A resident_budget above the safe floor is honored as-is (the floor may only
    RAISE the budget, never lower a larger explicit request)."""
    plan = _fake_plan(4)
    big = 10_000
    cfg = MonarchConfig(tile_cache=True, resident_budget=big)
    _, budget = _tile_cache_schedule(plan, recon_batch=2, cfg=cfg)
    assert budget == big
