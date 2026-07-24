"""CPU contracts for the validated dispatch preparation boundary and resident gate."""

from __future__ import annotations

import asyncio

from dataclasses import dataclass

import pytest

from biahub.settings import MonarchConfig
from biahub.tile_stitch.monarch.backend import _prepare_dispatch
from biahub.tile_stitch.monarch.execution import ResidentGate

TILE, OVERLAP = 8, 2
STRIDE = TILE - OVERLAP
DIMS = ("z", "y", "x")


@dataclass
class _Tile:
    tile_id: int
    slices: dict


@dataclass
class _InputTile:
    tile_id: int
    shape: tuple[int, ...] = (TILE, TILE, TILE)


@dataclass
class _Plan:
    output_to_inputs: dict
    output_tiles: list
    tile_dims: tuple
    input_order: list
    input_tiles: list


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
                    for iz in range(n)
                    if ov(iz, oz)
                    for iy in range(n)
                    if ov(iy, oy)
                    for ix in range(n)
                    if ov(ix, ox)
                ]
                out_to_in[gid(oz, oy, ox)] = ins
                all_in.update(ins)
                out_tiles.append(
                    _Tile(
                        gid(oz, oy, ox),
                        {
                            "z": slice(oz * STRIDE, oz * STRIDE + STRIDE),
                            "y": slice(oy * STRIDE, oy * STRIDE + STRIDE),
                            "x": slice(ox * STRIDE, ox * STRIDE + STRIDE),
                        },
                    )
                )
    input_order = sorted(all_in)
    return _Plan(
        out_to_in,
        out_tiles,
        DIMS,
        input_order,
        [_InputTile(tile_id) for tile_id in input_order],
    )


def test_resident_gate_bounds_concurrency_no_deadlock():
    """Resident set never exceeds the budget; units of n acquire atomically; the
    whole batch of work-units completes (no partial-hold deadlock)."""

    async def main():
        budget = 3
        gate = ResidentGate(budget)
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


def test_dispatch_schedule_morton_valid_and_budget_safe():
    plan = _fake_plan(6)
    max_fanin = max(len(v) for v in plan.output_to_inputs.values())
    cfg = MonarchConfig(resident_budget="auto")
    execution = _prepare_dispatch(plan, recon_batch=4, cfg=cfg, n_actors=8)
    assert sorted(execution.input_order) == sorted(plan.input_order)
    assert execution.resident_budget is not None
    assert execution.resident_budget >= max_fanin
    assert execution.input_order != tuple(plan.input_order)
    assert execution.metrics["scheduler"] == "morton"
    assert execution.metrics["n_actors"] == 8
    assert (
        execution.metrics["resident_safe_floor"]
        == execution.metrics["peak_resident_tiles"] + 32
    )


def test_explicit_budget_below_floor_is_raised():
    """A resident_budget below the mandatory deadlock-safe floor (auto_peak +
    recon_batch*n_gpus, the stranded-slots fix) is raised, not honored."""
    plan = _fake_plan(4)
    cfg = MonarchConfig(resident_budget=1)  # far below the floor
    max_fanin = max(len(v) for v in plan.output_to_inputs.values())
    execution = _prepare_dispatch(plan, recon_batch=2, cfg=cfg, n_actors=1)
    assert execution.resident_budget is not None
    assert execution.resident_budget >= max_fanin
    assert execution.resident_budget > cfg.resident_budget


def test_explicit_budget_above_floor_is_honored():
    """A resident_budget above the safe floor is honored as-is (the floor may only
    RAISE the budget, never lower a larger explicit request)."""
    plan = _fake_plan(4)
    big = 10_000
    cfg = MonarchConfig(resident_budget=big)
    execution = _prepare_dispatch(plan, recon_batch=2, cfg=cfg, n_actors=1)
    assert execution.resident_budget == big


def test_unbounded_dispatch_still_validates_and_uses_plan_strategy():
    plan = _fake_plan(3)
    execution = _prepare_dispatch(
        plan,
        recon_batch=2,
        cfg=MonarchConfig(),
        n_actors=2,
    )
    assert execution.input_order == tuple(plan.input_order)
    assert execution.resident_budget is None
    assert execution.metrics["scheduler"] == "plan"
    assert execution.metrics["bounded"] == 0
    with pytest.raises(TypeError):
        execution.metrics["mutated"] = 1
