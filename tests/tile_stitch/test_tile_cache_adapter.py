"""Adapter tests: a fake int-keyed plan (the ``RunPlan`` duck type — ``tile_id`` +
``slices: dict[str, slice]`` + ``output_to_inputs`` + ``tile_dims``) fed through
``output_to_inputs_and_order`` → ``WindowedScheduler``. Pins that the bridge
produces a valid traversal, byte-identical output, and that Morton (from real slice
coords) spills less than raster — without building a full waveorder plan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biahub.tile_stitch.tile_cache import DictSpillStore, TileCache, WindowedScheduler
from biahub.tile_stitch.tile_cache_adapter import output_to_inputs_and_order

TS, TILE, OVERLAP = 4, 8, 2
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


def _fake_plan(n: int):
    def gid(z, y, x):
        return z * n * n + y * n + x

    def ov(a, b):
        return a * STRIDE < b * STRIDE + STRIDE and a * STRIDE + TILE > b * STRIDE

    out_to_in, out_tiles, in_coord = {}, [], {}
    for z in range(n):
        for y in range(n):
            for x in range(n):
                in_coord[gid(z, y, x)] = (z, y, x)
    for oz in range(n):
        for oy in range(n):
            for ox in range(n):
                oid = gid(oz, oy, ox)
                out_to_in[oid] = [
                    gid(iz, iy, ix)
                    for iz in range(n) if ov(iz, oz)
                    for iy in range(n) if ov(iy, oy)
                    for ix in range(n) if ov(ix, ox)
                ]
                out_tiles.append(_Tile(oid, {
                    "z": slice(oz * STRIDE, oz * STRIDE + STRIDE),
                    "y": slice(oy * STRIDE, oy * STRIDE + STRIDE),
                    "x": slice(ox * STRIDE, ox * STRIDE + STRIDE),
                }))
    return _Plan(out_to_in, out_tiles, DIMS), in_coord


def _recon_for(in_coord):
    def recon(in_id):
        z, y, x = in_coord[in_id]
        zz, yy, xx = np.mgrid[0:TS, 0:TS, 0:TS].astype(np.float32)
        return (z * TS + zz) + (y * TS + yy) * 1e3 + (x * TS + xx) * 1e6
    return recon


_W = (np.hanning(TS + 2)[1:-1].astype(np.float32) + 1e-3)
_WK = _W[:, None, None] * _W[None, :, None] * _W[None, None, :]


def _blend(arrs):
    num = np.zeros((TS, TS, TS), np.float32)
    den = np.zeros((TS, TS, TS), np.float32)
    for a in arrs:
        num += a * _WK
        den += _WK
    return num / den


def _eager(out_to_in, recon):
    cache = {t: recon(t) for ins in out_to_in.values() for t in ins}
    return {o: _blend([cache[t] for t in out_to_in[o]]) for o in out_to_in}


def _run(out_to_in, order, recon, budget_bytes=None, spill=None):
    sched = WindowedScheduler(out_to_in, order)
    cache = TileCache(ram_budget_bytes=budget_bytes or 1 << 60, spill=spill)
    out: dict = {}
    sched.run(cache, recon, _blend, lambda o, a: out.__setitem__(o, a))
    return out, cache.stats


def test_adapter_order_is_valid_and_parity_exact():
    plan, in_coord = _fake_plan(6)
    recon = _recon_for(in_coord)
    ref = _eager(plan.output_to_inputs, recon)
    for kind in ("morton", "raster"):
        out_to_in, order = output_to_inputs_and_order(plan, order=kind)
        assert sorted(order) == sorted(plan.output_to_inputs)  # valid permutation, no dupes/drops
        out, _ = _run(out_to_in, order, recon)
        assert all(np.array_equal(ref[o], out[o]) for o in ref), kind


def test_adapter_morton_spills_less_than_raster():
    plan, in_coord = _fake_plan(8)
    recon = _recon_for(in_coord)
    tile_bytes = recon(0).nbytes
    out_to_in, ras = output_to_inputs_and_order(plan, order="raster")
    _, mor = output_to_inputs_and_order(plan, order="morton")
    nat = _run(out_to_in, ras, recon)[1].peak_tiles
    budget = int(nat * 0.6) * tile_bytes
    ras_stats = _run(out_to_in, ras, recon, budget_bytes=budget, spill=DictSpillStore())[1]
    mor_stats = _run(out_to_in, mor, recon, budget_bytes=budget, spill=DictSpillStore())[1]
    assert mor_stats.spills < ras_stats.spills, (mor_stats.spills, ras_stats.spills)
