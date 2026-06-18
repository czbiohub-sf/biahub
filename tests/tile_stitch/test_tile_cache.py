"""Unit tests for ``biahub.tile_stitch.tile_cache`` — the engine-agnostic streaming
cache core. No GPU, no Monarch, no waveorder plan: synthetic numpy tiles on a
synthetic overlap grid. Pins the four locally-validated properties:

  H1  windowed sweep == eager (byte-identical) AND offline peak == measured peak,
  cache  configurable in-memory byte budget is respected (spill keeps RAM <= budget),
  H2  Morton order spills less than raster at the same budget.
"""

from __future__ import annotations

import numpy as np

from biahub.tile_stitch.tile_cache import (
    DictSpillStore,
    LocalDirSpillStore,
    TileCache,
    WindowedScheduler,
    ZarrSpillStore,
    morton_order,
    raster_order,
)

TS = 4          # px per tile axis (tiny)
TILE, OVERLAP = 8, 2
STRIDE = TILE - OVERLAP


def _build_plan(n: int):
    """n tiles/axis. Returns out_to_in: {ocell(z,y,x) -> [input tile (z,y,x), ...]}.
    Input tiles overlap (stride = tile - overlap); an output cell's contributors are
    the input tiles whose span intersects it — the real recon→multi-output DAG."""
    def ov(a, b):
        return a * STRIDE < b * STRIDE + STRIDE and a * STRIDE + TILE > b * STRIDE

    return {
        (oz, oy, ox): [
            (iz, iy, ix)
            for iz in range(n) if ov(iz, oz)
            for iy in range(n) if ov(iy, oy)
            for ix in range(n) if ov(ix, ox)
        ]
        for oz in range(n) for oy in range(n) for ox in range(n)
    }


def _recon(tid) -> np.ndarray:
    z, y, x = tid  # deterministic in absolute position → eager/windowed must agree
    zz, yy, xx = np.mgrid[0:TS, 0:TS, 0:TS].astype(np.float32)
    return (z * TS + zz) + (y * TS + yy) * 1e3 + (x * TS + xx) * 1e6


_W = (np.hanning(TS + 2)[1:-1].astype(np.float32) + 1e-3)
_WK = _W[:, None, None] * _W[None, :, None] * _W[None, None, :]


def _blend(arrs):
    num = np.zeros((TS, TS, TS), np.float32)
    den = np.zeros((TS, TS, TS), np.float32)
    for a in arrs:
        num += a * _WK
        den += _WK
    return num / den


def _eager(out_to_in):
    cache = {t: _recon(t) for o in out_to_in for t in out_to_in[o]}
    return {o: _blend([cache[t] for t in out_to_in[o]]) for o in out_to_in}


def _run(out_to_in, order, batch=1, budget_bytes=None, spill=None, io=None):
    sched = WindowedScheduler(out_to_in, order, batch=batch)
    cache = TileCache(ram_budget_bytes=budget_bytes or 1 << 60, spill=spill, io=io)
    out: dict = {}
    stats = sched.run(cache, _recon, _blend, lambda o, a: out.__setitem__(o, a))
    return out, stats, sched


def test_windowed_matches_eager_and_predicted_peak():
    """H1: byte-identical output to eager; offline interval-overlap == measured peak."""
    o2i = _build_plan(6)
    ref = _eager(o2i)
    for kind, order_fn in (("raster", raster_order), ("morton", morton_order)):
        out, stats, sched = _run(o2i, order_fn(o2i))
        assert all(np.array_equal(ref[o], out[o]) for o in ref), f"{kind} diverged"
        assert stats.peak_tiles == sched.predict_peak_tiles(), kind
        assert stats.recons == len({t for ts in o2i.values() for t in ts})  # recon-once


def test_in_memory_budget_is_respected():
    """The configurable in-memory cache size caps resident bytes (via MIN spill),
    output stays exact, and tightening the budget forces more spills."""
    o2i = _build_plan(6)
    ref = _eager(o2i)
    order = morton_order(o2i)
    tile_bytes = _recon((0, 0, 0)).nbytes
    nat = _run(o2i, order)[1].peak_tiles  # natural peak (tiles)

    last_spills = -1
    for frac in (1.0, 0.5, 0.33):
        budget = max(tile_bytes, int(nat * frac) * tile_bytes)
        out, stats, _ = _run(o2i, order, budget_bytes=budget, spill=DictSpillStore())
        assert all(np.array_equal(ref[o], out[o]) for o in ref), frac
        assert stats.peak_bytes <= budget, (stats.peak_bytes, budget)
        if last_spills >= 0:
            assert stats.spills >= last_spills  # tighter budget → >= spills
        last_spills = stats.spills


def test_morton_spills_less_than_raster():
    """H2: at a fixed budget, Morton order evicts/spills fewer tiles than raster."""
    o2i = _build_plan(8)
    tile_bytes = _recon((0, 0, 0)).nbytes
    nat = _run(o2i, raster_order(o2i))[1].peak_tiles
    budget = int(nat * 0.6) * tile_bytes

    ras = _run(o2i, raster_order(o2i), budget_bytes=budget, spill=DictSpillStore())[1]
    mor = _run(o2i, morton_order(o2i), budget_bytes=budget, spill=DictSpillStore())[1]
    assert mor.spills < ras.spills, (mor.spills, ras.spills)


def test_local_dir_spill_parity(tmp_path):
    """Real disk round-trip (LocalDirSpillStore, np.save/np.load) preserves
    byte-identical output and respects the in-memory budget."""
    o2i = _build_plan(6)
    ref = _eager(o2i)
    order = morton_order(o2i)
    tile_bytes = _recon((0, 0, 0)).nbytes
    nat = _run(o2i, order)[1].peak_tiles
    budget = max(tile_bytes, int(nat * 0.4) * tile_bytes)
    out, stats, _ = _run(o2i, order, budget_bytes=budget, spill=LocalDirSpillStore(root=tmp_path))
    assert all(np.array_equal(ref[o], out[o]) for o in ref)
    assert stats.peak_bytes <= budget
    assert stats.spills > 0  # real disk round-trips actually exercised


def test_async_prefetch_spill_ahead_parity(tmp_path):
    """The async IO lane (prefetch reload-ahead + spill-ahead write-back) on a real
    ThreadPoolExecutor + node-local disk preserves byte-identical output and budget,
    and actually exercises spills. Batch > 1 to stress the union/eviction path."""
    from concurrent.futures import ThreadPoolExecutor

    o2i = _build_plan(6)
    ref = _eager(o2i)
    order = morton_order(o2i)
    tile_bytes = _recon((0, 0, 0)).nbytes
    nat = _run(o2i, order)[1].peak_tiles
    budget = max(tile_bytes, int(nat * 0.4) * tile_bytes)
    with ThreadPoolExecutor(max_workers=2) as io:
        out, stats, _ = _run(
            o2i, order, batch=4, budget_bytes=budget,
            spill=LocalDirSpillStore(root=tmp_path), io=io,
        )
    assert all(np.array_equal(ref[o], out[o]) for o in ref)
    assert stats.peak_bytes <= budget
    assert stats.spills > 0


def test_no_spill_store_runs_ram_only():
    """Without a spill store the cache is RAM-only (budget advisory) and still correct."""
    o2i = _build_plan(4)
    ref = _eager(o2i)
    out, stats, _ = _run(o2i, morton_order(o2i), budget_bytes=1)  # tiny budget, no spill
    assert all(np.array_equal(ref[o], out[o]) for o in ref)
    assert stats.spills == 0


def test_zarr_spill_parity(tmp_path):
    """ZarrSpillStore on a node-local LocalStore (the disk-backed Zarr spill) preserves
    byte-identical output, respects the budget, and actually round-trips through zarr."""
    o2i = _build_plan(6)
    ref = _eager(o2i)
    order = morton_order(o2i)
    tile_bytes = _recon((0, 0, 0)).nbytes
    nat = _run(o2i, order)[1].peak_tiles
    budget = max(tile_bytes, int(nat * 0.4) * tile_bytes)
    out, stats, _ = _run(o2i, order, budget_bytes=budget, spill=ZarrSpillStore(root=tmp_path))
    assert all(np.array_equal(ref[o], out[o]) for o in ref)
    assert stats.peak_bytes <= budget
    assert stats.spills > 0  # real zarr round-trips actually exercised


def test_zarr_memory_spill_parity():
    """In-memory ZarrSpillStore (zarr MemoryStore) — the 'in-memory Zarr store' tier —
    preserves byte-identical output and respects the budget."""
    from zarr.storage import MemoryStore

    o2i = _build_plan(6)
    ref = _eager(o2i)
    order = morton_order(o2i)
    tile_bytes = _recon((0, 0, 0)).nbytes
    nat = _run(o2i, order)[1].peak_tiles
    budget = max(tile_bytes, int(nat * 0.4) * tile_bytes)
    out, stats, _ = _run(o2i, order, budget_bytes=budget, spill=ZarrSpillStore(store=MemoryStore()))
    assert all(np.array_equal(ref[o], out[o]) for o in ref)
    assert stats.peak_bytes <= budget
    assert stats.spills > 0


def test_zarr_async_prefetch_spill_ahead_parity(tmp_path):
    """ZarrSpillStore through the async prefetch/spill-ahead lane (ThreadPoolExecutor)
    stays byte-identical + budget-safe — exercises concurrent put/pop on the zarr group."""
    from concurrent.futures import ThreadPoolExecutor

    o2i = _build_plan(6)
    ref = _eager(o2i)
    order = morton_order(o2i)
    tile_bytes = _recon((0, 0, 0)).nbytes
    nat = _run(o2i, order)[1].peak_tiles
    budget = max(tile_bytes, int(nat * 0.4) * tile_bytes)
    with ThreadPoolExecutor(max_workers=2) as io:
        out, stats, _ = _run(
            o2i, order, batch=4, budget_bytes=budget,
            spill=ZarrSpillStore(root=tmp_path), io=io,
        )
    assert all(np.array_equal(ref[o], out[o]) for o in ref)
    assert stats.peak_bytes <= budget
    assert stats.spills > 0
