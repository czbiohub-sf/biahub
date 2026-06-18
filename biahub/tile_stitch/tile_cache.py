"""Streaming tile cache — bounded-memory reconstruct-and-stitch core.

Engine-agnostic on purpose: keyed by an opaque tile-id type ``K`` (hashable), the
plan is a plain ``output_id -> [input_id, ...]`` dict, and recon / blend / write are
injected callables. So this whole layer unit-tests with synthetic numpy tiles (no
GPU, no Monarch, no waveorder plan), and the real engine adapts ``plan.output_to_inputs``
+ Tikhonov recon + ``_core.blend_contributors`` + the shard writer onto it.

Design + validation: ``tile_cache_design.md`` (same dir).

STATUS: this layer is validated standalone (``tests/tile_stitch/test_tile_cache*``),
but the driver-side spill mechanism is **not yet wired into the live Monarch engine**
— only the *policy* (``WindowedScheduler.predict_peak_tiles`` for the recon gate +
``morton_order``) is used today. Production spill is worker-side (RDMA re-registration;
see ``rdma_spill_design.md``), and the config knobs a wired cache would add (an
in-memory byte budget + a spill path) do **not** exist in ``MonarchConfig`` yet —
``ram_budget_bytes`` and the spill ``root`` are constructor args here, not config.

The behaviour here mirrors
the locally-validated prototype: output-windowed traversal (Morton order) + ref-count
**Belady-MIN** eviction bounds peak host RAM to a configurable **in-memory cache size**
(``TileCache.ram_budget_bytes``), spilling cold tiles to a pluggable ``SpillStore``
(node-local NVMe in production), and produces byte-identical output to the eager path.

Optional async IO lane (``io=`` executor): **prefetch** reloads the next batch's
spilled tiles ahead of need and **spill-ahead** writes evictions back asynchronously,
so NVMe IO overlaps the main thread's recon/blend (the schedule gives free lookahead).
Spill stores are lock-guarded, so any worker count is safe.

Python 3.12: PEP 695 generics (``TileCache[K]`` parameterised on the tile-id type),
the ``type`` alias statement, and ``itertools.batched`` for the windowed sweep.
"""

from __future__ import annotations

import bisect
import heapq
import os
import tempfile
import threading

from collections.abc import Callable, Hashable, Iterable, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Protocol

import numpy as np

type OKey = Hashable  # output tile / cell id (the tile-id type is the generic K)


# --------------------------------------------------------------------------- #
# Spill backend (mechanism). The cache evicts MIN victims here when over budget.
# Lock-guarded dict mutations (IO kept outside the lock) → thread-safe for the
# cache's async IO lane.
# --------------------------------------------------------------------------- #
class SpillStore[K](Protocol):
    def put(self, tid: K, arr: np.ndarray) -> None: ...
    def pop(self, tid: K) -> np.ndarray: ...  # remove + return
    def drop(self, tid: K) -> None: ...  # discard if present
    def __contains__(self, tid: K) -> bool: ...


class DictSpillStore[K]:
    """In-RAM spill backend — for logic validation / tests. Validates the spill
    PATH and counts; it does not actually reduce process RSS (use a disk-backed
    store for that). The real P2 backend is ``LocalDirSpillStore`` (np.save to
    node-local NVMe); P5 graduates to the log-structured ``TileStore``."""

    def __init__(self) -> None:
        self._d: dict[K, np.ndarray] = {}
        self._lock = threading.Lock()

    def put(self, tid: K, arr: np.ndarray) -> None:
        with self._lock:
            self._d[tid] = arr

    def pop(self, tid: K) -> np.ndarray:
        with self._lock:
            return self._d.pop(tid)

    def drop(self, tid: K) -> None:
        with self._lock:
            self._d.pop(tid, None)

    def __contains__(self, tid: K) -> bool:
        with self._lock:
            return tid in self._d


def node_local_scratch(override: str | os.PathLike | None = None) -> Path:
    """Pick a node-local scratch dir for spilling (NVMe in production), preferring
    SLURM/TMPDIR over /tmp. NOT a networked FS like /hpc — the spill is
    write-once-read-few and wants local bandwidth + no inode pressure on the shared
    store. Pass ``override`` to force a location (a config field for this is part of
    the deferred engine-wiring; see the module STATUS note)."""
    for cand in (override, os.environ.get("SLURM_TMPDIR"), os.environ.get("TMPDIR"), "/tmp"):
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            return Path(cand)
    return Path(tempfile.gettempdir())


class LocalDirSpillStore[K]:
    """Disk-backed spill to a node-local directory (NVMe scratch) — the real
    bytes-saving backend (vs ``DictSpillStore``). One ``.npy`` per spilled tile;
    an in-RAM map tracks ``tile-id -> path`` so keys can be arbitrary hashables.
    Files are unlinked on pop/drop; the path map is lock-guarded while the np.save /
    np.load IO runs outside the lock (so the async lane actually overlaps). P5
    graduates this to the log-structured ``TileStore`` — same seam, no caller change."""

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        self.root = node_local_scratch(root) / f"tilecache_{os.getpid()}"
        self.root.mkdir(parents=True, exist_ok=True)
        self._paths: dict[K, Path] = {}
        self._n = 0
        self._lock = threading.Lock()

    def put(self, tid: K, arr: np.ndarray) -> None:
        with self._lock:
            p = self._paths.get(tid)
            if p is None:
                p = self.root / f"t{self._n:08d}.npy"
                self._n += 1
                self._paths[tid] = p
        np.save(p, arr, allow_pickle=False)

    def pop(self, tid: K) -> np.ndarray:
        with self._lock:
            p = self._paths.pop(tid)
        arr = np.load(p)
        p.unlink(missing_ok=True)
        return arr

    def drop(self, tid: K) -> None:
        with self._lock:
            p = self._paths.pop(tid, None)
        if p is not None:
            p.unlink(missing_ok=True)

    def __contains__(self, tid: K) -> bool:
        with self._lock:
            return tid in self._paths


class ZarrSpillStore[K]:
    """Zarr-backed spill: each tile is one (compressed) array in a zarr group. The
    backing store is pluggable:

    * ``ZarrSpillStore()`` → a node-local NVMe ``LocalStore`` (the disk-backed Zarr
      store on node temp, via :func:`node_local_scratch`).
    * ``ZarrSpillStore(store=zarr.storage.MemoryStore())`` → an in-RAM Zarr store.

    Zarr's group/array objects aren't thread-safe, so every operation is serialized
    under a lock; the cache's io executor still overlaps this spill IO with the
    main-thread recon/blend."""

    def __init__(self, store=None, root: str | os.PathLike | None = None) -> None:
        import zarr
        from zarr.storage import LocalStore

        if store is None:
            self.root = node_local_scratch(root) / f"tilecache_zarr_{os.getpid()}"
            self.root.mkdir(parents=True, exist_ok=True)
            store = LocalStore(str(self.root))
        self._group = zarr.open_group(store=store, mode="w")
        self._slots: dict[K, str] = {}
        self._n = 0
        self._lock = threading.Lock()

    def put(self, tid: K, arr: np.ndarray) -> None:
        with self._lock:
            name = self._slots.get(tid)
            if name is None:
                name = f"t{self._n:08d}"
                self._n += 1
                self._slots[tid] = name
            za = self._group.create_array(
                name=name, shape=arr.shape, dtype=arr.dtype, chunks=arr.shape, overwrite=True
            )
            za[...] = arr

    def pop(self, tid: K) -> np.ndarray:
        with self._lock:
            name = self._slots.pop(tid)
            arr = np.asarray(self._group[name])
            del self._group[name]
            return arr

    def drop(self, tid: K) -> None:
        with self._lock:
            name = self._slots.pop(tid, None)
            if name is not None:
                del self._group[name]

    def __contains__(self, tid: K) -> bool:
        with self._lock:
            return tid in self._slots


# --------------------------------------------------------------------------- #
# Traversal orders (policy input). Morton keeps a cell's contributors co-resident.
# --------------------------------------------------------------------------- #
def raster_order(cells: Iterable[Sequence[int]]) -> list:
    return sorted(cells)


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


# --------------------------------------------------------------------------- #
# The cache: RAM byte-budget + MIN spill (+ optional async prefetch/spill-ahead).
# --------------------------------------------------------------------------- #
@dataclass
class CacheStats:
    peak_bytes: int = 0
    peak_tiles: int = 0
    spills: int = 0
    reloads: int = 0
    recons: int = 0


class TileCache[K]:
    """RAM-resident tile cache bounded by ``ram_budget_bytes`` (the in-memory cache
    size). When resident bytes exceed the budget, evicts the Belady-MIN victim
    (tile whose next consumer is furthest in the schedule) to ``spill``. Budget is
    enforced only when a ``spill`` store is given — without one there is nowhere to
    evict, so the cache runs RAM-only (budget advisory).

    With an ``io`` executor, eviction writes are submitted async (spill-ahead, the
    array stays referenced by the in-flight future) and ``prefetch`` issues async
    reloads — overlapping NVMe IO with the caller's recon/blend. ``resident_bytes``
    is the logical resident total; actual RSS transiently includes in-flight-write
    arrays (bounded by the executor queue depth). Call ``wait_io`` at run end."""

    def __init__(
        self, ram_budget_bytes: int, spill: SpillStore[K] | None = None, io: Executor | None = None
    ) -> None:
        self.ram_budget_bytes = int(ram_budget_bytes)
        self.spill = spill
        self.io = io
        self._ram: dict[K, np.ndarray] = {}
        self._bytes = 0
        self._iolock = threading.Lock()
        self._loads: dict[K, Future] = {}   # in-flight prefetch reloads
        self._writing: dict[K, Future] = {}  # in-flight spill-ahead writes
        self.stats = CacheStats()

    @property
    def resident_bytes(self) -> int:
        return self._bytes

    def prefetch(self, tids: Sequence[K]) -> None:
        """Async reload-ahead of already-spilled upcoming tiles — hides reload
        latency behind the caller's recon/blend. No-op without ``io`` + ``spill``."""
        if self.io is None or self.spill is None:
            return
        for tid in tids:
            with self._iolock:
                if tid in self._loads:
                    continue
            if tid in self._ram or tid not in self.spill:
                continue
            fut = self.io.submit(self.spill.pop, tid)
            with self._iolock:
                self._loads[tid] = fut

    def _fetch(self, tid: K, recon_fn: Callable[[K], np.ndarray]) -> np.ndarray:
        """Resolve a tile: pending prefetch → pending spill-ahead → on-disk → recon."""
        with self._iolock:
            load = self._loads.pop(tid, None)
            write = self._writing.pop(tid, None) if load is None else None
        if load is not None:
            self.stats.reloads += 1
            return load.result()
        if write is not None:
            write.result()  # finish the write so the tile is on disk, then read it
            self.stats.reloads += 1
            return self.spill.pop(tid)
        if self.spill is not None and tid in self.spill:
            self.stats.reloads += 1
            return self.spill.pop(tid)
        self.stats.recons += 1
        return recon_fn(tid)

    def ensure(
        self,
        tids: Sequence[K],
        recon_fn: Callable[[K], np.ndarray],
        key_fn: Callable[[K], float],
        protect: set[K],
    ) -> None:
        """Make every tid resident, evicting MIN victims to stay under budget.
        ``protect`` = tiles needed right now (never evict). ``key_fn(tid)`` =
        next-use distance; higher → evict first (Belady-MIN)."""
        for tid in tids:
            if tid in self._ram:
                continue
            arr = self._fetch(tid, recon_fn)
            self._ram[tid] = arr
            self._bytes += arr.nbytes
            self._evict(key_fn, protect)
        self.stats.peak_bytes = max(self.stats.peak_bytes, self._bytes)
        self.stats.peak_tiles = max(self.stats.peak_tiles, len(self._ram))

    def _evict(self, key_fn: Callable[[K], float], protect: set[K]) -> None:
        if self.spill is None:
            return  # RAM-only: nowhere to evict to
        while self._bytes > self.ram_budget_bytes:
            victim = max((t for t in self._ram if t not in protect), key=key_fn, default=None)
            if victim is None:
                return  # everything resident is needed by the current batch
            arr = self._ram.pop(victim)
            self._bytes -= arr.nbytes
            self.stats.spills += 1
            if self.io is not None:  # spill-ahead: write off the critical path
                fut = self.io.submit(self.spill.put, victim, arr)
                with self._iolock:
                    self._writing[victim] = fut
                fut.add_done_callback(lambda f, k=victim: self._forget_write(k, f))
            else:
                self.spill.put(victim, arr)

    def _forget_write(self, tid: K, fut: Future) -> None:
        with self._iolock:
            if self._writing.get(tid) is fut:
                self._writing.pop(tid, None)

    def get(self, tid: K) -> np.ndarray:
        return self._ram[tid]

    def drop(self, tid: K) -> None:
        """Free a tile permanently (its ref-count hit zero)."""
        with self._iolock:
            load = self._loads.pop(tid, None)
            write = self._writing.pop(tid, None)
        if load is not None:
            load.result()  # let the in-flight reload finish (file already removed)
        if write is not None:
            write.result()  # let the in-flight write finish before dropping
        arr = self._ram.pop(tid, None)
        if arr is not None:
            self._bytes -= arr.nbytes
        if self.spill is not None:
            self.spill.drop(tid)

    def wait_io(self) -> None:
        """Drain all in-flight async IO (call once at run end)."""
        with self._iolock:
            futs = [*self._writing.values(), *self._loads.values()]
            self._writing.clear()
            self._loads.clear()
        for f in futs:
            f.result()


# --------------------------------------------------------------------------- #
# The scheduler (policy): windowed sweep + ref-count lifetimes + next-use keys.
# --------------------------------------------------------------------------- #
class WindowedScheduler[K]:
    """Drives an output-windowed sweep over ``order`` in groups of ``batch``,
    using ``TileCache`` for residency. Owns ref-count lifetimes (free on last
    consumer), the next-use eviction key, and one-batch-ahead prefetch."""

    def __init__(
        self, out_to_in: dict[OKey, Sequence[K]], order: Sequence[OKey], batch: int = 1
    ) -> None:
        self.out_to_in = {o: list(out_to_in[o]) for o in order}
        self.order = list(order)
        self.batch = max(1, int(batch))
        self._pos = {o: i for i, o in enumerate(self.order)}
        self.refcount: dict[K, int] = {}
        uses: dict[K, list[int]] = {}
        for o in self.order:
            for t in self.out_to_in[o]:
                self.refcount[t] = self.refcount.get(t, 0) + 1
                uses.setdefault(t, []).append(self._pos[o])
        self._uses = {t: sorted(p) for t, p in uses.items()}  # for next-use lookup

    def _next_use_key(self, cur_idx: int) -> Callable[[K], float]:
        def key(t: K) -> float:
            ps = self._uses[t]
            i = bisect.bisect_left(ps, cur_idx)
            return ps[i] if i < len(ps) else float("inf")

        return key

    def run(
        self,
        cache: TileCache[K],
        recon_fn: Callable[[K], np.ndarray],
        blend_fn: Callable[[list[np.ndarray]], np.ndarray],
        write_fn: Callable[[OKey, np.ndarray], None],
    ) -> CacheStats:
        refcount = dict(self.refcount)
        groups = list(batched(self.order, self.batch))
        unions = [list(dict.fromkeys(t for o in g for t in self.out_to_in[o])) for g in groups]
        for i, grp in enumerate(groups):
            cache.ensure(unions[i], recon_fn, self._next_use_key(self._pos[grp[0]]), protect=set(unions[i]))
            if i + 1 < len(groups):
                cache.prefetch(unions[i + 1])  # reload next batch's spilled tiles ahead
            for o in grp:
                write_fn(o, blend_fn([cache.get(t) for t in self.out_to_in[o]]))
            for o in grp:  # ref-count AFTER the whole batch blends (no premature free)
                for t in self.out_to_in[o]:
                    refcount[t] -= 1
                    if refcount[t] == 0:
                        cache.drop(t)
        cache.wait_io()
        return cache.stats

    def predict_peak_tiles(self) -> int:
        """Offline interval-overlap = exact peak resident tiles for batch=1
        (Cubed / register-allocation liveness model). Plan-time memory gate."""
        live = sorted((ps[0], ps[-1]) for ps in self._uses.values())
        cur = peak = 0
        ends: list[int] = []
        for s, e in live:
            while ends and ends[0] < s:
                heapq.heappop(ends)
                cur -= 1
            heapq.heappush(ends, e)
            cur += 1
            peak = max(peak, cur)
        return peak
