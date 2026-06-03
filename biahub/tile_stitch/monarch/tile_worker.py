"""Monarch Actor port of v7's ``reconstruct_tile_memory_gpu``.

One Actor per GPU. Reconstructs assigned tiles, stores the CPU result on
the actor (keeping it alive), and returns an :class:`RDMABuffer` handle.
Downstream Stage B receives these handles in a dict, ``read_into`` each
via RDMA (ibverbs over IB intra-node, no Monarch mailbox involved for
the bulk data), then blends locally and writes the zarr chunk.

Monarch 0.5 RDMABuffer is CPU-only — GPU tensors would be bounced
through CPU automatically. We already pay the GPU→CPU move at the end
of recon (the v7 dask path does too), so there is no additional cost.
"""

from typing import Any

from monarch.actor import Actor, current_rank, endpoint
from monarch.rdma import RDMABuffer

from biahub.tile_stitch import _core
from biahub.tile_stitch._core import PrefetchReader
from biahub.tile_stitch.config import MonarchConfig


def _read_numa_node_for_gpu(gpu_idx: int) -> int | None:
    """Resolve NUMA node for a CUDA device via pynvml→sysfs.

    Returns ``None`` if the PCI device reports node ``-1`` (kernel-managed,
    no affinity) or anything fails. H200 nodes report a real node for
    every GPU, so ``None`` should only happen on misconfigured boxes.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            bus_id = pci.busIdLegacy.decode().lower()
        finally:
            pynvml.nvmlShutdown()
        with open(f"/sys/bus/pci/devices/{bus_id}/numa_node") as f:
            node = int(f.read().strip())
        if node < 0:
            return None
        return node
    except Exception:
        return None


def _pin_to_numa_for_gpu(gpu_idx: int) -> int | None:
    """Set CPU affinity of the current thread to the GPU's NUMA node.

    Returns the NUMA node id we pinned to (or ``None`` if no affinity was
    applied — for example, on a host without sysfs NUMA info).
    """
    import os

    node = _read_numa_node_for_gpu(gpu_idx)
    if node is None:
        # H200 Bruno fallback: GPU 0 → node 2, GPU 1 → node 3.
        node = gpu_idx + 2
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            cpulist = f.read().strip()
    except FileNotFoundError:
        return None
    cpus: set[int] = set()
    for part in cpulist.split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        return None
    # SLURM cgroup may restrict us to a subset of host CPUs. Intersect so
    # ``sched_setaffinity`` does not EINVAL on CPUs we don't actually own.
    allowed = os.sched_getaffinity(0)
    target = cpus & allowed
    if not target:
        return None
    os.sched_setaffinity(0, target)
    return node


class TileHandle:
    """Picklable handle: ``RDMABuffer`` + shape + dtype name.

    The buffer holds the byte-flat view of the recon tensor; shape + dtype
    let the consumer rebuild the (C, Z, Y, X) tensor after ``read_into``.
    """

    __slots__ = ("buffer", "shape", "dtype_name")

    def __init__(self, buffer: RDMABuffer, shape: tuple[int, ...], dtype_name: str):
        self.buffer = buffer
        self.shape = shape
        self.dtype_name = dtype_name


class TileWorker(Actor):
    """One actor per GPU. Holds plan + TF cache + recon results."""

    def __supervise__(self, failure) -> bool:
        """Log mesh failures with context; propagate so the controller sees them.

        Returning False lets the failure bubble up to the parent actor
        (the controller / run-script). We don't try to auto-recover from
        a worker crash — a recon or stitch failure means lost data and we
        prefer to surface it loudly rather than silently retry.
        """
        import logging as _logging

        log = _logging.getLogger("TileWorker.supervise")
        log.error(
            "supervision: mesh=%s gpu_idx=%s report=%s",
            getattr(failure, "mesh_name", "?"),
            getattr(self, "gpu_idx", "?"),
            getattr(failure, "report", lambda: str(failure))(),
        )
        return False

    def __init__(self, plan_path: str):
        """Pin to this rank's GPU and pre-load the plan + transfer function."""
        import torch

        from biahub.tile_stitch.plan import load_plan

        rank = current_rank()
        try:
            gpu_idx = rank["gpus"]
        except (KeyError, TypeError):
            gpu_idx = int(rank)
        torch.cuda.set_device(gpu_idx)
        self.gpu_idx = gpu_idx
        # Pin CPU affinity to the GPU-local NUMA node. asyncio thread pool
        # workers (used by ``asyncio.to_thread`` for blend + zarr write)
        # inherit this affinity on Linux, keeping CPU blend work on the
        # same socket as the GPU we DMA from.
        self._numa_node = _pin_to_numa_for_gpu(gpu_idx)
        self.plan = load_plan(plan_path)
        self.plan_path = plan_path
        # Monarch knobs ride on the plan (carried across setup + swap_to). An
        # older pickle (or a plan built without config) has ``monarch=None`` —
        # fall back to defaults so the actor still runs.
        self._cfg = self.plan.monarch or MonarchConfig()
        # Per-tile recon storage. Tensors stay resident until the actor
        # tears down (or ``forget`` is called) so the RDMABuffer handles
        # we hand to peers remain valid.
        self.recons: dict[int, torch.Tensor] = {}
        # Serialize the GPU-bound recon. Default ``recon_concurrency=1`` —
        # Tikhonov holds ~30 GB on device, so >1 risks OOM. Sized from config
        # at first use; lazy-init avoids binding an asyncio loop here (this
        # ``__init__`` may run before the loop exists in some transport paths).
        self._recon_sem = None
        # Background prefetch reader (streaming mode only). Primed per-TP via
        # the ``prime_reader`` endpoint with this actor's assigned tile order.
        self._reader: PrefetchReader | None = None
        # Streaming tiles is the default: per-tile zarr reads, no resident
        # volume. ``monarch.resident_volume=true`` opts into GPU-resident mode
        # (uint16 source halves HBM vs float32; ~34 GB volume + ~30 GB Tikhonov
        # ≈ 64 GB on the 80 GB H200, fits only for our current dataset size).
        # At 2× spatial growth or multi-node scale, streaming is mandatory.
        self._stream_tiles = not self._cfg.resident_volume
        if self._stream_tiles:
            self._volume_gpu = None
        else:
            self._volume_gpu = self._load_volume_to_gpu()
        # GPU H2D-overlap (``monarch.gpu_overlap``, default off). When on, the
        # prefetch reader produces pinned float32 host tensors and H2D copies
        # run on a dedicated copy stream; v2 stages the NEXT gpu_depth work-
        # units' inputs onto the device on the copy stream while the CURRENT
        # unit's FFT runs on the compute stream (exactly one FFT in flight —
        # the ~30 GB Tikhonov peak forbids concurrent compute). Streaming only
        # (resident volume already lives on the GPU). CUDA-graph capture
        # (reduce-overhead) can't compose with manual streams, so overlap forces
        # eager compile (see _get_compiled_recon).
        self._gpu_overlap = self._stream_tiles and self._cfg.gpu_overlap
        if self._gpu_overlap:
            self._copy_stream = torch.cuda.Stream(device=gpu_idx)
        else:
            self._copy_stream = None
        # GPU input stager (v2). ``gpu_depth`` = how many work-units ahead to
        # stage on the device (1 = one FFT's inputs ahead). ``_gpu_staged`` maps
        # tile_id -> (device tensor, H2D event); the FFT's compute stream waits
        # each tile's event before reading it. ``_assigned_order`` +
        # ``_stage_cursor`` track this actor's flat tile sequence (set by
        # prime_reader) so stage-ahead knows what comes next.
        self._gpu_depth = self._cfg.gpu_depth
        self._gpu_staged: dict[int, Any] = {}
        self._assigned_order: list[int] = []
        self._stage_cursor = 0    # index in _assigned_order staged up to
        self._consumed = 0        # tiles consumed by recon (bounds the window)
        # Lazily-built compiled recon callable. ``torch.compile`` with
        # ``mode="reduce-overhead"`` captures CUDA graphs after a couple
        # warm-up calls, eliminating Python+kernel-launch overhead for
        # the remaining ~98 tiles per actor.
        self._compiled_recon = None
        # Stitch-side geometric caches. Volume contents change per TP but
        # tile geometry, output slices, and blend kernels are constant
        # for the lifetime of the actor — pre-compute once and reuse.
        # ``_kernel_cache`` is keyed on ``(tile.shape, sample_dtype)`` so
        # a single recomputation per shape covers all 180 stitches × N TPs.
        self._blend_kernel = self.plan.settings.blend.build()
        self._tiles_by_id = {t.tile_id: t for t in self.plan.input_tiles}
        self._stitch_geom: dict[int, dict] = _core.build_stitch_geom(self.plan)
        self._kernel_cache: dict[tuple, Any] = {}
        self._reset_recon_stats()

    def _reset_recon_stats(self) -> None:
        """Zero the per-actor Stage A timing counters (call per TP)."""
        self._rs_n = 0          # tiles reconstructed
        self._rs_io_s = 0.0     # zarr read / volume slice + H2D
        self._rs_fft_s = 0.0    # recon_fn (FFT + Tikhonov)
        self._rs_d2h_s = 0.0    # GPU->CPU + RDMABuffer setup
        self._rs_first = None   # monotonic ts of first recon start
        self._rs_last = None    # monotonic ts of last recon end

    def _load_volume_to_gpu(self):
        """Read the whole TP+channel slab from zarr into HBM once.

        Stored in source dtype (typically uint16). ``_reconstruct_blocking``
        casts the per-tile view to float32 on the GPU when slicing.
        """
        import logging as _logging
        import time as _time

        import numpy as np
        import torch

        from iohub.ngff import open_ome_zarr

        log = _logging.getLogger("TileWorker.volume")
        t0 = _time.monotonic()
        src = open_ome_zarr(self.plan.input_path, layout="fov", mode="r")
        z_arr = src["0"]
        sl = (
            slice(self.plan.timepoint, self.plan.timepoint + 1),
            slice(self.plan.channel_idx, self.plan.channel_idx + 1),
        ) + tuple(slice(None) for _ in self.plan.tile_dims)
        arr = np.asarray(z_arr[sl]).squeeze(axis=(0, 1))  # (1,1,Z,Y,X) → (Z,Y,X), drop T, C
        t_read = _time.monotonic() - t0
        t1 = _time.monotonic()
        vol = torch.as_tensor(arr, device=f"cuda:{self.gpu_idx}")
        t_h2d = _time.monotonic() - t1
        log.info(
            "gpu_idx=%d volume loaded shape=%s dtype=%s read=%.1fs h2d=%.1fs",
            self.gpu_idx,
            tuple(vol.shape),
            vol.dtype,
            t_read,
            t_h2d,
        )
        return vol

    def _get_compiled_recon(self):
        """Return a (possibly torch.compile'd) callable ``zyx -> recon``.

        Binds the TF tensors + ``apply_inverse`` kwargs into a closure so
        the compiled graph only sees a single dynamic input. Mode
        ``reduce-overhead`` triggers CUDA-graph capture after a warm-up.
        On compile failure or env override, falls back to eager.
        """
        if self._compiled_recon is not None:
            return self._compiled_recon

        import logging as _logging

        import torch

        device = f"cuda:{self.gpu_idx}"
        cuda_tf, recon_settings = _core.get_tf_cuda(self.plan.settings, device)
        _eager = _core.make_eager_recon(cuda_tf, recon_settings)

        mode = self._cfg.compile_mode.value
        log = _logging.getLogger("TileWorker.compile")
        if self._gpu_overlap and mode != "none":
            # CUDA-graph capture (reduce-overhead) records a fixed stream and
            # cannot compose with the manual copy stream / cross-stream waits
            # the overlap path issues. Force eager so the two don't fight.
            log.info(
                "gpu_idx=%d GPU overlap on — forcing eager (was mode=%s)",
                self.gpu_idx, mode,
            )
            mode = "none"
        if mode == "none":
            log.info("gpu_idx=%d compile disabled (monarch.compile_mode=none)", self.gpu_idx)
            self._compiled_recon = _eager
            return _eager
        try:
            compiled = torch.compile(_eager, mode=mode, dynamic=False)
            log.info("gpu_idx=%d torch.compile mode=%s ready", self.gpu_idx, mode)
            self._compiled_recon = compiled
        except Exception as exc:
            log.warning(
                "gpu_idx=%d torch.compile failed (%s) — falling back to eager",
                self.gpu_idx, exc,
            )
            self._compiled_recon = _eager
        return self._compiled_recon

    @endpoint
    async def hostinfo(self) -> dict:
        """Diagnostic: report host, pid, GPU index, and plan tile counts."""
        import os
        import socket

        import torch

        affinity = sorted(os.sched_getaffinity(0))
        return {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_idx": self.gpu_idx,
            "numa_node": getattr(self, "_numa_node", None),
            "affinity_cpus": f"{affinity[0]}-{affinity[-1]}" if affinity else "?",
            "n_affinity_cpus": len(affinity),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "torch_cuda_current": torch.cuda.current_device(),
            "n_input_tiles": len(self.plan.input_tiles),
            "n_output_tiles": len(self.plan.output_tiles),
        }

    def _load_tile_gpu(self, tile):
        """Load one input tile to a ``(Z, Y, X)`` float32 GPU tensor.

        Streams from zarr (default) or slices the resident volume. The
        resident path's ``.clone()`` (in ``_core.load_tile_zyx``) decouples
        the slice so CUDA graphs see a stable input buffer.
        """
        volume = None if self._stream_tiles else self._volume_gpu
        return _core.load_tile_zyx(
            self.plan, tile, volume=volume, device=f"cuda:{self.gpu_idx}"
        )

    def _load_one(self, tile_id: int, tile):
        """Load one tile to a ``(Z,Y,X)`` f32 cuda tensor, preferring prefetch.

        Used by both the single-tile and batched recon paths so batched
        recon benefits from the same background read-ahead. Falls back to a
        synchronous load when prefetch is off or missed this tile.

        GPU-overlap mode: the prefetched value is a pinned float32 host tensor,
        so the H2D is issued ``non_blocking`` on ``self._copy_stream`` (off the
        recon critical path). ``record_stream`` ties the GPU buffer's lifetime
        to the compute stream so the caching allocator can't recycle it before
        the FFT runs; the caller (``_reconstruct*_blocking``) makes the compute
        stream ``wait_stream(self._copy_stream)`` once before compute.
        """
        import torch

        pre = self._reader.get(tile_id) if self._reader is not None else None
        if pre is None:
            return self._load_tile_gpu(tile)

        device = f"cuda:{self.gpu_idx}"
        if self._gpu_overlap and self._copy_stream is not None:
            # ``pre`` is a pinned float32 CPU tensor from the reader thread.
            with torch.cuda.stream(self._copy_stream):
                gpu = pre.to(device, non_blocking=True)
            gpu.record_stream(torch.cuda.current_stream(self.gpu_idx))
            # The H2D is async, so the pinned host buffer must outlive the
            # in-flight DMA. Tie its lifetime to the GPU tensor (which lives
            # through the FFT) so Python can't free it mid-copy.
            gpu._pinned_src = pre
            return gpu
        return torch.as_tensor(pre, dtype=torch.float32, device=device)

    def _stage_tile(self, tile_id: int) -> bool:
        """Queue one tile's H2D on the copy stream and record its event (v2).

        Pulls the pinned float32 host tensor from the prefetch reader and
        issues a ``non_blocking`` H2D on ``self._copy_stream``, recording a
        CUDA event the compute stream later waits on. Stores
        ``(device_tensor, event)`` in ``self._gpu_staged[tile_id]``. Returns
        False (no-op) if the reader has no pinned tensor for this tile — the
        consumer falls back to a synchronous load. Idempotent: already-staged
        tiles are skipped.
        """
        import torch

        if tile_id in self._gpu_staged:
            return True
        pre = self._reader.get(tile_id) if self._reader is not None else None
        if pre is None:
            return False
        device = f"cuda:{self.gpu_idx}"
        with torch.cuda.stream(self._copy_stream):
            gpu = pre.to(device, non_blocking=True)
            evt = torch.cuda.Event()
            evt.record(self._copy_stream)
        # Mark the buffer as used by both streams so the allocator can't
        # recycle it before the FFT (compute stream) reads it. Keep the pinned
        # host buffer alive through the in-flight DMA.
        gpu.record_stream(torch.cuda.current_stream(self.gpu_idx))
        gpu._pinned_src = pre
        self._gpu_staged[tile_id] = (gpu, evt)
        return True

    def _stage_fill_to(self, target_cursor: int, max_staged: int) -> None:
        """Stage tiles until ``_stage_cursor`` reaches ``target_cursor``.

        Walks ``self._assigned_order`` from the current cursor and queues each
        tile's H2D on the copy stream, advancing the cursor. Stops at the end
        of the assignment, the first tile the reader can't supply (loaded
        synchronously when consumed), or once ``len(_gpu_staged) >=
        max_staged``. ``target_cursor`` is an absolute index into
        ``_assigned_order`` so callers keep a bounded look-ahead relative to
        consumption; ``max_staged`` is a hard size gate on the staged set so
        the device-buffer footprint stays bounded even if recon units are
        consumed out of ``_assigned_order`` order (a stage that ran ahead of an
        out-of-order consumer would otherwise accumulate device buffers → OOM).
        """
        limit = min(target_cursor, len(self._assigned_order))
        while self._stage_cursor < limit:
            if len(self._gpu_staged) >= max_staged:
                break
            tid = self._assigned_order[self._stage_cursor]
            if not self._stage_tile(tid):
                break
            self._stage_cursor += 1

    def _take_staged(self, tile_id: int, tile):
        """Consume a pre-staged device tensor, or load synchronously (v2).

        Returns ``(gpu_tensor, event_or_None)``. If the tile was pre-staged,
        the caller must make the compute stream wait ``event`` before reading
        it. Otherwise falls back to the v1/synchronous ``_load_one`` path
        (event None) so a stage miss never corrupts the result.
        """
        staged = self._gpu_staged.pop(tile_id, None)
        if staged is not None:
            return staged  # (gpu, event)
        return self._load_one(tile_id, tile), None

    def _store_recon(self, tile_id: int, recon_cpu) -> TileHandle:
        """Keep the CPU tensor alive in ``self.recons`` and wrap an RDMABuffer."""
        import torch

        self.recons[tile_id] = recon_cpu
        flat = recon_cpu.view(torch.uint8).flatten()
        return TileHandle(
            buffer=RDMABuffer(flat),
            shape=tuple(recon_cpu.shape),
            dtype_name=str(recon_cpu.dtype),
        )

    @endpoint
    async def prime_reader(self, tile_ids: list[int]) -> dict:
        """Start a background reader over this actor's assigned tile order.

        Called by the driver once per TP with ``input_order[g::n_gpus]`` —
        the contiguous sequence this actor (gpu ``g``) will reconstruct. The
        reader pulls the next tile's zarr bytes while the GPU runs the
        current tile's FFT. No-op in resident-volume mode (reads slice HBM,
        nothing to prefetch) or when ``monarch.prefetch_depth=0``.
        """
        # Tear down any reader from the previous TP first.
        if self._reader is not None:
            self._reader.stop()
            self._reader = None

        # Reset the v2 GPU stager for the new TP (drop any device tensors
        # staged for the prior TP and record this actor's tile sequence so
        # stage-ahead knows what's next).
        self._gpu_staged.clear()
        self._assigned_order = list(tile_ids)
        self._stage_cursor = 0
        self._consumed = 0

        depth = self._cfg.prefetch_depth
        if depth <= 0 or not self._stream_tiles or not tile_ids:
            return {"gpu_idx": self.gpu_idx, "prefetch": False, "depth": depth}

        # Bind a per-tile loader over this actor's current plan. By default
        # the loader returns a source-dtype ``(Z, Y, X)`` numpy array and the
        # recon path casts to float32 during H2D (mirroring ``_load_tile_gpu``).
        # In GPU-overlap mode it instead returns a pinned float32 host tensor
        # so the cast + page-lock happen in this background thread, leaving the
        # recon path with just an async H2D on the copy stream.
        tiles_by_id = self._tiles_by_id
        plan = self.plan
        pin_float32 = self._gpu_overlap

        def _load(tid: int):
            return _core.read_tile_block(
                plan, tiles_by_id[tid], pin_float32=pin_float32
            )

        self._reader = PrefetchReader(_load, list(tile_ids), depth)
        return {
            "gpu_idx": self.gpu_idx,
            "prefetch": True,
            "depth": depth,
            "n_assigned": len(tile_ids),
        }

    def _reconstruct_blocking(self, tile_id: int) -> TileHandle:
        """Pure-blocking torch work for one tile (B=1 path)."""
        import torch

        # ``set_device`` is thread-local; the asyncio.to_thread worker
        # inherits cuda:0 by default. Re-pin so any ``"cuda"`` defaults
        # inside waveorder resolve to the right device.
        torch.cuda.set_device(self.gpu_idx)

        if self._gpu_overlap and self._copy_stream is not None:
            return self._reconstruct_overlap([tile_id])[0]
        return self._reconstruct_sync([tile_id])[0]

    def _reconstruct_batch_blocking(self, tile_ids: list[int]) -> list[TileHandle]:
        """Reconstruct B same-shape tiles in one batched waveorder call.

        Stacks the tiles into ``(B, Z, Y, X)`` so ``apply_inverse_transfer_
        function`` runs one batched FFT instead of B separate calls —
        amortizes kernel-launch + cuFFT-plan overhead. Caller guarantees
        all tiles in ``tile_ids`` share the same spatial shape.
        """
        import torch

        torch.cuda.set_device(self.gpu_idx)

        if self._gpu_overlap and self._copy_stream is not None:
            return self._reconstruct_overlap(tile_ids)
        return self._reconstruct_sync(tile_ids)

    def _reconstruct_sync(self, tile_ids: list[int]) -> list[TileHandle]:
        """Run synchronous recon for one work-unit (OFF path + overlap fallback).

        Loads each tile (prefetch or zarr), stacks into ``(B, Z, Y, X)`` (B may
        be 1), runs the batched FFT, copies each result to CPU, and wraps an
        RDMABuffer. Uses ``synchronize``-based io/fft/d2h timing — unchanged
        from the pre-overlap path so the OFF default is byte- and
        timing-semantics-identical.
        """
        import time as _t

        import torch

        tiles_by_id = self._tiles_by_id

        t0 = _t.monotonic()
        if self._rs_first is None:
            self._rs_first = t0
        zyx_list = [self._load_one(tid, tiles_by_id[tid]) for tid in tile_ids]
        batch = torch.stack(zyx_list, dim=0)  # (B, Z, Y, X)
        torch.cuda.synchronize(self.gpu_idx)
        t_io = _t.monotonic()

        recon_fn = self._get_compiled_recon()
        recons = recon_fn(batch)  # (B, Z, Y, X)
        torch.cuda.synchronize(self.gpu_idx)
        t_fft = _t.monotonic()

        handles: list[TileHandle] = []
        for i, tid in enumerate(tile_ids):
            recon_cpu = (
                recons[i].unsqueeze(0).to(torch.float32).contiguous().detach().cpu()
            )
            handles.append(self._store_recon(tid, recon_cpu))
        t_end = _t.monotonic()

        self._rs_n += len(tile_ids)
        self._rs_io_s += t_io - t0
        self._rs_fft_s += t_fft - t_io
        self._rs_d2h_s += t_end - t_fft
        self._rs_last = t_end
        return handles

    def _reconstruct_overlap(self, tile_ids: list[int]) -> list[TileHandle]:
        """v2 recon for one work-unit with cross-unit input H2D overlap.

        Consumes the unit's inputs from the GPU stager (their H2D was queued on
        the copy stream during the PRIOR unit's FFT), makes the compute stream
        wait each tile's H2D event, runs the one batched FFT, then — before the
        blocking D2H — stages the NEXT ``gpu_depth`` units' inputs on the copy
        stream so their transfer overlaps this unit's D2H + the next FFT.
        Exactly one FFT is in flight (the compute stream is serial and the
        recon semaphore is 1).

        Timing uses CUDA events (not ``synchronize``) so the copy stream is
        free to run ahead of compute — a per-tile device barrier would defeat
        the overlap. ``io_s`` is the copy-event→fft-start gap (≈0 when the H2D
        fully overlapped the prior FFT), ``fft_s`` the batched FFT, ``d2h_s``
        the GPU→CPU + RDMA wrap.
        """
        import time as _t

        import torch

        tiles_by_id = self._tiles_by_id
        compute = torch.cuda.current_stream(self.gpu_idx)
        unit_size = len(tile_ids)

        if self._rs_first is None:
            self._rs_first = _t.monotonic()

        # Maintain a bounded look-ahead of gpu_depth units beyond what's been
        # consumed: stage up to (consumed + (gpu_depth+1)*unit_size). On the
        # first call this primes the current unit + gpu_depth ahead; on later
        # calls the current unit is already staged (from the prior call's
        # fill) and this refills the window by one unit. The max_staged size
        # gate caps _gpu_staged at (gpu_depth+1) units regardless of whether
        # units are consumed in _assigned_order order, so device buffers can't
        # accumulate under out-of-order delivery.
        window = (self._gpu_depth + 1) * unit_size
        self._stage_fill_to(self._consumed + window, window)

        # Consume each tile's staged device tensor; the compute stream waits the
        # tile's H2D event so the FFT never reads an incomplete buffer. Stage
        # misses fall back to a synchronous load (event None).
        ev_io_start = torch.cuda.Event(enable_timing=True)
        ev_io_start.record(compute)
        zyx_list = []
        for tid in tile_ids:
            gpu, evt = self._take_staged(tid, tiles_by_id[tid])
            if evt is not None:
                compute.wait_event(evt)
            zyx_list.append(gpu)
        batch = torch.stack(zyx_list, dim=0)  # (B, Z, Y, X)
        ev_io_done = torch.cuda.Event(enable_timing=True)
        ev_io_done.record(compute)

        self._consumed += unit_size

        recon_fn = self._get_compiled_recon()
        recons = recon_fn(batch)  # (B, Z, Y, X)
        ev_fft_done = torch.cuda.Event(enable_timing=True)
        ev_fft_done.record(compute)

        # D2H + RDMA wrap. ``.cpu()`` blocks the host until the compute stream's
        # FFT (and the D2H) complete — that's the single sync point per unit.
        # The next units' input H2Ds were already queued on the copy stream (by
        # the _stage_fill_to at the top) and keep running concurrently with this
        # FFT + D2H, so the next unit's input is ready when its FFT starts.
        handles: list[TileHandle] = []
        for i, tid in enumerate(tile_ids):
            recon_cpu = (
                recons[i].unsqueeze(0).to(torch.float32).contiguous().detach().cpu()
            )
            handles.append(self._store_recon(tid, recon_cpu))
        ev_d2h_done = torch.cuda.Event(enable_timing=True)
        ev_d2h_done.record(compute)
        ev_d2h_done.synchronize()  # ensure events below are valid to query

        self._rs_n += unit_size
        # elapsed_time is in ms → seconds.
        self._rs_io_s += ev_io_start.elapsed_time(ev_io_done) / 1000.0
        self._rs_fft_s += ev_io_done.elapsed_time(ev_fft_done) / 1000.0
        self._rs_d2h_s += ev_fft_done.elapsed_time(ev_d2h_done) / 1000.0
        self._rs_last = _t.monotonic()
        return handles

    @endpoint
    async def reconstruct(self, tile_id: int) -> TileHandle:
        """Reconstruct one tile via a worker thread so the event loop stays free.

        The semaphore caps in-actor recon parallelism at 1 so we don't OOM
        stacking Tikhonov intermediates on the GPU.
        """
        import asyncio

        if self._recon_sem is None:
            self._recon_sem = asyncio.Semaphore(self._cfg.recon_concurrency)
        async with self._recon_sem:
            return await asyncio.to_thread(self._reconstruct_blocking, tile_id)

    @endpoint
    async def reconstruct_batch(self, tile_ids: list[int]) -> list[TileHandle]:
        """Reconstruct B same-shape tiles in one batched call.

        Same semaphore as ``reconstruct`` — one batched FFT on the GPU at
        a time per actor. Returns one TileHandle per input tile, in order.
        """
        import asyncio

        if self._recon_sem is None:
            self._recon_sem = asyncio.Semaphore(self._cfg.recon_concurrency)
        async with self._recon_sem:
            return await asyncio.to_thread(
                self._reconstruct_batch_blocking, tile_ids
            )

    @endpoint
    async def recon_stats(self) -> dict:
        """Per-actor Stage A timing: tile count, IO/FFT/D2H split, span.

        ``busy_s`` is summed per-tile work; ``span_s`` is wall from first
        recon start to last recon end. ``util = busy/span`` near 1.0 means
        the actor was saturated (IO- or compute-bound); well below 1.0
        means it idled between tiles (controller-/dispatch-bound).
        """
        import socket

        busy = self._rs_io_s + self._rs_fft_s + self._rs_d2h_s
        span = (
            (self._rs_last - self._rs_first)
            if (self._rs_first is not None and self._rs_last is not None)
            else 0.0
        )
        return {
            "host": socket.gethostname(),
            "gpu_idx": self.gpu_idx,
            "n_tiles": self._rs_n,
            "io_s": round(self._rs_io_s, 2),
            "fft_s": round(self._rs_fft_s, 2),
            "d2h_s": round(self._rs_d2h_s, 2),
            "busy_s": round(busy, 2),
            "span_s": round(span, 2),
            "util": round(busy / span, 3) if span > 0 else 0.0,
        }

    @endpoint
    async def swap_to(self, plan_path: str) -> dict:
        """Switch this actor to a new plan: release prior volume, reload.

        Used by the multi-TP loop: each TP has its own plan (different
        ``timepoint``). The TF cache and compiled recon survive the swap
        (same modality + shape across TPs), so only the resident volume
        is rebuilt.
        """
        import ctypes
        import gc

        import torch

        from biahub.tile_stitch.plan import load_plan

        # Release the prior volume and any cached recons before reloading.
        # ``self.recons`` holds ~200 CPU recon tensors (+ their RDMABuffer
        # ibverbs registrations) per TP; without an explicit release +
        # malloc_trim, freed host pages are retained by glibc and RSS grows
        # ~100 GB/TP across the persistent actor → OOM after a few TPs.
        # Stop the prior TP's prefetch reader (it references the old plan's
        # timepoint/zarr). The next TP re-primes via ``prime_reader``.
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        # Drop any GPU-staged inputs from the prior TP (they hold device buffers
        # + pinned host buffers); the next TP re-primes the stager via
        # prime_reader. Clearing here releases the HBM before empty_cache.
        self._gpu_staged.clear()
        self._volume_gpu = None
        self.recons.clear()
        torch.cuda.empty_cache()
        gc.collect()
        try:
            # Return freed heap pages to the OS (glibc retains them otherwise).
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        self.plan = load_plan(plan_path)
        self.plan_path = plan_path
        # Config is identical across TPs, but re-read from the new plan so the
        # actor never carries a stale reference (and an older plan without
        # config falls back to defaults rather than crashing).
        self._cfg = self.plan.monarch or MonarchConfig()
        self._reset_recon_stats()
        if not self._stream_tiles:
            self._volume_gpu = self._load_volume_to_gpu()
        import resource

        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        return {
            "gpu_idx": self.gpu_idx,
            "timepoint": self.plan.timepoint,
            "stream_tiles": self._stream_tiles,
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "host_maxrss_gb": round(rss_gb, 1),
        }

    @endpoint
    async def forget(self, tile_ids: list[int]) -> int:
        """Drop the cached recon tensors for ``tile_ids``. Returns count freed."""
        n = 0
        for tid in tile_ids:
            if tid in self.recons:
                del self.recons[tid]
                n += 1
        return n

    @endpoint
    async def stitch(self, out_tile_id: int, contributors: dict) -> dict:
        """Blend contributor tiles (pulled via RDMA) and write the output chunk.

        Async to allow Monarch to interleave multiple stitch calls on the same
        actor: while one stitch is awaiting RDMA pulls or a zarr write, the
        actor's event loop can advance another stitch's CPU blend. Net effect:
        I/O and CPU work overlap within a single actor.
        """
        import asyncio
        import time

        import numpy as np
        import torch
        import zarr

        if not contributors:
            return {"out_tile_id": out_tile_id, "n_inputs": 0, "wall_s": 0.0}

        t_start = time.monotonic()

        # Local-fast-path: if this actor produced the contributor, skip the
        # RDMA round-trip and use the cached tensor directly. Remote
        # contributors are pulled via ``RDMABuffer.read_into`` issued
        # concurrently (asyncio.gather). We use the low-level call rather
        # than ``RDMAAction`` because the latter hard-codes the 3s default
        # timeout — too short for cross-node IB pulls under Stage B load.
        # ``monarch.rdma_timeout_s`` controls it.
        rdma_timeout = self._cfg.rdma_timeout_s
        contribs_np: dict[int, np.ndarray] = {}
        remote_pulls: list[tuple[int, torch.Tensor]] = []
        pull_futs = []
        for tid, handle in contributors.items():
            local_t = self.recons.get(tid)
            if local_t is not None:
                contribs_np[tid] = local_t.numpy()
                continue
            dst = torch.empty(
                handle.shape,
                dtype=getattr(torch, handle.dtype_name.split(".")[-1]),
            )
            fut = handle.buffer.read_into(
                dst.view(torch.uint8).flatten(), timeout=rdma_timeout
            )
            pull_futs.append(fut)
            remote_pulls.append((tid, dst))

        if pull_futs:
            async def _await(f):
                return await f

            await asyncio.gather(*[_await(f) for f in pull_futs])
            for tid, dst in remote_pulls:
                contribs_np[tid] = dst.numpy()
        t_rdma = time.monotonic() - t_start

        # Blend + zarr write on a worker thread so the actor's event loop is
        # free to start the next stitch's RDMA pulls while this one blends.
        # Blend stays on CPU (numpy): GPU blend contends with concurrent
        # recon work on the same device (default CUDA stream). All
        # per-output-tile geometry is pre-built in ``_stitch_geom`` so the
        # hot loop is pure numpy + cached kernel views.
        geom = self._stitch_geom[out_tile_id]
        out_spatial = geom["out_spatial"]
        blend_kernel = self._blend_kernel
        kernel_cache = self._kernel_cache
        t_off = self.plan.timepoint
        leading_c = self.plan.leading_shape[1:]
        output_path = self.plan.output_path

        def _blend_and_write() -> dict:
            t_blend_start = time.monotonic()
            # CPU numpy weighted-mean blend (geometry pre-built in
            # ``_stitch_geom``); GPU blend would contend with concurrent
            # recon on the same device.
            result = _core.blend_contributors(
                geom, contribs_np, blend_kernel, kernel_cache
            )
            t_blend = time.monotonic() - t_blend_start

            t_write_start = time.monotonic()
            out_arr = zarr.open_group(output_path, mode="a")["0"]
            # T axis is carried by an explicit slice(t_off, t_off + 1); the
            # remaining leading dims use leading_shape[1:] and spatial dims
            # come from the pre-built geometry. This convention is pinned
            # CPU-side by test_core.test_timepoint_write_region_carries_t_axis.
            write_region = (
                (slice(t_off, t_off + 1),)
                + tuple(slice(0, n) for n in leading_c)
                + tuple(slice(lo, hi) for lo, hi in out_spatial)
            )
            out_arr[write_region] = result[None]
            t_write = time.monotonic() - t_write_start

            return {"t_blend_s": t_blend, "t_write_s": t_write}

        bw = await asyncio.to_thread(_blend_and_write)

        return {
            "out_tile_id": out_tile_id,
            "n_inputs": len(contributors),
            "t_rdma_s": t_rdma,
            "t_blend_s": bw["t_blend_s"],
            "t_write_s": bw["t_write_s"],
            "wall_s": time.monotonic() - t_start,
        }
