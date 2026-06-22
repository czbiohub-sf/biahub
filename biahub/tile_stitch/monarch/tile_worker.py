"""Per-GPU Monarch actor: reconstruct assigned tiles and hold them.

One Actor per GPU. Reconstructs assigned tiles, stores the CPU result on
the actor (keeping it alive), and returns an :class:`RDMABuffer` handle.
Downstream Stage B receives these handles in a dict, ``read_into`` each
via RDMA (ibverbs over IB intra-node, no Monarch mailbox involved for
the bulk data), then blends locally and writes the zarr chunk.

Monarch 0.5 RDMABuffer is CPU-only, so GPU tensors bounce through CPU —
but recon already ends with a GPU→CPU move, so there's no extra cost.
"""

import threading

from typing import Any

from monarch.actor import Actor, current_rank, endpoint
from monarch.rdma import RDMABuffer

from biahub.tile_stitch import _core
from biahub.tile_stitch._core import PrefetchReader
from biahub.tile_stitch.config import MonarchConfig


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
        # The RDMABuffer for each cached recon, kept so we can explicitly
        # ``drop()`` it on free. ``RDMABuffer`` has no ``__del__``: dropping the
        # Python ref does NOT deregister the underlying ibverbs MR, so the
        # recon's host pages stay pinned (and ``malloc_trim`` can't reclaim
        # them). Across a persistent multi-TP actor that leaks ~100 GB/TP →
        # OOM. ``forget``/``swap_to`` drop these explicitly.
        self._rdma_buffers: dict[int, Any] = {}
        # Serialize the GPU-bound recon. Default ``recon_concurrency=1`` —
        # Tikhonov holds ~30 GB on device, so >1 risks OOM. Sized from config
        # at first use; lazy-init avoids binding an asyncio loop here (this
        # ``__init__`` may run before the loop exists in some transport paths).
        self._recon_sem = None
        # Background prefetch reader. Primed per-TP via the ``prime_reader``
        # endpoint with this actor's assigned tile order: it pulls tile N+1's
        # zarr bytes while the GPU runs tile N's FFT.
        self._reader: PrefetchReader | None = None
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
        # D2H via a reused pinned scratch buffer. A pageable ``.cpu()`` DMAs at
        # ~2 GB/s (fresh alloc + page-faults + torch's chunked staging); a
        # PINNED destination DMAs at ~25 GB/s (~11×). We can't pin every held
        # recon (~150 GB at peak → mlocked), so we keep ONE reusable pinned
        # scratch per actor sized to the largest recon seen: GPU→pinned scratch
        # (fast DMA) then scratch→fresh pageable recon (CPU memcpy). Net ~3× on
        # the copy half, pinned footprint bounded to one tile. Lazily grown,
        # guarded by a lock (correct even at recon_concurrency>1), and falls
        # back to pageable ``.cpu()`` once if the pin is refused (memlock).
        self._pinned_scratch: Any = None
        self._d2h_lock = threading.Lock()
        self._pin_failed = False
        # Stored/transmitted recon dtype (compute stays float32). float16 halves
        # both the pinned-copy and the ibverbs-MR bytes; cast GPU-side in _d2h
        # BEFORE the pinned copy (cast-during-copy drops to the pageable rate).
        self._recon_dtype = (
            torch.float16
            if self._cfg.recon_dtype == "float16"
            else torch.float32
        )
        self._reset_recon_stats()

    def _reset_recon_stats(self) -> None:
        """Zero the per-actor Stage A timing counters (call per TP)."""
        self._rs_n = 0  # tiles reconstructed
        self._rs_io_s = 0.0  # zarr read / volume slice + H2D
        self._rs_fft_s = 0.0  # recon_fn (FFT + Tikhonov)
        self._rs_d2h_s = 0.0  # GPU->CPU copy + RDMABuffer setup (total; = copy + rdma)
        self._rs_copy_s = 0.0  # GPU->CPU copy only (the .cpu())
        self._rs_rdma_s = 0.0  # RDMABuffer ibverbs MR registration only
        self._rs_first = None  # monotonic ts of first recon start
        self._rs_last = None  # monotonic ts of last recon end

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
        log = _logging.getLogger("TileWorker.compile")

        cuda_tf, recon_settings = _core.get_tf_cuda(self.plan.settings, device)
        _eager = _core.make_eager_recon(cuda_tf, recon_settings)

        mode = self._cfg.compile_mode.value
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
                self.gpu_idx,
                exc,
            )
            self._compiled_recon = _eager
        return self._compiled_recon

    def _load_tile_gpu(self, tile):
        """Load one input tile to a ``(Z, Y, X)`` float32 GPU tensor.

        Streams the tile from zarr (per-tile read, no resident volume).
        """
        return _core.load_tile_zyx(
            self.plan, tile, volume=None, device=f"cuda:{self.gpu_idx}"
        )

    def _load_one(self, tile_id: int, tile):
        """Load one tile to a ``(Z,Y,X)`` f32 cuda tensor, preferring prefetch.

        Used by both the single-tile and batched recon paths so batched
        recon benefits from the same background read-ahead. Falls back to a
        synchronous load when prefetch is off or missed this tile.
        """
        import torch

        pre = self._reader.get(tile_id) if self._reader is not None else None
        if pre is None:
            return self._load_tile_gpu(tile)
        return torch.as_tensor(pre, dtype=torch.float32, device=f"cuda:{self.gpu_idx}")

    def _d2h(self, gpu_tensor):
        """GPU→host copy of one recon via a reused pinned scratch.

        ``gpu_tensor`` is the (already unsqueezed) device recon. Casts to
        contiguous float32 on the GPU, DMAs it into the per-actor pinned
        scratch (fast — pinned dest is a direct DMA), then memcpys out into a
        fresh PAGEABLE tensor that becomes the held recon (RDMA-registered by
        ``_store_recon``). The scratch is reused across tiles/work-units and
        grown only when a larger recon appears. Falls back to a plain pageable
        ``.cpu()`` if pinning is refused (e.g. a low ``memlock`` ulimit).
        """
        import torch

        # Cast to the stored dtype ON THE GPU first (float16 halves the bytes
        # the copy + ibverbs MR move). Casting during the host copy instead
        # would fall off the fast pinned path — keep the copy a same-dtype DMA.
        dtype = self._recon_dtype
        g = gpu_tensor.to(dtype).contiguous().detach()
        if self._pin_failed:
            return g.cpu()

        shape = tuple(g.shape)
        numel = g.numel()
        with self._d2h_lock:
            scratch = self._pinned_scratch
            # Scratch is sized in BYTES (uint8) so one buffer serves any dtype.
            need_bytes = numel * g.element_size()
            if scratch is None or scratch.numel() < need_bytes:
                try:
                    scratch = torch.empty(need_bytes, dtype=torch.uint8, pin_memory=True)
                except RuntimeError as exc:  # pinning refused (memlock ulimit)
                    import logging as _logging

                    self._pin_failed = True
                    _logging.getLogger("TileWorker.d2h").warning(
                        "pinned D2H scratch alloc failed (%s); "
                        "falling back to pageable .cpu()",
                        exc,
                    )
                    return g.cpu()
                self._pinned_scratch = scratch
            view = scratch[:need_bytes].view(dtype).view(shape)
            view.copy_(g)  # GPU → pinned scratch (fast DMA), blocks until done
            recon_cpu = torch.empty(shape, dtype=dtype)  # pageable, held
            recon_cpu.copy_(view)  # pinned scratch → pageable (CPU memcpy)
        return recon_cpu

    def _store_recon(self, tile_id: int, recon_cpu) -> TileHandle:
        """Keep the CPU tensor alive in ``self.recons`` and wrap an RDMABuffer."""
        import torch

        self.recons[tile_id] = recon_cpu
        flat = recon_cpu.view(torch.uint8).flatten()
        buf = RDMABuffer(flat)
        # Retain the buffer so ``forget``/``swap_to`` can ``drop()`` it (the MR
        # outlives the Python ref otherwise — see ``self._rdma_buffers``).
        self._rdma_buffers[tile_id] = buf
        return TileHandle(
            buffer=buf,
            shape=tuple(recon_cpu.shape),
            dtype_name=str(recon_cpu.dtype),
        )

    @endpoint
    async def prime_reader(self, tile_ids: list[int]) -> dict:
        """Start a background reader over this actor's assigned tile order.

        Called by the driver once per TP with ``input_order[g::n_gpus]`` —
        the contiguous sequence this actor (gpu ``g``) will reconstruct. The
        reader pulls the next tile's zarr bytes while the GPU runs the
        current tile's FFT. No-op when the effective prefetch depth is 0.
        """
        # Tear down any reader from the previous TP first.
        if self._reader is not None:
            self._reader.stop()
            self._reader = None

        depth = self._cfg.effective_prefetch_depth
        if depth <= 0 or not tile_ids:
            return {"gpu_idx": self.gpu_idx, "prefetch": False, "depth": depth}

        # Bind a per-tile loader over this actor's current plan. The loader
        # returns a source-dtype ``(Z, Y, X)`` numpy array and the recon path
        # casts to float32 during H2D (mirroring ``_load_tile_gpu``).
        tiles_by_id = self._tiles_by_id
        plan = self.plan

        def _load(tid: int):
            return _core.read_tile_block(plan, tiles_by_id[tid])

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

        return self._reconstruct_sync(tile_ids)

    def _reconstruct_sync(self, tile_ids: list[int]) -> list[TileHandle]:
        """Run synchronous recon for one work-unit.

        Loads each tile (prefetch or zarr), stacks into ``(B, Z, Y, X)`` (B may
        be 1), runs the batched FFT, copies each result to CPU, and wraps an
        RDMABuffer. Uses ``synchronize``-based io/fft/d2h timing.
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
        copy_s = 0.0
        rdma_s = 0.0
        for i, tid in enumerate(tile_ids):
            tc = _t.monotonic()
            recon_cpu = self._d2h(recons[i].unsqueeze(0))
            tr = _t.monotonic()
            handles.append(self._store_recon(tid, recon_cpu))
            copy_s += tr - tc
            rdma_s += _t.monotonic() - tr
        t_end = _t.monotonic()

        self._rs_n += len(tile_ids)
        self._rs_io_s += t_io - t0
        self._rs_fft_s += t_fft - t_io
        self._rs_d2h_s += t_end - t_fft
        self._rs_copy_s += copy_s
        self._rs_rdma_s += rdma_s
        self._rs_last = t_end
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
            return await asyncio.to_thread(self._reconstruct_batch_blocking, tile_ids)

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
            "copy_s": round(self._rs_copy_s, 2),  # GPU->CPU copy portion of d2h
            "rdma_s": round(self._rs_rdma_s, 2),  # RDMABuffer registration portion
            "busy_s": round(busy, 2),
            "span_s": round(span, 2),
            "util": round(busy / span, 3) if span > 0 else 0.0,
        }

    @endpoint
    async def swap_to(self, plan_path: str) -> dict:
        """Switch this actor to a new plan: release prior recons, reload.

        Used by the multi-TP loop: each TP has its own plan (different
        ``timepoint``). The TF cache and compiled recon survive the swap
        (same modality + shape across TPs).
        """
        import ctypes
        import gc

        import torch

        from biahub.tile_stitch.plan import load_plan

        # Release any cached recons before reloading.
        # ``self.recons`` holds ~200 CPU recon tensors (+ their RDMABuffer
        # ibverbs registrations) per TP; without an explicit release +
        # malloc_trim, freed host pages are retained by glibc and RSS grows
        # ~100 GB/TP across the persistent actor → OOM after a few TPs.
        # Stop the prior TP's prefetch reader (it references the old plan's
        # timepoint/zarr). The next TP re-primes via ``prime_reader``.
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        # Deregister any outstanding RDMABuffers before clearing recons.
        # Refcount-free ``forget`` drops them as outputs stitch, so usually
        # none remain — this is the backstop for a TP that ended early.
        await self._drop_buffers(list(self._rdma_buffers.keys()))
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
        import os
        import resource

        maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        # Current RSS (post clear/drop/malloc_trim) — unlike ru_maxrss (a
        # monotonic peak), this falls when memory is actually released, so it's
        # the signal for whether recon host memory is freed between TPs.
        with open("/proc/self/statm") as _f:
            rss_pages = int(_f.read().split()[1])
        cur_rss_gb = rss_pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
        return {
            "gpu_idx": self.gpu_idx,
            "timepoint": self.plan.timepoint,
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "host_maxrss_gb": round(maxrss_gb, 1),
            "host_rss_gb": round(cur_rss_gb, 1),
        }

    async def _drop_buffers(self, tile_ids: list[int]) -> None:
        """Deregister the RDMABuffers for ``tile_ids`` (releases pinned host
        pages). ``RDMABuffer`` has no ``__del__``, so this explicit ``drop()``
        is what actually frees the ibverbs MR; without it RSS grows per TP."""
        for tid in tile_ids:
            buf = self._rdma_buffers.pop(tid, None)
            if buf is None:
                continue
            try:
                await buf.drop()
            except Exception:
                pass

    @endpoint
    async def forget(self, tile_ids: list[int]) -> int:
        """Free cached recons for ``tile_ids``: drop their RDMABuffers (the MR
        keeps the host pages pinned otherwise) and the cached tensors. Returns
        the count of tensors freed."""
        await self._drop_buffers(tile_ids)
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
        out_c_idx = self.plan.output_channel_index
        output_path = self.plan.output_path

        def _blend_and_write() -> dict:
            t_blend_start = time.monotonic()
            # CPU numpy weighted-mean blend (geometry pre-built in
            # ``_stitch_geom``); GPU blend would contend with concurrent
            # recon on the same device.
            result = _core.blend_contributors(geom, contribs_np, blend_kernel, kernel_cache)
            t_blend = time.monotonic() - t_blend_start

            t_write_start = time.monotonic()
            out_arr = zarr.open_group(output_path, mode="a")["0"]
            # T axis carried by an explicit slice(t_off, t_off + 1); C axis by
            # this channel's slot slice(out_c_idx, out_c_idx + 1) in the (possibly
            # multi-channel) shared output; spatial dims from the pre-built
            # geometry. Pinned CPU-side by
            # test_core.test_timepoint_write_region_carries_t_axis.
            write_region = (
                (slice(t_off, t_off + 1),)
                + (slice(out_c_idx, out_c_idx + 1),)
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
