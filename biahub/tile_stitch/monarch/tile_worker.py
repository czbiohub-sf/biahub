"""Per-GPU Monarch actor for reconstruction and output stitching.

Reconstructed CPU tensors remain actor-owned while peers pull their registered
buffers via RDMA. The actor then blends available contributors and writes each
output chunk. Monarch's CPU-only RDMABuffer matches the reconstruction path's
existing device-to-host boundary.
"""

import threading

from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_rank, endpoint
from monarch.rdma import RDMAAction, RDMABuffer

from biahub.tile_stitch import _core, _nvtx
from biahub.tile_stitch.dataflow import (
    OutputCapture,
    ReconstructionLoader,
    read_zarr_tile,
    write_capture,
)
from biahub.tile_stitch.plan import StitchWorkUnit, load_program
from biahub.tile_stitch.transfer import get_transfer_tensors

_NO_PREFETCH = object()


class TileHandle:
    """Serializable reference to an actor-owned reconstruction.

    Parameters
    ----------
    buffer : RDMABuffer
        Registered byte buffer containing the reconstruction.
    shape : tuple[int, ...]
        Tensor shape restored by a consumer.
    dtype_name : str
        Torch dtype name restored by a consumer.
    """

    __slots__ = ("buffer", "shape", "dtype_name")

    def __init__(self, buffer: RDMABuffer, shape: tuple[int, ...], dtype_name: str):
        self.buffer = buffer
        self.shape = shape
        self.dtype_name = dtype_name


def _rdma_backend_name(buffer: RDMABuffer) -> str:
    """Report the routed backend; Monarch's availability property ignores disable."""
    import monarch

    if bool(monarch.get_global_config().get("rdma_disable_ibverbs")):
        return "tcp"
    return str(buffer.backend)


async def _pull_contributors(
    local_recons: dict[int, Any],
    contributors: dict[int, TileHandle],
    *,
    timeout_s: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Resolve local contributors and batch all remote reads for one output."""
    import torch

    arrays: dict[int, Any] = {}
    remote: list[tuple[int, torch.Tensor]] = []
    backends: set[str] = set()
    remote_bytes = 0
    action: RDMAAction | None = None

    for tile_id, handle in contributors.items():
        local = local_recons.get(tile_id)
        if local is not None:
            arrays[tile_id] = local.numpy()
            continue

        destination = torch.empty(
            handle.shape,
            dtype=getattr(torch, handle.dtype_name.split(".")[-1]),
        )
        destination_bytes = destination.view(torch.uint8).flatten()
        if action is None:
            action = RDMAAction()
        action.read_remote(destination_bytes, handle.buffer)
        remote.append((tile_id, destination))
        remote_bytes += int(destination_bytes.numel())
        backends.add(_rdma_backend_name(handle.buffer))

    if action is not None:
        await action.submit(timeout=timeout_s)
        for tile_id, destination in remote:
            arrays[tile_id] = destination.numpy()

    return arrays, {
        "rdma_backend": ",".join(sorted(backends)) if backends else "local",
        "rdma_batches": int(action is not None),
        "rdma_ops": len(remote),
        "rdma_bytes": remote_bytes,
    }


class TileWorker(Actor):
    """Reconstruct and stitch tiles on one GPU actor.

    Parameters
    ----------
    program_path : str
        Serialized static :class:`~biahub.tile_stitch.plan.StitchProgram`.
    work : StitchWorkUnit
        Initial position and timepoint binding.
    """

    def __supervise__(self, failure) -> bool:
        """Report an actor failure to the controller.

        Parameters
        ----------
        failure : object
            Monarch supervision failure report.

        Returns
        -------
        bool
            Always ``False`` so the failure propagates.
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

    def __init__(self, program_path: str, work: StitchWorkUnit):
        """Initialize the actor on its assigned GPU.

        Parameters
        ----------
        program_path : str
            Serialized static stitch program.
        work : StitchWorkUnit
            Initial input and output binding.
        """
        import torch

        rank = current_rank()
        try:
            gpu_idx = rank["gpus"]
        except (KeyError, TypeError):
            gpu_idx = int(rank)
        torch.cuda.set_device(gpu_idx)
        self.gpu_idx = gpu_idx
        self.program = load_program(program_path)
        self.work = work
        self._cfg = self.program.monarch
        # Keep tensors alive for as long as their RDMA handles are valid.
        self.recons: dict[int, torch.Tensor] = {}
        # RDMABuffer has no finalizer; explicit ``drop`` is required to
        # deregister its memory region and release pinned host pages.
        self._rdma_buffers: dict[int, Any] = {}
        # Serialize memory-heavy reconstruction; initialize against the actor's
        # running event loop rather than the construction transport.
        self._recon_sem = None
        # Each actor owns a bounded, process-local source reader.
        self._loader: ReconstructionLoader[Any] | None = None
        # Built lazily because torch.compile captures after warm-up calls.
        self._compiled_recon = None
        self._blend_kernel = self.program.settings.blend.build()
        self._tiles_by_id = {tile.tile_id: tile for tile in self.program.input_tiles}
        self._stitch_geom: dict[int, dict] = _core.build_stitch_geom(self.program)
        self._kernel_cache: dict[tuple, Any] = {}
        self._write_output = self._make_output_writer()
        # Reuse one pinned D2H scratch buffer, then copy into pageable tensors
        # that can remain resident. Pinning every held recon would exhaust the
        # actor's locked-memory allowance.
        self._pinned_scratch: Any = None
        self._d2h_lock = threading.Lock()
        self._pin_failed = False
        # Reconstruction computes in float32; storage may use float16.
        self._recon_dtype = (
            torch.float16 if self._cfg.recon_dtype == "float16" else torch.float32
        )
        self._reset_recon_stats()

    def _reset_recon_stats(self) -> None:
        """Reset per-timepoint Stage A timing counters."""
        self._rs_n = 0
        self._rs_batches = 0
        self._rs_h2d_bytes = 0
        self._rs_resident_bytes = 0
        self._rs_resident_peak_bytes = 0
        self._rs_resident_peak_tiles = 0
        self._rs_io_s = 0.0
        self._rs_fft_s = 0.0
        self._rs_d2h_s = 0.0
        self._rs_copy_s = 0.0
        self._rs_rdma_s = 0.0
        self._rs_first = None
        self._rs_last = None

    def _get_compiled_recon(self):
        """Return the cached compiled reconstruction callable.

        Returns
        -------
        Callable
            Compiled or eager function mapping a source batch to reconstructions.

        Notes
        -----
        Compilation binds transfer functions and inverse settings into a
        single-input graph. Configuration or compile failure selects eager mode.
        """
        if self._compiled_recon is not None:
            return self._compiled_recon

        import logging as _logging

        import torch

        device = f"cuda:{self.gpu_idx}"
        log = _logging.getLogger("TileWorker.compile")

        cuda_tf, recon_settings = get_transfer_tensors(self.program.settings, device)
        eager = _core.make_eager_recon(cuda_tf, recon_settings)

        mode = self._cfg.compile_mode.value
        if mode == "none":
            log.info("gpu_idx=%d compile disabled (monarch.compile_mode=none)", self.gpu_idx)
            self._compiled_recon = eager
            return eager
        try:
            compiled = torch.compile(eager, mode=mode, dynamic=False)
            log.info("gpu_idx=%d torch.compile mode=%s ready", self.gpu_idx, mode)
            self._compiled_recon = compiled
        except Exception as exc:
            log.warning(
                "gpu_idx=%d torch.compile failed (%s) — falling back to eager",
                self.gpu_idx,
                exc,
            )
            self._compiled_recon = eager
        return self._compiled_recon

    def _load_one(self, tile_id: int, tile, prefetched=_NO_PREFETCH):
        """Move one loader-owned source tile to this GPU."""
        import torch

        del tile
        value = (
            self._loader.get(tile_id)
            if prefetched is _NO_PREFETCH and self._loader is not None
            else prefetched
        )
        if value is _NO_PREFETCH:
            raise RuntimeError("reconstruction loader was not primed")
        if value is None:
            raise RuntimeError(f"source read failed for tile {tile_id}")
        self._rs_h2d_bytes += int(value.nbytes)
        return torch.as_tensor(value, device=f"cuda:{self.gpu_idx}").to(torch.float32)

    def _make_output_writer(self):
        """Bind the active work unit's output write callable."""
        from iohub.ngff import open_ome_zarr

        output_path = self.work.output_path

        def write(region, payload) -> None:
            with open_ome_zarr(output_path, layout="fov", mode="a") as output:
                output.data[region] = payload

        return write

    def _d2h(self, gpu_tensor):
        """Copy one reconstruction through a reusable pinned host buffer.

        Parameters
        ----------
        gpu_tensor : torch.Tensor
            Device-resident reconstruction.

        Returns
        -------
        torch.Tensor
            Pageable CPU tensor retained by the actor.

        Notes
        -----
        Only the reusable scratch remains pinned. Allocation failure falls back
        to ``Tensor.cpu``.
        """
        import torch

        # Convert on-device so the host transfer remains a same-dtype DMA.
        dtype = self._recon_dtype
        g = gpu_tensor.to(dtype).contiguous().detach()
        if self._pin_failed:
            return g.cpu()

        shape = tuple(g.shape)
        numel = g.numel()
        with self._d2h_lock:
            scratch = self._pinned_scratch
            # A byte buffer can be reinterpreted for either stored dtype.
            need_bytes = numel * g.element_size()
            if scratch is None or scratch.numel() < need_bytes:
                try:
                    scratch = torch.empty(need_bytes, dtype=torch.uint8, pin_memory=True)
                except RuntimeError as exc:
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
            view.copy_(g)
            recon_cpu = torch.empty(shape, dtype=dtype)
            recon_cpu.copy_(view)
        return recon_cpu

    def _store_recon(self, tile_id: int, recon_cpu) -> TileHandle:
        """Register and retain one CPU reconstruction.

        Parameters
        ----------
        tile_id : int
            Input tile identifier.
        recon_cpu : torch.Tensor
            Pageable CPU reconstruction.

        Returns
        -------
        TileHandle
            Serializable metadata and registered RDMA buffer.
        """
        import torch

        self.recons[tile_id] = recon_cpu
        self._rs_resident_bytes += recon_cpu.nbytes
        self._rs_resident_peak_bytes = max(
            self._rs_resident_peak_bytes, self._rs_resident_bytes
        )
        self._rs_resident_peak_tiles = max(self._rs_resident_peak_tiles, len(self.recons))
        flat = recon_cpu.view(torch.uint8).flatten()
        buf = RDMABuffer(flat)
        # Retention permits explicit deregistration during ``forget`` or swap.
        self._rdma_buffers[tile_id] = buf
        return TileHandle(
            buffer=buf,
            shape=tuple(recon_cpu.shape),
            dtype_name=str(recon_cpu.dtype),
        )

    @endpoint
    async def prime_loader(self, tile_ids: list[int]) -> dict:
        """Start bounded source read-ahead for this actor.

        Parameters
        ----------
        tile_ids : list[int]
            Disjoint reconstruction order assigned to the actor.

        Returns
        -------
        dict
            Effective prefetch state and worker counts.
        """
        if self._loader is not None:
            self._loader.close()
            self._loader = None

        depth = self._cfg.effective_prefetch_depth
        if not tile_ids:
            return {"gpu_idx": self.gpu_idx, "prefetch": False, "depth": depth}

        tiles_by_id = self._tiles_by_id
        program = self.program
        work = self.work
        self._loader = ReconstructionLoader(
            lambda tile_id: read_zarr_tile(
                program,
                work,
                tiles_by_id[tile_id],
            ),
            tile_ids,
            depth,
            num_workers=self._cfg.prefetch_workers,
            read_timeout_s=self._cfg.read_timeout_s,
            retries=self._cfg.read_retries,
        )
        return {
            "gpu_idx": self.gpu_idx,
            "prefetch": True,
            "depth": depth,
            "workers": min(self._cfg.prefetch_workers, depth),
            "n_assigned": len(tile_ids),
        }

    def _reconstruct_blocking(self, tile_id: int) -> TileHandle:
        """Reconstruct one tile synchronously.

        Parameters
        ----------
        tile_id : int
            Input tile identifier.

        Returns
        -------
        TileHandle
            Registered actor-owned reconstruction.
        """
        import torch

        # ``set_device`` is thread-local; the asyncio.to_thread worker
        # inherits cuda:0 by default. Re-pin so any ``"cuda"`` defaults
        # inside waveorder resolve to the right device.
        torch.cuda.set_device(self.gpu_idx)

        return self._reconstruct_sync([tile_id])[0]

    def _reconstruct_batch_blocking(self, tile_ids: list[int]) -> list[TileHandle]:
        """Reconstruct one same-shape batch synchronously.

        Parameters
        ----------
        tile_ids : list[int]
            Input tile identifiers sharing one spatial shape.

        Returns
        -------
        list[TileHandle]
            Handles aligned with ``tile_ids``.
        """
        import torch

        torch.cuda.set_device(self.gpu_idx)

        return self._reconstruct_sync(tile_ids)

    def _reconstruct_sync(self, tile_ids: list[int]) -> list[TileHandle]:
        """Execute one synchronous reconstruction work unit.

        Parameters
        ----------
        tile_ids : list[int]
            Same-shape tile IDs in batch order.

        Returns
        -------
        list[TileHandle]
            Registered reconstruction handles in input order.
        """
        import time as _t

        import torch

        tiles_by_id = self._tiles_by_id

        t0 = _t.monotonic()
        h2d_bytes_before = self._rs_h2d_bytes
        if self._rs_first is None:
            torch.cuda.reset_peak_memory_stats(self.gpu_idx)
            self._rs_first = t0
        with _nvtx.stage(f"load b={len(tile_ids)}", "cyan"):
            prefetched = (
                self._loader.get_batch(tile_ids)
                if self._loader is not None
                else (_NO_PREFETCH,) * len(tile_ids)
            )
            zyx_list = [
                self._load_one(tile_id, tiles_by_id[tile_id], prefetched_value)
                for tile_id, prefetched_value in zip(tile_ids, prefetched, strict=True)
            ]
            batch = torch.stack(zyx_list, dim=0)  # (B, Z, Y, X)
            torch.cuda.synchronize(self.gpu_idx)
        t_io = _t.monotonic()
        _nvtx.counter("bytes_h2d", unit="bytes").sample(self._rs_h2d_bytes - h2d_bytes_before)
        self._rs_batches += 1

        recon_fn = self._get_compiled_recon()
        with _nvtx.stage(f"recon_fft b={len(tile_ids)}", "green"):
            recons = recon_fn(batch)  # (B, Z, Y, X)
            torch.cuda.synchronize(self.gpu_idx)
        t_fft = _t.monotonic()
        _nvtx.counter("gpu_mem_mb").sample(
            int(torch.cuda.memory_allocated(self.gpu_idx) / 1e6)
        )

        handles: list[TileHandle] = []
        copy_s = 0.0
        rdma_s = 0.0
        for i, tid in enumerate(tile_ids):
            tc = _t.monotonic()
            with _nvtx.stage("d2h", "orange"):
                recon_cpu = self._d2h(recons[i].unsqueeze(0))
            tr = _t.monotonic()
            _nvtx.counter("bytes_d2h", unit="bytes").sample(recon_cpu.nbytes)
            with _nvtx.stage("rdma", "red"):
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

    @concurrent_endpoint
    async def reconstruct(self, tile_id: int) -> TileHandle:
        """Reconstruct one tile under the actor GPU semaphore.

        Parameters
        ----------
        tile_id : int
            Input tile identifier.

        Returns
        -------
        TileHandle
            Registered actor-owned reconstruction.
        """
        import asyncio

        if self._recon_sem is None:
            self._recon_sem = asyncio.Semaphore(self._cfg.recon_concurrency)
        async with self._recon_sem:
            return await asyncio.to_thread(self._reconstruct_blocking, tile_id)

    @concurrent_endpoint
    async def reconstruct_batch(self, tile_ids: list[int]) -> list[TileHandle]:
        """Reconstruct a same-shape batch under the actor GPU semaphore.

        Parameters
        ----------
        tile_ids : list[int]
            Same-shape input tile IDs.

        Returns
        -------
        list[TileHandle]
            Handles aligned with ``tile_ids``.
        """
        import asyncio

        if self._recon_sem is None:
            self._recon_sem = asyncio.Semaphore(self._cfg.recon_concurrency)
        async with self._recon_sem:
            return await asyncio.to_thread(self._reconstruct_batch_blocking, tile_ids)

    @endpoint
    async def recon_stats(self) -> dict:
        """Return Stage A telemetry for this actor.

        Returns
        -------
        dict
            Host, GPU, tile count, timing split, utilization, and loader state.

        Notes
        -----
        ``busy_s`` sums active work, while ``span_s`` measures wall time from
        the first reconstruction start to the last completion.
        """
        import os
        import resource
        import socket

        import torch

        busy = self._rs_io_s + self._rs_fft_s + self._rs_d2h_s
        span = (
            (self._rs_last - self._rs_first)
            if (self._rs_first is not None and self._rs_last is not None)
            else 0.0
        )
        with open("/proc/self/statm") as statm:
            rss_pages = int(statm.read().split()[1])
        result = {
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
            "n_batches": self._rs_batches,
            "h2d_bytes": self._rs_h2d_bytes,
            "hbm_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.gpu_idx),
            "host_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "host_rss_bytes": rss_pages * os.sysconf("SC_PAGE_SIZE"),
            "resident_peak_tiles": self._rs_resident_peak_tiles,
            "resident_peak_bytes": self._rs_resident_peak_bytes,
        }
        if self._loader is not None:
            result["loader"] = self._loader.snapshot()
        return result

    @endpoint
    async def bind_work_unit(self, work: StitchWorkUnit) -> dict:
        """Release unit-owned state and bind another input/output pair.

        Parameters
        ----------
        work : StitchWorkUnit
            Position and timepoint binding for the next execution.

        Returns
        -------
        dict
            Actor identity and post-release host/device memory telemetry.
        """
        import ctypes
        import gc

        import torch

        if self._loader is not None:
            self._loader.close()
            self._loader = None
        # Usually empty after normal refcount release; required after early exit.
        await self._drop_buffers(list(self._rdma_buffers.keys()))
        self.recons.clear()
        torch.cuda.empty_cache()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        self.work = work
        self._write_output = self._make_output_writer()
        self._reset_recon_stats()
        import os
        import resource

        maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        # Unlike ru_maxrss, current RSS verifies that swap released host pages.
        with open("/proc/self/statm") as _f:
            rss_pages = int(_f.read().split()[1])
        cur_rss_gb = rss_pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
        return {
            "gpu_idx": self.gpu_idx,
            "timepoint": self.work.timepoint,
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "host_maxrss_gb": round(maxrss_gb, 1),
            "host_rss_gb": round(cur_rss_gb, 1),
        }

    async def _drop_buffers(self, tile_ids: list[int]) -> None:
        """Deregister selected reconstruction buffers.

        Parameters
        ----------
        tile_ids : list[int]
            Input tile IDs whose memory registrations should be dropped.
        """
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
        """Release selected buffers and cached reconstructions.

        Parameters
        ----------
        tile_ids : list[int]
            Input tile IDs to release.

        Returns
        -------
        int
            Number of cached tensors removed.
        """
        await self._drop_buffers(tile_ids)
        n = 0
        for tid in tile_ids:
            if tid in self.recons:
                self._rs_resident_bytes -= self.recons[tid].nbytes
                del self.recons[tid]
                n += 1
        return n

    @concurrent_endpoint
    async def stitch(self, out_tile_id: int, contributors: dict) -> dict:
        """Blend contributors and write one output chunk.

        Parameters
        ----------
        out_tile_id : int
            Output tile identifier.
        contributors : dict[int, TileHandle]
            Available reconstruction handles keyed by input tile ID.

        Returns
        -------
        dict
            Contributor count and RDMA, blend, write, and wall timings.

        Notes
        -----
        Awaiting RDMA and offloading CPU work allows other actor calls to
        advance.
        """
        import asyncio
        import time

        if not contributors:
            return {"out_tile_id": out_tile_id, "n_inputs": 0, "wall_s": 0.0}

        t_start = time.monotonic()

        contribs_np, rdma_metrics = await _pull_contributors(
            self.recons,
            contributors,
            timeout_s=self._cfg.rdma_timeout_s,
        )
        t_rdma = time.monotonic() - t_start

        # Offload CPU blend/write so the event loop can start more RDMA pulls.
        geom = self._stitch_geom[out_tile_id]
        out_spatial = geom["out_spatial"]
        blend_kernel = self._blend_kernel
        kernel_cache = self._kernel_cache
        t_off = self.work.timepoint
        out_c_idx = self.work.output_channel_index

        def _blend_and_write() -> dict:
            t_blend_start = time.monotonic()
            result = _core.blend_contributors(geom, contribs_np, blend_kernel, kernel_cache)
            t_blend = time.monotonic() - t_blend_start

            write_region = (
                (slice(t_off, t_off + 1),)
                + (slice(out_c_idx, out_c_idx + 1),)
                + tuple(slice(lo, hi) for lo, hi in out_spatial)
            )
            capture = OutputCapture(
                tile_id=out_tile_id,
                region=write_region,
                payload=result[None],
            )
            receipt = write_capture(self._write_output, capture)
            t_write = receipt.elapsed_s

            return {"t_blend_s": t_blend, "t_write_s": t_write}

        bw = await asyncio.to_thread(_blend_and_write)

        return {
            "out_tile_id": out_tile_id,
            "n_inputs": len(contributors),
            "t_rdma_s": t_rdma,
            **rdma_metrics,
            "t_blend_s": bw["t_blend_s"],
            "t_write_s": bw["t_write_s"],
            "wall_s": time.monotonic() - t_start,
        }
