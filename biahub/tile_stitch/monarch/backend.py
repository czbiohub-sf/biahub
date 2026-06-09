"""``MonarchBackend`` — the Monarch distributed engine for tile-stitch.

Lifts the mesh bring-up + per-TP pipelined drive + volume swap + stats
collection out of the ``m3_driver`` script into a single reusable class.
``m3_driver`` (and, in Stage 3, ``cli.py``) build the shared scaffold (TP
resolve/shard, engine plan, output zarr, per-TP plan pickles) and delegate
the run loop here.

Lifecycle:

    backend = MonarchBackend(gpus_per_node=..., nodes=..., port=..., ...)
    backend.setup(plan_entries[0].plan_path)        # spawn the actor mesh
    for i, (tp, plan_path, plan) in enumerate(plan_entries):
        if i > 0:
            backend.swap(plan_path)                  # per-TP volume/reader reset
        summary = backend.drive_tp(plan_path, plan, recon_batch=recon_batch)
        backend.collect_recon_stats()
    backend.teardown()

``swap`` enforces that the prior TP's Stage B fully drained before issuing
``swap_to`` (which clears the prior TP's recons + RDMABuffers). Splitting the
old monolithic ``_drive_one_tp`` into ``drive_tp`` + ``swap`` would otherwise
lose that structural guarantee and free buffers mid-RDMA-pull.
"""

import asyncio
import logging
import time

from collections import defaultdict

logger = logging.getLogger("MonarchBackend")


def _await_initialized_sync(host_mesh) -> None:
    """Block until every attached host has connected."""

    async def _await(hm):
        await hm.initialized

    asyncio.run(_await(host_mesh))


def _wait_for_ready(ready_dir: str, node_list: list[str], timeout_s: int = 300) -> None:
    """Block until every node has dropped its ``<hostname>.ready`` file.

    Workers write the file just before binding their listen socket, so a
    small margin is added after all appear to let the sockets come up.
    """
    import os
    import time as _t

    expected = {f"{n}.ready" for n in node_list}
    deadline = _t.monotonic() + timeout_s
    last_n = -1
    while _t.monotonic() < deadline:
        present = set(os.listdir(ready_dir)) if os.path.isdir(ready_dir) else set()
        have = expected & present
        if len(have) != last_n:
            logger.info("ready workers: %d/%d", len(have), len(expected))
            last_n = len(have)
        if expected <= present:
            _t.sleep(3.0)  # margin: file written ~just before socket bind
            logger.info("all %d workers ready", len(expected))
            return
        _t.sleep(2.0)
    raise RuntimeError(
        f"timed out waiting for workers: have {last_n}/{len(expected)} after {timeout_s}s"
    )


def _grid_coords(tiles_by_id: dict, dims: tuple) -> dict:
    """Map tile_id -> integer grid index per dim (rank of its slice start)."""
    starts = {d: sorted({t.slices[d].start for t in tiles_by_id.values()}) for d in dims}
    rank = {d: {s: i for i, s in enumerate(starts[d])} for d in dims}
    return {
        tid: tuple(rank[d][t.slices[d].start] for d in dims) for tid, t in tiles_by_id.items()
    }


def _morton(coord: tuple) -> int:
    """Z-order key by bit-interleaving integer grid coords (locality-preserving)."""
    key = 0
    n = len(coord)
    for b in range(20):
        for j, c in enumerate(coord):
            key |= ((c >> b) & 1) << (n * b + j)
    return key


def build_recon_batches(
    order: list[int],
    tiles_by_id: dict,
    bsize: int,
    *,
    locality: bool = False,
    tile_dims: tuple | None = None,
) -> list[list[int]]:
    """Group ``order`` into batches of ``bsize`` same-shape tiles.

    All tiles in a batch must share spatial shape (the batched FFT stacks
    them into ``(B, Z, Y, X)``). Tiles are bucketed by shape first — so
    interior tiles form full batches and ragged edge shapes form their
    own (possibly smaller) batches — then each bucket is chunked.

    Default ordering within a shape is ``input_order`` (raster). With
    ``locality=True`` each shape bucket is Z-order (Morton) sorted first, so a
    ``bsize`` chunk is a spatially compact cluster (raster chunks straddle
    row/plane wraps). ``tile_dims`` is required when ``locality`` is set.
    """
    from collections import OrderedDict

    by_shape: OrderedDict[tuple, list[int]] = OrderedDict()
    for tid in order:
        shp = tuple(tiles_by_id[tid].shape)
        by_shape.setdefault(shp, []).append(tid)

    if locality:
        if tile_dims is None:
            raise ValueError("build_recon_batches(locality=True) needs tile_dims")
        gc = _grid_coords(tiles_by_id, tile_dims)
        for shp in by_shape:
            by_shape[shp] = sorted(by_shape[shp], key=lambda t: _morton(gc[t]))

    batches: list[list[int]] = []
    for tids in by_shape.values():
        for i in range(0, len(tids), bsize):
            batches.append(tids[i : i + bsize])
    return batches


class MonarchBackend:
    """Monarch actor-mesh engine for a single tile-stitch run.

    Multi-host (``nodes``, ``port``, ``ready_dir``) and the per-host GPU count
    are runtime/SLURM-dependent, so they are constructor args, not config.
    """

    def __init__(
        self,
        *,
        gpus_per_node: int | None = None,
        nodes: list[str] | None = None,
        port: int = 26000,
        ready_dir: str | None = None,
        window_per_actor: int = 6,
        device: str = "cuda",
    ):
        self._gpus_per_node = gpus_per_node
        self._node_list = [n for n in (nodes or []) if n]
        self._port = port
        self._ready_dir = ready_dir
        self._window_per_actor = window_per_actor
        self._device = device

        self._is_multihost = len(self._node_list) > 1
        self._procs = None
        self._workers = None
        self._gpn = 0
        self._n_gpus = 0
        # Stage-B drain guard: True only between a completed ``drive_tp`` and
        # the next ``swap``. ``swap`` refuses to release the prior TP's recons
        # / RDMABuffers unless the last drive fully drained (reviewer E1).
        self._drained = True

    # --- context manager: guarantee teardown on any exit -------------------
    def __enter__(self) -> "MonarchBackend":
        # ``setup`` is still called explicitly inside the ``with`` block — it
        # needs the first plan path, which the CLI scaffold computes.
        return self

    def __exit__(self, *exc) -> bool:
        self.teardown()
        return False  # never swallow a driver error

    # --- mesh lifecycle ----------------------------------------------------
    def setup(self, first_plan_path: str) -> None:
        """Spawn the actor mesh, initialised with the first TP's plan."""
        # CPU is a configured device knob, but the actor's CUDA-only path
        # (set_device, resident volume, CUDA-graph compile, the cuda:{idx}
        # stream wiring) is not yet device-guarded. Fail loud rather than ship
        # an untested half-wired CPU path (lead ruling 1).
        if self._device == "cpu":
            raise NotImplementedError(
                "CPU device not yet wired for the Monarch backend; use device=cuda"
            )
        import monarch.actor as ma
        import torch

        from monarch.actor import this_host

        from biahub.tile_stitch.monarch.tile_worker import TileWorker

        local_gpus = torch.cuda.device_count()
        if self._is_multihost:
            # Multi-host: attach to per-node worker loops, form a HostMesh.
            # The local_gpus fallback assumes homogeneous nodes (it reflects
            # the controller's device count, not the workers') — pass
            # --gpus-per-node explicitly for heterogeneous allocations.
            self._gpn = self._gpus_per_node or local_gpus
            ma.enable_transport("tcp")
            addrs = [f"tcp://{n}:{self._port}" for n in self._node_list]
            # Gate the attach on every worker signalling readiness — non-batch
            # nodes cold-start uv/monarch/cuda slower than the batch node, and
            # a one-shot attach races them ("config push failed on 1 host").
            if self._ready_dir:
                _wait_for_ready(self._ready_dir, self._node_list, timeout_s=300)
            logger.info("attaching to %d host workers: %s", len(addrs), addrs)
            host_mesh = ma.attach_to_workers(workers=addrs, ca="trust_all_connections")
            _await_initialized_sync(host_mesh)
            self._procs = host_mesh.spawn_procs(per_host={"gpus": self._gpn})
            self._n_gpus = len(self._node_list) * self._gpn
            logger.info(
                "multi-host mesh: %d nodes × %d gpus = %d actors",
                len(self._node_list),
                self._gpn,
                self._n_gpus,
            )
        else:
            self._gpn = self._gpus_per_node or local_gpus
            self._n_gpus = self._gpn
            logger.info(
                "single host: spawning %d actors (one per CUDA device)",
                self._n_gpus,
            )
            self._procs = this_host().spawn_procs(per_host={"gpus": self._n_gpus})
        # Init actors with the FIRST plan; subsequent TPs swap_to.
        self._workers = self._procs.spawn(
            "tile_workers", TileWorker, plan_path=first_plan_path
        )
        logger.info("actor mesh extent: %s", self._procs.extent)

    def _actor_one(self, flat_idx: int):
        """Single-actor slice for a flat actor index.

        Multi-host meshes have dims ``{hosts, gpus}`` — slice both so
        ``call_one`` sees exactly one actor. Single-host has only ``gpus``.
        """
        if self._is_multihost:
            return self._workers.slice(hosts=flat_idx // self._gpn, gpus=flat_idx % self._gpn)
        return self._workers.slice(gpus=flat_idx)

    # --- per-TP volume swap ------------------------------------------------
    def swap(self, plan_path: str) -> None:
        """Switch every actor to a new plan (per-TP volume + reader reset).

        Enforces that the prior TP's Stage B fully drained before issuing
        ``swap_to`` — ``swap_to`` clears the prior TP's cached recons and the
        RDMABuffers peers may still be pulling, so swapping mid-drain would
        race in-flight RDMA reads (reviewer E1). ``drive_tp`` sets
        ``_drained`` only after its final drain loop completes.
        """
        if not self._drained:
            raise RuntimeError(
                "swap() called before the prior TP's Stage B drained — refusing "
                "to release recons/RDMABuffers mid-pull"
            )

        async def _swap():
            vm = await self._workers.swap_to.call(plan_path=plan_path)
            return [st for _, st in vm.items()]

        t_swap = time.monotonic()
        stats = asyncio.run(_swap())
        self._drained = False  # new TP not yet driven/drained
        # Current (post-cleanup) host RSS across actors — watch this stay flat
        # across TPs. A steady climb means recon host memory isn't being
        # released between TPs (e.g. undropped RDMABuffer registrations).
        cur = max((s.get("host_rss_gb", 0.0) for s in stats), default=0.0)
        peak = max((s.get("host_maxrss_gb", 0.0) for s in stats), default=0.0)
        logger.info(
            "volume swap: %.1fs (max actor host RSS now %.1f GB, peak %.1f GB)",
            time.monotonic() - t_swap,
            cur,
            peak,
        )

    # --- per-TP drive ------------------------------------------------------
    def drive_tp(self, plan_path: str, plan, *, recon_batch: int = 1) -> dict:
        """Stage A → Stage B pipelined drive for a single TP's plan.

        Returns a summary dict: ``stage_a_s`` (Stage A wall), ``pipe_s``
        (full A+B pipelined wall), ``n_outputs`` (completed non-empty
        stitches), and ``summaries`` (per-stitch timing dicts).
        """
        t_a, t_pipe, summaries = asyncio.run(self._drive_one_tp(plan, recon_batch))
        # The drive's final ``while stitch_in_flight > 0`` loop has run, so
        # every Stage B stitch (and its RDMA pulls) for this TP has completed.
        self._drained = True
        n_completed = sum(1 for s in summaries if s["n_inputs"] > 0)
        return {
            "stage_a_s": t_a,
            "pipe_s": t_pipe,
            "n_outputs": n_completed,
            "summaries": summaries,
        }

    async def _drive_one_tp(self, run_plan, recon_batch: int):
        """Stage A → Stage B pipelined drive for a single TP's plan."""
        from monarch.actor import Channel, send

        workers = self._workers
        n_gpus = self._n_gpus
        window = self._window_per_actor * n_gpus

        done_send, done_recv = Channel.open()

        # Inverse map per-TP (cheap to rebuild; structure is identical across
        # TPs but we rebuild for safety in case a config tweak ever changes it).
        input_to_outputs: dict[int, list[int]] = defaultdict(list)
        for oid, inputs in run_plan.output_to_inputs.items():
            for tid in inputs:
                input_to_outputs[tid].append(oid)

        tiles_by_id = {t.tile_id: t for t in run_plan.input_tiles}

        recon_handles: dict[int, object] = {}
        pending_outputs = {
            oid: set(inputs) for oid, inputs in run_plan.output_to_inputs.items()
        }
        stitch_count = 0
        stitch_in_flight = 0
        summaries: list[dict] = []

        # Refcount each input tile by how many output tiles still need it. When a
        # tile's last output finishes stitching, no in-flight RDMA pull can need
        # it anymore, so free its recon immediately — drop the driver's RDMABuffer
        # handle and ``forget`` the actor's cached CPU tensor — instead of holding
        # the whole TP's ~100 GB of recons in host RAM until ``swap_to``.
        tile_remaining = {tid: len(outs) for tid, outs in input_to_outputs.items()}
        freed_pending: list[int] = []

        async def _flush_forget() -> None:
            # ``.call`` returns a Monarch Future (awaitable), not a coroutine, so
            # await it directly — don't wrap in asyncio.create_task. Batched
            # (caller flushes every ~32 frees), so the round-trip cost is small.
            if freed_pending:
                ids = list(freed_pending)
                freed_pending.clear()
                await self._workers.forget.call(tile_ids=ids)

        t_a_start = time.monotonic()
        t_a_end_local = [0.0]
        a_remaining = [len(run_plan.input_order)]

        async def _drain_one_stitch() -> None:
            nonlocal stitch_in_flight
            s = await done_recv.recv()
            stitch_in_flight -= 1
            summaries.append(s)
            # This output is done -> decrement its contributors; free any whose
            # last output just completed (safe: no pending/in-flight stitch needs
            # them once tile_remaining hits 0).
            oid = s.get("out_tile_id")
            if oid is not None:
                for tid in run_plan.output_to_inputs.get(oid, ()):
                    r = tile_remaining.get(tid)
                    if r is None:
                        continue
                    tile_remaining[tid] = r - 1
                    if r <= 1:
                        recon_handles.pop(tid, None)  # drop driver's RDMABuffer ref
                        freed_pending.append(tid)
                if len(freed_pending) >= 32:
                    await _flush_forget()
            n_done = len(summaries)
            if n_done % 20 == 0:
                logger.info("Stage B: %d completed", n_done)

        async def _maybe_dispatch_outputs(ready_outputs: list[int]) -> None:
            nonlocal stitch_count, stitch_in_flight
            for oid in ready_outputs:
                while stitch_in_flight >= window:
                    await _drain_one_stitch()
                contribs = {tid: recon_handles[tid] for tid in pending_outputs[oid]}
                send(
                    workers.stitch,
                    args=(oid, contribs),
                    kwargs={},
                    port=done_send,
                    selection="choose",
                )
                del pending_outputs[oid]
                stitch_count += 1
                stitch_in_flight += 1

        def _ready_outputs(tile_ids: list[int]) -> list[int]:
            seen: dict[int, None] = {}
            for tid in tile_ids:
                for oid in input_to_outputs.get(tid, []):
                    if (
                        oid in pending_outputs
                        and oid not in seen
                        and pending_outputs[oid] <= recon_handles.keys()
                    ):
                        seen[oid] = None
            return list(seen)

        async def _do_recon(tile_id: int, gpu: int) -> None:
            fut = self._actor_one(gpu).reconstruct.call_one(tile_id=tile_id)
            handle = await fut
            recon_handles[tile_id] = handle
            a_remaining[0] -= 1
            if a_remaining[0] == 0:
                t_a_end_local[0] = time.monotonic() - t_a_start
                logger.info("Stage A done in %.1fs", t_a_end_local[0])
            ready = _ready_outputs([tile_id])
            if ready:
                await _maybe_dispatch_outputs(ready)

        async def _do_recon_batch(tile_ids: list[int], gpu: int) -> None:
            fut = self._actor_one(gpu).reconstruct_batch.call_one(tile_ids=tile_ids)
            handles = await fut
            for tid, handle in zip(tile_ids, handles, strict=True):
                recon_handles[tid] = handle
            a_remaining[0] -= len(tile_ids)
            if a_remaining[0] == 0:
                t_a_end_local[0] = time.monotonic() - t_a_start
                logger.info("Stage A done in %.1fs", t_a_end_local[0])
            ready = _ready_outputs(tile_ids)
            if ready:
                await _maybe_dispatch_outputs(ready)

        if recon_batch > 1:
            batches = build_recon_batches(
                run_plan.input_order,
                tiles_by_id,
                recon_batch,
            )
            # Prefetch the next work-unit's read during the current FFT (the
            # IO-bound read overlaps compute): prime each actor's tile reader
            # with the FLATTENED order of its assigned batches. Assignment is
            # round-robin (``i % n_gpus``).
            per_actor_order: dict[int, list[int]] = {g: [] for g in range(n_gpus)}
            for i, b in enumerate(batches):
                per_actor_order[i % n_gpus].extend(b)
            prime_futs = [
                self._actor_one(g).prime_reader.call_one(tile_ids=per_actor_order[g])
                for g in range(n_gpus)
            ]
            await asyncio.gather(*prime_futs)
            recon_tasks = [
                asyncio.create_task(_do_recon_batch(b, i % n_gpus))
                for i, b in enumerate(batches)
            ]
            logger.info(
                "Stage A+B pipelined: %d batches (B=%d) dispatched",
                len(recon_tasks),
                recon_batch,
            )
        else:
            # Prime each actor's prefetch reader with its assigned tile
            # sequence (round-robin ``input_order[g::n_gpus]`` — the order it
            # will reconstruct). The reader then pulls tile N+1 from zarr
            # while the GPU runs FFT on tile N. No-op if prefetch is disabled.
            prime_futs = [
                self._actor_one(g).prime_reader.call_one(
                    tile_ids=run_plan.input_order[g::n_gpus]
                )
                for g in range(n_gpus)
            ]
            await asyncio.gather(*prime_futs)
            recon_tasks = [
                asyncio.create_task(_do_recon(tile_id, i % n_gpus))
                for i, tile_id in enumerate(run_plan.input_order)
            ]
            logger.info("Stage A+B pipelined: %d recon tasks dispatched", len(recon_tasks))
        await asyncio.gather(*recon_tasks)

        while stitch_in_flight > 0:
            await _drain_one_stitch()
        # Free the last batch of recons before returning (swap_to/teardown will
        # clear whatever remains anyway).
        await _flush_forget()
        logger.info("Stage B: %d/%d completed (final)", len(summaries), stitch_count)
        return t_a_end_local[0], time.monotonic() - t_a_start, summaries

    # --- stats + teardown --------------------------------------------------
    def collect_recon_stats(self) -> list[dict]:
        """Gather per-actor Stage A timing from every actor."""

        async def _collect():
            vm = await self._workers.recon_stats.call()
            return [st for _, st in vm.items()]

        return asyncio.run(_collect())

    def teardown(self) -> None:
        """Release the actor mesh.

        Monarch tears procs down on GC; we drop our references so the
        controller can exit cleanly.
        """
        self._workers = None
        self._procs = None
