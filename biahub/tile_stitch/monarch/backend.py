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
import contextlib
import logging
import os
import subprocess
import time

from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("MonarchBackend")

# Pin each actor (CUDA device i) to its GPU's local NUMA node so H2D/D2H staging
# and RDMA buffers use local memory bandwidth + the GPU's local NIC. Opt-out via
# TILE_NUMA_BIND=0. Derivation falls back to no-binding if the topology can't be read.
_NUMA_BIND = os.environ.get("TILE_NUMA_BIND", "1") != "0"

# Budget for draining Monarch's actor-context shutdown in teardown (see teardown).
_SHUTDOWN_TIMEOUT_S = float(os.environ.get("TILE_SHUTDOWN_TIMEOUT_S", "15"))

# Drive-loop debug heartbeat interval (seconds); 0 = off. When >0, the gated
# drive loop logs its state (recon-left / in-flight / free gate slots / freed /
# done) every interval — the signal that diagnoses a Stage-B stall/deadlock.
_DRIVE_HB_S = float(os.environ.get("TILE_DRIVE_HB_S", "0"))

# Recon-dispatch knob DEFAULTS — used only when no MonarchConfig is present (the
# legacy cfg-less path). MonarchConfig is the source of truth otherwise, via
# recon_max_inflight_per_gpu / recon_rpc_timeout_s / recon_rpc_retries (promoted
# from env vars so the durable knobs live in one place; config.py:docstring).
#
# max_inflight bounds concurrent in-flight recon work-units PER GPU on the gated
# (tile_cache) path: without it, all recon tasks hit the gate and fire RPCs at
# once, flooding the Monarch mesh until calls stop flowing (driver+workers idle,
# GPUs 0%). 0 = unbounded (legacy).
#
# rpc_{timeout,retries}: Monarch occasionally fails to deliver/pick up a
# reconstruct call under load (it never returns, wedging the drive); a call
# exceeding the timeout is re-sent (rotating GPU), up to retries times, then the
# TP fails loudly.
_DEFAULT_MAX_INFLIGHT_PER_GPU = 3
_DEFAULT_RECON_RPC_TIMEOUT_S = 90.0
_DEFAULT_RECON_RPC_RETRIES = 3


def _numa_proc_bind(n_gpus: int) -> list[dict[str, str]] | None:
    """Per-proc NUMA binding list (proc i -> the NUMA node local to CUDA device i),
    derived from the node's PCI topology. Returns None if it can't be resolved.
    Assumes CUDA_DEVICE_ORDER=PCI_BUS_ID, so CUDA index == nvidia-smi index.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except Exception as exc:
        logger.warning("NUMA bind: nvidia-smi query failed (%s); skipping", exc)
        return None
    by_idx: dict[int, str] = {}
    for line in out.strip().splitlines():
        idx_s, bus = (p.strip() for p in line.split(","))
        parts = bus.split(":")  # nvidia-smi 00000000:18:00.0 -> /sys 0000:18:00.0
        if len(parts) != 3:
            return None
        # /sys uses a 4-digit, lowercase domain (nvidia-smi gives 8-digit, uppercase).
        dom = f"{parts[0][-4:]}:{parts[1]}:{parts[2]}".lower()
        try:
            numa = Path(f"/sys/bus/pci/devices/{dom}/numa_node").read_text().strip()
        except OSError:
            return None
        if int(numa) < 0:  # -1 => kernel reports no NUMA affinity
            return None
        by_idx[int(idx_s)] = numa
    if len(by_idx) < n_gpus:
        return None
    # membind only: pin each actor's allocations (H2D staging + RDMA buffers) to its
    # GPU's local NUMA memory. cpunodebind is omitted on purpose — a partial CPU
    # allocation may not include that NUMA's cores, which would make the proc
    # unschedulable (spawn hang); membind is valid regardless of the cpuset and
    # captures the H2D/NIC-locality win (the IO/D2H bottleneck).
    bindings = [{"membind": by_idx[i]} for i in range(n_gpus)]
    logger.info("NUMA bind (membind): proc->NUMA = %s", [b["membind"] for b in bindings])
    return bindings


class _ResidentGate:
    """Bounds the resident reconstructed-tile set (the P3b OOM fix).

    Acquire ``n`` before dispatching a recon work-unit (n tiles); release one per
    tile as Stage B frees it (ref-count → 0). ``acquire`` takes all ``n`` slots
    atomically under the condition, so two batches can't each hold a partial set
    and deadlock. Deadlock-safe when ``budget >= max tiles any single output needs
    co-resident`` (the auto budget = the order's interval-overlap peak ensures it).
    """

    def __init__(self, budget: int) -> None:
        self._free = budget
        self._cond = asyncio.Condition()

    async def acquire(self, n: int) -> None:
        async with self._cond:
            while self._free < n:
                await self._cond.wait()
            self._free -= n

    async def release(self, n: int = 1) -> None:
        async with self._cond:
            self._free += n
            self._cond.notify_all()


def _tile_cache_schedule(run_plan, recon_batch: int, cfg) -> tuple[list[int], int]:
    """Recon-dispatch order + resident budget for the bounded tile-cache path.

    Morton/raster-orders the output tiles, then orders input tiles by their
    earliest-consuming output's rank (so recon follows the stitch sweep → tight
    band). Budget = the chosen order's interval-overlap peak (min feasible),
    clamped to >= recon_batch and >= max output fan-in (deadlock-safe).
    """
    from biahub.tile_stitch.tile_cache import WindowedScheduler
    from biahub.tile_stitch.tile_cache_adapter import output_to_inputs_and_order

    out_to_in = run_plan.output_to_inputs
    kind = cfg.tile_cache_order.value
    if kind == "plan":
        in_order = list(run_plan.input_order)
        out_order = list(out_to_in)
    else:
        _, out_order = output_to_inputs_and_order(run_plan, order=kind)
        rank = {oid: i for i, oid in enumerate(out_order)}
        input_to_outputs: dict[int, list[int]] = defaultdict(list)
        for oid, ins in out_to_in.items():
            for tid in ins:
                input_to_outputs[tid].append(oid)
        in_order = sorted(
            input_to_outputs, key=lambda tid: min(rank[o] for o in input_to_outputs[tid])
        )
    auto = WindowedScheduler(out_to_in, out_order).predict_peak_tiles()
    max_fanin = max((len(v) for v in out_to_in.values()), default=1)
    n_gpus = getattr(cfg, "gpus_per_node", 1) or 1
    # ``auto`` is the *batch=1* liveness peak. Recon dispatches in ATOMIC batches
    # of ``recon_batch`` and up to ``n_gpus`` run concurrently, so a budget of
    # exactly ``auto`` strands the last <recon_batch free slots: ``acquire(K)``
    # blocks forever once free < K and nothing is in flight to release a slot
    # (the stranded-slots deadlock — froze the full FOV at ~output 50 with
    # gate_free=7 < recon_batch=8). Add one full batch of headroom per concurrent
    # GPU so a batch can always be acquired at the liveness peak. This floor is
    # mandatory; a configured resident_budget may only raise the budget above it.
    safe_floor = auto + recon_batch * n_gpus
    budget = max(cfg.resident_budget or 0, safe_floor, max_fanin, recon_batch)
    logger.info(
        "tile_cache budget=%d tiles (auto_peak=%d + headroom %dx%d; max_fanin=%d cfg_req=%s)",
        budget, auto, recon_batch, n_gpus, max_fanin, cfg.resident_budget,
    )
    return in_order, budget


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
            self._procs = host_mesh.spawn_procs(
                per_host={"gpus": self._gpn},
                proc_bind=(_numa_proc_bind(self._gpn) if _NUMA_BIND else None),
            )
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
            self._procs = this_host().spawn_procs(
                per_host={"gpus": self._n_gpus},
                proc_bind=(_numa_proc_bind(self._n_gpus) if _NUMA_BIND else None),
            )
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

        # P3b: bounded recon-dispatch path (behind the tile_cache flag). Reorder
        # recon to a locality (Morton) sweep and cap resident reconstructed tiles
        # at a budget so recon can't outrun Stage B and OOM host RAM. Off → the
        # legacy dispatch-all path, byte-for-byte unchanged.
        cfg = getattr(run_plan, "monarch", None)
        # Recon-dispatch knobs: MonarchConfig is the source of truth; fall back to
        # the module defaults only on the legacy cfg-less path.
        max_inflight = _DEFAULT_MAX_INFLIGHT_PER_GPU
        rpc_timeout = _DEFAULT_RECON_RPC_TIMEOUT_S
        rpc_retries = _DEFAULT_RECON_RPC_RETRIES
        if cfg is not None:
            max_inflight = getattr(cfg, "recon_max_inflight_per_gpu", max_inflight)
            rpc_timeout = getattr(cfg, "recon_rpc_timeout_s", rpc_timeout)
            rpc_retries = getattr(cfg, "recon_rpc_retries", rpc_retries)
        gate: _ResidentGate | None = None
        input_order = list(run_plan.input_order)
        if cfg is not None and getattr(cfg, "tile_cache", False):
            input_order, budget = _tile_cache_schedule(run_plan, recon_batch, cfg)
            gate = _ResidentGate(budget)
            logger.info(
                "tile_cache ON: order=%s resident_budget=%d tiles",
                cfg.tile_cache_order.value,
                budget,
            )

        recon_handles: dict[int, object] = {}
        pending_outputs = {
            oid: set(inputs) for oid, inputs in run_plan.output_to_inputs.items()
        }
        stitch_count = 0
        stitch_in_flight = 0
        recon_done = [False]  # set True after all recons finish; concurrent-drainer stop signal
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
        a_remaining = [len(input_order)]

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
                        if gate is not None:
                            await gate.release(1)  # open a recon slot
                if len(freed_pending) >= 32:
                    await _flush_forget()
            n_done = len(summaries)
            if n_done % 20 == 0:
                logger.info("Stage B: %d completed", n_done)

        async def _maybe_dispatch_outputs(ready_outputs: list[int]) -> None:
            nonlocal stitch_count, stitch_in_flight
            for oid in ready_outputs:
                while stitch_in_flight >= window:
                    # Gated path: the concurrent drainer owns done_recv; just wait
                    # for it to reduce in-flight (draining here would double-recv).
                    # Sleep a short interval (matching the drainer cadence) instead
                    # of sleep(0), which would hot-spin a core during backpressure.
                    if gate is not None:
                        await asyncio.sleep(0.002)
                    else:
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

        async def _recon_rpc(method: str, gpu: int, **kw):
            """Issue a reconstruct[_batch] RPC with timeout+retry. A call that
            doesn't return within the timeout is re-sent to the next GPU (Monarch
            occasionally drops one under load, which would otherwise wedge the
            drive). Non-timeout errors propagate immediately.

            KNOWN TRADEOFF: on a timeout we re-send without cancelling the original
            on its worker — a *merely slow* (not dropped) call then runs twice, so
            two workers transiently hold that tile (extra host RAM not counted by
            the gate) until the broadcast ``forget`` reclaims both. The timeout is
            set well above normal batch latency to make this rare. A proper
            cancel-original + forget-superseded-handle fix needs a live-mesh run to
            validate and is deferred to the worker-side-spill increment.
            """
            for attempt in range(rpc_retries + 1):
                g = (gpu + attempt) % n_gpus
                try:
                    call = getattr(self._actor_one(g), method).call_one(**kw)
                    return await asyncio.wait_for(call, timeout=rpc_timeout)
                except TimeoutError:
                    logger.warning(
                        "recon RPC %s timed out on gpu=%d (attempt %d/%d) — re-sending",
                        method, g, attempt + 1, rpc_retries + 1,
                    )
            raise TimeoutError(f"reconstruct stuck after {rpc_retries} retries ({method})")

        async def _do_recon(tile_id: int, gpu: int) -> None:
            if gate is not None:
                await gate.acquire(1)
            handle = await _recon_rpc("reconstruct", gpu, tile_id=tile_id)
            recon_handles[tile_id] = handle
            a_remaining[0] -= 1
            if a_remaining[0] == 0:
                t_a_end_local[0] = time.monotonic() - t_a_start
                logger.info("Stage A done in %.1fs", t_a_end_local[0])
            ready = _ready_outputs([tile_id])
            if ready:
                await _maybe_dispatch_outputs(ready)

        async def _do_recon_batch(tile_ids: list[int], gpu: int) -> None:
            if gate is not None:
                await gate.acquire(len(tile_ids))
            handles = await _recon_rpc("reconstruct_batch", gpu, tile_ids=tile_ids)
            for tid, handle in zip(tile_ids, handles, strict=True):
                recon_handles[tid] = handle
            a_remaining[0] -= len(tile_ids)
            if a_remaining[0] == 0:
                t_a_end_local[0] = time.monotonic() - t_a_start
                logger.info("Stage A done in %.1fs", t_a_end_local[0])
            ready = _ready_outputs(tile_ids)
            if ready:
                await _maybe_dispatch_outputs(ready)

        # Bound concurrent recon dispatch on the gated path: excess tasks park on
        # this semaphore (FIFO, cheap) rather than all hitting the gate and firing
        # recon RPCs at once, which floods the Monarch mesh until calls stop
        # flowing. ``None`` (unbounded) preserves the legacy dispatch-all behavior.
        _dispatch_sem = (
            asyncio.Semaphore(max(max_inflight * n_gpus, recon_batch))
            if (gate is not None and max_inflight > 0)
            else None
        )

        async def _dispatch(coro):
            if _dispatch_sem is None:
                await coro
            else:
                async with _dispatch_sem:
                    await coro

        if recon_batch > 1:
            batches = build_recon_batches(
                input_order,
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
                asyncio.create_task(_dispatch(_do_recon_batch(b, i % n_gpus)))
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
                    tile_ids=input_order[g::n_gpus]
                )
                for g in range(n_gpus)
            ]
            await asyncio.gather(*prime_futs)
            recon_tasks = [
                asyncio.create_task(_dispatch(_do_recon(tile_id, i % n_gpus)))
                for i, tile_id in enumerate(input_order)
            ]
            logger.info("Stage A+B pipelined: %d recon tasks dispatched", len(recon_tasks))
        if gate is not None:
            # Concurrent drainer (the deadlock fix): owns done_recv, frees tiles +
            # releases gate slots independent of recon dispatch, so parked recon
            # tasks get unblocked. Recon = gated producer; this = consumer. Drains
            # while recon runs or any stitch is in flight; the ``stitch_in_flight>0``
            # guard avoids blocking on recv() when nothing is dispatched (and means
            # empty/never-ready outputs aren't waited on — same as the flag-off loop).
            async def _drainer() -> None:
                while not recon_done[0] or stitch_in_flight > 0:
                    if stitch_in_flight > 0:
                        await _drain_one_stitch()
                    else:
                        await asyncio.sleep(0.002)  # let recon dispatch the next ready output

            async def _heartbeat() -> None:
                while True:
                    await asyncio.sleep(_DRIVE_HB_S)
                    ready = sum(1 for ins in pending_outputs.values()
                                if ins <= recon_handles.keys())
                    logger.info(
                        "DRIVE hb: a_remaining=%d in_flight=%d gate_free=%d held=%d "
                        "ready_undispatched=%d freed_pending=%d done=%d recon_done=%s",
                        a_remaining[0], stitch_in_flight, gate._free,
                        len(recon_handles), ready, len(freed_pending),
                        len(summaries), recon_done[0],
                    )

            drainer = asyncio.create_task(_drainer())
            hb = asyncio.create_task(_heartbeat()) if _DRIVE_HB_S > 0 else None
            try:
                await asyncio.gather(*recon_tasks)
            except BaseException:
                # A recon work-unit failed (e.g. _recon_rpc exhausted its retries).
                # Signal + cancel the drainer and re-raise so the failure surfaces,
                # instead of leaving recon_done False -> the drainer orphaned looping
                # and the gate's slots leaked -> the exact deadlock this path exists
                # to prevent. Aborting the TP loudly is correct here.
                recon_done[0] = True
                drainer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drainer
                raise
            else:
                recon_done[0] = True
                await drainer
            finally:
                if hb is not None:
                    hb.cancel()
        else:
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
        # Drain Monarch's global actor-context shutdown here, with a generous
        # budget, rather than leaving it to the interpreter's atexit handler.
        # That handler caps the shutdown at 1s and, when teardown runs longer,
        # raises a TimeoutError the interpreter prints as "Exception ignored in
        # atexit callback" — benign end-of-run noise. Completing it now sets
        # Monarch's _shutdown_done flag, so the atexit handler finds an
        # already-resolved future and no-ops. Best-effort: swallow everything,
        # since this is end-of-life cleanup and must never fail the run.
        try:
            from monarch._src.actor.actor_mesh import shutdown_context

            shutdown_context().get(timeout=_SHUTDOWN_TIMEOUT_S)
        except Exception as exc:
            # End-of-life cleanup must never fail the run, but a genuine timeout
            # here means the mesh did not shut down within the budget — workers/GPUs
            # may still be held on a shared node. Surface it as a warning rather
            # than swallowing it silently.
            logger.warning("actor-context shutdown did not complete cleanly: %s", exc)
