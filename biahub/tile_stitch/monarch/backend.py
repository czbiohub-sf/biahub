"""Monarch actor-mesh execution for tile-stitch plans.

The backend owns mesh setup, validated dispatch, reconstruction/stitch
pipelining, timepoint swaps, and actor teardown. A timepoint must drain before
``swap`` releases its reconstruction buffers.
"""

import asyncio
import logging
import os
import time

from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import IPv6Address
from pathlib import Path
from types import MappingProxyType

from biahub.tile_stitch.monarch.execution import TimepointExecution

logger = logging.getLogger("MonarchBackend")

# Actor shutdown and diagnostics are environment-tunable for failure recovery.
_SHUTDOWN_TIMEOUT_S = float(os.environ.get("TILE_SHUTDOWN_TIMEOUT_S", "15"))

_DRIVE_HB_S = float(os.environ.get("TILE_DRIVE_HB_S", "0"))

_INFINIBAND_SYSFS = Path("/sys/class/infiniband")
_MELLANOX_PCI_VENDOR_ID = 0x15B3


@dataclass(frozen=True, slots=True)
class _RdmaPreflight:
    """Local sysfs evidence needed to decide whether mlx5 is safe to use."""

    sysfs_readable: bool
    device_count: int
    active_mellanox_ports: tuple[str, ...]
    unusable_mellanox_ports: tuple[str, ...]


def _is_monarch_global_gid(value: str) -> bool:
    """Mirror Monarch's IPv6-form GID scope classification."""
    try:
        address = IPv6Address(value)
    except ValueError:
        return False
    if int(address) == 0 or address.is_loopback:
        return False
    if address.ipv4_mapped is not None:
        return not (address.ipv4_mapped.is_loopback or address.ipv4_mapped.is_link_local)
    first, second, *_ = address.packed
    return not (first == 0xFE and (second & 0xC0) in {0x80, 0xC0})


def _port_has_global_rocev2_gid(port: Path) -> bool:
    """Return whether an active mlx5 port satisfies Monarch 0.6 QP setup."""
    try:
        gid_files = sorted(
            (path for path in (port / "gids").iterdir() if path.name.isdecimal()),
            key=lambda path: int(path.name),
        )
    except OSError:
        return False

    for gid_file in gid_files:
        try:
            gid = gid_file.read_text().strip()
            gid_type = (port / "gid_attrs" / "types" / gid_file.name).read_text().strip()
        except OSError:
            continue
        if gid_type == "RoCE v2" and _is_monarch_global_gid(gid):
            return True
    return False


def _inspect_rdma_fabric(sysfs_root: Path = _INFINIBAND_SYSFS) -> _RdmaPreflight:
    """Inspect active NVIDIA/Mellanox ports without initializing Monarch RDMA."""
    try:
        devices = sorted(sysfs_root.iterdir())
    except OSError:
        return _RdmaPreflight(False, 0, (), ())

    active: list[str] = []
    unusable: list[str] = []
    for device in devices:
        try:
            vendor_id = int((device / "device" / "vendor").read_text().strip(), 0)
        except (OSError, ValueError):
            continue
        if vendor_id != _MELLANOX_PCI_VENDOR_ID:
            continue
        try:
            ports = sorted((device / "ports").iterdir())
        except OSError:
            continue
        for port in ports:
            try:
                state = (port / "state").read_text().strip()
            except OSError:
                continue
            if state.partition(":")[0].strip() != "4":
                continue
            name = f"{device.name}/port{port.name}"
            active.append(name)
            if not _port_has_global_rocev2_gid(port):
                unusable.append(name)

    return _RdmaPreflight(
        True,
        len(devices),
        tuple(active),
        tuple(unusable),
    )


def _configure_rdma_transport(sysfs_root: Path = _INFINIBAND_SYSFS) -> str:
    """Select native ibverbs only when local mlx5 QP setup can succeed."""
    import monarch

    from monarch.rdma import is_ibverbs_available

    config = monarch.get_global_config()
    if bool(config.get("rdma_disable_ibverbs")):
        monarch.configure(rdma_allow_tcp_fallback=True)
        logger.info("RDMA preflight: ibverbs explicitly disabled; using TCP fallback")
        return "tcp"

    if not is_ibverbs_available():
        monarch.configure(rdma_allow_tcp_fallback=True)
        logger.info("RDMA preflight: ibverbs unavailable; using TCP fallback")
        return "tcp"

    preflight = _inspect_rdma_fabric(sysfs_root)
    reason = ""
    if not preflight.sysfs_readable or preflight.device_count == 0:
        reason = "InfiniBand sysfs is unavailable"
    elif preflight.unusable_mellanox_ports:
        reason = "active NVIDIA/Mellanox ports lack a global RoCEv2 GID: " + ", ".join(
            preflight.unusable_mellanox_ports
        )

    if reason:
        monarch.configure(
            rdma_allow_tcp_fallback=True,
            rdma_disable_ibverbs=True,
        )
        logger.warning("RDMA preflight: %s; forcing TCP fallback", reason)
        return "tcp"

    if preflight.active_mellanox_ports:
        logger.info(
            "RDMA preflight: all %d active NVIDIA/Mellanox ports expose a "
            "global RoCEv2 GID; keeping ibverbs",
            len(preflight.active_mellanox_ports),
        )
    else:
        logger.info(
            "RDMA preflight: no active NVIDIA/Mellanox ports found; "
            "keeping the available non-mlx ibverbs backend"
        )
    return "ibverbs"


@dataclass(frozen=True, slots=True)
class _DispatchExecutionPlan:
    """Validated scheduler output and runtime memory bound.

    Attributes
    ----------
    input_order : tuple[int, ...]
        Reconstruction order dispatched to actors.
    resident_budget : int or None
        Maximum resident tiles, or ``None`` for unbounded dispatch.
    metrics : Mapping[str, int | float | str]
        Immutable scheduler and budget telemetry.
    """

    input_order: tuple[int, ...]
    resident_budget: int | None
    metrics: Mapping[str, int | float | str]


def _prepare_dispatch(
    program,
    recon_batch: int,
    cfg,
    *,
    n_actors: int,
) -> _DispatchExecutionPlan:
    """Prepare validated scheduling state before actor execution."""
    from biahub.tile_stitch.dispatch import (
        SchedulerContext,
        build_dispatch_plan,
        build_dispatch_problem,
    )

    problem = build_dispatch_problem(program)
    requested_budget = cfg.resident_budget if cfg is not None else None
    bounded = requested_budget is not None
    scheduler = str(cfg.dispatch_scheduler) if bounded else "plan"
    context = SchedulerContext(
        window=int(getattr(cfg, "scheduler_window", 32)),
        recon_batch=recon_batch,
        n_actors=n_actors,
    )
    schedule = build_dispatch_plan(problem, scheduler, context=context)
    metrics = dict(schedule.metrics)
    metrics["bounded"] = int(bounded)
    if not bounded:
        return _DispatchExecutionPlan(
            schedule.input_order,
            None,
            MappingProxyType(metrics),
        )

    max_fanin = max(
        (len(inputs) for inputs in problem.output_to_inputs.values()),
        default=1,
    )
    safe_floor = schedule.peak_resident_tiles + recon_batch * n_actors
    explicit_budget = 0 if requested_budget == "auto" else int(requested_budget)
    budget = max(explicit_budget, safe_floor, max_fanin, recon_batch)
    metrics.update(
        {
            "resident_budget": budget,
            "resident_safe_floor": safe_floor,
            "max_output_fanin": max_fanin,
        }
    )
    logger.info(
        "dispatch scheduler=%s window=%d budget=%d tiles "
        "(planner_peak=%d + headroom %dx%d; max_fanin=%d requested=%s)",
        scheduler,
        context.window,
        budget,
        schedule.peak_resident_tiles,
        recon_batch,
        n_actors,
        max_fanin,
        requested_budget,
    )
    return _DispatchExecutionPlan(
        schedule.input_order,
        budget,
        MappingProxyType(metrics),
    )


def _await_initialized_sync(host_mesh) -> None:
    """Block until every attached host has connected.

    Parameters
    ----------
    host_mesh : monarch.actor.HostMesh
        Attached host mesh exposing an ``initialized`` future.
    """

    async def _await(hm):
        await hm.initialized

    asyncio.run(_await(host_mesh))


def _slurm_topology() -> tuple[list[str], int, str | None]:
    """Resolve multi-node topology from the SLURM allocation.

    Returns
    -------
    hosts : list[str]
        Allocated hostnames, empty for a local run.
    port : int
        Deterministic worker port, or zero for a local run.
    ready_dir : str or None
        Shared worker-readiness directory.
    """
    import subprocess

    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    nnodes = int(os.environ.get("SLURM_NNODES", "1") or "1")
    if not nodelist or nnodes <= 1:
        return [], 0, None
    hosts = subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist], text=True
    ).split()
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    port = 26000 + (int(job_id) % 2000)
    base = os.environ.get("SLURM_SUBMIT_DIR") or os.environ.get("TMPDIR") or "/tmp"
    ready_dir = os.path.join(base, f".tile_ready_{job_id}")
    return hosts, port, ready_dir


def _wait_for_ready(ready_dir: str, node_list: list[str], timeout_s: int = 300) -> None:
    """Wait for every worker readiness file.

    Parameters
    ----------
    ready_dir : str
        Shared directory in which workers publish readiness.
    node_list : list[str]
        Expected worker hostnames.
    timeout_s : int, optional
        Maximum wait in seconds.

    Raises
    ------
    RuntimeError
        If all workers do not become ready before the deadline.
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


class _MonarchExecutionTransport:
    """Adapt Monarch actor calls to the transport-independent driver."""

    def __init__(self, backend) -> None:
        from monarch.actor import Channel

        self._backend = backend
        self._send, self._receive = Channel.open()

    async def prime(self, assignments) -> None:
        futures = [
            self._backend._actor_one(gpu).prime_loader.call_one(tile_ids=list(tile_ids))
            for gpu, tile_ids in assignments.items()
        ]
        await asyncio.gather(*futures)

    async def reconstruct(self, gpu: int, tile_ids, timeout_s: float) -> list:
        actor = self._backend._actor_one(gpu)
        if len(tile_ids) == 1:
            future = actor.reconstruct.call_one(tile_id=tile_ids[0])
            handle = await asyncio.wait_for(future, timeout=timeout_s)
            return [handle]
        future = actor.reconstruct_batch.call_one(tile_ids=list(tile_ids))
        return await asyncio.wait_for(future, timeout=timeout_s)

    def dispatch_stitch(self, output_id: int, contributors) -> None:
        from monarch.actor import send

        send(
            self._backend._workers.stitch,
            args=(output_id, dict(contributors)),
            kwargs={},
            port=self._send,
            selection="choose",
        )

    async def receive_stitch(self) -> dict:
        return await self._receive.recv()

    async def forget(self, tile_ids) -> None:
        await self._backend._workers.forget.call(tile_ids=list(tile_ids))


class MonarchBackend:
    """Execute tile-stitch plans over a Monarch actor mesh.

    Parameters
    ----------
    gpus_per_node : int or None, optional
        Actors per host. ``None`` uses locally visible CUDA devices.
    window_per_actor : int, optional
        Maximum in-flight output stitches per actor.
    device : str, optional
        Reconstruction device. Only ``"cuda"`` is currently supported.

    Notes
    -----
    :meth:`setup` infers single- or multi-node topology from SLURM.
    """

    def __init__(
        self,
        *,
        gpus_per_node: int | None = None,
        window_per_actor: int = 6,
        device: str = "cuda",
    ):
        self._gpus_per_node = gpus_per_node
        self._window_per_actor = window_per_actor
        self._device = device

        self._node_list: list[str] = []
        self._port = 0
        self._ready_dir: str | None = None
        self._is_multihost = False
        self._procs = None
        self._workers = None
        self._gpn = 0
        self._n_gpus = 0
        # Prevent ``swap`` from releasing buffers used by in-flight RDMA reads.
        self._drained = True

    def __enter__(self) -> "MonarchBackend":
        return self

    def __exit__(self, *exc) -> bool:
        self.teardown()
        return False

    def setup(self, program_path: str, work) -> None:
        """Spawn the actor mesh with one static program and initial work unit.

        Parameters
        ----------
        program_path : str
            Serialized static program loaded by every actor.
        work : StitchWorkUnit
            Initial input and output binding.

        Raises
        ------
        NotImplementedError
            If the backend is configured for CPU execution.
        """
        self._node_list, self._port, self._ready_dir = _slurm_topology()
        self._is_multihost = len(self._node_list) > 1
        # CPU is a configured device knob, but the actor's CUDA-only path
        # (set_device, resident volume, CUDA-graph compile, the cuda:{idx}
        # stream wiring) is not yet device-guarded. Fail loud rather than ship
        # an untested half-wired CPU path.
        if self._device == "cpu":
            raise NotImplementedError(
                "CPU device not yet wired for the Monarch backend; use device=cuda"
            )
        _configure_rdma_transport()
        import monarch.actor as ma
        import torch

        from monarch.actor import this_host

        from biahub.tile_stitch.monarch.tile_worker import TileWorker

        local_gpus = torch.cuda.device_count()
        if self._is_multihost:
            # Auto-detection assumes homogeneous workers; heterogeneous
            # allocations must configure ``gpus_per_node``.
            self._gpn = self._gpus_per_node or local_gpus
            ma.enable_transport("tcp")
            addrs = [f"tcp://{n}:{self._port}" for n in self._node_list]
            # Wait for every worker: cold-start time varies across nodes.
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
        self._workers = self._procs.spawn(
            "tile_workers", TileWorker, program_path=program_path, work=work
        )
        logger.info("actor mesh extent: %s", self._procs.extent)

    def _actor_one(self, flat_idx: int):
        """Select one actor by flat index.

        Parameters
        ----------
        flat_idx : int
            Actor index across all hosts and GPUs.

        Returns
        -------
        object
            Monarch mesh slice containing exactly one actor.
        """
        if self._is_multihost:
            return self._workers.slice(hosts=flat_idx // self._gpn, gpus=flat_idx % self._gpn)
        return self._workers.slice(gpus=flat_idx)

    def bind_work_unit(self, work) -> None:
        """Release prior unit resources and bind every actor to new work.

        Parameters
        ----------
        work : StitchWorkUnit
            Next input and output binding.

        Raises
        ------
        RuntimeError
            If the previous unit still has in-flight stitch or RDMA work.
        """
        if not self._drained:
            raise RuntimeError(
                "bind_work_unit() called before Stage B drained; refusing to "
                "release reconstructions during an RDMA pull"
            )

        async def _bind():
            values = await self._workers.bind_work_unit.call(work=work)
            return [stats for _, stats in values.items()]

        started = time.monotonic()
        stats = asyncio.run(_bind())
        self._drained = False
        current = max((item.get("host_rss_gb", 0.0) for item in stats), default=0.0)
        peak = max((item.get("host_maxrss_gb", 0.0) for item in stats), default=0.0)
        logger.info(
            "work binding: %.1fs (max actor host RSS now %.1f GB, peak %.1f GB)",
            time.monotonic() - started,
            current,
            peak,
        )

    def drive_tp(self, program) -> dict:
        """Validate, schedule, and execute one bound work unit.

        Parameters
        ----------
        program : StitchProgram
            Static geometry and engine settings.

        Returns
        -------
        dict
            Stage timings, output count, per-stitch summaries, and dispatch
            diagnostics.
        """
        cfg = program.monarch
        recon_batch = int(cfg.recon_batch)
        execution = _prepare_dispatch(
            program,
            recon_batch,
            cfg,
            n_actors=self._n_gpus,
        )
        t_a, t_pipe, summaries = asyncio.run(
            self._drive_one_tp(program, recon_batch, execution)
        )
        # The drive's final ``while stitch_in_flight > 0`` loop has run, so
        # every Stage B stitch (and its RDMA pulls) for this TP has completed.
        self._drained = True
        n_completed = sum(1 for s in summaries if s["n_inputs"] > 0)
        return {
            "stage_a_s": t_a,
            "pipe_s": t_pipe,
            "n_outputs": n_completed,
            "summaries": summaries,
            "dispatch": dict(execution.metrics),
        }

    async def _drive_one_tp(
        self,
        program,
        recon_batch: int,
        execution: _DispatchExecutionPlan,
    ):
        """Execute a prevalidated dispatch plan through the Monarch transport."""
        driver = TimepointExecution(
            program=program,
            input_order=execution.input_order,
            resident_budget=execution.resident_budget,
            recon_batch=recon_batch,
            n_actors=self._n_gpus,
            window=self._window_per_actor * self._n_gpus,
            transport=_MonarchExecutionTransport(self),
            heartbeat_s=_DRIVE_HB_S,
        )
        return await driver.run()

    # --- stats + teardown --------------------------------------------------
    def collect_recon_stats(self) -> list[dict]:
        """Collect Stage A telemetry from every actor.

        Returns
        -------
        list[dict]
            One reconstruction telemetry mapping per actor.
        """

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
            from monarch.actor import shutdown_context

            shutdown_context().get(timeout=_SHUTDOWN_TIMEOUT_S)
        except Exception as exc:
            # End-of-life cleanup must never fail the run, but a genuine timeout
            # here means the mesh did not shut down within the budget — workers/GPUs
            # may still be held on a shared node. Surface it as a warning rather
            # than swallowing it silently.
            logger.warning("actor-context shutdown did not complete cleanly: %s", exc)
