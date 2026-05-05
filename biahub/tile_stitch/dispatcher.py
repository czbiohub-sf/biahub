"""Cluster lifecycle.

Verbatim port of the legacy v4_dask ``make_cluster`` and v7
``gpu_worker_setup``. Don't re-derive; the legacy is the source of
truth for what works on Bruno.
"""

import logging
import os
import socket
import sys

from typing import Any

from biahub.tile_stitch.config import CpuSlurmConfig, GpuLocalCudaConfig

logger = logging.getLogger(__name__)


# --- CPU SLURMCluster (port of v4_dask.py:1676-1738) -----------------------


def make_cpu_cluster(cfg: CpuSlurmConfig, *, run_dir: str) -> Any:
    """Build a SLURMCluster for CPU dask-jobqueue workers.

    Verbatim port of `drivers/v4_dask.py:make_cluster`. The legacy driver
    converged on this exact set of kwargs over c0017→c0041; don't change
    them without an iteration to back the change.
    """
    from dask_jobqueue import SLURMCluster

    job_extra: list[str] = [f"--cpus-per-task={cfg.worker_cores}"]
    if cfg.preempt_signal_secs:
        job_extra.append(f"--signal=USR1@{cfg.preempt_signal_secs}")

    nthreads = cfg.dask_threads or cfg.worker_cores

    log_dir = os.path.join(run_dir, "dask_logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.scratch_dir, exist_ok=True)

    cluster = SLURMCluster(
        cores=cfg.worker_cores,
        processes=1,
        memory=cfg.worker_memory,
        walltime=cfg.worker_walltime,
        queue=cfg.partition,
        local_directory=cfg.scratch_dir,
        log_directory=log_dir,
        python=sys.executable,
        job_extra_directives=job_extra,
        job_script_prologue=[
            f"export OMP_NUM_THREADS={cfg.blas_threads}",
            f"export MKL_NUM_THREADS={cfg.blas_threads}",
            f"export OPENBLAS_NUM_THREADS={cfg.blas_threads}",
            f"export TS_ZARR_ASYNC_CONCURRENCY={cfg.zarr_async_concurrency}",
            f"export TS_ZARR_THREADING_MAX_WORKERS={cfg.zarr_threading_max_workers}",
            "export PYTHONUNBUFFERED=1",
            "export MALLOC_TRIM_THRESHOLD_=0",
            "export MALLOC_ARENA_MAX=2",
        ],
        worker_extra_args=["--nthreads", str(nthreads)],
    )
    logger.info(
        "created SLURMCluster on %s (cores=%d, dask_threads=%d, blas_threads=%d)",
        cfg.partition,
        cfg.worker_cores,
        cfg.dask_threads,
        cfg.blas_threads,
    )
    return cluster


# --- GPU LocalCUDACluster (port of v7 single-node path) --------------------


def make_gpu_cluster(cfg: GpuLocalCudaConfig) -> Any:
    """Single-node LocalCUDACluster for c0041-style runs.

    Driver must already be in a SLURM allocation with --gres=gpu:N --exclusive.
    LocalCUDACluster auto-detects from CUDA_VISIBLE_DEVICES.
    """
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(
        rmm_pool_size=cfg.rmm_pool_size,
        protocol=cfg.protocol,
        threads_per_worker=1,
    )
    logger.info(
        "created LocalCUDACluster (rmm_pool_size=%s, protocol=%s)",
        cfg.rmm_pool_size,
        cfg.protocol,
    )
    return cluster


# --- GPU worker setup (port of v7.py:gpu_worker_setup) --------------------


def gpu_worker_setup() -> dict[str, Any]:
    """RMM-backed torch allocator + cuFFT plan cache. Run via client.run().

    Port of src/tile_stitch/v7.py:gpu_worker_setup. Idempotent.
    """
    import torch

    status: dict[str, Any] = {"host": socket.gethostname(), "pid": os.getpid()}

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_worker_setup: no CUDA device on this worker")

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    status["device_name"] = props.name
    status["vram_gb"] = round(props.total_memory / (1024**3), 2)

    try:
        from rmm.allocators.torch import rmm_torch_allocator

        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        status["rmm_torch_allocator"] = "active"
    except Exception as exc:
        status["rmm_torch_allocator"] = f"failed: {exc}"

    try:
        torch.backends.cuda.cufft_plan_cache.max_size = 32
        torch.backends.cuda.cufft_plan_cache[device].max_size = 32
        status["cufft_plan_cache_max"] = 32
    except Exception as exc:
        status["cufft_plan_cache"] = f"failed: {exc}"

    return status


def wait_for_workers(client: Any, target: int, timeout_s: float) -> int:
    """Block until ``target`` workers register, or fall back after timeout.

    Returns the actual worker count. Logs the registered count on every
    change so operators can see worker churn during prewarm.
    """
    import time

    start = time.monotonic()
    deadline = start + timeout_s
    last = -1
    while time.monotonic() < deadline:
        n = len(client.scheduler_info().get("workers", {}))
        if n >= target:
            elapsed = time.monotonic() - start
            logger.info("wait_for_workers: %d/%d workers in %.1fs", n, target, elapsed)
            return n
        if n != last:
            logger.info("wait_for_workers: %d/%d workers …", n, target)
            last = n
        time.sleep(1.0)

    n = len(client.scheduler_info().get("workers", {}))
    logger.warning(
        "wait_for_workers timed out after %.0fs; only %d of %d landed",
        timeout_s,
        n,
        target,
    )
    return n
