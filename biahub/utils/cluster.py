"""Sizing a step's per-position work and choosing where to run it.

``estimate_resources`` derives the CPU/RAM/wall-time a position needs,
``echo_resources`` publishes that as the contract the Nextflow pipeline parses,
and ``get_submitit_cluster`` picks the executor the CLI submits to.
"""

import json
import os

import click
import numpy as np

from numpy.typing import DTypeLike


def echo_resources(num_cpus: int, mem_gb: int, time_minutes: int) -> None:
    """Emit the per-position resource request consumed by the Nextflow pipeline.

    Every step CLI calls this from its ``--init`` path so there is a single
    source of truth for per-position CPU, memory, and wall-clock time. The
    Nextflow ``init_*`` process captures this line on stdout and
    ``parse_resources`` (``nextflow/modules/common.nf``) reads the JSON payload
    to set the per-position task's ``cpus``/``memory``/``time`` directives. The
    same values also feed the CLI's own ``slurm_*`` submission args, so the
    SLURM fan-out and the Nextflow fan-out request identical resources.

    A single JSON payload keeps the contract order-independent and extensible
    (new fields can be added without breaking the positional parsing).

    Parameters
    ----------
    num_cpus : int
        CPUs per position.
    mem_gb : int
        TOTAL memory per position in GB (not per-CPU).
    time_minutes : int
        Wall-clock budget per position in minutes.
    """
    # Coerce to plain int: estimators may return numpy integers, which json
    # cannot serialize.
    payload = {"cpus": int(num_cpus), "mem_gb": int(mem_gb), "time_minutes": int(time_minutes)}
    click.echo("RESOURCES:" + json.dumps(payload))


def get_submitit_cluster(
    local: bool = False,
    cluster: str | None = None,
) -> str:
    """Return the submitit cluster type.

    'debug' is forced in CI. Otherwise the explicit `cluster` string wins;
    if no cluster is given, falls back to the legacy `local` boolean.
    """
    if os.environ.get("CI") == "true":
        return "debug"
    if cluster is not None:
        return cluster
    return "local" if local else "slurm"


def estimate_resources(
    shape: tuple[int, int, int, int, int],
    dtype: DTypeLike = np.float32,
    ram_multiplier: float = 1.0,
    time_multiplier: float = 1.0,
    max_num_cpus: int = 64,
    min_ram_per_cpu: int = 4,
    min_time_minutes: int = 30,
):
    """Estimate wall-time, CPUs, and RAM required to process a data volume.

    Both RAM and wall-time key on the ZYX volume, the natural unit of work here:
    RAM scales with a single volume (the per-CPU working set), and wall-time
    scales with the NUMBER of volumes processed (T * C).

    Counting volumes -- rather than voxels -- is deliberate. Per-voxel
    throughput is not a stable quantity: it depends on the CPU/GPU model, the
    filesystem write speed, and the chunking, so a voxel-rate calibrated on one
    run does not transfer to the next. Volume count is a property of the
    dataset alone. The spread in per-volume cost between, say, an A549 volume
    and a neuromast volume is absorbed by ``time_multiplier``, which is a fudge
    factor, not a physical constant -- over-requesting 2x on one dataset and
    1.5x on another is fine and expected.

    ``time_multiplier`` mirrors ``ram_multiplier``: it is the per-step scaling
    knob, in minutes of wall-time per ZYX volume, calibrated from observed
    COMPLETED runs (see each call site). Callers that only need CPUs and RAM can
    ignore the time estimate:

        _, num_cpus, gb_ram_per_cpu = estimate_resources(shape, ram_multiplier=8)

    Parameters
    ----------
    shape : Tuple[int, int, int, int, int]
        The shape of the data as a tuple (T, C, Z, Y, X).
    dtype : DTypeLike, optional
        The data type of the elements. Default is np.float32.
    ram_multiplier : float, optional
        Multiplier to scale the required memory for processing a given ZYX volume.
        For example, if a pipeline makes two copies of the input data, the
        ram_multiplier should be at least 3. Default is 1.0.
    time_multiplier : float, optional
        Wall-time in minutes per ZYX volume processed (T*C volumes total). The
        per-step calibration knob, analogous to ram_multiplier. Default is 1.0.
    max_num_cpus : int, optional
        Maximum number of available CPUs. Default is 64.
    min_ram_per_cpu : int, optional
        Minimum amount of RAM per CPU in GB. Default is 4.
    min_time_minutes : int, optional
        Minimum wall-time so tiny inputs still get a sane request. Default 30.

    Returns
    -------
    Tuple[int, int, int]
        (time_minutes, num_cpus, gb_ram_per_cpu). time_minutes is rounded up to
        the nearest 10 minutes; num_cpus and gb_ram_per_cpu map to sbatch's
        --time, --cpus_per_task, and --mem_per_cpu.
    """
    if len(shape) != 5:
        raise ValueError("The shape must be a 5-tuple (T, C, Z, Y, X).")
    if ram_multiplier <= 0 or time_multiplier <= 0:
        raise ValueError("ram_multiplier and time_multiplier must be > 0.")

    T, C, Z, Y, X = shape
    gb_per_element = np.dtype(dtype).itemsize / 2**30  # bytes_per_element / bytes_per_gb
    # In CI/tests, run serially: the test data is tiny, so spawning a worker
    # pool costs far more (per-process re-imports) than the work itself.
    num_cpus = 1 if os.environ.get("CI") == "true" else min(T * C, max_num_cpus)
    gb_ram_per_volume = Z * Y * X * gb_per_element
    gb_ram_per_cpu = np.ceil(max(min_ram_per_cpu, gb_ram_per_volume * ram_multiplier))

    # Wall-time from the number of ZYX volumes processed, scaled by the per-step
    # time_multiplier, then rounded up to the nearest 10 minutes for tidy SLURM
    # requests.
    num_volumes = T * C
    minutes = max(min_time_minutes, num_volumes * time_multiplier)
    time_minutes = int(np.ceil(minutes / 10.0) * 10)

    return time_minutes, int(num_cpus), int(gb_ram_per_cpu)
