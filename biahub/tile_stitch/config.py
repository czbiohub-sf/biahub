"""Flat config — mirrors the legacy v4_dask SlurmConfig field-for-field.

We deliberately match the legacy field layout so operators familiar
with the experimental driver map their YAMLs over without re-thinking.
Validators are intentionally minimal — Pydantic basics + cluster_type
gating. Everything else is "user knows what they're doing"; if a value
breaks at runtime the SLURM/dask error tells you exactly what.
"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
)
from waveorder.api.tile_stitch import TileStitchSettings


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CpuSlurmConfig(_Base):
    """SLURMCluster kwargs for CPU dask-jobqueue workers (legacy v4 v6 path)."""

    partition: str = Field(default="cpu")
    workers_min: PositiveInt = 1
    workers_max: PositiveInt = 8
    worker_cores: PositiveInt = 16
    worker_memory: str = "64GB"
    worker_walltime: str = "01:00:00"
    dask_threads: PositiveInt = 1
    blas_threads: PositiveInt = 16
    zarr_async_concurrency: PositiveInt = 4
    zarr_threading_max_workers: PositiveInt = 4
    preempt_signal_secs: NonNegativeInt = 90
    scratch_dir: str = Field(
        description="dask local_directory — point at a project mount with TB free, NOT /tmp"
    )
    pool_mode: Literal["scale", "adapt"] = "adapt"
    prewarm_workers: NonNegativeInt = 0
    prewarm_timeout_s: PositiveInt = 600
    batch_size: PositiveInt = Field(
        default=4,
        description="K input tiles per Stage A batched dispatch (c0032 winner=4)",
    )


class GpuLocalCudaConfig(_Base):
    """LocalCUDACluster kwargs for single-node GPU runs (legacy v7 c0041 path)."""

    rmm_pool_size: str = "30GB"
    protocol: Literal["tcp", "ucxx"] = "tcp"
    gpu_constraint: str | None = None  # for sbatch-side --constraint, not used here


class TileStitchRun(_Base):
    """Top-level run config: tile_stitch settings + one of cpu/gpu pool."""

    tile_stitch: TileStitchSettings = Field(
        description="waveorder TileStitchSettings (tile + blend + recon)"
    )
    cpu_pool: CpuSlurmConfig | None = None
    gpu_pool: GpuLocalCudaConfig | None = None
    retries: NonNegativeInt = 1
    run_dir: str = Field(description="absolute path for run artifacts")
