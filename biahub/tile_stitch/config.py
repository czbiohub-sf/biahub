"""Tile-stitch run config — Monarch is the only distributed backend.

``MonarchConfig`` is the **single source of truth** for the durable,
operator-tunable engine knobs (read from YAML — no env-var overrides of durable
knobs, no duplicate CLI flags). Only the SLURM-/runtime topology that genuinely
varies per allocation — ``--nodes``, ``--port``, ``--ready-dir``,
``--shard-by-proc`` — stays on the CLI, since it can't be a static config value.

A few **debug escape-hatch** env vars remain on the backend (not durable
tuning): ``TILE_SHUTDOWN_TIMEOUT_S``, ``TILE_DRIVE_HB_S`` (and the ``_core``
read-timeout knobs).
"""

import warnings

from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)
from waveorder.api.tile_stitch import TileStitchSettings


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Device(StrEnum):
    """Recon device. CPU is a device knob, not a separate code path."""

    CUDA = "cuda"
    CPU = "cpu"


class ReconDtype(StrEnum):
    """Dtype the recon is stored + RDMA-transmitted in (compute stays float32)."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"


class CompileMode(StrEnum):
    """``torch.compile`` mode for the recon callable (wire values match tile_worker)."""

    REDUCE_OVERHEAD = "reduce-overhead"
    DEFAULT = "default"
    NONE = "none"


class MonarchConfig(BaseModel):
    """Durable knobs for the Monarch tile-stitch engine.

    Frozen: built once from YAML and never mutated. Carried on the pickled
    ``RunPlan`` so each actor reads the same values across ``setup`` and every
    per-TP ``swap_to``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gpus_per_node: PositiveInt | None = Field(
        default=None,
        description="GPUs per node; null = auto-detect from the SLURM allocation.",
    )
    recon_batch: PositiveInt = Field(
        default=4, description="Tiles per batched FFT (1 = per-tile); bench best = 4."
    )
    prefetch_depth: NonNegativeInt = Field(
        default=6,
        description="Read-ahead depth in TILES (0 disables); prefer prefetch_batches.",
    )
    prefetch_batches: NonNegativeInt | None = Field(
        default=None,
        description="Read-ahead in recon-batches (effective depth = batches * "
        "recon_batch); overrides prefetch_depth, 0 disables.",
    )
    rdma_timeout_s: PositiveInt = 60
    # >1 stacks Tikhonov intermediates on one GPU and risks HBM OOM.
    recon_concurrency: PositiveInt = Field(
        default=1, description="In-actor concurrent recon limit."
    )
    window_per_actor: PositiveInt = Field(
        default=6, description="Stage B backpressure window (per actor)."
    )
    # REDUCE_OVERHEAD trips a thread-local-storage assertion on >2 GPUs and
    # JIT-stalls on Blackwell; eager (NONE) is the reliable default. Opt in only
    # for proven single-shape, <=2-GPU runs.
    compile_mode: CompileMode = Field(
        default=CompileMode.NONE, description="torch.compile mode; default eager."
    )
    device: Device = Device.CUDA
    recon_dtype: ReconDtype = Field(
        default=ReconDtype.FLOAT32,
        description="Stored/transmitted recon dtype; float16 halves D2H+RDMA bytes "
        "(lossy ~3 digits), compute stays float32.",
    )

    bounded_dispatch: bool = Field(
        default=False,
        description="Gate recon dispatch to a resident budget over a Morton sweep "
        "(off = unbounded dispatch-all).",
    )
    resident_budget: PositiveInt | None = Field(
        default=None,
        description="Max resident recon tiles; null = auto (the order's overlap "
        "peak). Raised to >= recon_batch and max output fan-in.",
    )
    # 0 = unbounded: every recon task hits the gate and fires RPCs at once,
    # flooding the Monarch mesh until calls stop flowing (driver+workers idle).
    recon_max_inflight_per_gpu: NonNegativeInt = Field(
        default=3,
        description="Max in-flight recon RPCs per GPU on the gated path; only when "
        "bounded_dispatch is on.",
    )
    recon_rpc_timeout_s: PositiveInt = Field(
        default=90,
        description="Per-recon-RPC timeout; a call exceeding it is re-sent (rotating GPU).",
    )
    recon_rpc_retries: NonNegativeInt = Field(
        default=3, description="Recon-RPC re-send attempts after a timeout before failing."
    )

    @property
    def effective_prefetch_depth(self) -> int:
        """Tile read-ahead depth the reader actually uses.

        ``prefetch_batches`` (in units of ``recon_batch``) takes precedence over
        the raw ``prefetch_depth``, so a batch-expressed setting can't fall below
        one batch.
        """
        if self.prefetch_batches is not None:
            return self.prefetch_batches * self.recon_batch
        return self.prefetch_depth

    @model_validator(mode="after")
    def _warn_prefetch_below_batch(self) -> Self:
        # A prefetch depth below the batch size can't keep a whole next batch
        # buffered during the current FFT, so IO overlap is only partial. This
        # is a perf footgun, not invalid — warn, don't raise. Checked on the
        # EFFECTIVE depth, so a prefetch_batches setting (which is batch-derived)
        # never trips it; only a raw prefetch_depth < recon_batch does.
        d = self.effective_prefetch_depth
        if 0 < d < self.recon_batch:
            warnings.warn(
                f"effective prefetch depth ({d}) < recon_batch "
                f"({self.recon_batch}): prefetch can't stay a full batch ahead; "
                "IO overlap will be partial. Set prefetch_batches >= 1 (or "
                "prefetch_depth >= recon_batch).",
                stacklevel=2,
            )
        return self


class TileStitchRun(_Base):
    """Top-level run config: tile_stitch settings + the Monarch engine knobs."""

    tile_stitch: TileStitchSettings = Field(
        description="waveorder TileStitchSettings (tile + blend + recon)"
    )
    monarch: MonarchConfig = Field(default_factory=MonarchConfig)
    retries: NonNegativeInt = Field(
        default=1,
        description="Advisory only — Monarch fails loud (no per-task retry yet).",
    )
    run_dir: str = Field(description="absolute path for run artifacts")
