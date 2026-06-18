"""Tile-stitch run config — Monarch is the only distributed backend.

``MonarchConfig`` is the **single source of truth** for the durable,
operator-tunable engine knobs (read from YAML — no env-var overrides of durable
knobs, no duplicate CLI flags). Only the SLURM-/runtime topology that genuinely
varies per allocation — ``--nodes``, ``--port``, ``--ready-dir``,
``--shard-by-proc`` — stays on the CLI, since it can't be a static config value.

A few **debug / topology escape-hatch** env vars remain on the backend (not
durable tuning): ``TILE_NUMA_BIND``, ``TILE_SHUTDOWN_TIMEOUT_S``,
``TILE_DRIVE_HB_S`` (and the ``_core`` read-timeout knobs). The durable
recon-dispatch knobs that used to be env vars are now config fields
(``recon_max_inflight_per_gpu``, ``recon_rpc_timeout_s``, ``recon_rpc_retries``).
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
    """Dtype the recon result is STORED + transmitted in (not the compute dtype).

    The FFT recon always runs in float32 on the GPU; this only controls the
    GPU→host D2H + RDMA payload. ``FLOAT16`` halves both the pinned-copy bytes
    and the ibverbs MR (the two halves of the per-actor ``d2h`` cost) at the
    price of ~3 significant digits — a reasonable match when the source is
    already float16, but a quality tradeoff, so the default stays lossless.
    """

    FLOAT32 = "float32"
    FLOAT16 = "float16"


class CompileMode(StrEnum):
    """``torch.compile`` mode for the recon callable.

    ``REDUCE_OVERHEAD`` captures CUDA graphs (fastest, the default);
    ``DEFAULT`` is plain torch.compile; ``NONE`` runs eager. The wire values
    match the strings ``tile_worker`` already compares against, so StrEnum
    members drop in unchanged.
    """

    REDUCE_OVERHEAD = "reduce-overhead"
    DEFAULT = "default"
    NONE = "none"


class TileCacheOrder(StrEnum):
    """Recon/stitch traversal order when the bounded tile-cache path is on.

    MORTON (Z-order) keeps a cell's contributors co-resident → smallest live band
    (measured −57% spills vs raster). RASTER = lexicographic. PLAN = the engine's
    existing ``input_order`` (only the recon-dispatch budget applies, no reorder).
    """

    MORTON = "morton"
    RASTER = "raster"
    PLAN = "plan"


class MonarchConfig(BaseModel):
    """Durable knobs for the Monarch tile-stitch engine.

    Frozen: built once from YAML and never mutated. Carried on the pickled
    ``RunPlan`` so each actor reads the same values across ``setup`` and every
    per-TP ``swap_to``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gpus_per_node: PositiveInt | None = Field(
        default=None,
        description="GPUs per node. Null (default) = auto-detect from the "
        "allocation (SLURM_GPUS_ON_NODE, else torch.cuda.device_count, which "
        "honors --gres/CUDA_VISIBLE_DEVICES), so it never has to be hand-synced "
        "with the sbatch --gres. Set explicitly only for non-SLURM boxes.",
    )
    recon_batch: PositiveInt = Field(
        default=4,
        description="Tiles per batched FFT (1 = per-tile). Bench best = 4.",
    )
    prefetch_depth: NonNegativeInt = Field(
        default=6,
        description="Background read-ahead depth in TILES (0 disables). Raw escape "
        "hatch; want >= recon_batch. Prefer prefetch_batches, which expresses this "
        "in the natural unit and can't drop below a batch.",
    )
    prefetch_batches: NonNegativeInt | None = Field(
        default=None,
        description="Background read-ahead in RECON BATCHES — the natural unit, "
        "since the consumer pulls a whole recon_batch at once. When set, the "
        "effective tile depth is prefetch_batches * recon_batch, so it auto-scales "
        "with recon_batch and never drops below one batch (the partial-overlap "
        "footgun). Takes precedence over prefetch_depth. 0 disables prefetch; 2 = a "
        "batch of slack to hide IO jitter on the IO/D2H-bound read path.",
    )
    rdma_timeout_s: PositiveInt = 60
    recon_concurrency: PositiveInt = Field(
        default=1,
        description="In-actor recon semaphore (>1 risks Tikhonov HBM OOM).",
    )
    window_per_actor: PositiveInt = Field(
        default=6, description="Stage B backpressure window (per actor)."
    )
    compile_mode: CompileMode = Field(
        default=CompileMode.NONE,
        description="torch.compile mode. Default NONE (eager): reliable across "
        "GPU counts/archs/tile shapes. REDUCE_OVERHEAD (CUDA-graph trees) trips a "
        "thread-local-storage assertion under the multi-actor worker-thread recon "
        "on >2 GPUs, and torch.compile JIT-stalls on Blackwell; opt into it only "
        "for proven single-shape, <=2-GPU runs. DEFAULT (inductor, no CUDA graphs) "
        "is a middle ground but the complex-FFT recon isn't inductor-fusible, so "
        "it gives little over eager.",
    )
    device: Device = Device.CUDA
    recon_dtype: ReconDtype = Field(
        default=ReconDtype.FLOAT32,
        description="Dtype the recon is stored + RDMA-transmitted in (compute "
        "stays float32). FLOAT16 halves the D2H copy AND the ibverbs MR bytes "
        "(both halves of the per-actor d2h cost) — cast GPU-side before the "
        "pinned copy. Lossy (~3 sig digits); default FLOAT32 is lossless. The "
        "Stage B blend upcasts to float32 regardless, so accumulation precision "
        "is unaffected — only the stored tile loses bits.",
    )

    tile_cache: bool = Field(
        default=False,
        description="Bound resident reconstructed tiles by a recon-dispatch budget "
        "+ traversal order (the P3b OOM fix). Off (default) = the legacy "
        "dispatch-all-recons path. On: recon dispatch is gated so at most "
        "``resident_budget`` tiles are resident at once — recon can't outrun Stage "
        "B and pile up host RAM.",
    )
    tile_cache_order: TileCacheOrder = Field(
        default=TileCacheOrder.MORTON,
        description="Recon/stitch traversal order when tile_cache is on.",
    )
    resident_budget: PositiveInt | None = Field(
        default=None,
        description="Max resident reconstructed tiles (recon-dispatch cap). Null = "
        "auto: the interval-overlap peak of the chosen order (the minimum that can "
        "stitch without deadlock). Higher = more recon-ahead (GPU util) at more "
        "host RAM; the cap is also raised to >= recon_batch and max output fan-in.",
    )
    recon_max_inflight_per_gpu: NonNegativeInt = Field(
        default=3,
        description="Max concurrent in-flight recon work-units PER GPU on the "
        "tile_cache (gated) path. Without a bound, all recon tasks hit the gate and "
        "fire RPCs at once, flooding the Monarch mesh until calls stop flowing "
        "(driver+workers idle, GPUs 0%). 0 = unbounded (legacy). Only applies when "
        "tile_cache is on.",
    )
    recon_rpc_timeout_s: PositiveInt = Field(
        default=90,
        description="Per-recon-RPC timeout. Monarch occasionally fails to deliver/"
        "pick up a reconstruct call under load (it never returns); a call exceeding "
        "this is re-sent (rotating GPU). Set well above normal batch latency.",
    )
    recon_rpc_retries: NonNegativeInt = Field(
        default=3,
        description="Recon-RPC re-send attempts after a timeout before failing the "
        "TP loudly.",
    )

    @property
    def effective_prefetch_depth(self) -> int:
        """Tile read-ahead depth the reader actually uses. ``prefetch_batches`` (in
        units of ``recon_batch``) takes precedence over the raw ``prefetch_depth``,
        so a batch-expressed setting can't fall below one batch."""
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
