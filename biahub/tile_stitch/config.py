"""Tile-stitch run config — Monarch is the only distributed backend.

``MonarchConfig`` is the **single source of truth** for the durable,
operator-tunable engine knobs (read from YAML — no env-var overrides, no
duplicate CLI flags). Only the SLURM-/runtime topology that genuinely varies
per allocation — ``--nodes``, ``--port``, ``--ready-dir``, ``--shard-by-proc``
— stays on the CLI, since it can't be a static config value.
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
        description="Background read-ahead depth (0 disables). Want >= recon_batch.",
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

    @model_validator(mode="after")
    def _warn_prefetch_below_batch(self) -> Self:
        # A prefetch depth below the batch size can't keep a whole next batch
        # buffered during the current FFT, so IO overlap is only partial. This
        # is a perf footgun, not invalid — warn, don't raise.
        if 0 < self.prefetch_depth < self.recon_batch:
            warnings.warn(
                f"prefetch_depth ({self.prefetch_depth}) < recon_batch "
                f"({self.recon_batch}): prefetch can't stay a full batch ahead; "
                "IO overlap will be partial. Set prefetch_depth >= recon_batch.",
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
