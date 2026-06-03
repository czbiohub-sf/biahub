# Stage 3 Design — MonarchConfig + unified CLI/API

Scope: Phase A design for Stage 3 of `monarch_packaging_plan.md` (config + API
surface), plus the dependent deletions/trim the lead folded in (Stages 4–5).
Stages 1 (`_core.py` + `test_core.py`) and 2 (`MonarchBackend`) already landed;
this stage unifies `cli.py` on Monarch, replaces the pool configs with
`MonarchConfig`, deletes the Dask path, and trims deps.

This is a structural + dependency change only. No recon/blend math or output
format changes — the Monarch path is already parity-verified.

---

## 0. Lead rulings (locked) — supersede the Open Questions below

1. **CPU device — DEFER.** Land the `device` field; `cpu` raises a clear
   `NotImplementedError` at backend `setup()` ("CPU device not yet wired; use
   cuda"). cuda-only at runtime this stage; do NOT half-wire the actor CUDA
   guards. The field + an explicit raise is the honest middle.
2. **flag > config precedence — YES.** CLI flag wins when set (default `None`),
   else config.
3. **Plan-carried config — YES.** Two guards: (a) `RunPlan.monarch` is
   **optional with a default** so a plan without it (or an older pickle) still
   loads; (b) confirm the (frozen) `MonarchConfig` pickles cleanly through
   plan → actor → swap — add a tiny round-trip assert in `test_core`.
4. **frozen + StrEnum — YES.** `MonarchConfig` is `frozen=True` (lead affirmed;
   guard 3b verifies pickle). `use_enum_values` stays unset; call `.value` at the
   env boundary. (This overrides the memo's general "skip blanket frozen" — the
   lead explicitly wants the leaf frozen and the pickle path checked.)
5. **Naming — rename `monarch_pool` → `monarch`** on `TileStitchRun` ("pool" is a
   vestigial Dask-ism with one backend).

---

## 1. `MonarchConfig` (config.py)

### 1.1 Field surface (per plan §6)

```python
class MonarchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)   # lead ruling 4
    gpus_per_node: PositiveInt = 2
    recon_batch: PositiveInt = 4          # tiles/batched FFT (1 = per-tile); bench best = 4
    prefetch_depth: NonNegativeInt = 6    # bg read-ahead depth; 0 disables; want >= recon_batch
    resident_volume: bool = False         # opt into GPU-resident volume
    rdma_timeout_s: PositiveInt = 60
    recon_concurrency: PositiveInt = 1    # in-actor recon semaphore
    window_per_actor: PositiveInt = 6     # Stage B backpressure window
    compile_mode: CompileMode = CompileMode.REDUCE_OVERHEAD
    device: Device = Device.CUDA          # CPU = device knob (deferred; raises at setup)
```

`MonarchConfig` does NOT inherit `_Base` — it needs its own frozen `model_config`.
(`_Base` stays the mutable `extra="forbid"` base for `TileStitchRun`.)

Dropped: `CpuSlurmConfig`, `GpuLocalCudaConfig`, `GpuSlurmConfig` classes and the
`cpu_pool` / `gpu_pool` / `gpu_slurm_pool` fields on `TileStitchRun`.

`TileStitchRun` becomes (note the `monarch` field name — lead ruling 5):

```python
class TileStitchRun(_Base):
    tile_stitch: TileStitchSettings
    monarch: MonarchConfig = Field(default_factory=MonarchConfig)
    retries: NonNegativeInt = 1           # advisory — Monarch fails loud (Risks E5)
    run_dir: str
```

### 1.2 py312 / pydantic-v2 patterns applied

Reconciled against `researcher`'s `.planning/py312_patterns.md` (ADOPT 1–6 +
SKIP list). Adopting rules **1** (model_validator+Self), **2** (CM lifecycle,
§2.2), **4** (StrEnum). `frozen=True` on `MonarchConfig` per lead ruling 4 (the
memo's general "skip blanket frozen" is overridden for this one leaf — see §0).
Deliberately NOT adopting **3** (Protocol), **5** (computed_field),
**7** (discriminated union) — rationale in §9.

**StrEnum for `device` and `compile_mode`** (ADOPT 4, replacing `Literal`):

```python
from enum import StrEnum

class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"

class CompileMode(StrEnum):
    REDUCE_OVERHEAD = "reduce-overhead"
    DEFAULT = "default"
    NONE = "none"
```

(Uppercase member names per the memo's example; values are the wire strings.)

Rationale and why it's safe here:
- StrEnum members **are** `str`, so every existing `mode == "none"` / `device ==
  "cuda"` comparison in `tile_worker.py` keeps working unchanged when we pass
  `config.compile_mode.value` (or the member itself — `CompileMode.NONE ==
  "none"` is True).
- Pydantic v2 serializes StrEnum back to the plain string in `model_dump`/YAML,
  so config round-trips identically to the `Literal` version — no YAML churn for
  operators. `use_enum_values=True` is **not** set; we keep enum instances on the
  model and call `.value` at the env-override boundary (§2.3).
- Only `device`/`compile_mode` become enums (value sets reused at the actor's
  point-of-use dispatch); single-use scalars stay plain (memo rule-4 caveat —
  don't manufacture an enum per field).

**`@model_validator(mode="after") -> Self`** for cross-field checks (ADOPT 1,
§1.3). Use `typing.Self` as the return annotation.

**`frozen=True` on `MonarchConfig`** (lead ruling 4): built once from YAML, never
mutated; freezing makes that explicit. No `cached_property` on this model (memo
rule 5 not adopted), so the frozen+cached interaction the memo warns about
doesn't apply. The config is carried on the pickled plan — guard 3b adds a
round-trip pickle assert in `test_core` to confirm the frozen model survives
plan → actor → swap.

### 1.3 Cross-field validators

```python
@model_validator(mode="after")
def _check_resident_batch(self) -> Self:
    # Resident volume (~34 GB) + Tikhonov (~30 GB) ≈ 64 GB on an 80 GB H200;
    # a batched (B,Z,Y,X) stack on top exceeds HBM (plan E4).
    if self.resident_volume and self.recon_batch > 1:
        raise ValueError(
            "resident_volume=True is incompatible with recon_batch>1 "
            "(HBM OOM: ~34 GB volume + ~30 GB Tikhonov + batched stack > 80 GB). "
            "Set recon_batch=1 or resident_volume=False."
        )
    return self

@model_validator(mode="after")
def _warn_prefetch_lt_batch(self) -> Self:
    # Partial overlap: a full next batch can't be buffered during the current
    # FFT, so prefetch only partially hides IO (plan E2 note).
    if 0 < self.prefetch_depth < self.recon_batch:
        warnings.warn(
            f"prefetch_depth ({self.prefetch_depth}) < recon_batch "
            f"({self.recon_batch}): prefetch can't stay a full batch ahead; "
            "IO overlap will be partial. Set prefetch_depth >= recon_batch.",
            stacklevel=2,
        )
    return self
```

Note the **default config trips neither**: defaults are `resident_volume=False`,
`recon_batch=4`, `prefetch_depth=6` — reject-clause false, warn-clause false
(6 >= 4). Good: the out-of-box config is valid and quiet.

---

## 2. API surface — how cli.py + MonarchBackend + MonarchConfig compose

### 2.1 Responsibility split (unchanged from plan §4/§5, made concrete)

| Concern | Owner |
|---|---|
| Pydantic config (durable knobs) | `MonarchConfig` (in YAML) |
| SLURM-/runtime-dependent topology | CLI flags (`--nodes/--port/--gpus-per-node/--ready-dir/--shard-by-proc`) |
| TP resolve/shard, engine plan, output zarr, per-TP plan pickles | `cli.py` scaffold |
| Mesh setup / drive / swap / teardown | `MonarchBackend` |

`gpus_per_node` exists in **both** `MonarchConfig` (default 2) and as a CLI flag
(`--gpus-per-node`, default None). Resolution rule: **CLI flag wins when set,
else config value**. This mirrors the m3_driver behavior (flag defaulted to
local device count) while letting the YAML carry a sane default. Documented in
the CLI help.

### 2.2 Backend lifecycle as a context manager

`MonarchBackend` gains `__enter__`/`__exit__` so the CLI uses a `with` block,
guaranteeing `teardown()` even on exceptions (today m3_driver calls
`backend.teardown()` only on the success path — a drive failure leaks the mesh).

```python
class MonarchBackend:
    def __enter__(self) -> "MonarchBackend":
        return self
    def __exit__(self, *exc) -> None:
        self.teardown()
```

`setup()` stays explicit (called inside the `with`, after construction) because
it needs `first_plan_path`, which the scaffold computes. Pattern:

```python
with MonarchBackend(gpus_per_node=gpn, nodes=node_list, port=port,
                    ready_dir=ready_dir,
                    window_per_actor=run.monarch.window_per_actor) as backend:
    backend.setup(plan_entries[0].plan_path)
    for i, (tp, plan_path, plan) in enumerate(plan_entries):
        if i > 0:
            backend.swap(plan_path)
        summary = backend.drive_tp(plan_path, plan, recon_batch=run.monarch.recon_batch)
        ...log recon_stats...
```

`recon_batch` resolution: CLI `--recon-batch` flag wins when explicitly passed,
else `config.recon_batch`. To detect "explicitly passed", the CLI flag default
becomes `None` (not `1`); `recon_batch = recon_batch_flag if recon_batch_flag is
not None else run.monarch.recon_batch`. (Keeps the bench-best default of 4
from config while letting a one-off override on the command line.)

### 2.3 Threading config knobs into the actor (plan §6 / reviewer E3)

This is the load-bearing plumbing, not a one-liner. Four knobs are read directly
in `tile_worker.py` with hardcoded defaults today:

| Knob | Read site (tile_worker.py) | Config field | Default today |
|---|---|---|---|
| `TILE_STITCH_RESIDENT_VOLUME` | `__init__` (`:165`) | `resident_volume` | off |
| `TILE_STITCH_PREFETCH_DEPTH` | `prime_reader` (`:480`) | `prefetch_depth` | 2 |
| `TILE_STITCH_RDMA_TIMEOUT_S` | `stitch` (`:818`) | `rdma_timeout_s` | 60 |
| `TILE_STITCH_COMPILE_MODE` | `_get_compiled_recon` (`:282`) | `compile_mode` | reduce-overhead |
| (also) `recon_concurrency` | recon semaphore — **TWO sites:** `reconstruct` + `reconstruct_batch` (both lazily build `Semaphore(1)`) | `recon_concurrency` | 1 |
| (also) `device` | every `cuda:{idx}` / `set_device` site | `device` | cuda |

`recon_concurrency` must be rewired at **both** semaphore sites or the batched
path silently keeps concurrency=1 while per-tile honors config (reviewer
should-fix #2).

**`TILE_STITCH_GPU_OVERLAP` (`:183`) and `TILE_STITCH_GPU_DEPTH` (`:195`) stay
env-only experimental gates, NOT config.** They drive the v2 GPU H2D-overlap /
cross-batch staging path, which is opt-in and not bench-validated; promoting
them to `MonarchConfig` would imply they're supported knobs. There are six env
reads in `tile_worker.py` total — the four above become config-backed; these two
remain env-gated experiments.

**Mechanism — env-OR-config, config threaded through plan:**

The clean carrier is the **`RunPlan`** (already pickled per-TP, already loaded in
`TileWorker.__init__` and re-loaded in `swap_to`). `RunPlan` already holds
`settings` (the `TileStitchSettings`). I will add the resolved `MonarchConfig`
(or just the six scalar knobs) to the plan at `from_engine_plan` time, so it
survives the pickle → actor → swap path with **zero new actor constructor args**
and no per-call passing. The actor reads `self.plan.monarch` once.

Each env read becomes `env_or(self.plan.monarch.<field>)`:

```python
def _env_or(name: str, default):
    v = os.environ.get(name)
    return type(default)(v) if v is not None else default
```

- `resident_volume`: `self._stream_tiles = not _env_or_bool("TILE_STITCH_RESIDENT_VOLUME", cfg.resident_volume)`
- `prefetch_depth`: `depth = _env_or("TILE_STITCH_PREFETCH_DEPTH", cfg.prefetch_depth)`
- `rdma_timeout_s`: `rdma_timeout = _env_or("TILE_STITCH_RDMA_TIMEOUT_S", cfg.rdma_timeout_s)`
- `compile_mode`: `mode = os.environ.get("TILE_STITCH_COMPILE_MODE", cfg.compile_mode.value)`
- `recon_concurrency`: `asyncio.Semaphore(cfg.recon_concurrency)` (no env knob — config-only)
- `device`: drives `cuda:{idx}` vs `cpu` (plan §11.1; see §2.4)

Env override is preserved for all four existing knobs (operators' sbatch scripts
that set them keep working); config supplies the default when env is unset.

**Why plan-carried, not constructor-arg:** `setup()` spawns the actor with
`plan_path=first_plan_path`; `swap_to` reloads a new plan_path each TP. Carrying
the config on the plan means a single source threads through both paths with no
backend signature change and no risk of the swap path forgetting to re-pass it.
`MonarchConfig` is frozen + pydantic → cleanly picklable.

### 2.4 CPU as a device knob (plan §11.1)

`device="cpu"` requires `tile_worker.py` to guard CUDA-only calls. Out of scope
for the *config/API design* sign-off, but the design admits it: the config field
exists, `_core` functions already take an explicit `device` string, and the
actor will branch `set_device/synchronize/empty_cache/torch.compile(reduce-
overhead→default)` on `device == "cuda"`. I will implement the guards in Phase B
since the field is config-visible; if the reviewer wants CPU-device deferred to a
later stage (it has no shipping requirement per Risks), I'll gate the field
behind a `NotImplementedError` for `cpu` instead of half-wiring it. **Question
for reviewer:** wire CPU guards now, or land `device` as cuda-only + explicit
`raise` for cpu this stage? Default proposal: cuda-only + explicit raise (keep it
simple; no untested CPU path shipped).

---

## 3. cli.py after refactor

Single Monarch path. No dask imports survive (reviewer D2). Structure:

```python
@click.command("tile-stitch", no_args_is_help=True)
# unchanged: --config -c, --input -i, --output -o, --timepoint, --timepoints, --channel
# --recon-batch: default None (None → use config.recon_batch)
# ADDED (from m3_driver): --nodes, --port, --gpus-per-node, --ready-dir, --shard-by-proc
# REMOVED: --gpu, --gpu-slurm, --scheduler-file
def tile_stitch_cli(config, input_path, output_path, timepoint, timepoints,
                    channel, recon_batch, nodes, port, gpus_per_node,
                    ready_dir, shard_by_proc):
    run = TileStitchRun.model_validate(yaml.safe_load(config.read_text()))
    global_tps = resolve_tps(timepoints, timepoint)         # --timepoints wins, else [timepoint or 0]
    procid, nprocs, tps = shard_tps(global_tps, shard_by_proc)   # SLURM_PROCID/NTASKS split
    if not tps: return                                       # empty shard exits
    engine_plan = build engine plan from tps[0]              # geometry once, batch_size=None
    create_output_zarr(final_output, full_shape(global_tps), chunk, gate_on_procid=procid)
    plan_entries = [write per-TP plan pickle for tp in tps]  # plan carries run.monarch

    gpn = gpus_per_node if gpus_per_node is not None else run.monarch.gpus_per_node
    rb  = recon_batch  if recon_batch  is not None else run.monarch.recon_batch
    with MonarchBackend(gpus_per_node=gpn, nodes=split(nodes), port=port,
                        ready_dir=ready_dir,
                        window_per_actor=run.monarch.window_per_actor) as backend:
        backend.setup(plan_entries[0].plan_path)
        for i, (tp, plan_path, plan) in enumerate(plan_entries):
            if i > 0: backend.swap(plan_path)
            summary = backend.drive_tp(plan_path, plan, recon_batch=rb)
            log recon_stats + per-tp wall
    write walls_proc{procid}.json (or walls.json single-proc)
```

The TP-shard split, zarr-gate, and per-shard walls logic move **verbatim from
m3_driver** into cli.py (m3_driver is then deleted). This folds the m3_driver
shim's 11 options into the one registered CLI.

`m3_driver.py` deleted; the sbatch `python -m ...m3_driver` invocations become
`biahub tile-stitch`. One driver.

---

## 4. `__init__.py`

Currently re-exports `CpuSlurmConfig`, `GpuLocalCudaConfig`, `TileStitchRun` in
`__all__`. Dropping the pool classes from `config.py` without this breaks
`import biahub.tile_stitch` (surfaces at CLI dispatch — reviewer C3). New:

```python
from biahub.tile_stitch.config import MonarchConfig, TileStitchRun
__all__ = ["MonarchConfig", "TileStitchRun"]
```

Same commit as the `config.py` change.

---

## 5. Deletions (folded Stages 4 + the distributed cut)

- Dask path: `dispatcher.py`, `pipeline.py`, `workers.py`, `monarch/m3_driver.py`.
- Monarch scaffolding now redundant: `monarch/m0_hello.py`, `monarch/m1_smoke.py`,
  `monarch/multinode_probe.py` (smoke value → `test_monarch_smoke.py`, GPU-gated).
- `biahub/tile_stitch/distributed/` (cuFFTMp experiment, e0–e4/parity/analyze) —
  the only importer of `nvmath-python` (plan §11.2).

Pre-deletion grep gate (Risks) — **ALREADY RUN, PASSES:** `cupy` / `dask_cuda` /
`rmm` / `ucxx` have zero source hits anywhere outside the delete-list; `nvmath`
appears only in `distributed/` (being cut). `dask`/`distributed` ARE used broadly
across biahub via `dask.array` (`track.py`, `estimate_stabilization.py`,
`estimate_crop.py`, `registration/{ants,beads}.py`, `visualize/animation_utils.py`,
`vendor/stitch/tile.py`) — so `dask`/`dask-jobqueue`/`numba`/`llvmlite` stay as
base deps (reviewer C1), exactly as the plan predicted.

---

## 6. pyproject + lock (Stage 5, own commit, last)

`tilestitch-gpu` extra: drop `dask-cuda`, `cupy-cuda13x`, `rmm-cu13`,
`numba-cuda`, `ucxx-cu13`, `distributed-ucxx-cu13`. Fold `torchmonarch` in:

```toml
tilestitch-gpu = [
  "torchmonarch>=0.1.0",
]
```

`tilestitch-distributed` extra: removed entirely (its only members were
`nvmath-python` — now orphaned — and `torchmonarch`, moved to `tilestitch-gpu`).
`uv lock` after; dep edits in their own final commit so a lock regression is
bisectable.

---

## 7. Tests

- `test_smoke.py`: replace `cpu_pool`/`gpu_pool` assertions with `monarch_pool`
  defaults + the two validators (assert default config valid; assert
  `resident_volume=True, recon_batch=2` raises; assert `prefetch_depth=1,
  recon_batch=4` warns). Same commit as `config.py`.
- `test_core.py`: unchanged, must stay green (the parity guard incl. `t_off>0`).
- `test_monarch_smoke.py`: GPU+monarch-gated, skips in CI (replaces m1_smoke).
  Added in the deletion commit.

---

## 8. Verification (Phase B exit)

- `ruff check biahub/tile_stitch tests/tile_stitch`
- AST parse of every touched file
- `uv run --no-sync python -m pytest tests/tile_stitch -q` (CPU-only green)
- `uv run --no-sync python -c "import biahub.tile_stitch"` (import succeeds)
- No SLURM submission (lead runs GPU integration separately).

---

## 9. py312-memo patterns NOT adopted (deliberate, with rationale)

- **Rule 3 — Protocol backend seam + `@override`:** SKIP. The whole point of
  Stage 3 is that Monarch is the *only* backend (the dask path is being deleted
  this stage). A `Backend` Protocol exists to let multiple implementations
  satisfy one contract — with exactly one implementation and no second backend
  on the roadmap, it's a speculative abstraction over a single class, which the
  team's keep-it-simple guidance explicitly flags as a smell. If a CPU/mock
  backend is ever added, introduce the Protocol then.
- **Rule 5 — `@computed_field` + `cached_property`:** SKIP. The one derived value
  the memo cites (`window = window_per_actor * n_gpus`, `backend.py:242`) depends
  on the *runtime* `n_gpus` (resolved at `setup()` from the live allocation), not
  on config alone — it can't be a `MonarchConfig` computed field. `n_actors =
  nodes * gpus_per_node` likewise needs the CLI `nodes` flag, which isn't config.
  No purely-config-derived quantity is worth caching here.
- **Rule 7 — discriminated union for pools:** N/A. The three `*_pool` fields are
  being *deleted*, not collapsed into a tagged union — there is one config
  (`MonarchConfig`), so there's nothing to discriminate.
- **Blanket `frozen=True`:** SKIP per memo (and §1.2).

## 10. Open questions for reviewer

1. **CPU device this stage?** Proposal: land `device` field but cuda-only at
   runtime (`raise NotImplementedError` for `cpu`), defer the actor CUDA-guards
   to a follow-up — no shipping CPU requirement (Risks). Or wire the guards now?
2. **`gpus_per_node` / `recon_batch` flag-vs-config precedence:** proposal is CLI
   flag wins when set (flag default `None`), else config. OK?
3. **Knob carrier:** thread `MonarchConfig` via the pickled `RunPlan`
   (`plan.monarch`) vs new `MonarchBackend`/`TileWorker` constructor args.
   Proposal: plan-carried (survives setup+swap with no signature churn). OK?
