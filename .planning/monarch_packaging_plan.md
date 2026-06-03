# Monarch Tile-Stitch — Packaging & Dask Replacement Plan

**Decisions (locked):**
1. **Hard replace Dask.** Monarch becomes the only distributed backend. Delete
   `dispatcher.py` / `pipeline.py` / `workers.py` (Dask) and the dask cluster
   flags/config. One code path.
2. **Extract shared `_core` now.** Lift the backend-neutral compute (geometry,
   TF prep, tile-load, recon, blend) out of the monolithic `tile_worker.py`
   into `_core.py`. The actor becomes a thin Monarch transport shell.

**Non-goal:** This does not change the recon math, blend math, or output
format. It is a structural refactor + dependency swap. Output must remain
bit-for-bit identical to the current Monarch path (which already has parity
with the Dask path from prior c0041/c0032 reproduction).

> **Review verdict (incorporated below): execute-with-fixes.** The single
> load-bearing correction: **all `_core` compute is lifted EXCLUSIVELY from
> `tile_worker.py`, never from `workers.py`.** `workers.py` carries two
> different leading-axis conventions — `stitch_output_tile_v6` uses
> `n_lead = len(leading_shape)` (**includes T**, `workers.py:238`) while the v7
> path and `tile_worker` use `leading_shape[1:]` (**strips T**, writes T via an
> explicit `slice(t_off, t_off+1)`). A lift that pulls from the v6 CPU blend
> would corrupt every multi-TP write. `workers.py` is reference-for-deletion
> only. All §3/§9/§10 text below reflects this.

---

## 1. Current state — a fork, not an integration

| Concern | Dask (committed, PR #246) | Monarch (untracked research) |
|---|---|---|
| CLI / scaffold | `cli.py` | `monarch/m3_driver.py` (**duplicates** all of cli.py) |
| Worker bring-up | `dispatcher.py` (cluster factories) | inline in `m3_driver.main` (this_host / attach_to_workers) |
| Per-TP drive | `pipeline.py` `drive_pipelined_v7` | `m3_driver._drive_one_tp` |
| Compute | `workers.py` (recon + GPU blend + geom) | `tile_worker.py` (recon + CPU blend + geom + RDMA + resident-vol + prefetch) |
| Plan / zarr | `plan.py`, `_zarr_util.py` | **shared** (both import these) |

The Monarch path re-implements the entire CLI scaffold and the compute. The
packaging job is to collapse that into the PR #246 module shape, with Monarch
as the engine.

Key compute nuance discovered during analysis:
- Dask `stitch_output_tile_v7` blends on **GPU** (torch/cuda accumulator);
  contributors arrive as cupy arrays over dask comms.
- Monarch `stitch` blends on **CPU** (numpy) deliberately — GPU blend contends
  with concurrent recon on the same device; contributors arrive via RDMA as CPU
  tensors.
- **We keep the Monarch CPU-blend** (it is the proven, shipping path). So
  `_core` shares the *pure* logic (geometry, TF, tile-load, recon, kernel cache)
  and the **CPU numpy blend**; the GPU-blend variant in `workers.py` is deleted
  with the rest of the Dask path.

---

## 2. Target module layout

```
biahub/tile_stitch/
  __init__.py
  cli.py            # unified entry: scaffold + MonarchBackend (dask flags removed)
  config.py         # TileStitchRun: drop cpu/gpu/gpu_slurm pools; add monarch_pool
  plan.py           # UNCHANGED (RunPlan, from_engine_plan, write/load_plan)
  _zarr_util.py     # UNCHANGED (create_multi_tp_zarr, parse_timepoints)
  _core.py          # NEW — backend-neutral compute (see §3)
  monarch/
    __init__.py
    backend.py      # NEW — MonarchBackend: mesh setup, drive_tp, swap, teardown (lifts m3_driver internals)
    tile_worker.py  # SLIMMED — Monarch actor = transport shell calling _core
    worker_loop.py  # KEEP — multi-host run_worker_loop_forever entry
  # DELETE: dispatcher.py, pipeline.py, workers.py
  # DELETE: monarch/m0_hello.py, monarch/m1_smoke.py, monarch/m3_driver.py, monarch/multinode_probe.py
  #         (smoke/probe value preserved as a pytest; m3_driver logic → backend.py + cli.py)
```

`scripts/distributed/*` and `settings/distributed/*` (untracked) are pruned to
the few sbatch wrappers we actually ship under `scripts/tile_stitch/` (single
node, tp-shard, multi-node), matching the PR #246 convention
(`scripts/tile_stitch/example_*`).

---

## 3. `_core.py` — the shared compute API

Pure functions + small caches, no Monarch / no Dask imports. **Every function
is cut-pasted from `tile_worker.py`** (the device-correct, multi-actor-safe
variant), not synthesized from a "merge" of the two backends. Proposed surface:

```python
# --- transfer function (ONE module-level cache, keyed (settings_json, device)) ---
def get_tf_cuda(settings, device: str) -> tuple[dict[str, Tensor], ReconSettings]
    # lifts tile_worker._ensure_tf VERBATIM. Cache key MUST be
    # (settings.model_dump_json(), device) and tensors built on the exact
    # device string (e.g. "cuda:1") — the gpu_idx in the key is load-bearing
    # for multi-actor-in-one-process correctness (tile_worker.py:451,462-467).
    # The single _TF_CUDA_CACHE replaces BOTH old module caches.

# --- tile IO ---
def load_tile_zyx(plan, tile, *, volume=None, device: str) -> Tensor   # (Z,Y,X) f32
    # lifts tile_worker._load_tile_gpu. Resident path keeps
    # `.to(float32).contiguous().clone()` — the clone is REQUIRED so
    # torch.compile reduce-overhead CUDA-graph capture sees a stable buffer
    # (tile_worker.py:519). Streams to device=f"cuda:{idx}", not bare "cuda".

def read_tile_block(plan, tile) -> np.ndarray    # (Z,Y,X) source dtype — leaf used by the actor's per-TP PrefetchReader loader closure

# --- recon ---
def make_eager_recon(cuda_tf, recon_settings) -> Callable   # zyx -> (Z,Y,X) recon
    # the bare apply_inverse_transfer_function closure (tile_worker._eager).
    # torch.compile wrapping + the compiled-callable cache STAY IN THE ACTOR
    # (_get_compiled_recon is actor-resident state — see B4 below); _core only
    # provides the pure eager closure both the eager and compiled paths build on.

# --- geometry (lifted from tile_worker._build_stitch_geom; uses leading_shape[1:]) ---
def build_stitch_geom(plan) -> dict[int, dict]

# --- blend (CPU numpy — lifted from tile_worker.stitch._blend_and_write ONLY) ---
def get_blend_kernel(blend, tile_shape, dtype, cache) -> np.ndarray
def blend_contributors(geom_entry, contribs_np, blend_kernel, kernel_cache) -> np.ndarray
    # weighted-Gaussian accumulate + divide; zarr write stays in the actor.
    # NOT derived from workers.stitch_output_tile_v6 (different leading-axis convention).

# --- prefetch (backend-neutral) ---
class PrefetchReader:        # moved verbatim from tile_worker.py
    ...                      # the per-TP _load closure + reader lifecycle (start/stop on swap) STAY in the actor
```

**Where compiled-recon lives (resolves reviewer B4):** `_get_compiled_recon`
holds `torch.compile` state + a TF closure and is inherently actor-resident. It
stays in the actor and calls `_core.make_eager_recon(...)` to build the eager
fn it compiles. `_core` does **not** own the compiled callable. This keeps the
hot batch path (`_reconstruct_batch_blocking` → `_get_compiled_recon`) intact.

Caches: the single `_core._TF_CUDA_CACHE` (key includes device) and the kernel
cache are multi-actor-safe.

---

## 4. `MonarchBackend` — `monarch/backend.py`

Lifts the mesh/setup/drive logic out of `m3_driver.py`. Single class, three
lifecycle methods + swap. Holds the asyncio drive loop currently in
`_drive_one_tp`.

```python
class MonarchBackend:
    def __init__(self, *, gpus_per_node, nodes=None, port=26000,
                 ready_dir=None, window_per_actor=6): ...

    def setup(self, first_plan_path: str) -> None:
        # single-host: this_host().spawn_procs({"gpus": n})
        # multi-host:  enable_transport tcp → attach_to_workers → HostMesh → spawn_procs
        # spawn TileWorker mesh with first plan; resolve n_gpus, gpn, _actor_one

    def swap(self, plan_path: str) -> None:
        # workers.swap_to.call(plan_path)  (per-TP volume/reader reset)

    def drive_tp(self, plan_path, plan, *, recon_batch=1) -> dict:
        # the Stage A↔B pipelined asyncio drive (Channel/send, prime_reader,
        # WINDOW_PER_ACTOR backpressure) — verbatim from _drive_one_tp

    def collect_recon_stats(self) -> list[dict]: ...
    def teardown(self) -> None: ...
```

Multi-host (`nodes`, `port`, `ready_dir`) and `--shard-by-proc` stay **CLI/runtime
flags** (they depend on the live SLURM allocation), not config. The shard TP-split
+ zarr-creation gating moves into the cli.py scaffold.

---

## 5. `cli.py` after refactor

```python
def tile_stitch_cli(config, input, output, timepoint, timepoints, channel,
                    recon_batch, nodes, port, gpus_per_node, ready_dir,
                    shard_by_proc):
    run = TileStitchRun.model_validate(...)
    global_tps = resolve_tps(timepoints, timepoint)
    tps = shard_tps(global_tps, shard_by_proc)            # SLURM_PROCID split
    engine_plan = build engine plan from tps[0]           # geometry once
    create_output_zarr(..., gate_on_procid=shard_by_proc) # proc 0 creates, others wait
    plan_entries = write per-TP plan pickles

    backend = MonarchBackend(gpus_per_node=..., nodes=..., port=..., ready_dir=...)
    backend.setup(plan_entries[0].plan_path)
    for i, (tp, plan_path, plan) in enumerate(plan_entries):
        if i > 0: backend.swap(plan_path)
        summary = backend.drive_tp(plan_path, plan, recon_batch=recon_batch)
        log recon_stats
    backend.teardown()
    write walls.json
```

Removed flags: `--gpu`, `--gpu-slurm`, `--scheduler-file`.
Added flags (from m3_driver): `--nodes`, `--port`, `--gpus-per-node`,
`--ready-dir`, `--shard-by-proc`.

`m3_driver.py` is deleted; the `python -m ...m3_driver` invocation in the sbatch
scripts becomes `biahub tile-stitch` (the registered CLI). One driver.

---

## 6. `config.py` changes

Drop `CpuSlurmConfig`, `GpuLocalCudaConfig`, `GpuSlurmConfig`, and the
`cpu_pool` / `gpu_pool` / `gpu_slurm_pool` fields. Add:

```python
class MonarchConfig(_Base):
    gpus_per_node: PositiveInt = 2
    recon_batch: PositiveInt = 4              # tiles per batched FFT (1 = per-tile); bench best = 4
    prefetch_depth: NonNegativeInt = 6        # bg read-ahead depth; 0 disables; want >= recon_batch
    resident_volume: bool = False             # opt into GPU-resident volume
    rdma_timeout_s: PositiveInt = 60
    recon_concurrency: PositiveInt = 1        # in-actor recon semaphore
    window_per_actor: PositiveInt = 6         # Stage B backpressure window
    compile_mode: Literal["reduce-overhead","default","none"] = "reduce-overhead"
    device: Literal["cuda","cpu"] = "cuda"    # CPU = device knob, not a separate path (§11.1)
    # validators: reject resident_volume + recon_batch>1 (HBM OOM, E4);
    #             warn if 0 < prefetch_depth < recon_batch (partial overlap).

class TileStitchRun(_Base):
    tile_stitch: TileStitchSettings
    monarch_pool: MonarchConfig = Field(default_factory=MonarchConfig)
    retries: NonNegativeInt = 1               # advisory — Monarch fails loud (Risks E5)
    run_dir: str
```

**Env-override is real plumbing, not a one-liner (reviewer E3).** The four env
knobs are read *directly at point-of-use* today with hardcoded defaults:
`TILE_STITCH_RESIDENT_VOLUME` (`tile_worker.py:249`), `TILE_STITCH_PREFETCH_DEPTH`
(`:550`), `TILE_STITCH_RDMA_TIMEOUT_S` (`:821`), `TILE_STITCH_COMPILE_MODE`
(`:423`). To make config the source of truth with env override, the config
values must be **threaded into the actor** (passed to `TileWorker.__init__` /
carried across `swap_to`) and each read rewritten as `env_or(config_value)`.
`compile_mode` is included in `MonarchConfig` above precisely because the
reviewer caught it missing. This plumbing is part of Stage 3, not free.

**`resident_volume=True` + `recon_batch>1` can OOM (reviewer E4).** Resident
volume (~34 GB) + Tikhonov (~30 GB) already ≈64 GB on an 80 GB H200
(`tile_worker.py:244-246`); a batched `(B,Z,Y,X)` stack on top will exceed HBM.
Add a config validator rejecting that combination (or document mutual
exclusion) rather than letting it OOM at runtime.

---

## 7. Dependency changes (`pyproject.toml` + `uv.lock`)

Current `tilestitch-gpu` is Dask-centric: `dask-cuda`, `cupy-cuda13x`,
`rmm-cu13`, `numba-cuda`, `ucxx-cu13`, `distributed-ucxx-cu13`.
`tilestitch-distributed` = `nvmath-python[cu13-distributed]`, `torchmonarch`.

**Correction (reviewer C1): `dask[array,distributed]`, `dask-jobqueue`,
`numba`, `llvmlite` are BASE dependencies (`pyproject.toml:40-45`), not extras,
and are used across biahub — they are NOT droppable here.** Only the
`tilestitch-gpu` extra is in scope to trim.

After hard-replace:
- **Required by Monarch:** `torch` (cuda, already base), `torchmonarch`, `iohub`,
  `zarr`, `numpy`, `waveorder`. (Monarch path uses **torch, not cupy.**)
- **Droppable from `tilestitch-gpu` extra:** `dask-cuda`, `cupy-cuda13x`,
  `rmm-cu13`, `numba-cuda`, `ucxx-cu13`, `distributed-ucxx-cu13` — verified
  imported only inside the files-to-delete (`workers.py`, `dispatcher.py`); no
  hits elsewhere in `biahub/`.
- **`nvmath-python` → DROP (decision: cut `distributed/`).** The 11-file
  cuFFTMp experiment under `biahub/tile_stitch/distributed/` (`e0`–`e4`,
  `parity`, `analyze_task_stream`) is its only importer; it is being deleted
  (§11.2), so `nvmath-python` has no remaining consumer and is removed from the
  `tilestitch-distributed` extra. Delete `distributed/` in Stage 4 alongside the
  Dask modules.
- **Restructure:** fold `torchmonarch` into `tilestitch-gpu` for one install path.

Dependency edits land in their **own commit**, last, after code is green, so a
lock regression is bisectable and revertable independently.

---

## 8. Tests (`tests/tile_stitch/`)

- Update `test_smoke.py`: replace `cpu_pool`/`gpu_pool` config assertions with
  `monarch_pool` defaults + validation. (Same commit as the `config.py` change —
  see Stage 3.)
- Add `test_core.py` (CPU-only, no GPU/Monarch) **in Stage 1, at the moment of
  the extraction** (reviewer D1): on a tiny synthetic plan, assert
  `build_stitch_geom` intersection slices and `blend_contributors` weighted-mean
  output against a hand-computed reference. **Must include a `timepoint > 0`
  case** so the leading-axis / `t_off` write convention is pinned (this is the
  exact bug the v6-vs-v7 divergence would introduce). This is the parity guard;
  it cannot wait until Stage 4.
- Add `test_monarch_smoke.py` (gated on `torch.cuda.is_available()` + monarch
  import; **skips** in CI): single-host 1-GPU, 1-TP, tiny phantom → asserts the
  output zarr is finite and shaped right. Replaces the deleted `m1_smoke.py`.
- Parity artifact: before deleting `workers.py`, capture a small reference
  output (one output tile from the smoke phantom via the current Monarch path)
  and commit it; `test_core` compares against it.

---

## 9. Migration stages (each leaves the tree importable / tests green)

**Gate 0 — prefetch bench validates first.** The in-flight A/B prefetch jobs
(33456645 off / 33456646 on) must land and confirm the prefetch reader is a win
(or neutral) before we carry it into `_core`. Do not refactor on top of
unvalidated code.

**Stage 1 — `_core.py` extraction (no behavior change) + `test_core.py`.**
Create `_core.py`; move `PrefetchReader`, geometry, TF, tile-load, eager-recon,
blend out of `tile_worker.py` **by cut-paste from the Monarch variant only**.
`tile_worker.py` imports from `_core` and keeps only actor endpoints + RDMA +
resident-vol + swap + the actor-resident `_get_compiled_recon`. **Add
`test_core.py` (with the `t_off>0` case) in this same stage** as the parity
guard. Run the monarch smoke (manual, 1-GPU) to confirm identical output.
*Dask path still present and untouched.*

**Stage 2 — `MonarchBackend` (`monarch/backend.py`).**
Move mesh setup + `_drive_one_tp` + swap + stats out of `m3_driver.py` into the
class. `backend.swap()` must **enforce Stage-B drain** before releasing the
prior TP's recons/RDMABuffers (reviewer E1). Make `m3_driver.py` a thin shim
that **re-exposes all 11 click options** (the multinode sbatch passes
`--nodes/--port/--gpus-per-node/--ready-dir`, `sbatch_m3_monarch_multinode.sh:63-65`)
so the existing sbatch keeps working through the transition.

**Stage 3 — unify `cli.py` on Monarch; add `config.MonarchConfig`.**
Rewire cli.py to scaffold + MonarchBackend (**no surviving dask imports** —
reviewer D2); add the runtime flags; move the shard/zarr-gate logic in; thread
the env-override config knobs into the actor (§6 / E3). **In the SAME commit:**
update `config.py`, **update `biahub/tile_stitch/__init__.py`** (it re-exports
`CpuSlurmConfig`/`GpuLocalCudaConfig` in `__all__`, `__init__.py:8-14` — dropping
them from `config.py` without this breaks `import biahub.tile_stitch`, surfacing
at CLI dispatch, reviewer C3), and update `test_smoke.py`. Point the sbatch
wrappers at `biahub tile-stitch`.

**Stage 4 — delete the Dask path.**
Remove `dispatcher.py`, `pipeline.py`, `workers.py`, `m3_driver.py` (shim now
unused), and the `m0/m1/multinode_probe` modules. Add `test_monarch_smoke.py`
(GPU-gated, skips in CI). Full `pytest tests/tile_stitch` green. (`config.py` +
`__init__.py` class removals already done in Stage 3.)

**Stage 5 — dependencies.**
Trim `pyproject.toml` extras, `uv lock`, sync, re-run smoke. Own commit.

**Stage 6 — integrated validation.**
Re-run the single-node 2-GPU and 2-node tp-shard sbatch through the new
`biahub tile-stitch` entry; confirm walls + output parity vs the pre-refactor
Monarch numbers. Only then is the branch PR-ready (per the cross-repo hold-all-
PRs-locally rule).

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Slimming the 908-line `tile_worker.py` regresses RDMA / resident-vol / **the just-added prefetch reader** | Stage 1 is pure move (no logic edits); manual 1-GPU smoke diff before/after; keep endpoints' bodies identical, only relocate helpers |
| Blend math drift during extraction | `test_core.py` pins `blend_contributors` against a committed reference; extract by cut-paste, not rewrite |
| Losing the Dask multi-node **comparison harness** (the A/B story) | The bench numbers are already captured in `.planning/tile_stitch_bench_iterations.md`; we keep that doc. If a live re-run is ever needed, git history has the Dask path |
| **(E1) `swap` frees recons/RDMABuffers mid-read** — splitting `_drive_one_tp` into `drive_tp` + `swap` loses the structural guarantee that Stage B fully drained before `swap_to.clear()` | `backend.swap()` asserts/awaits the prior TP's stitch completion before issuing `swap_to`; never expose a swap that can race in-flight RDMA pulls |
| ~~(E2) prefetch no-ops on `recon_batch>1`~~ **RESOLVED** — batched prefetch built (`_load_one` backs both paths; driver primes per-actor flattened batch order) and bench-validated: same-node B4+depth6 = **198s**, beats B1+prefetch (212s) | Carried into `_core` as-is. Config validator warns if `prefetch_depth < recon_batch` (partial overlap) |
| Dropping `retries` semantics (Dask had per-task retries; Monarch `__supervise__` returns False = fail loud) | Document the behavior change; `retries` config field retained but noted as advisory until Monarch retry exists |
| Dependency removal breaks an unrelated biahub module that imports cupy/dask | Grep all of `biahub/` for `cupy`/`dask`/`distributed` imports before trimming; only remove deps with zero hits outside the deleted files |
| CPU recon path disappears (Dask v6 was the only CPU path) | Confirm no shipping requirement for CPU recon; if needed later, Monarch CPU actors. Flag in Open Questions |
| `distributed/` cuFFTMp experiment + `nvmath-python` orphaned | Decide keep-vs-cut (Open Questions) before trimming that dep |

---

## 11. Resolved decisions + remaining questions

**RESOLVED:**
1. **CPU path → device knob, not a separate path.** Don't keep the Dask CPU
   recon. Instead make the **one Monarch backend device-parameterized**: CPU is
   `device="cpu"` on a CPU procmesh. waveorder recon already accepts `device`
   (the old v6 path passed `"cpu"`). Work required (folded into Stage 1/3):
   - `_core` functions take an explicit `device` (already needed for `cuda:{idx}`).
   - `TileWorker` guards CUDA-only calls behind a device check: `set_device`,
     `synchronize`, `empty_cache`, and `torch.compile(mode="reduce-overhead")`
     (CUDA-graph-specific → eager/`default` on CPU). RDMABuffer is already
     CPU-backed, so Stage B transport is unaffected.
   - `MonarchConfig` gains `device: Literal["cuda","cpu"] = "cuda"`. Single
     backend, both devices.
2. **`distributed/` cuFFTMp experiment → CUT.** Delete
   `biahub/tile_stitch/distributed/` (e0–e4, parity, analyze_task_stream) and
   drop `nvmath-python` from the `tilestitch-distributed` extra. Monarch is the
   chosen multi-node approach; cuFFTMp is not pursued. This UNBLOCKS the §7 dep
   trim — `nvmath-python` now has no remaining importer. (Recoverable from local
   history if ever revisited.)

**STILL OPEN (lower-stakes, can default):**
3. **Entry-point name:** keep `biahub tile-stitch` (replace internals)? — default yes.
4. **Cross-node HostMesh path:** TP-shard is the validated scaling pattern; do
   we ship the cross-node HostMesh (single-volume-split) mode or keep it behind
   an undocumented flag? — default: keep behind flag, TP-shard is the documented path.
