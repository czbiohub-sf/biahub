# Tile-stitch Bench Iterations — Monarch & Dask on Bruno H200

Working log of optimizations we've layered on the tile-stitch pipeline.
Single-TP benchmarks on c0041 (`l0_brightfield_fov.zarr`) or TP 0 of
`/hpc/projects/waveorder/tile-stitch/sample_datasets/deskewed_t100_c0.zarr`
(same spatial shape: `(1647, 2368, 4445)`, ~32 GB float16 per TP). All runs
on 2× H200 same node unless noted. Started 2026-05-27.

## Headline progression

| # | Config | Stage A | Stage B tail | Wall | Notes |
|---|---|---|---|---|---|
| 1 | Monarch baseline (semaphore-only sync) | — | — | **9:12** | Earliest 2-GPU attempt (actually 1-GPU due to device bug) |
| 2 | + W=2 async | — | — | **5:46** | (1-GPU under the hood) |
| 3 | + local-fastpath + RDMAAction batching | — | — | **5:16** | Skip RDMA for same-actor contribs; batch peer reads |
| 4 | + NUMA pinning | 150.7s | 137s | **4:51** | CPU blend stays on GPU-local socket |
| 5 | + W=4 | 183.5s | 54s | **3:58** | CPU saturated by concurrent blends |
| 6 | + W=6 | 192.2s | 29s | **3:41** | Still 1-GPU, hidden |
| — | **Device-bug fix (explicit `cuda:{gpu_idx}`)** | | | | Both GPUs actually running |
| 7 | + GPU-resident volume | 94s | 67s | **2:41** | Per-tile zarr eliminated, true 2-GPU |
| 8 | + torch.compile reduce-overhead | 120.4s | 67s | 3:07 ❌ | +26s compile cost; would amortize multi-TP |
| 9 | + zarr prefetch (sem=2) | 206s | 55s | 4:21 ❌ | CPU oversubscription |
| 10 | Multi-TP smoke (2 TPs, GPU-vol, no geom cache) | TP0: 80s / TP1: 62.9s | | TP0: 148s / TP1: 138s | Total 312.5s. Vol swap 25.8s. Actor persistence + warm caches confirmed. |
| 11 | + streaming + geom cache | 100.5s | 25.8s | **2:06** | Geom cache saves ~41s/TP. GPU-vol disabled here. |
| 12 | + GPU-vol + geom cache | **73.6s** | 44.6s | **1:58** | **Current best.** GPU-vol adds 8s on top of geom cache. |

## Dask reference points

| # | Config | Wall | Notes |
|---|---|---|---|
| D1 | Dask v7 baseline (per-tile zarr, TCP) | **3:06** | LocalCUDACluster, true 2-GPU |
| D2 | Dask v7 + GPU-vol preload | 5:47 ❌ | Unexplained regression vs D1; didn't dig in |

## Decomposed contributions (on top of dask 3:06 ref)

| Lever | Δwall | Mechanism |
|---|---|---|
| Real 2-GPU via explicit `cuda:gpu_idx` | huge | `set_device` is thread-local; asyncio.to_thread workers defaulted to cuda:0 |
| Geom cache + kernel cache | **~41s** | Precompute per-output intersections; one kernel rebuild per shape |
| GPU-resident volume | ~8s | HBM slice vs per-tile zarr (Vast page cache stays warm) |
| NUMA pin | ~25s | CPU blend on GPU-local socket |
| Concurrent blend window (W=6) | ~30s | CPU saturated, not idle waiting on zarr |
| local-fastpath + RDMAAction batch | ~30s | Skip RDMA for same-actor contributors |

Total observed: ~1:10 off dask 3:06 → **1:58** current Monarch best.

## Levers rolled back

| Lever | Why |
|---|---|
| GPU blend | Default-stream contention with recon: Stage A +25s |
| Zarr prefetch (sem=2) | CPU oversubscription pushed both stages up |
| torch.compile reduce-overhead | 26s/TP compile cost doesn't amortize within single TP |
| Dask v7 + GPU-vol preload | Regression vs dask baseline; cache hit/miss not verified |

## Key finding: geom cache

The single biggest optimization in the session, and easy to miss because the
per-call work *looks* lightweight in source.

### What was being recomputed every stitch call (per TP)

```python
blend_kernel = self.plan.settings.blend.build()                  # heavy object
tiles_by_id = {t.tile_id: t for t in self.plan.input_tiles}      # O(N) dict build
for tid, tile_full in contribs_np.items():
    ...isect computation in every dim...
    kernel_full = blend_kernel.weight_kernel(in_tile.shape).astype(...)  # 512³ float32!
```

Cost per TP: 180 stitches × ~5 contributors = **~900 calls** to `weight_kernel`.
Each rebuilds a fresh 512³ float32 array (~500 MB) — only to take a small view
of it. All input tiles share the same shape, so the kernel is identical every time.

### What the cache does (built once in `__init__`)

```python
self._blend_kernel       = self.plan.settings.blend.build()        # 1×
self._tiles_by_id        = {...}                                    # 1×
self._stitch_geom        = self._build_stitch_geom()                # 1× per out_tile_id
self._kernel_cache       = {}                                       # lazy fill per shape
```

`_stitch_geom[oid]` holds: `out_spatial`, `out_shape`, and per-contributor
`{tile_shape, in_local, in_full_idx, out_full_idx}`. Pre-computed once.

### Why it survives `swap_to` for multi-TP

Per-TP swap only changes input pixel data + output T offset. Tile geometry
(input/output tile slices, neighbor relationships, blend kernel weights) is
identical across TPs — imaged volume's spatial shape doesn't change over time.

## Architecture notes

### Stage A (recon)
- One actor per GPU. Volume loaded once into HBM (uint16, ~34 GB), per-tile cast to float32 cheaply.
- `torch.cuda.set_device(self.gpu_idx)` re-pinned in `_reconstruct_blocking` because thread-local in asyncio.to_thread workers.
- Semaphore size 1 per actor — Tikhonov intermediates would OOM otherwise.

### Stage B (stitch)
- CPU blend (numpy) — GPU blend contends with concurrent recon on default stream.
- `WINDOW_PER_ACTOR = 6` concurrent stitches. Higher = more CPU saturation; 6 with OMP=4 = 24 thread-requests ≤ 32 NUMA-local cores.
- Local-fast-path: skip RDMA when consumer == producer (~50% of contributors on average).
- `RDMAAction.submit()` batches multiple `read_into`s, Monarch groups by source actor.

### NUMA pinning (`tile_worker._pin_to_numa_for_gpu`)
- sysfs lookup: `pynvml` → GPU PCI bus → `/sys/bus/pci/devices/<bus>/numa_node` → `/sys/devices/system/node/nodeN/cpulist`.
- `sched_setaffinity(0, target_cpus)` — child threads (asyncio.to_thread) inherit on Linux.
- SLURM `--gres-flags=enforce-binding --cpus-per-task=64` to land us on GPU-local CPUs.
- Bruno H200 fallback: GPU 0 → NUMA 2, GPU 1 → NUMA 3 (4 NUMA nodes per node).

### Multi-TP loop (`m3_driver`)
- `--timepoints '0-9'` or `'0,3,7'` or single `'5'`.
- Actors persist across TPs; `swap_to(plan_path)` reloads only the volume.
- Output zarr pre-created with T=N dim via `_create_multi_tp_zarr` helper — bypasses iohub's `create_image` to avoid allocating the full zeros tensor on host.
- Per-TP wall + stitch stats written to `<run_dir>/walls.json`.

### Env knobs
- `TILE_STITCH_STREAM_TILES=1` — disable GPU-resident volume, fall back to per-tile zarr (for A/B isolation).
- `TILE_STITCH_COMPILE_MODE` — `reduce-overhead` (default; CUDA graphs), `default` (Inductor only), `none` (eager).

## Open work

| Item | Status |
|---|---|
| 10-TP multi-TP bench (Monarch + GPU-vol + geom cache) | Ready to fire |
| 10-TP + compile (amortization test) | Ready to fire |
| 10-TP Dask v7 TCP (per-tile zarr + geom cache) | Needs dask geom-cache port (~half-day refactor) |
| 10-TP Dask v7 UCXX | Same prerequisites + UCXX config |
| Multi-node UCXX over IB (16 GPUs across 2 nodes) | Scaffolding not yet written: `GpuSlurmConfig`, `make_gpu_slurm_cluster`, dedicated scheduler head SLURM job |
| Investigate dask GPU-vol preload regression | Add cache hit/miss counter to `reconstruct_tile_memory_gpu`, rerun |
| Scaling to volumes ≥ 2× current size | Sharded GPU volume (Z-band per actor) or shared-host mmap fallback |
| Double-buffer load(TP+1) during compute(TP) | Hides ~25s/TP volume swap cost |

## Multi-node Monarch — works, but streaming is FS-bound (does NOT scale)

Multi-host bootstrap validated (probe 33427158: cross-node RDMABuffer INTEGRITY_OK).
m3_driver ported to HostMesh via `attach_to_workers` + per-node `run_worker_loop_forever`
(worker_loop.py), gated on `<host>.ready` files (non-batch nodes cold-start uv slower
→ one-shot attach races them). Host-aware slicing `_actor_one(flat_idx)` for the
`{hosts, gpus}` mesh. RDMA timeout fix: RDMAAction hard-codes 3s; switched stitch to
direct `RDMABuffer.read_into(timeout=TILE_STITCH_RDMA_TIMEOUT_S, default 60)` via
asyncio.gather (3s too short for cross-node IB under Stage B load).

2-node × 2-GPU (4 actors), single TP, streaming, deskewed_t100 TP0 (33428263):
**wall 229.9s — WORSE than single-node 126s.** Per-actor Stage A diagnostic:

| Actor | tiles | io_s | fft_s | d2h_s | util |
|---|---|---|---|---|---|
| gpu-h-3 gpu0 | 50 | 29.4 | 12.6 | 13.8 | 1.00 |
| gpu-h-3 gpu1 | 50 | 28.9 | 12.6 | 14.2 | 1.00 |
| gpu-h-7 gpu0 | 50 | 62.3 | 13.2 | 19.6 | 1.00 |
| gpu-h-7 gpu1 | 50 | 71.4 | 13.3 | 39.9 | 1.00 |

Findings:
- **FFT scales perfectly** (~13s/actor, was ~26s at 100 tiles/actor single-node). util=1.0 everywhere → NOT controller/dispatch bound.
- **IO is the cap and asymmetric**: remote node (gpu-h-7) zarr reads 2-2.4× slower than batch node (gpu-h-3). d2h (ibverbs RDMABuffer registration) also 2-3× slower remote.
- Net: adding a node adds a slower-IO participant; slowest actor's IO+d2h sets the floor → regression vs single-node.

**Conclusion: Stage A is shared-filesystem-bound in streaming mode, not compute-
or dispatch-bound.** Multi-node scaling requires reading each tile ONCE per node
from node-local storage (resident volume / node-local cache / stage-to-scratch),
not per-tile re-reads from contended Vast. Per-actor diagnostic lives in
`TileWorker.recon_stats` (io/fft/d2h split + busy/span util).

Env: `TILE_STITCH_RESIDENT_VOLUME=1` (default off, streaming),
`TILE_STITCH_RDMA_TIMEOUT_S` (default 60).

## Dask multi-node (allocated pattern + TCP-over-IPoIB)

Validated the gpu_slurm/multi-node Dask path. Two patterns built:
- **dask-jobqueue** (`make_gpu_slurm_cluster`, `--gpu-slurm`): head job on cpu
  partition submits worker GPU jobs via SLURMCluster. Works but idle cpu seat +
  double fairshare wait. Abandoned.
- **Allocated** (`sbatch_dask_multinode_allocated.sh`, `--scheduler-file`): one
  SLURM job grabs all nodes; srun starts scheduler + dask-cuda-workers; driver
  connects via scheduler-file. Same shape as Monarch worker_loop. **This is the
  one to use.**

Bug fixes along the way:
- dask-jobqueue worker script had duplicate `--cpus-per-task` / `-o` (I added
  what dask already emits) → SLURM rejected → silent 0-worker hang. Fix: set
  `cores=cpus_per_task`, don't duplicate directives. `worker_command="dask_cuda.cli"`
  (module for `python -m`), `python=sys.executable` (venv python; workers run
  outside `uv run`).
- **UCXX is broken at the library level in this build** — both the standalone
  `dask scheduler --protocol ucxx` CLI AND the in-process form
  (`dask_cuda.initialize()` + `LocalCluster(n_workers=0, protocol="ucxx")`,
  probe 33434696) fail identically with `UCXXUnsupportedError: Unsupported
  operation` at scheduler-listener creation. Not a scheduler-form or dask-config
  issue — the `ucxx` 0.2.3 wheel can't create a UCX listener here, almost
  certainly a UCX ABI / version mismatch vs hpcx/2.19's UCX (or a missing
  transport feature). Bruno's IB fabric is fine (Monarch's native ibverbs work
  cross-node). Conclusion: **dask+UCXX unusable without fixing the ucxx/UCX
  install; use TCP-over-IPoIB.**

  **ROOT CAUSE (probe 33434893, UCX_LOG_LEVEL=debug):** the scheduler fails at
  `ucp_listener` creation — "cannot create listener: none of the available
  components supports it". The pip **`libucx-cu13` 1.19.0 wheel is a CUDA-only
  slim build**: its only transport plugins are `libuct_cma` (shm) + `libuct_cuda`
  (cuda_copy/ipc). **No `libuct_ib`, no `libuct_tcp`, no rdmacm** → cannot create
  a network listener or do cross-node transport at all. The catch-22:
  - pip libucx 1.19.0 → right ABI for ucxx 26.4, but CUDA-IPC/shm only (no net)
  - system /lib64 UCX 1.18.0 → full (ib/tcp/rdmacm) but wrong ABI
  - hpcx/2.19 UCX 1.18 → full but wrong ABI

  No UCX present is both ABI-1.19 AND a full networking build. The libucx fix
  (LD_LIBRARY_PATH → pip 1.19) loaded the right lib (confirmed via
  /proc/self/maps) but it has no networking transports. The old single-node
  LocalCUDACluster UCXX run "worked" only because single-node needs just
  cuda_ipc/sm, which this build has.

  **Real fix (if ever needed): conda RAPIDS env** (full version-matched UCX with
  ib/tcp/rdmacm — pip UCX wheels are CUDA-IPC-only by design), or build UCX
  1.19 from source `--with-verbs --with-rdmacm --with-tcp`. NOT pursued:
  multi-node is FS-bound (UCXX won't change it) and Monarch already does native
  cross-node ibverbs RDMA without UCX.
- **Bruno IB has no IPoIB on ib0** (native verbs only, no IPv4); `--interface ib0`
  → "doesn't have an IPv4 address". **ib3 DOES have an IPv4** → use it.
- **Working transport: TCP-over-IPoIB** — `--protocol tcp --interface ib3`. Binds
  dask to the IPoIB interface so TCP rides the IB fabric. 4/4 workers register in
  0.0s, clean run.

Result (4 GPU / 2 node, TCP-IPoIB, TP0): **267s** vs dask single-node 2-GPU 297s
→ mild ~1.1× scaling (unlike Monarch which regressed multi-node). But ~2× slower
than Monarch single-node (126s) — dask per-task graph overhead + TCP-IPoIB
transfers vs Monarch async actors + native RDMA.

Headline (same dataset/TP0):

| Framework | Layout | Transport | Wall |
|---|---|---|---|
| Monarch | 1-node 2-GPU streaming+geom | n/a | **126s** ← best |
| Monarch | 1-node 2-GPU resident+geom | n/a | 118s |
| Monarch | 2-node 4-GPU streaming | TCP+RDMA | 202s |
| Dask | 1-node 2-GPU streaming+geom | TCP | 297s |
| Dask | 2-node 4-GPU | TCP-IPoIB | 267s |

## Batched recon (B>1) — no win for our tile size

waveorder's `apply_inverse_transfer_function` accepts `(B, Z, Y, X)` and does
a batched FFT. We wired batching into both GPU paths (`--recon-batch N`):
- Monarch: `reconstruct_batch` endpoint, `_build_recon_batches` groups input_order by shape.
- Dask v7: `reconstruct_batch_memory_gpu` + `pluck_tile` (per-tile futures so Stage B unchanged), `drive_pipelined_v7(recon_batch=N)`.

Result (streaming, single TP, 2× H200):

| Config | Stage A | Wall |
|---|---|---|
| B=1 (33416633) | 100.5s | 126.3s |
| B=4 (33426758) | 101.1s | 128.9s |

No improvement. **Stage A is FFT-compute-bound, not dispatch-bound** — the
512³ FFT is large enough that per-call launch + cuFFT-plan overhead is in the
noise, so there's nothing to amortize. Batching only helps small tiles where
launch overhead dominates. Code kept (default B=1, harmless) but not a lever
for this workload. Confirms: GPU saturated by one FFT, concurrent batches
also pointless (no idle SMs).

The real Stage A reducer is **more GPUs (data-parallel)** — fewer tiles/GPU.

## Things that didn't apply (worth recording so we don't revisit)

- **NUFFT (torchkbnufft)**: non-uniform FFT; our grids are uniform, would be slower not faster.
- **Distributed FFT (cuFFTMp / TorchProcessGroup)**: comm overhead dominates compute for our 512³ tile size; we have many independent tiles, data-parallel is optimal.
- **Treat 2 GPUs as one mesh**: same issue as distributed FFT. Data-parallel across independent tiles wins.
- **bf16 / fp16 FFT**: cuFFT half-precision is shape-restricted and dynamic-range risky for biological data. TF32 matmul is already on by default.
- **zarrs-python**: io_uring blocked on Rocky 8.10 login node, compute nodes unverified. Even if working, the win is small (~5-7s/TP off the ~10s volume read) vs double-buffering which hides it entirely.

## File map

| Path | What changed |
|---|---|
| `biahub/tile_stitch/monarch/tile_worker.py` | TileWorker actor: volume load, recon, stitch, geom cache, NUMA pin, swap_to endpoint |
| `biahub/tile_stitch/monarch/m3_driver.py` | Multi-TP driver: TP loop, swap orchestration, output zarr pre-create, per-TP walls.json |
| `biahub/tile_stitch/plan.py` | `write_plan` now takes optional `filename` for per-TP plan pickles |
| `biahub/tile_stitch/dispatcher.py` | `gpu_worker_preload_volume(plan_path, dask_worker=None)` — dask side, currently unused after we removed the preload call |
| `biahub/tile_stitch/workers.py` | `_try_get_resident_volume` + opt-in fast path in `reconstruct_tile_memory_gpu` (dask side) |
| `biahub/tile_stitch/cli.py` | (Dask path) currently has preload call removed; needs multi-TP loop when we get to that bench |
| `scripts/distributed/sbatch_m3_monarch.sh` | `--gres-flags=enforce-binding --cpus-per-task=64`, 30 min walltime |
| `scripts/distributed/sbatch_2gpu_dask_1024tile.sh` | (Untouched; dask v7 reference path) |
