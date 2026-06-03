# `biahub tile-stitch`

Distributed tiled phase reconstruction over a dask cluster. Consumes
`waveorder.api.tile_stitch.*` and runs the engine across SLURMCluster
(CPU) or LocalCUDACluster (single-node multi-GPU).



## config

Copy one of the example YAMLs as a starting point:

- `settings/example_tile_stitch_cpu.yml` — CPU SLURMCluster +
  dask-jobqueue, K=4 batched recon.
- `settings/tile-rec-stitch/example_tile_stitch_gpu.yml` — single-node `LocalCUDACluster`,
  RMM pool + cupy DLPack handoff.

What to edit per run:

| field | meaning |
|---|---|
| `tile_stitch.recon.transfer_function.*` | your phase TF: yx_pixel_size, NA_detection, NA_illumination, wavelength_illumination, z_padding, etc. |
| `tile_stitch.tile.tile_size` | per-dim input tile size (e.g. `{z: 512, y: 512, x: 512}`). Output zarr chunks shadow this. |
| `tile_stitch.tile.overlap` | per-dim tile overlap (e.g. `{z: 32, y: 32, x: 32}`) |
| `tile_stitch.blend.kind` | `gaussian_mean` (default), `uniform_mean`, `max`, `min` |
| `cpu_pool.scratch_dir` | dask spill directory — point at a project mount with TB free |
| `cpu_pool.workers_min/max` | adapt() bounds; 8/32 is a reasonable starting point for ~200-tile workloads |
| `cpu_pool.worker_memory` | RAM per worker SLURM job (default 64 GB; bump if Stage A pushes 95% threshold) |
| `cpu_pool.batch_size` | K input tiles per Stage A batched dispatch (default 4) |
| `cpu_pool.prewarm_workers` | block dispatch until N workers register; `0` skips and dispatches immediately |
| `gpu_pool.rmm_pool_size` | per-GPU RMM pool (default 30 GB); should fit ≤ 0.85 × min VRAM of selected GPUs |
| `run_dir` | absolute path for run artifacts (plan.pkl, dask_logs/, scratch/) |

## submit

CPU:

```bash
sbatch scripts/tile_stitch/example_tile_stitch_cpu.sh \
    --config my_run.yml \
    --input  /path/to/input.zarr \
    --output /path/to/output \
    --channel "BF"
```

GPU (whole-node `--exclusive` allocation in the GPU partition):

```bash
sbatch scripts/tile_stitch/example_tile_stitch_gpu.sh \
    --config my_gpu_run.yml \
    --input  /path/to/input.zarr \
    --output /path/to/output \
    --channel "BF" \
    --gpu
```

You can also run the driver outside SLURM (e.g. inside an `salloc`):

```bash
uv run biahub tile-stitch \
    --config my_run.yml --input <in> --output <out> --channel "BF" [--gpu]
```

## what happens

1. Driver SLURM allocation lands (4 cores / 96 GB / 2 h for CPU; whole-node 8× H100 `--exclusive` for GPU).
2. Wrapper `cd`s to the biahub repo, runs `uv run --no-sync biahub tile-stitch …`.
3. CLI parses YAML → `TileStitchRun`.
4. Opens input zarr, picks channel, calls `waveorder.tile_stitch._engine.build_plan` with `batch_size=cpu_pool.batch_size` (or `1` for GPU per-tile dispatch).
5. Pre-creates output zarr (5D `(T, C, Z, Y, X)`, chunks shadow `tile_size` → 1 output tile = 1 chunk = single-writer guarantee).
6. Pickles `RunPlan` to `<run_dir>/plan.pkl` so workers `_load_plan` once and cache.
7. **CPU:** `make_cpu_cluster` constructs `SLURMCluster` (`python=sys.executable`, `local_directory=scratch_dir`, `--cpus-per-task=N` SBATCH directive, BLAS thread caps + `MALLOC_TRIM_THRESHOLD_=0` + `MALLOC_ARENA_MAX=2` in `job_script_prologue`). `cluster.adapt(min, max)` (or `cluster.scale(max)` if `pool_mode: scale`).
8. **GPU:** `make_gpu_cluster` constructs `LocalCUDACluster`; auto-detects GPUs from `CUDA_VISIBLE_DEVICES`. `client.run(gpu_worker_setup)` wires RMM-backed torch allocator + cuFFT plan cache on each worker.
9. Optional `wait_for_workers(prewarm_workers, prewarm_timeout_s)` blocks until N register.
10. **CPU:** `drive_pipelined_v6` — interleaved Stage A↔B via `as_completed`. `reconstruct_batch_memory` reconstructs K tiles per dispatch (one waveorder call per K tiles, returns `dict[tile_id, ndarray]`). When all batches contributing to an output tile complete, dispatches `stitch_output_tile_v6`: flatten contributors → in-place blend → zarr write.
11. **GPU:** `drive_pipelined_v7` — same shape, but per-tile (`reconstruct_tile_memory_gpu` returns cupy via DLPack). `stitch_output_tile_v7` accumulates GPU-resident, single CPU bounce at zarr write.
12. Driver writes one summary log line, exits. Cluster torn down.

## monitor

```bash
squeue --user $USER                                            # cluster state
tail -f <run_dir>/dask_logs/slurm-<jobid>.out                  # worker stdout
tail -f tile-stitch_<jobid>.log                                # driver stdout (writes to repo cwd)
```

Per-task metadata streams to `<run_dir>/stage_meta/{host}_{pid}.jsonl`
(timing, RSS, GPU util when applicable). One JSONL per worker process;
read with any line-delimited-JSON tool.

## output

`<output>.zarr/` — OME-Zarr FOV layout:

- shape `(T=1, C=1, Z, Y, X)`, dtype `float32`
- chunks shadow `tile_size`
- channel name `<input_channel>_recon`

Read:

```python
from iohub.ngff import open_ome_zarr
ds = open_ome_zarr("/path/to/output.zarr", layout="fov", mode="r")
recon = ds["0"][:]
```

## recovery

**Driver crashed mid-run.** Re-submit; the output zarr from the prior
run is overwritten on the next launch (driver creates it fresh at step 5
above). A `--resume` flag is on the deferred list — re-add if you need
to recover only the failed tiles instead of re-running the whole stage.

**Orphan SLURM workers** after `kill -9` of the driver (signals can't
be trapped):

```bash
bash scripts/tile_stitch/cleanup_orphans.sh           # scancel orphan dask-worker jobs
bash scripts/tile_stitch/cleanup_orphans.sh --dry-run # preview
```

## reference numbers

Reference dataset (1647 × 2368 × 4445 fp32, ~13 GB), phase recon, K=4 CPU
batched, 8..32 SLURMCluster workers @ 16 cores × 64 GB:

- 200 input tiles → 53 batches → 180 output tiles
- Wall: **5:58** total (with prewarm), **3:25** pipeline-only
- 0 failures
- Per-tile recon ~3.3 s (CPU), peak RSS ~14 GB per task

Same dataset, single-node 8× H100 GPU run: **3:27** wall.

## known constraints

- `output_chunk` is not a user knob — it always shadows `tile_size`. Output zarr writes are 1 chunk per output tile = no contention.
- CPU pool: `dask_threads: 1` is required. With > 1, K=4 batches OOM on 64 GB workers.
- GPU pool: single-node `LocalCUDACluster` only. Multi-node `slurm-cuda` (dask-cuda-worker via dask-jobqueue) is deferred — GPU dispatch overhead at multi-node currently exceeds the recon savings.
- `protocol: "ucx"` is rejected by Pydantic — `dask-cuda >= 26.04` removed UCX in favor of UCXX.
- Worker memory is checked at runtime inside `gpu_worker_setup`: if `rmm_pool_size > 0.85 × actual_vram`, the worker aborts before its first task. The static validator misses this when `gpu_constraint` is None; the runtime check is the safety net.
