# submitit DebugExecutor is unsuitable for Nextflow `--debug` path

**Date:** 2026-05-27
**Context:** PR #224 / #252 — consolidating biahub CLI with `--init` / `--debug` flags for Nextflow orchestration
**submitit version:** 1.5.4 (confirmed identical to latest `main` on <https://github.com/facebookincubator/submitit>)

## Background

The proposal is to consolidate the biahub CLI so that `--init` handles output zarr creation and `--debug` runs the per-position processing step in-process (i.e., Nextflow calls the same CLI command that users call interactively). For the `--debug` path, submitit's `AutoExecutor` with `cluster="debug"` would wrap the function call.

The `DebugExecutor` was designed for interactive unit testing and IDE debugging. It has three behaviors that are problematic when the calling process is a Nextflow task running non-interactively.

## Concern 1: `pdb.post_mortem()` obfuscates errors on exception

**Source:** `submitit/local/debug.py`, `DebugJob.results()` lines 89-103

```python
try:
    return [self._submission.result()]
except Exception as e:
    print(e)
    if os.environ.get("PYTHONBREAKPOINT", "").startswith("ipdb"):
        import ipdb
        ipdb.post_mortem()
    else:
        import pdb
        pdb.post_mortem()
    raise
```

When the submitted callable raises any exception, the debug executor catches it and unconditionally drops into `pdb.post_mortem()`. Under both Nextflow executors (local and Slurm/sbatch), stdin is `/dev/null` or a pipe, so pdb receives EOF and exits immediately — the process still exits with code 1 and the error is not lost. However, the error output is obfuscated in three ways:

1. **Split across streams.** pdb's `print(e)` sends the error message to stdout (`.command.out`), while the traceback goes to stderr (`.command.err`). In Nextflow's error report these appear as separate "Command output" and "Command error" sections, requiring the reader to cross-reference.

2. **pdb noise in stdout.** The `.command.out` content looks like:
   ```
   Simulated processing error (e.g. OOM, corrupted zarr, bad config)
   > toy_process.py(24)toy_transform()
   -> raise RuntimeError(
   (Pdb)
   ```
   The frame info (`> ...`, `-> ...`) and `(Pdb)` prompt are pdb's interactive debugger output, not part of the error. Without submitit, stdout is empty and the full error + traceback appears cleanly in stderr.

3. **Deeper traceback.** The traceback includes 3 extra submitit frames (`debug.py:results`, `utils.py:result`, `utils.py:function`) that are irrelevant to the actual error, pushing the real call site further down. For a real-world error deep inside torch/iohub/waveorder, these extra frames make triage noticeably harder.

Verified on Bruno HPC (Nextflow 24.10.5, Slurm):
- `nextflow run main.nf -profile slurm --mode fail` (sbatch): pdb exits on EOF, exits with code 1 in ~1s. `(Pdb)` prompt in `.command.out`, traceback split across `.command.out` and `.command.err`.
- `nextflow run main.nf --mode fail` (local executor): same behavior.

Note: under a TTY-attached stdin (e.g., `srun --pty`), pdb would hang indefinitely, but this is not a realistic Nextflow execution path.

**Severity:** Low-Medium. Errors propagate correctly but are harder to triage due to log pollution and stream splitting.

## Concern 2: Shadow log directory splits logging away from Nextflow

**Source:** `submitit/local/debug.py`, `DebugJob.results()` lines 77-88

```python
root_logger = logging.getLogger("")
self.paths.stdout.parent.mkdir(exist_ok=True, parents=True)
stdout_handler = logging.FileHandler(self.paths.stdout)
stdout_handler.setLevel(logging.DEBUG)
stderr_handler = logging.FileHandler(self.paths.stderr)
stderr_handler.setLevel(logging.WARNING)
root_logger.addHandler(stdout_handler)
root_logger.addHandler(stderr_handler)
```

The debug executor adds `FileHandler`s to the Python root logger pointing at submitit's own log directory (the `folder` argument to `AutoExecutor`, typically `<output>.zarr/../slurm_output/`). During execution:

- All `logging.*` calls from any library (iohub, torch, numpy, waveorder, etc.) are written to **both** the process stdout (Nextflow's `.command.log`) **and** submitit's shadow files.
- `print()` calls only go to stdout — they are not duplicated.
- The submitit log files live outside Nextflow's work directory and are not captured by `publishDir`, `trace.txt`, or `nextflow log`.

This means:

1. When triaging a failure, the logs in `.command.log` may appear complete but are actually a subset (missing `print`-only output that happened before the logger was mutated, or vice versa).
2. The shadow log directory accumulates silently under `slurm_output/` on the shared filesystem, consuming inodes and disk without any cleanup mechanism tied to Nextflow's `-resume` or `cleanup` lifecycle.
3. **Retry accumulation:** when Nextflow retries a failed task (via `errorStrategy 'retry'`), each attempt creates a new `submitit_logs/DEBUG_<id>_0_log.{out,err}` pair with a unique ID inside the same work directory. These orphaned files pile up across retries, making it ambiguous which log corresponds to the successful vs. failed attempts. They are not cleaned by `nextflow clean` since they are not Nextflow-managed artifacts.

Note: Nextflow's `-resume` cache correctness is **not** affected — cache keys are computed from input hashes, script content, and container, not from output artifacts in the work dir. But the `submitit_logs/` pollution persists across cached runs and retries.

**Severity:** Medium. Doesn't break execution or caching but degrades observability and wastes storage.

## Concern 3: `os.environ` clear/restore can strip Nextflow-injected variables

**Source:** `submitit/local/debug.py`, `DebugJob.results()` lines 72-76 and 104-106

```python
# Before execution:
environ_backup = dict(os.environ)
os.environ.clear()
os.environ.update(self.environ)  # self.environ is snapshot from job creation time

# After execution (in finally):
os.environ.clear()
os.environ.update(environ_backup)
```

The debug executor snapshots `os.environ` at `DebugJob.__init__()` time (when `executor.submit()` is called) and replays that snapshot before running the callable. Any environment variables set *after* job creation but *before* execution are silently dropped.

In practice this is a non-issue for the Nextflow path: debug mode is synchronous, so the gap between submit and execute is negligible. Slurm-injected vars (`SLURM_*`, `CUDA_VISIBLE_DEVICES` via gres) and Nextflow-injected vars (`NXF_*`) are all present in the environment before Python starts, so the snapshot captures them correctly.

The biahub multiprocessing pattern (`ProcessPoolExecutor` with `mp.get_context("spawn")`, used by `process_single_position`) is also safe — workers are spawned and torn down entirely within the function execution, while `os.environ` holds the snapshot. The only theoretical risk is concurrent threads in the main process reading env vars during the brief non-atomic `os.environ.clear()` window (e.g., background threads from torch, zarr, or cloud SDKs), but this has not been observed.

**Severity:** Low. Not a practical concern for the current Nextflow/Slurm execution path, but `os.environ.clear()` is an unnecessary side effect that a direct function call avoids entirely.

## Scope: existing workflows are not affected

The `debug` cluster is currently **only** used in the test suite (`tests/conftest.py` sets `CI=true`). Interactive and HPC users always get `local` or `slurm` from `get_submitit_cluster()`. The concerns below apply exclusively to the proposed Nextflow `--debug` path — not to any existing workflow.

## Recommendation

For the Nextflow `--debug` path, bypass submitit entirely and call the processing function directly:

```python
if init:
    create_output_zarr(...)
elif debug:
    # Direct call - Nextflow handles scheduling, retries, and log capture
    process_single_position(transform_fn, input_path, output_path, ...)
else:
    # Original submitit path for interactive/HPC use
    executor = submitit.AutoExecutor(folder=..., cluster=get_submitit_cluster(local))
    ...
```

This gives:
- Clean error propagation (Nextflow sees the exit code and traceback)
- Unified logging (everything in `.command.log`)
- No env-var interference
- No shadow log directories
- The submitit path remains intact for the existing non-Nextflow interactive workflow

## Reproducer

A toy Nextflow pipeline demonstrating concerns 1, 2, and 3 is in this directory. See `README.md` for exact invocation commands.
