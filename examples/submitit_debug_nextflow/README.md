# submitit DebugExecutor + Nextflow: failure mode demo

Toy pipeline demonstrating why submitit's `DebugExecutor` (cluster="debug") is
unsuitable as the execution backend for Nextflow task processes.

See the full writeup at `2026-05-27-submitit-debug-nextflow-concerns.md` in this directory.

## Prerequisites

```bash
pip install submitit   # any version; tested with 1.5.4
module load nextflow
```

## Reproducing the failure modes

All commands below assume `--python_bin` points to a Python interpreter with
submitit installed (e.g., a project venv). If `python` on your PATH already has
submitit, you can omit `--python_bin`.

```bash
# Example: use a specific venv
PYTHON_BIN=/path/to/venv/bin/python
```

Two executor profiles are available in `nextflow.config`:
- **`standard`** (default): runs tasks locally via Nextflow's local executor
- **`slurm`**: submits tasks via `sbatch` (edit `queue` in `nextflow.config` for your partition)

**Important:** when using `-profile slurm`, the `-work-dir` must be on a shared
filesystem (not `/tmp`), since compute nodes write output there.

### Concern 1: `pdb.post_mortem()` on exception

**With submitit (local executor):**

```bash
nextflow run main.nf --python_bin $PYTHON_BIN --mode fail
```

**With submitit (Slurm executor):**

```bash
nextflow run main.nf -profile slurm --python_bin $PYTHON_BIN --mode fail
```

Under both executors, stdin is `/dev/null` (or a pipe), so pdb gets EOF and
exits immediately — the process exits with code 1. However, the `(Pdb)` prompt
pollutes `.command.out` and the traceback is split across `.command.out` and
`.command.err`:

```
# .command.out — pdb noise instead of a clean error:
Simulated processing error (e.g. OOM, corrupted zarr, bad config)
> toy_process.py(24)toy_transform()
-> raise RuntimeError(
(Pdb)
```

Under a TTY-attached context (e.g., `srun --pty`), pdb **hangs indefinitely**.
You can verify this directly:

```bash
# Hangs until killed (5s timeout, exit 124):
timeout 5 script -q -c "$PYTHON_BIN toy_process.py --mode fail" /dev/null

# Exits immediately on pipe stdin (EOF → pdb exits):
echo "" | $PYTHON_BIN toy_process.py --mode fail
```

**Without submitit (clean failure):**

```bash
nextflow run main.nf --python_bin $PYTHON_BIN --mode fail --bypass_submitit true
# or with Slurm:
nextflow run main.nf -profile slurm --python_bin $PYTHON_BIN --mode fail --bypass_submitit true
```

The exception propagates immediately with a clean traceback in `.command.err`.

### Concern 2: shadow log directory

**With submitit:**

```bash
nextflow run main.nf --python_bin $PYTHON_BIN --mode success
# or with Slurm:
nextflow run main.nf -profile slurm --python_bin $PYTHON_BIN --mode success
```

After completion, check the Nextflow work directories:

```bash
# submitit's shadow logs — duplicated output in a separate directory
find work -name "*_log.*" -path "*/submitit_logs/*"
```

The `DebugExecutor` adds `FileHandler`s to the Python root logger that write to
`submitit_logs/` inside each work directory. These files are invisible to
`nextflow log`, `publishDir`, and `trace.txt`.

**Without submitit:**

```bash
nextflow run main.nf --python_bin $PYTHON_BIN --mode success --bypass_submitit true
```

All logging goes to stderr, captured by Nextflow in `.command.err`. No shadow
directory is created.

### Concern 3: `os.environ` clear/restore

**With submitit:**

```bash
NXF_DEMO_VAR=hello nextflow run main.nf --python_bin $PYTHON_BIN --mode success
```

Inside `toy_transform`, the `CUDA_VISIBLE_DEVICES` and `NXF_TASK_WORKDIR`
values are logged. When running through `DebugExecutor`, the env is
snapshotted at `executor.submit()` time and replayed before execution.
Variables injected between submit and execute (e.g., by Nextflow's
`beforeScript` or container runtime) would be silently dropped.

In this toy example the gap is negligible (debug mode is synchronous), but
the mechanism is fundamentally unsafe — and the `os.environ.clear()` call
could interact badly with libraries that cache env vars lazily.

## Recommendation

For the Nextflow `--debug` path, bypass submitit entirely:

```python
# Instead of:
executor = submitit.AutoExecutor(folder=..., cluster="debug")
job = executor.submit(process_fn, ...)
job.results()

# Do:
process_fn(...)
```

This gives clean error propagation, unified logging, and no env interference.
The submitit path remains available for the existing interactive/HPC workflow.
