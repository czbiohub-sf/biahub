#!/usr/bin/env bash
# EXAMPLE CPU sbatch wrapper for `biahub tile-stitch`. Adapt for your
# environment.
#
# Driver runs in this allocation; SLURMCluster spawns worker jobs via
# dask-jobqueue (per the cpu_pool config in the YAML). One driver, N
# workers — driver is lightweight enough to run on a small allocation.
#
# Tune these values for your cluster:
#   --partition  : SLURM partition name (Bruno uses 'cpu')
#   --time       : driver walltime (≥ longest expected pipeline run)
#   --mem        : driver memory (32 GB is plenty for the driver itself)
#   --cpus-per-task : driver core count (4 is enough for plan + dispatch)
#   --signal=USR1@90 : sends SIGUSR1 90 s before walltime — drives the
#                      install_preempt_handler graceful shutdown path
#
# Usage:
#   sbatch scripts/tile_stitch/example_tile_stitch_cpu.sh \
#       --config /path/to/dispatcher.yml \
#       --input  /path/to/input.zarr/0/0/0 \
#       --output /path/to/output_basename
#
#SBATCH --job-name=tile-stitch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@90
#SBATCH --output=%x_%j.log

set -euo pipefail

# `uv` typically lives in ~/.local/bin which isn't on the default
# non-login-shell PATH. Add it explicitly so sbatch can find uv.
export PATH="$HOME/.local/bin:$PATH"

# Resolve the biahub repo so `uv run` finds the project's pyproject.toml.
# Order:
#   1. $BIAHUB_REPO env var (explicit override)
#   2. $SLURM_SUBMIT_DIR (sbatch submission cwd — SLURM copies the script
#      itself to /var/spool/..., so BASH_SOURCE isn't the original path)
#   3. BASH_SOURCE/../.. (works for direct `bash <script>` invocation)
if [ -n "${BIAHUB_REPO:-}" ]; then
    REPO_DIR="$BIAHUB_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
    REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
fi
if [ ! -f "$REPO_DIR/pyproject.toml" ]; then
    echo "ERROR: no pyproject.toml at $REPO_DIR — set BIAHUB_REPO or sbatch from the repo dir" >&2
    exit 1
fi
cd "$REPO_DIR"

# Pin BLAS thread counts on the driver process. Workers inherit their own
# limits via dask-jobqueue's job_script_prologue (set per-pool in YAML).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# Force glibc malloc to release freed memory back to the OS — keeps
# unmanaged memory low so the dask nanny doesn't restart workers.
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-0}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}

exec uv run --no-sync biahub tile-stitch "$@"
