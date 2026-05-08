#!/usr/bin/env bash
# EXAMPLE GPU sbatch wrapper for `biahub tile-stitch --gpu` (single-node).
# Adapt for your environment.
#
# Whole-node `--exclusive` allocation; LocalCUDACluster auto-detects all
# allocated GPUs from CUDA_VISIBLE_DEVICES. The driver runs in this same
# allocation — there is no separate dask-jobqueue worker pool on this
# path (multi-node `slurm-cuda` deferred).
#
# Tune these values for your cluster:
#   --partition  : GPU partition name (Bruno uses 'gpu')
#   --gres=gpu:N : number of GPUs to allocate; LocalCUDACluster will
#                  auto-detect and create one dask worker per GPU
#   --constraint : GPU class selector — must satisfy the rmm_pool_size
#                  in your YAML (validator requires pool ≤ 0.85 × min VRAM
#                  across the alternatives)
#   --time       : driver walltime
#   module load  : your cluster's CUDA + UCX modules (Bruno: cuda 13.1, hpcx 2.19)
#
# Usage:
#   sbatch scripts/tile_stitch/example_tile_stitch_gpu.sh \
#       --config /path/to/dispatcher.yml \
#       --input  /path/to/input.zarr/0/0/0 \
#       --output /path/to/output_basename \
#       --gpu
#
#SBATCH --job-name=tile-stitch-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --constraint=h100|h200
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@90
#SBATCH --output=%x_%j.log

set -euo pipefail

# `uv` typically lives in ~/.local/bin which isn't on the default
# non-login-shell PATH. Add it explicitly so sbatch can find uv.
export PATH="$HOME/.local/bin:$PATH"

# Resolve the biahub repo so `uv run` finds the project's pyproject.toml.
# See example_tile_stitch_cpu.sh for the resolution order.
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

module load cuda/13.1.0_590.44.01 hpcx/2.19

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1

exec uv run --no-sync --extra tilestitch-gpu --index-strategy unsafe-best-match \
    biahub tile-stitch "$@"
