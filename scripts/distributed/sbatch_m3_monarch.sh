#!/usr/bin/env bash
# M3 — single-host Monarch full c0041 run on 2 H200.
#
# Submit:
#   sbatch scripts/distributed/sbatch_m3_monarch.sh \
#       --config settings/tile-rec-stitch/monarch_2gpu.yml \
#       --input  /hpc/projects/waveorder/tile-stitch/sample_datasets/l0_brightfield_fov.zarr \
#       --output /hpc/projects/waveorder/tile-stitch/runs/m3_monarch/output \
#       --channel "camera 22500102 view 0 @ 780nm"
#
#SBATCH --job-name=m3-monarch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@90
#SBATCH --output=%x_%j.log

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

if [ -n "${BIAHUB_REPO:-}" ]; then
    REPO_DIR="$BIAHUB_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
    REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
fi
cd "$REPO_DIR"

module load cuda/13.1.0_590.44.01 hpcx/2.19

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

exec uv run --no-sync \
    --extra tilestitch-gpu \
    --index-strategy unsafe-best-match \
    biahub tile-stitch "$@"
