#!/usr/bin/env bash
# M3 — single-host Monarch full c0041 run on 2 H200.
#
# Submit:
#   sbatch scripts/distributed/sbatch_tile_stitch_singlenode.sh \
#       --config settings/tile-rec-stitch/example_minimal.yml \
#       --input  /hpc/projects/waveorder/tile-stitch/sample_datasets/l0_brightfield_fov.zarr \
#       --output /hpc/projects/waveorder/tile-stitch/runs/tile_stitch/output \
#       --channel "camera 22500102 view 0 @ 780nm"
#
#SBATCH --job-name=tile-stitch-singlenode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@90
#SBATCH --output=%x_%j.log

set -euo pipefail

if [ -n "${BIAHUB_REPO:-}" ]; then
    REPO_DIR="$BIAHUB_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
    REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
fi
cd "$REPO_DIR"

# shared: PATH, CUDA module, runtime env exports, $RUN (uv invocation prefix)
source scripts/distributed/_common.sh

exec $RUN biahub tile-stitch "$@"
