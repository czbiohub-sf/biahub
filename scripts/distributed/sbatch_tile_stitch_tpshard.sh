#!/usr/bin/env bash
# TP-parallel Monarch tile-stitch — N nodes, each chews a contiguous shard
# of the TP range on its OWN local GPUs (independent single-host runtimes).
#
# This is the right scaling pattern for many timepoints: timepoints are
# independent, so we shard them ACROSS nodes (linear, each node reads its
# own TPs) rather than splitting one TP across nodes (FS-bound, sublinear).
# Each node runs its own Monarch this_host() actor mesh on 2 GPUs.
#
# Submit (10 TPs across 2 nodes → 5 TPs/node):
#   sbatch scripts/distributed/sbatch_tile_stitch_tpshard.sh \
#       --config settings/tile-rec-stitch/example_minimal.yml \
#       --input  /hpc/projects/waveorder/tile-stitch/sample_datasets/deskewed_t100_c0.zarr \
#       --output /hpc/projects/waveorder/tile-stitch/runs/bench_t10_monarch_tpshard/output \
#       --channel "camera 22500102 view 0 @ 780nm" \
#       --timepoints 0-9
#
#SBATCH --job-name=tile-stitch-tpshard
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.log

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
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

# One driver per node (--ntasks-per-node=1). Each task reads its
# SLURM_PROCID / SLURM_NTASKS and processes its contiguous TP shard on its
# 2 local GPUs (single-host this_host mesh — NOT a cross-node HostMesh).
# proc 0 creates the shared T=N output zarr; others wait for it.
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --gres=gpu:2 \
    uv run --no-sync --extra tilestitch-gpu \
    --index-strategy unsafe-best-match \
    biahub tile-stitch --shard-by-proc "$@"

echo "tpshard wrapper done"
