#!/usr/bin/env bash
# TP-parallel Monarch tile-stitch — shard the timepoint range ACROSS tasks,
# each task running its OWN single-host this_host() actor mesh on its local
# GPU(s). Timepoints are independent, so sharding them across tasks scales
# ~linearly (each task reads only its own TPs); splitting ONE TP across nodes
# (a cross-node HostMesh) is FS-bound and sublinear, so we don't.
#
# proc 0 creates the shared T=N output zarr; the others wait for it, then all
# write their disjoint T-slots concurrently (--shard-by-proc). The srun step
# INHERITS the job's GPU geometry, so the SAME wrapper serves both layouts —
# pick one at submit time:
#
#   Homogeneous (default: 2 nodes × 2 GPU — the bench-winning 2×2):
#     sbatch scripts/distributed/sbatch_tile_stitch_tpshard.sh \
#         --config settings/tile-rec-stitch/example_minimal.yml \
#         --input  /hpc/projects/waveorder/tile-stitch/sample_datasets/deskewed_t100_c0.zarr \
#         --output /hpc/projects/waveorder/tile-stitch/runs/bench_t10_monarch_tpshard/output \
#         --channel "camera 22500102 view 0 @ 780nm" --timepoints 0-9
#
#   Heterogeneous / availability (4 single-GPU tasks, any node mix, h100|h200) —
#   grabs scattered GPUs when no 2-GPU node is free (sublinear vs 2×2, but runs):
#     sbatch --ntasks=4 --gpus-per-task=1 --constraint="h100|h200" \
#         --mem-per-gpu=400G scripts/distributed/sbatch_tile_stitch_tpshard.sh ...
#
# MEM: each task reconstructs its TP's FULL input-tile grid on one mesh, with a
# large transient spike on the TP→TP volume swap (recons + blend accumulators +
# streaming buffers). 128G and 160G both OOM-killed on the 2nd-TP swap; the
# homogeneous default uses --mem=900G/node, the hetero override --mem-per-gpu=400G.
# srun fails the WHOLE step if ANY task OOMs, so provision every task for the worst.
#
#SBATCH --job-name=tile-stitch-tpshard
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=00:45:00
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

# One driver per task. Each reads SLURM_PROCID / SLURM_NTASKS and processes its
# contiguous TP shard on its local GPU(s) (single-host this_host mesh — NOT a
# cross-node HostMesh). No --gres / --gpus-per-task pinned here: the step inherits
# the job's GPU geometry, so a submit-time override reshapes it without editing
# this file. --gpus-per-task=1 scopes CUDA_VISIBLE_DEVICES per task when SLURM
# co-locates tasks on one node.
srun --ntasks="$SLURM_NTASKS" \
    bash -c 'echo "task PROCID=${SLURM_PROCID}/${SLURM_NTASKS} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; \
        $RUN biahub tile-stitch --shard-by-proc "$@"' _ "$@"

echo "tpshard wrapper done"
