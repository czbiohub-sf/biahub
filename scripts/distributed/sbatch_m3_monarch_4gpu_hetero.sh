#!/usr/bin/env bash
# TP-parallel Monarch tile-stitch — 4 single-GPU shards across FLEXIBLE node
# placement, mixing H100 and H200. Derived from sbatch_m3_monarch_tpshard.sh.
#
# Design: timepoints are independent, so we shard them across 4 single-GPU
# tasks. Each task runs its own single-host Monarch this_host() runtime on its
# ONE local GPU (torch.cuda.device_count()==1 via --gpus-per-task=1). 10 TPs /
# 4 shards = ceil(10/4)=3 → shards get [0-2],[3-5],[6-8],[9].
#
# We request 4 single-GPU tasks with constraint h100|h200 and DO NOT pin
# --nodes / --ntasks-per-node, so SLURM is free to place the 4 GPUs on
# whatever is available: 1x4, 2x2, 4x1, or any mix of node types. This is the
# cleanest way to satisfy a heterogeneous 4-GPU request — SLURM treats each
# task as an independent single-GPU consumer.
#
# If SLURM packs >1 task on a node, --gpus-per-task=1 scopes CUDA_VISIBLE_DEVICES
# per task, so each Monarch this_host() sees exactly one GPU. The single-host
# Monarch path uses ephemeral bootstrap ports (the fixed --port 26000 is only
# used by the multi-host attach_to_workers path, which we don't take here), so
# two independent runtimes on one node don't collide.
#
# proc 0 creates the shared T=10 output zarr (T-chunk=1); other procs wait for
# it, then write their disjoint T-slots concurrently.
#
# Submit (10 TPs across 4 single-GPU shards):
#   sbatch scripts/distributed/sbatch_m3_monarch_4gpu_hetero.sh \
#       --config settings/tile-rec-stitch/monarch_2gpu.yml \
#       --input  /hpc/projects/waveorder/tile-stitch/sample_datasets/deskewed_t100_c0.zarr \
#       --output /hpc/projects/waveorder/tile-stitch/runs/bench_t10_monarch_4gpu_hetero/output \
#       --channel "camera 22500102 view 0 @ 780nm" \
#       --timepoints 0-9
#
#SBATCH --job-name=m3-monarch-4gpu-hetero
#SBATCH --partition=gpu
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=400G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.log

# NOTE on memory: each single-GPU task reconstructs the FULL ~200-input-tile
# grid for its TP on ONE actor, with a large transient spike during the
# TP->TP volume swap (recons + blend accumulators + streaming buffers). The
# validated single-node config uses --mem=400G; empirically 128G and 160G both
# OOM-kill on the 2nd-TP swap spike (jobs 33435766/33435830). So we set
# --mem-per-gpu=400G: each GPU/task carries 400G, so a single-task node gets
# 400G and a co-located 2-task node gets 800G. GPU nodes have ~2 TB RAM, so
# even 4 tasks on one node (1.6 TB) fits. NOTE: srun fails the WHOLE step if
# any one task OOMs, so every task must be provisioned for the worst case.

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

echo "=== alloc: nodes=${SLURM_JOB_NODELIST:-?} ntasks=${SLURM_NTASKS:-?} ==="

# One driver per task (4 tasks total, 1 GPU each). Each task reads its
# SLURM_PROCID / SLURM_NTASKS and processes its contiguous TP shard on its
# single local GPU (single-host this_host mesh — NOT a cross-node HostMesh).
# --gpus-per-task=1 keeps torch.cuda.device_count()==1 per task even when SLURM
# co-locates tasks on one node. proc 0 creates the shared T=10 output zarr.
srun --ntasks="$SLURM_NTASKS" --gpus-per-task=1 \
    bash -c 'echo "task PROCID=${SLURM_PROCID} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; \
        uv run --no-sync --extra tilestitch-gpu \
        --index-strategy unsafe-best-match \
        biahub tile-stitch --shard-by-proc "$@"' _ "$@"

echo "4gpu-hetero wrapper done"
