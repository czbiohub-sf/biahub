#!/usr/bin/env bash
# Multi-host Monarch m3 tile-stitch — N nodes × G GPUs.
#
# Starts a Monarch host worker loop on every node (background srun), then
# runs the m3 driver on the batch node which attaches to all of them via
# a HostMesh and drives the streaming tile-stitch pipeline.
#
# Submit (2 nodes × 2 GPUs):
#   sbatch scripts/distributed/sbatch_m3_monarch_multinode.sh \
#       --config settings/tile-rec-stitch/monarch_2gpu.yml \
#       --input  /hpc/projects/waveorder/tile-stitch/sample_datasets/deskewed_t100_c0.zarr \
#       --output /hpc/projects/waveorder/tile-stitch/runs/bench_t10_monarch_2node/output \
#       --channel "camera 22500102 view 0 @ 780nm" \
#       --timepoints 0-9
#
#SBATCH --job-name=m3-monarch-mn
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:00:00
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

PORT=26000
GPUS_PER_NODE=2
NODES_CSV=$(scontrol show hostnames "$SLURM_NODELIST" | paste -sd, -)
READY_DIR="/hpc/projects/waveorder/tile-stitch/runs/scratch/ready_${SLURM_JOB_ID}"
rm -rf "$READY_DIR"; mkdir -p "$READY_DIR"
echo "nodelist: $NODES_CSV  port: $PORT  ready_dir: $READY_DIR"

RUN="uv run --no-sync --extra tilestitch-gpu --index-strategy unsafe-best-match"

# One host worker loop per node; each touches <host>.ready before binding.
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --gres=gpu:${GPUS_PER_NODE} \
    $RUN python -m biahub.tile_stitch.monarch.worker_loop \
    --port "$PORT" --ready-dir "$READY_DIR" &
WORKER_PID=$!

# Driver gates its attach on all <host>.ready files (robust to slow cold
# starts on non-batch nodes), so no fixed sleep needed.
# NOTE: gpus_per_node now lives in the YAML (config.monarch.gpus_per_node);
# keep it in sync with the --gres=gpu:N above.
$RUN biahub tile-stitch \
    --nodes "$NODES_CSV" --port "$PORT" \
    --ready-dir "$READY_DIR" "$@" || true

rm -rf "$READY_DIR" 2>/dev/null || true

kill "$WORKER_PID" 2>/dev/null || true
wait 2>/dev/null || true
echo "m3 multinode wrapper done"
