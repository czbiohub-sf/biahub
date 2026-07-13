#!/usr/bin/env bash
# tile-stitch launcher — one wrapper for single- AND multi-node.
#
# Branches on SLURM_NNODES: 1 node → run the driver (single-host this_host mesh);
# >1 node → bring up a worker_loop on every node (background srun) + run the
# driver, which auto-attaches via a HostMesh. Topology (nodes/port/ready-dir) is
# derived from the SLURM allocation by the engine — you pass only -i/-c/-o.
#
# Request resources on the sbatch line; the wrapper figures out the rest:
#   # single node, 2 GPUs
#   sbatch --nodes=1 --gres=gpu:2 scripts/distributed/sbatch_tile_stitch.sh \
#       -i input.zarr/*/*/* -c config.yml -o output.zarr
#   # multi node (2 nodes × 2 GPUs) — identical command, just --nodes=2
#   sbatch --nodes=2 --gres=gpu:2 scripts/distributed/sbatch_tile_stitch.sh \
#       -i input.zarr/*/*/* -c config.yml -o output.zarr
#
#SBATCH --job-name=tile-stitch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=01:00:00
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

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    # One worker loop per node (background); each derives its port + ready-dir
    # from the SLURM env (matching the driver). The driver attaches via HostMesh.
    srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
        $RUN python -m biahub.tile_stitch.monarch.worker_loop &
    WPID=$!
    $RUN biahub tile-stitch "$@" || true
    kill "$WPID" 2>/dev/null || true
    wait 2>/dev/null || true
else
    exec $RUN biahub tile-stitch "$@"
fi
echo "tile-stitch wrapper done"
