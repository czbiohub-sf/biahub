#!/usr/bin/env bash
# Shared runtime setup for the scripts/distributed/ Monarch tile-stitch sbatch
# wrappers. SOURCE this (don't execute it) AFTER the wrapper has resolved
# REPO_DIR and cd'd into it — it's sourced by cwd-relative path, which dodges
# the sbatch spool-copy ${BASH_SOURCE[0]} gotcha:
#
#   cd "$REPO_DIR"
#   source scripts/distributed/_common.sh
#   ... $RUN biahub tile-stitch "$@"
#
# Sets: PATH (uv on it), the CUDA module, the runtime env exports, and $RUN —
# the uv invocation prefix (exported so a `srun bash -c '... $RUN ...'` child
# expands it from the propagated environment).

export PATH="$HOME/.local/bin:$PATH"

module load cuda/13.1.0_590.44.01 hpcx/2.19

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

export RUN="uv run --no-sync --extra tilestitch-gpu --index-strategy unsafe-best-match"
