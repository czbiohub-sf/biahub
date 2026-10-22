#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
# Track a few positions with one tracker, in one GPU job (cellpose + the tracker).
#
# Copy this next to the track-<tracker>.yml configs in the run directory, fill in the
# variables, and submit one job per tracker:
#   sbatch --job-name=track_ultrack    run_track.sh track-ultrack.yml
#   sbatch --job-name=track_trackastra run_track.sh track-trackastra.yml
# Each writes <run dir>/<config stem>/track.zarr (labels + tracks_<fov>.csv per position).
# Keep the script there: it is the run's provenance record.
#
# Runs on `gpu`, not `preempted`: tracking cannot resume, so a preemption restarts it.
# For a whole plate, drop --cluster debug and let biahub track submit one job per position.
set -euo pipefail

BIAHUB=/path/to/biahub                     # checkout on main (segment + track extras)
PLATE=/path/to/<dataset>.zarr              # the assembled plate track reads
POSITIONS="A/1/000000 A/2/000000"          # a few FOVs per condition
CONFIG=${1:?usage: run_track.sh track-<tracker>.yml}

source "$BIAHUB/.venv/bin/activate"
OUT="$PWD/$(basename "$CONFIG" .yml)"
INPUTS=$(for p in $POSITIONS; do printf '%s ' "$PLATE/$p"; done)

mkdir -p "$OUT"
# --init downloads the model weights once (cellpose, and HOCT / Trackastra if selected).
biahub track --init -i $INPUTS -o "$OUT/track.zarr" -c "$CONFIG"
biahub track --cluster debug -i $INPUTS -o "$OUT/track.zarr" -c "$CONFIG"
