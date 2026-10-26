#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
# Track a few positions of a segmented projected plate with one tracker, in one job.
#
# The plate is the Z=1 projected plate (<N>-assemble/<dataset>_proj.zarr) with the
# segment label image labels/nuclei in every position: the A549 pipeline writes both.
# For a plate not segmented yet, segment the positions first (one GPU job), with the
# A549 segment config (nextflow/configs/a549/segment.yml):
#   biahub segment --init -i $INPUTS -c segment.yml
#   biahub segment --cluster debug -i $INPUTS -c segment.yml
#
# Track writes INTO the plate: labels/<output_label> and tracks_<output_label>_<fov>.csv
# (ultrack also tracks_<output_label>.geff) in every position, one output_label per
# config, so the runs sit side by side. No -o: it would only keep ultrack's databases.
# Copy this next to the track-<tracker>.yml configs in the run directory, fill in the
# variables, run the --inits one after another (each adds its label to the positions'
# label list, and concurrent --inits could drop each other's entry), then submit one
# job per tracker:
#   for c in track-*.yml; do INIT_ONLY=1 bash run_track.sh $c; done
#   for c in track-*.yml; do sbatch --job-name=track_$(basename $c .yml) run_track.sh $c; done
# Keep the script there: it is the run's provenance record.
#
# One header for all four: hoct and trackastra use the GPU; ultrack and tracksdata are
# CPU only and do not need it (drop --gres and use cpu for them if gpu is busy).
# Not `preempted`: tracking cannot resume, so a preemption restarts it.
# For a whole plate, drop --cluster debug and let biahub track submit one job per position.
set -euo pipefail

BIAHUB=/path/to/biahub                     # checkout on main (track extra; segment for cellpose)
PLATE=/path/to/<dataset>_proj.zarr         # the projected plate, segmented (labels/nuclei)
POSITIONS="A/1/000000 A/2/000000"          # a few FOVs per condition
CONFIG=${1:?usage: run_track.sh track-<tracker>.yml}

source "$BIAHUB/.venv/bin/activate"
INPUTS=$(for p in $POSITIONS; do printf '%s ' "$PLATE/$p"; done)

# --init checks the config against the plate, creates labels/<output_label> and
# downloads the HOCT / Trackastra weights once; again in the job it is a no-op.
biahub track --init -i $INPUTS -c "$CONFIG"
if [[ -n "${INIT_ONLY:-}" ]]; then exit 0; fi
biahub track --cluster debug -i $INPUTS -c "$CONFIG"
