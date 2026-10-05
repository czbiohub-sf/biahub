#!/usr/bin/env bash
# Estimate a registration (or stabilization) with biahub's registration engine.
#
# Copy this into the run's output directory next to the config and fill in the
# variables below. Keep it there: it is the run's provenance record of the exact command.
#
# Outputs, next to OUTPUT:
#   transforms.yml                 one transform per timepoint, with score and status
#   estimate_transform_report.json scores, flagged, repairs, stand-ins
#   run_journal.json               every repair / sweep attempt
#   timepoints/ repairs/           per-timepoint records (what --resume continues from)
#
# Any extra arguments are forwarded, e.g. `./run_estimate_transform.sh --resume`.
set -euo pipefail

BIAHUB=/path/to/biahub            # checkout on main
MOVING=/path/to/deskew.zarr/R/C/FOV                # light-sheet, beads well
REFERENCE=/path/to/reconstruct.zarr/R/C/FOV        # phase, same well; leave empty to stabilize
CONFIG=./estimate-transform-beads.yml
OUTPUT=./transforms.yml
SBATCH=""                         # optional: a file with `#SBATCH --time=...` for large volumes

source "$BIAHUB/.venv/bin/activate"
export PYTHONWARNINGS="ignore::FutureWarning"   # scikit-image deprecation noise per timepoint

args=(-m "$MOVING" -c "$CONFIG" -o "$OUTPUT" --cluster slurm)
[[ -n "$REFERENCE" ]] && args+=(-r "$REFERENCE")
[[ -n "$SBATCH" ]] && args+=(-sb "$SBATCH")

biahub estimate-transform "${args[@]}" "$@"
