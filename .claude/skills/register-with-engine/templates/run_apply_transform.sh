#!/usr/bin/env bash
# Apply a transforms file (from estimate-transform) to positions.
#
# Copy this next to the transforms file and fill in the variables below. Keep it there:
# it is the record of how the output was produced.
#
# With REFERENCE set: registration -- every moving position is transformed onto the
# reference grid; the reference channels are copied and every moving channel transformed,
# except one the reference store also has and the file does not list in moving_channels
# (set CHANNELS to choose). Without REFERENCE: stabilization -- each store is
# transformed onto its own grid. Timepoints written with a non-accepted transform are
# printed and recorded in the output metadata.
set -euo pipefail

BIAHUB=/path/to/biahub                         # checkout on main
MOVING="/path/to/deskew.zarr/*/*/*"            # positions to transform (quoted glob)
REFERENCE="/path/to/reconstruct.zarr/*/*/*"    # leave empty to stabilize
TRANSFORMS=./transforms.yml
OUTPUT=/path/to/registered.zarr
CHANNELS=()                                    # e.g. ("GFP EX488 EM525-45"); empty = all
CROP_TO_OVERLAP=false                          # true: crop to the box every transform covers
SBATCH=""                                      # optional sbatch file with a time limit

source "$BIAHUB/.venv/bin/activate"

args=(-m $MOVING -c "$TRANSFORMS" -o "$OUTPUT" --cluster slurm --monitor)
[[ -n "$REFERENCE" ]] && args+=(-r $REFERENCE)
for channel in "${CHANNELS[@]}"; do args+=(--channels "$channel"); done
[[ "$CROP_TO_OVERLAP" == true ]] && args+=(--crop-to-overlap)
[[ -n "$SBATCH" ]] && args+=(-sb "$SBATCH")

biahub apply-transform "${args[@]}" "$@"
