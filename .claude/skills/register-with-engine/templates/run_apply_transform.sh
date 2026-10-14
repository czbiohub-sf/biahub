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
                                               # -- set it when the moving store mixes cameras
                                               #    (DragonFly: the BF is on the reference camera)
CROP_TO_OVERLAP=false                          # true: crop to the box every transform covers
SBATCH=""                                      # optional sbatch file with a time limit
VENV=""                           # empty: $BIAHUB/.venv; or another biahub venv (runs BIAHUB's code)

# Relative paths are next to this script, wherever it is launched from.
cd "$(dirname "$0")"
source "${VENV:-$BIAHUB/.venv}/bin/activate"
export PYTHONPATH="$BIAHUB${PYTHONPATH:+:$PYTHONPATH}"   # run this checkout, whatever the venv

# Expand each glob into one argument per position, keeping spaces inside a path.
mapfile -t moving < <(compgen -G "$MOVING")
[[ ${#moving[@]} -gt 0 ]] || { echo "no positions match MOVING=$MOVING" >&2; exit 1; }
args=(-m "${moving[@]}" -c "$TRANSFORMS" -o "$OUTPUT" --cluster slurm --monitor)
if [[ -n "$REFERENCE" ]]; then
    mapfile -t reference < <(compgen -G "$REFERENCE")
    [[ ${#reference[@]} -gt 0 ]] || { echo "no positions match REFERENCE=$REFERENCE" >&2; exit 1; }
    args+=(-r "${reference[@]}")
fi
for channel in "${CHANNELS[@]}"; do args+=(--channels "$channel"); done
[[ "$CROP_TO_OVERLAP" == true ]] && args+=(--crop-to-overlap)
[[ -n "$SBATCH" ]] && args+=(-sb "$SBATCH")

biahub apply-transform "${args[@]}" "$@"
