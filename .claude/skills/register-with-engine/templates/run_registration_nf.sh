#!/usr/bin/env bash
# Estimate a registration (or stabilization) and apply it with biahub's Nextflow
# workflow (nextflow/registration.nf): one task per position x timepoint, then
# flagging, repair, the transforms file, and one apply task per position. Nextflow
# owns the fan-out, retries (preemption, time limit, memory) and -resume.
#
# Copy this into the run's output directory next to the config and fill in the
# variables below. Keep it there: it is the run's provenance record of the exact command.
#
# Outputs, under OUTPUT:
#   transforms.yml                 one transform per timepoint, with score and status
#   transforms/                    the run folder: report, journal, per-timepoint records
#   <moving store name>.zarr       the applied plate (with APPLY=true)
#   nextflow/                      work dir, report.html, trace.txt,
#                                  slurm_output/{estimate_transform,apply_transform}/ logs
#
# Manual registration is interactive and cannot run here: run it with
# run_estimate_transform.sh in a session with a display, then set TRANSFORMS to its file.
#
# Any extra arguments are forwarded to nextflow, e.g. `./run_registration_nf.sh -profile local`.

module load nextflow
set -euo pipefail

# --- fill in ---------------------------------------------------------------
BIAHUB=/path/to/biahub                       # checkout on main
MOVING=/path/to/deskew.zarr                  # plate with the moving channel (light-sheet)
REFERENCE=/path/to/reconstruct.zarr          # plate with the reference (phase); empty to stabilize
CONFIG=./estimate-transform-beads.yml        # estimate config; empty when TRANSFORMS is set
ESTIMATE_POSITIONS='C/1/000000'              # beads well (one shared list) or '*/*/*' (one each)
TRANSFORMS=""                                # apply this existing file instead of estimating
APPLY=true                                   # apply after estimating
APPLY_POSITIONS='*/*/*'
CROP_TO_OVERLAP=false                        # true: crop to the box every transform covers
CHANNELS=""                                  # comma-separated moving channels; empty = all
OUTPUT=.                                     # this run directory
# ---------------------------------------------------------------------------

# Nextflow tasks run in their own work directory: pass absolute paths.
OUTPUT=$(realpath "${OUTPUT}")
MOVING=$(realpath "${MOVING}")
[[ -n "${REFERENCE}" ]] && REFERENCE=$(realpath "${REFERENCE}")
[[ -n "${CONFIG}" ]] && CONFIG=$(realpath "${CONFIG}")
[[ -n "${TRANSFORMS}" ]] && TRANSFORMS=$(realpath "${TRANSFORMS}")

# The tasks call `biahub` bare; sbatch exports this shell's environment to them.
# shellcheck disable=SC1091
set +u; source "${BIAHUB}/.venv/bin/activate"; set -u
command -v biahub >/dev/null || { echo "biahub not on PATH after activation" >&2; exit 1; }
export PYTHONWARNINGS="ignore::FutureWarning"   # scikit-image deprecation noise per timepoint

# Nextflow 26.04 renders agent-mode output when CLAUDECODE is set (a Claude Code tmux
# pane inherits it); see the reconstruct-with-nextflow skill's run_mantis_v2.sh.
unset CLAUDECODE

args=(--moving "${MOVING}" --output "${OUTPUT}" --apply_positions "${APPLY_POSITIONS}")
[[ -n "${REFERENCE}" ]] && args+=(--reference "${REFERENCE}")
if [[ -n "${TRANSFORMS}" ]]; then
    args+=(--transforms "${TRANSFORMS}")
else
    args+=(--estimate_config "${CONFIG}" --estimate_positions "${ESTIMATE_POSITIONS}")
    [[ "${APPLY}" == true ]] && args+=(--apply)
fi
[[ "${CROP_TO_OVERLAP}" == true ]] && args+=(--crop_to_overlap)
[[ -n "${CHANNELS}" ]] && args+=(--channels "${CHANNELS}")

nextflow run "${BIAHUB}/nextflow/registration.nf" \
    -c "${BIAHUB}/nextflow/nextflow.config" \
    -profile slurm \
    "${args[@]}" \
    -resume \
    "$@"
