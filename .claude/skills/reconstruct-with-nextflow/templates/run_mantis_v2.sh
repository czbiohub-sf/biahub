#!/usr/bin/env bash
# Run the mantis-v2 reconstruction pipeline (biahub nextflow/mantis-v2.nf):
#   flat-field -> deskew -> reconstruct -> virtual-stain -> assemble -> track
#
# Copy this into the project output directory and fill in the five variables
# below. Keep it there: it is the run's provenance record of the exact command.
#
# --output is the project root, so each step writes a sibling directory:
#   0-flatfield/ 1-deskew/ 2-reconstruct/ 3-virtual-stain/ 4-track/ 5-assemble/
# The work dir defaults to <output>/nextflow/work; override with `-work-dir`.
#
# Any extra arguments are forwarded to nextflow, e.g.
#   ./run_mantis_v2.sh -profile local
#   ./run_mantis_v2.sh --max_positions 2      # quick smoke test

module load nextflow
module load uv
module load cuda/12.8.0_570.86.10
set -euo pipefail

# --- fill in ---------------------------------------------------------------
DATASET=""                 # e.g. 2026_07_14_A549_MAP4_ZIKV
RAW_STORE=""               # basename of the raw store, e.g. ${DATASET}_1.ome.zarr
PROJECT_DIR=""             # e.g. /hpc/projects/intracellular_dashboard/organelle_dynamics
BIAHUB_PROJECT=""          # e.g. /hpc/mydata/taylla.theodoro/repo/biahub
# Set to the 0-convert plate for acquisitions with no HCS plate (neuromast /
# zebrafish / dynatrack); leave empty to read the raw store directly.
CONVERTED_ZARR=""          # e.g. ${OUTPUT_DIR}/0-convert/${DATASET}.zarr
# ---------------------------------------------------------------------------

DATA_DIR="/hpc/instruments/cm.mantis"
OUTPUT_DIR="${PROJECT_DIR}/${DATASET}"
CONFIGS="${OUTPUT_DIR}/configs"
PIPELINE="${BIAHUB_PROJECT}/nextflow/mantis-v2.nf"
NF_CONFIG="${BIAHUB_PROJECT}/nextflow/nextflow.config"

INPUT_ZARR="${CONVERTED_ZARR:-${DATA_DIR}/${DATASET}/${RAW_STORE}}"

[[ -d "${INPUT_ZARR}" ]] || { echo "input not found: ${INPUT_ZARR}" >&2; exit 1; }
[[ -d "${CONFIGS}"    ]] || { echo "configs not found: ${CONFIGS}"  >&2; exit 1; }

nextflow run "${PIPELINE}" \
    -c "${NF_CONFIG}" \
    -profile slurm \
    --input                "${INPUT_ZARR}" \
    --output               "${OUTPUT_DIR}" \
    --flat_field_config    "${CONFIGS}/flat_field.yml" \
    --deskew_config        "${CONFIGS}/deskew.yml" \
    --reconstruct_config   "${CONFIGS}/reconstruct.yml" \
    --virtual_stain_config "${CONFIGS}/virtual_stain.yml" \
    --concatenate_config   "${CONFIGS}/concatenate.yml" \
    --track_config         "${CONFIGS}/track.yml" \
    --biahub_project       "${BIAHUB_PROJECT}" \
    -resume \
    "$@"
