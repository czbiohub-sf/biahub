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
#   ./run_mantis_v2.sh --max_positions 1      # quick smoke test, one position
#
# Smoke-testing with --max_positions 1 first is worth the few minutes: it walks
# all six steps on one position and catches config/schema errors before a
# full-plate run spends hours reaching virtual-stain.

module load nextflow
module load uv
# No `module load cuda`. The torch wheel bundles its own CUDA runtime
# (torch 2.13.0+cu130 -> CUDA 13.0), so the GPU steps need only the driver, which
# is already on the GPU nodes. Verified on an L40S with no cuda module loaded:
# torch.cuda.is_available() True, a GPU matmul, and a cellpose gpu=True eval all
# succeed. Loading a system cuda would put a *different* version (12.8) on
# LD_LIBRARY_PATH ahead of the bundled one — a mismatch to introduce, not fix.
set -euo pipefail

# --- fill in ---------------------------------------------------------------
# DATASET is the output stem: every step writes <DATASET>.zarr, so use the clean
# YYYY_MM_DD_<description> name even when the raw store is spelled differently.
DATASET=""                 # e.g. 2026_07_14_A549_MAP4_ZIKV
RAW_STORE=""               # basename of the raw store, e.g. ${DATASET}_1.ome.zarr
PROJECT_DIR=""             # e.g. /hpc/projects/intracellular_dashboard/organelle_dynamics
# The biahub checkout. Used to locate the pipeline files AND the venv this script
# activates below — it is no longer passed to the pipeline as a parameter.
BIAHUB_PROJECT=""          # e.g. /hpc/mydata/<user>/biahub

# Output directory. Defaults to ${PROJECT_DIR}/${DATASET}; set it explicitly for
# a sibling run — the convention for reprocessing the same data is
# "${PROJECT_DIR}/${DATASET}_rerun", and "_smoketest" for a short trial store.
OUTPUT_DIR=""

# The 0-convert plate, for acquisitions with no HCS plate (neuromast /
# zebrafish / dynatrack). Leave empty to read the raw store directly. Resolved
# after OUTPUT_DIR below, so it may reference it.
CONVERTED_ZARR=""          # e.g. ${OUTPUT_DIR}/0-convert/${DATASET}.zarr
# ---------------------------------------------------------------------------

DATA_DIR="/hpc/instruments/cm.mantis"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/${DATASET}}"
CONFIGS="${OUTPUT_DIR}/configs"
PIPELINE="${BIAHUB_PROJECT}/nextflow/mantis-v2.nf"
NF_CONFIG="${BIAHUB_PROJECT}/nextflow/nextflow.config"

# Resolve the environment ONCE, here, and let it propagate. The pipeline calls
# `biahub` and `viscy` as bare commands: sbatch exports this shell's environment
# to every task (--export=ALL is the default) and the venv is on shared storage,
# so the compute nodes resolve the same absolute paths. `uv sync` is the whole
# provisioning step — no per-task `uv run`, so tasks never contend on
# site-packages. mantis-v2.nf calls check_environment() and fails at launch if
# this activation is missing.
uv sync --project "${BIAHUB_PROJECT}"
# shellcheck disable=SC1091
set +u; source "${BIAHUB_PROJECT}/.venv/bin/activate"; set -u
command -v biahub >/dev/null || { echo "biahub not on PATH after activation" >&2; exit 1; }

BIAHUB_BRANCH="$(git -C "${BIAHUB_PROJECT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
BIAHUB_COMMIT="$(git -C "${BIAHUB_PROJECT}" log -1 --format='%H %cI %s' 2>/dev/null || echo '?')"
BIAHUB_DIRTY="$(git -C "${BIAHUB_PROJECT}" status --porcelain 2>/dev/null || true)"

# Default the converted plate to the conventional location if it exists there.
if [[ -z "${CONVERTED_ZARR}" && -d "${OUTPUT_DIR}/0-convert/${DATASET}.zarr" ]]; then
    CONVERTED_ZARR="${OUTPUT_DIR}/0-convert/${DATASET}.zarr"
    echo "using 0-convert plate: ${CONVERTED_ZARR}"
fi

INPUT_ZARR="${CONVERTED_ZARR:-${DATA_DIR}/${DATASET}/${RAW_STORE}}"

[[ -d "${INPUT_ZARR}" ]] || { echo "input not found: ${INPUT_ZARR}" >&2; exit 1; }
[[ -d "${CONFIGS}"    ]] || { echo "configs not found: ${CONFIGS}"  >&2; exit 1; }

# Record which code and inputs this run used, to a FILE as well as the console.
#
# The file is the durable record. The launch is deliberately not piped through
# `tee` (Nextflow only renders its live progress table when stdout is a terminal),
# and `.nextflow.log` captures the launch command line but not the git state — so
# without this, "which commit produced this output?" is unanswerable once the tmux
# pane is gone. Appends, so a `-resume` relaunch adds an entry rather than erasing
# the original run's provenance.
PROVENANCE="${OUTPUT_DIR}/nextflow/provenance.txt"
mkdir -p "${OUTPUT_DIR}/nextflow"
{
    echo "=== $(date -Is) ==="
    echo "dataset   ${DATASET}"
    echo "input     ${INPUT_ZARR}"
    echo "output    ${OUTPUT_DIR}"
    echo "biahub    ${BIAHUB_PROJECT}"
    echo "branch    ${BIAHUB_BRANCH}"
    echo "commit    ${BIAHUB_COMMIT}"
    echo "host      $(hostname)"
    echo "nextflow  $(nextflow -version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    if [[ -n "${BIAHUB_DIRTY}" ]]; then
        echo "dirty     YES — not reproducible from the commit above:"
        echo "${BIAHUB_DIRTY}" | sed 's/^/          /'
    fi
    if [[ "${BIAHUB_BRANCH}" != "main" ]]; then
        echo "warning   not on main"
    fi
} >> "${PROVENANCE}"

echo "biahub: ${BIAHUB_PROJECT}"
echo "  branch ${BIAHUB_BRANCH}  ${BIAHUB_COMMIT%% *}"
echo "  provenance appended to ${PROVENANCE}"
if [[ -n "${BIAHUB_DIRTY}" ]]; then
    echo "  WARNING: uncommitted changes — this run is not reproducible from the commit above" >&2
fi
if [[ "${BIAHUB_BRANCH}" != "main" ]]; then
    echo "  WARNING: not on main" >&2
fi

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
    -resume \
    "$@"
