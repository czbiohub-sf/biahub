BIAHUB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINE_DIR="${1:-/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_03_18_A549_CAAX_DRAQ5_DENV_ZIKV/pipeline}"

cd "${PIPELINE_DIR}"

nextflow run "${BIAHUB_ROOT}/nextflow/example-flatfield-deskew-reconstruct.nf" \
    -profile slurm \
    --biahub_project "${BIAHUB_ROOT}" \
    --input_zarr "${PIPELINE_DIR}/0-convert/2026_03_18_A549_CAAX_DRAQ5_DENV_ZIKV.zarr" \
    --output_dir "${PIPELINE_DIR}/1-nextflow-test-preprocess" \
    --flat_field_config "${PIPELINE_DIR}/1-preprocess/0-flatfield/flatfield_settings.yaml" \
    --deskew_config "${PIPELINE_DIR}/1-preprocess/1-deskew/deskew_settings.yml" \
    --reconstruct_config "${PIPELINE_DIR}/1-preprocess/2-reconstruct/phase_setting.yaml" \
    -resume
