nextflow run ../nextflow/example-flatfield-deskew-reconstruct.nf \
    -profile slurm \
    --venv_path /home/aliu/repos/biahub/.venv \
    --input_zarr /hpc/projects/intracellular_dashboard/organelle_dynamics/2026_03_18_A549_CAAX_DRAQ5_DENV_ZIKV/pipeline/0-convert/2026_03_18_A549_CAAX_DRAQ5_DENV_ZIKV.zarr \
    --output_dir /hpc/projects/intracellular_dashboard/organelle_dynamics/2026_03_18_A549_CAAX_DRAQ5_DENV_ZIKV/pipeline/1-nextflow-test-preprocess \
    --flat_field_config /hpc/projects/intracellular_dashboard/organelle_dynamics/2026_03_18_A549_CAAX_DRAQ5_DENV_ZIKV/pipeline/1-preprocess/0-flatfield/flatfield_settings.yaml \
    --deskew_config /hpc/projects/intracellular_dashboard/organelle_dynamics/2026_03_18_A549_CAAX_DRAQ5_DENV_ZIKV/pipeline/1-preprocess/1-deskew/deskew_settings.yml \
    --reconstruct_config /hpc/projects/intracellular_dashboard/organelle_dynamics/2026_03_18_A549_CAAX_DRAQ5_DENV_ZIKV/pipeline/1-preprocess/2-reconstruct/phase_setting.yaml