#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.input_zarr        = null
params.output_dir        = null
params.flat_field_config = null
params.deskew_config     = null
params.reconstruct_config = null
params.num_processes     = 1
params.biahub_project    = null
params.work_dir          = null

// Derive dataset name from input zarr basename (e.g. "experiment.zarr" -> "experiment")
def dataset_name = params.input_zarr ?
    new File(params.input_zarr).name.replaceAll(/\.zarr$/, '') : null

def biahub_cmd = params.biahub_project ?
    "uv run --project ${params.biahub_project} biahub" : "biahub"


// ---------------------------------------------------------------------------
// Processes
// ---------------------------------------------------------------------------

process list_positions {
    label 'cpu_small'

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf list-positions -i ${params.input_zarr}
    """
}


process init_flat_field {
    label 'cpu_small'

    output:
    val true

    script:
    """
    ${biahub_cmd} nf init-flat-field \
        -i ${params.input_zarr} \
        -o ${params.output_dir}/0-flatfield/${dataset_name}.zarr \
        -c ${params.flat_field_config}
    """
}


process run_flat_field {
    label 'cpu_medium'
    tag "${position}"
    maxRetries 1
    errorStrategy 'retry'

    input:
    val position

    output:
    val position

    script:
    """
    ${biahub_cmd} nf run-flat-field \
        -i ${params.input_zarr} \
        -o ${params.output_dir}/0-flatfield/${dataset_name}.zarr \
        -p ${position} \
        -c ${params.flat_field_config} \
        -j ${task.cpus}
    """
}


process init_deskew {
    label 'cpu_small'

    input:
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd} nf init-deskew \
        -i ${params.output_dir}/0-flatfield/${dataset_name}.zarr \
        -o ${params.output_dir}/1-deskew/${dataset_name}.zarr \
        -c ${params.deskew_config}
    """
}


process run_deskew {
    label 'gpu'
    tag "${position}"
    maxRetries 1
    errorStrategy 'retry'

    input:
    val position

    output:
    val position

    script:
    """
    ${biahub_cmd} nf run-deskew \
        -i ${params.output_dir}/0-flatfield/${dataset_name}.zarr \
        -o ${params.output_dir}/1-deskew/${dataset_name}.zarr \
        -p ${position} \
        -c ${params.deskew_config}
    """
}


process init_reconstruct {
    label 'cpu_medium'

    input:
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd} nf init-reconstruct \
        -i ${params.output_dir}/1-deskew/${dataset_name}.zarr \
        -o ${params.output_dir}/2-reconstruct/${dataset_name}.zarr \
        -t ${params.output_dir}/2-reconstruct/transfer_function_${dataset_name}.zarr \
        -c ${params.reconstruct_config}
    """
}


process run_apply_inv_tf {
    label 'cpu_medium'
    tag "${position}"
    maxRetries 1
    errorStrategy 'retry'

    input:
    val position

    output:
    val position

    script:
    """
    ${biahub_cmd} nf run-apply-inv-tf \
        -i ${params.output_dir}/1-deskew/${dataset_name}.zarr \
        -o ${params.output_dir}/2-reconstruct/${dataset_name}.zarr \
        -t ${params.output_dir}/2-reconstruct/transfer_function_${dataset_name}.zarr \
        -p ${position} \
        -c ${params.reconstruct_config} \
        -j ${params.num_processes}
    """
}


// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------

workflow {
    if (!params.input_zarr)        error "Provide --input_zarr"
    if (!params.output_dir)        error "Provide --output_dir"
    if (!params.flat_field_config) error "Provide --flat_field_config"
    if (!params.deskew_config)     error "Provide --deskew_config"
    if (!params.reconstruct_config) error "Provide --reconstruct_config"

    // Collect positions as a value channel (single list, reusable)
    all_positions = list_positions()
        | splitText
        | map { it.trim() }
        | filter { it }
        | collect

    // Step 1: Flat-field correction
    ff_init = init_flat_field()

    ff_done = all_positions
        .flatMap { it }
        .combine(ff_init)
        .map { pos, ready -> pos }
        | run_flat_field
        | collect

    // Step 2: Deskew
    dk_init = init_deskew(ff_done.map { 'ff_done' })

    dk_done = all_positions
        .flatMap { it }
        .combine(dk_init)
        .map { pos, ready -> pos }
        | run_deskew
        | collect

    // Step 3: Reconstruct (compute TF + apply per position)
    rc_init = init_reconstruct(dk_done.map { 'dk_done' })

    all_positions
        .flatMap { it }
        .combine(rc_init)
        .map { pos, ready -> pos }
        | run_apply_inv_tf
}
