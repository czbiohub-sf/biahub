#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.input_zarr         = null
params.output_dir         = null
params.flat_field_config  = null
params.deskew_config      = null
params.reconstruct_config = null
params.predict_config     = null
params.num_processes      = 1
params.biahub_project     = null
params.viscy_project      = null
params.work_dir           = null
params.max_positions      = 0

def dataset_name = params.input_zarr ?
    new File(params.input_zarr).name.replaceAll(/\.zarr$/, '') : null

def biahub_cmd = params.biahub_project ?
    "uv run --project ${params.biahub_project} biahub" : "biahub"

def viscy_cmd = params.viscy_project ?
    "uv run --project ${params.viscy_project} viscy" :
    "uv run --from 'viscy @ git+https://github.com/mehta-lab/VisCy@v0.3.4' viscy"


// ---------------------------------------------------------------------------
// Processes — shared
// ---------------------------------------------------------------------------

process estimate_resources {
    label 'cpu_small'

    input:
    val ram_multiplier

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf estimate-resources -i ${params.input_zarr} --ram-multiplier ${ram_multiplier}
    """
}


process list_positions {
    label 'cpu_small'

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf list-positions -i ${params.input_zarr}
    """
}


// ---------------------------------------------------------------------------
// Processes — flat-field
// ---------------------------------------------------------------------------

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
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb * meta.cpus} GB" }
    time '2h'
    queue 'cpu'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

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


// ---------------------------------------------------------------------------
// Processes — deskew
// ---------------------------------------------------------------------------

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


// ---------------------------------------------------------------------------
// Processes — reconstruct
// ---------------------------------------------------------------------------

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
// Processes — virtual stain
// ---------------------------------------------------------------------------

process init_virtual_stain {
    label 'cpu_small'

    input:
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd} nf init-virtual-stain \
        -i ${params.output_dir}/2-reconstruct/${dataset_name}.zarr \
        -o ${params.output_dir}/3-virtual-stain/${dataset_name}.zarr \
        -c ${params.predict_config}
    """
}


process run_virtual_stain_preprocess {
    label 'cpu_medium'

    input:
    val trigger

    output:
    val true

    script:
    """
    ${viscy_cmd} preprocess \
        --data_path ${params.output_dir}/2-reconstruct/${dataset_name}.zarr \
        --channel_names -1 \
        --num_workers ${task.cpus} \
        --block_size 32
    """
}


process run_virtual_stain {
    label 'gpu'
    tag "${position}"
    maxRetries 1
    errorStrategy 'retry'

    input:
    val position

    output:
    val position

    script:
    def temp_zarr = "${params.output_dir}/3-virtual-stain/temp/${position.replaceAll('/', '_')}.zarr"
    """
    ${viscy_cmd} predict \
        -c ${params.predict_config} \
        --data.init_args.data_path ${params.output_dir}/2-reconstruct/${dataset_name}.zarr/${position} \
        --trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter \
        --trainer.callbacks.output_store ${temp_zarr} \
        --trainer.default_root_dir ${params.output_dir}/3-virtual-stain/logs

    ${biahub_cmd} nf copy-virtual-stain \
        -t ${temp_zarr} \
        -o ${params.output_dir}/3-virtual-stain/${dataset_name}.zarr \
        -p ${position}
    """
}


// ---------------------------------------------------------------------------
// Subworkflows
// ---------------------------------------------------------------------------

workflow flat_field_wf {
    take:
    positions
    resources

    main:
    ff_init = init_flat_field()
    ff_done = positions
        .flatMap { it }
        .combine(ff_init)
        .map { pos, ready -> pos }
        .combine(resources)
        | run_flat_field
        | collect

    emit:
    done = ff_done
}


workflow deskew_wf {
    take:
    positions
    prev_done

    main:
    dk_init = init_deskew(prev_done.map { 'done' })
    dk_done = positions
        .flatMap { it }
        .combine(dk_init)
        .map { pos, ready -> pos }
        | run_deskew
        | collect

    emit:
    done = dk_done
}


workflow reconstruct_wf {
    take:
    positions
    prev_done

    main:
    rc_init = init_reconstruct(prev_done.map { 'done' })
    rc_done = positions
        .flatMap { it }
        .combine(rc_init)
        .map { pos, ready -> pos }
        | run_apply_inv_tf
        | collect

    emit:
    done = rc_done
}


workflow virtual_stain_wf {
    take:
    positions
    prev_done

    main:
    vs_init = init_virtual_stain(prev_done.map { 'done' })
    vs_preprocess = run_virtual_stain_preprocess(prev_done.map { 'done' })

    ready = vs_init.combine(vs_preprocess)

    vs_done = positions
        .flatMap { it }
        .combine(ready)
        .map { pos, init_done, preprocess_done -> pos }
        | run_virtual_stain
        | collect

    emit:
    done = vs_done
}


// ---------------------------------------------------------------------------
// Main workflow
// ---------------------------------------------------------------------------

workflow {
    if (!params.input_zarr)         error "Provide --input_zarr"
    if (!params.output_dir)         error "Provide --output_dir"
    if (!params.flat_field_config)  error "Provide --flat_field_config"
    if (!params.deskew_config)      error "Provide --deskew_config"
    if (!params.reconstruct_config) error "Provide --reconstruct_config"

    all_positions = list_positions()
        | splitText
        | map { it.trim() }
        | filter { it }
        | take( params.max_positions ?: -1 )
        | collect

    ff_resources = estimate_resources(Channel.value(5))
        .map { it.trim().split(' ') }
        .map { [cpus: it[0].toInteger(), mem_gb: it[1].toInteger()] }

    ff_done = flat_field_wf(all_positions, ff_resources)
    dk_done = deskew_wf(all_positions, ff_done.done)
    rc_done = reconstruct_wf(all_positions, dk_done.done)

    if (params.predict_config) {
        virtual_stain_wf(all_positions, rc_done.done)
    }
}
