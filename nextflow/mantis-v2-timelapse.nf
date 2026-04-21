#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.input_zarr          = null
params.output_dir          = null
params.flat_field_config   = null
params.deskew_config       = null
params.reconstruct_config  = null
params.predict_config      = null
params.track_config        = null
params.concatenate_config  = null
params.rename_prefix       = ""
params.rename_suffix       = ""
params.num_threads         = 1
params.biahub_project      = null
params.viscy_project       = null
params.work_dir            = null
params.max_positions       = 0

def dataset_name = params.input_zarr ?
    new File(params.input_zarr).name.replaceAll(/\.zarr$/, '') : null

def biahub_cmd = params.biahub_project ?
    "uv run --project ${params.biahub_project} biahub" : "biahub"

def viscy_cmd = params.viscy_project ?
    "uv run --project ${params.viscy_project} viscy" :
    "uv run --from 'viscy @ git+https://github.com/mehta-lab/VisCy@v0.3.4' viscy"

def parse_resources(stdout_text, prefix = 'RESOURCES:') {
    def matching = stdout_text.trim().readLines().findAll { it.startsWith(prefix) }
    if (!matching) {
        error "Expected a '${prefix}' line in command output but none was found. The underlying CLI may have failed."
    }
    def parts = matching.last().replace(prefix, '').trim().split(/\s+/)
    return [cpus: parts[0].toInteger(), mem_gb: parts[1].toInteger()]
}


// ---------------------------------------------------------------------------
// Processes — shared
// ---------------------------------------------------------------------------

process list_positions {
    label 'cpu_small'

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf list-positions -i "${params.input_zarr}"
    """
}


// ---------------------------------------------------------------------------
// Processes — flat-field
// ---------------------------------------------------------------------------

process init_flat_field {
    label 'cpu_small'

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf init-flat-field \
        -i "${params.input_zarr}" \
        -o "${params.output_dir}/0-flatfield/${dataset_name}.zarr" \
        -c "${params.flat_field_config}"
    """
}

process run_flat_field {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
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
        -i "${params.input_zarr}" \
        -o "${params.output_dir}/0-flatfield/${dataset_name}.zarr" \
        -p "${position}" \
        -c "${params.flat_field_config}" \
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
    stdout

    script:
    """
    ${biahub_cmd} nf init-deskew \
        -i "${params.output_dir}/0-flatfield/${dataset_name}.zarr" \
        -o "${params.output_dir}/1-deskew/${dataset_name}.zarr" \
        -c "${params.deskew_config}"
    """
}

process run_deskew {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'gpu'
    clusterOptions '--gres=gpu:1'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd} nf run-deskew \
        -i "${params.output_dir}/0-flatfield/${dataset_name}.zarr" \
        -o "${params.output_dir}/1-deskew/${dataset_name}.zarr" \
        -p "${position}" \
        -c "${params.output_dir}/1-deskew/deskew_resolved.yml"
    """
}


// ---------------------------------------------------------------------------
// Processes — reconstruct
// ---------------------------------------------------------------------------

process init_reconstruct {
    label 'cpu_small'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf init-reconstruct \
        -i "${params.output_dir}/1-deskew/${dataset_name}.zarr" \
        -o "${params.output_dir}/2-reconstruct/${dataset_name}.zarr" \
        -c "${params.reconstruct_config}" \
        -j ${params.num_threads}
    """
}

process compute_transfer_function {
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'cpu'

    input:
    val meta

    output:
    val true

    script:
    """
    ${biahub_cmd} nf compute-transfer-function \
        -i "${params.output_dir}/1-deskew/${dataset_name}.zarr" \
        -t "${params.output_dir}/2-reconstruct/transfer_function_${dataset_name}.zarr" \
        -c "${params.output_dir}/2-reconstruct/reconstruct_resolved.yml"
    """
}

process run_apply_inv_tf {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '6h'
    queue 'cpu'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd} nf run-apply-inv-tf \
        -i "${params.output_dir}/1-deskew/${dataset_name}.zarr" \
        -o "${params.output_dir}/2-reconstruct/${dataset_name}.zarr" \
        -t "${params.output_dir}/2-reconstruct/transfer_function_${dataset_name}.zarr" \
        -p "${position}" \
        -c "${params.output_dir}/2-reconstruct/reconstruct_resolved.yml" \
        -j ${params.num_threads}
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
    stdout

    script:
    """
    ${biahub_cmd} nf init-virtual-stain \
        -i "${params.output_dir}/2-reconstruct/${dataset_name}.zarr" \
        -o "${params.output_dir}/3-virtual-stain/${dataset_name}.zarr" \
        -c "${params.predict_config}"
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
        --data_path "${params.output_dir}/2-reconstruct/${dataset_name}.zarr" \
        --channel_names -1 \
        --num_workers ${task.cpus} \
        --block_size 32
    """
}

process run_virtual_stain {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'gpu'
    clusterOptions '--gres=gpu:1'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    def temp_zarr = "${params.output_dir}/3-virtual-stain/temp/${position.replaceAll('/', '_')}.zarr"
    """
    ${viscy_cmd} predict \
        -c "${params.predict_config}" \
        --data.init_args.data_path "${params.output_dir}/2-reconstruct/${dataset_name}.zarr/${position}" \
        --trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter \
        --trainer.callbacks.output_store "${temp_zarr}" \
        --trainer.default_root_dir "${params.output_dir}/3-virtual-stain/logs"

    ${biahub_cmd} nf copy-virtual-stain \
        -t "${temp_zarr}" \
        -o "${params.output_dir}/3-virtual-stain/${dataset_name}.zarr" \
        -p "${position}"
    """
}


// ---------------------------------------------------------------------------
// Processes — rename channels (on reconstructed zarr, pre-assembly)
// ---------------------------------------------------------------------------

process init_resources_rename {
    label 'cpu_small'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf init-resources \
        -i "${params.output_dir}/2-reconstruct/${dataset_name}.zarr" \
        -r 2
    """
}

process rename_channels {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '30m'
    queue 'cpu'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    def prefix_flag = params.rename_prefix ? "--prefix '${params.rename_prefix}'" : ""
    def suffix_flag = params.rename_suffix ? "--suffix '${params.rename_suffix}'" : ""
    """
    ${biahub_cmd} nf rename-channels \
        -i "${params.output_dir}/2-reconstruct/${dataset_name}.zarr" \
        -p "${position}" \
        ${prefix_flag} ${suffix_flag}
    """
}


// ---------------------------------------------------------------------------
// Processes — tracking
// ---------------------------------------------------------------------------

process init_track {
    label 'cpu_small'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf init-track \
        -i "${params.output_dir}/2-reconstruct/${dataset_name}.zarr" \
        -o "${params.output_dir}/4-track/${dataset_name}.zarr" \
        -c "${params.track_config}"
    """
}

process run_track {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'gpu'
    clusterOptions '--gres=gpu:1'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd} nf run-track \
        -o "${params.output_dir}/4-track/${dataset_name}.zarr" \
        -p "${position}" \
        -c "${params.track_config}" \
        --input-images-path "${params.output_dir}/3-virtual-stain/${dataset_name}.zarr"
    """
}


// ---------------------------------------------------------------------------
// Processes — assembly (estimate-crop / concatenate)
// ---------------------------------------------------------------------------

process estimate_crop {
    label 'cpu_medium'

    input:
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd} nf estimate-crop \
        -c "${params.concatenate_config}" \
        -o "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        --concat-data-paths "${params.output_dir}/1-deskew/${dataset_name}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/2-reconstruct/${dataset_name}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/3-virtual-stain/${dataset_name}.zarr/*/*/*"
    """
}

process init_concatenate {
    label 'cpu_small'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf init-concatenate \
        -c "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        -o "${params.output_dir}/5-assemble/${dataset_name}.zarr"
    """
}

process run_concatenate {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
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
    ${biahub_cmd} nf run-concatenate \
        -c "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        -o "${params.output_dir}/5-assemble/${dataset_name}.zarr" \
        -p "${position}" \
        -j ${task.cpus}
    """
}


// ---------------------------------------------------------------------------
// Subworkflows
// ---------------------------------------------------------------------------

workflow flat_field_wf {
    take:
    positions

    main:
    resources = init_flat_field().map { parse_resources(it) }
    ff_done = positions
        .flatMap { it }
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
    resources = init_deskew(prev_done.map { 'done' }).map { parse_resources(it) }
    dk_done = positions
        .flatMap { it }
        .combine(resources)
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
    init_out = init_reconstruct(prev_done.map { 'done' })
    tf_resources = init_out.map { parse_resources(it, 'TF_RESOURCES:') }
    run_resources = init_out.map { parse_resources(it) }

    tf_done = compute_transfer_function(tf_resources)

    rc_done = positions
        .flatMap { it }
        .combine(run_resources)
        .combine(tf_done)
        .map { pos, meta, tf -> [pos, meta] }
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
    resources = init_virtual_stain(prev_done.map { 'done' }).map { parse_resources(it) }
    vs_preprocess = run_virtual_stain_preprocess(prev_done.map { 'done' })

    ready = resources.combine(vs_preprocess)

    vs_done = positions
        .flatMap { it }
        .combine(ready)
        .map { pos, meta, preprocess_done -> [pos, meta] }
        | run_virtual_stain
        | collect

    emit:
    done = vs_done
}

workflow rename_wf {
    take:
    positions
    prev_done

    main:
    resources = init_resources_rename(prev_done.map { 'done' }).map { parse_resources(it) }
    rn_done = positions
        .flatMap { it }
        .combine(resources)
        | rename_channels
        | collect

    emit:
    done = rn_done
}

workflow track_wf {
    take:
    positions
    prev_done

    main:
    resources = init_track(prev_done.map { 'done' }).map { parse_resources(it) }
    tk_done = positions
        .flatMap { it }
        .combine(resources)
        | run_track
        | collect

    emit:
    done = tk_done
}

workflow assemble_wf {
    take:
    positions
    prev_done

    main:
    crop_done = estimate_crop(prev_done.map { 'done' })
    resources = init_concatenate(crop_done).map { parse_resources(it) }
    as_done = positions
        .flatMap { it }
        .combine(resources)
        | run_concatenate
        | collect

    emit:
    done = as_done
}


// ---------------------------------------------------------------------------
// Main workflow
// ---------------------------------------------------------------------------

workflow {
    if (!params.input_zarr)          error "Provide --input_zarr"
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.flat_field_config)   error "Provide --flat_field_config"
    if (!params.deskew_config)       error "Provide --deskew_config"
    if (!params.reconstruct_config)  error "Provide --reconstruct_config"
    if (!params.predict_config)      error "Provide --predict_config"
    if (!params.track_config)        error "Provide --track_config"
    if (!params.concatenate_config)  error "Provide --concatenate_config"
    if (!params.rename_prefix && !params.rename_suffix) {
        error "Provide --rename_prefix and/or --rename_suffix"
    }

    positions_ch = list_positions()
        | splitText
        | map { it.trim() }
        | filter { it }

    all_positions = params.max_positions > 0
        ? positions_ch | take(params.max_positions) | collect
        : positions_ch | collect

    // Phase 1: flat-field → deskew → reconstruct (linear)
    ff_done  = flat_field_wf(all_positions)
    dk_done  = deskew_wf(all_positions, ff_done.done)
    rc_done  = reconstruct_wf(all_positions, dk_done.done)

    // Phase 2: virtual stain + tracking (parallel, both read from reconstruct)
    vs_done  = virtual_stain_wf(all_positions, rc_done.done)
    tk_done  = track_wf(all_positions, rc_done.done)

    // Phase 2b: rename reconstruct channels (after VS + tracking finish reading)
    rename_trigger = vs_done.done.mix(tk_done.done) | collect
    rn_done  = rename_wf(all_positions, rename_trigger)

    // Phase 3: assembly (waits for rename)
    assemble_wf(all_positions, rn_done.done)
}
