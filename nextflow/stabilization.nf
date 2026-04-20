#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

def biahub_cmd = params.biahub_project ?
    "uv run --project ${params.biahub_project} biahub" : "biahub"

def parse_resources(stdout_text, prefix = 'RESOURCES:') {
    def matching = stdout_text.trim().readLines().findAll { it.startsWith(prefix) }
    if (!matching) {
        error "Expected a '${prefix}' line in command output but none was found. The underlying CLI may have failed."
    }
    def parts = matching.last().replace(prefix, '').trim().split(/\s+/)
    return [cpus: parts[0].toInteger(), mem_gb: parts[1].toInteger()]
}


// ---------------------------------------------------------------------------
// Processes — resource estimation for stabilization fan-out commands
// ---------------------------------------------------------------------------

process init_resources_stabilization {
    label 'cpu_small'

    input:
    tuple val(trigger), val(in_zarr), val(ram_mult)

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf init-resources \
        -i "${in_zarr}" \
        -r ${ram_mult}
    """
}


// ---------------------------------------------------------------------------
// Processes — Z-focus estimation (per-position fan-out)
// ---------------------------------------------------------------------------

process estimate_stabilization_z_focus {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'cpu'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta), val(in_zarr), val(config), val(out_dir)

    output:
    val position

    script:
    """
    ${biahub_cmd} nf estimate-stabilization-z-focus \
        -i "${in_zarr}" \
        -p "${position}" \
        -c "${config}" \
        -o "${out_dir}"
    """
}


// ---------------------------------------------------------------------------
// Processes — XY estimation (per-position fan-out, needs merged focus CSV)
// ---------------------------------------------------------------------------

process estimate_stabilization_xy {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'cpu'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta), val(in_zarr), val(config), val(out_dir), val(focus_csv)

    output:
    val position

    script:
    """
    ${biahub_cmd} nf estimate-stabilization-xy \
        -i "${in_zarr}" \
        -p "${position}" \
        -c "${config}" \
        --focus-csv "${focus_csv}" \
        -o "${out_dir}"
    """
}


// ---------------------------------------------------------------------------
// Processes — combine-transforms (per-position, composes A[t] @ B[t])
// ---------------------------------------------------------------------------

process combine_transforms {
    tag "${position}"
    label 'cpu_small'

    input:
    tuple val(position), val(out_dir)

    output:
    val position

    script:
    def pos_file = position.replaceAll('/', '_')
    """
    ${biahub_cmd} nf combine-transforms \
        -a "${out_dir}/z_transforms/${pos_file}.yml" \
        -b "${out_dir}/xy_transforms/${pos_file}.yml" \
        -o "${out_dir}/xyz_stabilization_settings/${pos_file}.yml"
    """
}


// ---------------------------------------------------------------------------
// Processes — PCC estimation (per-position fan-out)
// ---------------------------------------------------------------------------

process estimate_stabilization_pcc {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'cpu'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta), val(in_zarr), val(config), val(out_dir)

    output:
    val position

    script:
    """
    ${biahub_cmd} nf estimate-stabilization-pcc \
        -i "${in_zarr}" \
        -p "${position}" \
        -c "${config}" \
        -o "${out_dir}"
    """
}


// ---------------------------------------------------------------------------
// Processes — beads estimation (one-shot on single reference FOV)
// ---------------------------------------------------------------------------

process estimate_stabilization_beads {
    label 'cpu_medium'

    input:
    tuple val(trigger), val(in_zarr), val(config), val(out_dir), val(bead_pos)

    output:
    val true

    script:
    """
    ${biahub_cmd} nf estimate-stabilization-beads \
        -i "${in_zarr}" \
        -p "${bead_pos}" \
        -c "${config}" \
        -o "${out_dir}"
    """
}


// ---------------------------------------------------------------------------
// Processes — apply stabilization (shared across all estimation modes)
// ---------------------------------------------------------------------------

process init_stabilize {
    label 'cpu_small'

    input:
    tuple val(trigger), val(in_zarr), val(out_zarr), val(config)

    output:
    stdout

    script:
    """
    ${biahub_cmd} nf init-stabilize \
        -i "${in_zarr}" \
        -o "${out_zarr}" \
        -c "${config}"
    """
}

process run_stabilize {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'cpu'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta), val(in_zarr), val(out_zarr), val(config)

    output:
    val position

    script:
    """
    ${biahub_cmd} nf run-stabilize \
        -i "${in_zarr}" \
        -o "${out_zarr}" \
        -p "${position}" \
        -c "${config}" \
        -j ${task.cpus}
    """
}


// ---------------------------------------------------------------------------
// Subworkflows
// ---------------------------------------------------------------------------

workflow stabilization_estimate_focus_z_wf {
    take:
    positions      // collected list of position strings
    prev_done      // barrier channel
    input_zarr     // Channel.value(path)
    config         // Channel.value(path)
    output_dir     // Channel.value(path)

    main:
    resources = init_resources_stabilization(
        prev_done.map { 'done' }.combine(input_zarr).combine(Channel.value(8))
    ).map { parse_resources(it) }

    z_done = positions
        .flatMap { it }
        .combine(resources)
        .combine(input_zarr)
        .combine(config)
        .combine(output_dir)
        | estimate_stabilization_z_focus
        | collect

    emit:
    done = z_done
}

workflow stabilization_estimate_focus_xyz_wf {
    take:
    positions      // collected list of position strings
    prev_done      // barrier channel
    input_zarr     // Channel.value(path)
    config         // Channel.value(path)
    output_dir     // Channel.value(path)

    main:
    // Stage 1: Z-focus estimation (fan-out, ram_multiplier=8)
    z_resources = init_resources_stabilization(
        prev_done.map { 'done' }.combine(input_zarr).combine(Channel.value(8))
    ).map { parse_resources(it) }

    z_done = positions
        .flatMap { it }
        .combine(z_resources)
        .combine(input_zarr)
        .combine(config)
        .combine(output_dir)
        | estimate_stabilization_z_focus
        | collect

    // Stage 2: merge per-position focus CSVs into a single file
    merged_csv = z_done
        .flatMap { it }
        .combine(output_dir)
        .map { pos, out_dir ->
            file("${out_dir}/z_focus_positions/${pos.replaceAll('/', '_')}.csv")
        }
        | collectFile(
            name: 'positions_focus.csv',
            keepHeader: true,
            skip: 1
        )

    // Stage 3: XY estimation (fan-out, reuse ram_multiplier=8)
    xy_done = positions
        .flatMap { it }
        .combine(z_resources)
        .combine(input_zarr)
        .combine(config)
        .combine(output_dir)
        .combine(merged_csv.map { it.toString() })
        | estimate_stabilization_xy
        | collect

    // Stage 4: compose z @ xy per position
    xyz_done = xy_done
        .flatMap { it }
        .combine(output_dir)
        | combine_transforms
        | collect

    emit:
    done = xyz_done
}

workflow stabilization_estimate_pcc_wf {
    take:
    positions      // collected list of position strings
    prev_done      // barrier channel
    input_zarr     // Channel.value(path)
    config         // Channel.value(path)
    output_dir     // Channel.value(path)

    main:
    resources = init_resources_stabilization(
        prev_done.map { 'done' }.combine(input_zarr).combine(Channel.value(16))
    ).map { parse_resources(it) }

    pcc_done = positions
        .flatMap { it }
        .combine(resources)
        .combine(input_zarr)
        .combine(config)
        .combine(output_dir)
        | estimate_stabilization_pcc
        | collect

    emit:
    done = pcc_done
}

workflow stabilization_estimate_beads_wf {
    take:
    prev_done       // barrier channel
    input_zarr      // Channel.value(path)
    config          // Channel.value(path)
    output_dir      // Channel.value(path)
    beads_position  // Channel.value(position string)

    main:
    beads_done = estimate_stabilization_beads(
        prev_done.map { 'done' }
            .combine(input_zarr)
            .combine(config)
            .combine(output_dir)
            .combine(beads_position)
    )

    emit:
    done = beads_done
}

workflow stabilize_wf {
    take:
    positions        // collected list of position strings
    prev_done        // barrier channel
    input_zarr       // Channel.value(path)
    output_zarr      // Channel.value(path)
    transform_config // Channel.value(path to StabilizationSettings YAML)

    main:
    resources = init_stabilize(
        prev_done.map { 'done' }
            .combine(input_zarr)
            .combine(output_zarr)
            .combine(transform_config)
    ).map { parse_resources(it) }

    st_done = positions
        .flatMap { it }
        .combine(resources)
        .combine(input_zarr)
        .combine(output_zarr)
        .combine(transform_config)
        | run_stabilize
        | collect

    emit:
    done = st_done
}
