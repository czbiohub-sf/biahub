include { parse_resources; biahub_cmd } from './common'


process init_resources_stabilization {
    label 'cpu_small'

    input:
    tuple val(trigger), val(in_zarr), val(ram_mult)

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-resources \
        -i "${in_zarr}" \
        -r ${ram_mult}
    """
}

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
    ${biahub_cmd()} nf estimate-stabilization-z-focus \
        -i "${in_zarr}" \
        -p "${position}" \
        -c "${config}" \
        -o "${out_dir}"
    """
}

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
    ${biahub_cmd()} nf estimate-stabilization-xy \
        -i "${in_zarr}" \
        -p "${position}" \
        -c "${config}" \
        --focus-csv "${focus_csv}" \
        -o "${out_dir}"
    """
}

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
    ${biahub_cmd()} nf combine-transforms \
        -a "${out_dir}/z_transforms/${pos_file}.yml" \
        -b "${out_dir}/xy_transforms/${pos_file}.yml" \
        -o "${out_dir}/xyz_stabilization_settings/${pos_file}.yml"
    """
}

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
    ${biahub_cmd()} nf estimate-stabilization-pcc \
        -i "${in_zarr}" \
        -p "${position}" \
        -c "${config}" \
        -o "${out_dir}"
    """
}

process estimate_stabilization_beads {
    label 'cpu_medium'

    input:
    tuple val(trigger), val(in_zarr), val(config), val(out_dir), val(bead_pos)

    output:
    val true

    script:
    """
    ${biahub_cmd()} nf estimate-stabilization-beads \
        -i "${in_zarr}" \
        -p "${bead_pos}" \
        -c "${config}" \
        -o "${out_dir}"
    """
}

process init_stabilize {
    label 'cpu_small'

    input:
    tuple val(trigger), val(in_zarr), val(out_zarr), val(config)

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-stabilize \
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
    ${biahub_cmd()} nf run-stabilize \
        -i "${in_zarr}" \
        -o "${out_zarr}" \
        -p "${position}" \
        -c "${config}" \
        -j ${task.cpus}
    """
}


workflow stabilization_estimate_focus_z_wf {
    take:
    positions
    prev_done
    input_zarr
    config
    output_dir

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
    positions
    prev_done
    input_zarr
    config
    output_dir

    main:
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

    xy_done = positions
        .flatMap { it }
        .combine(z_resources)
        .combine(input_zarr)
        .combine(config)
        .combine(output_dir)
        .combine(merged_csv.map { it.toString() })
        | estimate_stabilization_xy
        | collect

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
    positions
    prev_done
    input_zarr
    config
    output_dir

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
    prev_done
    input_zarr
    config
    output_dir
    beads_position

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
    positions
    prev_done
    input_zarr
    output_zarr
    transform_config

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
