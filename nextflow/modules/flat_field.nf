include { dataset_name; parse_resources; biahub_cmd } from './common'


process init_flat_field {
    label 'cpu_small'

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-flat-field \
        -i "${params.input_zarr}" \
        -o "${params.output_dir}/0-flatfield/${dataset_name()}.zarr" \
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
    ${biahub_cmd()} nf run-flat-field \
        -i "${params.input_zarr}" \
        -o "${params.output_dir}/0-flatfield/${dataset_name()}.zarr" \
        -p "${position}" \
        -c "${params.flat_field_config}" \
        -j ${task.cpus}
    """
}


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
