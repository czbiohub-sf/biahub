include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_flat_field {
    label 'cpu_local'

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('flat_field')}"
    ${biahub_cmd()} nf init-flat-field \
        -i "${params.input_zarr}" \
        -o "${params.output_dir}/0-flatfield/${dataset_name()}.zarr" \
        -c "${params.flat_field_config}"
    """
}

process run_flat_field {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('flat_field') }
    maxForks 30
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '1h'
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
        -c "${params.flat_field_config}"
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
