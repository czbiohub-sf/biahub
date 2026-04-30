include { dataset_name; parse_resources; biahub_cmd } from './common'


process init_deskew {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-deskew \
        -i "${params.output_dir}/0-flatfield/${dataset_name()}.zarr" \
        -o "${params.output_dir}/1-deskew/${dataset_name()}.zarr" \
        -c "${params.deskew_config}"
    """
}

process run_deskew {
    tag "${position}"
    label 'cpu'
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { task.attempt == 1 ? '1h' : '2h' }
    maxRetries 1
    errorStrategy 'retry'
    beforeScript { task.attempt > 1 ? "${biahub_cmd()} nf clean-position -o '${params.output_dir}/1-deskew/${dataset_name()}.zarr' -p '${position}'" : '' }

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd()} nf run-deskew \
        -i "${params.output_dir}/0-flatfield/${dataset_name()}.zarr" \
        -o "${params.output_dir}/1-deskew/${dataset_name()}.zarr" \
        -p "${position}" \
        -c "${params.output_dir}/1-deskew/deskew_resolved.yml"
    """
}


workflow deskew_wf {
    take:
    positions
    prev_done

    main:
    init_out = init_deskew(prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    dk_done = positions
        .flatMap { it }
        .combine(resources)
        | run_deskew
        | collect

    emit:
    done = dk_done
}
