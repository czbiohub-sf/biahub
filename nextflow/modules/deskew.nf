include { dataset_name; parse_resources; biahub_cmd } from './common'


def parse_work_items(stdout_text) {
    return stdout_text.trim().readLines()
        .findAll { it.startsWith('WORK:') }
        .collect { it.replace('WORK:', '') }
}

process init_deskew {
    label 'cpu_small'

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
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta), val(work_json)

    output:
    val position

    script:
    """
    ${biahub_cmd()} nf run-deskew \
        -i "${params.output_dir}/0-flatfield/${dataset_name()}.zarr" \
        -o "${params.output_dir}/1-deskew/${dataset_name()}.zarr" \
        -p "${position}" \
        -c "${params.output_dir}/1-deskew/deskew_resolved.yml" \
        -w '${work_json}'
    """
}


workflow deskew_wf {
    take:
    positions
    prev_done

    main:
    init_out = init_deskew(prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }
    work_items = init_out.map { parse_work_items(it) }.flatMap { it }

    dk_done = positions
        .flatMap { it }
        .combine(work_items)
        .combine(resources)
        .map { pos, work, meta -> [pos, meta, work] }
        | run_deskew
        | collect

    emit:
    done = dk_done
}
