include { dataset_name; parse_resources; biahub_cmd } from './common'


process init_estimate_crop {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-estimate-crop \
        --concat-data-path "${params.output_dir}/1-deskew/${dataset_name()}.zarr/*/*/*"
    """
}

process estimate_crop {
    label 'cpu_preempted'
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '1h'

    input:
    val meta

    output:
    val true

    script:
    """
    ${biahub_cmd()} nf estimate-crop \
        -c "${params.concatenate_config}" \
        -o "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        --concat-data-paths "${params.output_dir}/1-deskew/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/3-virtual-stain/${dataset_name()}.zarr/*/*/*"
    """
}

process init_concatenate {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-concatenate \
        -c "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        -o "${params.output_dir}/5-assemble/${dataset_name()}.zarr"
    """
}

process run_concatenate {
    tag "${position}"
    label 'cpu_preempted'
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd()} nf run-concatenate \
        -c "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        -o "${params.output_dir}/5-assemble/${dataset_name()}.zarr" \
        -p "${position}"
    """
}


workflow assemble_wf {
    take:
    positions
    prev_done

    main:
    crop_resources = init_estimate_crop(prev_done.map { 'done' }).map { parse_resources(it) }
    crop_done = estimate_crop(crop_resources)
    resources = init_concatenate(crop_done).map { parse_resources(it) }
    as_done = positions
        .flatMap { it }
        .combine(resources)
        | run_concatenate
        | collect

    emit:
    done = as_done
}
