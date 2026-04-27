include { dataset_name; parse_resources; biahub_cmd } from './common'


process estimate_crop {
    label 'cpu_medium'

    input:
    val trigger

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
    label 'cpu_small'

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
    ${biahub_cmd()} nf run-concatenate \
        -c "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        -o "${params.output_dir}/5-assemble/${dataset_name()}.zarr" \
        -p "${position}" \
        -j ${task.cpus}
    """
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
