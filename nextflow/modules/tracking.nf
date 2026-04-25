include { dataset_name; parse_resources; biahub_cmd } from './common'


process init_track {
    label 'cpu_small'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-track \
        -i "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        -o "${params.output_dir}/4-track/${dataset_name()}.zarr" \
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
    ${biahub_cmd()} nf run-track \
        -o "${params.output_dir}/4-track/${dataset_name()}.zarr" \
        -p "${position}" \
        -c "${params.track_config}" \
        --input-images-path "${params.output_dir}/3-virtual-stain/${dataset_name()}.zarr"
    """
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
