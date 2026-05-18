include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_track {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('track')}"
    ${biahub_cmd()} nf init-track \
        -i "${params.output_dir}/5-assemble/${dataset_name()}.zarr" \
        -o "${params.output_dir}/4-track/${dataset_name()}.zarr" \
        -c "${params.track_config}"
    """
}

process run_track {
    tag "${position}"
    label 'gpu'
    clusterOptions { "--gres=gpu:1 " + slurm_logs('track') }
    maxForks 30
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
