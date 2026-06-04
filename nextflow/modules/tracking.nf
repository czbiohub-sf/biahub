include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir; step_dir } from './common'


process init_track {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    // biahub track --init creates the output plate and emits RESOURCES.
    // --cluster debug is used on the Nextflow path so the CLI runs in-process
    // rather than submitting its own Slurm jobs — Nextflow handles the fan-out
    // and resource scheduling.
    """
    mkdir -p "${slurm_log_dir('track')}"
    ${biahub_cmd()} track --init \
        -i "${params.output_dir}/${step_dir('reconstruct')}/${dataset_name()}.zarr"/*/*/* \
        -o "${params.output_dir}/${step_dir('track')}/${dataset_name()}.zarr" \
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
    // --cluster debug: run in-process; Nextflow handles per-position fan-out.
    """
    ${biahub_cmd()} track --cluster debug \
        -i "${params.output_dir}/${step_dir('reconstruct')}/${dataset_name()}.zarr/${position}" \
        -o "${params.output_dir}/${step_dir('track')}/${dataset_name()}.zarr" \
        -c "${params.track_config}" \
        --input-images-path "${params.output_dir}/${step_dir('virtual_stain')}/${dataset_name()}.zarr"
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
