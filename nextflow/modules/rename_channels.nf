include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_resources_rename {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('rename')}"
    ${biahub_cmd()} nf init-resources \
        -i "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        -r 2
    """
}

process rename_channels {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('rename') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '30m'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    def prefix_flag = params.rename_prefix ? "--prefix '${params.rename_prefix}'" : ""
    def suffix_flag = params.rename_suffix ? "--suffix '${params.rename_suffix}'" : ""
    // biahub rename-channels is a standalone command (metadata-only, no data copy).
    """
    ${biahub_cmd()} rename-channels \
        -i "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        -p "${position}" \
        ${prefix_flag} ${suffix_flag}
    """
}


workflow rename_wf {
    take:
    positions
    prev_done

    main:
    resources = init_resources_rename(prev_done.map { 'done' }).map { parse_resources(it) }
    rn_done = positions
        .flatMap { it }
        .combine(resources)
        | rename_channels
        | collect

    emit:
    done = rn_done
}
