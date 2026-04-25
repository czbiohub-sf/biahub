include { dataset_name; parse_resources; biahub_cmd } from './common'


process init_resources_rename {
    label 'cpu_small'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-resources \
        -i "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        -r 2
    """
}

process rename_channels {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '30m'
    queue 'cpu'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    def prefix_flag = params.rename_prefix ? "--prefix '${params.rename_prefix}'" : ""
    def suffix_flag = params.rename_suffix ? "--suffix '${params.rename_suffix}'" : ""
    """
    ${biahub_cmd()} nf rename-channels \
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
