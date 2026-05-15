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


// ---------------------------------------------------------------------------
// Rename channels using a mapping config (runs on multiple zarrs)
// ---------------------------------------------------------------------------

process rename_channels_map_process {
    tag "${position} @ ${zarr_path}"
    label 'cpu_local'
    time '10m'

    input:
    tuple val(position), val(zarr_path), val(config_flag)

    output:
    val position

    script:
    """
    ${biahub_cmd()} nf rename-channels-map \
        -i "${zarr_path}" \
        -p "${position}" \
        ${config_flag}
    """
}

workflow rename_channels_map_wf {
    take:
    positions
    prev_done

    main:
    def config_flag = params.rename_config
        ? "-c ${params.rename_config}"
        : ""

    // Apply to deskew, reconstruct, and virtual-stain zarrs
    def zarr_paths = [
        "${params.output_dir}/1-deskew/${dataset_name()}.zarr",
        "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr",
        "${params.output_dir}/3-virtual-stain/${dataset_name()}.zarr",
    ]

    rn_done = prev_done.map { 'done' }
        | combine( positions.flatMap { it } )
        | map { trigger, pos -> pos }
        | flatMap { pos -> zarr_paths.collect { zarr -> [pos, zarr, config_flag] } }
        | rename_channels_map_process
        | collect

    emit:
    done = rn_done
}
