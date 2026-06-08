// Rename channels subworkflow: init_resources → fan-out rename × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass the target zarr to rename
// channels on. Rename is metadata-only — operates on a single zarr in-place.

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_resources_rename {
    label 'cpu_local'

    input:
    val target_zarr
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('rename')}"
    ${biahub_cmd()} nf init-resources \
        -i "${target_zarr}" \
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
    val target_zarr

    output:
    val position

    script:
    def prefix_flag = params.rename_prefix ? "--prefix '${params.rename_prefix}'" : ""
    def suffix_flag = params.rename_suffix ? "--suffix '${params.rename_suffix}'" : ""
    """
    ${biahub_cmd()} rename-channels \
        -i "${target_zarr}" \
        -p "${position}" \
        ${prefix_flag} ${suffix_flag}
    """
}


// take:
//   positions    collected channel of position keys
//   target_zarr  zarr to rename channels on (e.g. reconstruct output)
//   prev_done    gating channel
workflow rename_wf {
    take:
    positions
    target_zarr
    prev_done

    main:
    init_out = init_resources_rename(target_zarr, prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    pos_meta = positions
        .flatMap { it }
        .combine(resources)

    rn_done = rename_channels(pos_meta, target_zarr) | collect

    emit:
    done = rn_done
}
