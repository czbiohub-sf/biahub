// Tracking subworkflow: init → parse_resources → fan-out run × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass the input zarr (plate
// structure), input images zarr (image data for tracking), output zarr,
// and config explicitly.
//
// Tracking is a 2-input step: reads reconstruct for plate structure and
// virtual-stain for image data.

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_track {
    label 'cpu_local'

    input:
    val input_zarr
    val output_zarr
    val config
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('track')}"
    ${biahub_cmd()} track --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
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
    val input_zarr
    val input_images_zarr
    val output_zarr
    val config

    output:
    val position

    script:
    """
    ${biahub_cmd()} track --cluster debug \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}" \
        --input-images-path "${input_images_zarr}"
    """
}


// take:
//   positions          collected channel of position keys
//   input_zarr         path to reconstruct output zarr (plate structure)
//   input_images_zarr  path to virtual-stain output zarr (image data for tracking)
//   output_zarr        path to tracking output zarr
//   config             path to track settings YAML
//   prev_done          gating channel
workflow track_wf {
    take:
    positions
    input_zarr
    input_images_zarr
    output_zarr
    config
    prev_done

    main:
    init_out = init_track(input_zarr, output_zarr, config, prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    pos_meta = positions
        .flatMap { it }
        .combine(resources)

    tk_done = run_track(pos_meta, input_zarr, input_images_zarr, output_zarr, config) | collect

    emit:
    done = tk_done
}
