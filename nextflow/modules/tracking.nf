// Tracking subworkflows: init (scaffold + resources + weights cache) and run
// (fan-out × N positions).
//
// This module is PATH-AGNOSTIC. Callers pass the input zarr (plate structure),
// input images zarr (image data for tracking), output zarr, and config
// explicitly.
//
// Tracking is a 2-input step: `input_zarr` supplies the plate structure and
// `input_images_zarr` the pixels. mantis-v2.nf passes the assembled plate for
// both.
//
// INIT AND RUN ARE SEPARATE SUBWORKFLOWS, and hoisting init is worth more here
// than anywhere else: the tracking config is the LAST one a run would otherwise
// parse, so a typo in it used to surface after every reconstruction step and
// the assembly had finished. `biahub track --init` reads only the input plate's
// shape and scale (z-slice resolution is data-free), so it runs against the
// assembled plate as soon as `concatenate --init` has scaffolded it. It also
// warms the shared cellpose weights cache, which is strictly better done before
// any GPU worker exists to race for it.

include { parse_resources; slurm_logs; slurm_log_dir } from './common'


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
    biahub track --init \
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
    biahub track --cluster debug \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}" \
        --input-images-path "${input_images_zarr}"
    """
}


// Validate the config, scaffold the output plate, warm the cellpose weights.
// Metadata-only and cheap, so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the input plate.zarr (plate structure)
//   output_zarr  path to the tracking output plate.zarr
//   config       path to the track settings YAML
//   trigger      gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once the output plate exists
workflow track_init_wf {
    take:
    input_zarr
    output_zarr
    config
    trigger

    main:
    init_out = init_track(input_zarr, output_zarr, config, trigger.map { 'done' })

    emit:
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }.first()
    done      = init_out.map { 'done' }.first()
}


// Fan out one tracking task per position.
//
// take:
//   positions          collected channel of position keys
//   input_zarr         plate structure store
//   input_images_zarr  image data store
//   output_zarr        path to the tracking output plate.zarr
//   config             path to the track settings YAML
//   resources          RESOURCES payload from track_init_wf
//   prev_done          gating channel — the input stores hold data
workflow track_run_wf {
    take:
    positions
    input_zarr
    input_images_zarr
    output_zarr
    config
    resources
    prev_done

    main:
    // GATE CHANNELS ARE MAPPED TO A TOKEN BEFORE COMBINING, never combined raw.
    // `combine` FLATTENS a list-valued item into the tuple, so what the producer
    // happens to emit leaks into the tuple's arity: a step's `done` is a COLLECTED
    // list of every position, which turned [pos, meta] into
    // [pos, meta, p1, p2, … p54] and blew up a three-parameter closure with
    // `MissingMethodException` — after the previous step had already run. Mapping
    // reads nothing out of the gate, so no producer's payload shape can reach here.
    pos_meta = positions
        .flatMap { items -> items }
        .combine(resources)
        .combine(prev_done.map { 'done' })
        .map { pos, meta, _gate -> [pos, meta] }

    tk_done = run_track(pos_meta, input_zarr, input_images_zarr, output_zarr, config) | collect

    emit:
    done = tk_done
}
