// Tracking subworkflows: init (label scaffold + resources) and run (fan-out × N
// positions).
//
// This module is PATH-AGNOSTIC. Callers pass the input zarr (plate structure and
// the label image to track), input images zarr (image data, for the legacy
// foreground+contour path), and config explicitly.
//
// Tracking reads the label image `biahub segment` wrote into `input_zarr`
// (`labels/<labels>`) and writes INTO the same positions — the tracked labels as
// `labels/<labels>_tracked`, `tracks.geff` and a tracks CSV — and nothing else
// is kept: each worker builds its Ultrack database in the node's $TMPDIR and
// deletes it after export (no -o), so tracking has no step directory. Its real
// logs are in nextflow/slurm_output/track/, like every step's; submitit's
// debug-mode placeholders land in the task's work directory. `input_images_zarr`
// only matters to a foreground+contour config; mantis-v2.nf passes the same
// plate for both.
//
// INIT AND RUN ARE SEPARATE SUBWORKFLOWS, and hoisting init is worth more here
// than anywhere else: the tracking config is the LAST one a run would otherwise
// parse, so a typo in it used to surface after every reconstruction step and
// the assembly had finished. `biahub track --init` reads only the input plate's
// shape and scale, checks that the label image it will track exists (segment's
// init creates it, so it runs after that one) and creates the empty tracked
// label image in each position.
//
// Tracking is CPU work: ultrack re-segments from the labels' contours and solves
// an ILP; the GPU segmentation is `biahub segment`'s. It cannot resume, so it
// runs on the non-preemptible `cpu` partition.

include { parse_resources; slurm_logs; slurm_log_dir } from './common'


process init_track {
    label 'cpu_local'

    input:
    val input_zarr
    val config
    path config_file  // staged only for the task hash: see common.nf, #397
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('track')}"
    biahub track --init \
        -i "${input_zarr}"/*/*/* \
        -c "${config}"
    """
}

process run_track {
    tag "${position}"
    label 'cpu_reserved'
    clusterOptions { slurm_logs('track') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)
    val input_zarr
    val input_images_zarr
    val config
    path config_file  // staged only for the task hash: see common.nf, #397

    output:
    val position

    script:
    """
    biahub track --cluster debug \
        -i "${input_zarr}/${position}" \
        -c "${config}" \
        --input-images-path "${input_images_zarr}"
    """
}


// Validate the config and create the empty tracked label images.
// Metadata-only and cheap, so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the input plate.zarr (plate structure and the label image
//                to track; receives the tracked labels)
//   config       path to the track settings YAML
//   trigger      gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once every position has its empty label image
workflow track_init_wf {
    take:
    input_zarr
    config
    trigger

    main:
    init_out = init_track(input_zarr, config, file(config), trigger.collect().map { 'done' })

    emit:
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }
    done      = init_out.map { 'done' }
}


// Fan out one tracking task per position.
//
// take:
//   positions          collected channel of position keys
//   input_zarr         plate structure store
//   input_images_zarr  image data store
//   config             path to the track settings YAML
//   resources          RESOURCES payload from track_init_wf
//   prev_done          gating channel — the input stores hold data
workflow track_run_wf {
    take:
    positions
    input_zarr
    input_images_zarr
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

    tk_done = run_track(pos_meta, input_zarr, input_images_zarr, config, file(config)) | collect

    emit:
    done = tk_done
}
