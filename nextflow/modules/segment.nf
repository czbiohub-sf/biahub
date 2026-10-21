// Segmentation subworkflows: init (label scaffold + resources + weights cache) and
// run (fan-out × N positions), the same split as tracking.
//
// This module is PATH-AGNOSTIC. Callers pass the input zarr and config
// explicitly. For 2D models the input is the PROJECTED (Z=1) plate that
// `biahub project` writes from the assembled plate; segment does not project.
//
// Segment writes INTO the plate it reads: one OME-NGFF label image per model,
// `labels/<model>`, in every position, which is what `biahub track` reads
// (track.yml `labels: <model>`). There is no step directory and no `-o`; the
// SLURM logs are in nextflow/slurm_output/segment/, like every step's, and
// submitit's debug-mode placeholders land in the task's work directory.
//
// `biahub segment --init` reads only the input plate's channels, shape and scale,
// so it runs as soon as the input plate is scaffolded: a config naming a channel
// the plate lacks, 2D models on a 3D plate (or 3D models on a Z=1 plate), or an
// unknown cellpose model fails before any GPU job. It creates the empty label
// images (so `track --init`, which needs them to exist, can follow it in the
// init phase) and warms the shared cellpose weights cache, better done once here
// than by N GPU workers racing for it.

include { parse_resources; slurm_logs; slurm_log_dir; retry_time } from './common'


process init_segment {
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
    mkdir -p "${slurm_log_dir('segment')}"
    biahub segment --init \
        -i "${input_zarr}"/*/*/* \
        -c "${config}"
    """
}

process run_segment {
    tag "${position}"
    // Preemptable, like virtual staining: `--resume` (below) makes a reclaimed task
    // cost at most the timepoint it was segmenting, so it runs on the `preempted`
    // partition (any GPU node) rather than `gpu`. It inherits nextflow.config's
    // errorStrategy, which retries preemptions; no SLURM --requeue (see
    // nextflow.config).
    label 'gpu_preempted'
    clusterOptions { "--gres=gpu:1 " + slurm_logs('segment') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { retry_time(meta.time_minutes, task) }

    input:
    tuple val(position), val(meta)
    val input_zarr
    val config
    path config_file  // staged only for the task hash: see common.nf, #397

    output:
    val position

    script:
    // --resume: a retried task (preemption, walltime) skips the timepoints the
    // previous attempt already wrote, as deskew does.
    """
    biahub segment --cluster debug --resume \
        -i "${input_zarr}/${position}" \
        -c "${config}"
    """
}


// Validate the config, create the empty label images, warm the cellpose weights.
// Metadata-only and cheap, so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the input plate.zarr (receives the label images)
//   config       path to the segment settings YAML
//   trigger      gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once every position has its empty label images
workflow segment_init_wf {
    take:
    input_zarr
    config
    trigger

    main:
    init_out = init_segment(input_zarr, config, file(config), trigger.collect().map { 'done' })

    emit:
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }
    done      = init_out.map { 'done' }
}


// Fan out one segmentation task per position.
//
// take:
//   positions    collected channel of position keys
//   input_zarr   input plate store (receives the label images)
//   config       path to the segment settings YAML
//   resources    RESOURCES payload from segment_init_wf
//   prev_done    gating channel — the input store holds data
workflow segment_run_wf {
    take:
    positions
    input_zarr
    config
    resources
    prev_done

    main:
    // Gate channels are mapped to a token before combining (see tracking.nf):
    // `combine` flattens list-valued items into the tuple.
    pos_meta = positions
        .flatMap { items -> items }
        .combine(resources)
        .combine(prev_done.map { 'done' })
        .map { pos, meta, _gate -> [pos, meta] }

    sg_done = run_segment(pos_meta, input_zarr, config, file(config)) | collect

    emit:
    done = sg_done
}
