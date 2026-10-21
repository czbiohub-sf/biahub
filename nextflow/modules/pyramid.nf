// Pyramid subworkflows: init (validate + resources) and run (fan-out × N).
//
// This module is PATH-AGNOSTIC. Callers pass the plate and the number of levels
// explicitly; the module has no idea where the plate sits in the pipeline
// directory layout or which step wrote it. mantis-v2.nf includes it twice,
// under aliases, for the assembled and the projected plates.
//
// `biahub pyramid` writes IN PLACE: every position of the plate gets `levels`
// resolution levels (level 0 is the existing array), on its image and on every
// label image under `labels/`, so it must run after the LAST step that writes
// pixels or labels into that plate. Level 0 is never touched, so QC of the same
// plate can run concurrently with it. Rerunning recomputes the same levels,
// which is what a retried task wants.
//
// INIT AND RUN ARE SEPARATE SUBWORKFLOWS, like every step's (see the INIT PHASE
// comment in mantis-v2.nf). The init here scaffolds nothing: it checks the first
// position against the requested depth (a plate already carrying a pyramid of
// another depth is refused now rather than after tracking) and prints the
// RESOURCES line one position's task is sized from.
//
// run_pyramid uses `--cluster debug` so that submitit's DebugExecutor runs the
// work in-process; Nextflow owns the per-position fan-out. See modules/project.nf,
// the template this follows.

include { parse_resources; slurm_logs; slurm_log_dir; retry_time } from './common'


process init_pyramid {
    label 'cpu_local'

    input:
    val input_zarr
    val levels
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('pyramid')}"
    biahub pyramid --init \
        -i "${input_zarr}"/*/*/* \
        --levels ${levels}
    """
}

process run_pyramid {
    // Two plates share this process (under the aliased workflows), so the tag
    // names the store as well as the position.
    tag "${new File(input_zarr as String).name}/${position}"
    label 'cpu'
    clusterOptions { slurm_logs('pyramid') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { retry_time(meta.time_minutes, task) }

    input:
    tuple val(position), val(meta)
    val input_zarr
    val levels

    output:
    val position

    script:
    """
    biahub pyramid --cluster debug \
        -i "${input_zarr}/${position}" \
        --levels ${levels}
    """
}


// Validate the request against the plate and size one position's task.
// Metadata-only and cheap, so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the plate.zarr the pyramids are written into
//   levels       total number of resolution levels, counting level 0
//   trigger      gating channel — init starts once this emits (the plate exists)
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once the request has been validated
workflow pyramid_init_wf {
    take:
    input_zarr
    levels
    trigger

    main:
    init_out = init_pyramid(input_zarr, levels, trigger.collect().map { 'done' })

    emit:
    // Value channels by the contract every step module emits on; see
    // flat_field_init_wf for why the guarantee is made at the input.
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }
    done      = init_out.map { 'done' }
}


// Fan out one pyramid task per position.
//
// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the plate.zarr the pyramids are written into
//   levels       total number of resolution levels, counting level 0
//   resources    RESOURCES payload from pyramid_init_wf
//   prev_done    gating channel — the last step writing into the plate has finished
workflow pyramid_run_wf {
    take:
    positions
    input_zarr
    levels
    resources
    prev_done

    main:
    // Gate channels are mapped to a token before combining, never combined raw;
    // see flat_field_run_wf.
    pos_meta = positions
        .flatMap { items -> items }
        .combine(resources)
        .combine(prev_done.map { 'done' })
        .map { pos, meta, _gate -> [pos, meta] }

    py_done = run_pyramid(pos_meta, input_zarr, levels) | collect

    emit:
    done = py_done
}
