// Deskew subworkflows: init (scaffold + resources) and run (fan-out × N).
//
// This module is PATH-AGNOSTIC. Callers pass the input zarr, output zarr, and
// config explicitly; the module has no idea where it sits in the pipeline
// directory layout. That's deliberate — deskew can read raw input, a 0-convert
// store, a flat-field corrected store, or anything else, and write anywhere.
// The orchestrating pipeline (see mantis-v2.nf) owns the layout and
// the order of steps; this module just deskews whatever it's handed.
//
// INIT AND RUN ARE SEPARATE SUBWORKFLOWS. The pipeline runs every step's init
// ahead of every step's compute, so that a bad config fails in the first
// minutes rather than hours in. `biahub deskew --init` reads only the input
// plate's METADATA — channel names and shape, with the deskewed shape derived
// analytically — so it runs happily against a store its upstream step has
// scaffolded but not yet filled. See the INIT PHASE comment in mantis-v2.nf.
//
// run_deskew uses `--cluster debug` so that submitit's DebugExecutor runs the
// work in-process.  Nextflow already handles per-position fan-out and resource
// scheduling, so the CLI must NOT submit its own SLURM jobs — debug mode
// ensures the process_single_position call executes synchronously inside the
// Nextflow task.  See also:
// examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { parse_resources; slurm_logs; slurm_log_dir } from './common'


process init_deskew {
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
    mkdir -p "${slurm_log_dir('deskew')}"
    biahub deskew --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
    """
}

process run_deskew {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('deskew') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { "${meta.time_minutes * task.attempt} min" }

    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val config

    output:
    val position

    script:
    // --resume: a preempted task finishes the write it is in and stops early, so
    // the retry (or a later `nextflow -resume`) recomputes only the (t, c) units
    // this position had not finished. The completion record is keyed by the
    // resolved settings, so a config change recomputes instead of being skipped.
    """
    biahub deskew --cluster debug --resume \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// Validate the config and scaffold the output plate. Metadata-only and cheap,
// so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the input plate.zarr to deskew (any starting point)
//   output_zarr  path to the deskewed output plate.zarr
//   config       path to the deskew settings YAML
//   trigger      gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once the output plate exists
workflow deskew_init_wf {
    take:
    input_zarr
    output_zarr
    config
    trigger

    main:
    init_out = init_deskew(input_zarr, output_zarr, config, trigger.map { 'done' })

    emit:
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }.first()
    done      = init_out.map { 'done' }.first()
}


// Fan out one deskew task per position.
//
// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr to deskew
//   output_zarr  path to the deskewed output plate.zarr
//   config       path to the deskew settings YAML
//   resources    RESOURCES payload from deskew_init_wf
//   prev_done    gating channel — compute starts once this emits
workflow deskew_run_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    resources
    prev_done

    main:
    pos_meta = positions
        .flatMap { items -> items }
        .combine(resources)
        .combine(prev_done)
        .map { pos, meta, _gate -> [pos, meta] }

    dk_done = run_deskew(pos_meta, input_zarr, output_zarr, config) | collect

    emit:
    done = dk_done
}
