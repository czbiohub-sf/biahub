// Flat-field subworkflows: init (scaffold + resources) and run (fan-out × N).
//
// This module is PATH-AGNOSTIC. Callers pass the input zarr, output zarr, and
// config explicitly; the module has no idea where it sits in the pipeline
// directory layout. The orchestrating pipeline (see mantis-v2.nf) owns the
// layout and the order of steps; this module just applies flat-field correction
// to whatever it's handed.
//
// INIT AND RUN ARE SEPARATE SUBWORKFLOWS. The pipeline runs every step's init
// ahead of every step's compute, so that a bad config fails in the first
// minutes rather than hours in — which means it, not this module, decides when
// each phase happens. See the INIT PHASE comment in mantis-v2.nf.
//
// run_flat_field uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs — debug
// mode ensures the process_single_position call executes synchronously inside
// the Nextflow task.  See also:
// examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { parse_resources; slurm_logs; slurm_log_dir } from './common'


process init_flat_field {
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
    mkdir -p "${slurm_log_dir('flat_field')}"
    biahub flat-field --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
    """
}

process run_flat_field {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('flat_field') }
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
    biahub flat-field --cluster debug --resume \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// Validate the config and scaffold the output plate. Metadata-only and cheap,
// so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the input plate.zarr (raw input)
//   output_zarr  path to the flat-field corrected output plate.zarr
//   config       path to the flat-field settings YAML
//   trigger      gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once the output plate exists
workflow flat_field_init_wf {
    take:
    input_zarr
    output_zarr
    config
    trigger

    main:
    init_out = init_flat_field(input_zarr, output_zarr, config, trigger.map { 'done' })

    emit:
    // `.first()` turns the init's one-shot stdout into a VALUE channel, which is
    // the contract every step module here emits on: both outputs cross a
    // subworkflow boundary and are combined against a per-position queue
    // channel, and a value channel makes that a gate rather than a one-item
    // cross product whose behaviour depends on how many consumers read it.
    //
    // Nextflow logs one deduplicated "operator `first` is useless when applied
    // to a value channel" per run because THIS pipeline happens to trigger every
    // init from a value channel, which already makes `stdout` one. That is a
    // property of the caller, not of the module, so the guard stays.
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }.first()
    done      = init_out.map { 'done' }.first()
}


// Fan out one flat-field task per position.
//
// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr (raw input)
//   output_zarr  path to the flat-field corrected output plate.zarr
//   config       path to the flat-field settings YAML
//   resources    RESOURCES payload from flat_field_init_wf
//   prev_done    gating channel — compute starts once this emits
workflow flat_field_run_wf {
    take:
    positions
    input_zarr
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

    ff_done = run_flat_field(pos_meta, input_zarr, output_zarr, config) | collect

    emit:
    done = ff_done
}
