#!/usr/bin/env nextflow

// Asserts the shape of every gate expression the step modules use.
//
// WHY THIS EXISTS. `nextflow lint` and `nextflow run -preview` both pass on a
// gate that is wired wrong, because the failure is a RUNTIME arity mismatch:
// it fires only when the gate channel actually emits, which for `deskew_run_wf`
// is after flat-field has run all N positions. A full run found it the
// expensive way once (54 positions of flat-field, then
// `MissingMethodException`); this file finds it in ten seconds.
//
// THE TRAP. `combine` FLATTENS a list-valued item into the tuple, so what the
// producer happens to emit leaks into the tuple's arity. A step's `done` is a
// COLLECTED list of every position, so combining it raw turns [pos, meta] into
// [pos, meta, p1, p2, … pN] and no fixed-parameter closure can be spread across
// it. Mapping the gate to a token first reads nothing out of it, so no
// producer's payload shape can reach the closure.
//
// Run it directly; it exits non-zero if a gate regresses:
//     nextflow run nextflow/tests/gate_shapes.nf
//
// Keep the channel shapes below in step with the real ones:
//   positions       collect() of position keys              (common.nf)
//   resources       value map from parse_resources          (<step>_init_wf)
//   collected_done  `run_<step> | collect` — a LIST         (<step>_run_wf)
//   single_done     run_concatenate's single output path    (assemble_run_wf)
//   tf_done         compute_transfer_function's `val true`  (reconstruct_tf_wf)

nextflow.enable.dsl = 2

def check(label, got, want) {
    if (got != want) {
        error "${label}: expected ${want}, got ${got}"
    }
    println "ok  ${label}  ->  ${got}"
}

workflow {
    def keys = ['A/1/0', 'A/1/1', 'A/1/2']
    def meta = [cpus: 16, mem_gb: 224, time_minutes: 110]

    positions      = channel.of(keys).collect()
    resources      = channel.value(meta)
    collected_done = channel.fromList(keys).collect()      // a step's `done`: a LIST
    single_done    = channel.value('/path/out.zarr')  // run_concatenate's `done`
    tf_done        = channel.value(true)

    // flat_field_run_wf / deskew_run_wf / track_run_wf
    positions.flatMap { it }
        .combine(resources)
        .combine(collected_done.map { 'done' })
        .map { pos, m, _gate -> [pos, m] }
        .collect()
        .subscribe { check('per-position gate', it.size(), keys.size() * 2) }

    // reconstruct_run_wf — two gates
    positions.flatMap { it }
        .combine(resources)
        .combine(tf_done.map { 'ready' })
        .combine(collected_done.map { 'done' })
        .map { pos, m, _tf, _gate -> [pos, m] }
        .collect()
        .subscribe { check('reconstruct gate', it.size(), keys.size() * 2) }

    // assemble_run_wf — single-shot, gate only
    resources.combine(collected_done.map { 'done' })
        .map { m, _gate -> m }
        .subscribe { check('assemble gate', it, meta) }

    // A gate that is a single value, not a list, must work through the same
    // expression — assemble's `done` is one path, and track gates on it.
    positions.flatMap { it }
        .combine(resources)
        .combine(single_done.map { 'done' })
        .map { pos, m, _gate -> [pos, m] }
        .collect()
        .subscribe { check('scalar gate', it.size(), keys.size() * 2) }
}
