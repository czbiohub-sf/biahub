// Assembly subworkflows: init (scaffold + resources) and run (fan-out × N).
//
// Concatenate combines N source stores channel-wise at each position. The CLI
// takes one `-i` per source store, in the order of the config's per-source
// entries, so init passes each store's `*/*/*` glob and a worker passes the
// same position of each store — the same `-i` idiom as deskew, just repeated.
// The concatenate config itself holds only parameters (which channels, crop,
// chunking); the source paths never enter it.
//
// INIT AND RUN ARE SEPARATE SUBWORKFLOWS. `concatenate --init` reads each
// source position's shape/dtype/channel names to resolve the output channel
// mapping and creates the plate, stamping the `biahub-concatenate` provenance
// record on every position ONCE. It reads no pixel, so it runs against source
// stores that have been scaffolded but not yet filled, which is what lets it
// join the pipeline's up-front init phase — a concatenate config naming a
// channel no source store has fails in the first minutes rather than after
// every reconstruction step has completed. See the INIT PHASE comment in
// mantis-v2.nf.
//
// run_concatenate uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process. Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs. A worker
// finds its output position already scaffolded and creates nothing, so the
// provenance record is written by init alone, not once per worker.
//
// This replaced the single-shot design (biahub#279), in which one task copied
// the whole plate: on the 2026_08_11 A549 SEC61B run that task took 4 h 19 min
// for 54 positions, half the pipeline's wall-clock, CPU-bound on compressing
// the sharded output (biahub#301).

include { parse_resources; slurm_logs; slurm_log_dir; retry_time } from './common'


// Create the output plate and emit the RESOURCES line sizing one position's
// task. Metadata-only, so it stays on the login node (cpu_local).
process init_concatenate {
    label 'cpu_local'

    input:
    val deskew_zarr
    val reconstruct_zarr
    val virtual_stain_zarr
    val output_zarr
    val config
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('assemble')}"
    biahub concatenate --init \
        -i "${deskew_zarr}"/*/*/* \
        -i "${reconstruct_zarr}"/*/*/* \
        -i "${virtual_stain_zarr}"/*/*/* \
        -c "${config}" \
        -o "${output_zarr}"
    """
}

process run_concatenate {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('assemble') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { retry_time(meta.time_minutes, task) }

    input:
    tuple val(position), val(meta)
    val deskew_zarr
    val reconstruct_zarr
    val virtual_stain_zarr
    val output_zarr
    val config

    output:
    val position

    script:
    // --resume: a preempted task finishes the write it is in and stops early, so
    // the retry (or a later `nextflow -resume`) recopies only the (t, c) units
    // this position had not finished. The completion record is keyed by the
    // resolved settings, so a config change recopies instead of being skipped.
    """
    biahub concatenate --cluster debug --resume \
        -i "${deskew_zarr}/${position}" \
        -i "${reconstruct_zarr}/${position}" \
        -i "${virtual_stain_zarr}/${position}" \
        -c "${config}" \
        -o "${output_zarr}"
    """
}


// Validate the config and scaffold the assembled plate.
//
// take:
//   deskew_zarr        LF source store to concatenate
//   reconstruct_zarr   phase source store to concatenate
//   virtual_stain_zarr virtual-stain source store to concatenate
//   output_zarr        path to the assembled output plate.zarr
//   config             path to the concatenate settings YAML (parameters only)
//   trigger            gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once the assembled plate exists
workflow assemble_init_wf {
    take:
    deskew_zarr
    reconstruct_zarr
    virtual_stain_zarr
    output_zarr
    config
    trigger

    main:
    init_out = init_concatenate(deskew_zarr, reconstruct_zarr, virtual_stain_zarr,
                                output_zarr, config, trigger.collect().map { 'done' })

    emit:
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }
    done      = init_out.map { 'done' }
}


// Fan out one concatenate task per position.
//
// take:
//   positions          collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   deskew_zarr        LF source store to concatenate
//   reconstruct_zarr   phase source store to concatenate
//   virtual_stain_zarr virtual-stain source store to concatenate
//   output_zarr        path to the assembled output plate.zarr
//   config             path to the concatenate settings YAML
//   resources          RESOURCES payload from assemble_init_wf
//   prev_done          gating channel — every source store holds data
workflow assemble_run_wf {
    take:
    positions
    deskew_zarr
    reconstruct_zarr
    virtual_stain_zarr
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

    as_done = run_concatenate(pos_meta, deskew_zarr, reconstruct_zarr, virtual_stain_zarr,
                              output_zarr, config) | collect

    emit:
    done = as_done
}
