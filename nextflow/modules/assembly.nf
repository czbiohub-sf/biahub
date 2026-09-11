// Assembly subworkflows: init (resolve concat config → scaffold → RESOURCES)
// and run (single-shot concatenate).
//
// Unlike the per-position steps (deskew, reconstruct, …), concatenate combines
// N source stores channel-wise at each position, so there is no single `-i` to
// fan out over. Rather than Nextflow-managed per-position fan-out, this step
// runs the WHOLE plate in ONE task: `concatenate --cluster debug` iterates
// every position in-process (see biahub/concatenate.py). The task is a reserved
// SLURM compute node (label 'cpu'), NOT the login node, so the login node stays
// free. `--cluster debug` runs in-process and submits no SLURM jobs of its own,
// so there is no scheduler-in-scheduler nesting.
//
// This is the "reserve a compute node + --cluster debug" approach. To parallelise
// positions across the reserved node's cores later, switch the run step to
// `--cluster local` (submitit spawns one subprocess per position) and size the
// resources for the concurrent fan-out. Making this step work ACROSS positions
// is biahub#301, and is also what a per-position pipeline DAG (biahub#304)
// would need.
//
// BOTH init steps read only METADATA — `resolve_concatenate_config` templates
// paths into the config, and `concatenate --init` reads each source position's
// shape/dtype/channel names to resolve the output channel mapping and create
// the plate (create_empty_plate is idempotent). Neither reads a pixel, so both
// run against source stores that have been scaffolded but not yet filled, which
// is what lets them join the pipeline's up-front init phase. Resolving the
// channel mapping there is the point: a concatenate config naming a channel no
// source store has now fails in the first minutes rather than after every
// reconstruction step has completed.

include { parse_resources; slurm_logs; slurm_log_dir } from './common'


process resolve_concatenate_config {
    label 'cpu_local'

    input:
    val deskew_zarr
    val reconstruct_zarr
    val virtual_stain_zarr
    val config_dir
    val config
    val trigger

    output:
    path "concatenate_resolved.yml"

    // Write the resolved config alongside the source config (config_dir) so it
    // sits with the rest of the run's configs. `rm -f` first because resolve
    // mode's `-o` refuses to overwrite an existing file, so a rerun would
    // otherwise fail on the stale copy. NOTE: this is not hermetic — the file
    // lands next to the user's configs rather than in the work dir, so two runs
    // sharing a config directory overwrite each other's copy. Unchanged by the
    // move into the init phase, except that it now happens in the run's first
    // minutes rather than hours in.
    script:
    def resolved = "${config_dir}/concatenate_resolved.yml"
    """
    mkdir -p "${config_dir}"
    rm -f "${resolved}"
    biahub concatenate \
        -c "${config}" \
        -o "${resolved}" \
        --concat-data-paths "${deskew_zarr}/*/*/*" \
        --concat-data-paths "${reconstruct_zarr}/*/*/*" \
        --concat-data-paths "${virtual_stain_zarr}/*/*/*"
    cp "${resolved}" concatenate_resolved.yml
    """
}


// Create the output plate and emit the RESOURCES line used to size the compute
// node. Cheap and metadata-only, so it stays on the login node (cpu_local).
process init_concatenate {
    label 'cpu_local'

    input:
    path resolved_config
    val output_zarr

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('assemble')}"
    biahub concatenate --init \
        -c "${resolved_config}" \
        -o "${output_zarr}"
    """
}


// Single-shot concatenation of the whole plate on a reserved compute node.
// cpus/memory/time come from the RESOURCES payload emitted by init_concatenate
// (parsed via parse_resources), matching the other CLIs.
// NOTE: label 'cpu' routes to the 'preempted' partition; if the node is
// reclaimed mid-run the whole task restarts (the global errorStrategy retries
// it). Acceptable while the step is quick; route to a non-preempted partition
// if it grows long.
// The single-shot copy is memory-bandwidth-bound, so exclude the slow, small-
// memory cpu-c nodes (2017 Intel Xeon Gold 6126, 24 cores, 128 GB/node) — they
// ran this ~6x slower than the AMD EPYC nodes. All other cpu-* nodes are EPYC
// with >=750 GB, so a plain --exclude of cpu-c is enough.
process run_concatenate {
    label 'cpu'
    clusterOptions { "${slurm_logs('assemble')} --exclude=cpu-c-[1-4]" }
    cpus   { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time   { "${meta.time_minutes * task.attempt} min" }

    input:
    val output_zarr
    val resolved_config_path
    val meta

    output:
    val output_zarr

    script:
    // --resume matters more here than for the per-position steps: this is a
    // single job covering every position, so a preemption or walltime kill near
    // the end would otherwise discard hours of copying. The retry recomputes
    // only the (t, c) units that had not finished. The completion record is
    // keyed by the resolved settings, so a config change recomputes instead of
    // being skipped.
    """
    biahub concatenate --cluster debug --resume \
        -c "${resolved_config_path}" \
        -o "${output_zarr}"
    """
}


// The resolved config lives beside the source config. Both subworkflows need
// the path and the pipeline invokes them separately, so derive it in one place.
def resolved_config_path(config) {
    return "${new File(config.toString()).parent}/concatenate_resolved.yml"
}


// Resolve the source paths into the config and scaffold the assembled plate.
//
// take:
//   deskew_zarr        LF source store to concatenate
//   reconstruct_zarr   phase source store to concatenate
//   virtual_stain_zarr virtual-stain source store to concatenate
//   output_zarr        path to the assembled output plate.zarr
//   config             path to the concatenate settings YAML (placeholder paths)
//   trigger            gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing the single-shot task
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
    def config_dir = new File(config.toString()).parent

    resolved = resolve_concatenate_config(
        deskew_zarr, reconstruct_zarr, virtual_stain_zarr,
        config_dir, config, trigger.map { 'done' }
    )
    init_out = init_concatenate(resolved, output_zarr)

    emit:
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }.first()
    done      = init_out.map { 'done' }.first()
}


// Concatenate the whole plate in one task.
//
// take:
//   output_zarr  path to the assembled output plate.zarr
//   config       path to the concatenate settings YAML (locates the resolved copy)
//   resources    RESOURCES payload from assemble_init_wf
//   prev_done    gating channel — every source store holds data
workflow assemble_run_wf {
    take:
    output_zarr
    config
    resources
    prev_done

    main:
    ready = resources
        .combine(prev_done)
        .map { meta, _gate -> meta }

    as_done = run_concatenate(output_zarr, resolved_config_path(config), ready)

    emit:
    // `.first()` so this is a VALUE channel like every other step's `done`:
    // run_concatenate is single-shot, so its output is a one-item queue, and
    // the steps gated on it would each have to re-signal it otherwise.
    done = as_done.first()
}
