// Assembly subworkflow: resolve concat config → init (RESOURCES) → single-shot
// concatenate.
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
// resources for the concurrent fan-out.
//
// Like the other steps, a cheap `--init` step on the login node creates the
// output plate (create_empty_plate is idempotent) and emits the RESOURCES line
// that sizes the compute node. Path injection: the source zarr paths are
// Nextflow runtime values, so `--resolve-config` templates them into
// concat_data_paths — also a login-node step — before init/run read the config.

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


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
    // sits with the rest of the run's configs. `rm -f` first because
    // `--resolve-config -o` refuses to overwrite an existing file, so a rerun
    // would otherwise fail on the stale copy.
    script:
    def resolved = "${config_dir}/concatenate_resolved.yml"
    """
    mkdir -p "${config_dir}"
    rm -f "${resolved}"
    ${biahub_cmd()} concatenate --resolve-config \
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
    ${biahub_cmd()} concatenate --init \
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
    """
    ${biahub_cmd()} concatenate --cluster debug \
        -c "${resolved_config_path}" \
        -o "${output_zarr}"
    """
}


// take:
//   deskew_zarr        LF source store to concatenate
//   reconstruct_zarr   phase source store to concatenate
//   virtual_stain_zarr virtual-stain source store to concatenate
//   output_zarr        path to the assembled output plate.zarr
//   config             path to the concatenate settings YAML (placeholder paths)
//   prev_done          gating channel — assembly starts once this emits
workflow assemble_wf {
    take:
    deskew_zarr
    reconstruct_zarr
    virtual_stain_zarr
    output_zarr
    config
    prev_done

    main:
    def config_dir = new File(config.toString()).parent
    def resolved_config_path = "${config_dir}/concatenate_resolved.yml"

    resolved = resolve_concatenate_config(
        deskew_zarr, reconstruct_zarr, virtual_stain_zarr,
        config_dir, config, prev_done.map { 'done' }
    )
    resources = init_concatenate(resolved, output_zarr).map { parse_resources(it) }
    as_done = run_concatenate(output_zarr, resolved_config_path, resources)

    emit:
    done = as_done
}
