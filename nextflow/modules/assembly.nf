// Assembly subworkflow: resolve concat config → single-shot concatenate.
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
// Path injection: the source zarr paths are Nextflow runtime values, so
// `--resolve-config` templates them into concat_data_paths — a cheap login-node
// step — before the compute step reads the resolved config.

include { biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process resolve_concatenate_config {
    label 'cpu_local'

    input:
    val deskew_zarr
    val reconstruct_zarr
    val virtual_stain_zarr
    val output_dir
    val config
    val trigger

    output:
    path "concatenate_resolved.yml"

    script:
    def resolved = "${output_dir}/concatenate_resolved.yml"
    """
    mkdir -p "${output_dir}"
    ${biahub_cmd()} concatenate --resolve-config \
        -c "${config}" \
        -o "${resolved}" \
        --concat-data-paths "${deskew_zarr}/*/*/*" \
        --concat-data-paths "${reconstruct_zarr}/*/*/*" \
        --concat-data-paths "${virtual_stain_zarr}/*/*/*"
    cp "${resolved}" concatenate_resolved.yml
    """
}


// Single-shot concatenation of the whole plate on a reserved compute node.
// Resources are static (sized for the whole plate run, not one position) rather
// than driven by a `--init` RESOURCES payload — the single-shot model does not
// need the separate init/list-positions step. Tune cpus/memory/time to the data.
// NOTE: label 'cpu' routes to the 'preempted' partition; if the node is
// reclaimed mid-run the whole task restarts (the global errorStrategy retries
// it). Acceptable while the step is quick; route to a non-preempted partition
// if it grows long.
process run_concatenate {
    label 'cpu'
    clusterOptions { slurm_logs('assemble') }
    cpus 16
    memory '128 GB'
    time '4h'

    input:
    path resolved_config
    val output_zarr

    output:
    val output_zarr

    script:
    """
    mkdir -p "${slurm_log_dir('assemble')}"
    rm -rf "${output_zarr}"
    ${biahub_cmd()} concatenate --cluster debug \
        -c "${resolved_config}" \
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
    def output_dir = new File(output_zarr).parent

    resolved = resolve_concatenate_config(
        deskew_zarr, reconstruct_zarr, virtual_stain_zarr,
        output_dir, config, prev_done.map { 'done' }
    )
    as_done = run_concatenate(resolved, output_zarr)

    emit:
    done = as_done
}
