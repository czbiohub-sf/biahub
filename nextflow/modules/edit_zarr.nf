// Edit-zarr subworkflow: init → parse_resources → fan-out run × N positions.
//
// Like deskew (see deskew.nf), this subworkflow is PATH-AGNOSTIC: callers pass
// the input zarr, output path, and config explicitly. `edit-zarr` crops
// (T/ZYX), drops/renames channels, and/or divides one store into several; the
// orchestrating pipeline owns the layout and the order of steps.
//
// run_edit_zarr uses `--cluster debug` so submitit's DebugExecutor runs the
// work in-process. Nextflow already handles per-position fan-out and resource
// scheduling, so the CLI must NOT submit its own SLURM jobs.
//
// NOTE on divide: when the config sets `divide`, `output_zarr` is the PARENT
// directory that holds the per-group `<name>.zarr` stores (the CLI creates
// them). init fans out every division; each per-position worker writes only to
// the store(s) that contain its position.

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_edit_zarr {
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
    mkdir -p "${slurm_log_dir('edit_zarr')}"
    ${biahub_cmd()} edit-zarr --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
    """
}

process run_edit_zarr {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('edit_zarr') }
    maxForks 30
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { "${meta.time_minutes * task.attempt} min" }
    maxRetries 1
    errorStrategy 'retry'


    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val config

    output:
    val position

    script:
    """
    ${biahub_cmd()} edit-zarr --cluster debug \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr to edit
//   output_zarr  output plate.zarr (or, with divide, the parent dir)
//   config       path to the edit-zarr settings YAML
//   prev_done    gating channel — edit-zarr starts once this emits
workflow edit_zarr_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    prev_done

    main:
    init_out = init_edit_zarr(input_zarr, output_zarr, config, prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    pos_meta = positions
        .flatMap { it }
        .combine(resources)

    ez_done = run_edit_zarr(pos_meta, input_zarr, output_zarr, config) | collect

    emit:
    done = ez_done
}
