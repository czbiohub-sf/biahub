// Flat-field subworkflow: init → parse_resources → fan-out run × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass the input zarr, output zarr,
// and config explicitly; the module has no idea where it sits in the pipeline
// directory layout. The orchestrating pipeline (see mantis-v2.nf) owns the
// layout and the order of steps; this module just applies flat-field correction
// to whatever it's handed.
//
// run_flat_field uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs — debug
// mode ensures the process_single_position call executes synchronously inside
// the Nextflow task.  See also:
// examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


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
    ${biahub_cmd()} flat-field --init \
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
    ${biahub_cmd()} flat-field --cluster debug \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr (raw input)
//   output_zarr  path to the flat-field corrected output plate.zarr
//   config       path to the flat-field settings YAML
//   prev_done    gating channel — flat-field starts once this emits
workflow flat_field_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    prev_done

    main:
    init_out = init_flat_field(input_zarr, output_zarr, config, prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    pos_meta = positions
        .flatMap { it }
        .combine(resources)

    ff_done = run_flat_field(pos_meta, input_zarr, output_zarr, config) | collect

    emit:
    done = ff_done
}
