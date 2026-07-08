// Deskew subworkflow: init → parse_resources → fan-out run × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass the input zarr, output zarr,
// and config explicitly; the module has no idea where it sits in the pipeline
// directory layout. That's deliberate — deskew can read raw input, a 0-convert
// store, a flat-field corrected store, or anything else, and write anywhere.
// The orchestrating pipeline (see mantis-v2.nf) owns the layout and
// the order of steps; this module just deskews whatever it's handed.
//
// run_deskew uses `--cluster debug` so that submitit's DebugExecutor runs the
// work in-process.  Nextflow already handles per-position fan-out and resource
// scheduling, so the CLI must NOT submit its own SLURM jobs — debug mode
// ensures the process_single_position call executes synchronously inside the
// Nextflow task.  See also:
// examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { parse_resources; biahub_cmd; preemptible_logs; slurm_log_dir } from './common'


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
    ${biahub_cmd()} deskew --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
    """
}

process run_deskew {
    tag "${position}"
    label 'cpu'
    clusterOptions { preemptible_logs('deskew') }
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
    """
    ${biahub_cmd()} deskew --cluster debug \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr to deskew (any starting point)
//   output_zarr  path to the deskewed output plate.zarr
//   config       path to the deskew settings YAML
//   prev_done    gating channel — deskew starts once this emits
workflow deskew_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    prev_done

    main:
    init_out = init_deskew(input_zarr, output_zarr, config, prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    pos_meta = positions
        .flatMap { it }
        .combine(resources)

    dk_done = run_deskew(pos_meta, input_zarr, output_zarr, config) | collect

    emit:
    done = dk_done
}
