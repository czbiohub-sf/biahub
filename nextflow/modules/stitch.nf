// Stitch subworkflow: estimate → init → run.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass input zarr, output zarr,
// and stitch config path explicitly.
//
// estimate_stitch extracts stage positions and computes stitching parameters.
//
// init_stitch creates the per-well output stores and emits RESOURCES.
//
// run_stitch uses `--cluster debug` so that submitit's DebugExecutor runs
// all chunks in-process.  Nextflow handles resource scheduling; the CLI must
// NOT submit its own SLURM jobs.
// See: examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process estimate_stitch {
    label 'cpu_local'

    input:
    val input_zarr
    val stitch_config
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd()} estimate-stitch \
        -i "${input_zarr}"/*/*/* \
        -o "${stitch_config}"
    """
}

process init_stitch {
    label 'cpu_local'

    input:
    val input_zarr
    val output_zarr
    val stitch_config
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('stitch')}"
    ${biahub_cmd()} stitch --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${stitch_config}"
    """
}

process run_stitch {
    label 'cpu'
    clusterOptions { slurm_logs('stitch') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { task.attempt == 1 ? '4h' : '8h' }
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(trigger), val(meta)
    val input_zarr
    val output_zarr
    val stitch_config

    output:
    val true

    script:
    """
    ${biahub_cmd()} stitch --cluster debug \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${stitch_config}"
    """
}


// take:
//   input_zarr     path to the input plate.zarr
//   output_zarr    path to the stitched output plate.zarr
//   stitch_config  path to the stitch config YAML (written by estimate)
//   prev_done      gating channel
workflow stitch_wf {
    take:
    input_zarr
    output_zarr
    stitch_config
    prev_done

    main:
    init_out = init_stitch(input_zarr, output_zarr, stitch_config, prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    trigger_meta = resources.map { meta -> tuple('done', meta) }
    st_done = run_stitch(trigger_meta, input_zarr, output_zarr, stitch_config) | collect

    emit:
    done = st_done
}
