// Stitch subworkflow: estimate → init → run.
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
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd()} estimate-stitch \
        -i "${params.stitch_input_zarr}"/*/*/* \
        -o "${params.stitch_config}"
    """
}

process init_stitch {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('stitch')}"
    ${biahub_cmd()} stitch --init \
        -i "${params.stitch_input_zarr}"/*/*/* \
        -o "${params.stitch_output_zarr}" \
        -c "${params.stitch_config}"
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

    output:
    val true

    script:
    """
    ${biahub_cmd()} stitch --cluster debug \
        -i "${params.stitch_input_zarr}"/*/*/* \
        -o "${params.stitch_output_zarr}" \
        -c "${params.stitch_config}"
    """
}


workflow stitch_wf {
    take:
    prev_done

    main:
    init_out = init_stitch(prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    st_done = resources
        .map { meta -> tuple('done', meta) }
        | run_stitch
        | collect

    emit:
    done = st_done
}
