// Deconvolution subworkflow: init → fan-out run × N positions.
//
// init_deconvolve creates the output plate, computes the transfer function
// from the PSF, and emits RESOURCES.  The TF computation is lightweight (FFT)
// so it is folded into init rather than being a separate process.
//
// run_deconvolve uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs.
// See: examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_deconvolve {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('deconvolve')}"
    ${biahub_cmd()} deconvolve --init \
        -i "${params.deconvolve_input_zarr}"/*/*/* \
        -p "${params.psf_zarr}" \
        -o "${params.deconvolve_output_zarr}" \
        -c "${params.deconvolve_config}"
    """
}

process run_deconvolve {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('deconvolve') }
    maxForks 30
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { task.attempt == 1 ? '2h' : '4h' }
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd()} deconvolve --cluster debug \
        -i "${params.deconvolve_input_zarr}/${position}" \
        -o "${params.deconvolve_output_zarr}" \
        -c "${params.deconvolve_config}"
    """
}


workflow deconvolve_wf {
    take:
    positions
    prev_done

    main:
    init_out = init_deconvolve(prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    dc_done = positions
        .flatMap { it }
        .combine(resources)
        | run_deconvolve
        | collect

    emit:
    done = dc_done
}
