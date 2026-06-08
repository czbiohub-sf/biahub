// Deconvolution subworkflow: init → fan-out run × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass input zarr, PSF zarr,
// output zarr, and config explicitly.
//
// init_deconvolve creates the output plate, computes the transfer function
// from the PSF, and emits RESOURCES.  The TF computation is lightweight (FFT)
// so it is folded into init rather than being a separate process.
//
// run_deconvolve uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs.
// See: examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_deconvolve {
    label 'cpu_local'

    input:
    val input_zarr
    val psf_zarr
    val output_zarr
    val config
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('deconvolve')}"
    ${biahub_cmd()} deconvolve --init \
        -i "${input_zarr}"/*/*/* \
        -p "${psf_zarr}" \
        -o "${output_zarr}" \
        -c "${config}"
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
    val input_zarr
    val output_zarr
    val config

    output:
    val position

    script:
    """
    ${biahub_cmd()} deconvolve --cluster debug \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// take:
//   positions    collected channel of position keys
//   input_zarr   path to the input plate.zarr
//   psf_zarr     path to the PSF zarr (only needed for init)
//   output_zarr  path to the deconvolved output plate.zarr
//   config       path to the deconvolve settings YAML
//   prev_done    gating channel
workflow deconvolve_wf {
    take:
    positions
    input_zarr
    psf_zarr
    output_zarr
    config
    prev_done

    main:
    init_out = init_deconvolve(input_zarr, psf_zarr, output_zarr, config, prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    pos_meta = positions
        .flatMap { it }
        .combine(resources)

    dc_done = run_deconvolve(pos_meta, input_zarr, output_zarr, config) | collect

    emit:
    done = dc_done
}
