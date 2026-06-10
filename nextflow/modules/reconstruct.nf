// Reconstruct subworkflow: init → compute TF → fan-out apply-inv-tf × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass the input zarr, output zarr,
// and config explicitly; the module has no idea where it sits in the pipeline
// directory layout. The TF zarr and resolved config paths are derived from the
// output_zarr's parent directory (the module's internal convention).
//
// Three-phase pattern:
// 1. init_apply_inv_tf: creates the output plate, resolves pixel sizes from
//    zarr metadata into reconstruct_resolved.yml, emits RESOURCES:
// 2. compute_transfer_function: one-shot TF computation using hardcoded resources
// 3. run_apply_inv_tf: per-position inverse TF application using RESOURCES
//
// run_apply_inv_tf uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs.
// See: examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_apply_inv_tf {
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
    mkdir -p "${slurm_log_dir('reconstruct')}"
    ${biahub_cmd()} apply-inv-tf --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
    """
}

process compute_transfer_function {
    label 'cpu'
    clusterOptions { slurm_logs('reconstruct') }
    // Hardcoded resources for the one-shot transfer-function computation.
    //
    // waveorder upsamples the volume to Nyquist internally before building the
    // TF (waveorder/models/phase_thick_3d.py::calculate_transfer_function); the
    // peak footprint scales with that upsampled volume, not the input volume.
    // For properly-sampled label-free data the Nyquist upsampling factor is 1
    // in every axis, so even for the largest volume we expect
    // (~2048 x 2048 x 128 = 2 GB float32) the phase TF needs ~64 GB (2 GB x
    // waveorder's x32 Fourier multiplier).  128 GB is a 2x margin over that and
    // also covers fluorescence (x32) and combined birefringence+phase (x36,
    // which additionally downsamples in XY).  The factors only blow past 1 for
    // badly-undersampled data, which is not reconstructable phase to begin with.
    cpus 1
    memory '128 GB'
    time '30m'

    input:
    val trigger
    val input_zarr
    val tf_zarr
    val resolved_config

    output:
    val true

    script:
    """
    ${biahub_cmd()} compute-tf \
        -i "${input_zarr}"/*/*/* \
        -o "${tf_zarr}" \
        -c "${resolved_config}"
    """
}

process run_apply_inv_tf {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('reconstruct') }
    maxForks 30
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '6h'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val tf_zarr
    val resolved_config

    output:
    val position

    script:
    """
    ${biahub_cmd()} apply-inv-tf --cluster debug \
        -i "${input_zarr}/${position}" \
        -t "${tf_zarr}" \
        -o "${output_zarr}" \
        -c "${resolved_config}"
    """
}


// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr (deskew output)
//   output_zarr  path to the reconstructed output plate.zarr
//   config       path to the reconstruct settings YAML
//   prev_done    gating channel — reconstruct starts once this emits
workflow reconstruct_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    prev_done

    main:
    def output_dir      = new File(output_zarr).parent
    def tf_zarr         = "${output_dir}/transfer_function_reconstruct_resolved.zarr"
    def resolved_config = "${output_dir}/reconstruct_resolved.yml"

    init_out = init_apply_inv_tf(input_zarr, output_zarr, config, prev_done.map { 'done' })
    run_resources = init_out.map { parse_resources(it) }
    // compute_transfer_function uses hardcoded resources (see process body),
    // but must still wait for init to write reconstruct_resolved.yml.
    init_done = init_out.map { 'done' }

    tf_done = compute_transfer_function(init_done, input_zarr, tf_zarr, resolved_config)

    pos_meta = positions
        .flatMap { it }
        .combine(run_resources)
        .combine(tf_done)
        .map { pos, meta, tf -> [pos, meta] }

    rc_done = run_apply_inv_tf(pos_meta, input_zarr, output_zarr, tf_zarr, resolved_config) | collect

    emit:
    done = rc_done
}
