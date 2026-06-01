// Reconstruct subworkflow: init → compute TF → fan-out apply-inv-tf × N positions.
//
// Three-phase pattern:
// 1. init_apply_inv_tf: creates the output plate, resolves pixel sizes from
//    zarr metadata into reconstruct_resolved.yml, emits RESOURCES: and TF_RESOURCES:
// 2. compute_transfer_function: one-shot TF computation using TF_RESOURCES
// 3. run_apply_inv_tf: per-position inverse TF application using RESOURCES
//
// run_apply_inv_tf uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs.
// See: examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_apply_inv_tf {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('reconstruct')}"
    ${biahub_cmd()} apply-inv-tf --init \
        -i "${params.output_dir}/1-deskew/${dataset_name()}.zarr"/*/*/* \
        -o "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        -c "${params.reconstruct_config}"
    """
}

process compute_transfer_function {
    label 'cpu'
    clusterOptions { slurm_logs('reconstruct') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'

    input:
    val meta

    output:
    val true

    script:
    """
    ${biahub_cmd()} compute-tf \
        -i "${params.output_dir}/1-deskew/${dataset_name()}.zarr"/*/*/* \
        -o "${params.output_dir}/2-reconstruct/transfer_function_reconstruct_resolved.zarr" \
        -c "${params.output_dir}/2-reconstruct/reconstruct_resolved.yml"
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

    output:
    val position

    script:
    """
    ${biahub_cmd()} apply-inv-tf --cluster debug \
        -i "${params.output_dir}/1-deskew/${dataset_name()}.zarr/${position}" \
        -t "${params.output_dir}/2-reconstruct/transfer_function_reconstruct_resolved.zarr" \
        -o "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        -c "${params.output_dir}/2-reconstruct/reconstruct_resolved.yml"
    """
}


workflow reconstruct_wf {
    take:
    positions
    prev_done

    main:
    init_out = init_apply_inv_tf(prev_done.map { 'done' })
    tf_resources = init_out.map { parse_resources(it, 'TF_RESOURCES:') }
    run_resources = init_out.map { parse_resources(it) }

    tf_done = compute_transfer_function(tf_resources)

    rc_done = positions
        .flatMap { it }
        .combine(run_resources)
        .combine(tf_done)
        .map { pos, meta, tf -> [pos, meta] }
        | run_apply_inv_tf
        | collect

    emit:
    done = rc_done
}
