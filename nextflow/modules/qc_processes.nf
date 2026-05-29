include { slurm_logs; slurm_log_dir; biahub_cmd } from './common'

def qc_cmd() {
    return params.qc_project ?
        "uv run --project ${params.qc_project} imaging-qc" :
        "uv run --from 'imaging-qc-pipeline @ git+https://github.com/czbiohub-sf/imaging-qc-pipeline@v0.3.2' imaging-qc"
}


process plan_stage {
    label 'cpu_local'
    tag "${zarr_path}"

    input:
    tuple val(zarr_path), val(config_path)

    output:
    tuple val(zarr_path), val(config_path), stdout

    script:
    def chunk_arg = params.qc_chunk_size ? "--chunk-size ${params.qc_chunk_size}" : ""
    """
    mkdir -p "${slurm_log_dir('qc')}"
    ${qc_cmd()} plan-stage --config ${config_path} ${chunk_arg} ${zarr_path}
    """
}


process run_step {
    tag "${zarr_path}/${position ?: 'store'}/${step_id}"
    label 'cpu'
    clusterOptions { slurm_logs('qc') }
    memory { "${(meta?.memory_gb ?: 16).toFloat() * task.attempt} GB" }
    time '2h'
    maxRetries 1
    errorStrategy { task.exitStatus in [137, 140, 143] ? 'retry' : 'terminate' }

    input:
    tuple val(zarr_path), val(config_path), val(step_id),
          val(position), val(chunk_id), val(time_indices), val(meta)

    output:
    tuple val(zarr_path), val(step_id), val(position)

    script:
    def pos_arg = position ? "--positions '${position}'" : ""
    def chunk_arg = (chunk_id && time_indices) ? "--chunk-id ${chunk_id} --time-indices ${time_indices}" : ""
    """
    ${qc_cmd()} run-step --config ${config_path} --step-id ${step_id} \
        ${pos_arg} ${chunk_arg} ${zarr_path}
    """
}


process finalize_wave {
    label 'cpu'
    clusterOptions { slurm_logs('qc') }
    memory { task.attempt == 1 ? '32 GB' : '48 GB' }
    time '30m'
    maxRetries 1
    errorStrategy { task.exitStatus in [137, 140, 143] ? 'retry' : 'terminate' }
    tag "${zarr_path}/wave${wave_id}"

    input:
    tuple val(zarr_path), val(config_path), val(wave_id)

    output:
    tuple val(zarr_path), val(wave_id)

    script:
    """
    ${qc_cmd()} finalize-wave --config ${config_path} --wave-id ${wave_id} ${zarr_path}
    """
}


process finalize_stage {
    label 'cpu'
    clusterOptions { slurm_logs('qc') }
    memory { task.attempt == 1 ? '32 GB' : '48 GB' }
    time '1h'
    maxRetries 1
    errorStrategy { task.exitStatus in [137, 140, 143] ? 'retry' : 'terminate' }
    tag "${zarr_path}"

    input:
    tuple val(zarr_path), val(config_path)

    output:
    tuple val(zarr_path), stdout

    script:
    """
    ${qc_cmd()} finalize-stage --config ${config_path} ${zarr_path}
    """
}

process generate_report_spec {
    label 'cpu_local'

    input:
    val zarr_paths

    output:
    path 'report_spec.yaml'

    script:
    def config_flag = params.qc_config_dir ? "--config-dir \"${params.qc_config_dir}\"" : ''
    def zarr_args = zarr_paths.collect { "\"${it}\"" }.join(' ')
    """
    ${biahub_cmd()} nf generate-report-spec \
        -o report_spec.yaml \
        ${config_flag} \
        ${zarr_args}
    """
}

process run_report {
    label 'cpu'
    clusterOptions { slurm_logs('qc') }
    memory '32 GB'
    time '1h'

    input:
    path report_spec
    val report_dir

    output:
    val true

    script:
    def static_flag = params.qc_report_static ? '--static' : ''
    """
    ${qc_cmd()} report \
        --report-spec "${report_spec}" \
        "${report_dir}" \
        ${static_flag}
    """
}
