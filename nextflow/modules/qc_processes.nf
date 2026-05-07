include { biahub_cmd } from './common'


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
    ${qc_cmd()} plan-stage --config ${config_path} ${chunk_arg} ${zarr_path}
    """
}


process run_step {
    tag "${zarr_path}/${position ?: 'store'}/${step_id}"
    label 'cpu'
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
    memory { "${32 * task.attempt} GB" }
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
    memory { "${32 * task.attempt} GB" }
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


process consolidate_qc {
    label 'cpu_local'

    input:
    val all_qc_done
    val step_zarrs
    val assembly_zarr

    output:
    val true

    script:
    def step_args = step_zarrs.collect { "-s ${it}" }.join(' ')
    """
    if [ -n "${assembly_zarr}" ] && [ -d "${assembly_zarr}" ]; then
        ${biahub_cmd()} nf qc consolidate ${step_args} -a "${assembly_zarr}"
    else
        echo "No assembly zarr provided or found — skipping consolidation"
    fi
    """
}


process log_qc_summary {
    label 'cpu_local'

    input:
    val summaries

    output:
    val true

    script:
    """
    ${biahub_cmd()} nf qc log-summary ${summaries}
    """
}


process final_merge_and_report {
    label 'cpu'
    memory '32 GB'
    time '1h'

    input:
    val output_dir
    val report_dir
    val consolidated

    output:
    val true

    script:
    def static_flag = params.qc_report_static ? '--static' : ''
    def config_flag = params.qc_config_dir ? "--config \"${params.qc_config_dir}\"" : ''
    def path_prefix = params.quarto_bin ? "export PATH=\"${params.quarto_bin}:\${PATH}\"" : ''
    """
    ${path_prefix}
    ${qc_cmd()} report \
        --multi-store "${output_dir}" \
        ${config_flag} \
        "${report_dir}" \
        ${static_flag}
    """
}
