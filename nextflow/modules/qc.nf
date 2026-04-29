include { dataset_name; biahub_cmd } from './common'


def qc_cmd() {
    return params.qc_project ?
        "uv run --project ${params.qc_project} imaging-qc" :
        "uv run --from 'imaging-qc-pipeline @ git+https://github.com/czbiohub-sf/imaging-qc-pipeline@v0.3.1' imaging-qc"
}


process init_qc_fanout {
    label 'cpu_local'

    input:
    tuple val(zarr_path), val(config_path)

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf qc init-qc-fanout -i "${zarr_path}" -c "${config_path}"
    """
}


process run_qc_chunked {
    tag "${position}/${group}/${chunk_id}"
    label 'cpu_preempted'
    time '2h'
    maxRetries 1
    errorStrategy { task.exitStatus in [0, 1] ? 'ignore' : 'retry' }

    input:
    tuple val(position), val(group), val(start), val(end), val(chunk_id),
          val(zarr_path), val(config_path)

    output:
    val position

    script:
    """
    ${qc_cmd()} run \
        --config "${config_path}" \
        "${zarr_path}" \
        --mode metrics_only \
        --metric-group ${group} \
        --no-merge \
        --time-indices ${start}:${end} \
        --chunk-id ${chunk_id} \
        'positions=["${position}"]'
    """
}


process run_qc_position {
    tag "${position}/${group}"
    label 'cpu_preempted'
    time '2h'
    maxRetries 1
    errorStrategy { task.exitStatus in [0, 1] ? 'ignore' : 'retry' }

    input:
    tuple val(position), val(group), val(zarr_path), val(config_path)

    output:
    val position

    script:
    """
    ${qc_cmd()} run \
        --config "${config_path}" \
        "${zarr_path}" \
        --mode metrics_only \
        --metric-group ${group} \
        --no-merge \
        --chunk-id ${group} \
        'positions=["${position}"]'
    """
}


process merge_qc_stage {
    label 'cpu_preempted'
    time '30m'
    errorStrategy { task.exitStatus in [0, 1] ? 'ignore' : 'retry' }

    input:
    tuple val(zarr_path), val(done), val(config_path), val(stage_name)

    output:
    path "${stage_name}_qc_summary.txt"

    script:
    """
    ${qc_cmd()} merge-metrics "${zarr_path}"
    ${qc_cmd()} run \
        --config "${config_path}" \
        "${zarr_path}" \
        --mode gate_only 2>&1 | tee gate_log.txt >&2
    ${qc_cmd()} merge-gates "${zarr_path}"
    grep 'QC_SUMMARY' gate_log.txt > ${stage_name}_qc_summary.txt || echo "no_summary" > ${stage_name}_qc_summary.txt
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
    ${biahub_cmd()} nf qc consolidate ${step_args} -a "${assembly_zarr}"
    """
}


process log_qc_summary {
    label 'cpu_local'

    input:
    path summaries

    output:
    val true

    script:
    """
    ${biahub_cmd()} nf qc log-summary ${summaries}
    """
}


process final_merge_and_report {
    label 'cpu_preempted'
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


workflow qc_stage_wf {
    take:
    positions
    prev_done
    zarr_path
    config_path
    stage_name

    main:
    fanout = prev_done
        .map { tuple(zarr_path, config_path) }
        | init_qc_fanout
        | splitCsv()

    // position-scoped groups: time-chunked (start/end non-empty)
    chunked_done = fanout
        .filter { row -> row[2] != '' }
        .map { row -> tuple(row[0], row[1], row[2], row[3], row[4],
                            zarr_path, config_path) }
        | run_qc_chunked

    // temporal-scoped groups: per-position, all timepoints (start/end empty)
    position_done = fanout
        .filter { row -> row[2] == '' }
        .map { row -> tuple(row[0], row[1], zarr_path, config_path) }
        | run_qc_position

    qc_done = chunked_done.mix(position_done) | collect

    summary = qc_done
        .map { done -> tuple(zarr_path, done, config_path, stage_name) }
        | merge_qc_stage

    emit:
    done    = summary
    summary = summary
}


workflow qc_report_wf {
    take:
    all_qc_done
    all_summaries
    assembly_zarr
    step_zarrs
    output_dir
    report_dir

    main:
    consolidated = consolidate_qc(all_qc_done, step_zarrs, assembly_zarr)
    final_merge_and_report(output_dir, report_dir, consolidated)
    log_qc_summary(all_summaries)

    emit:
    done = log_qc_summary.out
}
