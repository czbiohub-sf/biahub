include { dataset_name; biahub_cmd } from './common'


def qc_cmd() {
    return params.qc_project ?
        "uv run --project ${params.qc_project} imaging-qc" :
        "uv run --from 'imaging-qc-pipeline @ git+https://github.com/czbiohub-sf/imaging-qc-pipeline@v0.3.1' imaging-qc"
}


process run_qc_position {
    tag "${position}"
    label 'cpu_medium'
    time '1h'
    queue 'cpu'
    maxRetries 1
    errorStrategy { task.exitStatus in [0, 1] ? 'ignore' : 'retry' }

    input:
    tuple val(position), val(zarr_path), val(config_path)

    output:
    val position

    script:
    """
    ${qc_cmd()} run \
        --config "${config_path}" \
        "${zarr_path}" \
        --mode metrics_only \
        --no-merge \
        --chunk-id default \
        'positions=["${position}"]'
    """
}


process merge_qc_stage {
    label 'cpu_small'
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
    label 'cpu_small'

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
    label 'cpu_small'

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
    label 'cpu_medium'

    input:
    val assembly_zarr
    val report_dir
    val config_path
    val consolidated

    output:
    val true

    script:
    """
    ${qc_cmd()} merge "${assembly_zarr}"
    ${qc_cmd()} report \
        --config "${config_path}" \
        "${assembly_zarr}" \
        "${report_dir}" \
        --static
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
    barrier = prev_done.map { 'done' }
    qc_done = positions
        .flatMap { it }
        .combine(barrier)
        .map { pos, _barrier -> [pos, zarr_path, config_path] }
        | run_qc_position
        | collect

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
    report_dir
    config_path

    main:
    consolidated = consolidate_qc(all_qc_done, step_zarrs, assembly_zarr)
    final_merge_and_report(assembly_zarr, report_dir, config_path, consolidated)
    log_qc_summary(all_summaries)

    emit:
    done = log_qc_summary.out
}
