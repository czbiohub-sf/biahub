include { slurm_logs; slurm_log_dir } from './common'


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
    imaging-qc plan-stage --config ${config_path} ${chunk_arg} ${zarr_path}
    """
}


// Data-driven per-task memory: defer to imaging-qc's own `estimate-resources`
// rather than reimplementing the estimate here. The CLI reads store metadata
// and metric scopes and prints a JSON payload whose `estimate_gb` drives
// `compute_step`'s dynamic `memory` (parsed in qc.nf and threaded through as
// `meta.memory_gb`). `--num-workers 1` because Nextflow fans out one position
// per compute task, so the relevant peak is per-position, not per-store.
process estimate_resources {
    label 'cpu_local'
    tag "${zarr_path}"

    input:
    tuple val(zarr_path), val(config_path)

    output:
    tuple val(zarr_path), val(config_path), stdout

    script:
    """
    imaging-qc estimate-resources --config ${config_path} --num-workers 1 ${zarr_path}
    """
}


// `cpus` is not decoration: imaging-qc dispatches a position's (T, C) units through
// a ThreadPoolExecutor sized from `max_concurrent`, which — unset in our configs —
// defaults to this process's CPU affinity (cli/composable.py). At one CPU the
// timepoints of a position are walked serially, so this is the knob that decides
// whether a per-position task uses the node it reserved.
process compute_step {
    tag "${zarr_path}/${position ?: 'store'}/${step_id}"
    label 'cpu'
    clusterOptions { slurm_logs('qc') }
    cpus { params.qc_cpus as int }
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
    imaging-qc compute --config ${config_path} --step-id ${step_id} \
        ${pos_arg} ${chunk_arg} ${zarr_path}
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
    tuple val(zarr_path), val(config_path), stdout

    script:
    """
    imaging-qc consolidate --config ${config_path} ${zarr_path}
    imaging-qc gate --config ${config_path} ${zarr_path}
    """
}

// ONE report over every QC'd store, driven by the report spec written at launch
// (see `qc_report_spec()` in qc.nf). Each tab is one store; imaging-qc discovers
// each tab's stage from its `qc_dir` and loads that stage's config, which is why
// the configs carry `stage{N}_` filename prefixes.
//
// `--report-spec` is mutually exclusive with a positional zarr path and with
// `--qc-dir`: everything per-store lives in the spec.
process generate_unified_report {
    label 'cpu'
    clusterOptions { slurm_logs('qc') }
    cpus 2
    memory '32 GB'
    time '1h'

    input:
    tuple path(report_spec), val(report_dir)

    output:
    val report_dir

    script:
    def static_flag = params.qc_report_static ? '--static' : ''
    """
    imaging-qc report \
        --report-spec "${report_spec}" \
        ${static_flag} \
        "${report_dir}"
    """
}
