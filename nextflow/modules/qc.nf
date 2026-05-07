include { dataset_name; biahub_cmd } from './common'


def qc_cmd() {
    return params.qc_project ?
        "uv run --project ${params.qc_project} imaging-qc" :
        "uv run --from 'imaging-qc-pipeline @ git+https://github.com/czbiohub-sf/imaging-qc-pipeline@v0.3.2' imaging-qc"
}


// ---------------------------------------------------------------------------
//  Processes: each wraps a single imaging-qc CLI call
// ---------------------------------------------------------------------------

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
    ${qc_cmd()} plan-stage --format csv --config ${config_path} ${chunk_arg} ${zarr_path}
    """
}


process run_step {
    tag "${zarr_path}/${position ?: 'store'}/${step_id}"
    label 'cpu'
    memory { "${16 * task.attempt} GB" }
    time '2h'
    maxRetries 1
    errorStrategy { task.exitStatus in [137, 140, 143] ? 'retry' : 'terminate' }

    input:
    tuple val(zarr_path), val(config_path), val(step_id),
          val(position), val(chunk_id), val(time_indices)

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


// ---------------------------------------------------------------------------
//  qc_stage_wf: plan-driven QC stage execution
//
//  plan-stage --format csv emits flat CSV rows. Nextflow splits, branches
//  by wave_id, and feeds rows to run_step with barriers between waves.
//  Fixed 3-tier structure: wave 0 → finalize_wave → wave 1 → wave 2 →
//  finalize_stage. Empty waves are no-ops (ifEmpty propagates).
// ---------------------------------------------------------------------------

workflow qc_stage_wf {
    take:
    prev_done        // Channel: trigger from upstream
    zarr_path        // val: path to zarr store
    config_path      // val: path to stage YAML config
    stage_name       // val: human-readable stage name

    main:
    // Plan: Python emits CSV fan-out rows to stdout
    plan_input = prev_done.map { tuple(zarr_path, config_path) }
    plan_out = plan_stage(plan_input)

    // Parse CSV and branch by wave_id
    parsed = plan_out
        .flatMap { z, cfg, csv_text ->
            csv_text.trim().tokenize('\n').tail().collect { line ->
                def c = line.split(',', -1)
                tuple(z, cfg, c[0].toInteger(), c[1], c[2], c[3], c[4], c[5], c[6])
                // z, cfg, wave_id, step_id, scope, position, chunk_id, time_indices, finalize_after
            }
        }
        .branch {
            w0: it[2] == 0
            w1: it[2] == 1
            w2: it[2] == 2
        }

    // Wave 0: position-scoped (may be chunked)
    w0_input = parsed.w0
        .map { z, cfg, wid, sid, scope, pos, cid, ti, fin ->
            [z, cfg, sid, pos ?: null, cid ?: null, ti ?: null]
        }
    w0_done = run_step(w0_input)

    // Finalize wave 0 (merge chunks before dependent metrics can read them)
    // finalize-wave is a no-op if nothing to merge, so always call it as barrier
    w0_barrier = w0_done
        .map { z, sid, pos -> z }
        .ifEmpty('none')
        .collect()

    fw0_done = w0_barrier
        .map { done -> [zarr_path, config_path, 0] }
        | finalize_wave

    // Wave 1: dependent metrics (runs after finalize_wave)
    w1_input = parsed.w1
        .map { z, cfg, wid, sid, scope, pos, cid, ti, fin ->
            [z, cfg, sid, pos ?: null, null, null]
        }
        .combine(fw0_done.collect())
        .map { z, cfg, sid, pos, cid, ti, barrier ->
            [z, cfg, sid, pos, cid, ti]
        }
    w1_done = run_step(w1_input)

    w1_barrier = w1_done
        .map { z, sid, pos -> z }
        .ifEmpty('none')
        .mix(fw0_done.map { z, wid -> z })
        .collect()

    // Wave 2: store-scoped (no position fan-out)
    w2_input = parsed.w2
        .map { z, cfg, wid, sid, scope, pos, cid, ti, fin ->
            [z, cfg, sid, null, null, null]
        }
        .combine(w1_barrier)
        .map { z, cfg, sid, pos, cid, ti, barrier ->
            [z, cfg, sid, pos, cid, ti]
        }
    w2_done = run_step(w2_input)

    // Finalize stage: aggregate + gate + summary
    all_compute = w0_done.mix(w1_done, w2_done)
        .map { z, sid, pos -> z }
        .ifEmpty('none')
        .collect()

    merged = all_compute
        .map { done -> [zarr_path, config_path] }
        | finalize_stage

    emit:
    done    = merged.map { z, summary -> z }
    summary = merged.map { z, summary -> summary }
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
