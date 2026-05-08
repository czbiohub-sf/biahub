#!/usr/bin/env nextflow
//
// Standalone QC pipeline — runs imaging-qc against arbitrary zarr stores
// listed in a CSV manifest. Python owns all dispatch logic via
// plan-stage (JSON); Nextflow owns fan-out, barriers, and retries.
//
// See nextflow/README-qc-standalone.md for usage and examples.
//

nextflow.enable.dsl = 2

params.stages_manifest  = null
params.output_dir       = null
params.positions        = null
params.qc_project       = null
params.biahub_project   = null
params.quarto_bin       = null
params.qc_report_static = false
params.qc_report_dir    = null
params.qc_chunk_size    = 10
params.max_positions    = 0

include { qc_stage_wf }   from './modules/qc'
include { qc_report_wf }  from './modules/qc'


workflow {
    if (!params.stages_manifest) {
        error "Provide --stages_manifest (CSV with header: zarr_path,config_path,stage_name)"
    }
    if (!params.output_dir) {
        error "Provide --output_dir"
    }

    // ── Parse manifest ──────────────────────────────────────────────
    plan_inputs = Channel
        .fromPath(params.stages_manifest)
        .splitCsv(header: true)
        .map { row -> tuple(row.zarr_path.trim(), row.config_path.trim()) }

    qc = qc_stage_wf(plan_inputs)

    all_qc_done = qc.done.collect()

    // ── Report ──────────────────────────────────────────────────────
    def report_dir = params.qc_report_dir ?: "${params.output_dir}/qc/report"
    qc_report_wf(all_qc_done, params.output_dir, report_dir)
}
