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
        error "Provide --stages_manifest (CSV with header: zarr_path,config_path,stage_name,assembly)"
    }
    if (!params.output_dir) {
        error "Provide --output_dir"
    }

    // ── Parse manifest ──────────────────────────────────────────────
    manifest = Channel
        .fromPath(params.stages_manifest)
        .splitCsv(header: true)
        .map { row ->
            def is_asm = (row.assembly ?: 'false').trim().toLowerCase() in ['true', '1', 'yes']
            [row.zarr_path.trim(), row.config_path.trim(), row.stage_name.trim(), is_asm]
        }
        .toList()

    // ── Run QC via shared workflow ──────────────────────────────────
    plan_inputs = manifest
        .flatMap { rows -> rows.collect { row -> tuple(row[0], row[1]) } }

    qc = qc_stage_wf(plan_inputs)

    all_qc_done   = qc.done.collect()
    all_summaries  = qc.summary.collect()

    // ── Consolidate (local) ─────────────────────────────────────────
    step_zarrs = manifest
        .map { rows -> rows.findAll { !it[3] }.collect { it[0] } }

    assembly_zarr = manifest
        .map { rows ->
            def asm = rows.find { it[3] }
            asm ? asm[0] : ''
        }

    // ── Report ──────────────────────────────────────────────────────
    def report_dir = params.qc_report_dir ?: "${params.output_dir}/qc/report"
    qc_report_wf(all_qc_done, all_summaries, assembly_zarr, step_zarrs, params.output_dir, report_dir)
}
