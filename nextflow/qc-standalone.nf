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

include { biahub_cmd }                         from './modules/common'
include { plan_stage }                         from './modules/qc_processes'
include { run_step as run_step_w0 }            from './modules/qc_processes'
include { run_step as run_step_w1 }            from './modules/qc_processes'
include { run_step as run_step_w2 }            from './modules/qc_processes'
include { finalize_wave }                      from './modules/qc_processes'
include { finalize_stage }                     from './modules/qc_processes'
include { consolidate_qc }                     from './modules/qc_processes'
include { log_qc_summary }                     from './modules/qc_processes'
include { final_merge_and_report }             from './modules/qc_processes'


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

    // ── Plan each stage ─────────────────────────────────────────────
    plan_inputs = manifest
        .flatMap { rows -> rows.collect { row -> tuple(row[0], row[1]) } }

    plans = plan_stage(plan_inputs)

    // Parse plan JSON, flatten into work items, branch by wave_id
    parsed = plans
        .flatMap { z, cfg, json_text ->
            def plan = new groovy.json.JsonSlurper().parseText(json_text.trim())
            plan.waves.collectMany { w ->
                (w.items ?: []).collect { i ->
                    [z, cfg, w.wave_id, i.step_id,
                     i.position ?: null, i.chunk_id ?: null,
                     i.time_indices ?: null]
                }
            }
        }
        .branch {
            w0: it[2] == 0
            w1: it[2] == 1
            w2: it[2] == 2
        }

    // ── Wave 0: position-scoped (chunked) ───────────────────────────
    w0_in = parsed.w0.map { z,c,wid,sid,pos,cid,ti -> [z,c,sid,pos,cid,ti,[:]] }
    w0_done = run_step_w0(w0_in)
    w0_count = w0_done.count()

    // Finalize wave 0 (merge chunks)
    fw0_input = plans
        .map { z, cfg, json_text -> [z, cfg] }
        .combine(w0_count)
        .map { z, cfg, n -> [z, cfg, 0] }

    fw0_done = finalize_wave(fw0_input)
    fw0_count = fw0_done.count()

    // ── Wave 1: dependent metrics ───────────────────────────────────
    w1_in = parsed.w1
        .combine(fw0_count)
        .map { z,c,wid,sid,pos,cid,ti,n -> [z,c,sid,pos,null,null,[:]] }
    w1_done = run_step_w1(w1_in)
    w1_count = w1_done.mix(fw0_done).count()

    // ── Wave 2: store-scoped ────────────────────────────────────────
    w2_in = parsed.w2
        .combine(w1_count)
        .map { z,c,wid,sid,pos,cid,ti,n -> [z,c,sid,null,null,null,[:]] }
    w2_done = run_step_w2(w2_in)

    // ── Finalize each stage ─────────────────────────────────────────
    all_done = w0_done.mix(w1_done, w2_done).count()

    fs_inputs = manifest
        .flatMap { rows -> rows.collect { row -> [row[0], row[1]] } }
        .combine(all_done)
        .map { zarr, config, n -> [zarr, config] }

    summaries = finalize_stage(fs_inputs)
    all_summaries = summaries
        .map { zarr, summary -> summary }
        .collect()

    // ── Consolidate (local) ─────────────────────────────────────────
    step_zarrs = manifest
        .map { rows -> rows.findAll { !it[3] }.collect { it[0] } }

    assembly_zarr = manifest
        .map { rows ->
            def asm = rows.find { it[3] }
            asm ? asm[0] : ''
        }

    consolidate_done = consolidate_qc(all_summaries, step_zarrs, assembly_zarr)

    // ── Report ──────────────────────────────────────────────────────
    def report_dir = params.qc_report_dir ?: "${params.output_dir}/qc/report"
    final_merge_and_report(params.output_dir, report_dir, consolidate_done)
    log_qc_summary(all_summaries)
}
