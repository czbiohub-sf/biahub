#!/usr/bin/env nextflow
//
// Standalone QC pipeline — runs imaging-qc against arbitrary zarr stores
// listed in a CSV manifest. Python owns all dispatch logic via
// plan-stage --format csv; Nextflow owns fan-out, barriers, and retries.
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

include { biahub_cmd }             from './modules/common'
include { plan_stage }             from './modules/qc'
include { run_step }               from './modules/qc'
include { finalize_wave }          from './modules/qc'
include { finalize_stage }         from './modules/qc'
include { consolidate_qc }         from './modules/qc'
include { log_qc_summary }         from './modules/qc'
include { final_merge_and_report } from './modules/qc'


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

    // ── Plan each stage via plan-stage --format csv ─────────────────
    plan_inputs = manifest
        .flatMap { rows -> rows.collect { row -> tuple(row[0], row[1]) } }

    plans = plan_stage(plan_inputs)

    // Parse CSV rows and branch by wave_id
    parsed = plans
        .flatMap { z, cfg, csv_text ->
            csv_text.trim().tokenize('\n').tail().collect { line ->
                def c = line.split(',', -1)
                tuple(z, cfg, c[0].toInteger(), c[1], c[2], c[3], c[4], c[5], c[6])
            }
        }
        .branch {
            w0: it[2] == 0
            w1: it[2] == 1
            w2: it[2] == 2
        }

    // ── Wave 0: position-scoped (chunked) ───────────────────────────
    w0_input = parsed.w0
        .map { z, cfg, wid, sid, scope, pos, cid, ti, fin ->
            [z, cfg, sid, pos ?: null, cid ?: null, ti ?: null]
        }
    w0_done = run_step(w0_input)

    // Finalize wave 0 (merge chunks)
    w0_zarrs = w0_done
        .map { z, sid, pos -> z }
        .ifEmpty('none')
        .collect()

    // Get unique zarrs that had wave 0 work
    fw0_input = plans
        .map { z, cfg, csv_text -> [z, cfg] }
        .combine(w0_zarrs)
        .map { z, cfg, done -> [z, cfg, 0] }

    fw0_done = finalize_wave(fw0_input)

    // ── Wave 1: dependent metrics ───────────────────────────────────
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

    // ── Wave 2: store-scoped ────────────────────────────────────────
    w2_input = parsed.w2
        .map { z, cfg, wid, sid, scope, pos, cid, ti, fin ->
            [z, cfg, sid, null, null, null]
        }
        .combine(w1_barrier)
        .map { z, cfg, sid, pos, cid, ti, barrier ->
            [z, cfg, sid, pos, cid, ti]
        }
    w2_done = run_step(w2_input)

    // ── Finalize each stage ─────────────────────────────────────────
    all_compute = w0_done.mix(w1_done, w2_done)
        .map { z, sid, pos -> z }
        .ifEmpty('none')
        .collect()

    fs_inputs = manifest
        .flatMap { rows -> rows.collect { row -> [row[0], row[1]] } }
        .combine(all_compute)
        .map { zarr, config, done -> [zarr, config] }

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
