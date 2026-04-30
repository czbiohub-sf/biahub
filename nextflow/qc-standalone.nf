#!/usr/bin/env nextflow
//
// Standalone QC pipeline — runs imaging-qc against arbitrary zarr stores
// listed in a CSV manifest, with two-pass metric computation (chunked
// position-scoped metrics first, then dependent/temporal metrics after merge),
// timepoint batching as distributed Slurm jobs, and a chained consolidation +
// report at the end.
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

include { biahub_cmd } from './modules/common'
include { init_qc_fanout }          from './modules/qc'
include { estimate_qc_resources }   from './modules/qc'
include { run_qc_chunked }          from './modules/qc'
include { run_qc_position }         from './modules/qc'
include { merge_qc_metrics }        from './modules/qc'
include { merge_qc_stage }          from './modules/qc'
include { consolidate_qc }          from './modules/qc'
include { log_qc_summary }          from './modules/qc'
include { final_merge_and_report }  from './modules/qc'


process discover_positions {
    label 'cpu_local'

    input:
    val zarr_path

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf list-positions -i "${zarr_path}"
    """
}


workflow {
    if (!params.stages_manifest) {
        error "Provide --stages_manifest (CSV with header: zarr_path,config_path,stage_name,assembly)"
    }
    if (!params.output_dir) {
        error "Provide --output_dir"
    }

    // ── Parse manifest ──────────────────────────────────────────────
    // Collect into a value channel (list of tuples) so it can be reused.
    manifest = Channel
        .fromPath(params.stages_manifest)
        .splitCsv(header: true)
        .map { row ->
            def is_asm = (row.assembly ?: 'false').trim().toLowerCase() in ['true', '1', 'yes']
            [row.zarr_path.trim(), row.config_path.trim(), row.stage_name.trim(), is_asm]
        }
        .toList()  // value channel: [[zarr, config, name, asm], ...]

    // ── Discover positions ──────────────────────────────────────────
    first_zarr = manifest.map { rows -> rows[0][0] }

    if (params.positions) {
        positions_ch = Channel.of(params.positions.tokenize(','))
            .flatMap { it }
            .map { it.trim() }
    } else {
        positions_ch = first_zarr
            | discover_positions
            | splitText
            | map { it.trim() }
            | filter { it }
    }

    all_positions = params.max_positions > 0
        ? positions_ch | take(params.max_positions) | collect
        : positions_ch | collect

    // ── Fan-out per stage ───────────────────────────────────────────
    // Emit each manifest row as a queue channel for fan-out.
    // init_qc_fanout takes tuple(zarr, config) and emits CSV rows:
    //   position, group, start, end, chunk_id
    fanout_inputs = manifest
        .flatMap { rows -> rows.collect { row -> tuple(row[0], row[1]) } }

    fanout_raw = init_qc_fanout(fanout_inputs)

    // Re-attach stage metadata. init_qc_fanout preserves input order,
    // so merge pairs each (zarr, config, name) with its stdout output.
    stage_meta = manifest
        .flatMap { rows -> rows.collect { row -> tuple(row[0], row[1], row[2]) } }

    fanout_rows = stage_meta
        .merge(fanout_raw)
        .flatMap { zarr, config, name, csv_text ->
            csv_text.trim().tokenize('\n').collect { line ->
                def cols = line.split(',', -1)
                tuple(zarr, config, name, cols[0], cols[1], cols[2], cols[3], cols[4])
            }
        }

    // ── Resource estimation ─────────────────────────────────────────
    // One estimate per unique (zarr, group, config)
    unique_groups = fanout_rows
        .map { zarr, config, name, pos, group, start, end, chunk_id ->
            tuple(zarr, group, config)
        }
        .unique()

    estimates = estimate_qc_resources(unique_groups)
        .map { group, mem_text ->
            tuple(group, mem_text.trim().toFloat().round(1))
        }

    // ── Pass 1: chunked position-scoped metrics (Slurm) ────────────
    // Rows with non-empty start/end are time-chunked
    chunked = fanout_rows
        .filter { zarr, config, name, pos, group, start, end, chunk_id -> start != '' }
        .map { zarr, config, name, pos, group, start, end, chunk_id ->
            tuple(group, pos, start, end, chunk_id, zarr, config)
        }
        .combine(estimates, by: 0)
        .map { group, pos, start, end, chunk_id, zarr, config, mem_gb ->
            tuple(pos, group, start, end, chunk_id, zarr, config, mem_gb)
        }

    chunked_done = run_qc_chunked(chunked)

    // ── Merge metrics per zarr (Slurm) ──────────────────────────────
    // After all chunked jobs finish, merge chunk parquets per zarr
    // so that pass-2 dependent metrics can read consolidated results.
    zarrs_with_chunks = chunked
        .map { pos, group, start, end, chunk_id, zarr, config, mem_gb -> zarr }
        .unique()

    metrics_merged = zarrs_with_chunks
        .combine(chunked_done.ifEmpty('none').collect())
        .map { zarr, done -> tuple(zarr, done) }
        | merge_qc_metrics

    // ── Pass 2: dependent/temporal metrics (Slurm) ──────────────────
    // Rows with empty start/end. Wait for merge_qc_metrics to finish
    // because these metrics (e.g. bleach_rate) depend on upstream
    // results (e.g. intensity_stats) written by pass-1 chunks.
    dependent = fanout_rows
        .filter { zarr, config, name, pos, group, start, end, chunk_id -> start == '' }
        .map { zarr, config, name, pos, group, start, end, chunk_id ->
            tuple(group, pos, zarr, config)
        }
        .combine(estimates, by: 0)
        .map { group, pos, zarr, config, mem_gb ->
            tuple(pos, group, zarr, config, mem_gb)
        }
        .combine(metrics_merged.collect())
        .map { pos, group, zarr, config, mem_gb, trigger ->
            tuple(pos, group, zarr, config, mem_gb)
        }

    dependent_done = run_qc_position(dependent)

    // ── Merge + gate per stage (Slurm) ──────────────────────────────
    all_qc_done = chunked_done.mix(dependent_done).collect()

    stage_merge_inputs = manifest
        .flatMap { rows -> rows.collect { row -> tuple(row[0], row[1], row[2]) } }
        .combine(all_qc_done)
        .map { zarr, config, name, done ->
            tuple(zarr, done, config, name)
        }

    summaries = merge_qc_stage(stage_merge_inputs)
    all_summaries = summaries | collect

    // ── Consolidate (local) ─────────────────────────────────────────
    // Copy QC parquets from step zarrs into the assembly zarr.
    // When no stage is marked assembly=true, the process is a no-op
    // (guarded in the consolidate_qc process script).
    step_zarrs = manifest
        .map { rows -> rows.findAll { !it[3] }.collect { it[0] } }

    assembly_zarr = manifest
        .map { rows ->
            def asm = rows.find { it[3] }
            asm ? asm[0] : ''
        }

    consolidate_done = consolidate_qc(all_summaries, step_zarrs, assembly_zarr)

    // ── Report (Slurm) ──────────────────────────────────────────────
    def report_dir = params.qc_report_dir ?: "${params.output_dir}/qc/report"
    final_merge_and_report(params.output_dir, report_dir, consolidate_done)
    log_qc_summary(all_summaries)
}
