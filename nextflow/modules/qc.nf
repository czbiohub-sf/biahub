include { plan_stage }                         from './qc_processes'
include { run_step as run_step_w0 }            from './qc_processes'
include { run_step as run_step_w1 }            from './qc_processes'
include { run_step as run_step_w2 }            from './qc_processes'
include { finalize_wave }                      from './qc_processes'
include { finalize_stage }                     from './qc_processes'
include { consolidate_qc }                     from './qc_processes'
include { log_qc_summary }                     from './qc_processes'
include { final_merge_and_report }             from './qc_processes'


// ---------------------------------------------------------------------------
//  qc_stage_wf: plan-driven QC stage execution
//
//  plan-stage emits plan.json v2 to stdout. Nextflow parses JSON, branches
//  items by wave_id, and uses .count() barriers between waves.
//  Fixed 3-tier structure: wave 0 → finalize_wave → wave 1 → wave 2 →
//  finalize_stage. Empty waves are no-ops (.count() emits 0).
// ---------------------------------------------------------------------------

workflow qc_stage_wf {
    take:
    plan_inputs      // Channel of tuple(zarr_path, config_path)

    main:
    plan_out = plan_stage(plan_inputs)

    // Parse plan JSON, flatten into work items, branch by wave_id
    items = plan_out
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
        .branch { w0: it[2] == 0; w1: it[2] == 1; w2: it[2] == 2 }

    // Wave 0: position-scoped (may be chunked)
    w0_in = items.w0.map { z,c,wid,sid,pos,cid,ti -> [z,c,sid,pos,cid,ti,[:]] }
    w0_done = run_step_w0(w0_in)
    w0_count = w0_done.count()

    // Finalize wave 0 (merge chunks before dependent wave)
    fw0 = finalize_wave(
        plan_out.map { z, cfg, json -> [z, cfg] }
            .combine(w0_count)
            .map { z, cfg, n -> [z, cfg, 0] }
    )
    fw0_count = fw0.count()

    // Wave 1: dependent-scoped (after finalize wave 0)
    w1_in = items.w1
        .combine(fw0_count)
        .map { z,c,wid,sid,pos,cid,ti,n -> [z,c,sid,pos,null,null,[:]] }
    w1_done = run_step_w1(w1_in)
    w1_count = w1_done.mix(fw0).count()

    // Wave 2: store-scoped (after wave 1)
    w2_in = items.w2
        .combine(w1_count)
        .map { z,c,wid,sid,pos,cid,ti,n -> [z,c,sid,null,null,null,[:]] }
    w2_done = run_step_w2(w2_in)

    // Finalize stage: aggregate + gate + summary
    all_done = w0_done.mix(w1_done, w2_done).count()
    merged = plan_out.map { z, cfg, json -> [z, cfg] }
        .combine(all_done)
        .map { z, cfg, n -> [z, cfg] }
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
