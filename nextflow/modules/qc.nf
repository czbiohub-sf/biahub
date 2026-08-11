include { plan_stage }                       from './qc_processes'
include { estimate_resources }               from './qc_processes'
include { compute_step as compute_step_w0 }  from './qc_processes'
include { compute_step as compute_step_w1 }  from './qc_processes'
include { compute_step as compute_step_w2 }  from './qc_processes'
include { finalize_wave }                    from './qc_processes'
include { finalize_stage }                   from './qc_processes'
include { generate_report_spec }             from './qc_processes'
include { run_report }                       from './qc_processes'


// ---------------------------------------------------------------------------
//  qc_stage_wf: plan-driven QC stage execution
//
//  plan-stage emits plan.json v3 to stdout. Nextflow parses JSON, branches
//  items by wave_id, and uses .count() barriers between waves. Only the
//  waves/items keys are read, so additive schema bumps stay compatible.
//  Fixed 3-tier structure: wave 0 → finalize_wave → wave 1 → wave 2 →
//  finalize_stage. Empty waves are no-ops (.count() emits 0).
// ---------------------------------------------------------------------------

workflow qc_stage_wf {
    take:
    plan_inputs      // Channel of tuple(zarr_path, config_path)

    main:
    plan_out = plan_stage(plan_inputs)

    // Per-(zarr,config) memory estimate from imaging-qc's estimate-resources CLI.
    // Keyed by [zarr, config] and joined 1:1 into the plan so every work item
    // carries `mem` (GB), which becomes compute_step's meta.memory_gb.
    est_mem = estimate_resources(plan_inputs)
        .map { z, cfg, est_json ->
            def line = est_json.trim().readLines().findAll { it.trim().startsWith('{') }.last()
            def r = new groovy.json.JsonSlurper().parseText(line)
            tuple([z, cfg], (r.estimate_gb ?: 16) as Double)
        }

    // Parse plan JSON, flatten into work items, branch by wave_id
    items = plan_out
        .map { z, cfg, json_text -> tuple([z, cfg], json_text) }
        .join(est_mem)
        .flatMap { key, json_text, mem ->
            def (z, cfg) = key
            def plan = new groovy.json.JsonSlurper().parseText(json_text.trim())
            plan.waves.collectMany { w ->
                (w.items ?: []).collect { i ->
                    [z, cfg, w.wave_id, i.step_id,
                     i.position ?: null, i.chunk_id ?: null,
                     i.time_indices ?: null, mem]
                }
            }
        }
        .branch { w0: it[2] == 0; w1: it[2] == 1; w2: it[2] == 2 }

    // Wave 0: position-scoped (may be chunked)
    w0_in = items.w0.map { z,c,wid,sid,pos,cid,ti,mem -> [z,c,sid,pos,cid,ti,[memory_gb: mem]] }
    w0_done = compute_step_w0(w0_in)
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
        .map { z,c,wid,sid,pos,cid,ti,mem,n -> [z,c,sid,pos,null,null,[memory_gb: mem]] }
    w1_done = compute_step_w1(w1_in)
    w1_count = w1_done.mix(fw0).count()

    // Wave 2: store-scoped (after wave 1)
    w2_in = items.w2
        .combine(w1_count)
        .map { z,c,wid,sid,pos,cid,ti,mem,n -> [z,c,sid,null,null,null,[memory_gb: mem]] }
    w2_done = compute_step_w2(w2_in)

    // Finalize stage: aggregate + gate + summary
    all_done = w0_done.mix(w1_done, w2_done).count()
    merged = plan_out.map { z, cfg, json -> [z, cfg] }
        .combine(all_done)
        .map { z, cfg, n -> [z, cfg] }
        | finalize_stage

    emit:
    done = merged.map { z, summary -> z }
}


workflow qc_report_wf {
    take:
    all_qc_done      // collected list of zarr paths
    report_dir

    main:
    spec = generate_report_spec(all_qc_done)
    run_report(spec, report_dir)

    emit:
    done = run_report.out
}
