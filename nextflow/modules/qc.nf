include { plan_stage }                         from './qc_processes'
include { run_step as run_step_w0 }            from './qc_processes'
include { run_step as run_step_w1 }            from './qc_processes'
include { run_step as run_step_w2 }            from './qc_processes'
include { finalize_wave }                      from './qc_processes'
include { finalize_stage }                     from './qc_processes'
include { final_merge_and_report }             from './qc_processes'


// ---------------------------------------------------------------------------
//  qc_stage_wf: plan-driven QC stage execution
//
//  plan-stage emits plan.json v2 to stdout. Nextflow parses JSON, branches
//  items by wave_id, and uses .count() barriers between waves.
//  Fixed 3-tier structure: wave 0 -> finalize_wave -> wave 1 -> wave 2 ->
//  finalize_stage. Empty waves are no-ops (.count() emits 0).
// ---------------------------------------------------------------------------

workflow qc_stage_wf {
    take:
    plan_inputs

    main:
    plan_out = plan_stage(plan_inputs)

    items = plan_out
        .flatMap { zarr_path, config_path, json_text ->
            def plan = new groovy.json.JsonSlurper().parseText(json_text.trim())
            plan.waves.collectMany { wave ->
                (wave.items ?: []).collect { item ->
                    [
                        zarr_path,
                        config_path,
                        wave.wave_id,
                        item.step_id,
                        item.position ?: null,
                        item.chunk_id ?: null,
                        item.time_indices ?: null,
                    ]
                }
            }
        }
        .branch { w0: it[2] == 0; w1: it[2] == 1; w2: it[2] == 2 }

    w0_in = items.w0.map { zarr_path, config_path, wave_id, step_id, position, chunk_id, time_indices ->
        [zarr_path, config_path, step_id, position, chunk_id, time_indices, [:]]
    }
    w0_done = run_step_w0(w0_in)
    w0_count = w0_done.count()

    fw0 = finalize_wave(
        plan_out.map { zarr_path, config_path, json_text -> [zarr_path, config_path] }
            .combine(w0_count)
            .map { zarr_path, config_path, count -> [zarr_path, config_path, 0] }
    )
    fw0_count = fw0.count()

    w1_in = items.w1
        .combine(fw0_count)
        .map { zarr_path, config_path, wave_id, step_id, position, chunk_id, time_indices, count ->
            [zarr_path, config_path, step_id, position, null, null, [:]]
        }
    w1_done = run_step_w1(w1_in)
    w1_count = w1_done.mix(fw0).count()

    w2_in = items.w2
        .combine(w1_count)
        .map { zarr_path, config_path, wave_id, step_id, position, chunk_id, time_indices, count ->
            [zarr_path, config_path, step_id, null, null, null, [:]]
        }
    w2_done = run_step_w2(w2_in)

    all_done = w0_done.mix(w1_done, w2_done).count()
    merged = plan_out
        .map { zarr_path, config_path, json_text -> [zarr_path, config_path] }
        .combine(all_done)
        .map { zarr_path, config_path, count -> [zarr_path, config_path] }
        | finalize_stage

    emit:
    done = merged.map { zarr_path, summary -> zarr_path }
}


workflow qc_report_wf {
    take:
    all_qc_done
    output_dir
    report_dir

    main:
    final_merge_and_report(output_dir, report_dir, all_qc_done)

    emit:
    done = final_merge_and_report.out
}
