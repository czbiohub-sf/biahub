include { plan_stage }                       from './qc_processes'
include { estimate_resources }               from './qc_processes'
include { compute_step }                      from './qc_processes'
include { finalize_stage }                    from './qc_processes'
include { generate_unified_report }          from './qc_processes'


// ---------------------------------------------------------------------------
//  qc_stage_wf: plan-driven QC stage execution
//
//  plan-stage emits plan.json v5 to stdout: {version, stage, items[]}, one flat
//  list. Nextflow fans every item out to a compute task, counts them as the
//  barrier, and merges once. imaging-qc's Phase 20 removed the wave mechanism —
//  `waves[]`, `wave_id`, `scope:` on a metric group, and `consolidate
//  --wave-id`, which the old two-merge structure here was built on — after
//  finding nothing could ever put a second wave in a plan.
//
//  Only the items keys are read, so additive schema bumps stay compatible.
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

    items = plan_out
        .map { z, cfg, json_text -> tuple([z, cfg], json_text) }
        .join(est_mem)
        .flatMap { key, json_text, mem ->
            def (z, cfg) = key
            def plan = new groovy.json.JsonSlurper().parseText(json_text.trim())
            // REFUSE a plan with no readable items rather than defaulting to an
            // empty list. An envelope this driver cannot read and a stage that
            // planned nothing are the same empty channel, and the second is
            // legitimate — so a default would turn a schema bump into a silent
            // no-op that re-gates the previous run's table at exit 0.
            if (!plan.containsKey('items') || !(plan.items instanceof List)) {
                error "Plan JSON for ${z} has no readable 'items' list (keys: ${plan.keySet()}). " +
                      "plan.json v5 is {version, stage, items[]}; a plan carrying 'waves' comes " +
                      "from an imaging-qc older than Phase 20, which this driver no longer speaks."
            }
            plan.items.collect { i ->
                [z, cfg, i.step_id, i.position ?: null, i.chunk_id ?: null,
                 i.time_indices ?: null, [memory_gb: mem]]
            }
        }

    done = compute_step(items)

    // `.count()` is the barrier: every item's shard is on disk before the stage
    // merge reads them. finalize_stage consolidates, then gates — one merge per
    // stage, which is all there is now that waves are gone.
    merged = plan_out.map { z, cfg, json -> [z, cfg] }
        .combine(done.count())
        .map { z, cfg, n -> [z, cfg] }
        | finalize_stage

    emit:
    // (zarr, config) rather than the zarr alone: the report needs the config that
    // produced the store's tables, and rediscovering it downstream would be a guess.
    done = merged.map { z, cfg, summary -> tuple(z, cfg) }
}


// Write the report-spec manifest that `imaging-qc report --report-spec` consumes:
// one tab per QC'd store, in the order given.
//
// A LAUNCH-TIME function, not a process. Its content is fully determined before
// any compute runs — the store paths, the labels and the table location are all
// known from params and the pipeline's own directory layout — so there is nothing
// to schedule and nothing to wait for. Writing it here also means a malformed
// spec (an empty label, a bad path) fails at launch instead of after every
// compute task has already run, which is exactly how the two bugs this replaces
// were found. `qc_dir` is DECLARED as the store's own `tables/qc/` group rather
// than probed on disk: this pipeline passes no `--output-dir`, so that is where
// its tables go, and a probe run before the tables exist would guess wrong.
//
// `tabs` is a list of maps: [label: <tab label>, zarr: <store>, config: <stage config>].
// Labels must be unique within a spec; a step directory name gives that for free,
// and it is also what identifies the tab to a reader — "5-assemble", "4-track" —
// rather than the stage number, which is imaging-qc's internal table namespace.
def qc_report_spec(tabs, spec_path, title) {
    def spec = file(spec_path)
    spec.parent.mkdirs()

    def lines = ["title: \"${title}\"", "tabs:"]
    tabs.each { t ->
        // The config DIRECTORY, not the file, for two reasons that pull the same
        // way. imaging-qc's report verb does not compose Hydra `defaults:`, so a
        // config inheriting its `report:` block from base.yaml loses every metric
        // plot when handed the file alone (imaging-qc-pipeline#201) — the
        // directory scan finds both files. And the scan takes the FIRST YAML with
        // a `report:` section, so the directory must hold only this step's
        // configs: point two tabs at one shared directory and both render the
        // same block, which is how a cell-count tab ends up with pixel metric
        // labels and no instance_count plot, at exit 0.
        //
        // Hence `nextflow/configs/qc/<step>/`. Nothing here keys on the stage
        // NUMBER: that lives only in each config's `stage:` key, so renumbering a
        // stage renames no files and changes nothing in this function.
        def config_dir = file(t.config).parent
        lines << "  - label: \"${t.label}\""
        lines << "    qc_dir: ${t.zarr}/tables/qc"
        lines << "    zarr_path: ${t.zarr}"
        lines << "    config: ${config_dir}"
    }
    spec.text = lines.join('\n') + '\n'
    return spec
}


// One unified report over every store QC'd in this run.
workflow qc_report_wf {
    take:
    qc_done          // Channel of tuple(zarr_path, config_path), one per finalized store
    report_spec      // spec file from qc_report_spec()
    report_dir

    main:
    // The barrier: one report reads every store's consolidated tables, so all of
    // them must be final first. `.count()` waits for the whole channel.
    ready = qc_done.count().map { n -> tuple(report_spec, report_dir) }
    reports = generate_unified_report(ready)

    emit:
    done = reports
}
