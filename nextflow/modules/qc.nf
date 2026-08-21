// QC stage execution and reporting: the processes that call the external
// `imaging-qc` CLI, and the two workflows that wire them. One file, as every
// other step module in this directory is.
include { slurm_logs; slurm_log_dir } from './common'


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
    mkdir -p "${slurm_log_dir('qc')}"
    imaging-qc plan-stage --config ${config_path} ${chunk_arg} ${zarr_path}
    """
}


// Data-driven per-task memory: defer to imaging-qc's own `estimate-resources`
// rather than reimplementing the estimate here. The CLI reads store metadata
// and metric scopes and prints a JSON payload whose `estimate_gb` drives
// `compute_step`'s dynamic `memory` (parsed in qc.nf and threaded through as
// `meta.memory_gb`). `--num-workers 1` because Nextflow fans out one position
// per compute task, so the relevant peak is per-position, not per-store.
process estimate_resources {
    label 'cpu_local'
    tag "${zarr_path}"

    input:
    tuple val(zarr_path), val(config_path)

    output:
    tuple val(zarr_path), val(config_path), stdout

    script:
    """
    imaging-qc estimate-resources --config ${config_path} --num-workers 1 ${zarr_path}
    """
}


// `cpus` is not decoration: imaging-qc dispatches a position's (T, C) units through
// a ThreadPoolExecutor sized from `max_concurrent`, which — unset in our configs —
// defaults to this process's CPU affinity (cli/composable.py). At one CPU the
// timepoints of a position are walked serially, so this is the knob that decides
// whether a per-position task uses the node it reserved.
process compute_step {
    tag "${zarr_path}/${position ?: 'store'}/${step_id}"
    label 'cpu'
    clusterOptions { slurm_logs('qc') }
    cpus { params.qc_cpus as int }
    memory { "${(meta?.memory_gb ?: 16).toFloat() * task.attempt} GB" }
    time '2h'
    maxRetries 1
    errorStrategy { task.exitStatus in [137, 140, 143] ? 'retry' : 'terminate' }

    input:
    tuple val(zarr_path), val(config_path), val(step_id),
          val(position), val(chunk_id), val(time_indices), val(meta)

    output:
    tuple val(zarr_path), val(step_id), val(position)

    script:
    def pos_arg = position ? "--positions '${position}'" : ""
    def chunk_arg = (chunk_id && time_indices) ? "--chunk-id ${chunk_id} --time-indices ${time_indices}" : ""
    """
    imaging-qc compute --config ${config_path} --step-id ${step_id} \
        ${pos_arg} ${chunk_arg} ${zarr_path}
    """
}


process finalize_stage {
    label 'cpu'
    clusterOptions { slurm_logs('qc') }
    memory { task.attempt == 1 ? '32 GB' : '48 GB' }
    time '1h'
    maxRetries 1
    errorStrategy { task.exitStatus in [137, 140, 143] ? 'retry' : 'terminate' }
    tag "${zarr_path}"

    input:
    tuple val(zarr_path), val(config_path)

    output:
    tuple val(zarr_path), val(config_path), stdout

    script:
    """
    imaging-qc consolidate --config ${config_path} ${zarr_path}
    imaging-qc gate --config ${config_path} ${zarr_path}
    """
}

// ONE report over every QC'd store, driven by the report spec written at launch
// (see `qc_report_spec()` in qc.nf). Each tab is one store; imaging-qc discovers
// each tab's stage from its `qc_dir` and loads that stage's config, which is why
// the configs carry `stage{N}_` filename prefixes.
//
// `--report-spec` is mutually exclusive with a positional zarr path and with
// `--qc-dir`: everything per-store lives in the spec.
process generate_unified_report {
    label 'cpu'
    clusterOptions { slurm_logs('qc') }
    cpus 2
    memory '32 GB'
    time '1h'

    input:
    tuple path(report_spec), val(report_dir)

    output:
    val report_dir

    script:
    def static_flag = params.qc_report_static ? '--static' : ''
    """
    imaging-qc report \
        --report-spec "${report_spec}" \
        ${static_flag} \
        "${report_dir}"
    """
}


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
// and it is also what identifies the tab to a reader — "4-assemble", "5-track" —
// rather than the stage number, which is imaging-qc's internal table namespace.
def qc_report_spec(tabs, spec_path, title) {
    def spec = file(spec_path)
    spec.parent.mkdirs()

    def lines = ["title: \"${title}\"", "tabs:"]
    tabs.each { t ->
        // The config FILE, which is the same file the compute steps are given,
        // so a tab renders the settings that stage was actually run with. The
        // report verb composes the file's Hydra `defaults:`, so a `report:` block
        // inherited from base.yaml arrives intact and each tab resolves its own
        // config independently of what sits beside it on disk.
        //
        // Nothing here keys on the stage NUMBER: that lives only in each config's
        // `stage:` key, so renumbering a stage changes nothing in this function.
        lines << "  - label: \"${t.label}\""
        lines << "    qc_dir: ${t.zarr}/tables/qc"
        lines << "    zarr_path: ${t.zarr}"
        lines << "    config: ${file(t.config)}"
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
