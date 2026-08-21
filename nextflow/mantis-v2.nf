#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// ---------------------------------------------------------------------------
//  mantis-v2 timelapse reconstruction pipeline.
//
//  This file is the ORCHESTRATION layer. It owns two things the step modules
//  must not know about:
//    1. the directory LAYOUT (the DIRECTORY_LAYOUT map returned by
//       directory_layout() below), and
//    2. the ORDER steps run in and what each step reads/writes.
//
//  Each step's subworkflow (e.g. deskew_wf) is path-agnostic and speaks only in
//  zarr: this pipeline hands it explicit input_zarr/output_zarr paths. The
//  pipeline itself speaks in `input`/`output`, where `input` may NOT be a zarr
//  (in some pipelines the first step converts raw input to zarr). To reorder
//  steps, change where a step reads from here; the modules stay untouched.
//
//  Flat-field → deskew → reconstruct → virtual-stain → assemble → track → QC is
//  the full chain: assemble concatenates the deskew/reconstruct/virtual-stain
//  channels into one plate, track reads that assembled plate as its single
//  input, and QC reads the finished stores. Follow the chaining below for the
//  pattern.
//
//  The last three are OPTIONAL and selected by whether their config is given, so
//  a run performs the prefix it asks for: A549 wants assemble + track + QC, a
//  neuromast run wants assemble + QC and no tracking (issue #306). A skipped step
//  never renumbers the directories of the ones around it.
// ---------------------------------------------------------------------------

params.input = null   // raw source — may not be a zarr store
params.output = null   // output directory for all step zarrs
params.deskew_config = null
params.flat_field_config = null
params.reconstruct_config = null
params.virtual_stain_config = null
params.track_config = null
params.concatenate_config = null
params.max_positions = 0
// QC, off by default. Each param points at a stage config for one store; set
// either, both, or neither. Both QC'd stores become tabs of ONE report.
//   --qc_config       nextflow/configs/qc/assemble/pixel_metrics.yaml
//   --qc_track_config nextflow/configs/qc/track/cell_count.yaml
params.qc_config = null
params.qc_track_config = null

include { collect_positions; dataset_name; check_environment } from './modules/common'
include { deskew_wf } from './modules/deskew'
include { flat_field_wf } from './modules/flat_field'
include { reconstruct_wf } from './modules/reconstruct'
include { virtual_stain_wf } from './modules/virtual_stain'
include { track_wf } from './modules/tracking'
include { assemble_wf } from './modules/assembly'
include { qc_stage_wf; qc_report_wf; qc_report_spec } from './modules/qc'
include { notify_step; notify_run_start; notify_run_end } from './modules/notify'

// Output directory layout for the reconstruction steps — single source of
// truth. Each entry is a subdirectory under params.output where that step
// writes its <dataset>.zarr. The pipeline's raw input/output live in the
// workflow body, not here (input may not even be a zarr).
//
// THE NUMBER IS A POSITION, NOT A NAME. It is the step's index among the steps
// THIS RUN PERFORMS, so it always describes the order a reader is looking at:
// skip tracking and assemble is `4-assemble`, and a pipeline that skips
// flat-field or deskew numbers everything after it accordingly instead of
// leaving a hole. That is why the numbers appear nowhere in the source — only
// the order does, and reordering a step is still a one-line edit.
//
// The trade-off, deliberately taken: a store's directory name now depends on
// which steps ran, so `4-assemble` from a neuromast run and `4-assemble` from an
// A549 run are the same step, but an older A549 run on disk says `5-assemble`
// because tracking used to be numbered ahead of it. Numbering by execution order
// is what makes the assembled store the SAME name in both families; the previous
// fixed map gave it two different names depending on whether an unrelated later
// step ran.
//
// Defined as a function rather than a bare top-level assignment: Nextflow's DSL2
// parser only allows declarations (include/process/workflow/function) at script
// scope, so a `DIRECTORY_LAYOUT = [...]` statement fails to compile. The workflow
// body calls step_directories() once to get the map.
def step_directories(performed) {
    // Steps that write a directory, in EXECUTION order. Order here is the only
    // thing that decides numbering; the numbers themselves are not written down.
    def order = [
        // convert    : 'convert',      // first step when raw input isn't zarr
        flat_field    : 'flatfield',
        deskew        : 'deskew',
        reconstruct   : 'reconstruct',
        virtual_stain : 'virtual-stain',
        assemble      : 'assemble',
        track         : 'track',
    ]
    def layout = [:]
    order.each { key, name ->
        if (performed.contains(key)) {
            layout[key] = "${layout.size()}-${name}"
        }
    }
    return layout
}


// The reconstruction steps, in the order they run, as they are named in Slack.
// One list so the run-start announcement cannot disagree with the per-step
// messages. A function rather than a bare assignment for the same reason
// directory_layout() is one: Nextflow's DSL2 parser allows only declarations at
// script scope.
workflow {
    if (!params.input)              error "Provide --input"
    if (!params.output)             error "Provide --output"
    if (!params.flat_field_config)  error "Provide --flat_field_config"
    if (!params.deskew_config)      error "Provide --deskew_config"
    if (!params.reconstruct_config) error "Provide --reconstruct_config"
    if (!params.virtual_stain_config) error "Provide --virtual_stain_config"

    // A STEP IS SELECTED BY THE PRESENCE OF ITS CONFIG. Reconstruction proper —
    // flat-field through virtual staining — is what this pipeline is for and is
    // always run. Assemble, track and QC are deliverables some runs want and
    // others do not: tracking is tuned for A549 and is not what a neuromast run
    // is for, and its parameters do not transfer (issue #306). Naming a config is
    // how a run asks for the step; omitting it is how a run declines, with no
    // placeholder to author and no output to discard.
    //
    // Skipping a step DOES renumber the ones after it: the number is a position
    // among the steps performed, so a neuromast run's assembled store is
    // `4-assemble` where an A549 run also has `5-track` after it.
    def assemble_on = params.concatenate_config as boolean
    def track_on    = params.track_config as boolean
    def qc_image_on = params.qc_config as boolean
    def qc_track_on = params.qc_track_config as boolean
    def qc_on       = qc_image_on || qc_track_on

    // A step cannot outlive the step whose output it reads. Refuse the
    // combination at launch, naming the config to add or the one to drop, rather
    // than failing hours in with a missing store.
    if (track_on && !assemble_on) {
        error "--track_config needs --concatenate_config: tracking reads the assembled plate."
    }
    if (qc_image_on && !assemble_on) {
        error "--qc_config needs --concatenate_config: it QCs the assembled store."
    }
    if (qc_track_on && !track_on) {
        error "--qc_track_config needs --track_config: it QCs the tracking store."
    }

    // Tasks call `biahub`/`viscy`/`imaging-qc` bare, so fail now if the env isn't
    // activated. `imaging-qc` is only required when QC is actually wired in.
    check_environment(qc_on ? ['biahub', 'viscy', 'imaging-qc'] : ['biahub', 'viscy'])

    def ds  = dataset_name()
    def out = params.output

    // The steps this run performs, in execution order — the list the directory
    // numbering is derived from. Reconstruction proper is always in it.
    def performed = ['flat_field', 'deskew', 'reconstruct', 'virtual_stain']
    if (assemble_on) performed << 'assemble'
    if (track_on)    performed << 'track'
    def layout = step_directories(performed)

    collect_positions(params.input)
    all_positions = collect_positions.out

    // ----- Flat-field -------------------------------------------------------
    // The pipeline input is already a zarr, so flat-field reads it directly and
    // writes the flat-field step directory. When a convert step is added ahead
    // of flat-field, point ff_input at the convert output instead — flat_field_wf
    // doesn't care where its input comes from.
    ff_trigger = channel.value(true)
    ff_input  = params.input
    ff_output = "${out}/${layout.flat_field}/${ds}.zarr"

    ff_done = flat_field_wf(all_positions, ff_input, ff_output, params.flat_field_config, ff_trigger)

    // ----- Deskew -----------------------------------------------------------
    // Deskew reads flat-field's output and waits on ff_done before starting.
    deskew_trigger = ff_done.done
    deskew_input  = ff_output
    deskew_output = "${out}/${layout.deskew}/${ds}.zarr"

    deskew_done = deskew_wf(all_positions, deskew_input, deskew_output, params.deskew_config, deskew_trigger)

    // ----- Reconstruct ------------------------------------------------------
    // Phase reconstruction runs on the deskewed output and waits on deskew_done.
    // It reads the deskewed brightfield channel — which channel is reconstructed
    // is set by `input_channel_names` in the reconstruct config, not here.
    reconstruct_trigger = deskew_done.done
    reconstruct_input   = deskew_output
    reconstruct_output  = "${out}/${layout.reconstruct}/${ds}.zarr"

    reconstruct_done = reconstruct_wf(all_positions, reconstruct_input, reconstruct_output, params.reconstruct_config, reconstruct_trigger)

    // ----- Virtual stain ----------------------------------------------------
    // Virtual staining runs cytoland (VisCy) prediction on the reconstructed
    // output and waits on reconstruct_done. A `viscy preprocess` step inside the
    // subworkflow computes the normalization statistics the model needs; which
    // source/target channels are used is set by the virtual-stain config, not
    // here.
    virtual_stain_trigger = reconstruct_done.done
    virtual_stain_input   = reconstruct_output
    virtual_stain_output  = "${out}/${layout.virtual_stain}/${ds}.zarr"

    virtual_stain_done = virtual_stain_wf(all_positions, virtual_stain_input, virtual_stain_output, params.virtual_stain_config, virtual_stain_trigger)

    // ----- Assemble ---------------------------------------------------------
    // Concatenate the deskew, reconstruct, and virtual-stain outputs channel-wise
    // into a single multichannel plate, waiting on virtual_stain_done. Unlike the
    // per-position steps this runs single-shot on ONE reserved compute node
    // (`concatenate --cluster debug` iterates every position in-process); which
    // channels/crops come from each source is set by the concatenate config, not
    // here. The config's concat_data_paths are placeholders — the subworkflow
    // injects the three source paths via --concat-data-paths (resolve mode).
    if (assemble_on) {
        assemble_output = "${out}/${layout.assemble}/${ds}.zarr"
        assemble_done = assemble_wf(
            deskew_output,
            reconstruct_output,
            virtual_stain_output,
            assemble_output,
            params.concatenate_config,
            virtual_stain_done.done
        )
    }

    // ----- Track ------------------------------------------------------------
    // Track reads the ASSEMBLED plate for both of its inputs: assemble already
    // carries the phase and virtual-stain channels (concatenate preserves channel
    // names, so the track config's channel names resolve unchanged), so the plate
    // structure and the image data come from the same store. It waits on
    // assemble_done, which means tracking now runs AFTER assemble rather than in
    // parallel with it — the tradeoff for the single input is that the whole plate
    // must be assembled first. Two consequences of reading the assembled plate:
    // any Z/Y/X crop or time_indices subset in the concatenate config is what
    // tracking sees, and the intermediate stores are no longer needed once
    // assemble is verified. To go back to the parallel wiring, point track_input
    // at reconstruct_output, track_input_images at virtual_stain_output, and gate
    // on virtual_stain_done.
    if (track_on) {
        track_output = "${out}/${layout.track}/${ds}.zarr"
        track_done = track_wf(all_positions, assemble_output, assemble_output, track_output,
                              params.track_config, assemble_done.done)
    }

    // ----- QC ---------------------------------------------------------------
    // QC reads the finished stores, gated on the step that wrote each one:
    // the ASSEMBLED plate on assemble_done (the same signal track waits on, so
    // image QC runs CONCURRENTLY with tracking), and the tracking store on
    // track_done. Neither extends the critical path ahead of itself.
    //
    // A QC verdict cannot fail the pipeline: `imaging-qc gate` exits 0 whether
    // positions pass or fail, recording the verdict in each store's own
    // `tables/qc/` tables and a QC_SUMMARY line. Only a broken config or a
    // genuine compute error exits non-zero, which is what should stop a run.
    qc_report_dir = params.qc_report_dir ?: "${out}/qc/report"

    if (qc_on) {
        // Tab label is the step directory, which is unique per store by
        // construction — the spec requires unique labels.
        def qc_stores = []
        if (params.qc_config) {
            qc_stores << [label: layout.assemble, zarr: assemble_output,
                          config: params.qc_config, trigger: assemble_done.done]
        }
        if (params.qc_track_config) {
            qc_stores << [label: layout.track, zarr: track_output,
                          config: params.qc_track_config, trigger: track_done.done]
        }

        // Written at launch, before any QC task runs: everything in it is known
        // from the layout above, so a malformed spec fails now rather than after
        // hours of compute.
        def spec = qc_report_spec(qc_stores, "${out}/qc/report_spec.yaml", "QC — ${ds}")

        // Each store waits only on its own producer, so they QC independently.
        //
        // The trigger is mapped, NOT combined. `combine` concatenates the two
        // items, so what the producer emits leaks into the tuple's arity: assemble
        // emits one path and gave `[zarr, config, path]`, but tracking emits a
        // COLLECTED LIST of every position's output, so the tuple became
        // `[zarr, config, p1, p2, … p30]` and a three-parameter closure could not
        // be spread across it — `MissingMethodException`, after seven hours, with
        // every reconstruction step already finished. Mapping reads nothing out of
        // the trigger, so no producer's payload shape can reach this.
        //
        // `.first()` because a trigger is a signal, not a stream: one QC run per
        // store however many items its producer emits.
        qc_inputs = channel.empty()
        qc_stores.each { st ->
            qc_inputs = qc_inputs.mix( st.trigger.first().map { tuple(st.zarr, st.config) } )
        }

        qc = qc_stage_wf(qc_inputs)
        qc_report = qc_report_wf(qc.done, spec, qc_report_dir)
    }

    // ----- Notifications ----------------------------------------------------
    // One Slack message as each step finishes, plus the run-start announcement.
    // Step ORDER and labels live here with the rest of the wiring, not in
    // notify.nf — same reason the layout map does: this file owns the order steps
    // run in.
    //
    // The done channels are MIXED into one and notify_step is invoked ONCE: a
    // process can only be invoked a single time per workflow context, so one
    // notify_step(...) call per step would not compile.
    //
    // Nothing here reads a position count. It is the same for every step, so
    // saying it six times adds nothing; the run-start message reports it once.
    // That also removes a trap: assemble_wf's output carries a single path
    // String rather than the collected position list, and `('a/b.zarr' as List)`
    // explodes into characters.
    // ONE list of the steps this run actually performed, in order, each with the
    // channel that says it finished and the artifact it produced. Both the
    // run-start announcement and the per-step messages read it, so they cannot
    // disagree about what ran — and the "i/n" counters are positions in it rather
    // than hard-coded numbers, which is what lets a skipped step renumber the
    // messages without renumbering any directory.
    //
    // QC contributes ONE entry, not one per store: the report is a single task
    // gated on every store's finalize, so its completion is exactly "QC of all
    // stores is done". Its report directory stands in for an output zarr.
    def step_events = [
        [label: 'flat-field',           done: ff_done.done,            output: ff_output],
        [label: 'deskew',               done: deskew_done.done,        output: deskew_output],
        [label: 'phase reconstruction', done: reconstruct_done.done,   output: reconstruct_output],
        [label: 'virtual staining',     done: virtual_stain_done.done, output: virtual_stain_output],
    ]
    if (assemble_on) step_events << [label: 'assemble', done: assemble_done.done, output: assemble_output]
    if (track_on)    step_events << [label: 'track',    done: track_done.done,    output: track_output]
    if (qc_on)       step_events << [label: 'QC',       done: qc_report.done,     output: qc_report_dir]

    steps = step_events.collect { event -> event.label }
    n_steps = steps.size()

    notify_events = channel.empty()
    step_events.eachWithIndex { e, i ->
        notify_events = notify_events.mix( e.done.map { [e.label, e.output, "${i + 1}/${n_steps}"] } )
    }

    notify_step(notify_events, ds)

    // Announce the run once the position count is known. all_positions carries a
    // single collected list, so this fires exactly once; registering the
    // subscribe here rather than earlier is fine because the whole body is graph
    // construction and nothing executes until it finishes.
    all_positions.subscribe { positions ->
        notify_run_start(ds, 'mantis_v2', positions.size(), steps)
    }

    // Report the finished run to Slack, with an @-mention.
    //
    // This already fires AFTER QC and needs no wiring to say so: onComplete runs
    // at session teardown, once every task in the DAG has finished, and the QC
    // tasks are in the DAG like any other. Gating it on a QC channel would be
    // both redundant and wrong — it has to fire on a failed run too, where no QC
    // channel ever emits.
    //
    // onComplete ONLY — onError fires in ADDITION to onComplete, so handling
    // both would double-post every failure. notify_run_end branches on
    // workflow.success instead.
    //
    // Registered inside the workflow body, not at script scope: Nextflow 26's
    // strict syntax rejects statements outside a declaration, the same
    // restriction that forces directory_layout() to be a function. The handler
    // still runs at session teardown, not here.
    //
    // `wf` is captured OUT here on purpose. Inside the handler closure the
    // implicit `workflow` resolves to null, so reading `workflow.stats` there
    // throws an NPE that Nextflow swallows into a bare "Failed to invoke
    // workflow.onComplete event handler" — and the run-end message is silently
    // never sent. The captured reference is the same mutable metadata object, so
    // the stats read at teardown are still the final ones.
    def wf = workflow
    workflow.onComplete {
        notify_run_end(ds, 'mantis_v2', wf, assemble_on ? assemble_output : null)
    }
}
