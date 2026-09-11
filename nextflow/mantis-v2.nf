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
//  Each step's subworkflows (e.g. deskew_init_wf / deskew_run_wf) are
//  path-agnostic and speak only in zarr: this pipeline hands them explicit
//  input_zarr/output_zarr paths. The pipeline itself speaks in `input`/`output`,
//  where `input` may NOT be a zarr (in some pipelines the first step converts
//  raw input to zarr). To reorder steps, change where a step reads from here;
//  the modules stay untouched.
//
//  EVERY STEP IS SPLIT INTO AN INIT AND A RUN SUBWORKFLOW, and this file runs
//  all the inits before any of the runs. See the INIT PHASE comment in the
//  workflow body for why that is possible and what it buys.
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
//   --qc_config       nextflow/configs/<family>/qc.yaml        (assembled store)
//   --qc_track_config nextflow/configs/<family>/qc_track.yaml  (tracking store)
// One self-contained file per store, beside that family's other step configs.
params.qc_config = null
params.qc_track_config = null

include { collect_positions; dataset_name; check_environment } from './modules/common'
include { flat_field_init_wf; flat_field_run_wf } from './modules/flat_field'
include { deskew_init_wf; deskew_run_wf } from './modules/deskew'
include { reconstruct_init_wf; reconstruct_tf_wf; reconstruct_run_wf } from './modules/reconstruct'
include { virtual_stain_init_wf; virtual_stain_run_wf } from './modules/virtual_stain'
include { assemble_init_wf; assemble_run_wf } from './modules/assembly'
include { track_init_wf; track_run_wf } from './modules/tracking'
include { qc_plan_wf; qc_compute_wf; qc_report_wf; qc_report_spec } from './modules/qc'
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

    // ----- Where each step reads and writes ---------------------------------
    // Resolved up front, because BOTH phases below need them and the init phase
    // wires steps together by path before any of them has run. The pipeline
    // input is already a zarr, so flat-field reads it directly. When a convert
    // step is added ahead of flat-field, point ff_input at the convert output
    // instead — the step modules don't care where their input comes from.
    ff_input             = params.input
    ff_output            = "${out}/${layout.flat_field}/${ds}.zarr"
    deskew_output        = "${out}/${layout.deskew}/${ds}.zarr"
    reconstruct_output   = "${out}/${layout.reconstruct}/${ds}.zarr"
    virtual_stain_output = "${out}/${layout.virtual_stain}/${ds}.zarr"
    assemble_output      = assemble_on ? "${out}/${layout.assemble}/${ds}.zarr" : null
    track_output         = track_on    ? "${out}/${layout.track}/${ds}.zarr"    : null

    // ========================================================================
    //  INIT PHASE — every config parsed, every output store scaffolded, before
    //  any compute is submitted.
    //
    //  WHY THIS IS POSSIBLE. Every `--init` is metadata-only: it reads the first
    //  input position's channel names, shape and scale, derives the output
    //  geometry from the config, creates the output plate (create_empty_plate is
    //  idempotent) and prints the RESOURCES line. Nothing reads a pixel. So a
    //  step's init needs its input store to EXIST, not to be FILLED — and the
    //  store that has to exist is the one the previous step's init just created.
    //  That is why the chain below is serial: flat-field's init makes the plate
    //  deskew's init reads, and so on down to tracking, whose init reads the
    //  assembled plate that `concatenate --init` scaffolded.
    //
    //  WHY IT IS WORTH IT. Before this, a config was not parsed until the step
    //  that consumes it was about to run, so a typo in track.yml surfaced seven
    //  hours in, with every reconstruction step already paid for. Now the whole
    //  chain runs on the head node in the run's first minutes, and it is real
    //  validation rather than a schema check: concatenate resolves its channel
    //  mapping against all three source stores, virtual-stain validates its
    //  predict config against VisCy's schema, and every settings model is
    //  constructed with `extra="forbid"`.
    //
    //  DO NOT PARALLELISE THIS. Each init depends on the plate the one before it
    //  created; running them concurrently would race on stores that do not exist
    //  yet. It is cheap precisely because it is metadata-only — seconds per step
    //  — so there is nothing to win.
    //
    //  Provenance depends on this ordering too. `create_empty_plate` inherits a
    //  source store's `biahub-*` zattrs at plate-creation time and only for
    //  positions it creates, so each step must stamp its OWN record during init
    //  (iohub's `extra_metadata`) or the chain would be empty by the time the
    //  next plate is made. See stamp-at-init in biahub/utils/ngff.py's callers.
    // ========================================================================
    ff_init = flat_field_init_wf(ff_input, ff_output, params.flat_field_config,
                                 channel.value('start'))
    dk_init = deskew_init_wf(ff_output, deskew_output, params.deskew_config,
                             ff_init.done)
    rc_init = reconstruct_init_wf(deskew_output, reconstruct_output, params.reconstruct_config,
                                  dk_init.done)
    vs_init = virtual_stain_init_wf(reconstruct_output, virtual_stain_output,
                                    params.virtual_stain_config, rc_init.done)

    // Collected so the gate below can name them all. The chain is serial, so
    // waiting on the last one already implies the rest — they are listed anyway
    // because QC planning hangs off the side of the chain rather than extending
    // it, and because a future step inserted here should not have to notice.
    def init_signals = [vs_init.done]

    if (assemble_on) {
        // Assemble reads the deskew, reconstruct and virtual-stain stores. Its
        // init resolves those three paths into the config and creates the
        // assembled plate from their metadata — which is how a concatenate
        // config naming a channel no source store has fails here rather than
        // after virtual staining.
        as_init = assemble_init_wf(deskew_output, reconstruct_output, virtual_stain_output,
                                   assemble_output, params.concatenate_config, vs_init.done)
        init_signals << as_init.done
    }

    if (track_on) {
        // Track reads the ASSEMBLED plate for both of its inputs: assemble
        // already carries the phase and virtual-stain channels (concatenate
        // preserves channel names, so the track config's channel names resolve
        // unchanged), so the plate structure and the image data come from the
        // same store. Its init also warms the shared cellpose weights cache,
        // which is better done here than with N GPU workers racing for it.
        tk_init = track_init_wf(assemble_output, track_output, params.track_config,
                                as_init.done)
        init_signals << tk_init.done
    }

    // QC's planning verbs belong in this phase for the same reason the step
    // inits do — `plan-stage` and `estimate-resources` read the store's
    // structure and validate the config, and read no pixels. Each store's plan
    // is gated on the init that scaffolded it. The tab label is the step
    // directory, which is unique per store by construction.
    def qc_stores = []
    if (qc_on) {
        if (qc_image_on) {
            qc_stores << [label: layout.assemble, zarr: assemble_output,
                          config: params.qc_config, scaffolded: as_init.done]
        }
        if (qc_track_on) {
            qc_stores << [label: layout.track, zarr: track_output,
                          config: params.qc_track_config, scaffolded: tk_init.done]
        }

        // The trigger is mapped, NOT combined. `combine` concatenates the two
        // items, so what the producer emits leaks into the tuple's arity: an
        // init emits one token and gave `[zarr, config, token]`, but a collected
        // producer emits a LIST of every position's output, so the tuple became
        // `[zarr, config, p1, p2, … p30]` and a three-parameter closure could not
        // be spread across it — `MissingMethodException`, after seven hours.
        // Mapping reads nothing out of the trigger, so no producer's payload
        // shape can reach this.
        //
        // Every step's `done` is a VALUE channel by contract (see the step
        // modules' emit blocks), so this is one plan per store however many
        // items the producer itself emitted.
        qc_plan_inputs = channel.empty()
        qc_stores.each { st ->
            qc_plan_inputs = qc_plan_inputs.mix( st.scaffolded.map { tuple(st.zarr, st.config) } )
        }
        qc_plan = qc_plan_wf(qc_plan_inputs)
        init_signals << qc_plan.done
    }

    // ONE gate for the whole compute phase: nothing reaches SLURM until every
    // config in the run has been parsed and every output store exists. Mixed
    // and collected the same way notify_events is below, because a workflow body
    // is graph construction and `init_signals` is a plain Groovy list of
    // channels, not a channel of channels.
    init_gate = channel.empty()
    init_signals.each { signal -> init_gate = init_gate.mix(signal) }
    init_done = init_gate.collect().map { 'done' }

    // ========================================================================
    //  COMPUTE PHASE
    //
    //  The per-step barriers are unchanged: each step waits for every position
    //  of the one before it. Letting a single position flow through the steps
    //  independently is issue #304, and is blocked on two whole-plate steps —
    //  `viscy preprocess` and single-shot `concatenate` — not on this wiring.
    // ========================================================================
    ff_done = flat_field_run_wf(all_positions, ff_input, ff_output,
                                params.flat_field_config, ff_init.resources, init_done)

    deskew_done = deskew_run_wf(all_positions, ff_output, deskew_output,
                                params.deskew_config, dk_init.resources, ff_done.done)

    // The transfer function is built from the deskewed plate's SHAPE alone, so
    // it is gated on init_done rather than on the deskew data. It overlaps
    // flat-field and deskew instead of sitting between them and reconstruction.
    tf = reconstruct_tf_wf(deskew_output, reconstruct_output, params.reconstruct_config,
                           init_done)

    // Phase reconstruction runs on the deskewed output. Which channel is
    // reconstructed is set by `input_channel_names` in the reconstruct config.
    reconstruct_done = reconstruct_run_wf(all_positions, deskew_output, reconstruct_output,
                                          params.reconstruct_config, rc_init.resources,
                                          tf.done, deskew_done.done)

    // Virtual staining runs cytoland (VisCy) prediction on the reconstructed
    // output. Its `viscy preprocess` pass reads every position's pixels and
    // writes the normalization statistics prediction reads back, which is why
    // this step keeps a whole-plate barrier ahead of it.
    virtual_stain_done = virtual_stain_run_wf(all_positions, reconstruct_output,
                                              virtual_stain_output,
                                              params.virtual_stain_config,
                                              vs_init.resources, reconstruct_done.done)

    // Concatenate the deskew, reconstruct and virtual-stain outputs channel-wise
    // into a single multichannel plate. Unlike the per-position steps this runs
    // single-shot on ONE reserved compute node (`concatenate --cluster debug`
    // iterates every position in-process).
    if (assemble_on) {
        assemble_done = assemble_run_wf(assemble_output, params.concatenate_config,
                                        as_init.resources, virtual_stain_done.done)
    }

    // Tracking runs AFTER assemble rather than in parallel with it — the
    // tradeoff for reading one store instead of two is that the whole plate must
    // be assembled first. Two consequences: any Z/Y/X crop or time_indices subset
    // in the concatenate config is what tracking sees, and the intermediate
    // stores are no longer needed once assemble is verified. To go back to the
    // parallel wiring, point the first input at reconstruct_output, the second at
    // virtual_stain_output, and gate on virtual_stain_done.
    if (track_on) {
        track_done = track_run_wf(all_positions, assemble_output, assemble_output,
                                  track_output, params.track_config,
                                  tk_init.resources, assemble_done.done)
    }

    // ----- QC compute -------------------------------------------------------
    // Planned in the init phase; the compute it planned is gated on the step
    // that wrote each store — the ASSEMBLED plate on assemble_done (the same
    // signal track waits on, so image QC runs CONCURRENTLY with tracking), and
    // the tracking store on track_done. Neither extends the critical path ahead
    // of itself.
    //
    // A QC verdict cannot fail the pipeline: `imaging-qc gate` exits 0 whether
    // positions pass or fail, recording the verdict in each store's own
    // `tables/qc/` tables and a QC_SUMMARY line. Only a broken config or a
    // genuine compute error exits non-zero, which is what should stop a run.
    qc_report_dir = params.qc_report_dir ?: "${out}/qc/report"

    if (qc_on) {
        // Written at launch, before any QC task runs: everything in it is known
        // from the layout above, so a malformed spec fails now rather than after
        // hours of compute.
        def spec = qc_report_spec(qc_stores, "${out}/qc/report_spec.yaml", "QC — ${ds}")

        // Keyed [zarr, config] to match what qc_plan_wf emitted, so each store's
        // planned work is released by its own producer and no other. Mapped, not
        // combined, for the reason spelled out at qc_plan_inputs above.
        qc_compute_ready = channel.empty()
        if (qc_image_on) {
            qc_compute_ready = qc_compute_ready.mix(
                assemble_done.done.map { tuple(assemble_output, params.qc_config) } )
        }
        if (qc_track_on) {
            qc_compute_ready = qc_compute_ready.mix(
                track_done.done.map { tuple(track_output, params.qc_track_config) } )
        }

        qc = qc_compute_wf(qc_plan.items, qc_plan.stores, qc_compute_ready)
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
    // That also removes a trap: assemble's output carries a single path String
    // rather than the collected position list, and `('a/b.zarr' as List)`
    // explodes into characters.
    //
    // ONE list of the steps this run actually performed, in order, each with the
    // channel that says it finished and the artifact it produced. Both the
    // run-start announcement and the per-step messages read it, so they cannot
    // disagree about what ran — and the "i/n" counters are positions in it rather
    // than hard-coded numbers, which is what lets a skipped step renumber the
    // messages without renumbering any directory.
    //
    // The init phase leads the list. It is not a reconstruction step, but it is
    // the first thing that can fail and the first thing that finishing means
    // something: every config is good and every output store exists. Its
    // "artifact" is the run directory.
    //
    // QC contributes ONE entry, not one per store: the report is a single task
    // gated on every store's finalize, so its completion is exactly "QC of all
    // stores is done". Its report directory stands in for an output zarr.
    def step_events = [
        [label: 'init (configs validated)', done: init_done,                 output: out],
        [label: 'flat-field',               done: ff_done.done,              output: ff_output],
        [label: 'deskew',                   done: deskew_done.done,          output: deskew_output],
        [label: 'phase reconstruction',     done: reconstruct_done.done,     output: reconstruct_output],
        [label: 'virtual staining',         done: virtual_stain_done.done,   output: virtual_stain_output],
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
