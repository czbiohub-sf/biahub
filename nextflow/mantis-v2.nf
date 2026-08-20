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
//  Flat-field → deskew → reconstruct → virtual-stain → assemble → track is
//  wired today: assemble concatenates the deskew/reconstruct/virtual-stain
//  channels into one plate, and track reads that assembled plate as its single
//  input. Follow the chaining below for the pattern.
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
// Configs are grouped one directory per step; see qc_report_spec() for why.
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
// workflow body, not here (input may not even be a zarr). A Dragonfly pipeline
// would define its own map; reordering or renaming a step is a one-line edit.
//
// Defined as a function rather than a bare top-level assignment: Nextflow's DSL2
// parser only allows declarations (include/process/workflow/function) at script
// scope, so a `DIRECTORY_LAYOUT = [...]` statement fails to compile. The workflow
// body calls directory_layout() once to get the map.
def directory_layout() {
    return [
        // convert    : '0-convert',     // first step when raw input isn't zarr
        flat_field    : '0-flatfield',
        deskew        : '1-deskew',
        reconstruct   : '2-reconstruct',
        virtual_stain : '3-virtual-stain',
        track         : '4-track',
        assemble      : '5-assemble',
    ]
}


// The reconstruction steps, in the order they run, as they are named in Slack.
// One list so the run-start announcement cannot disagree with the per-step
// messages. A function rather than a bare assignment for the same reason
// directory_layout() is one: Nextflow's DSL2 parser allows only declarations at
// script scope.
def reconstruction_steps(with_qc) {
    def steps = [
        'flat-field',
        'deskew',
        'phase reconstruction',
        'virtual staining',
        'assemble',
        'track',
    ]
    // QC is a step like any other when it is on, and absent when it is not, so
    // the announced list and the per-step counters agree either way (N/7 or N/6).
    return with_qc ? steps + ['QC'] : steps
}


workflow {
    if (!params.input)              error "Provide --input"
    if (!params.output)             error "Provide --output"
    if (!params.flat_field_config)  error "Provide --flat_field_config"
    if (!params.deskew_config)      error "Provide --deskew_config"
    if (!params.reconstruct_config) error "Provide --reconstruct_config"
    if (!params.virtual_stain_config) error "Provide --virtual_stain_config"
    if (!params.track_config)       error "Provide --track_config"
    if (!params.concatenate_config) error "Provide --concatenate_config"

    // Tasks call `biahub`/`viscy`/`imaging-qc` bare, so fail now if the env isn't
    // activated. `imaging-qc` is only required when QC is actually wired in.
    def qc_on = (params.qc_config ?: params.qc_track_config) as boolean
    check_environment(qc_on ? ['biahub', 'viscy', 'imaging-qc'] : ['biahub', 'viscy'])

    // Resolved here rather than beside the notification wiring below because the
    // QC block needs its own label and the final index. Still one list, read by
    // the run-start announcement and by every per-step message.
    steps = reconstruction_steps(qc_on)
    n_steps = steps.size()

    def ds     = dataset_name()
    def out    = params.output
    def layout = directory_layout()

    collect_positions(params.input)
    all_positions = collect_positions.out

    // ----- Flat-field -------------------------------------------------------
    // The pipeline input is already a zarr, so flat-field reads it directly and
    // writes the flat-field step directory. When a convert step is added ahead
    // of flat-field, point ff_input at the convert output instead — flat_field_wf
    // doesn't care where its input comes from.
    ff_trigger = Channel.value(true)
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
    assemble_trigger = virtual_stain_done.done
    assemble_output  = "${out}/${layout.assemble}/${ds}.zarr"

    assemble_done = assemble_wf(
        deskew_output,
        reconstruct_output,
        virtual_stain_output,
        assemble_output,
        params.concatenate_config,
        assemble_trigger
    )

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
    track_trigger      = assemble_done.done
    track_input        = assemble_output
    track_input_images = assemble_output
    track_output       = "${out}/${layout.track}/${ds}.zarr"

    track_done = track_wf(all_positions, track_input, track_input_images, track_output, params.track_config, track_trigger)

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
    // Empty when QC is off, so the mix below is unconditional either way.
    qc_notify = Channel.empty()

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
        def report_dir = params.qc_report_dir ?: "${out}/qc/report"
        def spec = qc_report_spec(qc_stores, "${out}/qc/report_spec.yaml", "QC — ${ds}")

        // Each store waits only on its own producer, so they QC independently.
        qc_inputs = Channel.empty()
        qc_stores.each { st ->
            def one = Channel.of(tuple(st.zarr, st.config))
                .combine(st.trigger)
                .map { z, cfg, done -> tuple(z, cfg) }
            qc_inputs = qc_inputs.mix(one)
        }

        qc = qc_stage_wf(qc_inputs)
        qc_report = qc_report_wf(qc.done, spec, report_dir)

        // ONE message for the whole QC step, not one per store: the report is a
        // single task gated on every store's finalize, so its completion is
        // exactly "QC of all stores is done". The report directory is the
        // artifact worth naming, so it stands in for the step's output zarr.
        qc_notify = qc_report.done.map { rd -> [steps[-1], rd, "${n_steps}/${n_steps}"] }
    }

    // ----- Notifications ----------------------------------------------------
    // One Slack message as each step finishes, plus the run-start announcement.
    // Step ORDER and labels live here with the rest of the wiring, not in
    // notify.nf — same reason the layout map does: this file owns the order steps
    // run in. reconstruction_steps() is the only list of labels, so the announced
    // list cannot drift from what the per-step messages actually say.
    //
    // The six done channels are MIXED into one and notify_step is invoked ONCE:
    // a process can only be invoked a single time per workflow context, so six
    // separate notify_step(...) calls would not compile.
    //
    // Nothing here reads a position count. It is the same for every step, so
    // saying it six times adds nothing; the run-start message reports it once.
    // That also removes a trap: assemble_wf's `done` carries a single path
    // String rather than the collected position list, and `('a/b.zarr' as List)`
    // explodes into characters.
    notify_events = ff_done.done      .map { [steps[0], ff_output,            "1/${n_steps}"] }
        .mix( deskew_done.done        .map { [steps[1], deskew_output,        "2/${n_steps}"] } )
        .mix( reconstruct_done.done   .map { [steps[2], reconstruct_output,   "3/${n_steps}"] } )
        .mix( virtual_stain_done.done .map { [steps[3], virtual_stain_output, "4/${n_steps}"] } )
        .mix( assemble_done.done      .map { [steps[4], assemble_output,      "5/${n_steps}"] } )
        .mix( track_done.done         .map { [steps[5], track_output,         "6/${n_steps}"] } )
        .mix( qc_notify )

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
        notify_run_end(ds, 'mantis_v2', wf)
    }
}
