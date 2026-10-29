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
//  Flat-field → deskew → reconstruct → virtual-stain → assemble → project →
//  segment → track → pyramid → QC → DCA finalize is the full chain: assemble
//  concatenates the deskew/reconstruct/virtual-stain channels into one plate,
//  project reduces that 3D plate to the 2D (Z=1) plate 2D segmentation needs,
//  segment writes cellpose labels into it, track links those labels and writes
//  the tracked labels beside them, pyramid adds the coarser resolution levels
//  to the assembled and projected plates once nothing else writes pixels or
//  labels into them, QC reads the finished stores, and DCA finalize writes the
//  DCA metadata onto the assembled plate from its QC statistics. Follow the
//  chaining below for the pattern.
//
//  Assemble, project, segment, track, QC and DCA finalize are OPTIONAL and
//  selected by whether their config is given, so a run performs the prefix it
//  asks for: A549 wants assemble + project + segment + track + QC, a neuromast
//  run wants assemble + QC and no segmentation or tracking (issue #306).
//  Pyramids are on by default for whichever of the two plates exist
//  (--pyramid_levels 1 turns them off). A skipped step never renumbers the
//  directories of the ones around it.
// ---------------------------------------------------------------------------

params.input = null   // raw source — may not be a zarr store
params.output = null   // output directory for all step zarrs
params.deskew_config = null
params.flat_field_config = null
params.reconstruct_config = null
params.virtual_stain_config = null
params.project_config = null
params.segment_config = null
params.track_config = null
params.concatenate_config = null
params.max_positions = 0
// QC, off by default. Each param points at a stage config for one store; set
// either, both, or neither. Both QC'd stores become tabs of ONE report.
//   --qc_config         nextflow/configs/<family>/qc.yaml          (assembled store)
//   --qc_project_config nextflow/configs/<family>/qc_project.yaml  (projected plate,
//                       which holds the labels segment and track write; counts
//                       the tracked ones)
// One self-contained file per store, beside that family's other step configs.
params.qc_config = null
params.qc_project_config = null
params.qc_track_config = null  // removed; refused below
// DCA finalize, off by default: --dca_config nextflow/configs/<family>/dca_finalize.yml
// writes the DCA metadata onto the assembled store and validates it, after QC.
params.dca_config = null
// Resolution levels of the assembled and projected plates, counting level 0:
// 3 is the DCA v0.2 target (issue #359) and adds a 2x and a 4x level, in Y and X
// (and Z when the plate has more than one plane), to every image and label
// image. 1 turns the step off. Images are mean-pooled, labels max-pooled.
params.pyramid_levels = 3

include { collect_positions; dataset_name; check_environment } from './modules/common'
include { flat_field_init_wf; flat_field_run_wf } from './modules/flat_field'
include { deskew_init_wf; deskew_run_wf } from './modules/deskew'
include { reconstruct_init_wf; reconstruct_tf_wf; reconstruct_run_wf } from './modules/reconstruct'
include { virtual_stain_init_wf; virtual_stain_run_wf } from './modules/virtual_stain'
include { assemble_init_wf; assemble_run_wf } from './modules/assembly'
include { project_init_wf; project_run_wf } from './modules/project'
include { segment_init_wf; segment_run_wf } from './modules/segment'
include { track_init_wf; track_run_wf } from './modules/tracking'
include { qc_plan_wf; qc_compute_wf; qc_report_wf; qc_report_spec } from './modules/qc'
include { dca_finalize_wf } from './modules/dca'
// One module, two plates: a process can only be invoked once per workflow
// context, so the assembled and projected plates get their own aliases.
include { pyramid_init_wf as pyramid_assembled_init_wf;
          pyramid_run_wf as pyramid_assembled_run_wf } from './modules/pyramid'
include { pyramid_init_wf as pyramid_projected_init_wf;
          pyramid_run_wf as pyramid_projected_run_wf } from './modules/pyramid'
include { notify_step; notify_run_start; notify_run_end } from './modules/notify'
include { cleanup_decision; cleanup_targets; cleanup_intermediates_wf } from './modules/cleanup'

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
        // project writes beside the assembled plate (<assemble>/<ds>_proj.zarr),
        // and segment, track and pyramid write into the plates they read; none has a
        // directory of its own or takes a number.
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
    // among the steps that write a directory, so A549 and neuromast runs both
    // end at `4-assemble`.
    //
    // Project turns the assembled 3D plate into the 2D (Z=1) plate that 2D
    // segmentation and tracking read: neither `biahub segment` nor `biahub track`
    // selects or projects Z-planes, so an A549 run names --project_config as
    // well as --segment_config and --track_config. Without it they read the
    // assembled plate itself, which suits 3D work. The projected plate is a
    // companion of the assembled one and lives beside it as
    // `<N>-assemble/<ds>_proj.zarr`; segment writes its label images into the
    // plate it reads and track writes the tracked labels and tracks beside
    // them. None of the three has a step directory.
    def assemble_on   = params.concatenate_config as boolean
    def project_on    = params.project_config as boolean
    def segment_on    = params.segment_config as boolean
    def track_on      = params.track_config as boolean
    def qc_image_on   = params.qc_config as boolean
    def qc_project_on = params.qc_project_config as boolean
    def qc_on         = qc_image_on || qc_project_on
    def dca_on        = params.dca_config as boolean
    // Pyramids go on the deliverable plates, so without assemble there is
    // nothing to pyramid and the default is silently moot rather than an error.
    def pyramid_levels = params.pyramid_levels as int
    if (pyramid_levels < 1) {
        error "--pyramid_levels must be at least 1 (1 = no pyramid), got ${params.pyramid_levels}."
    }
    def pyramid_on    = pyramid_levels > 1 && assemble_on

    // A step cannot outlive the step whose output it reads. Refuse the
    // combination at launch, naming the config to add or the one to drop, rather
    // than failing hours in with a missing store.
    if (project_on && !assemble_on) {
        error "--project_config needs --concatenate_config: project reads the assembled plate."
    }
    if (segment_on && !assemble_on) {
        error "--segment_config needs --concatenate_config: segment reads the assembled (or projected) plate."
    }
    if (track_on && !assemble_on) {
        error "--track_config needs --concatenate_config: tracking reads the assembled (or projected) plate."
    }
    // Track's `labels:` must name a label image segment wrote; `biahub track
    // --init` refuses a plate without it, so a run naming --track_config but
    // not --segment_config fails in the init phase with that message.
    if (qc_image_on && !assemble_on) {
        error "--qc_config needs --concatenate_config: it QCs the assembled store."
    }
    if (qc_project_on && !project_on) {
        error "--qc_project_config needs --project_config: it QCs the projected plate."
    }
    if (params.qc_track_config) {
        error "--qc_track_config was removed: tracking writes its labels into the plate it " +
              "reads, and --qc_project_config counts them there " +
              "(see nextflow/configs/a549/qc_project.yaml)."
    }
    if (dca_on && !qc_image_on) {
        error "--dca_config needs --qc_config: finalize reads the QC stage-5 statistics."
    }

    // Whether this run deletes its intermediates once it finishes. Resolved now
    // from --cleanup_intermediates and concatenate.yml (see cleanup_decision),
    // so `true` without assemble, or a bad value, fails before anything runs.
    // When on, also switch on Nextflow's own `cleanup`, which empties the work
    // directory at the end of a SUCCESSFUL run: with the intermediate stores
    // gone, every cached task would point at deleted data. This is set here
    // rather than in nextflow.config because `auto` is only resolved here.
    def cleanup_plan = cleanup_decision(params.cleanup_intermediates, params.concatenate_config)
    def cleanup_on   = cleanup_plan.on
    log.info "cleanup_intermediates: ${cleanup_on ? 'on' : 'off'} (${cleanup_plan.reason})"
    if (cleanup_on) {
        workflow.session.config.cleanup = true
    }

    // Tasks call `biahub`/`viscy`/`imaging-qc`/`dca-validate` bare, so fail now if
    // the env isn't activated. `imaging-qc` and `dca-validate` are only required
    // when their step is actually wired in.
    def tools = ['biahub', 'viscy']
    if (qc_on)  tools << 'imaging-qc'
    if (dca_on) tools << 'dca-validate'
    check_environment(tools)

    def ds  = dataset_name()
    def out = params.output

    // The steps this run performs, in execution order — the list the directory
    // numbering is derived from. Reconstruction proper is always in it.
    def performed = ['flat_field', 'deskew', 'reconstruct', 'virtual_stain']
    if (assemble_on) performed << 'assemble'
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
    // Beside the assembled plate, with a `_proj` suffix: the 2D companion of
    // that store, not a step of its own.
    project_output       = project_on  ? "${out}/${layout.assemble}/${ds}_proj.zarr" : null
    // What segment and track read AND write into (labels/, tracks.geff, the
    // tracks CSV): the projected (Z=1) plate when project runs, else the
    // assembled plate. Both the plate structure and the pixels come from it.
    track_input          = project_on  ? project_output : assemble_output

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
        // init takes those three stores as `-i` groups and creates the
        // assembled plate from their metadata — which is how a concatenate
        // config naming a channel no source store has fails here rather than
        // after virtual staining.
        as_init = assemble_init_wf(deskew_output, reconstruct_output, virtual_stain_output,
                                   assemble_output, params.concatenate_config, vs_init.done)
        init_signals << as_init.done
    }

    if (project_on) {
        // Project reads the ASSEMBLED plate (the phase channel for focus finding,
        // the virtual-stain channels to project) and scaffolds the Z=1 plate from
        // its metadata, so a project config naming a channel the assembled plate
        // lacks fails here.
        pj_init = project_init_wf(assemble_output, project_output, params.project_config,
                                  as_init.done)
        init_signals << pj_init.done
    }

    // The plate segment and track work in is scaffolded by project's init when
    // project runs, else by assemble's.
    def plate_scaffolded = project_on ? pj_init.done : as_init.done

    if (segment_on) {
        // Segment reads the projected plate (or the assembled one): its init
        // checks that the models' channels exist and that their dimensionality
        // matches the plate's Z, creates the empty label image per model in every
        // position, and warms the shared cellpose weights cache, which is better
        // done here than with N GPU workers racing for it.
        sg_init = segment_init_wf(track_input, params.segment_config, plate_scaffolded)
        init_signals << sg_init.done
    }

    if (track_on) {
        // Track reads ONE plate for both of its inputs (plate structure and the
        // label image to track; image data for a legacy foreground+contour
        // config): the projected plate when project runs, else the assembled
        // plate. Its init checks that the plate's Z matches output_mode and that
        // the label image `labels:` names exists — segment's init created it,
        // which is why this init follows that one — and creates the empty
        // tracked label image in every position.
        tk_init = track_init_wf(track_input, params.track_config,
                                segment_on ? sg_init.done : plate_scaffolded)
        init_signals << tk_init.done
    }

    if (pyramid_on) {
        // Pyramid's init scaffolds nothing: it checks the plate's first position
        // against the requested depth and sizes the task. Gated on the init that
        // scaffolded each plate; the label images tracking adds later get their
        // levels at compute time, from the same depth.
        pya_init = pyramid_assembled_init_wf(assemble_output, pyramid_levels, as_init.done)
        init_signals << pya_init.done
        if (project_on) {
            pyp_init = pyramid_projected_init_wf(project_output, pyramid_levels, pj_init.done)
            init_signals << pyp_init.done
        }
    }

    // QC's planning verbs belong in this phase for the same reason the step
    // inits do — `plan-stage` and `estimate-resources` read the store's
    // structure and validate the config, and read no pixels. Each store's plan
    // is gated on the init that scaffolded what it reads: for the projected
    // plate that is the last of project's, segment's and track's inits that
    // runs, because a config counting instances in `labels/` is refused at plan
    // time until the label image exists. The tab label is the step directory,
    // plus the store name for the projected plate, which shares the assemble
    // directory.
    def qc_stores = []
    if (qc_on) {
        if (qc_image_on) {
            qc_stores << [label: layout.assemble, zarr: assemble_output,
                          config: params.qc_config, scaffolded: as_init.done]
        }
        if (qc_project_on) {
            def proj_scaffolded = track_on ? tk_init.done : (segment_on ? sg_init.done : pj_init.done)
            qc_stores << [label: "${layout.assemble}/${ds}_proj", zarr: project_output,
                          config: params.qc_project_config, scaffolded: proj_scaffolded]
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
    //  independently is issue #304, and is blocked on one whole-plate step —
    //  `viscy preprocess` — not on this wiring.
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
    // into a single multichannel plate, one task per position like every other
    // step (one `-i` per source store; see modules/assembly.nf).
    if (assemble_on) {
        assemble_done = assemble_run_wf(all_positions, deskew_output, reconstruct_output,
                                        virtual_stain_output, assemble_output,
                                        params.concatenate_config, as_init.resources,
                                        virtual_stain_done.done)
    }

    // Projection reads the assembled plate, one task per position, and writes the
    // 2D plate segmentation and tracking read.
    if (project_on) {
        project_done = project_run_wf(all_positions, assemble_output, project_output,
                                      params.project_config, pj_init.resources,
                                      assemble_done.done)
    }
    def plate_written = project_on ? project_done.done : assemble_done.done

    // Segmentation (cellpose, GPU) writes its label images into the plate, one
    // task per position, once that plate holds data.
    if (segment_on) {
        segment_done = segment_run_wf(all_positions, track_input, params.segment_config,
                                      sg_init.resources, plate_written)
    }

    // Tracking runs AFTER assemble, project and segment rather than in parallel
    // with them — the tradeoff for reading one store instead of two is that the
    // whole plate must be assembled first. Two consequences: any Z/Y/X crop or
    // time_indices subset in the concatenate config is what tracking sees, and
    // the intermediate stores are no longer needed once assemble is verified. To
    // go back to the parallel wiring, point the first input at
    // reconstruct_output, the second at virtual_stain_output, and gate on
    // virtual_stain_done.
    if (track_on) {
        track_done = track_run_wf(all_positions, track_input, track_input,
                                  params.track_config, tk_init.resources,
                                  segment_on ? segment_done.done : plate_written)
    }

    // ----- Pyramids ---------------------------------------------------------
    // After the LAST step that writes pixels or labels into each plate: the
    // assembled plate's tracking when it writes there (track without project),
    // else assemble; the projected plate's tracking, else project. `biahub
    // pyramid` rewrites a position's multiscales metadata and adds its own
    // record to the position's zarr.json, so it must not overlap another writer
    // of that file, and QC (which stamps the tables it wrote into the same
    // attrs) is held until it has finished. Level 0 is never touched.
    // The last step that writes into each plate: segment and track write into
    // the projected plate when project runs, else into the assembled one.
    def labels_written = track_on ? track_done.done : (segment_on ? segment_done.done : null)
    def assembled_written = (labels_written != null && !project_on) ? labels_written : assemble_done.done
    def projected_written = project_on ? (labels_written ?: project_done.done) : null
    if (pyramid_on) {
        pyramid_assembled_done = pyramid_assembled_run_wf(all_positions, assemble_output,
                                                          pyramid_levels, pya_init.resources,
                                                          assembled_written)
        if (project_on) {
            pyramid_projected_done = pyramid_projected_run_wf(all_positions, project_output,
                                                              pyramid_levels, pyp_init.resources,
                                                              projected_written)
        }
    }

    // ----- QC compute -------------------------------------------------------
    // The last step except DCA finalize. Planned in the init phase; the compute it planned is
    // gated on the last writer of each store: its pyramid when pyramids are on,
    // else the last of segment and track where they write into that store (the
    // projected plate, or the assembled one without project), else the step
    // that created it. QC used to run concurrently with projection, segmentation
    // and tracking; it now waits, so that imaging-qc's stamps on the position
    // attrs never race another writer of the same zarr.json and the report
    // describes the finished plates.
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
            def assembled_final = pyramid_on ? pyramid_assembled_done.done : assembled_written
            qc_compute_ready = qc_compute_ready.mix(
                assembled_final.map { tuple(assemble_output, params.qc_config) } )
        }
        if (qc_project_on) {
            def projected_final = pyramid_on ? pyramid_projected_done.done : projected_written
            qc_compute_ready = qc_compute_ready.mix(
                projected_final.map { tuple(project_output, params.qc_project_config) } )
        }

        qc = qc_compute_wf(qc_plan.items, qc_plan.stores, qc_compute_ready)
        qc_report = qc_report_wf(qc.done, spec, qc_report_dir)
    }

    // ----- DCA finalize -----------------------------------------------------
    // After QC, which itself waits for the assembled plate's last writer (its
    // pyramid, or segment/track when they write there), so the QC report is the
    // one signal that nothing else will write the plate and its stage tables are
    // final. Finalize rewrites every position's zarr.json and reads the assembled
    // store's QC statistics. The validator's report lands next to the plate as
    // <stem>.dca-validate.json.
    if (dca_on) {
        dca = dca_finalize_wf([assemble_output], params.dca_config, qc_report.done)
    }

    // ----- Cleanup of intermediates (--cleanup_intermediates) ---------------
    // Once the LAST step has finished, delete what only existed to feed the
    // assembled plate: the flat-field, deskew, reconstruct and virtual-stain step
    // directories (store, slurm_output/ and .iohub-progress/ alike). The
    // assembled and projected plates (with the labels and tracks inside) are DELIVERABLES and
    // are never targets. Beside them, delete only what served the run:
    // slurm_output/ (submitit placeholders and a pointer README; the real logs
    // are in nextflow/slurm_output/) and the assembled and projected stores'
    // .iohub-progress/ resume markers, which left behind would make concatenate
    // or project skip every write unit of a later run into this directory,
    // keeping data built from the old intermediates. Segment's resume markers
    // (for its label images) live under the projected plate's entry in the same
    // .iohub-progress/; tracking writes none.
    //
    // What was removed is appended to nextflow/intermediates_cleaned.txt rather
    // than logged. Nextflow's `cleanup`, switched on above, empties the work
    // directory at the end of the same successful run, so a cleaned run is
    // final: rerunning it recomputes everything. See modules/cleanup.nf.
    //
    // The gate is every final step's `done`, so QC — when it runs — must have
    // FINISHED, whatever its verdict: `imaging-qc gate` exits 0 on a failing
    // verdict. A step that fails terminates the run, so none of these emit.
    // The targets are checked now, at launch, not when the gate opens.
    if (cleanup_on) {
        def cleanup_paths = [ff_output, deskew_output, reconstruct_output, virtual_stain_output]
            .collect { zarr -> new File(zarr).parent }
        def assemble_dir = new File(assemble_output).parent
        // The projected plate shares the assemble directory, so these two paths
        // cover its slurm_output/ and .iohub-progress/ as well.
        cleanup_paths << "${assemble_dir}/slurm_output" << "${assemble_dir}/.iohub-progress"
        def cleanup_list = cleanup_targets(cleanup_paths, out)
        cleanup_record = "${out}/nextflow/intermediates_cleaned.txt"

        def final_signals = [assemble_done.done]
        if (project_on) final_signals << project_done.done
        if (segment_on) final_signals << segment_done.done
        if (track_on) final_signals << track_done.done
        if (pyramid_on) {
            final_signals << pyramid_assembled_done.done
            if (project_on) final_signals << pyramid_projected_done.done
        }
        if (qc_on)    final_signals << qc_report.done
        if (dca_on)   final_signals << dca.done
        final_gate = channel.empty()
        final_signals.each { signal -> final_gate = final_gate.mix(signal) }

        cleanup_run = cleanup_intermediates_wf(cleanup_list, out, cleanup_record,
                                               "on (${cleanup_plan.reason})", final_gate)
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
        [label: 'initialize',           done: init_done,                 output: out],
        [label: 'flat-field',           done: ff_done.done,              output: ff_output],
        [label: 'deskew',               done: deskew_done.done,          output: deskew_output],
        [label: 'phase reconstruction', done: reconstruct_done.done,     output: reconstruct_output],
        [label: 'virtual staining',     done: virtual_stain_done.done,   output: virtual_stain_output],
    ]
    if (assemble_on) step_events << [label: 'assemble', done: assemble_done.done, output: assemble_output]
    if (project_on)  step_events << [label: 'project',  done: project_done.done,  output: project_output]
    if (segment_on)  step_events << [label: 'segment',  done: segment_done.done,  output: track_input]
    if (track_on)    step_events << [label: 'track',    done: track_done.done,    output: track_input]
    if (pyramid_on) {
        step_events << [label: 'pyramid (assembled)', done: pyramid_assembled_done.done, output: assemble_output]
        if (project_on) {
            step_events << [label: 'pyramid (projected)', done: pyramid_projected_done.done, output: project_output]
        }
    }
    if (qc_on)       step_events << [label: 'QC',       done: qc_report.done,     output: qc_report_dir]
    if (dca_on)      step_events << [label: 'DCA finalize', done: dca.done,       output: assemble_output]
    if (cleanup_on)  step_events << [label: 'cleanup intermediates', done: cleanup_run.done, output: cleanup_record]

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
        notify_run_start(ds, 'mantis_v2', positions.size(), steps,
                         "${cleanup_on ? 'on' : 'off'} (${cleanup_plan.reason})")
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
