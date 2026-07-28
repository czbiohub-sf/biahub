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
//  Flat-field → deskew → reconstruct → virtual-stain is wired today. The
//  remaining steps (track, assemble) arrive with their own PRs — follow the
//  chaining below for the pattern.
// ---------------------------------------------------------------------------

params.input = null   // raw source — may not be a zarr store
params.output = null   // output directory for all step zarrs
params.deskew_config = null
params.flat_field_config = null
params.reconstruct_config = null
params.virtual_stain_config = null
params.concatenate_config = null
params.biahub_project = null
params.max_positions = 0

include { collect_positions; dataset_name } from './modules/common'
include { deskew_wf } from './modules/deskew'
include { flat_field_wf } from './modules/flat_field'
include { reconstruct_wf } from './modules/reconstruct'
include { virtual_stain_wf } from './modules/virtual_stain'
include { assemble_wf } from './modules/assembly'

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
        // track         : '4-track',
        assemble      : '5-assemble',
    ]
}


workflow {
    if (!params.input)              error "Provide --input"
    if (!params.output)             error "Provide --output"
    if (!params.flat_field_config)  error "Provide --flat_field_config"
    if (!params.deskew_config)      error "Provide --deskew_config"
    if (!params.reconstruct_config) error "Provide --reconstruct_config"
    if (!params.virtual_stain_config) error "Provide --virtual_stain_config"
    if (!params.concatenate_config) error "Provide --concatenate_config"

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

    assemble_wf(
        deskew_output,
        reconstruct_output,
        virtual_stain_output,
        assemble_output,
        params.concatenate_config,
        assemble_trigger
    )
}
