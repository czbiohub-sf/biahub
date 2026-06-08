#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// ---------------------------------------------------------------------------
//  mantis-v2 timelapse reconstruction pipeline.
//
//  This file is the ORCHESTRATION layer. It owns two things the step modules
//  must not know about:
//    1. the directory LAYOUT (the DIRECTORY_LAYOUT map below), and
//    2. the ORDER steps run in and what each step reads/writes.
//
//  Each step's subworkflow (e.g. deskew_wf) is path-agnostic and speaks only in
//  zarr: this pipeline hands it explicit input_zarr/output_zarr paths. The
//  pipeline itself speaks in `input`/`output`, where `input` may NOT be a zarr
//  (in some pipelines the first step converts raw input to zarr). To reorder
//  steps, change where a step reads from here; the modules stay untouched.
//
//  Only deskew is wired today. Other steps (flat-field, reconstruct,
//  virtual-stain, track, assemble) arrive with their own PRs — see the
//  commented flat-field example below for the pattern to follow.
// ---------------------------------------------------------------------------

params.input             = null   // raw source — may not be a zarr store
params.output            = null   // output directory for all step zarrs
params.deskew_config     = null
params.flat_field_config = null   // reserved for the upcoming flat-field step
params.biahub_project    = null
params.max_positions     = 0

include { collect_positions; dataset_name } from './modules/common'
include { deskew_wf } from './modules/deskew'
// include { flat_field_wf } from './modules/flat_field'   // upcoming — PR #257

// Output directory layout for the reconstruction steps — single source of
// truth. Each entry is a subdirectory under params.output where that step
// writes its <dataset>.zarr. The pipeline's raw input/output live in the
// workflow body, not here (input may not even be a zarr). A Dragonfly pipeline
// would define its own map; reordering or renaming a step is a one-line edit.
DIRECTORY_LAYOUT = [
    // convert    : '0-convert',     // first step when raw input isn't zarr
    flat_field    : '0-flatfield',
    deskew        : '1-deskew',
    // reconstruct   : '2-reconstruct',
    // virtual_stain : '3-virtual-stain',
    // track         : '4-track',
    // assemble      : '5-assemble',
]


workflow {
    if (!params.input)         error "Provide --input"
    if (!params.output)        error "Provide --output"
    if (!params.deskew_config) error "Provide --deskew_config"

    def ds  = dataset_name()
    def out = params.output

    trigger = Channel.value(true)

    // ----- Deskew -----------------------------------------------------------
    // Here the pipeline input is already a zarr, so deskew reads it directly
    // and writes the deskew step directory. When a convert step is added ahead
    // of deskew, point deskew_input at the convert output instead — deskew_wf
    // doesn't care where its input comes from.
    deskew_input  = params.input
    deskew_output = "${out}/${DIRECTORY_LAYOUT.deskew}/${ds}.zarr"

    collect_positions(deskew_input)
    all_positions = collect_positions.out

    dk_done = deskew_wf(all_positions, deskew_input, deskew_output, params.deskew_config, trigger)

    // ----- Flat-field → Deskew (build-out example; not yet active) ----------
    // When the flat-field module lands, chaining it ahead of deskew is purely
    // an orchestration change — deskew_wf itself stays identical:
    //
    //   ff_input  = params.input
    //   ff_output = "${out}/${DIRECTORY_LAYOUT.flat_field}/${ds}.zarr"
    //   ff_done   = flat_field_wf(all_positions, ff_input, ff_output, params.flat_field_config, trigger)
    //
    //   // deskew now reads flat-field's output and waits on ff_done:
    //   deskew_input  = "${out}/${DIRECTORY_LAYOUT.flat_field}/${ds}.zarr"
    //   deskew_output = "${out}/${DIRECTORY_LAYOUT.deskew}/${ds}.zarr"
    //   dk_done = deskew_wf(all_positions, deskew_input, deskew_output, params.deskew_config, ff_done.done)
}
