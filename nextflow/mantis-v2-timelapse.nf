#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.input_zarr          = null
params.output_dir          = null
params.flat_field_config   = null
params.deskew_config       = null
params.reconstruct_config  = null
params.predict_config      = null
params.track_config        = null
params.concatenate_config  = null
params.rename_prefix       = ""
params.rename_suffix       = ""
params.num_threads         = 1
params.biahub_project      = null
params.viscy_project       = null
params.work_dir            = null
params.max_positions       = 0

include { list_positions }    from './modules/common'
include { flat_field_wf }     from './modules/flat_field'
include { deskew_wf }         from './modules/deskew'
include { reconstruct_wf }    from './modules/reconstruct'
include { virtual_stain_wf }  from './modules/virtual_stain'
include { rename_wf }         from './modules/rename_channels'
include { track_wf }          from './modules/tracking'
include { assemble_wf }       from './modules/assembly'


workflow {
    if (!params.input_zarr)          error "Provide --input_zarr"
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.flat_field_config)   error "Provide --flat_field_config"
    if (!params.deskew_config)       error "Provide --deskew_config"
    if (!params.reconstruct_config)  error "Provide --reconstruct_config"
    if (!params.predict_config)      error "Provide --predict_config"
    if (!params.track_config)        error "Provide --track_config"
    if (!params.concatenate_config)  error "Provide --concatenate_config"
    if (!params.rename_prefix && !params.rename_suffix) {
        error "Provide --rename_prefix and/or --rename_suffix"
    }

    positions_ch = list_positions()
        | splitText
        | map { it.trim() }
        | filter { it }

    all_positions = params.max_positions > 0
        ? positions_ch | take(params.max_positions) | collect
        : positions_ch | collect

    // Phase 1: flat-field → deskew → reconstruct (linear)
    ff_done  = flat_field_wf(all_positions)
    dk_done  = deskew_wf(all_positions, ff_done.done)
    rc_done  = reconstruct_wf(all_positions, dk_done.done)

    // Phase 2: virtual stain + tracking (parallel, both read from reconstruct)
    vs_done  = virtual_stain_wf(all_positions, rc_done.done)
    tk_done  = track_wf(all_positions, rc_done.done)

    // Phase 2b: rename reconstruct channels (after VS + tracking finish reading)
    rename_trigger = vs_done.done.mix(tk_done.done) | collect
    rn_done  = rename_wf(all_positions, rename_trigger)

    // Phase 3: assembly (waits for rename)
    assemble_wf(all_positions, rn_done.done)
}
