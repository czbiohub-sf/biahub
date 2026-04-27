#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.input_zarr         = null
params.output_dir         = null
params.flat_field_config  = null
params.deskew_config      = null
params.reconstruct_config = null
params.predict_config     = null
params.num_threads      = 1
params.biahub_project     = null
params.viscy_project      = null
params.work_dir           = null
params.max_positions      = 0

include { list_positions }    from './modules/common'
include { flat_field_wf }     from './modules/flat_field'
include { deskew_wf }         from './modules/deskew'
include { reconstruct_wf }    from './modules/reconstruct'
include { virtual_stain_wf }  from './modules/virtual_stain'


workflow {
    if (!params.input_zarr)         error "Provide --input_zarr"
    if (!params.output_dir)         error "Provide --output_dir"
    if (!params.flat_field_config)  error "Provide --flat_field_config"
    if (!params.deskew_config)      error "Provide --deskew_config"
    if (!params.reconstruct_config) error "Provide --reconstruct_config"

    positions_ch = list_positions()
        | splitText
        | map { it.trim() }
        | filter { it }

    all_positions = params.max_positions > 0
        ? positions_ch | take(params.max_positions) | collect
        : positions_ch | collect

    ff_done = flat_field_wf(all_positions)
    dk_done = deskew_wf(all_positions, ff_done.done)
    rc_done = reconstruct_wf(all_positions, dk_done.done)

    if (params.predict_config) {
        virtual_stain_wf(all_positions, rc_done.done)
    }
}
