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
params.qc_config_dir       = null
params.qc_project          = null
params.qc_report_dir       = null

include { list_positions; dataset_name } from './modules/common'
include { flat_field_wf }     from './modules/flat_field'
include { deskew_wf }         from './modules/deskew'
include { reconstruct_wf }    from './modules/reconstruct'
include { virtual_stain_wf }  from './modules/virtual_stain'
include { rename_wf }         from './modules/rename_channels'
include { track_wf }          from './modules/tracking'
include { assemble_wf }       from './modules/assembly'
include { qc_stage_wf as qc_post_flatfield }      from './modules/qc'
include { qc_stage_wf as qc_post_deskew }         from './modules/qc'
include { qc_stage_wf as qc_post_reconstruct }    from './modules/qc'
include { qc_stage_wf as qc_post_virtual_stain }  from './modules/qc'
include { qc_stage_wf as qc_post_assembly }        from './modules/qc'
include { qc_report_wf }                           from './modules/qc'


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

    // Phase 2: virtual stain (reads from reconstruct), then tracking (reads from virtual stain)
    vs_done  = virtual_stain_wf(all_positions, rc_done.done)
    tk_done  = track_wf(all_positions, vs_done.done)

    // Phase 2b: rename reconstruct channels (after tracking finishes reading)
    rn_done  = rename_wf(all_positions, tk_done.done)

    // Phase 3: assembly (waits for rename)
    asm_done = assemble_wf(all_positions, rn_done.done)

    // QC stages (parallel with next processing step, non-blocking)
    if (params.qc_config_dir) {
        def qc_dir  = params.qc_config_dir
        def ff_zarr  = "${params.output_dir}/0-flatfield/${dataset_name()}.zarr"
        def dk_zarr  = "${params.output_dir}/1-deskew/${dataset_name()}.zarr"
        def rc_zarr  = "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr"
        def vs_zarr  = "${params.output_dir}/3-virtual-stain/${dataset_name()}.zarr"
        def asm_zarr = "${params.output_dir}/5-assemble/${dataset_name()}.zarr"

        qc1 = qc_post_flatfield(all_positions, ff_done.done,  ff_zarr,  "${qc_dir}/qc_stage1_post_flatfield.yaml", "flatfield")
        qc2 = qc_post_deskew(all_positions, dk_done.done,     dk_zarr,  "${qc_dir}/qc_stage2_post_deskew.yaml", "deskew")
        qc3 = qc_post_reconstruct(all_positions, rc_done.done, rc_zarr, "${qc_dir}/qc_stage3_post_reconstruct.yaml", "reconstruct")
        qc4 = qc_post_virtual_stain(all_positions, vs_done.done, vs_zarr, "${qc_dir}/qc_stage4_post_virtual_stain.yaml", "virtual_stain")
        qc5 = qc_post_assembly(all_positions, asm_done.done,   asm_zarr, "${qc_dir}/qc_stage5_post_assembly.yaml", "assembly")

        all_qc = qc1.done.mix(qc2.done, qc3.done, qc4.done, qc5.done) | collect
        all_summaries = qc1.summary.mix(qc2.summary, qc3.summary, qc4.summary, qc5.summary) | collect

        def report_dir = params.qc_report_dir ?: "${params.output_dir}/qc/report"
        qc_report_wf(all_qc, all_summaries, asm_zarr, [ff_zarr, dk_zarr, rc_zarr, vs_zarr], report_dir, "${qc_dir}/qc_stage5_post_assembly.yaml")
    }
}
