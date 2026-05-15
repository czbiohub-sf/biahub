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
params.rename_prefix       = null
params.rename_suffix       = null
params.rename_config       = null
params.biahub_project      = null
params.viscy_project       = null
params.work_dir            = null
params.max_positions       = 0
params.qc_config_dir       = null
params.qc_project          = null
params.qc_report_dir       = null
params.qc_report_static    = false
params.quarto_bin          = null
params.clean_intermediates = false

include { list_positions; dataset_name } from './modules/common'
include { flat_field_wf }     from './modules/flat_field'
include { deskew_wf }         from './modules/deskew'
include { reconstruct_wf }    from './modules/reconstruct'
include { virtual_stain_wf }  from './modules/virtual_stain'
include { rename_wf; rename_channels_map_wf } from './modules/rename_channels'
include { track_wf }          from './modules/tracking'
include { assemble_wf_mantisv2 } from './modules/assembly'
include { qc_stage_wf as qc_post_flatfield }      from './modules/qc'
include { qc_stage_wf as qc_post_deskew }         from './modules/qc'
include { qc_stage_wf as qc_post_reconstruct }    from './modules/qc'
include { qc_stage_wf as qc_post_virtual_stain }  from './modules/qc'
include { qc_stage_wf as qc_post_assembly }        from './modules/qc'
include { qc_report_wf }                           from './modules/qc'


// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

def collect_positions() {
    positions_ch = list_positions()
        | splitText
        | map { it.trim() }
        | filter { it }

    return params.max_positions > 0
        ? positions_ch | take(params.max_positions) | collect
        : positions_ch | collect
}

def run_qc(all_positions, Map stages) {
    if (!params.qc_config_dir) return

    def qc_dir   = params.qc_config_dir
    def ff_zarr  = "${params.output_dir}/0-flatfield/${dataset_name()}.zarr"
    def dk_zarr  = "${params.output_dir}/1-deskew/${dataset_name()}.zarr"
    def rc_zarr  = "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr"
    def vs_zarr  = "${params.output_dir}/3-virtual-stain/${dataset_name()}.zarr"
    def asm_zarr = "${params.output_dir}/5-assemble/${dataset_name()}.zarr"

    def qc_done_list = []
    def qc_summary_list = []

    if (stages.ff_done) {
        qc1 = qc_post_flatfield(all_positions, stages.ff_done, ff_zarr, "${qc_dir}/qc_stage1_post_flatfield.yaml", "flatfield")
        qc_done_list.add(qc1.done)
        qc_summary_list.add(qc1.summary)
    }
    if (stages.dk_done) {
        qc2 = qc_post_deskew(all_positions, stages.dk_done, dk_zarr, "${qc_dir}/qc_stage2_post_deskew.yaml", "deskew")
        qc_done_list.add(qc2.done)
        qc_summary_list.add(qc2.summary)
    }
    if (stages.rc_done) {
        qc3 = qc_post_reconstruct(all_positions, stages.rc_done, rc_zarr, "${qc_dir}/qc_stage3_post_reconstruct.yaml", "reconstruct")
        qc_done_list.add(qc3.done)
        qc_summary_list.add(qc3.summary)
    }
    if (stages.vs_done) {
        qc4 = qc_post_virtual_stain(all_positions, stages.vs_done, vs_zarr, "${qc_dir}/qc_stage4_post_virtual_stain.yaml", "virtual_stain")
        qc_done_list.add(qc4.done)
        qc_summary_list.add(qc4.summary)
    }
    if (stages.asm_done) {
        qc5 = qc_post_assembly(all_positions, stages.asm_done, asm_zarr, "${qc_dir}/qc_stage5_post_assembly.yaml", "assembly")
        qc_done_list.add(qc5.done)
        qc_summary_list.add(qc5.summary)
    }

    if (qc_done_list.size() > 0) {
        all_qc = qc_done_list.inject { a, b -> a.mix(b) } | collect
        all_summaries = qc_summary_list.inject { a, b -> a.mix(b) } | collect

        def report_dir = params.qc_report_dir ?: "${params.output_dir}/qc/report"
        qc_report_wf(all_qc, all_summaries, asm_zarr, [ff_zarr, dk_zarr, rc_zarr, vs_zarr], params.output_dir, report_dir)
    }
}


// ---------------------------------------------------------------------------
//  Entry: full pipeline (default)
//  Usage: nextflow run ... -entry full
// ---------------------------------------------------------------------------

workflow full {
    if (!params.input_zarr)          error "Provide --input_zarr"
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.flat_field_config)   error "Provide --flat_field_config"
    if (!params.deskew_config)       error "Provide --deskew_config"
    if (!params.reconstruct_config)  error "Provide --reconstruct_config"
    if (!params.predict_config)      error "Provide --predict_config"
    if (!params.track_config)        error "Provide --track_config"
    if (!params.concatenate_config)  error "Provide --concatenate_config"

    all_positions = collect_positions()

    ff_done  = flat_field_wf(all_positions)
    dk_done  = deskew_wf(all_positions, ff_done.done)
    rc_done  = reconstruct_wf(all_positions, dk_done.done)
    vs_done  = virtual_stain_wf(all_positions, rc_done.done)
    pre_rename = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, vs_done.done).done
        : vs_done.done
    pre_asm  = params.rename_config
        ? rename_channels_map_wf(all_positions, pre_rename).done
        : pre_rename
    asm_done = assemble_wf_mantisv2(all_positions, pre_asm)
    tk_done  = track_wf(all_positions, asm_done.done)

    run_qc(all_positions, [
        ff_done:  ff_done.done,
        dk_done:  dk_done.done,
        rc_done:  rc_done.done,
        vs_done:  vs_done.done,
        asm_done: asm_done.done,
    ])
}


// ---------------------------------------------------------------------------
//  Entry: from deskew (assumes 0-flatfield exists)
//  Usage: nextflow run ... -entry from_deskew
// ---------------------------------------------------------------------------

workflow from_deskew {
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.deskew_config)       error "Provide --deskew_config"
    if (!params.reconstruct_config)  error "Provide --reconstruct_config"
    if (!params.predict_config)      error "Provide --predict_config"
    if (!params.track_config)        error "Provide --track_config"
    if (!params.concatenate_config)  error "Provide --concatenate_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)

    dk_done  = deskew_wf(all_positions, trigger)
    rc_done  = reconstruct_wf(all_positions, dk_done.done)
    vs_done  = virtual_stain_wf(all_positions, rc_done.done)
    pre_rename = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, vs_done.done).done
        : vs_done.done
    pre_asm  = params.rename_config
        ? rename_channels_map_wf(all_positions, pre_rename).done
        : pre_rename
    asm_done = assemble_wf_mantisv2(all_positions, pre_asm)
    tk_done  = track_wf(all_positions, asm_done.done)

    run_qc(all_positions, [
        dk_done:  dk_done.done,
        rc_done:  rc_done.done,
        vs_done:  vs_done.done,
        asm_done: asm_done.done,
    ])
}


// ---------------------------------------------------------------------------
//  Entry: from reconstruct (assumes 0-flatfield + 1-deskew exist)
//  Usage: nextflow run ... -entry from_reconstruct
// ---------------------------------------------------------------------------

workflow from_reconstruct {
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.reconstruct_config)  error "Provide --reconstruct_config"
    if (!params.predict_config)      error "Provide --predict_config"
    if (!params.track_config)        error "Provide --track_config"
    if (!params.concatenate_config)  error "Provide --concatenate_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)

    rc_done  = reconstruct_wf(all_positions, trigger)
    vs_done  = virtual_stain_wf(all_positions, rc_done.done)
    pre_rename = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, vs_done.done).done
        : vs_done.done
    pre_asm  = params.rename_config
        ? rename_channels_map_wf(all_positions, pre_rename).done
        : pre_rename
    asm_done = assemble_wf_mantisv2(all_positions, pre_asm)
    tk_done  = track_wf(all_positions, asm_done.done)

    run_qc(all_positions, [
        rc_done:  rc_done.done,
        vs_done:  vs_done.done,
        asm_done: asm_done.done,
    ])
}


// ---------------------------------------------------------------------------
//  Entry: from virtual stain (assumes through 2-reconstruct exist)
//  Usage: nextflow run ... -entry from_virtual_stain
// ---------------------------------------------------------------------------

workflow from_virtual_stain {
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.predict_config)      error "Provide --predict_config"
    if (!params.track_config)        error "Provide --track_config"
    if (!params.concatenate_config)  error "Provide --concatenate_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)

    vs_done  = virtual_stain_wf(all_positions, trigger)
    pre_rename = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, vs_done.done).done
        : vs_done.done
    pre_asm  = params.rename_config
        ? rename_channels_map_wf(all_positions, pre_rename).done
        : pre_rename
    asm_done = assemble_wf_mantisv2(all_positions, pre_asm)
    tk_done  = track_wf(all_positions, asm_done.done)

    run_qc(all_positions, [
        vs_done:  vs_done.done,
        asm_done: asm_done.done,
    ])
}


// ---------------------------------------------------------------------------
//  Entry: from assembly (assumes through 3-virtual-stain exist)
//  Usage: nextflow run ... -entry from_assembly
// ---------------------------------------------------------------------------

workflow from_assembly {
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.concatenate_config)  error "Provide --concatenate_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)

    asm_done = assemble_wf_mantisv2(all_positions, trigger)

    run_qc(all_positions, [
        asm_done: asm_done.done,
    ])
}


// ---------------------------------------------------------------------------
//  Entry: from tracking (assumes through 5-assemble exist)
//  Usage: nextflow run ... -entry from_tracking
// ---------------------------------------------------------------------------

workflow from_tracking {
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.track_config)        error "Provide --track_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)

    tk_done  = track_wf(all_positions, trigger)
}


// ---------------------------------------------------------------------------
//  Standalone entries: rerun a single step only
//  Usage: nextflow run ... -entry only_flat_field
// ---------------------------------------------------------------------------

workflow only_flat_field {
    if (!params.input_zarr)        error "Provide --input_zarr"
    if (!params.output_dir)        error "Provide --output_dir"
    if (!params.flat_field_config) error "Provide --flat_field_config"

    all_positions = collect_positions()
    flat_field_wf(all_positions)
}

workflow only_deskew {
    if (!params.output_dir)    error "Provide --output_dir"
    if (!params.deskew_config) error "Provide --deskew_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)
    deskew_wf(all_positions, trigger)
}

workflow only_reconstruct {
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.reconstruct_config)  error "Provide --reconstruct_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)
    reconstruct_wf(all_positions, trigger)
}

workflow only_virtual_stain {
    if (!params.output_dir)      error "Provide --output_dir"
    if (!params.predict_config)  error "Provide --predict_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)
    virtual_stain_wf(all_positions, trigger)
}

workflow only_rename_channels_map {
    if (!params.output_dir) error "Provide --output_dir"

    all_positions = collect_positions()
    trigger = Channel.value(true)
    rename_channels_map_wf(all_positions, trigger)
}

workflow only_tracking {
    if (!params.output_dir)    error "Provide --output_dir"
    if (!params.track_config)  error "Provide --track_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)
    track_wf(all_positions, trigger)
}

workflow only_assembly {
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.concatenate_config)  error "Provide --concatenate_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)
    assemble_wf_mantisv2(all_positions, trigger)
}


// ---------------------------------------------------------------------------
//  Default entry (anonymous workflow delegates to full)
// ---------------------------------------------------------------------------

workflow {
    full()
}
