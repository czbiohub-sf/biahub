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
params.biahub_project      = null
params.viscy_project       = null
params.work_dir            = null
params.max_positions       = 0
params.qc_config_dir       = null
params.qc_project          = null
params.qc_report_dir       = null
params.qc_report_static    = false
params.clean_intermediates = false

include { list_positions; dataset_name; clean_intermediates } from './modules/common'
include { flat_field_wf }     from './modules/flat_field'
include { deskew_wf }         from './modules/deskew'
include { reconstruct_wf }    from './modules/reconstruct'
include { virtual_stain_wf }  from './modules/virtual_stain'
include { rename_wf }         from './modules/rename_channels'
include { track_wf }          from './modules/tracking'
include { assemble_wf_mantisv2 } from './modules/assembly'
include { qc_stage_wf }      from './modules/qc'
include { qc_report_wf }     from './modules/qc'


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

// Build a channel of (zarr_path, config_path) tuples for QC stages.
// Each entry in stage_channels is [done_channel, zarr_path, config_filename].
// Null entries are filtered out; done_channels gate when each stage can start.
def build_qc_inputs(List stage_channels) {
    if (!params.qc_config_dir) return Channel.empty()

    def inputs = stage_channels
        .findAll { it[0] != null }
        .collect { done_ch, zarr, cfg -> done_ch.map { [zarr, cfg] } }

    if (inputs.size() == 0) return Channel.empty()
    return inputs.inject { a, b -> a.mix(b) }
}


// ---------------------------------------------------------------------------
//  QC wrapper workflow
//
//  Takes a channel of (zarr_path, config_path) tuples, runs QC stages on
//  each, then generates a combined report. When the input channel is empty
//  (QC not requested), no processes execute and the output is empty.
// ---------------------------------------------------------------------------

workflow run_qc_wf {
    take:
    qc_inputs       // Channel of tuple(zarr_path, config_path)

    main:
    qc_stage_wf(qc_inputs)
    all_qc = qc_stage_wf.out.done
        | collect
        | filter { it.size() > 0 }

    def report_dir = params.qc_report_dir ?: "${params.output_dir}/qc/report"
    qc_report_wf(all_qc, report_dir)

    emit:
    done = qc_report_wf.out.done
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
    tk_done  = track_wf(all_positions, vs_done.done)
    pre_asm  = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, tk_done.done).done
        : tk_done.done
    asm_done = assemble_wf_mantisv2(all_positions, pre_asm)

    def ds      = dataset_name()
    def out     = params.output_dir
    def qc_dir  = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [ff_done.done,  "${out}/0-flatfield/${ds}.zarr",      "${qc_dir}/qc_stage1_post_flatfield.yaml"],
        [dk_done.done,  "${out}/1-deskew/${ds}.zarr",         "${qc_dir}/qc_stage2_post_deskew.yaml"],
        [rc_done.done,  "${out}/2-reconstruct/${ds}.zarr",    "${qc_dir}/qc_stage3_post_reconstruct.yaml"],
        [vs_done.done,  "${out}/3-virtual-stain/${ds}.zarr",  "${qc_dir}/qc_stage4_post_virtual_stain.yaml"],
        [asm_done.done, "${out}/5-assemble/${ds}.zarr",       "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
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
    tk_done  = track_wf(all_positions, vs_done.done)
    pre_asm  = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, tk_done.done).done
        : tk_done.done
    asm_done = assemble_wf_mantisv2(all_positions, pre_asm)

    def ds      = dataset_name()
    def out     = params.output_dir
    def qc_dir  = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [dk_done.done,  "${out}/1-deskew/${ds}.zarr",         "${qc_dir}/qc_stage2_post_deskew.yaml"],
        [rc_done.done,  "${out}/2-reconstruct/${ds}.zarr",    "${qc_dir}/qc_stage3_post_reconstruct.yaml"],
        [vs_done.done,  "${out}/3-virtual-stain/${ds}.zarr",  "${qc_dir}/qc_stage4_post_virtual_stain.yaml"],
        [asm_done.done, "${out}/5-assemble/${ds}.zarr",       "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
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
    tk_done  = track_wf(all_positions, vs_done.done)
    pre_asm  = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, tk_done.done).done
        : tk_done.done
    asm_done = assemble_wf_mantisv2(all_positions, pre_asm)

    def ds      = dataset_name()
    def out     = params.output_dir
    def qc_dir  = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [rc_done.done,  "${out}/2-reconstruct/${ds}.zarr",    "${qc_dir}/qc_stage3_post_reconstruct.yaml"],
        [vs_done.done,  "${out}/3-virtual-stain/${ds}.zarr",  "${qc_dir}/qc_stage4_post_virtual_stain.yaml"],
        [asm_done.done, "${out}/5-assemble/${ds}.zarr",       "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
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
    tk_done  = track_wf(all_positions, vs_done.done)
    pre_asm  = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, tk_done.done).done
        : tk_done.done
    asm_done = assemble_wf_mantisv2(all_positions, pre_asm)

    def ds      = dataset_name()
    def out     = params.output_dir
    def qc_dir  = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [vs_done.done,  "${out}/3-virtual-stain/${ds}.zarr",  "${qc_dir}/qc_stage4_post_virtual_stain.yaml"],
        [asm_done.done, "${out}/5-assemble/${ds}.zarr",       "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
}


// ---------------------------------------------------------------------------
//  Entry: from tracking (assumes through 3-virtual-stain exist)
//  Usage: nextflow run ... -entry from_tracking
// ---------------------------------------------------------------------------

workflow from_tracking {
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.track_config)        error "Provide --track_config"
    if (!params.concatenate_config)  error "Provide --concatenate_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)

    tk_done  = track_wf(all_positions, trigger)
    pre_asm  = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, tk_done.done).done
        : tk_done.done
    asm_done = assemble_wf_mantisv2(all_positions, pre_asm)

    def ds      = dataset_name()
    def out     = params.output_dir
    def qc_dir  = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [asm_done.done, "${out}/5-assemble/${ds}.zarr", "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
}


// ---------------------------------------------------------------------------
//  Entry: from assembly (assumes through rename exist)
//  Usage: nextflow run ... -entry from_assembly
// ---------------------------------------------------------------------------

workflow from_assembly {
    if (!params.output_dir)          error "Provide --output_dir"
    if (!params.concatenate_config)  error "Provide --concatenate_config"

    all_positions = collect_positions()
    trigger = Channel.value(true)

    asm_done = assemble_wf_mantisv2(all_positions, trigger)

    def ds      = dataset_name()
    def out     = params.output_dir
    def qc_dir  = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [asm_done.done, "${out}/5-assemble/${ds}.zarr", "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
}


// ---------------------------------------------------------------------------
//  Default entry (anonymous workflow delegates to full)
// ---------------------------------------------------------------------------

workflow {
    full()
}
