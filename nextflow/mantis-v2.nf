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
//  Each step's subworkflow (e.g. deskew_wf) is path-agnostic and speaks only
//  in zarr: this pipeline hands it explicit input_zarr/output_zarr paths. The
//  pipeline itself speaks in `input`/`output`, where `input` may NOT be a zarr
//  (in some pipelines the first step converts raw input to zarr). To reorder
//  steps, change where a step reads from here; the modules stay untouched.
// ---------------------------------------------------------------------------

include { collect_positions; dataset_name; clean_intermediates } from './modules/common'
include { flat_field_wf }          from './modules/flat_field'
include { deskew_wf }              from './modules/deskew'
include { reconstruct_wf }         from './modules/reconstruct'
include { virtual_stain_wf }       from './modules/virtual_stain'
include { rename_wf }              from './modules/rename_channels'
include { track_wf }               from './modules/tracking'
include { assemble_wf_mantisv2 }   from './modules/assembly'
include { qc_stage_wf }           from './modules/qc'
include { qc_report_wf }          from './modules/qc'

DIRECTORY_LAYOUT = [
    flat_field    : '0-flatfield',
    deskew        : '1-deskew',
    reconstruct   : '2-reconstruct',
    virtual_stain : '3-virtual-stain',
    track         : '4-track',
    assemble      : '5-assemble',
]


// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

def build_qc_inputs(List stage_channels) {
    if (!params.qc_config_dir) return Channel.empty()

    def inputs = stage_channels
        .findAll { it[0] != null }
        .collect { done_ch, zarr, cfg -> done_ch.map { [zarr, cfg] } }

    if (inputs.size() == 0) return Channel.empty()
    return inputs.inject { a, b -> a.mix(b) }
}


workflow run_qc_wf {
    take:
    qc_inputs

    main:
    qc_stage_wf(qc_inputs)
    all_qc = qc_stage_wf.out.done
        | collect
        | filter { it.size() > 0 }

    def report_dir = params.qc_report_dir ?: "${params.output}/qc/report"
    qc_report_wf(all_qc, report_dir)

    emit:
    done = qc_report_wf.out.done
}


// ---------------------------------------------------------------------------
//  Entry: full pipeline (default)
// ---------------------------------------------------------------------------

workflow full {
    if (!params.input)             error "Provide --input"
    if (!params.output)            error "Provide --output"
    if (!params.flat_field_config) error "Provide --flat_field_config"
    if (!params.deskew_config)     error "Provide --deskew_config"
    if (!params.reconstruct_config) error "Provide --reconstruct_config"
    if (!params.predict_config)    error "Provide --predict_config"
    if (!params.track_config)      error "Provide --track_config"
    if (!params.concatenate_config) error "Provide --concatenate_config"

    def ds  = dataset_name()
    def out = params.output

    collect_positions(params.input)
    all_positions = collect_positions.out
    trigger = Channel.value(true)

    // Flat-field
    ff_input  = params.input
    ff_output = "${out}/${DIRECTORY_LAYOUT.flat_field}/${ds}.zarr"
    ff_done   = flat_field_wf(all_positions, ff_input, ff_output, params.flat_field_config, trigger)

    // Deskew
    dk_input  = ff_output
    dk_output = "${out}/${DIRECTORY_LAYOUT.deskew}/${ds}.zarr"
    dk_done   = deskew_wf(all_positions, dk_input, dk_output, params.deskew_config, ff_done.done)

    // Reconstruct
    rc_input  = dk_output
    rc_output = "${out}/${DIRECTORY_LAYOUT.reconstruct}/${ds}.zarr"
    rc_done   = reconstruct_wf(all_positions, rc_input, rc_output, params.reconstruct_config, dk_done.done)

    // Virtual stain
    vs_input  = rc_output
    vs_output = "${out}/${DIRECTORY_LAYOUT.virtual_stain}/${ds}.zarr"
    vs_done   = virtual_stain_wf(all_positions, vs_input, vs_output, params.predict_config, rc_done.done)

    // Track
    tk_output = "${out}/${DIRECTORY_LAYOUT.track}/${ds}.zarr"
    tk_done   = track_wf(all_positions, rc_output, vs_output, tk_output, params.track_config, vs_done.done)

    // Rename (optional)
    pre_asm = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, rc_output, tk_done.done).done
        : tk_done.done

    // Assembly
    asm_output = "${out}/${DIRECTORY_LAYOUT.assemble}/${ds}.zarr"
    asm_done = assemble_wf_mantisv2(all_positions, dk_output, rc_output, vs_output,
        asm_output, params.concatenate_config, pre_asm)

    // QC
    def qc_dir = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [ff_done.done,  ff_output,  "${qc_dir}/qc_stage1_post_flatfield.yaml"],
        [dk_done.done,  dk_output,  "${qc_dir}/qc_stage2_post_deskew.yaml"],
        [rc_done.done,  rc_output,  "${qc_dir}/qc_stage3_post_reconstruct.yaml"],
        [vs_done.done,  vs_output,  "${qc_dir}/qc_stage4_post_virtual_stain.yaml"],
        [asm_done.done, asm_output, "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
}


// ---------------------------------------------------------------------------
//  Entry: from deskew (assumes 0-flatfield exists)
// ---------------------------------------------------------------------------

workflow from_deskew {
    if (!params.output)            error "Provide --output"
    if (!params.deskew_config)     error "Provide --deskew_config"
    if (!params.reconstruct_config) error "Provide --reconstruct_config"
    if (!params.predict_config)    error "Provide --predict_config"
    if (!params.track_config)      error "Provide --track_config"
    if (!params.concatenate_config) error "Provide --concatenate_config"

    def ds  = dataset_name()
    def out = params.output

    ff_output = "${out}/${DIRECTORY_LAYOUT.flat_field}/${ds}.zarr"
    collect_positions(ff_output)
    all_positions = collect_positions.out
    trigger = Channel.value(true)

    dk_input  = ff_output
    dk_output = "${out}/${DIRECTORY_LAYOUT.deskew}/${ds}.zarr"
    dk_done   = deskew_wf(all_positions, dk_input, dk_output, params.deskew_config, trigger)

    rc_input  = dk_output
    rc_output = "${out}/${DIRECTORY_LAYOUT.reconstruct}/${ds}.zarr"
    rc_done   = reconstruct_wf(all_positions, rc_input, rc_output, params.reconstruct_config, dk_done.done)

    vs_input  = rc_output
    vs_output = "${out}/${DIRECTORY_LAYOUT.virtual_stain}/${ds}.zarr"
    vs_done   = virtual_stain_wf(all_positions, vs_input, vs_output, params.predict_config, rc_done.done)

    tk_output = "${out}/${DIRECTORY_LAYOUT.track}/${ds}.zarr"
    tk_done   = track_wf(all_positions, rc_output, vs_output, tk_output, params.track_config, vs_done.done)

    pre_asm = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, rc_output, tk_done.done).done
        : tk_done.done

    asm_output = "${out}/${DIRECTORY_LAYOUT.assemble}/${ds}.zarr"
    asm_done = assemble_wf_mantisv2(all_positions, dk_output, rc_output, vs_output,
        asm_output, params.concatenate_config, pre_asm)

    def qc_dir = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [dk_done.done,  dk_output,  "${qc_dir}/qc_stage2_post_deskew.yaml"],
        [rc_done.done,  rc_output,  "${qc_dir}/qc_stage3_post_reconstruct.yaml"],
        [vs_done.done,  vs_output,  "${qc_dir}/qc_stage4_post_virtual_stain.yaml"],
        [asm_done.done, asm_output, "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
}


// ---------------------------------------------------------------------------
//  Entry: from reconstruct (assumes 0-flatfield + 1-deskew exist)
// ---------------------------------------------------------------------------

workflow from_reconstruct {
    if (!params.output)            error "Provide --output"
    if (!params.reconstruct_config) error "Provide --reconstruct_config"
    if (!params.predict_config)    error "Provide --predict_config"
    if (!params.track_config)      error "Provide --track_config"
    if (!params.concatenate_config) error "Provide --concatenate_config"

    def ds  = dataset_name()
    def out = params.output

    dk_output = "${out}/${DIRECTORY_LAYOUT.deskew}/${ds}.zarr"
    collect_positions(dk_output)
    all_positions = collect_positions.out
    trigger = Channel.value(true)

    rc_input  = dk_output
    rc_output = "${out}/${DIRECTORY_LAYOUT.reconstruct}/${ds}.zarr"
    rc_done   = reconstruct_wf(all_positions, rc_input, rc_output, params.reconstruct_config, trigger)

    vs_input  = rc_output
    vs_output = "${out}/${DIRECTORY_LAYOUT.virtual_stain}/${ds}.zarr"
    vs_done   = virtual_stain_wf(all_positions, vs_input, vs_output, params.predict_config, rc_done.done)

    tk_output = "${out}/${DIRECTORY_LAYOUT.track}/${ds}.zarr"
    tk_done   = track_wf(all_positions, rc_output, vs_output, tk_output, params.track_config, vs_done.done)

    pre_asm = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, rc_output, tk_done.done).done
        : tk_done.done

    asm_output = "${out}/${DIRECTORY_LAYOUT.assemble}/${ds}.zarr"
    asm_done = assemble_wf_mantisv2(all_positions, dk_output, rc_output, vs_output,
        asm_output, params.concatenate_config, pre_asm)

    def qc_dir = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [rc_done.done,  rc_output,  "${qc_dir}/qc_stage3_post_reconstruct.yaml"],
        [vs_done.done,  vs_output,  "${qc_dir}/qc_stage4_post_virtual_stain.yaml"],
        [asm_done.done, asm_output, "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
}


// ---------------------------------------------------------------------------
//  Entry: from virtual stain (assumes through 2-reconstruct exist)
// ---------------------------------------------------------------------------

workflow from_virtual_stain {
    if (!params.output)            error "Provide --output"
    if (!params.predict_config)    error "Provide --predict_config"
    if (!params.track_config)      error "Provide --track_config"
    if (!params.concatenate_config) error "Provide --concatenate_config"

    def ds  = dataset_name()
    def out = params.output

    dk_output = "${out}/${DIRECTORY_LAYOUT.deskew}/${ds}.zarr"
    rc_output = "${out}/${DIRECTORY_LAYOUT.reconstruct}/${ds}.zarr"
    collect_positions(rc_output)
    all_positions = collect_positions.out
    trigger = Channel.value(true)

    vs_input  = rc_output
    vs_output = "${out}/${DIRECTORY_LAYOUT.virtual_stain}/${ds}.zarr"
    vs_done   = virtual_stain_wf(all_positions, vs_input, vs_output, params.predict_config, trigger)

    tk_output = "${out}/${DIRECTORY_LAYOUT.track}/${ds}.zarr"
    tk_done   = track_wf(all_positions, rc_output, vs_output, tk_output, params.track_config, vs_done.done)

    pre_asm = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, rc_output, tk_done.done).done
        : tk_done.done

    asm_output = "${out}/${DIRECTORY_LAYOUT.assemble}/${ds}.zarr"
    asm_done = assemble_wf_mantisv2(all_positions, dk_output, rc_output, vs_output,
        asm_output, params.concatenate_config, pre_asm)

    def qc_dir = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [vs_done.done,  vs_output,  "${qc_dir}/qc_stage4_post_virtual_stain.yaml"],
        [asm_done.done, asm_output, "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
}


// ---------------------------------------------------------------------------
//  Entry: from tracking (assumes through 3-virtual-stain exist)
// ---------------------------------------------------------------------------

workflow from_tracking {
    if (!params.output)            error "Provide --output"
    if (!params.track_config)      error "Provide --track_config"
    if (!params.concatenate_config) error "Provide --concatenate_config"

    def ds  = dataset_name()
    def out = params.output

    dk_output = "${out}/${DIRECTORY_LAYOUT.deskew}/${ds}.zarr"
    rc_output = "${out}/${DIRECTORY_LAYOUT.reconstruct}/${ds}.zarr"
    vs_output = "${out}/${DIRECTORY_LAYOUT.virtual_stain}/${ds}.zarr"
    collect_positions(rc_output)
    all_positions = collect_positions.out
    trigger = Channel.value(true)

    tk_output = "${out}/${DIRECTORY_LAYOUT.track}/${ds}.zarr"
    tk_done   = track_wf(all_positions, rc_output, vs_output, tk_output, params.track_config, trigger)

    pre_asm = (params.rename_prefix || params.rename_suffix)
        ? rename_wf(all_positions, rc_output, tk_done.done).done
        : tk_done.done

    asm_output = "${out}/${DIRECTORY_LAYOUT.assemble}/${ds}.zarr"
    asm_done = assemble_wf_mantisv2(all_positions, dk_output, rc_output, vs_output,
        asm_output, params.concatenate_config, pre_asm)

    def qc_dir = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [asm_done.done, asm_output, "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
}


// ---------------------------------------------------------------------------
//  Entry: from assembly (assumes through rename exist)
// ---------------------------------------------------------------------------

workflow from_assembly {
    if (!params.output)            error "Provide --output"
    if (!params.concatenate_config) error "Provide --concatenate_config"

    def ds  = dataset_name()
    def out = params.output

    dk_output = "${out}/${DIRECTORY_LAYOUT.deskew}/${ds}.zarr"
    rc_output = "${out}/${DIRECTORY_LAYOUT.reconstruct}/${ds}.zarr"
    vs_output = "${out}/${DIRECTORY_LAYOUT.virtual_stain}/${ds}.zarr"
    collect_positions(dk_output)
    all_positions = collect_positions.out
    trigger = Channel.value(true)

    asm_output = "${out}/${DIRECTORY_LAYOUT.assemble}/${ds}.zarr"
    asm_done = assemble_wf_mantisv2(all_positions, dk_output, rc_output, vs_output,
        asm_output, params.concatenate_config, trigger)

    def qc_dir = params.qc_config_dir
    qc_inputs = build_qc_inputs([
        [asm_done.done, asm_output, "${qc_dir}/qc_stage5_post_assembly.yaml"],
    ])
    run_qc_wf(qc_inputs)

    if (params.clean_intermediates) {
        clean_intermediates(asm_done.done.mix(run_qc_wf.out.done).collect())
    }
}


// ---------------------------------------------------------------------------
//  Default entry
// ---------------------------------------------------------------------------

workflow {
    full()
}
