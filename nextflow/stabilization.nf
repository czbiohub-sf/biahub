#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { stabilization_estimate_focus_z_wf }   from './modules/stabilization'
include { stabilization_estimate_focus_xyz_wf }  from './modules/stabilization'
include { stabilization_estimate_pcc_wf }        from './modules/stabilization'
include { stabilization_estimate_beads_wf }      from './modules/stabilization'
include { stabilize_wf }                         from './modules/stabilization'
