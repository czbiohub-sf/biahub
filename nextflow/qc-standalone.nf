#!/usr/bin/env nextflow
//
// Standalone QC pipeline — runs the external imaging-qc CLI against arbitrary
// zarr stores listed in a CSV manifest, without the full mantis pipeline.
// Python owns all dispatch logic via plan-stage (JSON); Nextflow owns fan-out,
// barriers, and retries.
//
// Both CLIs are called bare, so activate one environment holding both before
// launching (`imaging-qc` comes from the `qc` extra, which `all` leaves out):
//
//   uv sync --project <BIAHUB> --extra qc
//   source <BIAHUB>/.venv/bin/activate
//   nextflow run nextflow/qc-standalone.nf -c nextflow/nextflow.config -profile slurm \
//       --stages_manifest stages.csv --output <experiment-dir> -resume
//
// Manifest is a CSV with header `zarr_path,config_path`; each row is one QC
// stage (one config on one zarr). Rows run in parallel.
//

nextflow.enable.dsl = 2

include { qc_stage_wf }        from './modules/qc'
include { qc_report_wf }       from './modules/qc'
include { check_environment }  from './modules/common'


workflow {
    // `imaging-qc` does the QC work; `biahub` builds the report spec.
    check_environment(['biahub', 'imaging-qc'])

    if (!params.stages_manifest) {
        error "Provide --stages_manifest (CSV with header: zarr_path,config_path)"
    }
    if (!params.output) {
        error "Provide --output (run/output directory)"
    }

    plan_inputs = Channel
        .fromPath(params.stages_manifest)
        .splitCsv(header: true)
        .map { row -> tuple(row.zarr_path.trim(), row.config_path.trim()) }

    qc = qc_stage_wf(plan_inputs)

    all_qc_done = qc.done.collect()

    def report_dir = params.qc_report_dir ?: "${params.output}/qc/report"
    qc_report_wf(all_qc_done, report_dir)
}
