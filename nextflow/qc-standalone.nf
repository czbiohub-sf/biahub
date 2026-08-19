#!/usr/bin/env nextflow
//
// Standalone QC pipeline — runs the external imaging-qc CLI against arbitrary
// zarr stores listed in a CSV manifest, without the full mantis pipeline.
// Python owns all dispatch logic via plan-stage (JSON); Nextflow owns fan-out,
// barriers, and retries.
//
// `imaging-qc` is called bare, so activate an environment holding it before
// launching (it comes from biahub's `qc` extra, which `all` leaves out):
//
//   uv sync --project <BIAHUB> --extra qc
//   source <BIAHUB>/.venv/bin/activate
//   nextflow run nextflow/qc-standalone.nf -c nextflow/nextflow.config -profile slurm \
//       --stages_manifest stages.csv --output <experiment-dir> -resume
//
// Manifest is a CSV with header `zarr_path,config_path`; each row is one QC
// stage (one config on one zarr). Rows run in parallel, and each store gets its
// own report beside it (`<store>_report/`).
//

nextflow.enable.dsl = 2

include { qc_stage_wf }        from './modules/qc'
include { qc_report_wf }       from './modules/qc'
include { check_environment }  from './modules/common'


workflow {
    // Nothing here calls `biahub` any more — the whole stage, report included, is
    // imaging-qc verbs.
    check_environment(['imaging-qc'])

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

    // Per-store reports: each row's report follows its own store as soon as that
    // store finalizes, so one slow store does not hold the others' reports back.
    qc_report_wf(qc.done)
}
