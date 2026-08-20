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
// stage (one config on one zarr). Rows run in parallel and every store becomes a
// tab of ONE report at `<output>/qc/report`.
//

nextflow.enable.dsl = 2

include { qc_stage_wf; qc_report_wf; qc_report_spec } from './modules/qc'
include { check_environment }                             from './modules/common'


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

    // The manifest is read a SECOND time here, directly rather than through the
    // channel above, because the report spec is written at launch — before any
    // task runs — and a channel's contents are not available then. One CSV, two
    // readers, no ordering dependency between them.
    def rows = file(params.stages_manifest).readLines()
        .findAll { it.trim() && !it.startsWith('zarr_path') }
        .collect { it.split(',').collect { c -> c.trim() } }

    // Label each tab by the store's parent directory (`4-assemble`, `5-track`),
    // which is what distinguishes stores of one dataset; fall back to the store
    // name when two stores would otherwise collide.
    def labels = rows.collect { file(it[0]).parent.name }
    def qc_stores = [rows, labels].transpose().collect { row, label ->
        [label: labels.count(label) > 1 ? "${label}/${file(row[0]).simpleName}" : label,
         zarr: row[0], config: row[1]]
    }

    def report_dir = params.qc_report_dir ?: "${params.output}/qc/report"
    def spec = qc_report_spec(qc_stores, "${params.output}/qc/report_spec.yaml", "QC report")

    qc = qc_stage_wf(plan_inputs)
    qc_report_wf(qc.done, spec, report_dir)
}
