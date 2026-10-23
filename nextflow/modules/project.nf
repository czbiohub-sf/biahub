// Projection subworkflows: init (scaffold + resources) and run (fan-out × N).
//
// This module is PATH-AGNOSTIC. Callers pass the input zarr, output zarr, and
// config explicitly; the module has no idea where it sits in the pipeline
// directory layout. The orchestrating pipeline (see mantis-v2.nf) owns the
// layout and the order of steps; this module just projects whatever 3D plate
// it is handed to a 2D (Z=1) plate.
//
// INIT AND RUN ARE SEPARATE SUBWORKFLOWS. The pipeline runs every step's init
// ahead of every step's compute, so that a bad config fails in the first
// minutes rather than hours in — which means it, not this module, decides when
// each phase happens. See the INIT PHASE comment in mantis-v2.nf.
//
// run_project uses `--cluster debug` so that submitit's DebugExecutor runs the
// work in-process. Nextflow already handles per-position fan-out and resource
// scheduling, so the CLI must NOT submit its own SLURM jobs. See
// modules/flat_field.nf, the template this follows.

include { parse_resources; slurm_logs; slurm_log_dir; slurm_output_readme; retry_time; retry_memory } from './common'


process init_project {
    label 'cpu_local'

    input:
    val input_zarr
    val output_zarr
    val config
    path config_file  // staged only for the task hash: see common.nf, #397
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('project')}"
    ${slurm_output_readme('project', output_zarr)}
    biahub project --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
    """
}

process run_project {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('project') }
    cpus { meta.cpus }
    memory { retry_memory(meta.mem_gb, task) }
    time { retry_time(meta.time_minutes, task) }

    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val config
    path config_file  // staged only for the task hash: see common.nf, #397

    output:
    val position

    script:
    // --resume: a preempted task finishes the write it is in and stops early, so
    // the retry (or a later `nextflow -resume`) recomputes only the (t, c) units
    // this position had not finished. The focus pre-pass runs again either way.
    """
    biahub project --cluster debug --resume \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// Validate the config and scaffold the output plate. Metadata-only and cheap,
// so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the input (3D) plate.zarr
//   output_zarr  path to the projected (Z=1) output plate.zarr
//   config       path to the project settings YAML
//   trigger      gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once the output plate exists
workflow project_init_wf {
    take:
    input_zarr
    output_zarr
    config
    trigger

    main:
    init_out = init_project(input_zarr, output_zarr, config, file(config), trigger.collect().map { 'done' })

    emit:
    // Value channels by the contract every step module emits on; see
    // flat_field_init_wf for why the guarantee is made at the input.
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }
    done      = init_out.map { 'done' }
}


// Fan out one projection task per position.
//
// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input (3D) plate.zarr
//   output_zarr  path to the projected (Z=1) output plate.zarr
//   config       path to the project settings YAML
//   resources    RESOURCES payload from project_init_wf
//   prev_done    gating channel — compute starts once this emits
workflow project_run_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    resources
    prev_done

    main:
    // Gate channels are mapped to a token before combining, never combined raw;
    // see flat_field_run_wf.
    pos_meta = positions
        .flatMap { items -> items }
        .combine(resources)
        .combine(prev_done.map { 'done' })
        .map { pos, meta, _gate -> [pos, meta] }

    pj_done = run_project(pos_meta, input_zarr, output_zarr, config, file(config)) | collect

    emit:
    done = pj_done
}
