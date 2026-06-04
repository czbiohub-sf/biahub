// Flat-field subworkflow: init → parse_resources → fan-out run × N positions.
//
// run_flat_field uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs — debug
// mode ensures the process_single_position call executes synchronously inside
// the Nextflow task.  See also:
// examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir; step_dir } from './common'


process init_flat_field {
    label 'cpu_local'

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('flat_field')}"
    ${biahub_cmd()} flat-field --init \
        -i "${params.input_zarr}"/*/*/* \
        -o "${params.output_dir}/${step_dir('flat_field')}/${dataset_name()}.zarr" \
        -c "${params.flat_field_config}"
    """
}

process run_flat_field {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('flat_field') }
    maxForks 30
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '1h'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd()} flat-field --cluster debug \
        -i "${params.input_zarr}/${position}" \
        -o "${params.output_dir}/${step_dir('flat_field')}/${dataset_name()}.zarr" \
        -c "${params.flat_field_config}"
    """
}


workflow flat_field_wf {
    take:
    positions

    main:
    resources = init_flat_field().map { parse_resources(it) }
    ff_done = positions
        .flatMap { it }
        .combine(resources)
        | run_flat_field
        | collect

    emit:
    done = ff_done
}
