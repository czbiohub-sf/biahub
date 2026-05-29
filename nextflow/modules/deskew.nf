// Deskew subworkflow: init → parse_resources → fan-out run × N positions.
//
// run_deskew uses `--cluster debug` so that submitit's DebugExecutor runs the
// work in-process.  Nextflow already handles per-position fan-out and resource
// scheduling, so the CLI must NOT submit its own SLURM jobs — debug mode
// ensures the process_single_position call executes synchronously inside the
// Nextflow task.  See also:
// examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_deskew {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('deskew')}"
    ${biahub_cmd()} deskew --init \
        -i "${params.output_dir}/0-flatfield/${dataset_name()}.zarr"/*/*/* \
        -o "${params.output_dir}/1-deskew/${dataset_name()}.zarr" \
        -c "${params.deskew_config}"
    """
}

process run_deskew {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('deskew') }
    maxForks 30
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { task.attempt == 1 ? '1h' : '2h' }
    maxRetries 1
    errorStrategy 'retry'


    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd()} deskew --cluster debug \
        -i "${params.output_dir}/0-flatfield/${dataset_name()}.zarr/${position}" \
        -o "${params.output_dir}/1-deskew/${dataset_name()}.zarr" \
        -c "${params.deskew_config}"
    """
}


workflow deskew_wf {
    take:
    positions
    prev_done

    main:
    init_out = init_deskew(prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    dk_done = positions
        .flatMap { it }
        .combine(resources)
        | run_deskew
        | collect

    emit:
    done = dk_done
}
