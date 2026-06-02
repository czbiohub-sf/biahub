// Registration subworkflow: estimate → init → fan-out run × N positions.
//
// estimate_registration runs once to compute the affine transform between
// source and target datasets and writes a registration config YAML.
//
// init_register creates the output plate (including crop/overlap computation)
// and emits RESOURCES.
//
// run_register uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs.
// See: examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process estimate_registration {
    label 'cpu'
    time '4h'

    input:
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd()} estimate-registration \
        -s "${params.register_source_zarr}"/*/*/* \
        -t "${params.register_target_zarr}"/*/*/* \
        -o "${params.register_config}" \
        -c "${params.estimate_register_config}"
    """
}

process init_register {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('register')}"
    ${biahub_cmd()} register --init \
        -s "${params.register_source_zarr}"/*/*/* \
        -t "${params.register_target_zarr}"/*/*/* \
        -o "${params.register_output_zarr}" \
        -c "${params.register_config}"
    """
}

process run_register {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('register') }
    maxForks 30
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { task.attempt == 1 ? '2h' : '4h' }
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd()} register --cluster debug \
        -s "${params.register_source_zarr}/${position}" \
        -t "${params.register_target_zarr}/${position}" \
        -o "${params.register_output_zarr}" \
        -c "${params.register_config}"
    """
}


workflow register_wf {
    take:
    positions
    prev_done

    main:
    init_out = init_register(prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    reg_done = positions
        .flatMap { it }
        .combine(resources)
        | run_register
        | collect

    emit:
    done = reg_done
}
