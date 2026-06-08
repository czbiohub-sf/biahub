// Registration subworkflow: estimate → init → fan-out run × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass source zarr, target zarr,
// output zarr, and config paths explicitly.
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
    val source_zarr
    val target_zarr
    val register_config
    val estimate_config
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd()} estimate-registration \
        -s "${source_zarr}"/*/*/* \
        -t "${target_zarr}"/*/*/* \
        -o "${register_config}" \
        -c "${estimate_config}"
    """
}

process init_register {
    label 'cpu_local'

    input:
    val source_zarr
    val target_zarr
    val output_zarr
    val register_config
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('register')}"
    ${biahub_cmd()} register --init \
        -s "${source_zarr}"/*/*/* \
        -t "${target_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${register_config}"
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
    val source_zarr
    val target_zarr
    val output_zarr
    val register_config

    output:
    val position

    script:
    """
    ${biahub_cmd()} register --cluster debug \
        -s "${source_zarr}/${position}" \
        -t "${target_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${register_config}"
    """
}


// take:
//   positions        collected channel of position keys
//   source_zarr      path to the source plate.zarr
//   target_zarr      path to the target plate.zarr
//   output_zarr      path to the registered output plate.zarr
//   estimate_config  path to the estimate-registration settings YAML
//   register_config  path to the registration config YAML (written by estimate)
//   prev_done        gating channel
workflow register_wf {
    take:
    positions
    source_zarr
    target_zarr
    output_zarr
    estimate_config
    register_config
    prev_done

    main:
    init_out = init_register(source_zarr, target_zarr, output_zarr, register_config, prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    pos_meta = positions
        .flatMap { it }
        .combine(resources)

    reg_done = run_register(pos_meta, source_zarr, target_zarr, output_zarr, register_config) | collect

    emit:
    done = reg_done
}
