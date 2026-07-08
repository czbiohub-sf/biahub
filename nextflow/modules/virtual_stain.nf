// Virtual-stain subworkflow: init + preprocess → fan-out run × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass the input zarr, output zarr,
// and config explicitly; the module has no idea where it sits in the pipeline
// directory layout. The orchestrating pipeline (see mantis-v2.nf) owns the
// layout and the order of steps; this module just virtually stains whatever
// it's handed.
//
// Since PR #267 the `biahub virtual-stain` CLI runs cytoland (modular VisCy)
// prediction IN-PROCESS, so per-position work is a single `biahub virtual-stain
// --cluster debug` call — no temp per-position zarr and no `--copy` merge step
// (the old #259 flow). `--cluster debug` makes submitit's DebugExecutor run the
// work synchronously inside the Nextflow task; Nextflow already handles
// per-position fan-out and resource scheduling, so the CLI must NOT submit its
// own SLURM jobs. See:
// examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md
//
// Three-phase pattern:
// 1. init_virtual_stain: validates the config, creates the output plate with
//    the predicted channels, emits RESOURCES:
// 2. run_virtual_stain_preprocess: `viscy preprocess` over the whole input
//    plate. virtual_stain_position reads precomputed normalization statistics
//    from the input store (viscy_data.read_norm_meta) and errors if they are
//    missing, so this must run before fan-out. NOTE: this MUTATES the input
//    store by writing normalization metadata into it.
// 3. run_virtual_stain: per-position GPU prediction using RESOURCES.
//
// Both `biahub virtual-stain` and `viscy` live in biahub's optional `stain`
// extra (cytoland → viscy-utils provides the `viscy` console script), so the
// tasks here run in that extra's environment rather than the plain biahub env.

include { parse_resources; slurm_logs; slurm_log_dir } from './common'

// Command prefix for tools that require biahub's `stain` extra. Both
// `biahub virtual-stain` (it imports cytoland) and `viscy preprocess` need it.
// Falls back to the bare tool on the active environment when biahub_project is
// unset (assumes that env already has the stain extra installed).
def stain_cmd(tool) {
    return params.biahub_project ?
        "uv run --project ${params.biahub_project} --extra stain ${tool}" : tool
}


process init_virtual_stain {
    label 'cpu_local'

    input:
    val input_zarr
    val output_zarr
    val config
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('virtual_stain')}"
    ${stain_cmd('biahub')} virtual-stain --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
    """
}

process run_virtual_stain_preprocess {
    label 'cpu'
    clusterOptions { slurm_logs('virtual_stain') }
    cpus 16
    memory { "${64 * task.attempt} GB" }
    time '1h'
    maxRetries 1
    errorStrategy 'retry'

    input:
    val input_zarr
    val trigger

    output:
    val true

    // `--trainer.logger false` disables the viscy CLI's default WandbLogger.
    // The VisCy LightningCLI sets trainer.logger to a lazy WandbLogger for every
    // subcommand (viscy_utils/cli.py); preprocess needs no logger, and W&B isn't
    // in the `stain` extra, so instantiating it fails with a missing-wandb error.
    //
    // `unset SLURM_NTASKS`: sbatch exports the submit environment, so when this
    // pipeline is launched from inside a SLURM allocation, the submit shell's
    // SLURM_NTASKS leaks into the job. Lightning's Trainer then auto-detects a
    // SLURMEnvironment and rejects SLURM_NTASKS>1 (it expects --ntasks-per-node).
    // preprocess is a single-process CPU job, so clearing it lets Lightning fall
    // back to LightningEnvironment. The dataloader uses --num_workers, not tasks.
    script:
    """
    unset SLURM_NTASKS
    ${stain_cmd('viscy')} preprocess \
        --data_path "${input_zarr}" \
        --channel_names -1 \
        --num_workers ${task.cpus} \
        --block_size 32 \
        --trainer.logger false
    """
}

process run_virtual_stain {
    tag "${position}"
    label 'gpu'
    clusterOptions { "--gres=gpu:1 " + slurm_logs('virtual_stain') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { "${meta.time_minutes * task.attempt} min" }
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val config

    output:
    val position

    script:
    """
    ${stain_cmd('biahub')} virtual-stain --cluster debug \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr (reconstruct output)
//   output_zarr  path to the virtual-stain output plate.zarr
//   config       path to the virtual-stain (viscy predict) settings YAML
//   prev_done    gating channel — virtual stain starts once this emits
workflow virtual_stain_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    prev_done

    main:
    init_out = init_virtual_stain(input_zarr, output_zarr, config, prev_done.map { 'done' })
    resources = init_out.map { parse_resources(it) }

    // Preprocess the whole plate in parallel with init; both gate the fan-out.
    vs_preprocess = run_virtual_stain_preprocess(input_zarr, prev_done.map { 'done' })

    ready = resources.combine(vs_preprocess)

    pos_meta = positions
        .flatMap { it }
        .combine(ready)
        .map { pos, meta, preprocess_done -> [pos, meta] }

    vs_done = run_virtual_stain(pos_meta, input_zarr, output_zarr, config) | collect

    emit:
    done = vs_done
}
