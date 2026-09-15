// Virtual-stain subworkflows: init (scaffold + resources) and run (preprocess +
// fan-out × N).
//
// This module is PATH-AGNOSTIC. Callers pass the input zarr, output zarr, and
// config explicitly; the module has no idea where it sits in the pipeline
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
// Three phases, split across TWO subworkflows by whether they need DATA:
// 1. init_virtual_stain: validates the predict config against VisCy's schema,
//    creates the output plate with the predicted channels, emits RESOURCES.
//    Metadata-only, so it joins the pipeline's up-front init phase.
// 2. run_virtual_stain_preprocess: `viscy preprocess` over the whole input
//    plate. virtual_stain_position reads precomputed normalization statistics
//    from the input store (viscy_data.read_norm_meta) and errors if they are
//    missing, so this must run before fan-out. NOTE: this MUTATES the input
//    store by writing normalization metadata into it, and it reads every
//    position's PIXELS — so unlike the other one-shot steps it cannot be
//    hoisted, and it is what keeps a whole-plate barrier ahead of this step
//    (biahub#304).
// 3. run_virtual_stain: per-position GPU prediction using RESOURCES.
//
// Both `biahub virtual-stain` and `viscy` live in biahub's optional `stain`
// extra (cytoland → viscy-utils provides the `viscy` console script). The
// activated environment must therefore carry that extra — the default
// `uv sync` does, because the `dev` dependency group depends on `biahub[all]`.
// See the ENVIRONMENT CONTRACT note in common.nf; these tasks call `biahub` and
// `viscy` bare, exactly like every other step.

include { parse_resources; slurm_logs; slurm_log_dir } from './common'


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
    biahub virtual-stain --init \
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
    viscy preprocess \
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

    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val config

    output:
    val position

    script:
    """
    biahub virtual-stain --cluster debug \
        -i "${input_zarr}/${position}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// Validate the predict config and scaffold the output plate. Metadata-only and
// cheap, so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the input plate.zarr (reconstruct output)
//   output_zarr  path to the virtual-stain output plate.zarr
//   config       path to the virtual-stain (viscy predict) settings YAML
//   trigger      gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once the output plate exists
workflow virtual_stain_init_wf {
    take:
    input_zarr
    output_zarr
    config
    trigger

    main:
    init_out = init_virtual_stain(input_zarr, output_zarr, config, trigger.map { 'done' })

    emit:
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }.first()
    done      = init_out.map { 'done' }.first()
}


// Compute normalization statistics over the whole input plate, then fan out one
// GPU prediction per position.
//
// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr (reconstruct output)
//   output_zarr  path to the virtual-stain output plate.zarr
//   config       path to the virtual-stain (viscy predict) settings YAML
//   resources    RESOURCES payload from virtual_stain_init_wf
//   prev_done    gating channel — the input store holds data for EVERY position,
//                which preprocess requires
workflow virtual_stain_run_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    resources
    prev_done

    main:
    vs_preprocess = run_virtual_stain_preprocess(input_zarr, prev_done.map { 'done' })

    pos_meta = positions
        .flatMap { items -> items }
        .combine(resources)
        .combine(vs_preprocess.first())
        .map { pos, meta, _preprocess_done -> [pos, meta] }

    vs_done = run_virtual_stain(pos_meta, input_zarr, output_zarr, config) | collect

    emit:
    done = vs_done
}
