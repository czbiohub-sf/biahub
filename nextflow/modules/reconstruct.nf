// Reconstruct subworkflows: init, transfer function, and fan-out × N positions.
//
// This module is PATH-AGNOSTIC. Callers pass the input zarr, output zarr, and
// config explicitly; the module has no idea where it sits in the pipeline
// directory layout. The TF zarr path is derived from the output_zarr's parent
// directory (the module's internal convention).
//
// The config is the single source of truth and must carry the pixel sizes; the
// CLI validates it and only warns if those pixel sizes disagree with the input
// zarr metadata.  No resolved config is written.
//
// Three phases, THREE SUBWORKFLOWS, because the pipeline schedules each of them
// at a different time (see mantis-v2.nf):
// 1. reconstruct_init_wf: validates the config, creates the output plate, emits
//    RESOURCES. Metadata-only, head node, part of the up-front init phase.
// 2. reconstruct_tf_wf: one-shot TF computation on hardcoded resources. This is
//    real compute, but it depends only on the input plate's SHAPE — waveorder
//    builds the transfer function from the first position's dimensions, reading
//    no pixels — so the pipeline starts it as soon as the input store has been
//    SCAFFOLDED, and it overlaps the upstream steps' compute instead of sitting
//    on the critical path between them and reconstruction.
// 3. reconstruct_run_wf: per-position inverse TF application, gated on the TF.
//
// run_apply_inv_tf uses `--cluster debug` so that submitit's DebugExecutor runs
// the work in-process.  Nextflow already handles per-position fan-out and
// resource scheduling, so the CLI must NOT submit its own SLURM jobs.
// See: examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md

include { parse_resources; slurm_logs; slurm_log_dir } from './common'


process init_apply_inv_tf {
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
    mkdir -p "${slurm_log_dir('reconstruct')}"
    biahub apply-inv-tf --init \
        -i "${input_zarr}"/*/*/* \
        -o "${output_zarr}" \
        -c "${config}"
    """
}

process compute_transfer_function {
    label 'cpu'
    clusterOptions { slurm_logs('reconstruct') }
    // Hardcoded resources for the one-shot transfer-function computation.
    //
    // waveorder upsamples the volume to Nyquist internally before building the
    // TF (waveorder/models/phase_thick_3d.py::calculate_transfer_function); the
    // peak footprint scales with that upsampled volume, not the input volume.
    // For properly-sampled label-free data the Nyquist upsampling factor is 1
    // in every axis, so even for the largest volume we expect
    // (~2048 x 2048 x 128 = 2 GB float32) the phase TF needs ~64 GB (2 GB x
    // waveorder's x32 Fourier multiplier).
    //
    // The TF computation is torch-CPU-FFT-bound (large 3D FFTs over the
    // upsampled volume in optics.compute_weak_object_transfer_function_3D) and
    // is not thread-pinned in the compute-tf path, so torch parallelizes the
    // FFTs across the granted cores.  8 sits in the sweet spot before FFT
    // thread-scaling tails off.
    cpus 8
    memory '64 GB'
    time '30m'

    input:
    val trigger
    val input_zarr
    val tf_zarr
    val config

    output:
    val true

    script:
    """
    biahub compute-tf \
        -i "${input_zarr}"/*/*/* \
        -o "${tf_zarr}" \
        -c "${config}"
    """
}

process run_apply_inv_tf {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('reconstruct') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { "${meta.time_minutes * task.attempt} min" }

    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val tf_zarr
    val config

    output:
    val position

    script:
    """
    biahub apply-inv-tf --cluster debug \
        -i "${input_zarr}/${position}" \
        -t "${tf_zarr}" \
        -o "${output_zarr}" \
        -c "${config}"
    """
}


// The TF zarr path, derived from the output plate's parent directory. Exposed
// as a function because both reconstruct_tf_wf and reconstruct_run_wf need it
// and the pipeline invokes them separately.
def transfer_function_path(output_zarr) {
    return "${new File(output_zarr).parent}/transfer_function.zarr"
}


// Validate the config and scaffold the output plate. Metadata-only and cheap,
// so it belongs in the pipeline's up-front init phase.
//
// take:
//   input_zarr   path to the input plate.zarr (deskew output)
//   output_zarr  path to the reconstructed output plate.zarr
//   config       path to the reconstruct settings YAML
//   trigger      gating channel — init starts once this emits
// emit:
//   resources    the RESOURCES payload sizing one position's task
//   done         fires once the output plate exists
workflow reconstruct_init_wf {
    take:
    input_zarr
    output_zarr
    config
    trigger

    main:
    init_out = init_apply_inv_tf(input_zarr, output_zarr, config, trigger.map { 'done' })

    emit:
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }.first()
    done      = init_out.map { 'done' }.first()
}


// Compute the transfer function once for the whole plate.
//
// take:
//   input_zarr   path to the input plate.zarr (deskew output); only its shape
//                is read, so a scaffolded-but-empty store is enough
//   output_zarr  path to the reconstructed output plate.zarr (locates the TF)
//   config       path to the reconstruct settings YAML
//   trigger      gating channel — the TF starts once this emits
workflow reconstruct_tf_wf {
    take:
    input_zarr
    output_zarr
    config
    trigger

    main:
    tf_zarr = transfer_function_path(output_zarr)
    tf_done = compute_transfer_function(trigger.map { 'done' }, input_zarr, tf_zarr, config)

    emit:
    done = tf_done.first()
}


// Fan out one inverse-TF task per position.
//
// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr (deskew output)
//   output_zarr  path to the reconstructed output plate.zarr
//   config       path to the reconstruct settings YAML
//   resources    RESOURCES payload from reconstruct_init_wf
//   tf_done      gating channel — the transfer function exists
//   prev_done    gating channel — the input store holds data
workflow reconstruct_run_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    resources
    tf_done
    prev_done

    main:
    tf_zarr = transfer_function_path(output_zarr)

    pos_meta = positions
        .flatMap { items -> items }
        .combine(resources)
        .combine(tf_done)
        .combine(prev_done)
        .map { pos, meta, _tf, _gate -> [pos, meta] }

    rc_done = run_apply_inv_tf(pos_meta, input_zarr, output_zarr, tf_zarr, config) | collect

    emit:
    done = rc_done
}
