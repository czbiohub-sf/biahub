// Virtual stain subworkflow: init → preprocess → fan-out (predict + copy) × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass the input zarr, output zarr,
// and config explicitly. Temp paths for predict output are derived from the
// output_zarr's parent directory.
//
// Virtual staining uses VisCy (external tool) for the GPU work, so the
// preprocess and predict steps call viscy directly rather than biahub CLI.
// Only init and copy are biahub commands:
//
// 1. init_virtual_stain: creates empty output plate with prediction channels,
//    cleans temp dir, emits RESOURCES:
// 2. run_virtual_stain_preprocess: calls `viscy preprocess` on the whole plate
// 3. run_virtual_stain: calls `viscy predict` per position, then
//    `biahub virtual-stain --copy` to merge the temp FOV zarr into the plate
//
// The predict step writes to a temp per-position zarr, and the --copy step
// moves data from that temp zarr into the output plate.

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'

def viscy_cmd() {
    return params.viscy_project ?
        "uv run --project ${params.viscy_project} viscy" :
        "uv run --from 'viscy @ git+https://github.com/mehta-lab/VisCy@v0.3.4' viscy"
}


process init_virtual_stain {
    label 'cpu_local'

    input:
    val input_zarr
    val output_zarr
    val config
    val output_dir
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('virtual_stain')}"
    rm -rf "${output_dir}/temp"
    ${biahub_cmd()} virtual-stain --init \
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

    script:
    """
    ${viscy_cmd()} preprocess \
        --data_path "${input_zarr}" \
        --channel_names -1 \
        --num_workers ${task.cpus} \
        --block_size 32
    """
}

process run_virtual_stain {
    tag "${position}"
    label 'gpu'
    clusterOptions { "--gres=gpu:1 " + slurm_logs('virtual_stain') }
    maxForks 30
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time { task.attempt == 1 ? '8h' : '12h' }
    maxRetries 2
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)
    val input_zarr
    val output_zarr
    val config
    val output_dir

    output:
    val position

    script:
    def temp_zarr = "${output_dir}/temp/${position.replaceAll('/', '_')}.zarr"
    """
    rm -rf "${temp_zarr}"

    ${viscy_cmd()} predict \
        -c "${config}" \
        --data.init_args.data_path "${input_zarr}/${position}" \
        --data.init_args.num_workers 0 \
        --trainer.callbacks+=viscy_utils.callbacks.prediction_writer.HCSPredictionWriter \
        --trainer.callbacks.output_store "${temp_zarr}" \
        --trainer.default_root_dir "${output_dir}/logs"

    ${biahub_cmd()} virtual-stain --copy \
        -t "${temp_zarr}" \
        -o "${output_zarr}" \
        -p "${position}" \
        -c "${config}"
    """
}


// take:
//   positions    collected channel of position keys (e.g. ['A/1/0', 'B/1/0'])
//   input_zarr   path to the input plate.zarr (reconstruct output)
//   output_zarr  path to the virtual stain output plate.zarr
//   config       path to the predict settings YAML
//   prev_done    gating channel — virtual stain starts once this emits
workflow virtual_stain_wf {
    take:
    positions
    input_zarr
    output_zarr
    config
    prev_done

    main:
    def output_dir = new File(output_zarr).parent

    resources = init_virtual_stain(input_zarr, output_zarr, config, output_dir, prev_done.map { 'done' })
        .map { parse_resources(it) }
    vs_preprocess = run_virtual_stain_preprocess(input_zarr, prev_done.map { 'done' })

    ready = resources.combine(vs_preprocess)

    pos_meta = positions
        .flatMap { it }
        .combine(ready)
        .map { pos, meta, preprocess_done -> [pos, meta] }

    vs_done = run_virtual_stain(pos_meta, input_zarr, output_zarr, config, output_dir) | collect

    emit:
    done = vs_done
}
