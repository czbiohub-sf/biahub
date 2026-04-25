include { dataset_name; parse_resources; biahub_cmd } from './common'

def viscy_cmd() {
    return params.viscy_project ?
        "uv run --project ${params.viscy_project} viscy" :
        "uv run --from 'viscy @ git+https://github.com/mehta-lab/VisCy@v0.3.4' viscy"
}


process init_virtual_stain {
    label 'cpu_small'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-virtual-stain \
        -i "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        -o "${params.output_dir}/3-virtual-stain/${dataset_name()}.zarr" \
        -c "${params.predict_config}"
    """
}

process run_virtual_stain_preprocess {
    label 'cpu_medium'

    input:
    val trigger

    output:
    val true

    script:
    """
    ${viscy_cmd()} preprocess \
        --data_path "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        --channel_names -1 \
        --num_workers ${task.cpus} \
        --block_size 32
    """
}

process run_virtual_stain {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'gpu'
    clusterOptions '--gres=gpu:1'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    def temp_zarr = "${params.output_dir}/3-virtual-stain/temp/${position.replaceAll('/', '_')}.zarr"
    """
    ${viscy_cmd()} predict \
        -c "${params.predict_config}" \
        --data.init_args.data_path "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr/${position}" \
        --trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter \
        --trainer.callbacks.output_store "${temp_zarr}" \
        --trainer.default_root_dir "${params.output_dir}/3-virtual-stain/logs"

    ${biahub_cmd()} nf copy-virtual-stain \
        -t "${temp_zarr}" \
        -o "${params.output_dir}/3-virtual-stain/${dataset_name()}.zarr" \
        -p "${position}"
    """
}


workflow virtual_stain_wf {
    take:
    positions
    prev_done

    main:
    resources = init_virtual_stain(prev_done.map { 'done' }).map { parse_resources(it) }
    vs_preprocess = run_virtual_stain_preprocess(prev_done.map { 'done' })

    ready = resources.combine(vs_preprocess)

    vs_done = positions
        .flatMap { it }
        .combine(ready)
        .map { pos, meta, preprocess_done -> [pos, meta] }
        | run_virtual_stain
        | collect

    emit:
    done = vs_done
}
