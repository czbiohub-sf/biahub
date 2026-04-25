include { dataset_name; parse_resources; biahub_cmd } from './common'


process init_reconstruct {
    label 'cpu_small'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-reconstruct \
        -i "${params.output_dir}/1-deskew/${dataset_name()}.zarr" \
        -o "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        -c "${params.reconstruct_config}" \
        -j ${params.num_threads}
    """
}

process compute_transfer_function {
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    queue 'cpu'

    input:
    val meta

    output:
    val true

    script:
    """
    ${biahub_cmd()} nf compute-transfer-function \
        -i "${params.output_dir}/1-deskew/${dataset_name()}.zarr" \
        -t "${params.output_dir}/2-reconstruct/transfer_function_${dataset_name()}.zarr" \
        -c "${params.output_dir}/2-reconstruct/reconstruct_resolved.yml"
    """
}

process run_apply_inv_tf {
    tag "${position}"
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '6h'
    queue 'cpu'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd()} nf run-apply-inv-tf \
        -i "${params.output_dir}/1-deskew/${dataset_name()}.zarr" \
        -o "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr" \
        -t "${params.output_dir}/2-reconstruct/transfer_function_${dataset_name()}.zarr" \
        -p "${position}" \
        -c "${params.output_dir}/2-reconstruct/reconstruct_resolved.yml" \
        -j ${params.num_threads}
    """
}


workflow reconstruct_wf {
    take:
    positions
    prev_done

    main:
    init_out = init_reconstruct(prev_done.map { 'done' })
    tf_resources = init_out.map { parse_resources(it, 'TF_RESOURCES:') }
    run_resources = init_out.map { parse_resources(it) }

    tf_done = compute_transfer_function(tf_resources)

    rc_done = positions
        .flatMap { it }
        .combine(run_resources)
        .combine(tf_done)
        .map { pos, meta, tf -> [pos, meta] }
        | run_apply_inv_tf
        | collect

    emit:
    done = rc_done
}
