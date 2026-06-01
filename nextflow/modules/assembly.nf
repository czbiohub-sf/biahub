include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_estimate_crop {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('assemble')}"
    ${biahub_cmd()} nf init-estimate-crop \
        --lf-data-path "${params.output_dir}/1-deskew/${dataset_name()}.zarr/*/*/*" \
        --ls-data-path "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr/*/*/*"
    """
}

process estimate_crop {
    tag "${lf_position}"
    label 'cpu'
    clusterOptions { slurm_logs('assemble') }
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '1h'
    maxRetries 1
    errorStrategy 'retry'

    input:
    tuple val(lf_position), val(ls_position), val(meta)

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf estimate-crop \
        --lf-position "${lf_position}" \
        --ls-position "${ls_position}"
    """
}

process reduce_crop_ranges {
    label 'cpu_local'

    input:
    path ranges_file
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd()} nf reduce-crop-ranges \
        -c "${params.concatenate_config}" \
        -o "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        --ranges-file "${ranges_file}" \
        --concat-data-paths "${params.output_dir}/1-deskew/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/3-virtual-stain/${dataset_name()}.zarr/*/*/*"
    """
}

process resolve_concatenate_config {
    label 'cpu_local'

    input:
    val trigger

    output:
    path "concatenate_resolved.yml"

    script:
    def resolved = "${params.output_dir}/5-assemble/concatenate_resolved.yml"
    """
    ${biahub_cmd()} nf resolve-concatenate-config \
        -c "${params.concatenate_config}" \
        -o "${resolved}" \
        --concat-data-paths "${params.output_dir}/1-deskew/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/3-virtual-stain/${dataset_name()}.zarr/*/*/*"
    cp "${resolved}" concatenate_resolved.yml
    """
}

process init_concatenate {
    label 'cpu_local'

    input:
    path resolved_config

    output:
    stdout

    script:
    def output_zarr = "${params.output_dir}/5-assemble/${dataset_name()}.zarr"
    """
    rm -rf "${output_zarr}"
    ${biahub_cmd()} nf init-concatenate \
        -c "${resolved_config}" \
        -o "${output_zarr}"
    """
}

process run_concatenate {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('assemble') }
    maxForks 30
    cpus { meta.cpus }
    memory { "${meta.mem_gb} GB" }
    time '2h'
    maxRetries 1
    errorStrategy 'retry'


    input:
    tuple val(position), val(meta)

    output:
    val position

    script:
    """
    ${biahub_cmd()} nf run-concatenate \
        -c "${params.output_dir}/5-assemble/concatenate_resolved.yml" \
        -o "${params.output_dir}/5-assemble/${dataset_name()}.zarr" \
        -p "${position}"
    """
}


process clean_intermediates {
    label 'cpu_local'

    input:
    val trigger

    script:
    """
    ${biahub_cmd()} nf clean-intermediates \
        -o "${params.output_dir}" \
        -d "${dataset_name()}"
    """
}


workflow assemble_wf_mantisv2 {
    take:
    positions
    prev_done

    main:
    resolved_config = resolve_concatenate_config(prev_done.map { 'done' })
    resources = init_concatenate(resolved_config).map { parse_resources(it) }
    as_done = positions
        .flatMap { it }
        .combine(resources)
        | run_concatenate
        | collect

    emit:
    done = as_done
}
