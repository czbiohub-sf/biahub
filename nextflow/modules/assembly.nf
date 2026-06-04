include { dataset_name; parse_resources; biahub_cmd; slurm_logs; slurm_log_dir; step_dir } from './common'


process init_estimate_crop {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    // biahub estimate-crop --init lists LF/LS position pairs and emits RESOURCES.
    """
    mkdir -p "${slurm_log_dir('assemble')}"
    ${biahub_cmd()} estimate-crop --init \
        --lf-data-path "${params.output_dir}/${step_dir('deskew')}/${dataset_name()}.zarr/*/*/*" \
        --ls-data-path "${params.output_dir}/${step_dir('reconstruct')}/${dataset_name()}.zarr/*/*/*"
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
    // Per-FOV crop estimation: calls estimate_crop_one_position, emits RANGES.
    """
    ${biahub_cmd()} estimate-crop \
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
    // Aggregate per-FOV ranges into a global standardized crop config.
    """
    ${biahub_cmd()} estimate-crop --reduce \
        -c "${params.concatenate_config}" \
        -o "${params.output_dir}/${step_dir('assemble')}/concatenate_cropped.yml" \
        --ranges-file "${ranges_file}" \
        --concat-data-paths "${params.output_dir}/${step_dir('deskew')}/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/${step_dir('reconstruct')}/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/${step_dir('virtual_stain')}/${dataset_name()}.zarr/*/*/*"
    """
}

process resolve_concatenate_config {
    label 'cpu_local'

    input:
    val trigger

    output:
    path "concatenate_resolved.yml"

    script:
    // Resolve placeholder concat_data_paths to actual glob patterns.
    def resolved = "${params.output_dir}/${step_dir('assemble')}/concatenate_resolved.yml"
    """
    mkdir -p "${params.output_dir}/5-assemble"
    ${biahub_cmd()} concatenate --resolve-config \
        -c "${params.concatenate_config}" \
        -o "${resolved}" \
        --concat-data-paths "${params.output_dir}/${step_dir('deskew')}/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/${step_dir('reconstruct')}/${dataset_name()}.zarr/*/*/*" \
        --concat-data-paths "${params.output_dir}/${step_dir('virtual_stain')}/${dataset_name()}.zarr/*/*/*"
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
    // biahub concatenate --init creates the empty output plate and emits RESOURCES.
    def output_zarr = "${params.output_dir}/${step_dir('assemble')}/${dataset_name()}.zarr"
    """
    rm -rf "${output_zarr}"
    ${biahub_cmd()} concatenate --init \
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
    // --cluster debug: run in-process; Nextflow handles per-position fan-out.
    """
    ${biahub_cmd()} concatenate --cluster debug \
        -c "${params.output_dir}/${step_dir('assemble')}/concatenate_resolved.yml" \
        -o "${params.output_dir}/${step_dir('assemble')}/${dataset_name()}.zarr" \
        -p "${position}"
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
