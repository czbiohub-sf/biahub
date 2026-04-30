include { dataset_name; parse_resources; biahub_cmd } from './common'


process init_estimate_crop {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-estimate-crop \
        --lf-data-path "${params.output_dir}/1-deskew/${dataset_name()}.zarr/*/*/*" \
        --ls-data-path "${params.output_dir}/2-reconstruct/${dataset_name()}.zarr/*/*/*"
    """
}

process estimate_crop {
    tag "${lf_position}"
    label 'cpu'
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

process init_concatenate {
    label 'cpu_local'

    input:
    val trigger

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf init-concatenate \
        -c "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        -o "${params.output_dir}/5-assemble/${dataset_name()}.zarr"
    """
}

process run_concatenate {
    tag "${position}"
    label 'cpu'
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
    def output_zarr = "${params.output_dir}/5-assemble/${dataset_name()}.zarr"
    """
    if [ ${task.attempt} -gt 1 ]; then
        ${biahub_cmd()} nf clean-position -o "${output_zarr}" -p "${position}"
    fi
    ${biahub_cmd()} nf run-concatenate \
        -c "${params.output_dir}/5-assemble/concatenate_cropped.yml" \
        -o "${output_zarr}" \
        -p "${position}"
    """
}


def parse_positions(stdout_text) {
    return stdout_text.trim().readLines()
        .findAll { it.startsWith('POSITION:') }
        .collect { line ->
            def parts = line.replace('POSITION:', '').trim().split('\t')
            [parts[0], parts[1]]
        }
}


workflow assemble_wf {
    take:
    positions
    prev_done

    main:
    init_out = init_estimate_crop(prev_done.map { 'done' })
    crop_resources = init_out.map { parse_resources(it) }
    crop_positions = init_out
        .flatMap { parse_positions(it) }
        .combine(crop_resources)

    ranges_ch = estimate_crop(crop_positions)
        | collectFile(name: 'all_ranges.txt', newLine: true)

    crop_done = reduce_crop_ranges(ranges_ch, ranges_ch.map { 'done' })

    resources = init_concatenate(crop_done).map { parse_resources(it) }
    as_done = positions
        .flatMap { it }
        .combine(resources)
        | run_concatenate
        | collect

    emit:
    done = as_done
}
