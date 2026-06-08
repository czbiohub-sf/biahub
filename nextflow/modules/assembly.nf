// Assembly subworkflow: resolve config → init → fan-out concatenate × N positions.
//
// This subworkflow is PATH-AGNOSTIC. Callers pass explicit source zarr paths
// (deskew, reconstruct, virtual-stain), output zarr, and config. Uses named
// vals for the 3 source zarrs (Decision 2: self-documenting, no ordering
// ambiguity).
//
// Assembly reads from 3 upstream zarrs and uses them in different ways:
// - init_estimate_crop: deskew (LF) + reconstruct (LS) for crop estimation
// - resolve_concatenate_config / reduce_crop_ranges: all 3 for concatenation
// - init_concatenate / run_concatenate: output zarr only

include { parse_resources; biahub_cmd; slurm_logs; slurm_log_dir } from './common'


process init_estimate_crop {
    label 'cpu_local'

    input:
    val deskew_zarr
    val reconstruct_zarr
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('assemble')}"
    ${biahub_cmd()} estimate-crop --init \
        --lf-data-path "${deskew_zarr}/*/*/*" \
        --ls-data-path "${reconstruct_zarr}/*/*/*"
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
    ${biahub_cmd()} estimate-crop \
        --lf-position "${lf_position}" \
        --ls-position "${ls_position}"
    """
}

process reduce_crop_ranges {
    label 'cpu_local'

    input:
    path ranges_file
    val deskew_zarr
    val reconstruct_zarr
    val virtual_stain_zarr
    val output_dir
    val config
    val trigger

    output:
    val true

    script:
    """
    ${biahub_cmd()} estimate-crop --reduce \
        -c "${config}" \
        -o "${output_dir}/concatenate_cropped.yml" \
        --ranges-file "${ranges_file}" \
        --concat-data-paths "${deskew_zarr}/*/*/*" \
        --concat-data-paths "${reconstruct_zarr}/*/*/*" \
        --concat-data-paths "${virtual_stain_zarr}/*/*/*"
    """
}

process resolve_concatenate_config {
    label 'cpu_local'

    input:
    val deskew_zarr
    val reconstruct_zarr
    val virtual_stain_zarr
    val output_dir
    val config
    val trigger

    output:
    path "concatenate_resolved.yml"

    script:
    def resolved = "${output_dir}/concatenate_resolved.yml"
    """
    mkdir -p "${output_dir}"
    ${biahub_cmd()} concatenate --resolve-config \
        -c "${config}" \
        -o "${resolved}" \
        --concat-data-paths "${deskew_zarr}/*/*/*" \
        --concat-data-paths "${reconstruct_zarr}/*/*/*" \
        --concat-data-paths "${virtual_stain_zarr}/*/*/*"
    cp "${resolved}" concatenate_resolved.yml
    """
}

process init_concatenate {
    label 'cpu_local'

    input:
    path resolved_config
    val output_zarr

    output:
    stdout

    script:
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
    val output_zarr
    val resolved_config_path

    output:
    val position

    script:
    """
    ${biahub_cmd()} concatenate --cluster debug \
        -c "${resolved_config_path}" \
        -o "${output_zarr}" \
        -p "${position}"
    """
}


// take:
//   positions          collected channel of position keys
//   deskew_zarr        LF source (for estimate-crop --lf-data-path and resolve/reduce)
//   reconstruct_zarr   LS source (for estimate-crop --ls-data-path and resolve/reduce)
//   virtual_stain_zarr VS source (for resolve/reduce --concat-data-paths)
//   output_zarr        path to assembled output plate.zarr
//   config             path to concatenate config YAML
//   prev_done          gating channel
workflow assemble_wf_mantisv2 {
    take:
    positions
    deskew_zarr
    reconstruct_zarr
    virtual_stain_zarr
    output_zarr
    config
    prev_done

    main:
    def output_dir = new File(output_zarr).parent
    def resolved_config_path = "${output_dir}/concatenate_resolved.yml"

    resolved_config = resolve_concatenate_config(
        deskew_zarr, reconstruct_zarr, virtual_stain_zarr,
        output_dir, config, prev_done.map { 'done' }
    )
    resources = init_concatenate(resolved_config, output_zarr).map { parse_resources(it) }

    pos_meta = positions
        .flatMap { it }
        .combine(resources)

    as_done = run_concatenate(pos_meta, output_zarr, resolved_config_path) | collect

    emit:
    done = as_done
}
