include { dataset_name; airtable_cmd; slurm_log_dir } from './common'


process register_airtable_after_tracking {
    label 'cpu_local'

    input:
    val trigger

    output:
    val true

    script:
    def assembled_zarr = "${params.output_dir}/5-assemble/${dataset_name()}.zarr"
    def dry_run_flag = params.airtable_registry_dry_run ? "--dry-run" : ""
    def dataset_flag = params.airtable_registry_dataset ?
        "--dataset \"${params.airtable_registry_dataset}\"" : ""
    """
    mkdir -p "${slurm_log_dir('airtable_registry')}"

    shopt -s nullglob
    positions=( "${assembled_zarr}"/*/*/* )
    if [ \${#positions[@]} -eq 0 ]; then
        echo "No assembled positions found under ${assembled_zarr}" >&2
        exit 1
    fi

    ${airtable_cmd()} register ${dry_run_flag} ${dataset_flag} "\${positions[@]}"
    ${airtable_cmd()} write ${dry_run_flag} "\${positions[@]}"
    """
}


workflow airtable_registry_wf {
    take:
    trigger

    main:
    if (params.airtable_registry_after_tracking && !params.airtable_project) {
        error "Provide --airtable_project when --airtable_registry_after_tracking is enabled"
    }

    registry_done = params.airtable_registry_after_tracking
        ? register_airtable_after_tracking(trigger)
        : trigger

    emit:
    done = registry_done
}
