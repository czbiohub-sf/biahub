def dataset_name() {
    return params.input_zarr ?
        new File(params.input_zarr).name.replaceAll(/\.zarr$/, '') : null
}

def parse_resources(stdout_text, prefix = 'RESOURCES:') {
    def matching = stdout_text.trim().readLines().findAll { it.startsWith(prefix) }
    if (!matching) {
        error "Expected a '${prefix}' line in command output but none was found. The underlying CLI may have failed."
    }
    def parts = matching.last().replace(prefix, '').trim().split(/\s+/)
    return [cpus: parts[0].toInteger(), mem_gb: parts[1].toInteger()]
}

def slurm_log_dir(step_name) {
    return "${params.output_dir}/slurm_output/${step_name}"
}

def slurm_logs(step_name) {
    def dir = slurm_log_dir(step_name)
    return "--output=${dir}/%x_%j.out --error=${dir}/%x_%j.err"
}

def biahub_cmd() {
    return params.biahub_project ?
        "uv run --project ${params.biahub_project} biahub" : "biahub"
}

def airtable_cmd() {
    if (!params.airtable_project) {
        error "Provide --airtable_project when Airtable registry integration is enabled"
    }
    def script = "${params.airtable_project}/applications/airtable/scripts/write_experiment_metadata.py"
    return "uv run --project ${params.airtable_project} --package airtable-utils ${script}"
}

process init_chunks {
    label 'cpu_local'

    input:
    val zarr_path

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf qc init-chunks -i "${zarr_path}"
    """
}

process list_positions {
    label 'cpu_local'

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf list-positions -i "${params.input_zarr}"
    """
}
