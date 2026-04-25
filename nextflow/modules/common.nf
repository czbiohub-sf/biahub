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

def biahub_cmd() {
    return params.biahub_project ?
        "uv run --project ${params.biahub_project} biahub" : "biahub"
}

process list_positions {
    label 'cpu_small'

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf list-positions -i "${params.input_zarr}"
    """
}
