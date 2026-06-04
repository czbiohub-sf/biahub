def dataset_name() {
    return params.input_zarr ?
        new File(params.input_zarr).name.replaceAll(/\.zarr$/, '') : null
}

// Look up a step's output directory name from the pipeline steps config YAML.
// The YAML is the single source of truth for directory layout; different
// workflows (mantis-v2, dragonfly, etc.) provide their own config.
def step_dir(step_name) {
    if (!params.pipeline_steps_config) {
        error "Provide --pipeline_steps_config (path to pipeline steps YAML)"
    }
    def yaml = new org.yaml.snakeyaml.Yaml()
    def config = yaml.load(new File(params.pipeline_steps_config).text)
    return config.steps[step_name].dir
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
