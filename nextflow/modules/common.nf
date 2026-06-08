def dataset_name() {
    return params.input ?
        new File(params.input).name.replaceAll(/(\.ome)?\.zarr$/, '') : null
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
    return "${params.output}/slurm_output/${step_name}"
}

def slurm_logs(step_name) {
    def dir = slurm_log_dir(step_name)
    // NOTE: the --output/--error targets are intentionally CROSSED.
    // Nextflow's task launcher tees the job's streams with an fd swap
    // (`... 3>&1 1>&2 2>&3 ...` in .command.run) so it can write the task's
    // stdout to .command.out and stderr to .command.err. A side effect is that
    // the *batch script's* own stdout/stderr streams — the ones SLURM captures
    // via --output/--error — are swapped relative to the program's streams:
    // the --output stream carries the program's stderr and vice versa.
    // Mapping --output to the .err file and --error to the .out file undoes
    // that swap so each file ends up with the stream its name implies.
    return "--output=${dir}/%x_%j.err --error=${dir}/%x_%j.out"
}

def biahub_cmd() {
    return params.biahub_project ?
        "uv run --project ${params.biahub_project} biahub" : "biahub"
}


// List the position keys of a plate zarr, one per line, for fan-out.
process list_positions {
    label 'cpu_local'

    input:
    val input_zarr

    output:
    stdout

    script:
    """
    ${biahub_cmd()} nf list-positions -i "${input_zarr}"
    """
}


// Collect position keys from a plate zarr into a single list channel for
// per-position fan-out. Shared by every pipeline (mantis-v2, dragonfly, …);
// honours params.max_positions (0 = all) for quick test runs. `input_zarr` is
// the zarr to fan out over — for pipelines that convert raw input first, that's
// the convert output, not the pipeline's raw `input`.
workflow collect_positions {
    take:
    input_zarr

    main:
    positions = list_positions(input_zarr)
        | splitText
        | map { it.trim() }
        | filter { it }

    out = params.max_positions > 0
        ? positions | take(params.max_positions) | collect
        : positions | collect

    emit:
    out
}
