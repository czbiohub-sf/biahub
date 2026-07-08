def dataset_name() {
    return params.input ?
        new File(params.input).name.replaceAll(/(\.ome)?\.zarr$/, '') : null
}

def parse_resources(stdout_text, prefix = 'RESOURCES:') {
    def matching = stdout_text.trim().readLines().findAll { it.startsWith(prefix) }
    if (!matching) {
        error "Expected a '${prefix}' line in command output but none was found. The underlying CLI may have failed."
    }
    // The CLI emits a JSON payload (see biahub.cli.utils.echo_resources): cpus,
    // total mem_gb, and per-position time_minutes. Parsing JSON keeps the contract
    // order-independent and extensible.
    def payload = matching.last().replace(prefix, '').trim()
    def res = new groovy.json.JsonSlurper().parseText(payload)
    return [cpus: res.cpus as int, mem_gb: res.mem_gb as int, time_minutes: res.time_minutes as int]
}

def slurm_log_dir(step_name) {
    return "${params.output}/nextflow/slurm_output/${step_name}"
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

// clusterOptions for the bulk CPU work on the preemptible `preempted` partition
// (see the slurm profile in nextflow.config). Composes the log routing from
// slurm_logs() with `--requeue`.
//
// `--requeue` is REQUIRED here, not optional: Nextflow's SLURM executor injects
// `#SBATCH --no-requeue` into every batch script (it wants to own retries), so a
// job is submitted with Requeue=0 and would be CANCELLED — not requeued — on
// preemption, despite the cluster default JobRequeue=1. clusterOptions is
// emitted as the last `#SBATCH` line, so this `--requeue` comes after Nextflow's
// `--no-requeue` and wins (SLURM honours the last of conflicting directives),
// flipping the job to Requeue=1. SLURM then requeues and reruns a preempted job
// itself, so preemption is transparent to Nextflow and never fails the pipeline.
// Ignored by the local executor, so it's a no-op under `-profile local`.
def preemptible_logs(step_name) {
    return "--requeue " + slurm_logs(step_name)
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
