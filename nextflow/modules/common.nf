def dataset_name() {
    return params.input ?
        new File(params.input).name.replaceAll(/(\.ome)?\.zarr$/, '') : null
}

def parse_resources(stdout_text, prefix = 'RESOURCES:') {
    def matching = stdout_text.trim().readLines().findAll { line -> line.startsWith(prefix) }
    if (!matching) {
        error "Expected a '${prefix}' line in command output but none was found. The underlying CLI may have failed."
    }
    // The CLI emits a JSON payload (see biahub.utils.cluster.echo_resources): cpus,
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

// ENVIRONMENT CONTRACT
//
// Every process calls its CLI as a BARE command: `biahub`, plus `viscy` in
// virtual_stain.nf and `imaging-qc` in qc.nf. There is no per-task
// environment wrapper: the pipeline inherits the environment of whatever shell
// launched it, and SLURM propagates that to the compute nodes (sbatch defaults
// to --export=ALL, and the venv lives on shared storage, so the same absolute
// paths resolve on every node). Activate once before launching:
//
//     uv sync --project <BIAHUB>
//     source <BIAHUB>/.venv/bin/activate
//     nextflow run <BIAHUB>/nextflow/mantis-v2.nf ...
//
// This replaced the `biahub_cmd()` and `qc_cmd()` helpers that prefixed every
// task with `uv run --project <path>` (and, for QC, `uv run --from <git-url>`).
// Those wrappers made each of up to `maxForks` tasks re-resolve and
// re-materialize the environment concurrently against one shared site-packages
// — the QC form additionally re-fetching a git dependency per task. Resolving
// the environment once, up front, is both faster and free of that write
// contention. `check_environment()` below turns a missing activation into one
// clear launch-time error instead of N task failures.
//
// Callers pass the tools they actually invoke, so a pipeline that never runs QC
// is not held to an `imaging-qc` it would never call.
def check_environment(tools) {
    // Which install provides each command beyond `biahub` itself, so a missing
    // one names the single thing to install rather than the whole extras matrix.
    def provided_by = [
        'viscy'     : "`viscy` comes from biahub's `stain` extra, which the default `uv sync` installs via the dev dependency group.",
        'imaging-qc': "`imaging-qc` comes from biahub's `qc` extra, which is NOT in `all` — sync it explicitly with `uv sync --extra qc`.",
    ]
    tools.each { tool ->
        def proc = ['bash', '-c', "command -v ${tool}"].execute()
        proc.waitFor()
        if (proc.exitValue() != 0) {
            error """
                `${tool}` is not on PATH, so every task would fail with "command not found".

                This pipeline expects an already-activated environment. Run:
                    uv sync --project <BIAHUB>
                    source <BIAHUB>/.venv/bin/activate
                then relaunch. ${provided_by[tool] ?: ''}
                """.stripIndent()
        }
    }
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
    biahub nf list-positions -i "${input_zarr}"
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
        | map { line -> line.trim() }
        | filter { line -> line }

    // COERCE max_positions TO int, and call `take` as a method rather than
    // through the `|` pipe. A param supplied on the command line arrives as a
    // STRING ("1"), not a number — only the defaults in this repo's
    // nextflow.config are typed. `take` has no String overload, so
    // `positions.take(params.max_positions)` finds no matching operator and
    // Nextflow falls back to resolving `take` as a process/function, aborting
    // the run at launch with:
    //   Missing process or function take([DataflowStream[?], 1])
    // The default path (max_positions = 0, from config, an int) never reaches
    // `take`, so this only ever broke the `--max_positions N` smoke-test path.
    def n = (params.max_positions ?: 0) as int

    emit:
    n > 0
        ? positions.take(n).collect()
        : positions.collect()
}
