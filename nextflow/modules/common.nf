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

def biahub_cmd() {
    return params.biahub_project ?
        "uv run --project ${params.biahub_project} biahub" : "biahub"
}


// Wrap a per-position command with clean-and-retry self-healing. A per-position
// task killed mid-write (SLURM preemption / timeout / OOM, or a transient storage
// I/O error) can leave a torn or partial zarr v3 shard in the output position.
// Because flat-field / deskew / apply-inv-tf / virtual-stain write partial chunks
// via read-modify-write, the NEXT attempt (a Nextflow retry, or a later `-resume`)
// reads that torn shard back and zarr's `zarrs` codec pipeline aborts with a
// non-signal exit (1) that the global errorStrategy would otherwise `terminate`
// on — so retries and `-resume` keep dying on the same position. The same root
// cause surfaces as several different messages depending on which codec hits the
// corruption (see czbiohub-sf/iohub#415, czbiohub-sf/biahub#286):
//   "the checksum is invalid" / "encoded shard is smaller than the expected size"
//   / "blosc encoded value is invalid" / ...
//
// Rather than enumerate every message, this heals on ANY failure. A per-position
// task always fully recomputes its position from the input, so on failure it is
// always safe to remove that position's chunk data (the zarr v3 `c/` directories
// under `${output_zarr}/${position}`, preserving every `zarr.json` scaffold from
// the `--init` step) and run `cmd` once more from clean chunks: this recovers
// every torn-shard flavour (present and future) and gives transient errors a
// second chance. If the retry ALSO fails, its exit status propagates and the
// global errorStrategy handles it exactly as before — so genuine bugs are not
// masked, they just cost one extra attempt. No-op on success (nothing is
// deleted), and the delete is scoped to a single position group, so concurrent
// per-position tasks never touch each other.
//
// `cmd` must be a single shell command (no trailing backslashes): it is run, and
// re-run verbatim on failure. `|| heal_status=$?` captures the exit code without
// tripping the `-e` shell option so the retry logic can run.
def checksum_heal(output_zarr, position, cmd) {
    def pos_dir = "${output_zarr}/${position}"
    return """
    heal_status=0
    ${cmd} || heal_status=\$?
    if [ "\$heal_status" -ne 0 ]; then
        echo "[self-heal] ${position}: attempt failed (exit \$heal_status); clearing chunk data under ${pos_dir} and retrying once"
        find "${pos_dir}" -type d -name c -prune -exec rm -rf {} + 2>/dev/null || true
        ${cmd}
    fi
    """.stripIndent()
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
