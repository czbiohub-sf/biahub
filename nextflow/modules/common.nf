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


// Wrap a per-position command with self-healing for zarr's "checksum is invalid"
// error. A per-position task killed mid-write (SLURM preemption / timeout / OOM
// — the 130..145 signal exits the global errorStrategy already retries) can
// leave a torn zarr v3 shard in the output position. flat-field / deskew /
// apply-inv-tf / virtual-stain all write partial chunks via read-modify-write,
// so the NEXT attempt (a Nextflow retry, or a later `-resume`) reads that torn
// shard back and zarr's codec pipeline raises `RuntimeError: the checksum is
// invalid` — a non-signal exit (1) that the global errorStrategy would otherwise
// `terminate` on, so retries and `-resume` both keep dying on the same position.
//
// This wrapper runs `cmd`, capturing its output. On failure it inspects the log
// and ONLY if it sees "checksum is invalid" does it remove this position's chunk
// data (the zarr v3 `c/` directories under `${output_zarr}/${position}`) while
// preserving every `zarr.json` scaffold created by the `--init` step, then runs
// `cmd` once more so the write starts from clean chunks. Any OTHER failure is
// re-raised with its original exit status, so the global errorStrategy handles
// it exactly as before. Safe and a no-op unless corruption is actually detected:
// nothing is deleted when `cmd` succeeds, and the delete is scoped to a single
// position group, so concurrent per-position tasks never touch each other.
//
// `cmd` must be a single-line shell command string (no trailing backslashes),
// since it is spliced into an `if !` pipeline and re-run verbatim. `set -o
// pipefail` is required so the pipeline's status reflects `cmd`, not `tee`.
def checksum_heal(output_zarr, position, cmd) {
    def pos_dir = "${output_zarr}/${position}"
    return """
    set -o pipefail
    if ! ${cmd} 2>&1 | tee .checksum_heal.log; then
        heal_status=\${PIPESTATUS[0]}
        if grep -q 'checksum is invalid' .checksum_heal.log; then
            echo "[checksum-heal] corrupt output chunk detected for ${position}; clearing chunk data under ${pos_dir} and retrying once"
            find "${pos_dir}" -type d -name c -prune -exec rm -rf {} + 2>/dev/null || true
            ${cmd}
        else
            exit \$heal_status
        fi
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
