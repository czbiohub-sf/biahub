// Slack notifications for a pipeline run.
//
// Three kinds of message, all posted by `biahub nf notify` (biahub/utils/notify.py):
//
//   run start        notify_run_start()  — once, before any work is submitted
//   step complete    notify_step         — once per step, as each one finishes
//   run end          notify_run_end()    — once, from workflow.onComplete
//
// WHY notify_step IS A PROCESS AND NOT A `.subscribe { }` ON THE done CHANNEL
//
// A `-resume` run replays the entire dataflow graph: cached tasks still emit
// their outputs, so `collect` still emits and a subscribe closure would fire for
// every step within seconds of relaunching — six "step complete" messages for
// work that was merely restored from cache. (Measured: a fully-cached relaunch
// reaches 341/341 CACHED in about nine seconds.)
//
// Making this an ordinary CACHEABLE process solves that with no state of its own:
// the notify task's inputs are unchanged, so on a resume it is cached too, so it
// does not execute, so no message is sent. Nextflow's own task cache IS the
// "already announced" record — it is per-output-directory by construction and it
// survives `-resume`, which is exactly the semantics wanted here. Only steps that
// genuinely re-ran announce themselves again.
//
// A subscribe closure would also run on a dataflow thread, putting a blocking
// HTTP POST directly between one step finishing and the next being submitted.
//
// CREDENTIALS
//
// The webhook URL ($BIAHUB_SLACK_WEBHOOK) and the operator's member ID
// ($BIAHUB_SLACK_ID) are read from the environment by the Python, never
// interpolated into a script body. So neither reaches `.command.sh`, the work
// directory, `.nextflow.log`, or a task hash — and changing either one never
// invalidates a cached task. The environment reaches every task per the
// ENVIRONMENT CONTRACT in common.nf (sbatch defaults to --export=ALL).

// One message per completed step. See the header for why this is a process.
//
// Resources and error handling are pinned by `withName: 'notify_step'` in
// nextflow.config rather than by a label: the slurm profile's
// `withLabel: 'cpu_local'` block would otherwise clobber them, and six 12 GB
// local tasks would contend with the init steps for the head node.
process notify_step {
    tag "${step}"

    input:
    tuple val(step), val(n_positions), val(output_zarr), val(index)
    val dataset

    output:
    val step

    script:
    // Emoji as a Slack shortcode rather than raw UTF-8: it renders the same and
    // keeps the payload ASCII.
    def positions = n_positions > 1 ? " (${n_positions} positions)" : ""
    // No --ping. Step messages post silently so the @-mention on the run-end
    // message keeps meaning "this needs you".
    // `|| true` so nothing the notifier does can fail the task, independent of
    // the errorStrategy.
    """
    biahub nf notify \\
        --level good \\
        --title ":white_check_mark: ${dataset} — ${step} complete [${index}]${positions}" \\
        --detail "output: ${output_zarr}" || true
    """
}

// Announce the run before any work is submitted.
//
// This doubles as a webhook smoke test: without it the first Slack message is
// the completion one, hours or days later, so a revoked webhook would be
// discovered far too late to be useful.
//
// Rate-limited by key, because debugging a config produces relaunch storms —
// five launches inside ten minutes is an observed pattern, which would otherwise
// be five notifications.
def notify_run_start(dataset) {
    // Say once, at launch, that Slack is not configured. Without a webhook every
    // message still prints and every exit status is still 0, but only the
    // run-level messages reach the console: notify_step runs as a task, so its
    // text goes to the task's .command.out and slurm_output/<step>/*.out, which
    // nobody watching the pane would think to look in.
    if (!System.getenv('BIAHUB_SLACK_WEBHOOK')) {
        log.warn "BIAHUB_SLACK_WEBHOOK unset — Slack notifications disabled. " +
            "Step messages will only appear in nextflow/slurm_output/<step>/*.out."
    }
    else if (!System.getenv('BIAHUB_SLACK_ID')) {
        log.warn "BIAHUB_SLACK_ID unset — the run-end message will not @-mention anyone."
    }

    def detail = [
        "input:  ${params.input}",
        "output: ${params.output}",
        "host:   ${java.net.InetAddress.localHost.hostName}",
        // `as int` because a param given on the command line arrives as a
        // String, and the String "0" is truthy in Groovy (same trap as
        // queueSize in nextflow.config). `?: 0` because a bare `as int` on an
        // unset param throws — mantis-v2.nf declares a default, but this module
        // must not blow up in a pipeline that doesn't.
        ((params.max_positions ?: 0) as int) > 0 ? "max_positions: ${params.max_positions}" : null,
    ].findAll { it }.join('\n')

    notify_send([
        '--level', 'info',
        '--title', ":rocket: ${dataset} — mantis-v2 started",
        '--detail', detail,
        '--key', 'run-start',
        '--min-interval', '900',
        '--state-dir', "${params.output}/nextflow/.notify",
    ])
}

// Report the finished run, pinging the operator.
//
// Called from a single `workflow.onComplete`, never from onComplete AND onError:
// onError fires in ADDITION to onComplete, so implementing both double-posts
// every failure.
def notify_run_end(dataset, wf) {
    def stats = wf.stats
    def summary = [
        "succeeded: ${stats.succeededCount}",
        "cached:    ${stats.cachedCount}",
        "failed:    ${stats.failedCount}",
        "retries:   ${stats.retriesCount}",
        "duration:  ${wf.duration}",
    ].join('\n')

    if (wf.success) {
        notify_send([
            '--level', 'good',
            '--ping',
            '--title', ":white_check_mark: ${dataset} — mantis-v2 complete",
            '--detail', "${summary}\nassembled: ${params.output}/5-assemble/${dataset}.zarr",
        ])
        return
    }

    // Ctrl-C also lands here. Calling that a failure would make every debug
    // relaunch read as a catastrophe, so distinguish it: an interrupt records no
    // process exit status.
    def aborted = wf.exitStatus == null
    def verb = aborted ? 'aborted' : 'FAILED'
    def icon = aborted ? ':warning:' : ':x:'
    def title = "${icon} ${dataset} — mantis-v2 ${verb}"
    if (wf.errorMessage) {
        title += ": ${wf.errorMessage.readLines()[0]}"
    }

    // workflow.errorReport embeds the failing task's `Command executed:` block
    // and a full Python traceback — backticks, quotes and newlines. Never build
    // a shell string out of it; write it to a file and pass the path.
    def report = [summary, wf.errorReport ?: ''].findAll { it }.join('\n\n')
    def detail_file = new File("${params.output}/nextflow/.notify/run-end.txt")
    def args = [
        '--level', aborted ? 'warn' : 'error',
        '--ping',
        '--title', title,
    ]
    try {
        detail_file.parentFile.mkdirs()
        detail_file.text = report
        args += ['--detail-file', detail_file.path]
    }
    catch (Exception e) {
        // An unwritable output dir is plausible here (it may be why the run
        // failed). Fall back to the counts, which need no file.
        args += ['--detail', summary]
    }

    notify_send(args)
}

// Invoke the notifier without a shell.
//
// List.execute() execs the command directly, so no argument is ever parsed by a
// shell — the only safe way to pass an error report containing quotes and
// backticks. `biahub` resolves on PATH because check_environment() already
// aborted the run at launch if it did not.
//
// Bounded and swallowed: a hung POST must not hold up the run's exit, and a
// notification problem must never surface as a pipeline failure after two days
// of compute.
def notify_send(args) {
    try {
        def proc = (['biahub', 'nf', 'notify'] + args.collect { it.toString() }).execute()
        // Drain both streams concurrently. The notifier is silent on a clean
        // post and only speaks up when something needs attention — no webhook
        // configured, a malformed member ID, a rejected POST — so anything it
        // does say belongs on the console, where it is the terminal fallback for
        // a message that did not reach Slack. Unlike notify_step, whose output
        // lands in a task's .command.out, these run-level calls have nowhere
        // else to surface.
        def out = new StringBuffer()
        def err = new StringBuffer()
        proc.consumeProcessOutput(out, err)
        proc.waitForOrKill(30000)

        def said = [out.toString().trim(), err.toString().trim()].findAll { it }.join('\n')
        if (said) {
            log.info said
            // Also append to a file. The run-end message is sent from
            // onComplete, i.e. during session teardown, by which point
            // Nextflow's console renderer has already been torn down and
            // anything written to it is lost — so without this a failed
            // run-end post would be invisible after the fact.
            notify_log(said)
        }
    }
    catch (Exception e) {
        log.warn "Slack notification failed (continuing): ${e.message}"
    }
}

// Append the notifier's own output to <output>/nextflow/.notify/notify.log.
//
// This is the only durable record of a notification problem: the run-end message
// is sent during session teardown, when the console is already gone. Best-effort
// by design — if it cannot be written there is nothing useful to do about it.
def notify_log(message) {
    try {
        def file = new File("${params.output}/nextflow/.notify/notify.log")
        file.parentFile.mkdirs()
        file << "${new Date().format('yyyy-MM-dd HH:mm:ss')} ${message}\n"
    }
    catch (Exception e) {
    }
}
