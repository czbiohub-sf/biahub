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
    tuple val(step), val(output_zarr), val(index)
    val dataset

    output:
    val step

    script:
    // Emoji as a Slack shortcode rather than raw UTF-8: it renders the same and
    // keeps the payload ASCII.
    //
    // No position count here. The plate has the same number of positions for
    // every step, so repeating it six times says nothing new — it is reported
    // once, in the run-start message.
    //
    // No --ping either. Step messages post silently so the @-mention on the
    // run-end message keeps meaning "this needs you".
    //
    // `|| true` so nothing the notifier does can fail the task, independent of
    // the errorStrategy.
    """
    biahub nf notify \\
        --level good \\
        --title ":white_check_mark: ${dataset} — ${step} complete [${index}]" \\
        --detail "output: ${output_zarr}" || true
    """
}

// Announce the run.
//
// This doubles as a webhook smoke test: without it the first Slack message is
// the completion one, hours or days later, so a revoked webhook would be
// discovered far too late to be useful.
//
// Sent once the position count is known rather than at graph-construction time,
// because the count comes from the list_positions task (about 40s) and the
// message reports it. The consequence to know: a config error that kills
// list_positions itself produces no start message — only the failure message
// from onComplete.
//
// Rate-limited by key, because debugging a config produces relaunch storms —
// five launches inside ten minutes is an observed pattern, which would otherwise
// be five notifications.
def notify_run_start(dataset, pipeline, n_positions, steps) {
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
        "pipeline:  ${pipeline}",
        "positions: ${n_positions}",
        "steps:     ${steps.size()} — ${steps.join(', ')}",
        "input:     ${params.input}",
        "output:    ${params.output}",
        "host:      ${java.net.InetAddress.localHost.hostName}",
        // `as int` because a param given on the command line arrives as a
        // String, and the String "0" is truthy in Groovy (same trap as
        // queueSize in nextflow.config). `?: 0` because a bare `as int` on an
        // unset param throws — mantis-v2.nf declares a default, but this module
        // must not blow up in a pipeline that doesn't.
        ((params.max_positions ?: 0) as int) > 0 ? "max_positions: ${params.max_positions}" : null,
    ].findAll { it }.join('\n')

    // --operator prepends the "operator:" line, resolved from the account database
    // by the Python. It is NOT taken from Slack: turning a member ID into a
    // display name needs a users.info call and a bot token, which an incoming
    // webhook cannot do — and an <@U…> mention would ping, while this message is
    // deliberately silent.
    notify_send([
        '--level', 'info',
        '--operator',
        '--title', ":rocket: ${dataset} — reconstruction started",
        '--detail', detail,
        '--key', 'run-start',
        '--min-interval', '900',
        '--state-dir', "${params.output}/nextflow/.notify",
    ])
}

// Ask SLURM what actually killed each attempt.
//
// The exit code alone CANNOT say. 143 is SIGTERM, which SLURM sends both when it
// reclaims a job on the `preempted` partition and when a job hits its time limit
// — nextflow.config says as much ("SIGTERM 143 (preempt/timeout)"). Calling every
// 143 a preemption would send the reader after the wrong cause: preemption needs
// no action, a time limit needs more --time. sacct knows the difference, and
// trace.txt hands us the SLURM job id in native_id.
//
// One batched query for every restarted attempt. Returns a label -> count map, or
// null when sacct cannot answer — no accounting records, not a SLURM run, or the
// JVM shutting down under us — in which case the caller falls back to exit codes.
// -X reports the allocation only, without the .batch/.extern sub-steps that would
// otherwise double-count.
def restart_causes(job_ids) {
    if (!job_ids) {
        return null
    }
    try {
        def proc = ['sacct', '-X', '-n', '-P', '-o', 'JobID,State',
                    '-j', job_ids.join(',')].execute()
        def out = new StringBuffer()
        def err = new StringBuffer()
        proc.consumeProcessOutput(out, err)
        proc.waitFor(15, java.util.concurrent.TimeUnit.SECONDS)

        def causes = [:]
        out.toString().readLines().each { line ->
            def field = line.split('\\|')
            if (field.size() > 1) {
                def label = sacct_label(field[1])
                causes[label] = (causes[label] ?: 0) + 1
            }
        }
        return causes ?: null
    }
    catch (Throwable t) {
        return null
    }
}

def sacct_label(state) {
    // "CANCELLED by 12345" carries the canceller's uid, so keep the first word.
    def name = state.trim().split(/\s+/)[0]
    if (name == 'PREEMPTED')     return 'preempted'
    if (name == 'TIMEOUT')       return 'timed out'
    if (name == 'OUT_OF_MEMORY') return 'out of memory'
    if (name == 'NODE_FAIL')     return 'node failure'
    return name.toLowerCase().replace('_', ' ')
}

// Split every failed ATTEMPT in trace.txt into "restarted" and "failed".
//
// workflow.stats cannot: it reports failedCount and retriesCount with no exit
// codes, so a position that was preempted once and then succeeded reads as
// "failed: 1, retries: 1" — as if the run were broken when nothing went wrong.
//
// Same rule the pipeline uses to decide whether to retry (errorStrategy in
// nextflow.config): 130..145 is "128 + signal", i.e. the job was killed rather
// than the code failing. Those attempts were restarted, so they are not failures.
//
// trace.txt is complete and readable by the time onComplete runs (verified); its
// path comes from `trace.file` in nextflow.config. Returns null if it cannot be
// read, and the caller then reports the raw counts instead.
def failed_attempts() {
    try {
        def trace = new File("${params.output}/nextflow/trace.txt")
        if (!trace.exists()) {
            return null
        }
        def outcome = [restarted: 0, failed: 0, exits: [:], job_ids: []]
        trace.readLines().drop(1).each { line ->
            def field = line.split('\\t')
            if (field.size() > 5 && field[4] == 'FAILED') {
                def code = field[5].isInteger() ? field[5] as int : -1
                if (code >= 130 && code <= 145) {
                    outcome.restarted = outcome.restarted + 1
                    outcome.exits[code] = (outcome.exits[code] ?: 0) + 1
                    if (field[2] && field[2].isInteger()) {
                        outcome.job_ids += field[2]
                    }
                }
                else {
                    outcome.failed = outcome.failed + 1
                }
            }
        }
        return outcome
    }
    catch (Exception e) {
        return null
    }
}

// "restarted: 3 (2 preempted, 1 out of memory)", or the exit codes when sacct
// could not tell us.
def restarted_line(outcome) {
    // Always reported, including "restarted: 0", so a clean run says so outright
    // instead of leaving the reader to infer it from an absent line. Nothing to
    // ask sacct about in that case.
    if (!outcome.restarted) {
        return 'restarted: 0'
    }
    def causes = restart_causes(outcome.job_ids)
    def detail = causes
        ? causes.sort { -it.value }.collect { label, n -> "${n} ${label}" }.join(', ')
        : outcome.exits.sort().collect { code, n -> "exit ${code}×${n}" }.join(', ')
    return "restarted: ${outcome.restarted} (${detail})".toString()
}

// Report the finished run, pinging the operator.
//
// Called from a single `workflow.onComplete`, never from onComplete AND onError:
// onError fires in ADDITION to onComplete, so implementing both double-posts
// every failure.
// Lines that carry no information about what went wrong. Groovy appends a
// `Possible solutions:` list to every MissingMethodException — that is what a
// real run put in a Slack title, verbatim and alone: "reconstruction aborted:
// Possible solutions: any(), any(), any(groovy.lang.Closure), …".
def _is_boilerplate(line) {
    def t = line.trim()
    return t.startsWith('at ') || t.startsWith('Possible solutions:') ||
           t ==~ /^\.\.\. \d+ more$/ || t.startsWith('Caused by:') && t.size() < 12
}

// The most informative single line, for the title.
//
// LAST non-boilerplate line, not the first. For a task that died in Python,
// errorMessage is the captured stderr, so the first line is "Traceback (most
// recent call last):" — true of every Python failure and useless in a title —
// while the last is the exception itself ("ValueError: bad config here"). For a
// Groovy error the informative line is followed by boilerplate, which is why the
// filter runs before `.last()` rather than after. The Python caps the length.
def error_headline(lines) {
    def useful = lines.findAll { !_is_boilerplate(it) }
    return useful ? useful.last().trim() : null
}

// Stack frames say where the interpreter was, not what the user must fix.
def strip_stack_frames(text) {
    return text.readLines().findAll { !_is_boilerplate(it) }.join('\n')
}

def notify_run_end(dataset, pipeline, wf, assembled = null) {
    def stats = wf.stats
    def lines = [
        "pipeline:  ${pipeline}",
        "succeeded: ${stats.succeededCount}",
        "cached:    ${stats.cachedCount}",
    ]

    def outcome = failed_attempts()
    if (outcome == null) {
        // No trace.txt to classify against; report the raw counts rather than
        // claim something we cannot substantiate.
        lines += ["failed:    ${stats.failedCount}", "retries:   ${stats.retriesCount}"]
    }
    else {
        // Restarted attempts were retried, so they are not failures and must
        // not be counted as any. Reported unconditionally — see restarted_line.
        lines += [restarted_line(outcome)]
        if (outcome.failed) {
            lines += ["failed:    ${outcome.failed}".toString()]
        }
        // A run that died with nothing but restarts burnt through maxRetries. Say
        // so, or the reader sees a FAILED title with no failures listed under it.
        if (!wf.success && !outcome.failed && outcome.restarted) {
            lines += ["note:      retries exhausted, no code-level failure".toString()]
        }
    }

    lines += ["duration:  ${wf.duration}"]
    def summary = lines.join('\n')

    if (wf.success) {
        notify_send([
            '--level', 'good',
            '--ping',
            // :checkered_flag: rather than the :white_check_mark: the six step
            // messages use — this is the one that pings, so it should not look
            // identical to six that do not. Reads 🚀 start, ✅ per step, 🏁 done.
            '--title', ":checkered_flag: ${dataset} — reconstruction complete",
            // The assembled store's path is PASSED IN, not built here: its
            // directory number is a position among the steps that ran, and the
            // step is optional, so there is no constant to write down. Null when
            // assemble was not performed, and then the line is simply absent.
            '--detail', assembled ? "${summary}\nassembled: ${assembled}" : summary,
        ])
        return
    }

    // THREE outcomes reach here, and they need different words. An interrupt
    // (Ctrl-C) records no error message at all — calling that a failure would
    // make every debug relaunch read as a catastrophe. A task failure names a
    // process. Anything else is the pipeline's own code failing to wire itself,
    // where NO task failed and there is nothing to retry: the run that prompted
    // this had 288 successful tasks and every reconstruction step on disk, and
    // still said "reconstruction aborted".
    def message_lines = (wf.errorMessage ?: '').readLines().findAll { it.trim() }
    def interrupted = message_lines.isEmpty()
    def task_match = (wf.errorReport ?: '') =~ /Process `([^`]+)`/
    def failed_task = task_match ? task_match[0][1] : null

    def icon = interrupted ? ':warning:' : ':x:'
    def title
    if (interrupted) {
        title = "${icon} ${dataset} — reconstruction interrupted"
    }
    else if (failed_task) {
        title = "${icon} ${dataset} — ${failed_task} failed"
    }
    else {
        title = "${icon} ${dataset} — pipeline error, no task failed"
    }
    def headline = error_headline(message_lines)
    if (headline) {
        title += ": ${headline}"
    }

    // workflow.errorReport embeds the failing task's `Command executed:` block
    // and a full Python traceback — backticks, quotes and newlines. Never build
    // a shell string out of it; write it to a file and pass the path.
    //
    // Stack frames are stripped first. The detail is truncated from the FRONT
    // (clean_and_truncate keeps the tail, where a Python diagnosis lives), so a
    // hundred lines of `at java.base/...` would push the counts and the actual
    // error out of the message and leave the reader with thread bookkeeping.
    def report = [summary, strip_stack_frames(wf.errorReport ?: '')].findAll { it }.join('\n\n')
    def detail_file = new File("${params.output}/nextflow/.notify/run-end.txt")
    def args = [
        '--level', interrupted ? 'warn' : 'error',
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
// NEVER USE waitForOrKill HERE. run-end is sent from onComplete, i.e. during JVM
// shutdown, and there `waitForOrKill` returns before the child has exited and
// then DESTROYS it — the notifier was SIGTERMed before it could POST, so the
// failure message, the one that matters most, was silently never sent. Measured
// against a local webhook: run start and run end on a successful run delivered,
// a failed run delivered nothing.
//
// `waitFor(timeout, unit)` does not kill the child, so it survives and completes
// its POST even if this thread is interrupted (it is: AnsiLogObserver's own join
// throws InterruptedException during teardown) or the JVM exits first — an
// orphaned process keeps running. We wait only so a fast post is ordered before
// shutdown; we do not depend on the wait succeeding.
//
// inheritIO() rather than capturing the child's streams. The notifier is silent
// on a clean post and only speaks up when something needs attention — no webhook
// configured, a malformed member ID, a rejected POST — and inheriting sends that
// straight to the console the user is watching, with no pipe for us to drain (an
// unread pipe would also block the child once it filled) and nothing for us to
// re-log at teardown, when the console is already gone. `--log-file` makes the
// notifier record delivery problems itself, so the durable diagnostic does not
// depend on this process being alive to write it.
def notify_send(args) {
    def command = ['biahub', 'nf', 'notify'] +
        ['--log-file', "${params.output}/nextflow/.notify/notify.log".toString()] +
        args.collect { it.toString() }
    try {
        def builder = new ProcessBuilder(command)
        builder.inheritIO()
        def proc = builder.start()
        proc.waitFor(30, java.util.concurrent.TimeUnit.SECONDS)
    }
    catch (Throwable t) {
        // Includes the InterruptedException thrown during shutdown. The child is
        // already spawned and unaffected, so there is nothing to recover: a
        // notification problem must never surface as a pipeline failure after two
        // days of compute.
        log.debug "notify: ${t.class.simpleName}: ${t.message}"
    }
}

