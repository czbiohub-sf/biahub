#!/usr/bin/env nextflow

// Asserts the resource request a RETRIED task makes.
//
// WHY THIS EXISTS. Every fan-out process is retried on the 130..145 exits and on
// an unreadable exit status (nextflow.config), and on the `preempted` partition
// nearly all of those are preemptions, which say nothing about a task's size.
// The old `* task.attempt` directives made a twice-preempted position ask for
// triple time — and above the partition's 48 h limit sbatch rejects the request
// with exit 1, which is not retried, so one preemption of a >24 h estimate took
// the whole run down (2026_08_14_dynatrack). retry_time / retry_memory in
// modules/common.nf now read the previous attempt's trace and grow only on a
// wall-time (140, or 143 at >=90% of the budget) or OOM (137) signature.
//
// The first block checks the rule directly against stand-in trace records; the
// second runs it through processes that fail their first attempt with each exit
// and print what the second attempt was granted. Runs locally in seconds:
//     nextflow run nextflow/tests/retry_resources.nf
// It exits non-zero if the rule regresses.

nextflow.enable.dsl = 2

include { retry_time; retry_memory } from '../modules/common'

process retried {
    tag "${label}"
    errorStrategy 'retry'
    maxRetries 1
    time   { retry_time(10, task) }
    memory { retry_memory(2, task) }

    input:
    tuple val(label), val(first_exit)

    output:
    tuple val(label), stdout

    script:
    """
    if [ ${task.attempt} -eq 1 ]; then exit ${first_exit}; fi
    echo "time=${task.time.toMinutes()} memory=${task.memory.toGiga()}"
    """
}

// Stand-in for the `task` a directive closure sees on a retry: the attempt and
// the previous attempt's trace record (durations in ms, memory in bytes).
def retry(attempt, exit, realtime_min, time_min, memory_gb) {
    def min_ms = 60000L
    def gb = 1024L ** 3
    return [attempt: attempt,
            previousTrace: [exit: exit, realtime: (realtime_min * min_ms) as long,
                            time: (time_min * min_ms) as long, memory: (memory_gb * gb) as long]]
}

def check(label, got, want) {
    if (got != want) {
        error "${label}: expected ${want}, got ${got}"
    }
    println "ok  ${label}  ->  ${got}"
}

workflow {
    def first = [attempt: 1]
    check('first attempt: time',                    retry_time(110, first),  '110 min')
    check('first attempt: memory',                  retry_memory(64, first), '64 GB')

    // Preemption signatures keep the previous request.
    check('preempted early (143 at 10%): time',     retry_time(110, retry(2, 143, 11, 110, 64)),  '110 min')
    check('preempted early (143 at 10%): memory',   retry_memory(64, retry(2, 143, 11, 110, 64)), '64 GB')
    check('unreadable exit (cancelled at 0 s)',      retry_time(110, retry(2, Integer.MAX_VALUE, 0, 110, 64)), '110 min')
    check('no trace at all',                         retry_time(110, [attempt: 2, previousTrace: null]), '110 min')

    // Wall-time signatures double time only.
    check('SIGUSR2 near limit (140): time',          retry_time(110, retry(2, 140, 109, 110, 64)),  '220 min')
    check('SIGUSR2 near limit (140): memory',        retry_memory(64, retry(2, 140, 109, 110, 64)), '64 GB')
    check('SIGTERM at 95% of budget (143): time',    retry_time(110, retry(2, 143, 105, 110, 64)),  '220 min')
    check('SIGTERM at 89% of budget (143): time',    retry_time(110, retry(2, 143, 97, 110, 64)),   '110 min')

    // OOM doubles memory only.
    check('OOM (137): memory',                       retry_memory(64, retry(2, 137, 5, 110, 64)), '128 GB')
    check('OOM (137): time',                         retry_time(110, retry(2, 137, 5, 110, 64)),  '110 min')

    // Escalation persists across a later preemption; a second wall-time hit
    // doubles again; the partition limit caps it.
    check('attempt 3 preempted after a doubling',    retry_time(110, retry(3, 143, 20, 220, 64)),  '220 min')
    check('attempt 3 hits wall-time again',          retry_time(110, retry(3, 140, 219, 220, 64)), '440 min')
    check('cap: 27 h base hits wall-time',           retry_time(1620, retry(2, 140, 1619, 1620, 64)), '2880 min')
    check('cap: base above the limit',               retry_time(3000, first), '2880 min')

    // End to end: what the second attempt is actually granted after each exit.
    retried(channel.of(['preempted', 143], ['timeout', 140], ['oom', 137]))
        .subscribe { label, out ->
            def want = [preempted: 'time=10 memory=2', timeout: 'time=20 memory=2', oom: 'time=10 memory=4'][label]
            check("retried process after ${label}", out.trim(), want)
        }
}
