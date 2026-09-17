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
// modules/common.nf now double the base once and cap time at that limit.
//
// Each process below fails its first two attempts with a preemption-style exit
// and, on the third, prints the time and memory it was granted; the workflow
// checks them against the rule. Runs locally in seconds:
//     nextflow run nextflow/tests/retry_resources.nf
// It exits non-zero if the rule regresses.

nextflow.enable.dsl = 2

include { retry_time; retry_memory } from '../modules/common'

process sized {
    errorStrategy 'retry'
    maxRetries 2
    time   { retry_time(meta.time_minutes, task) }
    memory { retry_memory(meta.mem_gb, task) }

    input:
    tuple val(label), val(meta)

    output:
    tuple val(label), stdout

    script:
    // Attempts 1 and 2 die like a preempted job; attempt 3 reports its request.
    """
    if [ ${task.attempt} -lt 3 ]; then exit 143; fi
    echo "attempt=${task.attempt} time=${task.time.toMinutes()} memory=${task.memory.toGiga()}"
    """
}

// Stand-in for the `task` a directive closure sees: only `attempt` is read.
def at(n) {
    return [attempt: n]
}

def check(label, got, want) {
    if (got != want) {
        error "${label}: expected ${want}, got ${got}"
    }
    println "ok  ${label}  ->  ${got}"
}

workflow {
    // Directive values, computed directly: base on the first attempt, double on
    // every later one (not attempt-fold), time capped at the partition limit.
    check('time attempt 1',            retry_time(110, at(1)), '110 min')
    check('time attempt 2',            retry_time(110, at(2)), '220 min')
    check('time attempt 3 (no growth)', retry_time(110, at(3)), '220 min')
    check('time attempt 6 (no growth)', retry_time(110, at(6)), '220 min')
    check('time cap: 27 h base doubled', retry_time(1620, at(2)), '2880 min')
    check('time cap: base above limit',  retry_time(3000, at(1)), '2880 min')
    check('memory attempt 1',          retry_memory(64, at(1)), '64.0 GB')
    check('memory attempt 2',          retry_memory(64, at(2)), '128.0 GB')
    check('memory attempt 3 (no growth)', retry_memory(64, at(3)), '128.0 GB')
    check('memory from a string meta',  retry_memory('16', at(2)), '32.0 GB')

    // End to end through a real retried process: the third attempt must run with
    // exactly double the base, not triple.
    sized(channel.of(['fanout', [time_minutes: 10, mem_gb: 2]]))
        .subscribe { label, out ->
            check("${label}: retried task request", out.trim(), 'attempt=3 time=20 memory=4')
        }
}
