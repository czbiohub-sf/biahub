#!/usr/bin/env nextflow

// Drives nextflow.config's errorStrategy through real task failures. Run it with
// the pipeline config, through nextflow/tests/retry_oserror.sh, which asserts on
// the outcome of each case:
//
//   --case retried     a bare `OSError: [Errno 71] ...` (exit 1) and a SIGTERM
//                      (exit 143) both fail their first attempt and must succeed
//                      on the second; a slow sibling must finish during the
//                      OSError's wait (12 s at the test's delay, slow takes 3 s),
//                      i.e. the wait blocks only its own task.
//   --case terminated  `FileNotFoundError: [Errno 2] ...` (exit 1), an OSError
//                      subclass, must terminate the run without a retry.

nextflow.enable.dsl = 2

params.case = 'retried'

process fails_once {
    tag "${kind}"

    input:
    val kind

    output:
    stdout

    script:
    """
    if [ ${task.attempt} -eq 1 ]; then
        case "${kind}" in
            oserror)  printf 'Traceback (most recent call last):\\n  File "w.py", line 1\\nOSError: [Errno 71] Protocol error\\n' >&2; exit 1 ;;
            missing)  printf 'Traceback (most recent call last):\\n  File "w.py", line 1\\nFileNotFoundError: [Errno 2] No such file\\n' >&2; exit 1 ;;
            sigterm)  exit 143 ;;
        esac
    fi
    if [ "${kind}" = "slow" ]; then sleep 3; fi
    echo "${kind} attempt=${task.attempt} at=\$(date +%s)"
    """
}

workflow {
    def kinds = params.case == 'terminated' ? ['missing'] : ['oserror', 'sigterm', 'slow']
    channel.fromList(kinds) | fails_once | view
}
