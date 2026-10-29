#!/usr/bin/env bash
# Asserts nextflow.config's errorStrategy on storage errors. Runs locally in
# under a minute, from anywhere:
#     bash nextflow/tests/retry_oserror.sh
# Exits non-zero if the rule regresses.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${HERE}/../nextflow.config"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
unset CLAUDECODE   # Nextflow's agent mode would reformat the console (nextflow#7478)
fail=0

run_case() {   # $1 = case; prints the console, returns nextflow's exit code
    (cd "${WORK}" && nextflow -q run "${HERE}/retry_oserror.nf" -c "${CONFIG}" -profile local \
        --output "${WORK}/out_$1" --case "$1" --oserror_retry_delay_minutes 0.2 -ansi-log false 2>&1)
}

out="$(run_case retried)"; code=$?
echo "${out}" | grep -E 'attempt=|ERROR' || true
if [[ ${code} -ne 0 ]]; then echo "FAIL retried: run exited ${code}"; fail=1; fi
for kind in oserror sigterm; do
    echo "${out}" | grep -q "^${kind} attempt=2 " || { echo "FAIL retried: ${kind} did not succeed on attempt 2"; fail=1; }
done
slow_at="$(echo "${out}" | sed -n 's/^slow attempt=1 at=//p')"
os_at="$(echo "${out}" | sed -n 's/^oserror attempt=2 at=//p')"
if [[ -n "${slow_at}" && -n "${os_at}" && "${slow_at}" -gt "${os_at}" ]]; then
    echo "FAIL retried: slow finished after the OSError retry, so the wait blocked other tasks"; fail=1
fi

out="$(run_case terminated)"; code=$?
if [[ ${code} -eq 0 ]]; then echo "FAIL terminated: run succeeded"; fail=1; fi
if echo "${out}" | grep -q 'attempt=2'; then echo "FAIL terminated: FileNotFoundError was retried"; fail=1; fi

[[ ${fail} -eq 0 ]] && echo "PASS: OSError and signal exits retried, OSError subclass terminates"
exit ${fail}
