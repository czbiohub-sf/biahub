#!/usr/bin/env bash
# One-screen status of a mantis-v2 run, for the unattended monitoring loop.
#
#   bash run_status.sh <OUTPUT>
#
# Read-only: it never relaunches, deletes or notifies; the loop decides that from
# the `state` line (SKILL.md § Monitoring pass). Prints `key  value` lines:
#
#   state  running     the head process is alive
#          completed   the last launch finished without an error
#          aborted     the last launch ended on an error (`cause`, `failed`,
#                      `error`, `workdir` say which and why)
#          head-lost   the head process is gone and the log has no ending: it was
#                      killed (allocation ended, node lost, Ctrl-C before startup)
#          not-started no .nextflow.log yet
#
# `cleaned yes` means cleanup deleted the intermediates and the run is final:
# relaunching it would recompute everything from raw.
#
# Liveness: on the host that launched the run (provenance.txt's last `host`), a
# process check for a `nextflow run` naming this output directory decides. From
# any other host, the log decides: Nextflow writes a heartbeat to .nextflow.log
# every 5 minutes while tasks are pending, so a log modified in the last
# 15 minutes counts as alive.
set -uo pipefail

OUT="${1:?usage: run_status.sh <OUTPUT>}"
OUT="$(cd "${OUT}" && pwd)" || exit 2
LOG="${OUT}/.nextflow.log"
PROV="${OUT}/nextflow/provenance.txt"
TRACE="${OUT}/nextflow/trace.txt"

echo "run       ${OUT}"
if [[ -f "${PROV}" ]]; then
    launches="$(grep -c '^=== ' "${PROV}")"
    echo "launched  $(grep '^=== ' "${PROV}" | tail -1 | tr -d '= ')  (launch ${launches})"
    launch_host="$(awk '$1 == "host" {h = $2} END {print h}' "${PROV}")"
    echo "commit    $(awk '$1 == "branch" {b = $2} $1 == "commit" {c = substr($2, 1, 8)} END {print b, c}' "${PROV}")"
fi
[[ -f "${OUT}/nextflow/intermediates_cleaned.txt" ]] && echo "cleaned   yes"

if [[ ! -f "${LOG}" ]]; then
    echo "state     not-started"
    exit 0
fi

alive=""
if [[ "${launch_host:-}" == "$(hostname)" ]]; then
    pid="$(pgrep -u "${USER}" -f "nextflow.*run .*--output +\"?${OUT}\"?( |$)" | head -1)"
    [[ -n "${pid}" ]] && alive="pid ${pid} on $(hostname)"
else
    age_min=$(( ($(date +%s) - $(stat -c %Y "${LOG}")) / 60 ))
    (( age_min < 15 )) && alive="log written ${age_min} min ago (launched on ${launch_host:-?}, checked from $(hostname))"
fi
echo "log_age   $(( ($(date +%s) - $(stat -c %Y "${LOG}")) / 60 )) min"

cause="$(grep -m1 'Session aborted -- Cause:' "${LOG}" | sed 's/.*Cause: //')"
if [[ -n "${alive}" ]]; then
    echo "state     running"
    echo "head      ${alive}"
elif [[ -n "${cause}" ]]; then
    echo "state     aborted"
    echo "cause     ${cause}"
elif grep -q 'Execution complete -- Goodbye' "${LOG}"; then
    echo "state     completed"
else
    echo "state     head-lost"
    echo "last_log  $(tail -1 "${LOG}" | cut -c1-160)"
fi

# The task that ended the run: Nextflow prints its work dir after the error.
if [[ -n "${cause}" ]]; then
    failed="$(sed -n "s/.*Error executing process > '\(.*\)'.*/\1/p" "${LOG}" | tail -1)"
    workdir="$(awk '/^Work dir:/ {getline; gsub(/^ +| +$/, ""); w = $0} END {print w}' "${LOG}")"
    [[ -n "${failed}" ]] && echo "failed    ${failed}"
    if [[ -n "${workdir}" && -d "${workdir}" ]]; then
        echo "workdir   ${workdir}"
        echo "exit      $(cat "${workdir}/.exitcode" 2>/dev/null || echo '?')"
        err="$(grep -E '^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception): ' "${workdir}/.command.err" 2>/dev/null | tail -1)"
        [[ -n "${err}" ]] && echo "error     ${err}"
    fi
fi

# Per-process outcome of this launch's tasks (trace.txt is rewritten at each launch).
if [[ -f "${TRACE}" ]]; then
    echo "steps     process: completed cached failed aborted (max attempt)"
    awk -F'\t' 'NR > 1 {
        p = $4; sub(/ \(.*/, "", p); sub(/.*:/, "", p)
        if (!(p in seen)) { order[++n] = p; seen[p] = 1 }
        count[p, $5]++
        if ($15 + 0 > max[p]) max[p] = $15 + 0
    } END {
        for (i = 1; i <= n; i++) { p = order[i]
            printf "          %-32s %4d %4d %4d %4d  (%d)\n", p, count[p, "COMPLETED"], count[p, "CACHED"], count[p, "FAILED"], count[p, "ABORTED"], max[p] }
    }' "${TRACE}"
    # Positions that needed 4+ attempts: one more failure and retries run out.
    awk -F'\t' 'NR > 1 && $15 + 0 >= 4 {print "attempt4  " $4 " attempt " $15 " " $5 " exit " $6}' "${TRACE}" | tail -5
fi

echo "squeue    $(squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -c '^nf-') nf-* jobs for ${USER} (all runs)"
