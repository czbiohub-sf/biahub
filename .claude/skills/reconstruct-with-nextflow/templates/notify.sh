#!/usr/bin/env bash
# Post a notification to Slack, falling back to the terminal.
#
#   notify.sh "✅ 2026_07_14_A549_MAP4_ZIKV: assemble verified"
#   notify.sh "❌ run_deskew failed on C/4/001001" "$(tail -20 slurm_output/deskew/*.out)"
#   notify.sh --level error --ping "❌ 2026_07_14: torn shard, needs a human"
#
# THE PIPELINE DOES NOT USE THIS SCRIPT. mantis-v2.nf posts run start, each
# step's completion, and run end itself (nextflow/modules/notify.nf). This is for
# the two cases that have no Nextflow context:
#
#   1. messages Claude sends while monitoring — a diagnosis mid-run, the wrap-up
#      summary after verifying the assembled plate;
#   2. hand-rolled reruns that bypass Nextflow (a bare sbatch of one step).
#
# It is a thin wrapper over `biahub nf notify`, which owns the payload shaping,
# truncation, mention handling, and HTTP retries. The direct-curl branch below is
# only for a broken venv — which is exactly when you most need to send an alert.
#
# Two environment variables, both belonging in ~/.bashrc and never in the repo:
#   BIAHUB_SLACK_WEBHOOK   incoming-webhook URL — a CREDENTIAL
#   BIAHUB_SLACK_ID        your Slack member ID, for --ping
#
# Without a webhook this is still not an error: the message is printed and the
# exit status is 0, so a monitoring loop is never broken by a missing webhook.

set -uo pipefail

LEVEL="info"
PING=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) LEVEL="$2"; shift 2 ;;
        --ping)  PING="--ping"; shift ;;
        --) shift; break ;;
        -*) echo "notify.sh: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

TITLE="${1:-}"
DETAIL="${2:-}"

[[ -n "${TITLE}" ]] || {
    echo "usage: notify.sh [--level info|good|warn|error] [--ping] <title> [detail]" >&2
    exit 2
}

# Prefer the CLI. BIAHUB_PROJECT lets a non-activated shell still find it.
BIAHUB="$(command -v biahub || true)"
if [[ -z "${BIAHUB}" && -n "${BIAHUB_PROJECT:-}" && -x "${BIAHUB_PROJECT}/.venv/bin/biahub" ]]; then
    BIAHUB="${BIAHUB_PROJECT}/.venv/bin/biahub"
fi

if [[ -n "${BIAHUB}" ]]; then
    # Pass the detail through a file: a traceback contains quotes and backticks,
    # and argv has a length limit a long log tail can exceed.
    DETAIL_ARGS=()
    if [[ -n "${DETAIL}" ]]; then
        DETAIL_FILE="$(mktemp)"
        trap 'rm -f "${DETAIL_FILE}"' EXIT
        printf '%s' "${DETAIL}" > "${DETAIL_FILE}"
        DETAIL_ARGS=(--detail-file "${DETAIL_FILE}")
    fi
    "${BIAHUB}" nf notify --level "${LEVEL}" ${PING} --title "${TITLE}" "${DETAIL_ARGS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Fallback: biahub is unavailable (venv broken or not synced). Keep this minimal
# and dependency-free — no severity colors, no mention validation, no retries.
# ---------------------------------------------------------------------------
echo "[notify] biahub not found — using the minimal curl fallback" >&2

STAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"
TEXT="${TITLE}"
if [[ -n "${PING}" && -n "${BIAHUB_SLACK_ID:-}" ]]; then
    TEXT="<@${BIAHUB_SLACK_ID}> ${TEXT}"
fi
if [[ -n "${DETAIL}" ]]; then
    # Keep the tail: a traceback ends with the exception. Cap it so a huge log
    # cannot produce a payload Slack rejects.
    TRIMMED="$(printf '%s' "${DETAIL}" | tail -c 2500 | tr -d '\000')"
    TEXT="${TEXT}"$'\n'"\`\`\`${TRIMMED}\`\`\`"
fi
TEXT="${TEXT}"$'\n'"_${STAMP} · $(hostname)_"

if [[ -z "${BIAHUB_SLACK_WEBHOOK:-}" ]]; then
    echo "[notify] BIAHUB_SLACK_WEBHOOK unset — printing instead of posting"
    echo "${TEXT}"
    exit 0
fi

PAYLOAD=$(TEXT="${TEXT}" python3 -c 'import json,os; print(json.dumps({"text": os.environ["TEXT"]}))')

# mktemp, not /tmp/notify_resp.$$: $TMPDIR is a shared multi-user /tmp on the
# login nodes, where a predictable name is a collision and symlink hazard.
RESP="$(mktemp)"
ERRF="$(mktemp)"
trap 'rm -f "${RESP}" "${ERRF}" "${DETAIL_FILE:-}"' EXIT

HTTP=$(curl -sS -o "${RESP}" -w '%{http_code}' \
    -X POST -H 'Content-Type: application/json' \
    --data "${PAYLOAD}" --max-time 15 \
    "${BIAHUB_SLACK_WEBHOOK}" 2>"${ERRF}") || true

if [[ "${HTTP}" != "200" ]]; then
    echo "[notify] Slack post failed (HTTP ${HTTP:-none}): $(cat "${RESP}" 2>/dev/null)" >&2
    cat "${ERRF}" >&2 2>/dev/null
    echo "${TEXT}"
fi

exit 0
