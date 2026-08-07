#!/usr/bin/env bash
# Post a pipeline notification to Slack, falling back to the terminal.
#
#   notify.sh "✅ 2026_07_14_A549_MAP4_ZIKV: mantis-v2 complete"
#   notify.sh "❌ run_deskew failed on C/4/001001" "$(tail -20 slurm_output/.../x.out)"
#
# Set BIAHUB_SLACK_WEBHOOK to an incoming-webhook URL to enable Slack:
#   export BIAHUB_SLACK_WEBHOOK="https://hooks.slack.com/services/T.../B.../..."
# Put it in ~/.bashrc (not in the repo) — the URL is a credential.
#
# Without the variable set this is not an error: the message is printed and the
# exit status is still 0, so a monitoring loop is never broken by a missing
# webhook.

set -uo pipefail

TITLE="${1:-}"
DETAIL="${2:-}"

[[ -n "${TITLE}" ]] || { echo "usage: notify.sh <title> [detail]" >&2; exit 2; }

STAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"
TEXT="${TITLE}"
[[ -n "${DETAIL}" ]] && TEXT="${TEXT}"$'\n'"\`\`\`${DETAIL}\`\`\`"
TEXT="${TEXT}"$'\n'"_${STAMP} · $(hostname)_"

if [[ -z "${BIAHUB_SLACK_WEBHOOK:-}" ]]; then
    echo "[notify] BIAHUB_SLACK_WEBHOOK unset — printing instead of posting"
    echo "${TEXT}"
    exit 0
fi

PAYLOAD=$(TEXT="${TEXT}" python3 -c 'import json,os; print(json.dumps({"text": os.environ["TEXT"]}))')

HTTP=$(curl -sS -o /tmp/notify_resp.$$ -w '%{http_code}' \
    -X POST -H 'Content-Type: application/json' \
    --data "${PAYLOAD}" --max-time 15 \
    "${BIAHUB_SLACK_WEBHOOK}" 2>/tmp/notify_err.$$) || true

if [[ "${HTTP}" != "200" ]]; then
    echo "[notify] Slack post failed (HTTP ${HTTP:-none}): $(cat /tmp/notify_resp.$$ 2>/dev/null)" >&2
    cat /tmp/notify_err.$$ >&2 2>/dev/null
    echo "${TEXT}"
fi

rm -f /tmp/notify_resp.$$ /tmp/notify_err.$$
exit 0
