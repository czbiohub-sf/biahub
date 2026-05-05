#!/usr/bin/env bash
# SIGKILL-resilient orphan cleanup for tile-stitch SLURM jobs.
#
# `install_preempt_handler` covers SIGUSR1 + SIGTERM + SIGINT + atexit
# for clean shutdown. SIGKILL cannot be trapped — when a driver gets
# `kill -9`'d (or the host hard-crashes), the dask-jobqueue worker
# allocations don't get torn down and persist on the cluster until
# their walltime expires, eating fairshare.
#
# This script is the documented oncall path for that case. Invoke it
# after confirming no live driver is holding the workers.
#
# Usage:
#   bash cleanup_orphans.sh                 # cancel all dask-worker jobs for $USER
#   bash cleanup_orphans.sh --dry-run       # print what would be cancelled
#   bash cleanup_orphans.sh --name <name>   # override the SLURM job-name pattern
#
# Exit codes:
#   0  cleanup succeeded (or nothing to clean up)
#   1  squeue/scancel not on PATH or otherwise unavailable
#   2  scancel returned non-zero

set -euo pipefail

JOB_NAME="dask-worker"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --name)
            JOB_NAME="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,/^# Exit codes:/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if ! command -v squeue >/dev/null 2>&1; then
    echo "cleanup_orphans: squeue not on PATH" >&2
    exit 1
fi
if ! command -v scancel >/dev/null 2>&1; then
    echo "cleanup_orphans: scancel not on PATH" >&2
    exit 1
fi

ORPHANS=$(squeue --user "$USER" --name "$JOB_NAME" --format="%i" --noheader || true)

if [[ -z "$ORPHANS" ]]; then
    echo "cleanup_orphans: no jobs matching --name=$JOB_NAME for $USER"
    exit 0
fi

ORPHAN_COUNT=$(echo "$ORPHANS" | wc -l)
echo "cleanup_orphans: found $ORPHAN_COUNT orphan job(s) matching --name=$JOB_NAME"

if (( DRY_RUN )); then
    echo "$ORPHANS"
    exit 0
fi

scancel --user "$USER" --name "$JOB_NAME"
echo "cleanup_orphans: scancel issued"
