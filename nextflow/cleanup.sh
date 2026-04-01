#!/usr/bin/env bash
# Clean up Nextflow work directories and cached outputs after a completed run.
#
# Usage:
#   bash nextflow/cleanup.sh            # clean current directory
#   bash nextflow/cleanup.sh /path/to   # clean specified directory

set -euo pipefail

target="${1:-.}"

echo "Cleaning Nextflow artifacts in: ${target}"

echo "Removing Nextflow work directory..."
rm -rf "${target}/work/"

echo "Removing Nextflow cache..."
rm -rf "${target}/.nextflow/"

echo "Removing Nextflow log files..."
rm -f "${target}/.nextflow.log" "${target}/.nextflow.log."*

echo "Removing Nextflow reports..."
rm -rf "${target}/nextflow/output/"

echo "Done."
