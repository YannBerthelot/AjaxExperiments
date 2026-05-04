#!/usr/bin/env bash
# Pull degradation_v2 results back from the remote machine via rsync.
#
# Usage:
#   ./fetch_results_v2.sh user@remote-host:/path/to/AjaxExperiments
#
# Pulls:
#   - degradation_v2_results/*.json   (the cell results, small)
#   - study_v2_logs/*.log             (per-cell stdout/stderr, optional)
#
# Does NOT pull TB events or checkpoints — those are large and regenerable.
# If you need them, append the path to the SUBPATHS array below and re-run.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <user@remote:/path/to/AjaxExperiments> [--with-logs]" >&2
  exit 1
fi

REMOTE_BASE="${1%/}"
shift || true
WITH_LOGS=0
for arg in "$@"; do
  case "$arg" in
    --with-logs) WITH_LOGS=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"

SUBPATHS=("degradation_v2_results")
if [ "$WITH_LOGS" -eq 1 ]; then
  SUBPATHS+=("study_v2_logs")
fi

for sub in "${SUBPATHS[@]}"; do
  mkdir -p "$LOCAL_ROOT/$sub"
  echo "[fetch_results_v2] pulling $sub/"
  rsync -avh --progress --human-readable \
    "$REMOTE_BASE/$sub/" \
    "$LOCAL_ROOT/$sub/"
done

n=$(ls "$LOCAL_ROOT/degradation_v2_results"/*.json 2>/dev/null | wc -l)
echo "[fetch_results_v2] $n result JSONs in $LOCAL_ROOT/degradation_v2_results/"
