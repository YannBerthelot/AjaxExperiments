#!/usr/bin/env bash
# Wrapper to launch run_degradation_study_v2.py from a non-interactive shell
# (e.g. ssh-and-run). Sets PATH, WANDB_MODE, and cd's to the project root so
# all subprocess dispatches work the same as from an interactive session.
#
# Usage (on the remote):
#   bash run_v2_study.sh --gpus 0 1 2 3                  # full work plan
#   bash run_v2_study.sh --gpus 0 --dry-run              # just print plan
#   bash run_v2_study.sh --gpus 0 1 --types obs_noise    # subset of types
#
# Usage (from local, ssh-and-run):
#   ssh user@remote -p PORT -i KEY 'bash /root/projects/AjaxExperiments/run_v2_study.sh --gpus 0 1 2 3'

set -euo pipefail

# Make poetry / uv visible regardless of shell startup (login vs non-login).
export PATH="$HOME/.local/bin:$PATH"

# Disable wandb online sync — we're on a remote that may not have wandb auth,
# and the v2 study only needs TB events for the per-seed metric extraction.
export WANDB_MODE="${WANDB_MODE:-disabled}"

REPO_ROOT="${REPO_ROOT:-/root/projects}"
cd "${REPO_ROOT}/AjaxExperiments"

# Sanity: poetry must resolve.
if ! command -v poetry >/dev/null 2>&1; then
  echo "[run_v2_study] ERROR: poetry not found on PATH ($PATH)." >&2
  echo "  Run bootstrap_remote.sh first." >&2
  exit 1
fi

exec poetry run python run_degradation_study_v2.py "$@"
