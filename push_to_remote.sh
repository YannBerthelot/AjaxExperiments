#!/usr/bin/env bash
# Push Ajax + TargetGym + AjaxExperiments from scool-bob to a remote
# (RunPod, etc.) via rsync. No GitHub auth required. Idempotent.
#
# Usage:
#   bash push_to_remote.sh
#
# Edit the REMOTE_* vars below for each new instance. The remote SSH user,
# host, port, and key path come from the RunPod (or other provider) "SSH
# command" panel. Example:
#   ssh root@213.173.108.11 -p 17693 -i ~/.ssh/id_ed25519_runpod
# becomes:
#   REMOTE_USER=root, REMOTE_HOST=213.173.108.11, REMOTE_PORT=17693,
#   REMOTE_KEY=$HOME/.ssh/id_ed25519_runpod

set -euo pipefail

# === EDIT FOR EACH NEW INSTANCE ===
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-213.173.108.11}"
REMOTE_PORT="${REMOTE_PORT:-17693}"
REMOTE_KEY="${REMOTE_KEY:-$HOME/.ssh/id_ed25519_runpod}"
REMOTE_DIR="${REMOTE_DIR:-/root/projects}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME}"
# ==================================

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
SSH="ssh -i ${REMOTE_KEY} -p ${REMOTE_PORT}"

echo "[push] target: ${REMOTE}:${REMOTE_DIR}"
echo "[push] source: ${LOCAL_ROOT}/{Ajax,TargetGym,AjaxExperiments}"
echo

# Pre-flight: rsync on remote (added to bootstrap_remote.sh, but verify here too)
$SSH "$REMOTE" "command -v rsync >/dev/null || (echo '[push] rsync missing on remote; install via: apt install -y rsync, or run bootstrap_remote.sh first' >&2; exit 1)"
$SSH "$REMOTE" "mkdir -p $REMOTE_DIR"

# Exclusions: caches, regenerable artifacts, large TB events.
# Includes .git/ — TargetGym needs it for dunamai version detection at install time.
# Use **/ patterns so depth-2 backup dirs (e.g. hp_results/<env>/<method>/pre_full_study_backup/)
# get caught regardless of where they appear in the tree.
EXCLUDES=(
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
  --exclude='.pytest_cache/'
  --exclude='.tox/'
  --exclude='.venv/'
  --exclude='*.pkl'
  --exclude='*.pkl.bak_*'
  --exclude='checkpoints/'
  --exclude='final_results/'
  --exclude='final_results_tb/'
  --exclude='wandb/'
  --exclude='*.log'
  --exclude='study_logs/'
  --exclude='study_v2_logs/'
  --exclude='plots/'
  --exclude='plots_*/'
  --exclude='.plot_cache/'
  --exclude='paper/build/'
  --exclude='paper/figs/*.pdf'
  --exclude='videos/'                  # TargetGym animation outputs
  --exclude='figures/'                 # TargetGym misc figures
  --exclude='tensorboard/'             # top-level TB sink (Ajax has 5+GB here)
  --exclude='hyperparams/'             # Ajax exploratory hparam configs
  --exclude='degradation_results/'     # produced by the run; start clean on remote
  --exclude='degradation_v2_results/'  # ditto
  --exclude='ablation_results/'
  # hp_results/: keep optuna.db + trial_*.json, drop TB events and stale subdirs
  --exclude='**/trial_*_tb/'
  --exclude='**/*backup*/'
  --exclude='**/legacy_*/'
  --exclude='**/buggy_*/'
  --exclude='**/r_edge_bow/'
  --exclude='**/seeds*_baseline/'
  --exclude='**/*pre_full_study*/'
  --exclude='**/*.bak*'
  --exclude='**/*.pre_*/'
  # Local-only working files
  --exclude='todo.md'
  --exclude='COMMANDS.md'
  --exclude='paper/paper_stash.tex'
  --exclude='preserved/'
  --exclude='preserved_params/'
)

push_repo() {
  local name="$1"
  echo "[push] mirroring $name"
  rsync -rlptD --no-owner --no-group --info=stats0 \
    -e "$SSH" \
    "${EXCLUDES[@]}" \
    "${LOCAL_ROOT}/${name}/" \
    "${REMOTE}:${REMOTE_DIR}/${name}/"
}

push_repo Ajax
push_repo TargetGym
push_repo AjaxExperiments

echo
echo "[push] done."
$SSH "$REMOTE" "du -sh ${REMOTE_DIR}/* 2>/dev/null"
echo
echo "[push] next: on the remote, run bootstrap_remote.sh to install deps and run the smoke test."
