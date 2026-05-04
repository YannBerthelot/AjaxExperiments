#!/usr/bin/env bash
# Bootstrap the AjaxExperiments + Ajax + TargetGym stack on a fresh remote
# machine, in sibling layout (AjaxExperiments/pyproject.toml resolves
# Ajax and TargetGym via ``path = "../Ajax"`` / ``path = "../TargetGym"``).
#
# Pre-requisites on the remote (NOT installed by this script):
#   - git, curl
#   - python >= 3.11
#   - poetry (install via ``curl -sSL https://install.python-poetry.org | python3 -``)
#   - CUDA driver + a JAX wheel matching the installed CUDA version
#     (poetry will pull the JAX in pyproject; if you need a CUDA-specific
#      build, install it manually after ``poetry install`` completes,
#      e.g. ``poetry run pip install -U "jax[cuda12]"``)
#   - GitHub access (HTTPS PAT or SSH key configured for ``git clone``)
#
# Usage:
#   bash setup_remote.sh                # clone into $HOME/projects (default)
#   REPO_ROOT=/scratch bash setup_remote.sh
#
# After this finishes, the study runner is:
#   cd $REPO_ROOT/AjaxExperiments
#   poetry run python run_degradation_study_v2.py --gpus 0 1 2

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/projects}"
GH_USER="${GH_USER:-YannBerthelot}"

echo "[setup_remote] REPO_ROOT=$REPO_ROOT  GH_USER=$GH_USER"
mkdir -p "$REPO_ROOT"
cd "$REPO_ROOT"

# 1. Clone (or pull) the three repos.
for repo in Ajax TargetGym AjaxExperiments; do
  if [ -d "$repo/.git" ]; then
    echo "[setup_remote] $repo: pulling latest on current branch"
    git -C "$repo" pull --ff-only
  else
    echo "[setup_remote] $repo: cloning"
    git clone "https://github.com/${GH_USER}/${repo}.git"
  fi
done

# 2. Resolve and install all deps from AjaxExperiments (sibling-path layout).
cd AjaxExperiments
echo "[setup_remote] poetry install (may take several minutes on first run)"
poetry install --no-root --no-interaction

# 3. Smoke test: confirm jax + a GPU device + a tiny op.
echo "[setup_remote] smoke test"
poetry run python - <<'PY'
import jax
import jax.numpy as jnp
print("jax devices:", jax.devices())
x = jnp.array([1.0, 2.0, 3.0])
y = (x * 2.0).sum()
print("ok, jnp.sum(x*2)=", float(y))
PY

# 4. Confirm the HPO essentials are present (committed via .gitignore whitelist).
missing=0
for env in plane3dcircle cheetahrun fourtank glassfurnace barkourjoystick plane3dfigureeight; do
  for method in sac_quality_aware ibrl jsrl_curriculum jsrl residual; do
    db="hp_results/${env}/${method}/optuna.db"
    if [ ! -f "$db" ]; then
      missing=$((missing+1))
    fi
  done
done
if [ "$missing" -gt 0 ]; then
  echo "[setup_remote] WARNING: $missing optuna.db files missing under hp_results/. "
  echo "  Some (env, method) pairs may not be ready for degradation v2 yet."
  echo "  This is expected if HPO for those cells hasn't been committed to git."
fi

cat <<'EOF'

============================================================
[setup_remote] done.

Next:
  cd $REPO_ROOT/AjaxExperiments
  # Dry-run to see the work plan:
  poetry run python run_degradation_study_v2.py --gpus 0 1 2 --dry-run
  # Real run:
  poetry run python run_degradation_study_v2.py --gpus 0 1 2

Results will be written to ./degradation_v2_results/
Logs per cell will be written to ./study_v2_logs/
============================================================
EOF
