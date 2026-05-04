#!/usr/bin/env bash
# Bootstrap the AjaxExperiments + Ajax + TargetGym stack on a fresh remote
# instance. Designed for the workflow:
#   1. (on local) bash push_to_remote.sh   -- rsyncs the 3 repos
#   2. (on remote) bash bootstrap_remote.sh -- installs deps + smoke tests
#
# Pre-requisites on the remote (typically present on RunPod images):
#   - curl, git, apt (or another way to install git/rsync if missing)
#   - CUDA driver matching the JAX wheel (CUDA 12.x for jax-cuda12)
#   - GPU(s) visible via nvidia-smi
#
# This script is idempotent: safe to re-run (skips uv install if present, etc).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/projects}"
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# 0. System tools (git, rsync) via apt — needed for git-based version
#    detection (TargetGym uses dunamai) and for the rsync push step.
# ---------------------------------------------------------------------------
need_apt=()
command -v git   >/dev/null || need_apt+=(git)
command -v rsync >/dev/null || need_apt+=(rsync)
if [ "${#need_apt[@]}" -gt 0 ]; then
  echo "[bootstrap] apt-installing: ${need_apt[*]}"
  apt-get -o Acquire::http::Timeout=15 -o Acquire::Retries=2 install -y "${need_apt[@]}"
fi

# ---------------------------------------------------------------------------
# 1. uv (single static binary, no python required)
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[bootstrap] uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 2. Python 3.12 via uv (avoids apt's python3-venv + ensurepip dance)
# ---------------------------------------------------------------------------
uv python install 3.12 >/dev/null
ln -sf "$(uv python find 3.12)" /usr/local/bin/python3
echo "[bootstrap] python3: $(python3 --version)"

# ---------------------------------------------------------------------------
# 3. Poetry via uv tool (handles its own venv, ensurepip-free)
# ---------------------------------------------------------------------------
if ! command -v poetry >/dev/null 2>&1; then
  echo "[bootstrap] installing poetry via uv tool"
  uv tool install poetry
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[bootstrap] poetry: $(poetry --version)"

# Persist ~/.local/bin on PATH for future shells (poetry / uv tools live there).
PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  if [ -f "$rc" ] && ! grep -qF "$PATH_LINE" "$rc"; then
    echo "$PATH_LINE" >> "$rc"
    echo "[bootstrap] persisted PATH addition to $rc"
  fi
done

# ---------------------------------------------------------------------------
# 4. Verify the 3 repos are present (rsynced from local)
# ---------------------------------------------------------------------------
for repo in Ajax TargetGym AjaxExperiments; do
  if [ ! -d "${REPO_ROOT}/${repo}" ]; then
    echo "[bootstrap] ERROR: ${REPO_ROOT}/${repo} missing." >&2
    echo "  Run push_to_remote.sh on the local machine first." >&2
    exit 1
  fi
done
echo "[bootstrap] 3 repos present in ${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 5. poetry install — resolves Ajax + TargetGym via path = "../X"
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}/AjaxExperiments"
echo "[bootstrap] poetry install (this can take several minutes on first run)"
poetry install --no-root --no-interaction

# ---------------------------------------------------------------------------
# 6. Pip-install runtime deps that poetry's path-dep resolution misses.
#    Empirically, poetry installs Ajax + TargetGym + dev deps but not all of
#    Ajax's transitive runtime deps when those deps are declared on Ajax's
#    side and Ajax itself is a path dep. The list below is what's hit at
#    runtime by run_degradation_v2.py / sac_hyperparam_search.py /
#    compute_obs_std.py.
# ---------------------------------------------------------------------------
EXTRA_RUNTIME_DEPS=(optuna tbparse wandb)
echo "[bootstrap] pip-installing runtime extras: ${EXTRA_RUNTIME_DEPS[*]}"
poetry run pip install --quiet "${EXTRA_RUNTIME_DEPS[@]}"

# ---------------------------------------------------------------------------
# 7. JAX smoke test — confirms GPU is visible and jit works
# ---------------------------------------------------------------------------
echo "[bootstrap] jax smoke test"
poetry run python - <<'PY'
import jax, jax.numpy as jnp
print("  jax version:", jax.__version__)
print("  jax devices:", jax.devices())
print("  jax backend:", jax.default_backend())
x = jnp.array([1.0, 2.0, 3.0])
print("  sanity:    ", float((x * 2.0).sum()))
PY

# ---------------------------------------------------------------------------
# 8. Sanity: HPO availability
# ---------------------------------------------------------------------------
missing=0
for env in plane3dcircle cheetahrun fourtank glassfurnace barkourjoystick plane3dfigureeight; do
  for method in sac_quality_aware ibrl jsrl_curriculum jsrl residual; do
    [ -f "hp_results/${env}/${method}/optuna.db" ] || missing=$((missing+1))
  done
done
[ "$missing" -gt 0 ] && echo "[bootstrap] ${missing} (env, method) HPO dbs missing (some cells will be skipped at orchestrator startup)"

cat <<EOF

============================================================
[bootstrap] done.

Next steps:
  cd ${REPO_ROOT}/AjaxExperiments

  # Dry-run to verify the work plan:
  poetry run python run_degradation_study_v2.py --gpus 0 --dry-run

  # Real run (pass all GPU IDs):
  WANDB_MODE=disabled poetry run python run_degradation_study_v2.py --gpus 0 1 2 3

To pull results back to scool-bob (run on scool-bob):
  ./fetch_results_v2.sh user@<remote-host>:${REPO_ROOT}/AjaxExperiments

To re-mirror from local after edits, on scool-bob:
  bash push_to_remote.sh
============================================================
EOF
