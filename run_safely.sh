#!/usr/bin/env bash
# Launch a long-running command that survives SSH logout / tmux teardown.
#
# Usage:
#   ./run_safely.sh poetry run python run_full_study.py --gpus 0 1 2
#   ./run_safely.sh poetry run python sac_hyperparam_search.py --env CheetahRun --method sac --gpus 2 ...
#
# Behavior:
#   * If already inside tmux ($TMUX) or screen ($STY), exec directly: tmux
#     and screen already detach the child from the SSH pty, so SIGHUP on
#     logout never reaches it.
#   * Otherwise, re-launch under setsid + nohup with stdout/stderr to a
#     timestamped log under logs/, print the PID and log path, and return.
#     The launched process is fully detached from the controlling pty and
#     survives SSH disconnect, logout, and parent shell exit.
#
# Diagnosing a previous logout-kill: the giveaway is a TB events stream
# that ends within a minute of the user's session ending, with no result
# JSON written and no OOM trace in dmesg. That's a controlling-terminal
# SIGHUP, not a cron or a kernel kill.

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

if [[ -n "${TMUX:-}" || -n "${STY:-}" ]]; then
  exec "$@"
fi

LOG_DIR="${LOG_DIR:-$(pwd)/logs}"
mkdir -p "$LOG_DIR"
ts=$(date +%Y%m%d-%H%M%S)
slug=$(basename "${1##*/}" | tr -c 'a-zA-Z0-9._-' '_')
log="$LOG_DIR/${ts}_${slug}.log"

echo "[run_safely] not inside tmux/screen; detaching from tty" >&2
echo "[run_safely] log: $log" >&2

setsid nohup "$@" </dev/null >"$log" 2>&1 &
pid=$!
disown -h "%1" 2>/dev/null || true

echo "[run_safely] launched pid=$pid"
echo "[run_safely] tail logs with:  tail -f $log"
echo "[run_safely] stop with:       kill -TERM -$pid   # negative pid kills the whole session"
