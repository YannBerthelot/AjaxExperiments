"""Master orchestrator for the full EDGE study.

Handles the complete experimental plan:
  - HPO for every (env, method) at 1M env steps, 20 seeds per trial
  - final evaluation at 2M env steps, 100 seeds per (env, method)
  - expert-degradation sensitivity at 2M env steps, 100 seeds per
    (env, method, sigma) in sigma in {0.25, 0.50, 0.75}

A per-GPU dispatcher runs 3 workers (one pinned to each of GPUs 0/1/2) that
pull work units from a shared queue. Each work unit uses one GPU, so three
different (env, method) studies run in parallel across the three GPUs.

Every spawn is gated by GPU_check.sh (SMACT + DRAMA + VRAM + context count)
to avoid colliding with external processes.

Idempotent: on startup we scan existing Optuna DBs and final-eval JSON
files and skip work units that are already done.

Usage:
    poetry run python run_full_study.py           # run the plan
    poetry run python run_full_study.py --dry-run # print the plan and exit
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Experimental plan
# ---------------------------------------------------------------------------

ENVS: List[str] = [
    "Plane3DCircle",     # headline env — first to finish so we can analyse
    "CheetahRun",        # generalisation env — second
    "Reactor",
    "FourTank",
    "BarkourJoystick",
]

# Core baseline pair. Every (env, sac) and (env, edge) HPO runs before any
# non-core HPO, so early results give us the flagship comparison without
# waiting for the full grid.
_CORE_METHODS = ("sac", "sac_quality_aware", "edge")

# Baselines that need HPO + final eval. Each is a --method value accepted
# by sac_hyperparam_search.py. If a method is missing dispatch there, the
# subprocess will error and the master will mark the unit FAILED.
METHODS: List[str] = [
    # Paper methods.
    "sac",                   # vanilla SAC baseline
    "edge",                  # edge-eps: SAC + ε-greedy expert gating (gating-ablation)
    "sac_quality_aware",     # SAC + LCB expert gating + obs+state aug (proposed)
]

# Display labels for the status table.
_METHOD_DISPLAY: dict = {
    "edge": "edge-eps",
    "sac_quality_aware": "edge-qa",
}

HPO_TIMESTEPS = 1_000_000
HPO_SEEDS = 20
HPO_TRIALS = 30

FINAL_TIMESTEPS = 2_000_000
FINAL_SEEDS = 100  # single vmapped run, adjust if VRAM tight

# Per-env final-eval budget override. Set to 1M for envs whose best-trial
# learning curves plateau within the HPO budget (checked manually at the end
# of each env's HPO phase). Falling back to ``FINAL_TIMESTEPS`` otherwise.
#
# Verification log:
#   CheetahRun : SAC trial 37 plateaus at ~660 (+12 over last 100k, 1.8%)
#   Plane3DCircle: both SAC and EDGE still climbing at 1M -> stays at 2M
#   Reactor, FourTank, BarkourJoystick: unknown until HPO finishes
_FINAL_TIMESTEPS_BY_ENV: dict = {
    "CheetahRun": 1_000_000,
}

# Per-env chunking of the final-eval train() call. The per-step metric tensor
# inside ajax.agent.train is materialized at call end with shape
# [n_seeds, n_steps] per scalar; for 100 seeds x 2M steps that is ~800 MB per
# metric and OOMs the L40S during post-scan autotune. Splitting into N chunks
# halves/thirds the tensor size while preserving the full 100-seed batch via
# checkpoint-resume between calls. 1 = off (default).
_FINAL_CHUNKS_BY_ENV: dict = {
    # Placeholder: no env currently chunks. Plane3DCircle was tried with
    # chunking but the dominant memory consumer is the 1M replay buffer,
    # which chunking doesn't shrink. We fall back to running 50 seeds at a
    # time instead (see FINAL_SEEDS workflow).
}


def final_timesteps_for(env: str) -> int:
    return _FINAL_TIMESTEPS_BY_ENV.get(env, FINAL_TIMESTEPS)


def final_chunks_for(env: str) -> int:
    return _FINAL_CHUNKS_BY_ENV.get(env, 1)

DEGRADATION_SIGMAS: List[float] = [0.25, 0.50, 0.75]
DEGRADATION_SEEDS = 100

# Tuned-expert reference scores per env (20-seed mean from the benchmark-
# table eval). Shown in the status table so each method's numbers land
# against the "beat the expert" goalpost at a glance.
EXPERT_REFERENCE: dict = {
    "Plane3DCircle": 3784.0,
    "Reactor": 21479.0,
    "FourTank": 246.0,
    "CheetahRun": 222.0,
    "BarkourJoystick": 6.6,  # 20-seed mean of hand-tuned trot CPG (std 4.56)
}

# EDGE design ablations (§5.3). Variants from run_ablation_eval.ABLATION_VARIANTS;
# duplicated here so we don't import jax just to build the queue.
ABLATION_VARIANTS: List[str] = [
    "gating_argmax", "mc_pretrain",
    "decay_0.05", "decay_0.30", "decay_0.50", "decay_1.00",
    "eps_0.1", "eps_0.25", "eps_0.75", "eps_0.9",
]
# Envs the ablation study runs on, in order of priority (FourTank first).
ABLATION_ENVS: List[str] = ["FourTank", "CheetahRun"]
ABLATION_SEEDS = 100

GPUS: List[int] = [0, 1, 2]
POLL_INTERVAL_S = 30
GPU_CHECK_RETRY_S = 60

ROOT = Path(__file__).resolve().parent
HP_RESULTS = ROOT / "hp_results"
FINAL_RESULTS = ROOT / "final_results"
DEGRADATION_RESULTS = ROOT / "degradation_results"
ABLATION_RESULTS = ROOT / "ablation_results"
LOG_DIR = ROOT / "study_logs"
GPU_CHECK = ROOT / "GPU_check.sh"


# ---------------------------------------------------------------------------
# Work unit types
# ---------------------------------------------------------------------------


@dataclass(order=True)
class WorkUnit:
    # Lower priority -> runs first.
    priority: int
    kind: str = field(compare=False)       # hpo | final_eval | degradation | ablation
    env: str = field(compare=False)
    method: str = field(compare=False)
    sigma: Optional[float] = field(default=None, compare=False)
    variant: Optional[str] = field(default=None, compare=False)

    def name(self) -> str:
        tag = f"{self.env}.{self.method}"
        if self.kind == "hpo":
            return f"HPO[{tag}]"
        if self.kind == "final_eval":
            return f"FINAL[{tag}]"
        if self.kind == "degradation":
            return f"DEGRADE[{tag}.sigma={self.sigma}]"
        if self.kind == "ablation":
            return f"ABLATE[{self.env}.{self.variant}]"
        raise ValueError(self.kind)


# ---------------------------------------------------------------------------
# Idempotence: skip work units whose artifact already exists
# ---------------------------------------------------------------------------


def _hpo_inflight_metric(env: str, method: str) -> Optional[tuple]:
    """For the most-recent RUNNING trial in (env, method), read the per-seed-
    aggregated eval mean and return (mean, step) or None if not available.

    Reads the worker's TB folder on disk; works while training is still
    in-flight, even before any trial completes.
    """
    candidates = [
        HP_RESULTS / env.lower() / method / "optuna.db",
        HP_RESULTS / env.lower() / "optuna.db",
    ]
    trial_num = None
    for db_path in candidates:
        if not db_path.exists():
            continue
        try:
            import optuna
            from optuna.trial import TrialState
            s = optuna.load_study(
                study_name=_study_name(env, method),
                storage=f"sqlite:///{db_path}",
            )
            running = [t for t in s.trials if t.state == TrialState.RUNNING]
            if running:
                trial_num = max(t.number for t in running)
                break
        except Exception:
            pass
    if trial_num is None:
        return None
    tb_root = HP_RESULTS / env.lower() / method / f"trial_{trial_num:03d}_tb" / "tensorboard"
    if not tb_root.exists():
        return None
    import glob, statistics, struct, os
    try:
        from tensorboardX.proto import event_pb2
    except Exception:
        return None
    def _iter(p):
        try:
            with open(p, "rb") as f:
                data = f.read()
        except Exception:
            return
        i, N = 0, len(data)
        while i + 12 <= N:
            ln = struct.unpack("<Q", data[i:i+8])[0]
            i += 12
            if i + ln + 4 > N:
                break
            try:
                yield event_pb2.Event.FromString(data[i:i+ln])
            except Exception:
                pass
            i += ln + 4
    # Match EXACTLY the canonical metric tags. Returns
    # (eval_mean, train_mean, step) where any of the means may be None.
    eval_tag = "Eval/episodic_mean_reward"
    train_tag = "Train/episodic_mean_reward"
    eval_per_step: dict = {}
    train_per_step: dict = {}
    for sd in sorted(os.listdir(tb_root)):
        for f in glob.glob(str(tb_root / sd / "events.out.tfevents.*")):
            for ev in _iter(f):
                for v in ev.summary.value:
                    if v.tag == eval_tag:
                        eval_per_step.setdefault(ev.step, []).append(v.simple_value)
                    elif v.tag == train_tag:
                        train_per_step.setdefault(ev.step, []).append(v.simple_value)
    if not eval_per_step and not train_per_step:
        return None
    eval_mean = train_mean = None
    step = 0
    if eval_per_step:
        eval_step = max(eval_per_step)
        eval_mean = statistics.mean(eval_per_step[eval_step])
        step = max(step, eval_step)
    if train_per_step:
        train_step = max(train_per_step)
        train_mean = statistics.mean(train_per_step[train_step])
        step = max(step, train_step)
    return eval_mean, train_mean, step


def _hpo_best_score(env: str, method: str) -> Optional[float]:
    """Return the best COMPLETE trial value in the study, or None."""
    candidates = [
        HP_RESULTS / env.lower() / "optuna.db",
        HP_RESULTS / env.lower() / method / "optuna.db",
    ]
    for db_path in candidates:
        if not db_path.exists():
            continue
        try:
            import optuna
            from optuna.trial import TrialState

            study_name = _study_name(env, method)
            s = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///{db_path}",
            )
            done = [t for t in s.trials if t.state == TrialState.COMPLETE]
            if done:
                return max(t.value for t in done)
        except Exception:
            pass
    return None


def _final_eval_score(env: str, method: str) -> Optional[float]:
    """Read the final-eval JSON for (env, method) and return the best score
    if present, else None. The JSON schema is set by ``run_final_eval.py``.
    """
    import json as _json
    p = FINAL_RESULTS / f"{env.lower()}__{method}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            payload = _json.load(f)
    except Exception:
        return None
    # Preferred: eval_metrics.episodic_mean_reward (final window mean)
    em = payload.get("eval_metrics")
    if isinstance(em, dict):
        for key in ("episodic_mean_reward", "final_mean",
                    "Eval/episodic_mean_reward"):
            v = em.get(key)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, list) and v and isinstance(v[-1], (int, float)):
                return float(v[-1])
    # Fallback: the best-HPO metric if final-eval didn't serialize returns
    v = payload.get("best_hpo_score")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _hpo_trial_count(env: str, method: str) -> Optional[int]:
    """Return the number of COMPLETE+PRUNED trials in the study, or None if
    the study doesn't exist yet."""
    candidates = [
        HP_RESULTS / env.lower() / "optuna.db",
        HP_RESULTS / env.lower() / method / "optuna.db",
    ]
    for db_path in candidates:
        if not db_path.exists():
            continue
        try:
            import optuna
            from optuna.trial import TrialState

            study_name = _study_name(env, method)
            s = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///{db_path}",
            )
            return sum(
                1
                for t in s.trials
                if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
            )
        except Exception:
            pass
    return None


def _hpo_db_complete(env: str, method: str) -> bool:
    """True if the Optuna study for (env, method) has >= HPO_TRIALS completed."""
    n = _hpo_trial_count(env, method)
    return n is not None and n >= HPO_TRIALS


def _study_name(env: str, method: str) -> str:
    # Matches the convention used inside sac_hyperparam_search.py.
    if method == "sac":
        return f"sac_hp_search_{env.lower()}"
    return f"{method}_hp_search_{env.lower()}"


def _final_eval_done(env: str, method: str) -> bool:
    p = FINAL_RESULTS / f"{env.lower()}__{method}.json"
    return p.exists()


def _degradation_done(env: str, method: str, sigma: float) -> bool:
    p = DEGRADATION_RESULTS / f"{env.lower()}__{method}__sigma{sigma}.json"
    return p.exists()


def _ablation_done(env: str, variant: str) -> bool:
    p = ABLATION_RESULTS / f"{env.lower()}__{variant}.json"
    return p.exists()


# ---------------------------------------------------------------------------
# Build the work queue
# ---------------------------------------------------------------------------


_HEADLINE_ENVS = ["Plane3DCircle", "CheetahRun"]
_OTHER_ENVS = [e for e in ENVS if e not in _HEADLINE_ENVS]


def _unit_priority(kind: str, env: str, method: str,
                   sigma: Optional[float] = None,
                   variant: Optional[str] = None) -> int:
    """Env-level depth-first with a two-tier env group:

      tier 0: headline envs (Plane3DCircle, CheetahRun) finish their
              HPO -> final_eval -> degradation -> non-core HPO -> ...
              pipeline before tier 1 starts.
      tier 1: remaining envs (Reactor, FourTank, BarkourJoystick) run
              the same pipeline afterwards.

    Within a tier, phase order is:
        0  core HPO          (sac, edge)
        1  core final_eval
        2  core degradation  (edge only)
        3  non-core HPO      (ibrl, jsrl, residual)
        4  non-core final_eval
        5  non-core degradation

    Ablation runs last (tier 2, phase 6), FourTank first then CheetahRun.
    """
    if kind == "ablation":
        env_rank = ABLATION_ENVS.index(env)
        var_rank = ABLATION_VARIANTS.index(variant) if variant else 0
        return 2 * 100_000 + 6 * 10_000 + env_rank * 100 + var_rank
    is_core = method in _CORE_METHODS
    phase_map = {
        ("hpo", True): 0,
        ("final_eval", True): 1,
        ("degradation", True): 2,
        ("hpo", False): 3,
        ("final_eval", False): 4,
        ("degradation", False): 5,
    }
    phase = phase_map[(kind, is_core)]
    is_headline = env in _HEADLINE_ENVS
    tier = 0 if is_headline else 1
    group = _HEADLINE_ENVS if is_headline else _OTHER_ENVS
    env_rank = group.index(env)
    method_rank = METHODS.index(method)
    sigma_rank = (
        DEGRADATION_SIGMAS.index(sigma) if sigma is not None else 0
    )
    return (
        tier * 100_000
        + phase * 10_000
        + env_rank * 100
        + method_rank * 10
        + sigma_rank
    )


def build_queue() -> "queue.PriorityQueue[WorkUnit]":
    q: "queue.PriorityQueue[WorkUnit]" = queue.PriorityQueue()
    # HPOs — core methods first across all envs, then non-core.
    for env in ENVS:
        for method in METHODS:
            if _hpo_db_complete(env, method):
                print(f"[skip] HPO already done: {env}/{method}")
                continue
            q.put(WorkUnit(
                priority=_unit_priority("hpo", env, method),
                kind="hpo", env=env, method=method,
            ))
    # Final evals.
    for env in ENVS:
        for method in METHODS:
            if _final_eval_done(env, method):
                print(f"[skip] final-eval already done: {env}/{method}")
                continue
            q.put(WorkUnit(
                priority=_unit_priority("final_eval", env, method),
                kind="final_eval", env=env, method=method,
            ))
    # Degradation study (only expert-using methods).
    degrade_methods = [m for m in METHODS if m in ("edge", "sac_quality_aware")]
    for env in ENVS:
        for method in degrade_methods:
            for sigma in DEGRADATION_SIGMAS:
                if _degradation_done(env, method, sigma):
                    print(
                        f"[skip] degradation already done: "
                        f"{env}/{method}/sigma={sigma}"
                    )
                    continue
                q.put(WorkUnit(
                    priority=_unit_priority("degradation", env, method, sigma),
                    kind="degradation", env=env, method=method, sigma=sigma,
                ))
    # EDGE design ablations (end of study; FourTank then CheetahRun).
    for env in ABLATION_ENVS:
        for variant in ABLATION_VARIANTS:
            if _ablation_done(env, variant):
                print(f"[skip] ablation already done: {env}/{variant}")
                continue
            q.put(WorkUnit(
                priority=_unit_priority("ablation", env, "edge", variant=variant),
                kind="ablation", env=env, method="edge", variant=variant,
            ))
    return q


# ---------------------------------------------------------------------------
# GPU availability check (wraps GPU_check.sh)
# ---------------------------------------------------------------------------


def _gpu_check_free_set() -> Optional[set]:
    try:
        result = subprocess.run(
            ["bash", str(GPU_CHECK)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Recommended: GPU "):
            after = line[len("Recommended: GPU "):].split("(", 1)[0].strip()
            return {int(tok) for tok in after.split() if tok.isdigit()}
        if line.startswith("No fully-free GPU"):
            return set()
    return set()


_gpu_check_cache = {"ts": 0.0, "set": None}
_gpu_check_lock = threading.Lock()


def gpu_is_free(gpu_id: int, ttl: float = 20.0) -> bool:
    """Threaded-safe free check with short cache. Fail-open on check failure."""
    with _gpu_check_lock:
        now = time.time()
        if now - _gpu_check_cache["ts"] > ttl or _gpu_check_cache["set"] is None:
            _gpu_check_cache["set"] = _gpu_check_free_set()
            _gpu_check_cache["ts"] = now
        free = _gpu_check_cache["set"]
    if free is None:
        return True
    return gpu_id in free


def wait_until_gpu_free(gpu_id: int):
    while not gpu_is_free(gpu_id):
        time.sleep(GPU_CHECK_RETRY_S)


# ---------------------------------------------------------------------------
# Work dispatch: turn a WorkUnit into a subprocess call
# ---------------------------------------------------------------------------


def _env_with_gpu(gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Ensure master stays on CPU if needed (we're spawning, not computing).
    env.pop("JAX_PLATFORMS", None)
    return env


def _log_path(unit: WorkUnit) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    parts = [unit.kind, unit.env, unit.method]
    if unit.sigma is not None:
        parts.append(f"sigma{unit.sigma}")
    return LOG_DIR / ("_".join(parts) + ".log")


def build_cmd(unit: WorkUnit, gpu_id: int) -> List[str]:
    python = sys.executable
    if unit.kind == "hpo":
        return [
            python, str(ROOT / "sac_hyperparam_search.py"),
            "--env", unit.env,
            "--method", unit.method,
            "--gpus", str(gpu_id),
            "--trials", str(HPO_TRIALS),
            "--n-timesteps", str(HPO_TIMESTEPS),
            "--n-seeds", str(HPO_SEEDS),
        ]
    if unit.kind == "final_eval":
        cmd = [
            python, str(ROOT / "run_final_eval.py"),
            "--env", unit.env,
            "--method", unit.method,
            "--gpu", str(gpu_id),
            "--n-timesteps", str(final_timesteps_for(unit.env)),
            "--n-seeds", str(FINAL_SEEDS),
        ]
        n_chunks = final_chunks_for(unit.env)
        if n_chunks > 1:
            cmd += ["--n-chunks", str(n_chunks)]
        return cmd
    if unit.kind == "degradation":
        cmd = [
            python, str(ROOT / "run_degradation_eval.py"),
            "--env", unit.env,
            "--method", unit.method,
            "--gpu", str(gpu_id),
            "--sigma", str(unit.sigma),
            "--n-timesteps", str(final_timesteps_for(unit.env)),
            "--n-seeds", str(DEGRADATION_SEEDS),
        ]
        n_chunks = final_chunks_for(unit.env)
        if n_chunks > 1:
            cmd += ["--n-chunks", str(n_chunks)]
        return cmd
    if unit.kind == "ablation":
        return [
            python, str(ROOT / "run_ablation_eval.py"),
            "--env", unit.env,
            "--variant", unit.variant,
            "--gpu", str(gpu_id),
            "--n-timesteps", str(final_timesteps_for(unit.env)),
            "--n-seeds", str(ABLATION_SEEDS),
        ]
    raise ValueError(unit.kind)


def dispatch(unit: WorkUnit, gpu_id: int) -> int:
    """Spawn the subprocess, wait for it, return exit code."""
    cmd = build_cmd(unit, gpu_id)
    log_path = _log_path(unit)
    env = _env_with_gpu(gpu_id)
    with open(log_path, "w") as log:
        log.write(f"[master] launching {unit.name()} on GPU {gpu_id}\n")
        log.write(f"[master] cmd: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ret = proc.wait()
        log.write(f"[master] exited with {ret}\n")
    return ret


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------


class GPUWorker(threading.Thread):
    def __init__(self, gpu_id: int, work_q: "queue.PriorityQueue[WorkUnit]",
                 done_counter: dict, done_lock: threading.Lock):
        super().__init__(daemon=True, name=f"gpu-{gpu_id}")
        self.gpu_id = gpu_id
        self.work_q = work_q
        self.done_counter = done_counter
        self.done_lock = done_lock

    def run(self):
        while True:
            try:
                unit = self.work_q.get(timeout=2)
            except queue.Empty:
                # Queue drained — nothing left to do.
                return
            try:
                print(f"[gpu{self.gpu_id}] picked {unit.name()}")
                wait_until_gpu_free(self.gpu_id)
                ret = dispatch(unit, self.gpu_id)
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[gpu{self.gpu_id}] {unit.name()} -> {status}")
                with self.done_lock:
                    self.done_counter[status] = (
                        self.done_counter.get(status, 0) + 1
                    )
            except Exception as e:
                print(f"[gpu{self.gpu_id}] {unit.name()} raised {e!r}")
            finally:
                self.work_q.task_done()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _running_units() -> dict:
    """Return a dict mapping (kind, env, method[, sigma]) -> (pid, gpu, etime)
    for every work unit currently running, by scanning ``ps`` output.

    Kinds:
      - HPO master:    ``sac_hyperparam_search.py --env E --method M --gpus G ...``
      - final_eval:    ``run_final_eval.py --env E --method M --gpu G ...``
      - degradation:   ``run_degradation_eval.py --env E --method M --gpu G --sigma S ...``
    """
    import re
    try:
        out = subprocess.run(
            # -ww disables the truncation ``ps`` applies when its terminal
            # is narrower than the cmdline; running under ``watch`` sets
            # a tight COLUMNS and silently trims ``--method X`` off the
            # tail, which breaks our regex below.
            ["ps", "-eww", "-o", "pid,etime,command"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return {}

    units: dict = {}
    re_gpus = re.compile(r"--gpus?\s+(\d+)")
    re_env = re.compile(r"--env\s+(\S+)")
    re_method = re.compile(r"--method\s+(\S+)")
    re_sigma = re.compile(r"--sigma\s+(\S+)")
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, etime, cmd = parts
        if "--worker" in cmd:
            continue  # workers are spawned by HPO masters; only count masters
        kind = None
        if "sac_hyperparam_search.py" in cmd:
            kind = "hpo"
        elif "run_final_eval.py" in cmd:
            kind = "final_eval"
        elif "run_degradation_eval.py" in cmd:
            kind = "degradation"
        elif "run_ablation_eval.py" in cmd:
            kind = "ablation"
        else:
            continue
        env_m = re_env.search(cmd)
        meth_m = re_method.search(cmd)
        gpu_m = re_gpus.search(cmd)
        if not env_m:
            continue
        # Ablation cmdlines don't carry --method (method is always edge).
        if not meth_m and kind != "ablation":
            continue
        env_name = env_m.group(1)
        method = meth_m.group(1) if meth_m else "edge"
        gpu = gpu_m.group(1) if gpu_m else "?"
        key = (kind, env_name, method)
        if kind == "degradation":
            sig = re_sigma.search(cmd)
            key = (kind, env_name, method, sig.group(1) if sig else "?")
        if kind == "ablation":
            re_variant = re.compile(r"--variant\s+(\S+)")
            var_m = re_variant.search(cmd)
            key = (kind, env_name, "edge", var_m.group(1) if var_m else "?")
        units[key] = (pid, gpu, etime.strip())
    return units


# Rough per-unit wall-time baselines (seconds). Updated from live data when
# available (see ``_eta_budget`` below).
#
# Calibrated from the first wave of HPO trials:
#   - 1M-step HPO trial: 33 min median, 85 min when a trial runs full-length
#     without pruning (~15% of trials). We use a 45-min average as the default
#     because TPE-pruner kicks most trials well below the full budget.
#   - 2M-step final/degradation/ablation unit: no pruning, full 2M every time.
#     At the measured 82 min/1M for full-length trials, 2M ≈ 167 min; round to
#     170 min (2.8 h).
_HPO_SEC_PER_TRIAL_DEFAULT = 45 * 60
# Per-1M-step cost for a full (unpruned) run. Used to derive final/degradation/
# ablation wall-time estimates: for an env at 2M budget -> 2 * _SEC_PER_M; for
# CheetahRun at 1M budget -> 1 * _SEC_PER_M.
_SEC_PER_MILLION_TIMESTEPS = 85 * 60


def _final_sec_for(env: str) -> int:
    ts = _FINAL_TIMESTEPS_BY_ENV.get(env, FINAL_TIMESTEPS)
    return (ts // 1_000_000) * _SEC_PER_MILLION_TIMESTEPS


def _parse_etime(etime: str) -> int:
    """Convert ``ps etime`` format (DD-HH:MM:SS or HH:MM:SS or MM:SS) to seconds."""
    days = 0
    if "-" in etime:
        d_part, etime = etime.split("-", 1)
        days = int(d_part)
    parts = [int(p) for p in etime.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def _fmt_hms(secs: float) -> str:
    """Format seconds as ``Xd YYhZZm`` / ``HhMMm`` / ``MMm``."""
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    if secs < 86400:
        return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"
    days = secs // 86400
    hours = (secs % 86400) // 3600
    return f"{days}d{hours:02d}h"


def _hpo_eta_seconds(n_done: int, etime_sec: int) -> int:
    if n_done > 0:
        per_trial = etime_sec / n_done
    else:
        per_trial = _HPO_SEC_PER_TRIAL_DEFAULT
    remaining = max(HPO_TRIALS - n_done, 0)
    return int(remaining * per_trial)


def print_status() -> None:
    """Print a per-category / per-env progress table with live-run annotations."""
    running = _running_units()
    hpo_running = {
        (key[1], key[2]): v for key, v in running.items() if key[0] == "hpo"
    }
    final_running = {
        (key[1], key[2]): v for key, v in running.items() if key[0] == "final_eval"
    }
    deg_running = {
        (key[1], key[2], key[3]): v for key, v in running.items()
        if key[0] == "degradation"
    }
    abl_running_by_env = {}  # env -> count of running variants
    for key in running:
        if key[0] == "ablation":
            abl_running_by_env[key[1]] = abl_running_by_env.get(key[1], 0) + 1

    print(
        f"{'env':<20} {'expert':>10} {'method':<10} {'HPO':<46} "
        f"{'HPO_best':<10} {'FINAL':<14} {'FINAL_best':<11} {'DEG':<8}"
    )
    print("-" * 110)
    n_hpo_done = n_final_done = n_deg_done = 0
    n_hpo_total = n_final_total = n_deg_total = 0
    total_remaining_sec = 0  # sum of remaining time across all pending/running units
    for env in ENVS:
        expert_ref = EXPERT_REFERENCE.get(env)
        expert_str = (
            f"{expert_ref:>10.1f}" if expert_ref is not None else f"{'tbd':>10}"
        )
        for i, method in enumerate(METHODS):
            # Only show the expert once per env block (first row).
            expert_cell = expert_str if i == 0 else " " * 10
            hpo_done = _hpo_db_complete(env, method)
            final_done = _final_eval_done(env, method)
            # HPO column
            if hpo_done:
                hpo_str = f"done {HPO_TRIALS}/{HPO_TRIALS}"
            elif (env, method) in hpo_running:
                pid, gpu, et = hpo_running[(env, method)]
                n_done = _hpo_trial_count(env, method) or 0
                eta_sec = _hpo_eta_seconds(n_done, _parse_etime(et))
                hpo_str = f"gpu{gpu} {n_done}/{HPO_TRIALS} ETA{_fmt_hms(eta_sec)}"
                inflight = _hpo_inflight_metric(env, method)
                if inflight is not None:
                    eval_m, train_m, step = inflight
                    eval_s = f"E={eval_m:.0f}" if eval_m is not None else "E=-"
                    train_s = f"T={train_m:.0f}" if train_m is not None else "T=-"
                    hpo_str += f" ({eval_s}/{train_s}@{step//1000}k)"
                total_remaining_sec += eta_sec
            else:
                hpo_str = "-"
                total_remaining_sec += _HPO_SEC_PER_TRIAL_DEFAULT * HPO_TRIALS
            hpo_best = _hpo_best_score(env, method)
            hpo_best_str = f"{hpo_best:10.2f}" if hpo_best is not None else "-"
            # FINAL column
            if final_done:
                final_str = "done"
            elif (env, method) in final_running:
                pid, gpu, et = final_running[(env, method)]
                # Assume unit finishes at nominal wall time.
                elapsed = _parse_etime(et)
                eta_sec = max(_final_sec_for(env) - elapsed, 0)
                final_str = f"gpu{gpu} ETA{_fmt_hms(eta_sec)}"
                total_remaining_sec += eta_sec
            else:
                final_str = "-"
                total_remaining_sec += _final_sec_for(env)
            final_best = _final_eval_score(env, method)
            final_best_str = (
                f"{final_best:11.2f}" if final_best is not None else "-"
            )
            # DEG column
            if method in ("edge", "sac_quality_aware"):
                deg_done = sum(
                    _degradation_done(env, method, s) for s in DEGRADATION_SIGMAS
                )
                running_sigs = sum(
                    1 for s in DEGRADATION_SIGMAS
                    if (env, method, str(s)) in deg_running
                )
                deg_str = f"{deg_done}/{len(DEGRADATION_SIGMAS)}"
                if running_sigs:
                    deg_str += f"+{running_sigs}r"
                n_deg_total += len(DEGRADATION_SIGMAS)
                n_deg_done += deg_done
                # Remaining degradation units for this (env, method):
                deg_remaining = len(DEGRADATION_SIGMAS) - deg_done
                # Approximate: unknown how much of each running one is left.
                total_remaining_sec += deg_remaining * _final_sec_for(env)
            else:
                deg_str = "n/a"
            disp = _METHOD_DISPLAY.get(method, method)
            print(
                f"{env:<20} {expert_cell} {disp:<10} {hpo_str:<46} "
                f"{hpo_best_str:<10} {final_str:<14} "
                f"{final_best_str:<11} {deg_str:<8}"
            )
            n_hpo_total += 1
            n_final_total += 1
            n_hpo_done += int(hpo_done)
            n_final_done += int(final_done)

    # Summary line for ablation progress (not per-env/method since ablations
    # are indexed by variant and live outside the main table).
    n_abl_total = len(ABLATION_ENVS) * len(ABLATION_VARIANTS)
    n_abl_done = sum(
        _ablation_done(env, v)
        for env in ABLATION_ENVS for v in ABLATION_VARIANTS
    )
    n_abl_running = sum(abl_running_by_env.values())
    # Ablations run on FourTank then CheetahRun; use per-env final cost.
    for abl_env in ABLATION_ENVS:
        remaining = sum(
            0 if _ablation_done(abl_env, v) else 1 for v in ABLATION_VARIANTS
        )
        total_remaining_sec += remaining * _final_sec_for(abl_env)
    print("-" * 110)
    abl_tag = f"{n_abl_done}/{n_abl_total}"
    if n_abl_running:
        abl_tag += f"+{n_abl_running}r"
    print(f"ablations: {abl_tag}  (FourTank then CheetahRun, at study tail)")
    n_running = len(running)
    n_gpus = max(len(GPUS), 1)
    # Parallel division across GPUs (assumes near-perfect load balance).
    study_eta_sec = total_remaining_sec // n_gpus
    print(
        f"totals: HPO {n_hpo_done}/{n_hpo_total}, "
        f"FINAL {n_final_done}/{n_final_total}, "
        f"DEG {n_deg_done}/{n_deg_total}, "
        f"RUNNING {n_running}"
    )
    print(
        f"study ETA (÷{n_gpus} GPUs): {_fmt_hms(study_eta_sec)} "
        f"(raw serial: {_fmt_hms(total_remaining_sec)})"
    )
    # Refresh the baseline-vs-edge-qa curve overlays (PNGs land at
    # baseline_curves_{env}.png in the project root).
    try:
        from plot_baseline_curves import update_curves
        written = update_curves()
        if written:
            print("baseline curves: " + ", ".join(str(p.name) for p in written))
    except Exception as e:
        print(f"[plot_baseline_curves] skipped ({e!r})")


def clean_results(confirm: bool) -> None:
    """Wipe all result artifacts for a clean-slate re-run.

    Removes: hp_results/, final_results/, degradation_results/,
    final_results_tb/, checkpoints/, study_logs/. Requires ``--confirm`` so
    this cannot be invoked by accident.
    """
    import shutil

    to_delete = [
        ROOT / "hp_results",
        FINAL_RESULTS,
        DEGRADATION_RESULTS,
        ABLATION_RESULTS,
        ROOT / "final_results_tb",
        ROOT / "checkpoints",
        LOG_DIR,
    ]
    present = [p for p in to_delete if p.exists()]
    print("[clean] the following directories will be removed:")
    for p in present:
        print(f"  {p}")
    if not present:
        print("  (none)")
        return
    if not confirm:
        print(
            "[clean] dry-run (no --confirm). Re-run with --confirm to actually delete."
        )
        return
    for p in present:
        shutil.rmtree(p)
        print(f"[clean] removed {p}")
    print("[clean] done. A fresh run_full_study.py run will start from zero.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without executing")
    parser.add_argument("--status", action="store_true",
                        help="Print a progress table and exit")
    parser.add_argument("--clean", action="store_true",
                        help="Remove all result artifacts for a clean slate")
    parser.add_argument("--confirm", action="store_true",
                        help="Required with --clean; otherwise --clean is a dry run")
    parser.add_argument("--gpus", nargs="+", type=int, default=GPUS,
                        help="GPU ids to dispatch onto (default: 0 1 2)")
    args = parser.parse_args()

    if args.status:
        print_status()
        return
    if args.clean:
        clean_results(confirm=args.confirm)
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_RESULTS.mkdir(parents=True, exist_ok=True)
    DEGRADATION_RESULTS.mkdir(parents=True, exist_ok=True)
    ABLATION_RESULTS.mkdir(parents=True, exist_ok=True)

    print("[master] scanning existing artifacts")
    q = build_queue()
    total = q.qsize()
    print(f"[master] {total} pending work units on GPUs {args.gpus}")
    if args.dry_run:
        while not q.empty():
            print(f"  {q.get().name()}")
        return

    done_counter: dict = {}
    done_lock = threading.Lock()
    workers = [
        GPUWorker(g, q, done_counter, done_lock) for g in args.gpus
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    print("[master] all workers finished")
    print(f"[master] summary: {dict(done_counter)}")


if __name__ == "__main__":
    main()
