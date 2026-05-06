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
import ctypes
import json
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# Linux-only: tell the kernel to send SIGTERM to a child the moment its
# parent dies (any reason: SIGKILL, segfault, hangup, anything). With
# this, killing the master cascades down to every per-method HPO master
# and every per-trial worker without us having to track PIDs. Replaces
# the start_new_session=True orphan trail.
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_PR_SET_PDEATHSIG = 1


def _set_pdeathsig() -> None:
    """preexec_fn helper: child dies (SIGTERM) when its parent dies."""
    _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)


# ---------------------------------------------------------------------------
# Experimental plan
# ---------------------------------------------------------------------------

ENVS: List[str] = [
    # Priority order: P3D first (headline), then locomotion, then control envs.
    # Reactor removed for now: post-env-fix, full 8640-step episodes ×
    # CONTROL_PERIOD=10 × rk4_50 make eval dominate even with sweep=True +
    # 12k log_freq (~12 st/s observed). Re-add once we have a real eval-cap fix.
    # GlassFurnace gets in (same long-ish episodes but rk4_1 + no
    # control_period, ~750x cheaper physics per env-step than Reactor).
    # Plane3DFigureEight is parked last as a generalization add-on — same
    # plane env as P3D Circle, weaker expert (~3% of optimum) makes for a
    # sharp edge-qa showcase but isn't headline-critical.
    "Plane3DCircle",
    "CheetahRun",
    "FourTank",
    "GlassFurnace",
    "BarkourJoystick",
    "Plane3DFigureEight",
    # BarkourJoystick: CPG returns ~3.5 standalone (joystick-blind forward trot
    # mismatches most random commands), BUT the gait-structured rollouts it
    # produces are far more useful to a learning agent than random actions —
    # the kickstart value (upright trajectories, diagonal-pair anti-phase
    # coordination) is what matters for JSRL/IBRL/edge-qa, not the expert's
    # standalone task reward.
]

# METHODS list. Order matters: it drives the per-env work-unit ordering
# (depth-first per env, methods in this list order). Headline methods
# (sac, sac_quality_aware) run first per env so we get the flagship
# comparison early; expert-RL baselines next; edge-eps last because
# it's only used for the §5.3 gating-mechanism ablation.
METHODS: List[str] = [
    "sac",                   # vanilla SAC baseline
    "sac_quality_aware",     # proposed: symmetric LCB gating (edge-qa)
    # NOTE: "edge_qa_optimistic" deprioritized 2026-05-04 to free GPU for
    # GlassFurnace HPO. Method definition is preserved in
    # sac_hyperparam_search.py and the env restriction below; to re-enable,
    # uncomment the entry and delete the deprioritized placeholders at
    # hp_results/fourtank/edge_qa_optimistic/trial_999_phase3_result.json
    # and degradation_results/fourtank__edge_qa_optimistic__sigma{0.25,0.5,0.75}.json.
    # "edge_qa_optimistic",  # FourTank-only variant of edge-qa: signed β + wider τ
    "ibrl",                  # baseline: SAC + argmax bootstrap on expert
    "jsrl_curriculum",       # baseline: SAC + per-episode JSRL curriculum (Uchendu 2023)
    "jsrl",                  # baseline: SAC + JSRL training-time handoff (simpler variant)
    "residual",              # baseline: SAC + residual on expert
    # NOTE: "edge" (ε-greedy mixing) is an *ablation* of edge-qa's
    # gating mechanism, surfaced via the "gating_argmax" + (future)
    # gating-eps ablation variants. It is NOT run as a standalone HPO
    # method to keep the comparison surface focused on actually
    # competing baselines. Removing it cuts ~62h serial from the queue.
]

# Display labels for the status table.
_METHOD_DISPLAY: dict = {
    "edge": "edge-eps",
    "sac_quality_aware": "edge-qa",
    "edge_qa_optimistic": "edge-qa-opt",
    "jsrl_curriculum": "jsrl-cur",
    "jsrl": "jsrl-tt",
}

# Two-phase HPO. Phase 1 sweeps all trials at a small budget to surface
# the promising configurations cheaply; Phase 2 re-runs the top-K of
# them at the full budget for a robust headline number.
HPO_PHASE1_TIMESTEPS = 300_000
HPO_PHASE1_SEEDS = 20  # cheap ranker for TPE — coverage is in phase2
HPO_PHASE2_TIMESTEPS = 1_000_000  # informational; sac_hyperparam_search owns the constant
HPO_PHASE2_SEEDS = 50              # informational; same
HPO_TRIALS = 30
HPO_TOP_K = 5  # Phase 2 confirms top-5 from Phase 1
# Legacy aliases used by the wall-time estimator.
HPO_TIMESTEPS = HPO_PHASE1_TIMESTEPS
HPO_SEEDS = HPO_PHASE1_SEEDS

# No separate final-eval phase: the HPO already runs at 1M steps × HPO_SEEDS
# seeds with pruning, so the best trial *is* the headline number. Degradation
# evals reuse the HPO best-trial budget (1M, HPO_SEEDS) for consistency.

def final_timesteps_for(env: str) -> int:
    """Eval timesteps shared by degradation evals (1M, matches HPO budget)."""
    return HPO_TIMESTEPS

DEGRADATION_SIGMAS: List[float] = [0.25, 0.50, 0.75]
DEGRADATION_SEEDS = 50

# Tuned-expert reference scores per env (20-seed mean from the benchmark-
# table eval). Shown in the status table so each method's numbers land
# against the "beat the expert" goalpost at a glance.
EXPERT_REFERENCE: dict = {
    "Plane3DCircle": 4765.8,  # actual eval-set expert (was 3784 before retune)
    "Plane3DFigureEight": 314.93,  # 20-seed mean (CV 0.94 — expert often fails)
    "Reactor": 21479.0,
    "GlassFurnace": 4315.31,  # 20-seed mean (CV 0.43)
    "FourTank": 246.0,
    "CheetahRun": 291.2,  # 16-seed mean of retuned CPG (cpg_tuned_cheetahrun.json)
    "BarkourJoystick": 3.5,  # joystick-blind forward-trot CPG; standalone reward
                              # is low because trot mismatches random joystick
                              # commands, but the gait-structured rollouts kickstart
                              # learning. Per-seed bias from TB is the actual signal.
}

# EDGE-QA design ablations (§5.3). Variants from run_ablation_eval.ABLATION_VARIANTS;
# duplicated here so we don't import jax just to build the queue. Each variant
# flips a single design knob of the tuned sac_quality_aware config.
ABLATION_VARIANTS: List[str] = [
    # Gating mechanism
    "gating_argmax",
    # Preprocessing
    "no_state_aug",
    "no_obs_norm",
    # Critic target / bootstrap (edge-qa + IBRL-style bootstrap)
    "bootstrap_argmax",
    "bootstrap_lcb_gated",
    # Expert quality
    "random_expert",
    # Off-policy correctness
    "store_policy_action",
    # Orthogonal augmentation: tt-prefill on top of edge-qa
    # Compositionality: edge-qa + JSRL-curriculum
    "edge_qa_with_curriculum",
    # Pessimism axis discrete: β = 0 with softmax gate retained
    "no_pessimism",
    # Ensemble probe (lowest priority: K=10 is ~2.5x critic cost)
    "large_ensemble",
]
# Envs the ablation study runs on, in order of priority (P3DCircle first).
# Switched away from FourTank: per the SAC vs IBRL one-sided permutation test
# (n=50, p=0.034, 4-point IQM gap, Cohen's d=0.43), FourTank's per-seed std
# (~10) makes it borderline even for fundamentally-different methods. Ablation
# variants of EDGE produce smaller effects than method swaps and would mostly
# return "not significantly different" — uninformative. P3DCircle's
# thousands-of-points spread between methods gives clean SNR for design-choice
# ablations.
ABLATION_ENVS: List[str] = ["Plane3DCircle", "CheetahRun"]
ABLATION_SEEDS = 50

GPUS: List[int] = [0, 1, 2]
POLL_INTERVAL_S = 30
GPU_CHECK_RETRY_S = 60

ROOT = Path(__file__).resolve().parent
HP_RESULTS = ROOT / "hp_results"
DEGRADATION_RESULTS = ROOT / "degradation_results"
# v2 deployment-side perturbations (action_bias, obs_noise). Source of truth
# for these intensity sweeps lives in run_degradation_study_v2.py:INTENSITIES;
# duplicated here for the --status display only.
DEGRADATION_V2_RESULTS = ROOT / "degradation_v2_results"
V2_DEGRADATION_INTENSITIES = {
    "action_bias": [0.10, 0.25, 0.50],
    "obs_noise":   [0.05, 0.15, 0.30],
}
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
    kind: str = field(compare=False)       # hpo | degradation | ablation
    env: str = field(compare=False)
    method: str = field(compare=False)
    sigma: Optional[float] = field(default=None, compare=False)
    variant: Optional[str] = field(default=None, compare=False)
    # How many times this unit has been requeued after a non-zero
    # dispatch exit (master killed for an online fix, hardware glitch,
    # OOM, etc.). Capped by GPUWorker so a permanently-broken config
    # does not spin forever; reset implicitly on master restart.
    retries: int = field(default=0, compare=False)

    def name(self) -> str:
        tag = f"{self.env}.{self.method}"
        if self.kind == "hpo":
            return f"HPO[{tag}]"
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


# IQM cache: keyed by (tb_dir, window_steps, latest_mtime) so repeated
# --status calls don't reparse TB events when nothing changed.
_IQM_CACHE: dict = {}


def _read_seed_curves(tb_dir: Path,
                      tag: str = "Eval/episodic_mean_reward") -> dict:
    """Return {step: [v_seed1, v_seed2, ...]} for the requested scalar tag.

    Skips subfolders whose event file is a tiny stub (<10 KB) — these
    are leftovers from worker processes that crashed at JAX init and
    never wrote real training data. Without this filter, the IQM
    function gets confused when a trial has been retried multiple
    times in the same TB folder.
    """
    base = tb_dir / "tensorboard"
    if not base.is_dir():
        return {}
    out: dict = {}
    try:
        from tensorboardX.proto import event_pb2
    except Exception:
        return {}
    for sd in base.iterdir():
        if not sd.is_dir():
            continue
        sized_files = []
        for f in sd.glob("events.out.tfevents.*"):
            try:
                if f.stat().st_size > 10_000:
                    sized_files.append(f)
            except OSError:
                continue
        if not sized_files:
            continue
        for f in sized_files:
            try:
                with open(f, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            i, N = 0, len(data)
            while i + 12 <= N:
                ln = struct.unpack("<Q", data[i:i + 8])[0]
                i += 12
                if i + ln + 4 > N:
                    break
                try:
                    ev = event_pb2.Event.FromString(data[i:i + ln])
                    for v in ev.summary.value:
                        if v.tag == tag:
                            out.setdefault(ev.step, []).append(float(v.simple_value))
                except Exception:
                    pass
                i += ln + 4
    return out


def _iqm_window_by_run_ids(tb_dir: Path, window_steps: int,
                           run_ids: List[str],
                           tag: str = "Eval/episodic_mean_reward",
                           ) -> Optional[float]:
    """Compute IQM over a specific set of seed-run subfolders, given by
    name. The phase2/phase3 result JSONs each store the ``run_ids``
    that backed their reported metric, so this is the ground-truth
    way to recover the right subset of seeds even when the trial
    folder has accumulated leftovers from killed-master retries or
    when phase3's seeds are mixed in with phase2's.
    """
    if not tb_dir.is_dir():
        return None
    base = tb_dir / "tensorboard"
    if not base.is_dir():
        return None
    wanted = set(run_ids)
    sub_dirs = [p for p in base.iterdir() if p.is_dir() and p.name in wanted]
    if not sub_dirs:
        return None
    out: dict = {}
    try:
        from tensorboardX.proto import event_pb2
    except Exception:
        return None
    for sd in sub_dirs:
        for f in sd.glob("events.out.tfevents.*"):
            try:
                with open(f, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            i, N = 0, len(data)
            while i + 12 <= N:
                ln = struct.unpack("<Q", data[i:i + 8])[0]
                i += 12
                if i + ln + 4 > N:
                    break
                try:
                    ev = event_pb2.Event.FromString(data[i:i + ln])
                    for v in ev.summary.value:
                        if v.tag == tag:
                            out.setdefault(ev.step, []).append(float(v.simple_value))
                except Exception:
                    pass
                i += ln + 4
    if not out:
        return None
    counts = [len(v) for v in out.values()]
    modal = max(set(counts), key=counts.count)
    steps = sorted(s for s, v in out.items() if len(v) == modal)
    if not steps:
        return None
    import numpy as _np
    arr = _np.array([out[s] for s in steps], dtype=_np.float32)
    last_step = steps[-1]
    mask = _np.array([s >= last_step - window_steps for s in steps])
    window = arr[mask]
    if window.size == 0:
        return None
    per_seed = window.mean(axis=0)
    sorted_seeds = _np.sort(per_seed)
    n = len(sorted_seeds)
    lo, hi = n // 4, (3 * n) // 4
    return float(sorted_seeds[lo:hi].mean()) if hi > lo else float(sorted_seeds.mean())


def _iqm_window_first_n(tb_dir: Path, window_steps: int,
                        max_seeds: Optional[int] = None,
                        prefer: str = "newest",
                        tag: str = "Eval/episodic_mean_reward",
                        ) -> Optional[float]:
    """Variant of ``_iqm_window`` that selects ``max_seeds`` seed subfolders.

    ``prefer="oldest"``: take the chronologically earliest subfolders.
    Use this for the phase2 50-seed snapshot when phase3 has merged in
    additional seeds (phase3 events are newer than phase2 events).

    ``prefer="newest"``: take the chronologically latest subfolders.
    Use this when phase3 hasn't run but the trial folder accumulated
    leftovers from earlier killed master attempts (the most recent run
    is the canonical one).
    """
    if not tb_dir.is_dir():
        return None
    base = tb_dir / "tensorboard"
    if not base.is_dir():
        return None
    sub_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not sub_dirs:
        return None
    # Order by max event-file mtime per subfolder, AND filter out empty
    # stubs (subfolders from crashed-at-init workers that left a file
    # but no eval data). Looking at file size alone is enough — a folder
    # with no eval events has events files of just a few hundred bytes.
    def _max_event_mtime(sd: Path) -> float:
        m = 0.0
        for f in sd.glob("events.out.tfevents.*"):
            try:
                m = max(m, f.stat().st_mtime)
            except OSError:
                continue
        return m

    def _has_real_data(sd: Path) -> bool:
        # A 1M-step run writes hundreds of KB; an init-only crash writes
        # < 5 KB (just the file-version event header).
        for f in sd.glob("events.out.tfevents.*"):
            try:
                if f.stat().st_size > 10_000:
                    return True
            except OSError:
                continue
        return False

    sub_dirs = [sd for sd in sub_dirs if _has_real_data(sd)]
    if not sub_dirs:
        return None
    sub_dirs.sort(key=_max_event_mtime)
    if max_seeds is not None and len(sub_dirs) > max_seeds:
        if prefer == "oldest":
            sub_dirs = sub_dirs[:max_seeds]
        else:  # "newest"
            sub_dirs = sub_dirs[-max_seeds:]
    # Read those subfolders only. Inline a small variant of _read_seed_curves.
    out: dict = {}
    try:
        from tensorboardX.proto import event_pb2
    except Exception:
        return None
    for sd in sub_dirs:
        for f in sd.glob("events.out.tfevents.*"):
            try:
                with open(f, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            i, N = 0, len(data)
            while i + 12 <= N:
                ln = struct.unpack("<Q", data[i:i + 8])[0]
                i += 12
                if i + ln + 4 > N:
                    break
                try:
                    ev = event_pb2.Event.FromString(data[i:i + ln])
                    for v in ev.summary.value:
                        if v.tag == tag:
                            out.setdefault(ev.step, []).append(float(v.simple_value))
                except Exception:
                    pass
                i += ln + 4
    if not out:
        return None
    counts = [len(v) for v in out.values()]
    modal = max(set(counts), key=counts.count)
    steps = sorted(s for s, v in out.items() if len(v) == modal)
    if not steps:
        return None
    import numpy as _np
    arr = _np.array([out[s] for s in steps], dtype=_np.float32)
    last_step = steps[-1]
    mask = _np.array([s >= last_step - window_steps for s in steps])
    window = arr[mask]
    if window.size == 0:
        return None
    per_seed = window.mean(axis=0)
    sorted_seeds = _np.sort(per_seed)
    n = len(sorted_seeds)
    lo, hi = n // 4, (3 * n) // 4
    return float(sorted_seeds[lo:hi].mean()) if hi > lo else float(sorted_seeds.mean())


def _iqm_window(tb_dir: Path, window_steps: int,
                tag: str = "Eval/episodic_mean_reward") -> Optional[float]:
    """Per-seed mean over trailing ``window_steps`` env-steps, then IQM
    across seeds (drop top + bottom 25%, mean of middle 50%). Returns
    None on missing/empty TB events."""
    if not tb_dir.is_dir():
        return None
    # Cache key: dir + window + max mtime across event files (cheap to stat).
    latest_mtime = 0.0
    try:
        for sd in (tb_dir / "tensorboard").iterdir():
            if not sd.is_dir():
                continue
            for f in sd.glob("events.out.tfevents.*"):
                latest_mtime = max(latest_mtime, f.stat().st_mtime)
    except FileNotFoundError:
        return None
    cache_key = (str(tb_dir), int(window_steps), tag, latest_mtime)
    if cache_key in _IQM_CACHE:
        return _IQM_CACHE[cache_key]
    curves = _read_seed_curves(tb_dir, tag=tag)
    if not curves:
        _IQM_CACHE[cache_key] = None
        return None
    # Per-seed: average the seed's eval values in [max_step - window, max_step].
    # Seeds may report different counts per step; group by run_id by index.
    counts = [len(v) for v in curves.values()]
    if not counts:
        _IQM_CACHE[cache_key] = None
        return None
    modal = max(set(counts), key=counts.count)
    steps = sorted(s for s, v in curves.items() if len(v) == modal)
    if not steps:
        _IQM_CACHE[cache_key] = None
        return None
    import numpy as _np
    arr = _np.array([curves[s] for s in steps], dtype=_np.float32)  # (n_steps, n_seeds)
    last_step = steps[-1]
    mask = _np.array([s >= last_step - window_steps for s in steps])
    window = arr[mask]
    if window.size == 0:
        _IQM_CACHE[cache_key] = None
        return None
    per_seed = window.mean(axis=0)
    sorted_seeds = _np.sort(per_seed)
    n = len(sorted_seeds)
    lo, hi = n // 4, (3 * n) // 4
    iqm = float(sorted_seeds[lo:hi].mean()) if hi > lo else float(sorted_seeds.mean())
    _IQM_CACHE[cache_key] = iqm
    return iqm


def _hpo_best_iqm_per_phase(env: str, method: str,
                            tag: str = "Eval/episodic_mean_reward",
                            ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """IQM-equivalent of ``_hpo_best_per_phase``. For each phase, find the
    best trial by stored metric (mean of per-seed window means), then
    re-read its TB events and compute IQM across seeds. This is what the
    paper actually reports.

    ``tag`` controls which TB scalar is reduced. Defaults to the trained
    return; pass ``"Eval/expert_bias"`` to compute per-seed bias IQM
    (rl_reward - expert_reward, computed seed-wise at log time)."""
    base = HP_RESULTS / env.lower() / method
    if not base.exists():
        return None, None, None

    p1 = p2 = p3 = None

    # Phase 1: best trial number from optuna.db (with 60k sidecar override).
    db_path = base / "optuna.db"
    best_p1_num = None
    if db_path.exists():
        try:
            import optuna
            from optuna.trial import TrialState
            s = optuna.load_study(
                study_name=_study_name(env, method),
                storage=f"sqlite:///{db_path}",
            )
            done = [t for t in s.trials if t.state == TrialState.COMPLETE]
            if done:
                import json as _json
                sidecar_path = base / "recomputed_p1_60k.json"
                sidecar = {}
                if sidecar_path.exists():
                    try:
                        sidecar = _json.loads(sidecar_path.read_text())
                    except Exception:
                        sidecar = {}
                best_t = max(
                    done, key=lambda t: sidecar.get(str(t.number), t.value)
                )
                best_p1_num = best_t.number
        except Exception:
            pass
    if best_p1_num is not None:
        p1 = _iqm_window(base / f"trial_{best_p1_num:03d}_tb", 60_000, tag=tag)

    # Phase 2/3: pick the best result JSON by metric, then read IQM
    # over the EXACT run_ids the JSON itself recorded. This is robust
    # to (a) phase3 having spawned its 50 seeds into the same TB
    # folder as phase2 and (b) leftover empty subfolders from killed
    # master retries. No mtime / file-size heuristics needed.
    import json as _json
    def _best_json(pattern: str):
        """Return (trial_number, json_path) of the result with highest metric."""
        best_n, best_v, best_path = None, None, None
        for f in base.glob(pattern):
            try:
                r = _json.loads(f.read_text())
                m = r.get("metric")
                if m is None:
                    continue
                m = float(m)
                if best_v is None or m > best_v:
                    best_v = m
                    best_n = int(f.name.split("_")[1])
                    best_path = f
            except Exception:
                pass
        return best_n, best_path

    n2, p2_json_path = _best_json("trial_*_phase2_result.json")
    if n2 is not None and p2_json_path is not None:
        try:
            run_ids = list(_json.loads(p2_json_path.read_text())
                                .get("run_ids", []))
        except Exception:
            run_ids = []
        if run_ids:
            p2 = _iqm_window_by_run_ids(
                base / f"trial_{n2:03d}_phase2_tb", 200_000, run_ids, tag=tag
            )

    n3, p3_json_path = _best_json("trial_*_phase3_result.json")
    if n3 is not None and p3_json_path is not None:
        # Phase3 reports the COMBINED 100-seed metric; build the union
        # of phase2 + phase3 run_ids to match.
        try:
            r3 = _json.loads(p3_json_path.read_text())
            run_ids_3 = list(r3.get("run_ids", []))
        except Exception:
            run_ids_3 = []
        run_ids_combined = list(run_ids_3)
        p2_companion = base / f"trial_{n3:03d}_phase2_result.json"
        if p2_companion.exists():
            try:
                run_ids_combined += list(
                    _json.loads(p2_companion.read_text()).get("run_ids", [])
                )
            except Exception:
                pass
        if run_ids_combined:
            p3 = _iqm_window_by_run_ids(
                base / f"trial_{n3:03d}_phase2_tb", 200_000,
                run_ids_combined, tag=tag,
            )
    return p1, p2, p3


def _hpo_best_per_phase(env: str, method: str,
                        ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return ``(p1_best, p2_best, p3_best)`` for one ``(env, method)``.

    Each entry is the maximum score seen at that phase, or None.
    Phase1 reads Optuna's ``t.value`` across COMPLETE trials.
    Phase2/3 read ``trial_*_phaseN_result.json`` ``metric`` fields.
    """
    base = HP_RESULTS / env.lower() / method
    if not base.exists():
        legacy = HP_RESULTS / env.lower() / "optuna.db"
        if not legacy.exists():
            return None, None, None
        base = HP_RESULTS / env.lower()

    p1 = p2 = p3 = None

    db_path = base / "optuna.db"
    if db_path.exists():
        try:
            import optuna
            from optuna.trial import TrialState
            s = optuna.load_study(
                study_name=_study_name(env, method),
                storage=f"sqlite:///{db_path}",
            )
            done = [t for t in s.trials if t.state == TrialState.COMPLETE]
            if done:
                # Prefer the recomputed 60k-window score (sidecar JSON
                # written by recompute_phase1_window.py) when present, so
                # old and new trials are scored on the same window.
                import json as _json
                sidecar_path = base / "recomputed_p1_60k.json"
                sidecar = {}
                if sidecar_path.exists():
                    try:
                        sidecar = _json.loads(sidecar_path.read_text())
                    except Exception:
                        sidecar = {}
                vals = [
                    sidecar.get(str(t.number), t.value) for t in done
                ]
                p1 = max(vals)
        except Exception:
            pass

    import json as _json

    def _max_metric(pattern: str) -> Optional[float]:
        best = None
        for f in base.glob(pattern):
            try:
                r = _json.loads(f.read_text())
                m = r.get("metric")
                if m is None:
                    continue
                m = float(m)
                if best is None or m > best:
                    best = m
            except Exception:
                pass
        return best

    p2 = _max_metric("trial_*_phase2_result.json")
    p3 = _max_metric("trial_*_phase3_result.json")
    return p1, p2, p3


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


def _hpo_phase_status(env: str, method: str) -> str:
    """Return ``'phase1'`` / ``'phase2'`` / ``'phase3'`` / ``'done'``.

    The HPO unit is considered ``'done'`` only when a phase3 result JSON
    exists. If phase1 has the full trial count but phase2 has fewer than
    ``HPO_TOP_K`` result JSONs, status is ``'phase2'`` (re-dispatch the
    same HPO master; phase1 will skip and phase2 will run). Likewise
    ``'phase3'`` when phase2 is complete but no phase3 result yet. This
    is what lets a master that died mid-pipeline resume cleanly on
    restart.
    """
    base = HP_RESULTS / env.lower() / method
    if (base.exists()
            and any(base.glob("trial_*_phase3_result.json"))):
        return "done"
    n_p2 = (
        len(list(base.glob("trial_*_phase2_result.json")))
        if base.exists() else 0
    )
    if _hpo_db_complete(env, method):
        if n_p2 >= HPO_TOP_K:
            return "phase3"
        return "phase2"
    return "phase1"


def _study_name(env: str, method: str) -> str:
    return f"{method}_hp_search_{env.lower()}"


def _degradation_done(env: str, method: str, sigma: float) -> bool:
    p = DEGRADATION_RESULTS / f"{env.lower()}__{method}__sigma{sigma}.json"
    return p.exists()


def _degradation_v2_done(env: str, method: str, deg_type: str,
                         intensity: float) -> bool:
    """True iff the v2 cell's output JSON exists locally.

    Path matches run_degradation_v2.py's writer: filenames use the float
    repr of intensity (so 0.10 → '0.1', 0.50 → '0.5', 0.05 → '0.05').
    """
    tag = f"{float(intensity)}"
    p = DEGRADATION_V2_RESULTS / (
        f"{env.lower()}__{method}__{deg_type}__{tag}.json"
    )
    return p.exists()


def _ablation_done(env: str, variant: str) -> bool:
    p = ABLATION_RESULTS / f"{env.lower()}__{variant}.json"
    return p.exists()


# ---------------------------------------------------------------------------
# Build the work queue
# ---------------------------------------------------------------------------


_PRIMARY_TIER = 0
_EDGE_TIER = 1
_ABLATION_TIER = 2


def _unit_priority(kind: str, env: str, method: str,
                   sigma: Optional[float] = None,
                   variant: Optional[str] = None) -> int:
    """Three-tier ordering:

      tier 0 (primary):    sac, sac_quality_aware, ibrl, jsrl, residual.
                           Depth-first per env: complete BOTH the HPO and the
                           degradation evaluation of every primary method on
                           env N before any work starts on env N+1. Within env,
                           HPO of all primary methods runs before any
                           degradation.
      tier 1 (edge-eps):   the LCB-vs-ε-greedy comparison, deferred to a tail
                           tier that runs after the entire primary tier
                           completes. Same depth-first ordering inside.
      tier 2 (ablations):  sac_quality_aware design ablations on FourTank
                           then CheetahRun.

    Encoding: tier * 10_000_000 + env_rank * 100_000 + phase * 10_000 +
              method_rank * 10 + sigma_rank.
    """
    if kind == "ablation":
        # User directive: ablations on P3DC + CheetahRun must complete BEFORE
        # the BarkourJoystick + Plane3DFigureEight HPO sweep starts.
        # Encoding: slot ablations between GF (env_rank=3) and BarkourJoystick
        # (env_rank=4) in the primary tier, so they take priority over those
        # two envs but defer to in-flight P3DC/Cheetah/FT/GF HPO+degradation.
        env_rank_in_abl_set = ABLATION_ENVS.index(env)  # 0 P3DC, 1 Cheetah
        var_rank = ABLATION_VARIANTS.index(variant) if variant else 0
        return (
            _PRIMARY_TIER * 10_000_000
            + 350_000  # between GF (300_000) and BarkourJoystick (400_000)
            + env_rank_in_abl_set * 1_000
            + var_rank
        )
    env_rank = ENVS.index(env)
    method_rank = METHODS.index(method)
    sigma_rank = (
        DEGRADATION_SIGMAS.index(sigma) if sigma is not None else 0
    )
    tier = _EDGE_TIER if method == "edge" else _PRIMARY_TIER
    # User directive: v1 mistune degradation for sac_quality_aware + residual
    # runs BEFORE ablations; mistune for the other expert methods runs AFTER
    # ablations. Encoded by splitting the degradation phase offset:
    #   - high-pri methods: phase_offset 10_000 (lands at env_rank*100k + 10k)
    #     => GF lands at 310_000-310_059, before ablations at 350_000+
    #   - low-pri methods:  phase_offset 60_000 (lands at env_rank*100k + 60k)
    #     => GF lands at 360_000+, after ablations at 350_000+
    # HPO stays at phase_offset 0 (always first within env).
    if kind == "hpo":
        phase_offset = 0
    else:
        # kind == "degradation"
        if method in ("sac_quality_aware", "residual"):
            phase_offset = 10_000
        else:
            phase_offset = 60_000
    return (
        tier * 10_000_000
        + env_rank * 100_000
        + phase_offset
        + method_rank * 10
        + sigma_rank
    )


def build_unit_list() -> list[WorkUnit]:
    """Return the full study-wide unit list, sorted by priority.

    The list is *static* across the master's lifetime: GPU workers
    rescan it on every iteration, skipping units already done on disk
    (terminal artifact present) or already running elsewhere. This
    means killing a unit mid-flight does NOT lose it: the next free
    GPU re-picks it from the same list.
    """
    units: list[WorkUnit] = []
    # Per-method env restriction. Methods restricted here only get
    # built for the listed envs (skipped on others). Used for variants
    # we only need on a single env for a targeted test.
    _METHOD_ENV_RESTRICT = {
        "edge_qa_optimistic": ("FourTank",),  # see METHODS comment
    }
    def _method_active_for(method: str, env: str) -> bool:
        envs = _METHOD_ENV_RESTRICT.get(method)
        return envs is None or env in envs

    # HPOs.
    for env in ENVS:
        for method in METHODS:
            if not _method_active_for(method, env):
                continue
            units.append(WorkUnit(
                priority=_unit_priority("hpo", env, method),
                kind="hpo", env=env, method=method,
            ))
    # Degradation: every expert-using method. Reuses HPO-best params.
    degrade_methods = [
        m for m in METHODS
        if m in ("edge", "sac_quality_aware", "edge_qa_optimistic",
                 "ibrl", "jsrl", "jsrl_curriculum", "residual")
    ]
    for env in ENVS:
        for method in degrade_methods:
            if not _method_active_for(method, env):
                continue
            for sigma in DEGRADATION_SIGMAS:
                units.append(WorkUnit(
                    priority=_unit_priority("degradation", env, method, sigma),
                    kind="degradation", env=env, method=method, sigma=sigma,
                ))
    # Decision-test ablation: bootstrap_lcb_gated on Plane3DCircle.
    units.append(WorkUnit(
        priority=90_000,
        kind="ablation",
        env="Plane3DCircle",
        method="edge",
        variant="bootstrap_lcb_gated",
    ))
    # EDGE design ablations.
    for env in ABLATION_ENVS:
        for variant in ABLATION_VARIANTS:
            units.append(WorkUnit(
                priority=_unit_priority("ablation", env, "edge",
                                        variant=variant),
                kind="ablation", env=env, method="edge", variant=variant,
            ))
    units.sort()
    return units


def build_queue() -> "queue.PriorityQueue[WorkUnit]":
    """Legacy adapter: keeps callers that expect a queue working.

    The new GPUWorker pulls from a shared unit list directly; this
    function is retained only so older entrypoints / tests / status
    helpers keep importing.
    """
    q: "queue.PriorityQueue[WorkUnit]" = queue.PriorityQueue()
    for u in build_unit_list():
        q.put(u)
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
        # Master uses --search-* CLI args; the worker subprocess it
        # spawns picks up its budget from those (via the master).
        # NOTE: --n-timesteps / --n-seeds are the *worker* CLI knobs and
        # don't influence what the master tells the worker — using them
        # here would silently fall back to SEARCH_TIMESTEPS defaults.
        # Three chained phases:
        #   1: 20 seeds × 300k × 30 trials (cheap TPE ranker, with pruning)
        #   2: 50 seeds × 1M × top-K (HPO confirmation)
        #   3: top-1 with PHASE2_SEEDS extra seeds (final eval, 100-seed total)
        cmd = [
            python, str(ROOT / "sac_hyperparam_search.py"),
            "--env", unit.env,
            "--method", unit.method,
            "--gpus", str(gpu_id),
            "--trials", str(HPO_TRIALS),
            "--search-timesteps", str(HPO_PHASE1_TIMESTEPS),
            "--search-seeds", str(HPO_PHASE1_SEEDS),
            "--auto-phase2",
            "--auto-phase3",
            "--top-k", str(HPO_TOP_K),
        ]
        # Locomotion envs converge slowly: the gait isn't stable until
        # 200k+ steps, so the 100k-warmup pruner kills trials that would
        # have been good. Disable pruning for these envs.
        if unit.env in ("CheetahRun", "BarkourJoystick"):
            cmd.append("--no-prune")
        return cmd
    if unit.kind == "degradation":
        return [
            python, str(ROOT / "run_degradation_eval.py"),
            "--env", unit.env,
            "--method", unit.method,
            "--gpu", str(gpu_id),
            "--sigma", str(unit.sigma),
            "--n-timesteps", str(final_timesteps_for(unit.env)),
            "--n-seeds", str(DEGRADATION_SEEDS),
        ]
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


def _hpo_truly_done(env: str, method: str) -> bool:
    """True iff (env, method) has a *real* phase3 result on disk.

    A warm-start dispatch writes a placeholder phase3 result with
    ``warm_start_status: in_progress`` and ``metric: NaN`` at the start
    of training, so ``_hpo_phase_status`` reports "done" even while the
    training is still in flight. That's what we want for the master's
    "skip re-dispatch" check (race protection), but degradation /
    ablation must NOT fire against an in-progress placeholder, so they
    use this stricter check.
    """
    base = HP_RESULTS / env.lower() / method
    if not base.exists():
        return False
    import json as _json
    import math as _math
    for r in base.glob("trial_*_phase3_result.json"):
        try:
            d = _json.loads(r.read_text())
        except Exception:
            continue
        status = d.get("warm_start_status")
        if status == "in_progress":
            continue
        metric = d.get("metric")
        if metric is None or (isinstance(metric, float) and _math.isnan(metric)):
            continue
        return True
    return False


def _unit_eligible(unit: "WorkUnit") -> bool:
    """True if this unit can actually be dispatched right now.

    Degradation/ablation depend on their target method having a fully
    complete HPO unit (real phase3 result, NOT a warm-start placeholder
    whose final metric is still NaN). Gate at dispatch time so the
    unit gets requeued instead of consumed.
    """
    if unit.kind == "degradation":
        return _hpo_truly_done(unit.env, unit.method)
    if unit.kind == "ablation":
        # Ablations clone edge-qa best params on the same env.
        return _hpo_truly_done(unit.env, "sac_quality_aware")
    return True


def _unit_complete_on_disk(unit: "WorkUnit") -> bool:
    """True iff this work unit's terminal artifact is on disk.

    Used by the GPUWorker to decide whether a non-zero dispatch exit
    should requeue the unit (incomplete -> retry, e.g. master killed
    for an online code fix) or drop it (complete -> nothing more to do,
    e.g. a phase3 result already exists and the rerun was a no-op).
    """
    if unit.kind == "hpo":
        # Use _hpo_truly_done so a warm-start placeholder result
        # (metric=NaN, warm_start_status=in_progress) does NOT count as
        # complete. Race protection for the in-flight warm-start is
        # handled by _unit_already_running's ps scan, not by this
        # check, so requiring a real metric here is safe.
        return _hpo_truly_done(unit.env, unit.method)
    if unit.kind == "degradation":
        return _degradation_done(unit.env, unit.method, float(unit.sigma))
    if unit.kind == "ablation":
        out = ABLATION_RESULTS / f"{unit.env.lower()}__{unit.variant}.json"
        return out.exists()
    return True


def _unit_already_running(unit: "WorkUnit") -> bool:
    """True if ``ps`` shows a master for the same (kind, env, method[, sigma/variant])
    is already alive (orphan from an earlier session, or a parallel launch)."""
    units = _running_units()
    if unit.kind == "hpo":
        return ("hpo", unit.env, unit.method) in units
    if unit.kind == "degradation":
        return ("degradation", unit.env, unit.method,
                str(unit.sigma)) in units
    if unit.kind == "ablation":
        return ("ablation", unit.env, "edge", unit.variant) in units
    return False


def dispatch(unit: WorkUnit, gpu_id: int) -> int:
    """Spawn the subprocess, wait for it, return exit code.

    Children are spawned with ``preexec_fn=_set_pdeathsig`` so the
    Linux kernel SIGTERMs them automatically if the master dies. We
    keep them in the master's process group (no ``start_new_session``)
    so a Ctrl-C on the master also propagates immediately.
    """
    cmd = build_cmd(unit, gpu_id)
    is_hpo_master = (unit.kind == "hpo")
    log_path = _log_path(unit)
    env = _env_with_gpu(gpu_id)
    if is_hpo_master:
        # The HPO master never trains: it only runs Optuna's TPE sampler,
        # spawns worker subprocesses, and reads TB events for pruning.
        # Force it to CPU so it doesn't claim ~10 GB of GPU memory just
        # for JAX init. The worker subprocess (spawned via
        # sac_hyperparam_search._spawn_worker) explicitly pops
        # JAX_PLATFORMS, so workers still use the GPU normally.
        env["JAX_PLATFORMS"] = "cpu"
    with open(log_path, "w") as log:
        log.write(f"[master] launching {unit.name()} on GPU {gpu_id}\n")
        log.write(f"[master] cmd: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=_set_pdeathsig,
        )
        ret = proc.wait()
        log.write(f"[master] exited with {ret}\n")
    return ret


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------


class GPUWorker(threading.Thread):
    """One worker thread per GPU.

    Instead of consuming a one-shot queue, every iteration rescans the
    shared (static) unit list from the top in priority order, picking
    the first one that is:
      - not already done on disk (terminal artifact missing),
      - not already running elsewhere (ps scan),
      - not currently dispatched by a sibling GPU worker (in-memory
        ``in_progress`` set guarded by ``pick_lock``),
      - eligible (HPO prereqs met for degradation/ablation).

    This means killing a mid-flight unit (e.g. for an online code fix)
    is harmless: the next free GPU re-picks the same unit from the top
    of the list. No per-unit retry counter, no requeue plumbing — the
    list IS the source of truth, scanned fresh every time.
    """

    def __init__(self, gpu_id: int, units: list[WorkUnit],
                 in_progress: set, pick_lock: threading.Lock,
                 done_counter: dict, done_lock: threading.Lock,
                 stop_event: threading.Event):
        super().__init__(daemon=True, name=f"gpu-{gpu_id}")
        self.gpu_id = gpu_id
        self.units = units
        self.in_progress = in_progress
        self.pick_lock = pick_lock
        self.done_counter = done_counter
        self.done_lock = done_lock
        self.stop_event = stop_event

    def _pick_next(self) -> Optional[WorkUnit]:
        """Atomically scan the unit list and claim the first runnable unit.

        Returns None if no unit is currently runnable (everything is
        either done, running, or ineligible). Caller should sleep and
        retry. The claimed unit is added to ``in_progress`` under the
        pick_lock so sibling GPUs don't double-claim it.
        """
        with self.pick_lock:
            for unit in self.units:
                if unit.name() in self.in_progress:
                    continue
                if _unit_complete_on_disk(unit):
                    continue
                if _unit_already_running(unit):
                    continue
                if not _unit_eligible(unit):
                    continue
                self.in_progress.add(unit.name())
                return unit
        return None

    def run(self):
        IDLE_SLEEP_S = 30  # rescan cadence when nothing is runnable
        while not self.stop_event.is_set():
            unit = self._pick_next()
            if unit is None:
                # Nothing to dispatch right now. Either we are done
                # (loop terminates via _all_done check below), or the
                # remaining units are blocked (ineligible / running on
                # other GPUs). Sleep and rescan.
                if self._all_done():
                    return
                time.sleep(IDLE_SLEEP_S)
                continue
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
                # Nothing to do on FAIL: next iteration rescans, and if
                # the terminal artifact is still missing the same unit
                # gets re-picked automatically (by us or a sibling GPU).
            except Exception as e:
                print(f"[gpu{self.gpu_id}] {unit.name()} raised {e!r}")
            finally:
                with self.pick_lock:
                    self.in_progress.discard(unit.name())

    def _all_done(self) -> bool:
        """True when every unit in the list is complete on disk."""
        return all(_unit_complete_on_disk(u) for u in self.units)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _running_units() -> dict:
    """Return a dict mapping (kind, env, method[, sigma]) -> (pid, gpu, etime)
    for every work unit currently running, by scanning ``ps`` output.

    Kinds:
      - HPO master:    ``sac_hyperparam_search.py --env E --method M --gpus G ...``
      - degradation:   ``run_degradation_eval.py --env E --method M --gpu G --sigma S ...``
      - ablation:      ``run_ablation_eval.py --env E --variant V --gpu G ...``
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
_HPO_SEC_PER_TRIAL_DEFAULT = 15 * 60  # empirical median across observed methods
# Per-1M-step cost for a full (unpruned) run. Used to derive final/degradation/
# ablation wall-time estimates: for an env at 2M budget -> 2 * _SEC_PER_M; for
# CheetahRun at 1M budget -> 1 * _SEC_PER_M.
_SEC_PER_MILLION_TIMESTEPS = 85 * 60


def _final_sec_for(env: str) -> int:
    """Wall-time estimate for one degradation/ablation unit. Float math
    because HPO_TIMESTEPS is 300k (not a clean multiple of 1M); the old
    integer-division version returned 0 for any sub-1M budget and silently
    dropped DEG+ABL from the global ETA."""
    return int((HPO_TIMESTEPS / 1_000_000) * _SEC_PER_MILLION_TIMESTEPS)


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


def _hpo_steps_per_sec(env: str, method: str) -> Optional[float]:
    """Estimate env-steps/sec rate from the active trial's TB events,
    skipping the FIRST eval point (which lumps in JAX compile time).

    Reads ``Eval/episodic_mean_reward`` event timestamps from the most
    recent ``trial_*_tb`` directory under ``hp_results/<env>/<method>/``.
    Returns ``None`` when fewer than 2 post-compile eval points exist
    (caller falls back to the static per-trial default).
    """
    method_dir = HP_RESULTS / env.lower() / method
    if not method_dir.is_dir():
        return None
    # Pick the trial with the most recent TB activity.
    candidates = sorted(method_dir.glob("trial_*_tb"))
    if not candidates:
        return None
    best_pairs: list = []
    best_mtime = -1.0
    for tb in candidates:
        events_root = tb / "tensorboard"
        if not events_root.is_dir():
            continue
        # Find the latest event file across run subdirs.
        latest_mtime = -1.0
        files = []
        for sd in events_root.iterdir():
            if not sd.is_dir():
                continue
            for f in sd.glob("events.out.tfevents.*"):
                files.append(f)
                latest_mtime = max(latest_mtime, f.stat().st_mtime)
        if not files or latest_mtime <= best_mtime:
            continue
        # Parse Eval/episodic_mean_reward (step, wall_time) pairs.
        from tensorboardX.proto import event_pb2
        pairs: dict = {}
        for f in files:
            try:
                with open(f, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            i, N = 0, len(data)
            while i + 12 <= N:
                ln = struct.unpack("<Q", data[i:i + 8])[0]
                i += 12
                if i + ln + 4 > N:
                    break
                try:
                    ev = event_pb2.Event.FromString(data[i:i + ln])
                    for v in ev.summary.value:
                        if v.tag == "Eval/episodic_mean_reward":
                            pairs.setdefault(ev.step, ev.wall_time)
                except Exception:
                    pass
                i += ln + 4
        if len(pairs) >= 2:
            best_pairs = sorted(pairs.items())  # [(step, wall_time), ...]
            best_mtime = latest_mtime
    if len(best_pairs) < 2:
        return None
    # Drop the first point: it includes JAX compile + buffer warmup,
    # which inflates wall-time disproportionately.
    if len(best_pairs) >= 3:
        best_pairs = best_pairs[1:]
    span = best_pairs[-1][1] - best_pairs[0][1]
    if span <= 0:
        return None
    delta_steps = best_pairs[-1][0] - best_pairs[0][0]
    return float(delta_steps) / span


def _phase_completed_metrics(env: str, method: str,
                             ) -> dict:
    """Per-phase ``{wall_sec, steps_per_sec, n_trials}`` for finished trials.

    Returns ``{'phase1': {...}, 'phase2': {...}, 'phase3': {...}}``,
    populated only for phases that actually have completed trials on
    disk. Used by ``--status`` so the HPO column shows the cumulative
    wall-time and average per-seed throughput retrospectively.

    Throughput is *per-seed env-steps/sec* (so it matches the live
    "X st/s" number reported for in-flight trials), not aggregate.
    """
    base = HP_RESULTS / env.lower() / method
    out: dict = {}

    # Phase1: from Optuna db trial start/complete timestamps.
    db = base / "optuna.db"
    p1_durs: list[float] = []
    if db.exists():
        try:
            import optuna
            from optuna.trial import TrialState
            s = optuna.load_study(
                study_name=_study_name(env, method),
                storage=f"sqlite:///{db}",
            )
            for t in s.trials:
                if (t.state == TrialState.COMPLETE
                        and t.datetime_start and t.datetime_complete):
                    p1_durs.append(
                        (t.datetime_complete - t.datetime_start).total_seconds()
                    )
        except Exception:
            pass
    if p1_durs:
        total = float(sum(p1_durs))
        n = len(p1_durs)
        # Per-trial budget = phase1_timesteps; total per-seed steps =
        # n_trials × phase1_timesteps. Aggregate wall = total.
        rate = (n * HPO_PHASE1_TIMESTEPS) / total if total > 0 else 0.0
        out["phase1"] = {
            "wall_sec": total, "steps_per_sec": rate, "n_trials": n,
        }

    # Phase2/3: read each result JSON's ``elapsed_s`` (the worker
    # writes it at trial-end and is the authoritative wall duration;
    # filesystem ctime is unreliable as it changes with metadata
    # writes). Per-seed throughput = n_timesteps / elapsed_s.
    import json as _json

    def _accumulate(pattern: str) -> Optional[dict]:
        total_wall = 0.0
        total_steps = 0
        n = 0
        for r in base.glob(pattern):
            try:
                d = _json.loads(r.read_text())
                wall = float(d.get("elapsed_s") or 0.0)
                n_steps = int(d.get("n_timesteps") or HPO_PHASE2_TIMESTEPS)
                if wall <= 0:
                    continue
                total_wall += wall
                total_steps += n_steps
                n += 1
            except Exception:
                pass
        if n == 0 or total_wall <= 0:
            return None
        return {
            "wall_sec": total_wall,
            "steps_per_sec": total_steps / total_wall,
            "n_trials": n,
        }

    p2 = _accumulate("trial_*_phase2_result.json")
    if p2:
        out["phase2"] = p2
    p3 = _accumulate("trial_*_phase3_result.json")
    if p3:
        out["phase3"] = p3
    return out


def _empirical_phase_durations(env: str, method: str,
                               ) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(median_phase1_sec, median_phase2_sec)`` from Optuna trial
    timestamps in ``hp_results/<env>/<method>/optuna.db``, or (None, None)
    if data missing. Phase2 cost is read from
    ``trial_*_phase2_result.json``'s file mtime relative to its log start.
    """
    base = HP_RESULTS / env.lower() / method
    db_path = base / "optuna.db"
    p1: Optional[float] = None
    p2: Optional[float] = None
    if db_path.exists():
        try:
            import optuna
            s = optuna.load_study(
                study_name=_study_name(env, method),
                storage=f"sqlite:///{db_path}",
            )
            durs = [
                (t.datetime_complete - t.datetime_start).total_seconds()
                for t in s.trials
                if t.datetime_start and t.datetime_complete
            ]
            if len(durs) >= 5:
                # Trust empirical median once we have a few samples; below
                # that, fall back to live rate / static default to avoid
                # over-reacting to a single outlier trial.
                durs.sort()
                p1 = float(durs[len(durs) // 2])
        except Exception:
            pass
    # Phase2 duration: log mtime - log ctime per phase2 log file.
    p2_durs = []
    for log in base.glob("trial_*_phase2.log"):
        try:
            st = log.stat()
            if st.st_mtime > st.st_ctime:
                p2_durs.append(st.st_mtime - st.st_ctime)
        except OSError:
            pass
    if p2_durs:
        p2_durs.sort()
        p2 = float(p2_durs[len(p2_durs) // 2])
    return p1, p2


def _hpo_eta_seconds(n_done: int, etime_sec: int,
                     env: Optional[str] = None,
                     method: Optional[str] = None) -> int:
    """Estimate remaining HPO wall-time for one (env, method).

    Order of preference for per-trial cost:
      1. Median per-trial wall-time from Optuna's datetime_start /
         datetime_complete (already accounts for pruning; phase1).
      2. Live env-steps/sec rate from the running trial (skips compile),
         scaled to the phase1 budget. (Used only if no completed trials.)
      3. Static per-trial default (very conservative; rarely hit now).

    Phase2/3 cost: median measured phase2 wall-time from
    ``trial_*_phase2.log`` mtimes when available; else the linear
    (timesteps × seeds) scaling from phase1 (sub-linear in seeds via
    vmap so this over-estimates, kept as fallback).
    """
    per_trial_p1 = None
    per_trial_p2 = None
    if env and method:
        emp_p1, emp_p2 = _empirical_phase_durations(env, method)
        per_trial_p1 = emp_p1
        per_trial_p2 = emp_p2
    if per_trial_p1 is None:
        if n_done > 0:
            per_trial_p1 = etime_sec / n_done
        elif env and method:
            rate = _hpo_steps_per_sec(env, method)
            if rate and rate > 0:
                per_trial_p1 = HPO_PHASE1_TIMESTEPS / rate
            else:
                # Cross-env fallback: borrow this *method*'s median from
                # whatever env has run it. Methods are far more variable
                # than envs in our setup (sac_qa is 5x slower than sac
                # regardless of env), so this is much closer than the
                # static default.
                for other_env in ENVS:
                    if other_env == env:
                        continue
                    p1_other, _ = _empirical_phase_durations(other_env, method)
                    if p1_other is not None:
                        per_trial_p1 = p1_other
                        break
                if per_trial_p1 is None:
                    per_trial_p1 = _HPO_SEC_PER_TRIAL_DEFAULT
        else:
            per_trial_p1 = _HPO_SEC_PER_TRIAL_DEFAULT
    if per_trial_p2 is None:
        # Phase2 scale: (timesteps ratio) × sqrt(seeds ratio). Linear in
        # timesteps, sub-linear in seeds because vmap shares JAX compile
        # and most env-step cost across seeds. Empirically, ibrl on
        # Plane3DCircle had phase1=631s and phase2=3124s, ratio ≈ 4.95;
        # this formula gives 3.33 × sqrt(2.5) ≈ 5.27, close to measured.
        import math as _math
        phase2_scale = (
            (HPO_PHASE2_TIMESTEPS / HPO_PHASE1_TIMESTEPS)
            * _math.sqrt(HPO_PHASE2_SEEDS / HPO_PHASE1_SEEDS)
        )
        per_trial_p2 = per_trial_p1 * phase2_scale

    remaining_p1 = max(HPO_TRIALS - n_done, 0)
    phase1_eta = remaining_p1 * per_trial_p1
    phase2_eta = HPO_TOP_K * per_trial_p2
    phase3_eta = 1 * per_trial_p2  # top-1 rerun at phase2 cost.
    return int(phase1_eta + phase2_eta + phase3_eta)


def print_status() -> None:
    """Print a per-category / per-env progress table with live-run annotations."""
    running = _running_units()
    hpo_running = {
        (key[1], key[2]): v for key, v in running.items() if key[0] == "hpo"
    }
    deg_running = {
        (key[1], key[2], key[3]): v for key, v in running.items()
        if key[0] == "degradation"
    }
    abl_running_by_env = {}  # env -> count of running variants
    for key in running:
        if key[0] == "ablation":
            abl_running_by_env[key[1]] = abl_running_by_env.get(key[1], 0) + 1

    # DEG columns: v1 mistune (existing param-perturbation sweep) +
    # v2 action_bias + v2 obs_noise. Each shows 'done/total' and an optional
    # '+Nr' suffix for cells currently running.
    print(
        f"{'env':<20} {'method':<10} {'HPO':<44} "
        f"{'p1 bias (n/N)':>16} {'p2 bias (n/N)':>16} {'p3 bias':>10} "
        f"{'mistune':<8} {'a-bias':<8} {'obs':<8}"
    )
    print("-" * 145)
    n_hpo_done = n_deg_done = 0
    n_hpo_total = n_deg_total = 0
    total_remaining_sec = 0  # sum of remaining time across all pending/running units
    for env in ENVS:
        for i, method in enumerate(METHODS):
            # ``hpo_done`` here means the *unit* is complete (phase3
            # result on disk), NOT just that the optuna db has 30
            # phase1 trials. The previous check (_hpo_db_complete)
            # returned True as soon as phase1 hit HPO_TRIALS, which
            # made the status row collapse to the retrospective
            # ``p1=...`` line even while phase2 / phase3 were
            # actively training; ``_hpo_phase_status == 'done'`` only
            # flips once phase3 actually lands its result JSON.
            hpo_phase = _hpo_phase_status(env, method)
            hpo_done = (hpo_phase == "done")
            # Per-phase finished-run summary (cumulative wall-time).
            # Shown for both running and done units so finished phases
            # keep their retrospective cost on screen even after the
            # master moves on.
            phase_done = _phase_completed_metrics(env, method)

            def _phase_summary() -> str:
                parts = []
                for key in ("phase1", "phase2", "phase3"):
                    m = phase_done.get(key)
                    if not m:
                        continue
                    short = {"phase1": "p1", "phase2": "p2",
                             "phase3": "p3"}[key]
                    parts.append(f"{short}={_fmt_hms(int(m['wall_sec']))}")
                return " ".join(parts)

            if hpo_done:
                hpo_str = _phase_summary()
            elif (env, method) in hpo_running:
                pid, gpu, et = hpo_running[(env, method)]
                n_done = _hpo_trial_count(env, method) or 0
                eta_sec = _hpo_eta_seconds(
                    n_done, _parse_etime(et), env=env, method=method
                )
                hpo_str = f"gpu{gpu} ETA{_fmt_hms(eta_sec)}"
                # Live throughput from TB wall-clock between checkpoints
                # (excludes JAX compile by skipping the first eval). Tells
                # you "how fast is this trial currently running" rather
                # than the latest reward, which is already implied by the
                # bias columns to the right.
                rate = _hpo_steps_per_sec(env, method)
                inflight = _hpo_inflight_metric(env, method)
                step_k = (inflight[2] // 1000) if inflight else None
                rate_s = f"{rate:.0f} st/s" if rate else "- st/s"
                step_s = f"@{step_k}k" if step_k is not None else ""
                hpo_str += f" ({rate_s}{step_s})"
                # Append finished-phase summary so we keep historical
                # cost visible alongside the live throughput.
                summary = _phase_summary()
                if summary:
                    hpo_str += f"  [{summary}]"
                total_remaining_sec += eta_sec
            else:
                hpo_str = ""
                # Pending unit: try to extrapolate using the per-method live
                # rate from another env's running trial of the same method,
                # else fall back to the static default. Conservative.
                pending_eta = _hpo_eta_seconds(0, 0, env=env, method=method)
                total_remaining_sec += pending_eta
            # Read Eval/expert_bias directly from TB. Per-seed bias
            # (rl_reward[seed] - expert_reward[seed]) is logged at eval
            # time inside Ajax (see ajax/log.py:199), so the IQM here is
            # the seed-wise bias IQM, not (rl IQM - expert reference).
            # This is robust to expert-reference drift / unit mismatches.
            p1_best, p2_best, p3_best = _hpo_best_iqm_per_phase(
                env, method, tag="Eval/expert_bias"
            )
            # Per-phase completion counts. Phase1 = COMPLETE Optuna trials
            # vs HPO_TRIALS target; phase2/3 = result-JSON counts vs the
            # top-K and top-1 targets respectively.
            _hp_base = HP_RESULTS / env.lower() / method
            n_p1 = _hpo_trial_count(env, method) or 0
            n_p2 = (
                len(list(_hp_base.glob("trial_*_phase2_result.json")))
                if _hp_base.exists() else 0
            )
            n_p3 = (
                len(list(_hp_base.glob("trial_*_phase3_result.json")))
                if _hp_base.exists() else 0
            )

            def _fmt(v: Optional[float], n: int, target: int,
                     show_count: bool = True) -> str:
                width = 16 if show_count else 10
                if v is None:
                    return f"{'-':>{width}}"
                if show_count:
                    return f"{v:+9.2f} ({n}/{target})"
                return f"{v:+9.2f}"
            p1_str = _fmt(p1_best, n_p1, HPO_TRIALS)
            p2_str = _fmt(p2_best, n_p2, HPO_TOP_K)
            # Phase 3 is always a single top-1 rerun, no count needed.
            p3_str = _fmt(p3_best, n_p3, 1, show_count=False)
            # DEG columns (expert-using methods only): v1 mistune + v2 bias + v2 obs.
            if method in ("edge", "sac_quality_aware", "ibrl", "jsrl", "jsrl_curriculum", "residual"):
                # v1 mistune (existing param-perturbation sweep)
                deg_done = sum(
                    _degradation_done(env, method, s) for s in DEGRADATION_SIGMAS
                )
                running_sigs = sum(
                    1 for s in DEGRADATION_SIGMAS
                    if (env, method, str(s)) in deg_running
                )
                v1_str = f"{deg_done}/{len(DEGRADATION_SIGMAS)}"
                if running_sigs:
                    v1_str += f"+{running_sigs}r"
                n_deg_total += len(DEGRADATION_SIGMAS)
                n_deg_done += deg_done
                deg_remaining = len(DEGRADATION_SIGMAS) - deg_done
                total_remaining_sec += deg_remaining * _final_sec_for(env)

                # v2 action_bias / obs_noise. No "running" annotation: the v2
                # orchestrator (run_degradation_study_v2.py) is typically off-host
                # and ``ps`` here doesn't see its workers.
                v2_strs = []
                for dtype in ("action_bias", "obs_noise"):
                    intensities = V2_DEGRADATION_INTENSITIES[dtype]
                    done = sum(
                        _degradation_v2_done(env, method, dtype, i)
                        for i in intensities
                    )
                    v2_strs.append(f"{done}/{len(intensities)}")
            else:
                v1_str = "n/a"
                v2_strs = ["n/a", "n/a"]
            disp = _METHOD_DISPLAY.get(method, method)
            print(
                f"{env:<20} {disp:<10} {hpo_str:<44} "
                f"{p1_str:>16} {p2_str:>16} {p3_str:>10} "
                f"{v1_str:<8} {v2_strs[0]:<8} {v2_strs[1]:<8}"
            )
            n_hpo_total += 1
            n_hpo_done += int(hpo_done)

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
        f"DEG {n_deg_done}/{n_deg_total}, "
        f"RUNNING {n_running}"
    )
    print(
        f"study ETA (÷{n_gpus} GPUs): {_fmt_hms(study_eta_sec)} "
        f"(raw serial: {_fmt_hms(total_remaining_sec)})"
    )
    # Refresh the baseline-vs-edge-qa curve overlays (PNGs land at


def clean_results(confirm: bool) -> None:
    """Wipe all result artifacts for a clean-slate re-run.

    Removes: hp_results/, final_results/, degradation_results/,
    final_results_tb/, checkpoints/, study_logs/. Requires ``--confirm`` so
    this cannot be invoked by accident.
    """
    import shutil

    to_delete = [
        ROOT / "hp_results",
        DEGRADATION_RESULTS,
        ABLATION_RESULTS,
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
    DEGRADATION_RESULTS.mkdir(parents=True, exist_ok=True)
    ABLATION_RESULTS.mkdir(parents=True, exist_ok=True)

    print("[master] scanning existing artifacts")
    units = build_unit_list()
    pending = [u for u in units if not _unit_complete_on_disk(u)]
    print(f"[master] {len(pending)}/{len(units)} pending work units "
          f"(static list, rescanned each iteration) on GPUs {args.gpus}")
    if args.dry_run:
        for u in pending:
            print(f"  {u.name()}")
        return

    done_counter: dict = {}
    done_lock = threading.Lock()
    pick_lock = threading.Lock()
    in_progress: set = set()
    stop_event = threading.Event()
    workers = [
        GPUWorker(g, units, in_progress, pick_lock,
                  done_counter, done_lock, stop_event)
        for g in args.gpus
    ]
    for w in workers:
        w.start()
    try:
        for w in workers:
            w.join()
    except KeyboardInterrupt:
        print("[master] interrupt received, stopping workers")
        stop_event.set()
        for w in workers:
            w.join()

    print("[master] all workers finished")
    print(f"[master] summary: {dict(done_counter)}")


if __name__ == "__main__":
    main()
