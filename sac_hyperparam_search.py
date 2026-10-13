"""
hyperparam_search.py

Bayesian hyperparameter optimisation for vanilla SAC on the Plane environment.

Optimisation method
─────────────────────────────────────────────────────────────────────────────
Tree-structured Parzen Estimator (TPE) — the engine behind Optuna's Bayesian
search.  After an initial random phase (~10 trials) TPE builds two KDE models
P(x|good) and P(x|bad) from past results and proposes the next config by
sampling regions where P(x|good)/P(x|bad) is large.  With 40+ trials this
consistently outperforms grid / random search by a large margin.

Why not ASHA / HyperBand?
─────────────────────────────────────────────────────────────────────────────
True async successive halving needs per-step callbacks to prune a trial while
it is still running.  The current SAC.train() is a monolithic JAX-jit call
with no such hooks.  As a practical substitute this script uses two phases:

  Phase 1 — Exploration  (all N_TRIALS × SHORT_BUDGET)
    TPE directs the search; it learns from final metrics (not intermediate
    ones) but still gives ~80 % of ASHA's sample efficiency at this scale.

  Phase 2 — Confirmation (top-K configs × FULL_BUDGET, more seeds)
    Run automatically after Phase 1, or manually with --phase2.

Why not Population Based Training (PBT)?
─────────────────────────────────────────────────────────────────────────────
PBT requires copying model weights between workers mid-training.  This needs
save/restore of the full SAC state dict which is non-trivial.  Use Ray Tune
with your own checkpoint hooks if you want PBT later.

Multi-GPU architecture
─────────────────────────────────────────────────────────────────────────────
• One Optuna study, persisted in SQLite (hp_results/optuna.db) — fully
  resume-safe: kill the process and relaunch at any time.
• Master process owns the GPU pool and the study.
• Each trial is a subprocess with CUDA_VISIBLE_DEVICES set by the master.
  Trial params  →  hp_results/trial_NNN_params.json   (written by master)
  Trial result  →  hp_results/trial_NNN_result.json   (written by worker)
  Trial log     →  hp_results/trial_NNN.log
• Master reads results, reports to Optuna, and recycles the GPU slot.
• On restart: stale RUNNING trials are marked FAILED so TPE can re-explore.

Usage
─────────────────────────────────────────────────────────────────────────────
  python hyperparam_search.py                     # all GPUs, 40 trials, env=Plane
  python hyperparam_search.py --env Plane         # pick a TargetGym env
  python hyperparam_search.py --gpus 0 1          # restrict to GPUs 0-1
  python hyperparam_search.py --trials 60         # more trials
  python hyperparam_search.py --results           # ranked table
  python hyperparam_search.py --dry-run           # show search space only
  python hyperparam_search.py --phase2 --top-k 5  # confirm top-5 configs

Per-env artefacts land under hp_results/<env>/ so you can run multiple envs
in parallel without collisions.  The winning config is saved back into
hyperparams/ajax_sac.yml under the env name.

This script manages GPUs internally (one launcher process spawns workers
per GPU); use it directly rather than wrapping it in another launcher.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

# Linux PR_SET_PDEATHSIG: child receives SIGTERM the moment its parent
# dies (any reason). With this set on workers, killing the HPO master
# cascades down to every running trial without orphan trail. Same
# mechanism is used in run_full_study.py to cascade master -> HPO masters.
_libc_pdeath = ctypes.CDLL("libc.so.6", use_errno=True)
_PR_SET_PDEATHSIG = 1


def _set_pdeathsig() -> None:
    """preexec_fn helper: child gets SIGTERM when parent dies."""
    _libc_pdeath.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from envs import make_env_by_name, registered_envs

# ---------------------------------------------------------------------------
# Constants (per-env paths/names are derived from the --env flag)
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("hp_results")
# Single-phase search: maximize raw eval reward (absolute units).
# Ranking by eval reward is equivalent to ranking by expert advantage since the
# expert score is constant for a given env.
EVAL_TAG = "Eval/episodic_mean_reward"

SUPPORTED_METHODS = (
    # Paper methods (the ones we'll keep for the final HPO + comparison):
    "sac",                    # vanilla SAC baseline (no expert)
    "edge",                   # edge-eps: SAC + ε-greedy expert gating (gating-ablation baseline)
    "sac_quality_aware",      # SAC + symmetric LCB expert gating + obs+state aug
    "edge_qa_optimistic",     # variant of sac_quality_aware: HPO over signed β
                              # (β ∈ ±[0.1, 5.0]) and wider τ ∈ [0.001, 10.0]
                              # so the (β, τ) surface contains IBRL ((0, 0)
                              # limit), vanilla TS (β=0), pessimistic TS
                              # (β>0), and optimistic TS (β<0). Restricted to
                              # FourTank in build_unit_list (HPO chose β=0.1
                              # at the lower bound there, suggesting the
                              # optimum is on the optimistic side).
    "jsrl_curriculum",        # JSRL with per-episode curriculum (Uchendu et al. 2023, faithful)
    # Other agents kept for legacy compatibility — not part of the paper.
    "residual", "ibrl", "pid_policy", "jsrl",
)


def _is_td3(method: str) -> bool:
    return False


def _opt_int(v):
    """Cast to int, passing None through cleanly (used by
    ``_hpparams_from_dict`` so that a null entry in
    ``trial_<n>_params.json`` doesn't blow up on ``int(None)``)."""
    return None if v is None else int(v)


def _opt_float(v):
    """Float counterpart of ``_opt_int``."""
    return None if v is None else float(v)


def _env_slug(env_name: str) -> str:
    return env_name.lower()


def _results_dir(env_name: str, method: str = "sac") -> Path:
    """Per-method results dir. Every method uses its own subdir under
    ``hp_results/<env>/<method>/`` — no legacy env-root convention,
    so a fresh sac HPO can't collide with leftover trial files from
    a prior run.
    """
    slug = _env_slug(env_name)
    return RESULTS_ROOT / slug / method


def _db_path(env_name: str, method: str = "sac") -> Path:
    return _results_dir(env_name, method) / "optuna.db"


def _study_name(env_name: str, method: str = "sac") -> str:
    return f"{method}_hp_search_{_env_slug(env_name)}"


def _project_name(env_name: str, method: str = "sac") -> str:
    return f"{method}_hyperparam_search_{_env_slug(env_name)}"

# Legacy Phase 1 / Phase 2 constants — retained for backwards-compat with the
# existing SAC results cached under hp_results/<env>/.  New runs use the
# single-phase CLI flags (--search-timesteps, --search-seeds, --trials).
PHASE1_TIMESTEPS = int(3e5)
PHASE1_SEEDS = 20
PHASE2_TIMESTEPS = int(1e6)
PHASE2_SEEDS = 50
# Phase 3 (final eval): rerun the top-1 phase2 trial with PHASE3_SEEDS
# *additional* seeds (offset by PHASE2_SEEDS) so the trial's TB folder
# ends up holding PHASE2_SEEDS + PHASE3_SEEDS = 100 seeds for the
# headline number, without paying the cost of 100 fresh seeds.
PHASE3_TIMESTEPS = int(1e6)
PHASE3_SEEDS = 50
# Eval window is now a fraction of n_timesteps (was a fixed 200k, which
# was 67% of phase1 budget — too much). Use _eval_window(n_timesteps).
EVAL_WINDOW_FRAC = 0.2

# Default single-phase search budget (symmetric across methods).
SEARCH_TIMESTEPS = int(1e6)
SEARCH_SEEDS = 20
SEARCH_TRIALS = 30


def _eval_window(n_timesteps: int) -> int:
    """Trailing eval window in env-steps: fraction of the run length."""
    return max(1, int(EVAL_WINDOW_FRAC * n_timesteps))

# Asynchronous median pruning on running trials.
# Survivors run the full budget; pruned trials are SIGTERM'd mid-flight.
PRUNE_ENABLED_DEFAULT = True
PRUNE_WARMUP_STEPS = 300_000      # no pruning before this step
PRUNE_INTERVAL_STEPS = 100_000    # report cadence
PRUNE_N_STARTUP_TRIALS = 5        # first N trials never pruned (median seed)

LOG_FREQUENCY = 10_000
NUM_EPISODE_TEST = 25


def _eval_schedule(env_name: str, phase: int) -> tuple[int, bool]:
    """Per-(env, phase) (log_frequency, sweep) override.

    Reactor's full-horizon eval is ~10x longer per call than other envs
    (8640 env-steps vs ~1000), which made eval cost dominate at the
    default LOG_FREQUENCY=10k (~12 st/s observed). Special-case it:

    * phase1 (HPO search, 300k steps): only the trailing-window IQM
      matters, so sweep=True suppresses evals before 80% of training
      and frequency=12k yields exactly 5 evals over the last 60k window.
    * phase2/3 (1M-step reruns): we want granular learning curves for
      plotting, so 50k throughout (20 evals per run, 5x sparser than
      default, ~3-5x faster than current).

    Other envs keep the global defaults (LOG_FREQUENCY, sweep=False).
    """
    if env_name == "Reactor":
        if phase == 1:
            return 12_000, True
        return 50_000, False
    return LOG_FREQUENCY, False


# ---------------------------------------------------------------------------
# Hyperparameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class HPParams:
    """One set of SAC hyperparameters — JSON serialisable.

    Shared base dims are searched for every method.  EDGE-specific dims
    (exploration_decay_frac, exploration_tau, fixed_exploration_prob,
    policy_update_start) are populated only when method == "edge"; for other
    methods they stay None and fall back to ajax's defaults.
    """

    trial_number: int
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    gamma: float
    tau: float
    alpha_init: float
    target_entropy_per_dim: float
    batch_size: int
    arch_width: int  # 256 or 512; expands to full arch tuple
    max_grad_norm: Optional[float]  # None = no clipping
    method: str = "sac"
    # EDGE-only (None for other methods)
    exploration_decay_frac: Optional[float] = None
    exploration_tau: Optional[float] = None
    fixed_exploration_prob: Optional[float] = None
    policy_update_start: Optional[int] = None
    # TD3-only (None for other methods)
    exploration_noise: Optional[float] = None
    target_policy_noise: Optional[float] = None
    target_noise_clip: Optional[float] = None
    policy_delay: Optional[int] = None
    # quality-aware-only (None for other methods)
    num_critics: Optional[int] = None
    qa_temperature: Optional[float] = None
    # Expert-buffer prefill (gentler than actor BC — populates replay
    # buffer with expert transitions, leaves actor weights at random init)
    expert_buffer_n_steps: Optional[int] = None
    # target_entropy ramp axes (sac_qa)
    target_entropy_initial_per_dim: Optional[float] = None
    target_entropy_ramp_frac: Optional[float] = None
    # Thompson ε-floor: min P(policy) per step
    epsilon_floor: Optional[float] = None
    # LCB gating axes (sac_qa with exploration_lcb=True)
    lcb_beta_init: Optional[float] = None
    lcb_temperature: Optional[float] = None
    # Residual-RL scale (Johannink et al. 2019). Executed action is
    # ``clip(a_expert + residual_scale * a_pi, -1, 1)``. Only used when
    # method == "residual"; None for other methods.
    residual_scale: Optional[float] = None
    # JSRL curriculum decay fraction (Uchendu et al. 2023). Per-episode
    # handoff step H_t shrinks from H_max=episode_length to 0 over
    # ``jsrl_decay_frac × T_total`` env steps. None for other methods.
    jsrl_decay_frac: Optional[float] = None

    @property
    def actor_architecture(self) -> tuple:
        w = str(self.arch_width)
        return (w, "relu", w, "relu")

    @property
    def critic_architecture(self) -> tuple:
        w = str(self.arch_width)
        return (w, "relu", w, "relu")

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> HPParams:
        with open(path) as f:
            data = json.load(f)
        # Tolerate legacy keys written by an older orchestrator after the
        # HPParams surface was pruned (e.g. qa_beta_init, expert_mix_fraction,
        # imitation_coef). Unknown keys are silently dropped.
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _params_path(n: int, env_name: str, method: str = "sac") -> Path:
    return _results_dir(env_name, method) / f"trial_{n:03d}_params.json"


def _phase_suffix(phase2: bool, phase3: bool) -> str:
    if phase3:
        return "_phase3"
    if phase2:
        return "_phase2"
    return ""


def _result_path(n: int, env_name: str, phase2: bool = False, method: str = "sac",
                 phase3: bool = False) -> Path:
    suffix = _phase_suffix(phase2, phase3)
    return _results_dir(env_name, method) / f"trial_{n:03d}{suffix}_result.json"


def _log_path(n: int, env_name: str, phase2: bool = False, method: str = "sac",
              phase3: bool = False) -> Path:
    suffix = _phase_suffix(phase2, phase3)
    return _results_dir(env_name, method) / f"trial_{n:03d}{suffix}.log"


def _tb_folder(n: int, env_name: str, phase2: bool = False, method: str = "sac",
               phase3: bool = False) -> str:
    # Phase 3 shares phase 2's TB folder so the combined 100 seeds get
    # aggregated automatically by read_final_metric / monitor_curves.
    suffix = "_phase2" if (phase2 or phase3) else ""
    return str(_results_dir(env_name, method) / f"trial_{n:03d}{suffix}_tb")


# ---------------------------------------------------------------------------
# Optuna search space
# ---------------------------------------------------------------------------


def suggest_params(trial, method: str = "sac") -> HPParams:
    """Sample one configuration from the search space.

    All methods share the base 10 dims.  EDGE adds 4 method-specific dims on
    top (decay fraction, softmax temperature, exploration floor, and the
    warmup length before policy updates start).  Other methods inherit ajax's
    defaults for those knobs.
    """
    actor_lr = trial.suggest_float("actor_lr", 1e-4, 1e-3, log=True)
    critic_lr = trial.suggest_float("critic_lr", 1e-4, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.97, 0.999)
    tau = trial.suggest_float("tau", 5e-4, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    arch_width = trial.suggest_categorical("arch_width", [256, 512])
    grad_clip = trial.suggest_categorical("grad_clip", [0, 0.5, 1.0])
    max_grad_norm = None if grad_clip == 0 else float(grad_clip)

    # SAC-only knobs (entropy-related); skipped for TD3 trials so the search
    # space stays comparably sized rather than exploring dead dimensions.
    if _is_td3(method):
        alpha_lr = 1e-4  # placeholder, unused
        alpha_init = 1.0  # placeholder, unused
        target_entropy_per_dim = -1.0  # placeholder, unused
    else:
        alpha_lr = trial.suggest_float("alpha_lr", 1e-4, 1e-3, log=True)
        alpha_init = trial.suggest_float("alpha_init", 0.05, 2.0, log=True)
        # All methods share the same target_entropy_per_dim range. The
        # earlier sac_qa-specific tightening to [-1.5, -0.2] was put in
        # place when BC + critic MC pretrain were causing α to collapse
        # toward 0; both are now removed. edge-eps's best Plane3DCircle
        # config uses -2.79, which sat outside the tightened range —
        # that was a structural unfairness against sac_qa.
        target_entropy_per_dim = trial.suggest_float(
            "target_entropy_per_dim", -4.0, -0.2
        )

    # TD3-specific exploration / target-smoothing knobs
    exploration_noise = None
    target_policy_noise = None
    target_noise_clip = None
    policy_delay = None
    if _is_td3(method):
        exploration_noise = trial.suggest_float("exploration_noise", 0.05, 0.4)
        target_policy_noise = trial.suggest_float("target_policy_noise", 0.1, 0.4)
        target_noise_clip = trial.suggest_float("target_noise_clip", 0.2, 0.6)
        policy_delay = trial.suggest_categorical("policy_delay", [1, 2, 4])

    exploration_decay_frac = None
    exploration_tau = None
    fixed_exploration_prob = None
    policy_update_start = None
    if method in ("edge", "td3_edge"):
        exploration_decay_frac = trial.suggest_float("exploration_decay_frac", 0.1, 0.8)
        exploration_tau = trial.suggest_float("exploration_tau", 0.3, 3.0, log=True)
        fixed_exploration_prob = trial.suggest_float(
            "fixed_exploration_prob", 0.1, 0.9
        )
        if method == "edge":
            policy_update_start = trial.suggest_int(
                "policy_update_start", 1_000, 50_000, log=True
            )
    num_critics = None
    qa_temperature = None
    expert_buffer_n_steps_value = None
    target_entropy_initial_per_dim = None
    target_entropy_ramp_frac = None
    epsilon_floor = None
    lcb_beta_init = None
    lcb_temperature = None
    if method in ("sac_quality_aware", "r_edge_bow", "edge_qa_optimistic"):
        # Proposed methods: SAC + critic ensemble + LCB gating between
        # policy and (stateful) expert + obs+expert_state augmentation
        # + agent-side obs normalisation. Method-specific HPO axes:
        #   num_critics  — ensemble size (drives LCB σ signal)
        #   lcb_beta_init — disagreement penalty weight
        #   lcb_temperature — sigmoid sharpness on advantage
        # ``r_edge_bow`` adds asymmetric pessimism (LCB on expert /
        # UCB on policy) and the IBRL bootstrap proposal; same HPO
        # axes as sac_quality_aware so we can compare apples-to-apples.
        # ``edge_qa_optimistic`` extends β to ±[0.1, 5.0] (signed-log
        # via a sign categorical) and τ to [0.001, 10.0] so the (β, τ)
        # surface contains IBRL ((0, 0) limit), vanilla TS (β=0),
        # pessimistic TS (β>0), and optimistic TS (β<0). Used on
        # FourTank to test whether the lower-bound β=0.1 saturation in
        # the standard sac_quality_aware HPO indicates an unexplored
        # optimistic optimum.
        target_entropy_initial_per_dim = None
        target_entropy_ramp_frac = None
        qa_temperature = 1.0  # not used under LCB
        num_critics = trial.suggest_categorical("num_critics", [2, 4])
        expert_buffer_n_steps_value = 0
        epsilon_floor = 0.0
        if method == "edge_qa_optimistic":
            beta_sign = trial.suggest_categorical("lcb_beta_sign", [-1.0, 1.0])
            beta_mag = trial.suggest_float("lcb_beta_mag", 0.1, 5.0, log=True)
            lcb_beta_init = float(beta_sign) * float(beta_mag)
            lcb_temperature = trial.suggest_float(
                "lcb_temperature", 0.001, 10.0, log=True
            )
        else:
            lcb_beta_init = trial.suggest_float("lcb_beta_init", 0.1, 5.0, log=True)
            lcb_temperature = trial.suggest_float(
                "lcb_temperature", 0.1, 10.0, log=True
            )
    residual_scale = None
    if method == "residual":
        # Johannink et al. 2019 / Silver et al. 2018: scale the policy
        # term in clip(a_expert + scale * a_pi, -1, 1) so the initial
        # behavior is "expert + small correction" rather than
        # "expert + ±1 random noise". Default 0.3 in the literature; we
        # search [0.05, 1.0] log-uniform to let TPE pick per env.
        residual_scale = trial.suggest_float(
            "residual_scale", 0.05, 1.0, log=True
        )
    jsrl_decay_frac = None
    if method == "jsrl_curriculum":
        # True JSRL: per-episode handoff with H_t curriculum (Uchendu 2023).
        # Search the curriculum decay fraction; H_max is fixed to the
        # env's episode length (passed at agent construction).
        jsrl_decay_frac = trial.suggest_float(
            "jsrl_decay_frac", 0.05, 0.8, log=True
        )
    if method == "jsrl":
        # JSRL hard handoff: expert acts deterministically during the initial
        # fraction of training, then the learner takes over. fixed_exploration_prob
        # stays at 1.0 inside the handoff window; exploration_decay_frac
        # is the handoff point in training-step fraction.
        exploration_decay_frac = trial.suggest_float("exploration_decay_frac", 0.05, 0.5)
        exploration_tau = 1.0
        fixed_exploration_prob = 1.0
        policy_update_start = trial.suggest_int(
            "policy_update_start", 1_000, 50_000, log=True
        )

    return HPParams(
        trial_number=trial.number,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        gamma=gamma,
        tau=tau,
        alpha_init=alpha_init,
        target_entropy_per_dim=target_entropy_per_dim,
        batch_size=batch_size,
        arch_width=arch_width,
        max_grad_norm=max_grad_norm,
        method=method,
        exploration_decay_frac=exploration_decay_frac,
        exploration_tau=exploration_tau,
        fixed_exploration_prob=fixed_exploration_prob,
        policy_update_start=policy_update_start,
        exploration_noise=exploration_noise,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        policy_delay=policy_delay,
        num_critics=num_critics,
        qa_temperature=qa_temperature,
        expert_buffer_n_steps=expert_buffer_n_steps_value,
        target_entropy_initial_per_dim=target_entropy_initial_per_dim,
        target_entropy_ramp_frac=target_entropy_ramp_frac,
        epsilon_floor=epsilon_floor,
        lcb_beta_init=lcb_beta_init,
        lcb_temperature=lcb_temperature,
        residual_scale=residual_scale,
        jsrl_decay_frac=jsrl_decay_frac,
    )


# ---------------------------------------------------------------------------
# Metric extraction from TensorBoard
# ---------------------------------------------------------------------------


def read_final_metric(
    tb_folder: str, run_ids: list, window_steps: Optional[int] = None
) -> float:
    """
    Compute the eval metric averaged across all seed run_ids.

    window_steps=None  → use the last logged value only (Phase 1, fast).
    window_steps=N     → average over the last N timesteps (Phase 2, robust).

    Returns -inf if no data is found (trial treated as failed by Optuna).
    """
    from ajax.logging.wandb_logging import load_scalars_from_tfevents

    seed_scores = []
    for run_id in run_ids:
        log_dir = os.path.join(tb_folder, "tensorboard", run_id)
        if not os.path.exists(log_dir):
            continue
        try:
            scalars = load_scalars_from_tfevents(log_dir)
        except Exception:
            continue
        if EVAL_TAG not in scalars or not scalars[EVAL_TAG]:
            continue
        entries = scalars[EVAL_TAG]  # sorted list of (step, value)
        if window_steps is None:
            seed_scores.append(entries[-1][1])
        else:
            max_step = entries[-1][0]
            window = [v for step, v in entries if step >= max_step - window_steps]
            if window:
                seed_scores.append(sum(window) / len(window))

    if not seed_scores:
        return float("-inf")
    return float(sum(seed_scores) / len(seed_scores))


# ---------------------------------------------------------------------------
# Worker — entry point for each trial subprocess
# ---------------------------------------------------------------------------


_ENV_EPISODE_LENGTH: Dict[str, int] = {
    "Plane3DCircle": 10_000,
    "Plane3DFigureEight": 10_000,
    "Reactor": 8_640,
    "GlassFurnace": 5_760,
    "FourTank": 500,
    "CheetahRun": 1_000,
    "BarkourJoystick": 1_000,
}


def _build_method_kwargs(method: str, expert_policy, params: HPParams,
                         env_name: Optional[str] = None) -> dict:
    """Return the method-specific SAC(**kwargs) for a given trial.

    Base HPs from `params` are applied by the caller; this fn only returns the
    method-specific structural flags + any EDGE-tuned knobs that need to flow
    in as SAC kwargs.
    """
    if method == "sac":
        # Vanilla SAC baseline. Agent-side obs normalisation is on
        # everywhere as a fixed preprocessing detail (not an HPO axis).
        return dict(
            expert_policy=None,
            residual=False,
            use_pid_policy=False,
            use_expert_guidance=False,
            use_expert_guided_exploration=False,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
            use_mc_critic_pretrain=False,
            use_online_bc=False,
            use_online_critic_light_pretrain=False,
            normalize_obs_running=True,
        )
    if method == "edge":
        # edge-eps: SAC + ε-greedy expert gating. Gating-mechanism ablation
        # against sac_quality_aware (LCB). Fair-comparison: same obs+state
        # augmentation and running normalisation as edge-qa, so the only
        # varied axis is LCB vs ε-greedy.
        return dict(
            expert_policy=expert_policy,
            residual=False,
            use_pid_policy=False,
            use_expert_guidance=False,
            use_expert_guided_exploration=True,
            exploration_boltzmann=False,
            exploration_argmax=False,
            augment_obs_with_expert_state=True,
            normalize_obs_running=True,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
            use_mc_critic_pretrain=False,
            use_online_bc=False,
            use_online_critic_light_pretrain=False,
            use_critic_blend=False,
            num_critic_updates=1,
            # EDGE-tuned knobs
            exploration_decay_frac=params.exploration_decay_frac,
            exploration_tau=params.exploration_tau,
            fixed_exploration_prob=params.fixed_exploration_prob,
            policy_update_start=params.policy_update_start,
            alpha_update_start=params.policy_update_start,
        )
    if method == "residual":
        # Residual RL (Johannink et al. 2019): execute
        # clip(a_expert + scale * a_policy, -1, 1). The literature scale
        # of 0.3 is what makes residual a fair baseline; without it the
        # policy adds ±1 random noise to the expert at init and the
        # trial collapses on reward-sharpened envs (Plane3DCircle).
        # Fair-comparison: same obs+expert_state augmentation and
        # running observation normalisation as edge-qa, so the only
        # varied axis is the action-integration mechanism.
        return dict(
            expert_policy=expert_policy,
            residual=True,
            residual_scale=(params.residual_scale
                            if params.residual_scale is not None else 0.3),
            use_pid_policy=False,
            use_expert_guidance=False,
            use_expert_guided_exploration=False,
            augment_obs_with_expert_state=True,
            normalize_obs_running=True,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
            use_mc_critic_pretrain=False,
            use_online_bc=False,
            use_online_critic_light_pretrain=False,
        )
    if method == "ibrl":
        # Full IBRL: argmax action selection + IBRL bootstrap (max(Q_policy, Q_expert)
        # TD target, consistent with the argmax policy).
        # Fair-comparison: norm + state aug as in edge-qa.
        return dict(
            expert_policy=expert_policy,
            residual=False,
            use_pid_policy=False,
            use_expert_guidance=False,
            use_expert_guided_exploration=True,
            exploration_argmax=True,
            exploration_decay_frac=0.0,
            exploration_boltzmann=False,
            ibrl_bootstrap=True,
            augment_obs_with_expert_state=True,
            normalize_obs_running=True,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
            use_mc_critic_pretrain=False,
            use_online_bc=False,
            use_online_critic_light_pretrain=False,
        )
    if method == "jsrl_curriculum":
        # True JSRL (Uchendu et al. 2023): per-episode handoff. Expert
        # acts during steps [0, H_t) of every episode; learner acts
        # [H_t, end]. H_t shrinks linearly from episode_length to 0
        # over jsrl_decay_frac × T_total env-steps. Implemented in
        # train_SAC's action pipeline using the per-env step_in_episode
        # counter we added to CollectorState.
        # Fair-comparison: norm + state aug as in edge-qa.
        ep_len = _ENV_EPISODE_LENGTH.get(env_name, 1000) if env_name else 1000
        return dict(
            expert_policy=expert_policy,
            residual=False,
            use_pid_policy=False,
            use_expert_guidance=False,
            use_expert_guided_exploration=False,
            jsrl_curriculum=True,
            jsrl_episode_length=int(ep_len),
            jsrl_decay_frac=(
                float(params.jsrl_decay_frac)
                if params.jsrl_decay_frac is not None else 0.5
            ),
            augment_obs_with_expert_state=True,
            normalize_obs_running=True,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
            use_mc_critic_pretrain=False,
            use_online_bc=False,
            use_online_critic_light_pretrain=False,
        )
    if method == "jsrl":
        # Jump-Start RL with a closed-form expert: deterministic expert
        # actions during an initial training-step window of length
        # ``exploration_decay_frac * total_timesteps``, then pure learner.
        # Because the handoff is applied at training-step granularity rather
        # than in-episode-step granularity (as in the original JSRL paper),
        # the implementation is a simplification: expert is "online" during
        # the early window and "offline" after. Matches JSRL's high-level
        # behavior for long-horizon tasks where training-step fraction
        # approximates episodic-horizon fraction.
        # Fair-comparison: norm + state aug as in edge-qa.
        return dict(
            expert_policy=expert_policy,
            residual=False,
            use_pid_policy=False,
            use_expert_guidance=False,
            use_expert_guided_exploration=True,
            exploration_argmax=False,
            exploration_boltzmann=False,
            fixed_exploration_prob=params.fixed_exploration_prob,
            exploration_decay_frac=params.exploration_decay_frac,
            exploration_tau=params.exploration_tau,
            ibrl_bootstrap=False,
            augment_obs_with_expert_state=True,
            normalize_obs_running=True,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
            use_mc_critic_pretrain=False,
            use_online_bc=False,
            use_online_critic_light_pretrain=False,
            policy_update_start=params.policy_update_start,
            alpha_update_start=params.policy_update_start,
        )
    if method in ("sac_quality_aware", "r_edge_bow", "edge_qa_optimistic"):
        # Minimal principled method + expert_state augmentation.
        # The diagnostic showed the critic CAN'T fit Q(s, a) on a
        # PID-expert dataset (RMSE/std ≈ 0.30) without the integrator
        # state, but with [s, a, expert_state] it drops to ≈ 0.04. The
        # policy needs the same state so it can represent the optimal
        # action (which depends on the integrator). Asymmetric to
        # `augment_obs_with_expert_action`: this exposes a STATE, not
        # an ACTION, so it doesn't bias the policy toward copying the
        # expert.
        #
        # ``r_edge_bow`` is the robust variant: same gate machinery but
        # (1) the LCB scoring is asymmetric (LCB on expert / UCB on
        # policy) so the gate hands control to the policy whenever the
        # critic ensemble disagrees on a policy action, and (2) the
        # IBRL bootstrap proposal is added to the TD target so the
        # critic uses the expert as an optimistic anchor when the
        # expert is reliable, and falls back to vanilla SAC otherwise.
        is_robust = (method == "r_edge_bow")
        return dict(
            expert_policy=expert_policy,
            residual=False,
            use_pid_policy=False,
            use_expert_guidance=False,
            use_expert_guided_exploration=True,
            augment_obs_with_expert_action=False,
            augment_obs_with_expert_state=True,
            normalize_obs_running=True,
            exploration_lcb=True,
            exploration_thompson=False,
            lcb_beta_init=params.lcb_beta_init,
            lcb_beta_decay_k=2.0,
            lcb_temperature=params.lcb_temperature,
            # r_edge_bow gets the asymmetric LCB (UCB on policy arm)
            # AND the IBRL bootstrap proposal; sac_quality_aware uses
            # symmetric LCB and the standard SAC TD target.
            lcb_asymmetric=is_robust,
            ibrl_bootstrap=is_robust,
            epsilon_floor=0.0,         # no floor
            target_entropy_initial_per_dim=None,
            target_entropy_ramp_frac=0.5,  # unused when initial is None
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
            # Disable competing EDGE flags
            exploration_argmax=False,
            exploration_boltzmann=False,
            use_mc_critic_pretrain=False,
            use_online_bc=False,
            use_online_critic_light_pretrain=False,
            use_critic_blend=False,
            num_critic_updates=1,
        )
    if method == "pid_policy":
        return dict(
            expert_policy=expert_policy,
            residual=False,
            use_pid_policy=True,
            use_expert_guidance=False,
            use_expert_guided_exploration=False,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
            use_mc_critic_pretrain=False,
            use_online_bc=False,
            use_online_critic_light_pretrain=False,
        )
    if method == "td3":
        # Vanilla TD3: no expert, default Gaussian exploration pipeline.
        return dict(expert_policy=None)
    if method in ("td3_edge", "td3_residual", "td3_ibrl", "td3_jsrl",
                  "td3_quality_aware", "td3_always_expert"):
        # The action_pipeline is built later in _construct_agent (where it
        # has access to total_timesteps + params). Here we just signal that
        # the variant uses the expert.
        return dict(expert_policy=expert_policy)
    raise ValueError(f"Unknown method {method!r}; supported: {SUPPORTED_METHODS}")


_CPG_EXPERT_FACTORIES = {
    "cheetahrun": "make_cheetah_cpg_expert",
    "barkourjoystick": "make_barkour_cpg_expert",
}


def _hpparams_from_dict(
    best_params: dict,
    method: str,
    trial_number: int = -1,
    n_timesteps: Optional[int] = None,
) -> HPParams:
    """Coerce an Optuna ``trial.params`` dict into a structured ``HPParams``.

    Missing EDGE/JSRL fields are filled with ``None`` so downstream dispatch
    falls back to ajax defaults. ``max_grad_norm`` is normalised from the
    ``grad_clip`` categorical.

    ``n_timesteps`` (optional) is the training budget this HPParams will be
    used with. If provided and different from ``SEARCH_TIMESTEPS`` (the HPO
    budget), fraction-based knobs that semantically encode an absolute
    duration (``exploration_decay_frac``) are rescaled so absolute duration
    is preserved. This lets HPO at 1M and final-eval at 2M share the same
    effective decay window without re-running the search.
    """
    grad_clip = best_params.get("grad_clip", 0)
    max_grad_norm = None if grad_clip in (0, None) else float(grad_clip)

    decay_frac = best_params.get("exploration_decay_frac")
    if decay_frac is not None and n_timesteps is not None and n_timesteps != SEARCH_TIMESTEPS:
        abs_decay_steps = decay_frac * SEARCH_TIMESTEPS
        decay_frac = abs_decay_steps / n_timesteps

    # SAC-only fields may be absent for TD3 trials; supply unused defaults.
    is_td3 = _is_td3(method)
    return HPParams(
        trial_number=trial_number,
        actor_lr=float(best_params["actor_lr"]),
        critic_lr=float(best_params["critic_lr"]),
        alpha_lr=float(best_params.get("alpha_lr", 1e-4)),
        gamma=float(best_params["gamma"]),
        tau=float(best_params["tau"]),
        alpha_init=float(best_params.get("alpha_init", 1.0)),
        target_entropy_per_dim=float(best_params.get("target_entropy_per_dim", -1.0)),
        batch_size=int(best_params["batch_size"]),
        arch_width=int(best_params["arch_width"]),
        max_grad_norm=max_grad_norm,
        method=method,
        exploration_noise=(
            float(best_params["exploration_noise"]) if is_td3 else None
        ),
        target_policy_noise=(
            float(best_params["target_policy_noise"]) if is_td3 else None
        ),
        target_noise_clip=(
            float(best_params["target_noise_clip"]) if is_td3 else None
        ),
        policy_delay=(
            int(best_params["policy_delay"]) if is_td3 else None
        ),
        # Use ``_opt_int`` / ``_opt_float`` helpers so a None value (which
        # happens when a hardcoded HP slot is null in trial_<n>_params.json)
        # passes through cleanly instead of crashing on ``int(None)``.
        num_critics=_opt_int(best_params.get("num_critics")),
        qa_temperature=_opt_float(best_params.get("qa_temperature")),
        expert_buffer_n_steps=_opt_int(
            best_params.get("expert_buffer_n_steps")
        ),
        exploration_decay_frac=decay_frac,
        exploration_tau=_opt_float(best_params.get("exploration_tau")),
        fixed_exploration_prob=_opt_float(
            best_params.get("fixed_exploration_prob")
        ),
        policy_update_start=_opt_int(best_params.get("policy_update_start")),
        lcb_beta_init=_opt_float(best_params.get("lcb_beta_init")),
        lcb_temperature=_opt_float(best_params.get("lcb_temperature")),
        residual_scale=_opt_float(best_params.get("residual_scale")),
        jsrl_decay_frac=_opt_float(best_params.get("jsrl_decay_frac")),
    )


def _construct_agent(method, env, env_params, expert_policy, params, method_kwargs,
                     total_timesteps=None, n_envs=1, recurrent=False):
    """Build either an SAC or TD3 agent from the resolved HPParams + method
    kwargs. Centralizes the algo dispatch so build_agent_and_train and
    run_worker stay in sync.

    For TD3 variants (td3_edge / td3_residual / td3_ibrl / td3_jsrl), the
    appropriate ``action_pipeline`` from ``td3_pipelines`` is constructed here
    and passed to TD3, with ``total_timesteps`` bound into the schedule.
    """
    from ajax.agents.SAC.SAC import SAC  # local import to avoid module-level dep
    if _is_td3(method):
        from ajax.agents.TD3.TD3 import TD3
        td3_kwargs = dict(method_kwargs)  # may contain action_pipeline
        # Number of critics is configurable for quality_aware (where the
        # disagreement signal is the whole method); other variants stay
        # at TD3's default of 2.
        if method == "td3_quality_aware" and params.num_critics is not None:
            td3_kwargs["num_critics"] = int(params.num_critics)
        if method != "td3":
            # The variants need a custom action_pipeline. method_kwargs from
            # _build_method_kwargs only carries expert_policy + the variant
            # name; we resolve the pipeline here so we have access to params.
            import td3_pipelines as tp
            common = dict(
                expert_policy=expert_policy, n_envs=n_envs,
                recurrent=recurrent, exploration_noise=params.exploration_noise,
            )
            if method == "td3_edge":
                pipeline = tp.make_edge_pipeline(
                    **common,
                    total_timesteps=total_timesteps,
                    fixed_exploration_prob=params.fixed_exploration_prob,
                    exploration_decay_frac=params.exploration_decay_frac,
                    exploration_tau=params.exploration_tau,
                )
            elif method == "td3_residual":
                pipeline = tp.make_residual_pipeline(**common)
            elif method == "td3_ibrl":
                pipeline = tp.make_ibrl_pipeline(**common)
            elif method == "td3_jsrl":
                pipeline = tp.make_jsrl_pipeline(
                    **common,
                    total_timesteps=total_timesteps,
                    handoff_frac=params.exploration_decay_frac,
                )
            else:
                raise ValueError(f"Unknown TD3 variant {method!r}")
            td3_kwargs["action_pipeline"] = pipeline
            td3_kwargs["expert_policy"] = expert_policy
        return TD3(
            env_id=env,
            env_params=env_params,
            actor_learning_rate=params.actor_lr,
            critic_learning_rate=params.critic_lr,
            gamma=params.gamma,
            tau=params.tau,
            batch_size=params.batch_size,
            max_grad_norm=params.max_grad_norm,
            actor_architecture=params.actor_architecture,
            critic_architecture=params.critic_architecture,
            policy_delay=params.policy_delay,
            target_policy_noise=params.target_policy_noise,
            target_noise_clip=params.target_noise_clip,
            exploration_noise=params.exploration_noise,
            **td3_kwargs,
        )
    sac_kwargs = dict(method_kwargs)
    if params.num_critics is not None:
        sac_kwargs["num_critics"] = int(params.num_critics)
    return SAC(
        env_id=env,
        env_params=env_params,
        eval_expert_policy=expert_policy,
        actor_learning_rate=params.actor_lr,
        critic_learning_rate=params.critic_lr,
        alpha_learning_rate=params.alpha_lr,
        gamma=params.gamma,
        tau=params.tau,
        batch_size=params.batch_size,
        alpha_init=params.alpha_init,
        target_entropy_per_dim=params.target_entropy_per_dim,
        max_grad_norm=params.max_grad_norm,
        actor_architecture=params.actor_architecture,
        critic_architecture=params.critic_architecture,
        **sac_kwargs,
    )


def build_agent_and_train(
    env_name: str,
    method: str,
    best_params: dict,
    seed_list: list,
    n_timesteps: int,
    checkpoint_path: Optional[str] = None,
    expert_override: Optional[object] = None,
    method_kwargs_override: Optional[dict] = None,
    num_episode_test: int = None,
    env_wrapper: Optional[Callable] = None,
):
    """Build a SAC agent with ``best_params`` and train on ``seed_list`` seeds
    for ``n_timesteps`` environment steps.

    This is the reusable training path consumed by ``run_final_eval.py`` and
    ``run_degradation_eval.py``. Mirrors ``run_worker``'s agent-construction
    logic but without Optuna/HPO bookkeeping.

    Parameters
    ----------
    env_name : str
        One of the supported environments (Plane3DCircle, NuclearReactor,
        FourTank, CheetahRun, BarkourJoystick, ...).
    method : str
        One of ``SUPPORTED_METHODS`` (sac, edge, residual, ibrl, pid_policy,
        jsrl).
    best_params : dict
        Optuna ``trial.params`` dictionary from the HPO study.
    seed_list : list[int]
        Training seeds (final eval typically uses 100 fresh seeds).
    n_timesteps : int
        Total environment steps per seed.
    checkpoint_path : str, optional
        If provided and the file exists, resume from that checkpoint. After
        training, the final state is saved to the same path (atomic write).
    expert_override : object, optional
        If provided, replace the env-default expert with this
        ``FunctionalExpertPolicy`` (used by degradation study).
    num_episode_test : int, optional
        Number of evaluation episodes between log checkpoints. Defaults to
        the module-level ``NUM_EPISODE_TEST``.
    env_wrapper : callable, optional
        Applied to the env the agent trains on, after the expert is loaded
        from the bare env (used by degradation study v2's observation noise).

    Returns
    -------
    A 3-tuple ``(state, run_ids, tb_folder)``: the train() result, the per-seed
    TB run identifiers, and the directory containing the TB events. The caller
    can pass ``run_ids`` and ``tb_folder`` to ``read_final_metric`` to compute
    the actual evaluation score from the on-disk TB logs (since ``state`` no
    longer carries per-step metrics after the Ajax scan-ys removal).
    """
    from ajax.agents.SAC.SAC import SAC
    from ajax.checkpoint import (
        checkpoint_exists,
        restore_into,
        save_checkpoint,
    )
    from ajax.logging.wandb_logging import LoggingConfig

    if num_episode_test is None:
        num_episode_test = NUM_EPISODE_TEST

    env, env_params = make_env_by_name(env_name)
    expert_policy = (
        expert_override
        if expert_override is not None
        else _load_expert_policy(env_name, env)
    )
    if env_wrapper is not None:
        env = env_wrapper(env)
    params = _hpparams_from_dict(
        best_params, method=method, n_timesteps=n_timesteps
    )
    method_kwargs = _build_method_kwargs(method, expert_policy, params,
                                         env_name=env_name)
    if method_kwargs_override:
        method_kwargs = {**method_kwargs, **method_kwargs_override}

    _log_freq, _sweep = _eval_schedule(env_name, phase=3)
    logging_config = LoggingConfig(
        project_name=_project_name(env_name, method),
        run_name=f"final_eval_{method}",
        config={"method": method, "mode": "final_eval", **best_params},
        log_frequency=_log_freq,
        horizon=10_000,
        folder=str(FINAL_RESULTS_TB_DIR(env_name, method)),
        use_tensorboard=True,
        use_wandb=False,
        sweep=_sweep,
    )

    agent = _construct_agent(
        method, env, env_params, expert_policy, params, method_kwargs,
        total_timesteps=n_timesteps,
    )

    initial_state = None
    if checkpoint_path is not None and checkpoint_exists(checkpoint_path):
        # Resume: load into a freshly-initialised skeleton.
        skeleton = agent.train(
            seed=seed_list,
            n_timesteps=1,  # near-zero compute, just to build state
            num_episode_test=1,
        )
        initial_state = restore_into(skeleton, checkpoint_path)

    result = agent.train(
        seed=seed_list,
        logging_config=logging_config,
        n_timesteps=n_timesteps,
        num_episode_test=num_episode_test,
        initial_state=initial_state,
    )

    if checkpoint_path is not None:
        save_checkpoint(result, checkpoint_path)

    return result, list(agent.run_ids), str(FINAL_RESULTS_TB_DIR(env_name, method))


def FINAL_RESULTS_TB_DIR(env_name: str, method: str) -> Path:
    p = Path(__file__).resolve().parent / "final_results_tb" / env_name.lower() / method
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_expert_policy(env_name: str, env):
    """Load the cached expert pickle if present, else construct the expert.

    For TargetGym-native envs we fall back to ``env.expert_policy``.
    For Mujoco Playground locomotion envs we use a CPG factory from
    ``cpg_experts`` (CheetahRun, BarkourJoystick).
    """
    import pickle

    pkl = Path(f"expert_policy_{env_name.lower()}.pkl")
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)

    key = env_name.lower()
    if key in _CPG_EXPERT_FACTORIES:
        import cpg_experts as cpg_mod
        factory = getattr(cpg_mod, _CPG_EXPERT_FACTORIES[key])
        return factory()

    return env.expert_policy


def run_worker(
    params: HPParams,
    n_timesteps: int,
    n_seeds: int,
    env_name: str,
    phase2: bool = False,
    method: str = "sac",
    eval_window: Optional[int] = None,
    seed_offset: int = 0,
    phase3: bool = False,
) -> None:
    """
    Runs inside a subprocess.  JAX (and GPU allocation) is initialised here,
    not in the master, so each worker gets a clean CUDA context.

    CUDA_VISIBLE_DEVICES is set in the environment by the master before
    launching this subprocess.
    """
    import jax
    from ajax import SAC
    from ajax.logging.wandb_logging import LoggingConfig

    if eval_window is None:
        eval_window = _eval_window(n_timesteps)

    tb = _tb_folder(params.trial_number, env_name, phase2=phase2, method=method,
                    phase3=phase3)

    env, env_params = make_env_by_name(env_name)

    expert_policy = _load_expert_policy(env_name, env)

    method_kwargs = _build_method_kwargs(method, expert_policy, params,
                                         env_name=env_name)

    phase_tag = _phase_suffix(phase2, phase3)
    _phase_num = 3 if phase3 else (2 if phase2 else 1)
    _log_freq, _sweep = _eval_schedule(env_name, _phase_num)
    logging_config = LoggingConfig(
        project_name=_project_name(env_name, method),
        run_name=f"trial_{params.trial_number:03d}{phase_tag}",
        config={
            "debug": False,
            "method": method,
            "log_frequency": _log_freq,
            "actor_lr": params.actor_lr,
            "critic_lr": params.critic_lr,
            "alpha_lr": params.alpha_lr,
            "gamma": params.gamma,
            "tau": params.tau,
            "alpha_init": params.alpha_init,
            "target_entropy_per_dim": params.target_entropy_per_dim,
            "batch_size": params.batch_size,
            "arch_width": params.arch_width,
            "max_grad_norm": params.max_grad_norm,
            "exploration_decay_frac": params.exploration_decay_frac,
            "exploration_tau": params.exploration_tau,
            "fixed_exploration_prob": params.fixed_exploration_prob,
            "policy_update_start": params.policy_update_start,
            "phase": _phase_num,
        },
        log_frequency=_log_freq,
        horizon=10_000,
        folder=tb,
        use_tensorboard=True,
        use_wandb=False,
        sweep=_sweep,
    )

    agent = _construct_agent(
        method, env, env_params, expert_policy, params, method_kwargs,
        total_timesteps=n_timesteps,
    )

    backend = jax.default_backend()
    print(
        f"[trial {params.trial_number:03d}] device={backend}  "
        f"actor_lr={params.actor_lr:.1e}  critic_lr={params.critic_lr:.1e}  "
        f"alpha_lr={params.alpha_lr:.1e}  γ={params.gamma:.4f}  τ={params.tau:.1e}  "
        f"tep={params.target_entropy_per_dim:.2f}  α₀={params.alpha_init:.2f}  "
        f"batch={params.batch_size}  arch={params.arch_width}  "
        f"clip={params.max_grad_norm}"
    )

    t0 = time.time()
    seed_list = list(range(seed_offset, seed_offset + n_seeds))
    if backend != "gpu":
        from tqdm import tqdm

        for seed in tqdm(seed_list, desc=f"trial_{params.trial_number:03d}"):
            agent.train(
                seed=[seed],
                logging_config=logging_config,
                n_timesteps=n_timesteps,
                num_episode_test=NUM_EPISODE_TEST,
            )
    else:
        agent.train(
            seed=seed_list,
            logging_config=logging_config,
            n_timesteps=n_timesteps,
            num_episode_test=NUM_EPISODE_TEST,
        )

    elapsed = time.time() - t0
    # Single-phase search: always evaluate on a trailing window so the metric
    # mirrors the ablation's last-N-steps average (stable, not noisy last tick).
    metric = read_final_metric(
        tb,
        agent.run_ids,
        window_steps=eval_window,
    )
    print(
        f"[trial {params.trial_number:03d}] done  metric={metric:.2f}  "
        f"elapsed={elapsed:.0f}s"
    )

    phase_num = 3 if phase3 else (2 if phase2 else 1)
    result = {
        "trial_number": params.trial_number,
        "metric": metric,
        "elapsed_s": elapsed,
        "run_ids": list(agent.run_ids),
        "tb_folder": tb,
        "phase": phase_num,
        "n_seeds": n_seeds,
        "seed_offset": seed_offset,
    }
    with open(
        _result_path(params.trial_number, env_name, phase2=phase2, method=method,
                     phase3=phase3), "w"
    ) as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Master — manages the Optuna study and GPU pool
# ---------------------------------------------------------------------------


def _get_gpu_ids(requested: Optional[List[int]] = None) -> List[int]:
    if requested:
        return requested
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi failed — is CUDA available?")
    return [int(ln.strip()) for ln in result.stdout.strip().splitlines() if ln.strip()]


_GPU_CHECK_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "GPU_check.sh"
)


def _gpu_check_free_set() -> Optional[set]:
    """Return the set of GPU ids currently FREE per GPU_check.sh.

    GPU_check.sh enforces SMACT (compute), DRAMA (memory bandwidth), free
    VRAM, and per-GPU CUDA context count — a GPU is FREE only if all four
    are permissive. Returns None if the check could not be run (in which
    case callers should fail-open).
    """
    try:
        result = subprocess.run(
            ["bash", _GPU_CHECK_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Recommended: GPU "):
            after = line[len("Recommended: GPU ") :].split("(", 1)[0].strip()
            return {int(tok) for tok in after.split() if tok.isdigit()}
        if line.startswith("No fully-free GPU"):
            return set()
    return set()


def _gpu_is_claimable(
    gpu_id: int,
    running: Dict[int, "Optional[RunningJob]"],
    _cache: Dict[str, object] = {},
    _cache_ttl_s: float = 20.0,
) -> bool:
    """Can the master safely spawn a new worker on gpu_id?

    Uses GPU_check.sh (SMACT + DRAMA + VRAM + context count) so external
    processes competing for the GPU block our spawn. Our own running
    workers are tolerated: if gpu_id is in `running`, we grant it.

    Result is cached for ~20 s to avoid burning 3 s on dcgmi for each poll
    cycle per GPU in the spawn loop.
    """
    # Tolerate our own workers: if we already have a worker on this GPU the
    # slot is "taken" from the master's perspective and we never try to
    # double-spawn anyway, but grant the check so callers don't get confused.
    if running.get(gpu_id) is not None:
        return True

    now = time.time()
    last_ts = _cache.get("ts", 0.0)
    if now - last_ts > _cache_ttl_s or "set" not in _cache:
        _cache["set"] = _gpu_check_free_set()
        _cache["ts"] = now
    free_set = _cache["set"]
    if free_set is None:
        return True  # fail-open: check unavailable
    return gpu_id in free_set


def _create_or_load_study(env_name: str, method: str = "sac", enable_prune: bool = True):
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, NopPruner

    _results_dir(env_name, method).mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(
        n_startup_trials=10,  # first 10 trials are purely random (cold start)
        seed=42,
        constant_liar=True,  # parallel-safe: pretend in-flight trials return
        # the current best so next suggestions aren't clones
        multivariate=True,  # model cross-parameter correlations (e.g. lr × γ)
    )
    if enable_prune:
        # n_warmup_steps is enforced master-side (we only call trial.report
        # once last_step >= PRUNE_WARMUP_STEPS), so leave it at 0 here.
        pruner = MedianPruner(
            n_startup_trials=PRUNE_N_STARTUP_TRIALS,
            n_warmup_steps=0,
        )
    else:
        pruner = NopPruner()
    study = optuna.create_study(
        study_name=_study_name(env_name, method),
        storage=f"sqlite:///{_db_path(env_name, method)}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    return study


def _fix_stale_running_trials(study) -> int:
    import optuna

    fixed = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
            fixed += 1
    if fixed:
        print(f"[master] Marked {fixed} stale RUNNING trial(s) as FAILED.")
    return fixed


@dataclass
class RunningJob:
    trial_number: int
    gpu_id: int
    process: subprocess.Popen
    optuna_trial: object  # optuna.Trial
    log_file: object
    pruned: bool = False
    reported_buckets: set = None

    def __post_init__(self):
        if self.reported_buckets is None:
            self.reported_buckets = set()


def _poll_trial_progress(
    tb_folder: str, window_steps: int
) -> Tuple[Optional[int], Optional[float]]:
    """Parse a trial's in-flight TB event files and return
    (last_step, trailing_mean_reward_across_seeds) or (None, None) if no eval
    scalars are available yet.

    Uses raw protobuf parsing (tensorboardX only, no tensorflow dep).
    """
    import struct
    import glob
    from tensorboardX.proto import event_pb2

    seed_dirs = sorted(glob.glob(os.path.join(tb_folder, "tensorboard", "*")))
    if not seed_dirs:
        return None, None

    seed_last_steps = []
    seed_trailing_means = []
    for seed_dir in seed_dirs:
        events_files = sorted(glob.glob(os.path.join(seed_dir, "events.out.*")))
        if not events_files:
            continue
        eval_entries: list = []  # (step, value) for EVAL_TAG
        for path in events_files:
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            i = 0
            while i < len(data):
                if len(data) - i < 12:
                    break
                (length,) = struct.unpack("<Q", data[i : i + 8])
                i += 12
                if len(data) - i < length + 4:
                    break
                payload = data[i : i + length]
                i += length + 4
                try:
                    ev = event_pb2.Event.FromString(payload)
                except Exception:
                    break
                for v in ev.summary.value:
                    if v.tag == EVAL_TAG:
                        eval_entries.append((int(ev.step), float(v.simple_value)))
        if not eval_entries:
            continue
        eval_entries.sort()
        last_step = eval_entries[-1][0]
        window = [val for st, val in eval_entries if st >= last_step - window_steps]
        if not window:
            continue
        seed_last_steps.append(last_step)
        seed_trailing_means.append(sum(window) / len(window))

    if not seed_trailing_means:
        return None, None
    # All seeds step in lockstep (vmapped), so min == max in practice.
    return min(seed_last_steps), sum(seed_trailing_means) / len(seed_trailing_means)


def _spawn_worker(
    params: HPParams,
    gpu_id: int,
    n_timesteps: int,
    n_seeds: int,
    env_name: str,
    phase2: bool = False,
    method: str = "sac",
    eval_window: Optional[int] = None,
    seed_offset: int = 0,
    phase3: bool = False,
) -> subprocess.Popen:
    """Launch one trial as a subprocess on gpu_id.

    ``seed_offset`` lets phase 3 reuse the same trial number with seeds
    [seed_offset, seed_offset + n_seeds) so its TB events land alongside
    phase 2's in the same trial_<n>_tb folder, giving a combined
    100-seed score on read.
    """
    if eval_window is None:
        eval_window = _eval_window(n_timesteps)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("JAX_PLATFORMS", None)

    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--env",
        env_name,
        "--method",
        method,
        "--trial-number",
        str(params.trial_number),
        "--n-timesteps",
        str(n_timesteps),
        "--n-seeds",
        str(n_seeds),
        "--eval-window",
        str(eval_window),
        "--seed-offset",
        str(seed_offset),
    ]
    if phase2:
        cmd.append("--phase2-worker")
    if phase3:
        cmd.append("--phase3-worker")

    log_phase = phase3 or phase2
    log = open(_log_path(params.trial_number, env_name, phase2=log_phase, method=method, phase3=phase3), "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=_set_pdeathsig,
    )
    return proc, log


def run_master(
    gpu_ids: List[int],
    n_trials: int,
    n_timesteps: int,
    n_seeds: int,
    poll_interval: int,
    env_name: str,
    method: str = "sac",
    eval_window: Optional[int] = None,
    enable_prune: bool = PRUNE_ENABLED_DEFAULT,
    prune_warmup_steps: int = PRUNE_WARMUP_STEPS,
    prune_interval_steps: int = PRUNE_INTERVAL_STEPS,
) -> None:
    import optuna

    if eval_window is None:
        eval_window = _eval_window(n_timesteps)

    study = _create_or_load_study(env_name, method, enable_prune=enable_prune)
    _fix_stale_running_trials(study)

    # Pruned trials count toward the budget: they consumed an Optuna suggestion.
    _DONE_STATES = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }

    n_complete = sum(1 for t in study.trials if t.state in _DONE_STATES)
    pending = n_trials - n_complete
    if pending <= 0:
        print(f"All {n_trials} trials already complete.  Run --results to view.")
        return

    print(
        f"\nSingle-phase search [{method}]: {pending} trials remaining  |  "
        f"env={env_name}  |  GPUs: {gpu_ids}"
    )
    print(f"Budget: {n_timesteps:,} steps × {n_seeds} seeds per trial")
    print(
        f"Study DB: {_db_path(env_name, method)}   (kill/relaunch safely at any time)\n"
    )

    running: Dict[int, Optional[RunningJob]] = {g: None for g in gpu_ids}

    from tqdm import tqdm

    progress = tqdm(
        total=n_trials, initial=n_complete, desc=f"{method} trials", unit="trial"
    )

    try:
        while True:
            # ── Collect finished workers ──────────────────────────────────
            for gpu_id, job in list(running.items()):
                if job is None:
                    continue
                ret = job.process.poll()
                if ret is None:
                    continue

                job.log_file.close()
                rp = _result_path(job.trial_number, env_name, method=method)

                if job.pruned:
                    # Already told PRUNED at SIGTERM time; just reap.
                    progress.update(1)
                    n_done = sum(1 for t in study.trials if t.state in _DONE_STATES)
                    tqdm.write(
                        f"  [GPU {gpu_id}] ✂ trial {job.trial_number:03d} "
                        f"pruned  ({n_done}/{n_trials} done)"
                    )
                elif ret == 0 and rp.exists():
                    with open(rp) as f:
                        metric = json.load(f)["metric"]
                    study.tell(job.optuna_trial, metric)
                    n_done = sum(1 for t in study.trials if t.state in _DONE_STATES)
                    best = study.best_trial
                    progress.update(1)
                    progress.set_postfix(
                        {"best": f"{best.value:.2f}", "trial": best.number}
                    )
                    tqdm.write(
                        f"  [GPU {gpu_id}] ✓ trial {job.trial_number:03d}  "
                        f"score={metric:.2f}  ({n_done}/{n_trials} done)"
                    )
                else:
                    study.tell(job.optuna_trial, state=optuna.trial.TrialState.FAIL)
                    tqdm.write(
                        f"  [GPU {gpu_id}] ✗ trial {job.trial_number:03d}  "
                        f"exit={ret} — marked FAILED  (log: "
                        f"{_log_path(job.trial_number, env_name, method=method)})"
                    )

                running[gpu_id] = None

            # ── Poll running workers for async pruning ───────────────────
            if enable_prune:
                for gpu_id, job in running.items():
                    if job is None or job.pruned:
                        continue
                    tb = _tb_folder(job.trial_number, env_name, method=method)
                    last_step, trailing_mean = _poll_trial_progress(
                        tb, window_steps=eval_window
                    )
                    if last_step is None or last_step < prune_warmup_steps:
                        continue
                    bucket = (last_step // prune_interval_steps) * prune_interval_steps
                    if bucket in job.reported_buckets:
                        continue
                    job.reported_buckets.add(bucket)
                    job.optuna_trial.report(trailing_mean, bucket)
                    if job.optuna_trial.should_prune():
                        try:
                            job.process.terminate()
                        except ProcessLookupError:
                            pass
                        study.tell(
                            job.optuna_trial,
                            state=optuna.trial.TrialState.PRUNED,
                        )
                        job.pruned = True
                        tqdm.write(
                            f"  [GPU {gpu_id}] ✂ trial {job.trial_number:03d} "
                            f"PRUNE @ step {bucket:,}  trailing={trailing_mean:.2f}"
                        )

            # ── Check completion ──────────────────────────────────────────
            n_done = sum(1 for t in study.trials if t.state in _DONE_STATES)
            if n_done >= n_trials and all(j is None for j in running.values()):
                break

            # ── Fill free GPU slots ───────────────────────────────────────
            for gpu_id in gpu_ids:
                if running[gpu_id] is not None:
                    continue

                in_flight = sum(1 for j in running.values() if j is not None)
                n_done = sum(1 for t in study.trials if t.state in _DONE_STATES)
                if n_done + in_flight >= n_trials:
                    break

                # Skip GPUs that are currently occupied by an external process
                # (another HPO, another user). We keep polling each cycle, so
                # once the external job releases the GPU we pick it up.
                if not _gpu_is_claimable(gpu_id, running):
                    continue

                optuna_trial = study.ask()
                params = suggest_params(optuna_trial, method=method)
                params.trial_number = optuna_trial.number
                params.to_json(_params_path(params.trial_number, env_name, method=method))

                proc, log = _spawn_worker(
                    params=params,
                    gpu_id=gpu_id,
                    n_timesteps=n_timesteps,
                    n_seeds=n_seeds,
                    env_name=env_name,
                    method=method,
                    eval_window=eval_window,
                )
                running[gpu_id] = RunningJob(
                    trial_number=params.trial_number,
                    gpu_id=gpu_id,
                    process=proc,
                    optuna_trial=optuna_trial,
                    log_file=log,
                )
                print(
                    f"  [GPU {gpu_id}] → trial {params.trial_number:03d}  "
                    f"actor_lr={params.actor_lr:.1e}  critic_lr={params.critic_lr:.1e}  "
                    f"alpha_lr={params.alpha_lr:.1e}  γ={params.gamma:.4f}  "
                    f"τ={params.tau:.1e}  tep={params.target_entropy_per_dim:.2f}  "
                    f"α₀={params.alpha_init:.2f}  batch={params.batch_size}  "
                    f"arch={params.arch_width}  clip={params.max_grad_norm}"
                )

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers...")
        for gpu_id, job in running.items():
            if job is not None and job.process.poll() is None:
                job.process.terminate()
                job.log_file.close()
                study.tell(job.optuna_trial, state=optuna.trial.TrialState.FAIL)
                tqdm.write(f"  [GPU {gpu_id}] terminated trial {job.trial_number:03d}")
        progress.close()
        sys.exit(1)

    progress.close()
    print_results(env_name, study, method=method)


# ---------------------------------------------------------------------------
# Phase 2: confirm top-K at full budget
# ---------------------------------------------------------------------------


def run_phase2(top_k: int, gpu_ids: List[int], env_name: str, method: str = "sac") -> None:
    """Run top-K Phase 1 configs at full budget in parallel across GPUs."""
    import optuna
    from tqdm import tqdm

    study = _create_or_load_study(env_name, method)
    complete = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True,
    )
    if not complete:
        print("No completed Phase 1 trials found.  Run Phase 1 first.")
        return

    top = complete[:top_k]
    print(
        f"\nPhase 2 — confirmation: top-{len(top)} configs × "
        f"{PHASE2_TIMESTEPS:,} steps × {PHASE2_SEEDS} seeds  |  "
        f"env={env_name}  |  GPUs: {gpu_ids}\n"
    )

    # Separate already-cached trials from those that need running
    queue: deque = deque()
    for rank, trial in enumerate(top, 1):
        pp = _params_path(trial.number, env_name, method=method)
        if not pp.exists():
            tqdm.write(f"  [skip] trial {trial.number:03d} — params file missing")
            continue
        rp = _result_path(trial.number, env_name, phase2=True, method=method)
        if rp.exists():
            with open(rp) as f:
                cached = json.load(f)
            tqdm.write(
                f"  [{rank}/{len(top)}] trial {trial.number:03d}  "
                f"Phase1={trial.value:.2f}  Phase2={cached['metric']:.2f}  (cached)"
            )
            continue
        queue.append((rank, trial))

    if not queue:
        print("All Phase 2 trials already cached.")
        _print_phase2_summary(top, env_name, method=method)
        return

    running: Dict[int, Optional[RunningJob]] = {g: None for g in gpu_ids}
    progress = tqdm(
        total=len(top),
        initial=len(top) - len(queue),
        desc="Phase 2 trials",
        unit="trial",
    )

    try:
        while queue or any(j is not None for j in running.values()):
            # ── Collect finished workers ──────────────────────────────────
            for gpu_id, job in list(running.items()):
                if job is None:
                    continue
                ret = job.process.poll()
                if ret is None:
                    continue
                job.log_file.close()
                rp = _result_path(job.trial_number, env_name, phase2=True, method=method)
                if ret == 0 and rp.exists():
                    with open(rp) as f:
                        r = json.load(f)
                    progress.update(1)
                    progress.set_postfix(
                        {"last": f"{r['metric']:.2f}", "trial": job.trial_number}
                    )
                    tqdm.write(
                        f"  [GPU {gpu_id}] ✓ trial {job.trial_number:03d}  "
                        f"Phase2={r['metric']:.2f}  ({r['elapsed_s']:.0f}s)"
                    )
                else:
                    tqdm.write(
                        f"  [GPU {gpu_id}] ✗ trial {job.trial_number:03d}  "
                        f"exit={ret} — see "
                        f"{_log_path(job.trial_number, env_name, phase2=True, method=method)}"
                    )
                running[gpu_id] = None

            # ── Fill free GPU slots ───────────────────────────────────────
            for gpu_id in gpu_ids:
                if running[gpu_id] is not None or not queue:
                    continue
                rank, trial = queue.popleft()
                params = HPParams.from_json(
                    _params_path(trial.number, env_name, method=method)
                )
                tqdm.write(
                    f"  [GPU {gpu_id}] → trial {trial.number:03d}  "
                    f"Phase1={trial.value:.2f}"
                )
                proc, log = _spawn_worker(
                    params=params,
                    gpu_id=gpu_id,
                    n_timesteps=PHASE2_TIMESTEPS,
                    n_seeds=PHASE2_SEEDS,
                    env_name=env_name,
                    phase2=True,
                    method=method,
                )
                running[gpu_id] = RunningJob(
                    trial_number=trial.number,
                    gpu_id=gpu_id,
                    process=proc,
                    optuna_trial=trial,
                    log_file=log,
                )

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nInterrupted — terminating Phase 2 workers...")
        for gpu_id, job in running.items():
            if job is not None and job.process.poll() is None:
                job.process.terminate()
                job.log_file.close()
                tqdm.write(f"  [GPU {gpu_id}] terminated trial {job.trial_number:03d}")
        progress.close()
        sys.exit(1)

    progress.close()
    _print_phase2_summary(top, env_name, method=method)


def run_phase3(gpu_ids: List[int], env_name: str, method: str = "sac") -> None:
    """Final eval: pick the top-1 phase 2 trial and rerun it with
    PHASE3_SEEDS *additional* seeds (offset PHASE2_SEEDS) so the trial's
    TB folder ends up holding PHASE2_SEEDS + PHASE3_SEEDS = 100 seeds.
    """
    import optuna

    study = _create_or_load_study(env_name, method)
    complete = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True,
    )
    if not complete:
        print("[phase3] No completed trials.")
        return
    # Top-1 by phase 2 metric (fall back to phase 1 if phase 2 missing).
    best_trial = None
    best_phase2 = -float("inf")
    for t in complete:
        rp = _result_path(t.number, env_name, phase2=True, method=method)
        if not rp.exists():
            continue
        with open(rp) as f:
            r = json.load(f)
        if r["metric"] > best_phase2:
            best_phase2 = r["metric"]
            best_trial = t
    if best_trial is None:
        # No phase 2 finished — defer phase 3.
        print("[phase3] No phase 2 results yet; skipping.")
        return
    rp3 = _result_path(best_trial.number, env_name, phase2=False, method=method,
                       phase3=True)
    if rp3.exists():
        print(f"[phase3] cached: trial {best_trial.number:03d}.")
        return
    print(
        f"\n[phase3] final eval: trial {best_trial.number:03d}  "
        f"Phase2={best_phase2:.2f}  +{PHASE3_SEEDS} seeds × {PHASE3_TIMESTEPS:,} steps"
    )
    params = HPParams.from_json(_params_path(best_trial.number, env_name, method=method))
    gpu_id = gpu_ids[0]
    proc, log = _spawn_worker(
        params=params,
        gpu_id=gpu_id,
        n_timesteps=PHASE3_TIMESTEPS,
        n_seeds=PHASE3_SEEDS,
        env_name=env_name,
        method=method,
        seed_offset=PHASE2_SEEDS,
        phase3=True,
    )
    ret = proc.wait()
    log.close()
    if ret == 0 and rp3.exists():
        with open(rp3) as f:
            r = json.load(f)
        print(
            f"[phase3] ✓ trial {best_trial.number:03d}  "
            f"Phase3={r['metric']:.2f}  ({r['elapsed_s']:.0f}s)"
        )
    else:
        print(
            f"[phase3] ✗ trial {best_trial.number:03d}  exit={ret} — see "
            f"{_log_path(best_trial.number, env_name, method=method, phase3=True)}"
        )


def _print_phase2_summary(top, env_name: str, method: str = "sac") -> None:
    """Print a ranked table of Phase 2 results."""
    print(f"\n{'='*60}")
    print(f"Phase 2 summary  env={env_name}")
    print(f"{'='*60}")
    print(f"{'Rank':>4}  {'Trial':>5}  {'Phase1':>8}  {'Phase2':>8}")
    print("-" * 60)
    results = []
    for rank, trial in enumerate(top, 1):
        rp = _result_path(trial.number, env_name, phase2=True, method=method)
        if rp.exists():
            with open(rp) as f:
                r = json.load(f)
            results.append((r["metric"], trial.number, trial.value))
        else:
            results.append((float("-inf"), trial.number, trial.value))
    results.sort(reverse=True)
    for rank, (p2, tnum, p1) in enumerate(results, 1):
        p2_s = f"{p2:.2f}" if p2 != float("-inf") else "FAILED"
        print(f"{rank:>4}  {tnum:>5}  {p1:>8.2f}  {p2_s:>8}")
    best = next((r for r in results if r[0] != float("-inf")), None)
    if best:
        print(f"\nBest Phase 2: trial #{best[1]}  score={best[0]:.2f}")
        _save_best_to_yaml(best[1], env_name, method=method)


def _save_best_to_yaml(trial_number: int, env_name: str, method: str = "sac") -> None:
    """Persist the best trial's HPs.

    For method=="sac": writes hyperparams/ajax_sac.yml under env_name (legacy path
    consumed by load_hyperparams). For other methods: writes a sibling file
    hyperparams/ajax_sac_<method>.yml so each method's search result is
    recoverable without overwriting SAC's tuned base HPs.
    """
    import yaml

    pp = _params_path(trial_number, env_name, method=method)
    if not pp.exists():
        print(
            f"[save] params file missing for trial {trial_number:03d}, skipping YAML save."
        )
        return

    params = HPParams.from_json(pp)
    # Human-readable YAML mirror of the best HPParams. Lives inside
    # AjaxExperiments (next to the script) rather than in the sibling
    # Ajax library directory: experiment artifacts shouldn't dirty the
    # library's working tree, and nothing in our pipeline reads these
    # YAMLs back (the source of truth is trial_NNN_params.json under
    # hp_results/). The file is purely for at-a-glance inspection of
    # best-trial settings per (env, method).
    yaml_name = "ajax_sac.yml" if method == "sac" else f"ajax_sac_{method}.yml"
    yaml_path = Path(__file__).resolve().parent / "hyperparams" / yaml_name

    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    entry = {
        "actor_learning_rate": params.actor_lr,
        "critic_learning_rate": params.critic_lr,
        "alpha_learning_rate": params.alpha_lr,
        "gamma": params.gamma,
        "tau": params.tau,
        "alpha_init": params.alpha_init,
        "target_entropy_per_dim": params.target_entropy_per_dim,
        "batch_size": params.batch_size,
        "arch_width": params.arch_width,
        "max_grad_norm": params.max_grad_norm,
    }
    if method == "edge":
        entry.update({
            "exploration_decay_frac": params.exploration_decay_frac,
            "exploration_tau": params.exploration_tau,
            "fixed_exploration_prob": params.fixed_exploration_prob,
            "policy_update_start": params.policy_update_start,
        })
    cfg[env_name] = entry

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(
        f"[save] Best hyperparameters for env {env_name!r} (method={method}) "
        f"written to {yaml_path}  (trial #{trial_number})"
    )


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def print_results(env_name: str, study=None, method: str = "sac") -> None:
    import optuna

    if study is None:
        study = _create_or_load_study(env_name, method)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    if not complete:
        print("No completed trials yet.")
        return

    ranked = sorted(complete, key=lambda t: t.value, reverse=True)

    print(f"\n{'='*105}")
    print(
        f"Results env={env_name}  method={method}: "
        f"{len(complete)} completed, {len(failed)} failed"
    )
    print(f"{'='*105}")
    hdr = (
        f"{'Rank':>4}  {'#':>5}  {'Score':>8}  "
        f"{'actor_lr':>8}  {'critic_lr':>9}  {'alpha_lr':>8}  "
        f"{'γ':>6}  {'τ':>8}  {'tep':>5}  "
        f"{'α₀':>5}  {'batch':>5}  {'arch':>4}  {'clip':>5}"
    )
    print(hdr)
    print("-" * 105)
    for rank, t in enumerate(ranked, 1):
        p = t.params
        clip = p.get("grad_clip", 0)
        clip_s = "None" if clip == 0 else f"{float(clip):.1f}"
        print(
            f"{rank:>4}  {t.number:>5}  {t.value:>8.2f}  "
            f"{p.get('actor_lr', 0):>8.2e}  "
            f"{p.get('critic_lr', 0):>9.2e}  "
            f"{p.get('alpha_lr', 0):>8.2e}  "
            f"{p.get('gamma', 0):>6.4f}  "
            f"{p.get('tau', 0):>8.2e}  "
            f"{p.get('target_entropy_per_dim', 0):>5.2f}  "
            f"{p.get('alpha_init', 0):>5.2f}  "
            f"{int(p.get('batch_size', 0)):>5}  "
            f"{int(p.get('arch_width', 0)):>4}  "
            f"{clip_s:>5}"
        )

    best = ranked[0]
    print(f"\nBest: trial #{best.number}  score={best.value:.2f}")
    print(json.dumps(best.params, indent=4))
    _save_best_to_yaml(best.number, env_name, method=method)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian HPO for SAC and its expert-assisted variants. "
            "Single-phase symmetric protocol (--method picks the variant)."
        )
    )

    # Worker flags (hidden — called internally by master)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase2-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase3-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trial-number", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--n-timesteps", type=int, default=SEARCH_TIMESTEPS, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--n-seeds", type=int, default=SEARCH_SEEDS, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--eval-window", type=int, default=0,
        help=argparse.SUPPRESS,  # 0 = use EVAL_WINDOW_FRAC * n_timesteps
    )

    # Master / user flags
    parser.add_argument(
        "--env",
        type=str,
        default="Plane",
        help="TargetGym environment to run the HP search against (default: Plane).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sac",
        choices=SUPPORTED_METHODS,
        help=(
            "Which variant to tune. 'sac' = vanilla; 'edge' adds expert-guided "
            "exploration + 4 method-specific dims; 'residual', 'ibrl', "
            "'pid_policy' inherit the same 10-dim base search."
        ),
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="GPU IDs to use (default: all detected by nvidia-smi).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=SEARCH_TRIALS,
        help=f"Total trials (default: {SEARCH_TRIALS}).",
    )
    parser.add_argument(
        "--search-timesteps",
        type=int,
        default=SEARCH_TIMESTEPS,
        help=f"Env steps per trial (default: {SEARCH_TIMESTEPS}).",
    )
    parser.add_argument(
        "--search-seeds",
        type=int,
        default=SEARCH_SEEDS,
        help=f"Seeds per trial (default: {SEARCH_SEEDS}).",
    )
    parser.add_argument(
        "--search-eval-window",
        type=int,
        default=0,
        help=(
            "Trailing-window size (env steps) over which the eval metric is "
            f"averaged when scoring a trial. 0 (default) = "
            f"{int(EVAL_WINDOW_FRAC * 100)}%% of --search-timesteps."
        ),
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=30,
        help="Master poll interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Disable async median pruning (run all trials to full budget).",
    )
    parser.add_argument(
        "--prune-warmup-steps",
        type=int,
        default=PRUNE_WARMUP_STEPS,
        help=(
            "No pruning decision before this env-step count "
            f"(default: {PRUNE_WARMUP_STEPS:,})."
        ),
    )
    parser.add_argument(
        "--prune-interval-steps",
        type=int,
        default=PRUNE_INTERVAL_STEPS,
        help=(
            "Re-evaluate prune decision every N steps "
            f"(default: {PRUNE_INTERVAL_STEPS:,})."
        ),
    )
    parser.add_argument(
        "--results", action="store_true", help="Print ranked results table and exit."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show search space info and exit."
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Run Phase 2 confirmation for top-K configs (legacy, not auto-run).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top configs to confirm in Phase 2 (default: 5).",
    )
    parser.add_argument(
        "--auto-phase2",
        action="store_true",
        help=(
            "After Phase 1 (single-phase master) finishes, automatically "
            "run Phase 2 in the same process. Used by run_full_study.py "
            "to chain the two phases without dispatching a second subprocess."
        ),
    )
    parser.add_argument(
        "--auto-phase3",
        action="store_true",
        help=(
            "After Phase 2, run Phase 3 (final eval, top-1 + 50 extra seeds) "
            "in the same process."
        ),
    )

    args = parser.parse_args()

    # ── Worker subprocess ─────────────────────────────────────────────────
    if args.worker:
        if args.trial_number is None:
            sys.exit("--trial-number required in --worker mode")
        pp = _params_path(args.trial_number, args.env, method=args.method)
        if not pp.exists():
            sys.exit(f"Params file not found: {pp}")
        params = HPParams.from_json(pp)
        run_worker(
            params=params,
            n_timesteps=args.n_timesteps,
            n_seeds=args.n_seeds,
            env_name=args.env,
            phase2=args.phase2_worker,
            phase3=args.phase3_worker,
            seed_offset=args.seed_offset,
            method=args.method,
            eval_window=(args.eval_window or None),
        )
        return

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        import optuna

        _study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42),
            direction="maximize",
        )
        print(f"\nSearch space for method={args.method!r} "
              "(10 random samples from TPE cold start):\n")
        hdr = (
            f"{'#':>2}  {'actor_lr':>8}  {'critic_lr':>9}  {'alpha_lr':>8}  "
            f"{'γ':>6}  {'τ':>8}  {'tep':>5}  "
            f"{'α₀':>5}  {'batch':>5}  {'arch':>4}  {'clip':>5}"
        )
        if args.method == "edge":
            hdr += f"  {'decay':>6}  {'tau':>6}  {'fxprb':>5}  {'pstart':>7}"
        print(hdr)
        print("-" * len(hdr))
        for i in range(10):
            t = _study.ask()
            p = suggest_params(t, method=args.method)
            clip_s = "None" if p.max_grad_norm is None else f"{p.max_grad_norm:.1f}"
            line = (
                f"{i:>2}  {p.actor_lr:>8.2e}  {p.critic_lr:>9.2e}  {p.alpha_lr:>8.2e}  "
                f"{p.gamma:>6.4f}  {p.tau:>8.2e}  "
                f"{p.target_entropy_per_dim:>5.2f}  {p.alpha_init:>5.2f}  "
                f"{p.batch_size:>5}  {p.arch_width:>4}  {clip_s:>5}"
            )
            if args.method == "edge":
                line += (
                    f"  {p.exploration_decay_frac:>6.3f}  "
                    f"{p.exploration_tau:>6.3f}  "
                    f"{p.fixed_exploration_prob:>5.2f}  "
                    f"{p.policy_update_start:>7d}"
                )
            print(line)
        print(
            f"\nBudget: {args.trials} trials × {args.search_seeds} seeds "
            f"× {args.search_timesteps:,} steps"
        )
        ew = args.search_eval_window or _eval_window(args.search_timesteps)
        print(f"Eval window (trailing): {ew:,} steps")
        print(f"Study DB: {_db_path(args.env, args.method)}")
        return

    # ── Results ───────────────────────────────────────────────────────────
    if args.results:
        print_results(args.env, method=args.method)
        return

    # ── Phase 2 (legacy confirmation) ─────────────────────────────────────
    if args.phase2:
        gpu_ids = _get_gpu_ids(args.gpus)
        run_phase2(
            top_k=args.top_k, gpu_ids=gpu_ids, env_name=args.env, method=args.method
        )
        return

    # ── Single-phase master ───────────────────────────────────────────────
    gpu_ids = _get_gpu_ids(args.gpus)
    # Stagger orchestrators so concurrent workers don't draw identical TPE
    # startup samples from the shared seed=42 RandomSampler. A 30s/gpu_id
    # offset is enough for the first study.ask() to serialize via storage.
    if gpu_ids:
        stagger_s = min(gpu_ids[0], 4) * 30
        if stagger_s > 0:
            print(f"[stagger] sleeping {stagger_s}s to avoid TPE startup collision")
            time.sleep(stagger_s)
    print(
        f"Candidate GPUs: {gpu_ids}  env={args.env}  method={args.method}  "
        f"budget={args.trials} × {args.search_seeds} seeds × "
        f"{args.search_timesteps:,} steps"
    )
    run_master(
        gpu_ids=gpu_ids,
        n_trials=args.trials,
        n_timesteps=args.search_timesteps,
        n_seeds=args.search_seeds,
        poll_interval=args.poll,
        env_name=args.env,
        method=args.method,
        eval_window=(args.search_eval_window or None),
        enable_prune=not args.no_prune,
        prune_warmup_steps=args.prune_warmup_steps,
        prune_interval_steps=args.prune_interval_steps,
    )

    # Phase 2 auto-chain: re-run the top-K Phase 1 trials at the larger
    # PHASE2_TIMESTEPS x PHASE2_SEEDS budget. Uses the same Optuna study;
    # the per-trial JSON gets a Phase-2 score that the master prefers when
    # picking the final best.
    if args.auto_phase2:
        print(f"\n[auto-phase2] Phase 1 done; running Phase 2 top-{args.top_k} confirmation.")
        run_phase2(
            top_k=args.top_k,
            gpu_ids=gpu_ids,
            env_name=args.env,
            method=args.method,
        )
        if args.auto_phase3:
            print("\n[auto-phase3] Phase 2 done; running Phase 3 final eval.")
            run_phase3(
                gpu_ids=gpu_ids,
                env_name=args.env,
                method=args.method,
            )


if __name__ == "__main__":
    main()
