"""Expert-degradation sensitivity study — v2: action-pipeline perturbations.

Complements ``run_degradation_eval.py`` (which perturbs *expert parameters*:
log-scale on PID gains, phase-wrap on CPG phases). v2 perturbs the
action / observation pipeline at deployment time, leaving the expert's
internal tuning intact:

  * ``action_bias``  : ``a' = clip(a* + b, -1, 1)`` with ``b ~ U(-σ, σ)`` drawn
                       once per call to this script (constant per run; encodes
                       a miscalibrated actuator zero).
  * ``latency``      : ``a'(t) = a*(t-k)`` for ``k`` env steps; the first ``k``
                       steps return zero ("clamped"). State is extended with a
                       ``(k, action_dim)`` ring buffer.
  * ``obs_noise``    : the expert sees ``obs + σ·N(0, I)`` per call (additive
                       Gaussian on raw obs, scale absolute — practitioners may
                       need to retune σ per env). The agent still sees clean
                       obs; this is "expert-only" obs noise to keep the change
                       local to the expert callsite. The "both" interpretation
                       (agent + expert see the same noisy obs) requires
                       wrapping the env or the data-collection loop and is
                       deliberately out of scope for v2.

Writes ``degradation_v2_results/<env>__<method>__<deg_type>__<intensity>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEGRADATION_V2_RESULTS = ROOT / "degradation_v2_results"

DEG_TYPES = ("action_bias", "latency", "obs_noise")


def _intensity_str(deg_type: str, intensity) -> str:
    """Return a stable filesystem-safe intensity tag for the output filename.

    Latency uses an integer step count; bias and obs_noise are floats.
    """
    if deg_type == "latency":
        return f"k{int(intensity)}"
    return f"{float(intensity)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--degradation-type", required=True, choices=DEG_TYPES,
        help="Which deployment-time perturbation to apply.",
    )
    parser.add_argument(
        "--intensity", required=True, type=float,
        help=(
            "Magnitude knob, interpreted per-type: action_bias σ "
            "(uniform half-range), latency k env-steps (cast to int), "
            "obs_noise σ (Gaussian std on raw obs)."
        ),
    )
    parser.add_argument("--n-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument(
        "--degradation-seed", type=int, default=0,
        help="Seed for any random draws in the perturbation (e.g. bias, obs noise).",
    )
    parser.add_argument(
        "--n-chunks", type=int, default=1,
        help="Split n-timesteps into N sequential train() calls with checkpoint-resume.",
    )
    args = parser.parse_args()
    if args.n_timesteps % args.n_chunks != 0:
        sys.stderr.write(
            f"[degradation_v2] --n-timesteps ({args.n_timesteps}) must be "
            f"divisible by --n-chunks ({args.n_chunks})\n"
        )
        return 5

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        from sac_hyperparam_search import (  # type: ignore
            build_agent_and_train,
            _load_expert_policy,
        )
    except ImportError as e:
        sys.stderr.write(
            f"[degradation_v2] cannot import build_agent_and_train: {e}\n"
        )
        return 2

    import jax
    import jax.numpy as jnp
    import optuna
    from optuna.trial import TrialState
    from envs import make_env_by_name  # type: ignore

    # ---------------------------------------------------------------- HPO load
    study_name = f"{args.method}_hp_search_{args.env.lower()}"
    candidates = [ROOT / "hp_results" / args.env.lower() / args.method / "optuna.db"]
    study = None
    for p in candidates:
        if not p.exists():
            continue
        try:
            study = optuna.load_study(study_name=study_name,
                                      storage=f"sqlite:///{p}")
            break
        except KeyError:
            continue
    if study is None:
        sys.stderr.write(
            f"[degradation_v2] No optuna study {study_name!r} in "
            f"{[str(c) for c in candidates if c.exists()]}\n"
        )
        return 3
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not done:
        sys.stderr.write("[degradation_v2] No COMPLETE trials\n")
        return 4
    best = max(done, key=lambda t: t.value)
    params_path = (
        ROOT / "hp_results" / args.env.lower() / args.method
        / f"trial_{best.number:03d}_params.json"
    )
    if params_path.exists():
        best_params = json.loads(params_path.read_text())
    else:
        best_params = dict(best.params)

    # ---------------------------------------------------------------- expert
    env, _env_params = make_env_by_name(args.env)
    anchor_expert = _load_expert_policy(args.env, env)

    # Probe the action_dim once for the latency wrapper. The expert is
    # vmapped per-env so we read it from ``anchor_gains`` shape OR from a
    # zero-state probe.
    action_dim = _probe_action_dim(anchor_expert, env)

    if args.degradation_type == "action_bias":
        sigma = float(args.intensity)
        bias_key = jax.random.PRNGKey(args.degradation_seed)
        bias = jax.random.uniform(
            bias_key, shape=(action_dim,), minval=-sigma, maxval=sigma,
        )
        degraded = BiasedExpertPolicy(anchor_expert, bias)
    elif args.degradation_type == "latency":
        k = int(args.intensity)
        if k < 0:
            sys.stderr.write(f"[degradation_v2] latency k={k} must be >= 0\n")
            return 6
        degraded = DelayedExpertPolicy(anchor_expert, k, action_dim)
    elif args.degradation_type == "obs_noise":
        sigma = float(args.intensity)
        noise_key = jax.random.PRNGKey(args.degradation_seed)
        degraded = ObsNoisyExpertPolicy(anchor_expert, sigma, noise_key)
    else:
        sys.stderr.write(
            f"[degradation_v2] unknown degradation_type={args.degradation_type!r}\n"
        )
        return 7

    print(
        f"[degradation_v2] {args.env}/{args.method} "
        f"{args.degradation_type}={args.intensity} "
        f"best trial={best.number} score={best.value:.2f}"
    )

    # ---------------------------------------------------------------- train
    seeds = list(range(args.n_seeds))
    CHECKPOINT_DIR = ROOT / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    intensity_tag = _intensity_str(args.degradation_type, args.intensity)
    ckpt_path = CHECKPOINT_DIR / (
        f"degrade_v2__{args.env.lower()}__{args.method}__"
        f"{args.degradation_type}__{intensity_tag}.pkl"
    )
    chunk_steps = args.n_timesteps // args.n_chunks
    run_ids: list = []
    tb_folder: str = ""
    for chunk in range(args.n_chunks):
        print(f"[degradation_v2] chunk {chunk+1}/{args.n_chunks}: "
              f"training {chunk_steps} steps (resume={chunk > 0})")
        final_state, run_ids, tb_folder = build_agent_and_train(
            env_name=args.env,
            method=args.method,
            best_params=best_params,
            seed_list=seeds,
            n_timesteps=chunk_steps,
            checkpoint_path=str(ckpt_path) if args.n_chunks > 1 else None,
            expert_override=degraded,
        )
        if args.n_chunks > 1:
            from ajax.checkpoint import save_checkpoint
            save_checkpoint(final_state, str(ckpt_path))

    # ---------------------------------------------------------------- write
    out = DEGRADATION_V2_RESULTS / (
        f"{args.env.lower()}__{args.method}__"
        f"{args.degradation_type}__{intensity_tag}.json"
    )
    DEGRADATION_V2_RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": args.env,
        "method": args.method,
        "degradation_type": args.degradation_type,
        "intensity": args.intensity,
        "intensity_tag": intensity_tag,
        "n_timesteps": args.n_timesteps,
        "n_seeds": args.n_seeds,
        "best_hpo_trial": best.number,
        "degradation_seed": args.degradation_seed,
    }
    try:
        from sac_hyperparam_search import read_final_metric  # type: ignore
        final_window = 200_000
        payload["final_metric"] = read_final_metric(
            tb_folder, run_ids, window_steps=final_window,
        )
        payload["final_metric_window_steps"] = final_window
        try:
            from ajax.logging.wandb_logging import load_scalars_from_tfevents
            EVAL_TAG = "Eval/episodic_mean_reward"
            per_seed = []
            for rid in run_ids:
                log_dir = os.path.join(tb_folder, "tensorboard", str(rid))
                if not os.path.exists(log_dir):
                    per_seed.append(None)
                    continue
                try:
                    scalars = load_scalars_from_tfevents(log_dir)
                except Exception:
                    per_seed.append(None)
                    continue
                if EVAL_TAG not in scalars or not scalars[EVAL_TAG]:
                    per_seed.append(None)
                    continue
                entries = scalars[EVAL_TAG]
                max_step = entries[-1][0]
                window = [v for step, v in entries
                          if step >= max_step - final_window]
                per_seed.append(
                    float(sum(window) / len(window)) if window else None
                )
            payload["per_seed_metric"] = per_seed
            payload["run_ids"] = [str(r) for r in run_ids]
        except Exception as e:
            payload["per_seed_metric_error"] = str(e)
    except Exception as e:
        payload["final_metric"] = None
        payload["final_metric_error"] = str(e)
    if args.n_chunks > 1:
        payload["n_chunks"] = args.n_chunks
        payload["chunk_steps"] = chunk_steps
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[degradation_v2] wrote {out}")
    return 0


# ===========================================================================
# Expert wrappers (stateful + stateless call forms; init_state for stateful)
# ===========================================================================


def _probe_action_dim(expert, env) -> int:
    """Best-effort action-dim probe.

    Tries:
      1. ``expert.anchor_gains`` shape (irrelevant — that's gain count, not action dim).
      2. ``env.action_space.shape`` for target_gym envs.
      3. Mujoco Playground envs: ``env.action_size`` if present, else
         ``env.unwrapped.sys.action_size`` for Brax-style.
    """
    try:
        return int(env.action_space().shape[0])  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return int(env.action_size)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return int(env.unwrapped.sys.action_size)  # type: ignore[attr-defined]
    except Exception:
        pass
    raise RuntimeError(
        "Could not probe action_dim for env; add an explicit handler."
    )


class BiasedExpertPolicy:
    """Adds a constant bias vector to expert actions, then clips to [-1, 1].

    Bias is drawn once at construction (per-run), shape ``(action_dim,)``,
    sampled from ``U(-σ, σ)``. Same bias used for every call (across env
    steps and across seeds in a vmapped batch).
    """

    def __init__(self, base, bias):
        self._base = base
        self._bias = bias

    def __call__(self, *args):
        import jax.numpy as jnp
        if len(args) == 2:
            state, obs = args
            action, new_state = self._base(state, obs)
            return jnp.clip(action + self._bias, -1.0, 1.0), new_state
        (obs,) = args
        action = self._base(obs)
        return jnp.clip(action + self._bias, -1.0, 1.0)

    def init_state(self, num_envs):
        return self._base.init_state(num_envs)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class DelayedExpertPolicy:
    """k-step latency on expert actions via a (num_envs, k, action_dim) buffer.

    Stateful call: pop oldest action from the buffer, push fresh action onto
    the right. The first ``k`` outputs are zeros (clamped). Stateless calls
    fall back to the unwrapped expert (delay is undefined for one-shot
    queries used by training-time bootstrap targets).
    """

    def __init__(self, base, k: int, action_dim: int):
        self._base = base
        self._k = int(k)
        self._action_dim = int(action_dim)

    def __call__(self, *args):
        import jax.numpy as jnp
        if len(args) == 2:
            state, obs = args
            base_state, buf = state
            fresh_action, new_base_state = self._base(base_state, obs)
            if self._k == 0:
                return fresh_action, (new_base_state, buf)
            output_action = buf[..., 0, :]
            new_buf = jnp.concatenate(
                [buf[..., 1:, :], fresh_action[..., None, :]],
                axis=-2,
            )
            return output_action, (new_base_state, new_buf)
        (obs,) = args
        return self._base(obs)

    def init_state(self, num_envs):
        import jax.numpy as jnp
        base_state = self._base.init_state(num_envs)
        if self._k == 0:
            buf = jnp.zeros((num_envs, 1, self._action_dim))
        else:
            buf = jnp.zeros((num_envs, self._k, self._action_dim))
        return (base_state, buf)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class ObsNoisyExpertPolicy:
    """Adds Gaussian noise to obs before passing to the wrapped expert.

    Per-step independent noise via PRNG fold_in on a step counter that
    lives in the wrapper's state. The agent still sees the clean obs from
    the env: only the expert's input is corrupted. This is "expert-only"
    obs noise; the "both" interpretation (agent + expert) requires hooking
    the env-step or the data-collection loop and is out of scope for v2.

    Stateless calls add noise per-call using a fresh fold_in on a hash of
    the obs (so the same obs gives the same noise — deterministic but
    decoupled from the rollout step counter).
    """

    def __init__(self, base, sigma: float, base_key):
        self._base = base
        self._sigma = float(sigma)
        self._base_key = base_key

    def __call__(self, *args):
        import jax
        import jax.numpy as jnp
        if len(args) == 2:
            state, obs = args
            base_state, step = state
            key = jax.random.fold_in(self._base_key, step)
            num_envs = obs.shape[0]
            keys = jax.random.split(key, num_envs)
            noise = jax.vmap(
                lambda k, o: jax.random.normal(k, shape=o.shape)
            )(keys, obs)
            noisy_obs = obs + self._sigma * noise
            action, new_base_state = self._base(base_state, noisy_obs)
            return action, (new_base_state, step + 1)
        # Stateless fallback: deterministic-per-obs noise via obs-hash fold.
        (obs,) = args
        key = jax.random.fold_in(self._base_key, 0)
        noise = jax.random.normal(key, shape=obs.shape)
        noisy_obs = obs + self._sigma * noise
        return self._base(noisy_obs)

    def init_state(self, num_envs):
        import jax.numpy as jnp
        base_state = self._base.init_state(num_envs)
        step = jnp.zeros((), dtype=jnp.int32)
        return (base_state, step)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


if __name__ == "__main__":
    sys.exit(main())
