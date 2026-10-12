"""Expert-degradation sensitivity study — v2: deployment-side perturbations.

Complements ``run_degradation_eval.py`` (which perturbs *expert parameters*:
log-scale on PID gains, phase-wrap on CPG phases). v2 perturbs the
deployment-side action / observation pipeline at training time, leaving the
expert's internal tuning intact:

  * ``action_bias`` : ``a' = clip(a* + b_i, -1, 1)`` with **per-seed**
                       ``b_i ~ U(-σ, σ)`` drawn deterministically from
                       ``fold_in(key, env_idx)``. Each of the ``n_seeds``
                       parallel envs gets its own constant bias for the run.
                       Models miscalibrated actuator zero.
  * ``obs_noise``   : the agent sees ``obs + σ·obs_std·N(0, I)``, drawn
                       afresh at every reset and env-step. Per-dim obs std
                       is estimated by a short warmup pass (cached to
                       ``obs_std/<env>.npy``) so σ is dimensionless and
                       comparable across envs. The training env is wrapped
                       in Ajax's ``ObservationNoiseWrapper`` (target_gym) or
                       ``ObservationNoiseWrapperBrax`` (Playground / Brax).
                       The expert reads the env's raw observation: the
                       noiseless ``get_obs`` on target_gym envs, the noisy
                       ``state.obs`` on Playground / Brax.
  * ``action_dropout`` (TBD, deferred): with prob ``p`` repeat the last
                       action instead of querying the expert. Models
                       intermittent expert availability. Implemented as a
                       stateful wrapper analogous to bias.

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

DEG_TYPES = ("action_bias", "obs_noise")
OBS_STD_DIR = ROOT / "obs_std"


def _intensity_str(deg_type: str, intensity) -> str:
    """Return a stable filesystem-safe intensity tag for the output filename."""
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

    action_dim = _probe_action_dim(anchor_expert, env)

    env_wrapper = None
    if args.degradation_type == "action_bias":
        sigma = float(args.intensity)
        # Per-seed independent bias: deterministic from (degradation_seed, env_idx).
        # Wrapper draws the per-env bias at each call via fold_in, so 50 seeds
        # in a vmapped batch get 50 different biases (each constant across
        # env steps within a seed).
        bias_key = jax.random.PRNGKey(args.degradation_seed)
        degraded = BiasedExpertPolicy(
            anchor_expert, sigma=sigma, base_key=bias_key,
            action_dim=action_dim,
        )
    elif args.degradation_type == "obs_noise":
        # build_agent_and_train builds its own env: the noise must wrap that
        # one, not the env built here.
        env_wrapper = _obs_noise_wrapper(args.env, float(args.intensity))
        # Expert is unwrapped: the noise is on the env, not the expert.
        degraded = anchor_expert
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
            env_wrapper=env_wrapper,
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
    """Adds a per-seed constant bias to expert actions, then clips to [-1, 1].

    Each of the ``num_envs`` parallel envs (= seeds in our vmapped training)
    gets its own bias vector ``b_i ~ U(-σ, σ)``, drawn deterministically
    from ``fold_in(base_key, env_idx)`` so it is constant across env steps
    within a seed but varies across seeds.

    The bias is materialized lazily on each call (cheap, deterministic),
    so we don't need to thread it through state. Stateless calls use
    ``num_envs`` inferred from ``obs.shape[0]``.
    """

    def __init__(self, base, sigma: float, base_key, action_dim: int):
        self._base = base
        self._sigma = float(sigma)
        self._base_key = base_key
        self._action_dim = int(action_dim)

    def _per_env_bias(self, num_envs):
        import jax
        import jax.numpy as jnp
        env_keys = jax.vmap(
            lambda i: jax.random.fold_in(self._base_key, jnp.uint32(i))
        )(jnp.arange(num_envs, dtype=jnp.uint32))
        return jax.vmap(
            lambda k: jax.random.uniform(
                k, shape=(self._action_dim,),
                minval=-self._sigma, maxval=self._sigma,
            )
        )(env_keys)

    def __call__(self, *args):
        import jax.numpy as jnp
        if len(args) == 2:
            state, obs = args
            action, new_state = self._base(state, obs)
            biases = self._per_env_bias(obs.shape[0])
            return jnp.clip(action + biases, -1.0, 1.0), new_state
        (obs,) = args
        action = self._base(obs)
        biases = self._per_env_bias(obs.shape[0])
        return jnp.clip(action + biases, -1.0, 1.0)

    def init_state(self, num_envs):
        return self._base.init_state(num_envs)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def _load_obs_std(env_name: str):
    """Load per-dim obs std for ``env_name`` from ``obs_std/<env>.npy``.

    Returns a JAX array. Falls back to scalar ``1.0`` (no per-dim scaling)
    if the cache file is missing — in that case σ is interpreted as
    absolute noise std on raw obs, and may need per-env retuning.

    To populate the cache, run ``compute_obs_std.py --env <name>``
    offline once per env; the resulting ``.npy`` file is committed to git.
    """
    import jax.numpy as jnp
    cache = OBS_STD_DIR / f"{env_name.lower()}.npy"
    if not cache.exists():
        sys.stderr.write(
            f"[obs_std] {cache} not found; using obs_std=1.0 (no per-dim scaling). "
            f"Run compute_obs_std.py to populate.\n"
        )
        return jnp.asarray(1.0)
    import numpy as np
    arr = np.load(cache)
    return jnp.asarray(arr)


def _obs_noise_wrapper(env_name: str, sigma: float):
    """Return a function wrapping an env so that its observations are
    ``obs + sigma * obs_std * N(0, I)`` (Ajax's observation-noise wrappers,
    the Brax one for Playground / Brax stacks)."""
    from ajax.environments.utils import get_env_type
    from ajax.wrappers import ObservationNoiseWrapper, ObservationNoiseWrapperBrax

    obs_std = _load_obs_std(env_name)

    def wrap(env):
        if get_env_type(env) == "brax":
            return ObservationNoiseWrapperBrax(env, sigma=sigma, scale=obs_std)
        return ObservationNoiseWrapper(env, sigma=sigma, scale=obs_std)

    return wrap


# NOTE: action_dropout (deferred per user; "do at end if time"). Will be
# implemented as a stateful wrapper:
#   class DropoutExpertPolicy:
#       state = (base_state, last_action, prng_step)
#       __call__: with prob p, return last_action; else fresh_action and update
#       last_action. Stateless calls fall back to unwrapped expert.
# Add `action_dropout` to DEG_TYPES, INTENSITIES (e.g. p in {0.1, 0.3, 0.5}),
# and a dispatch branch in main().


if __name__ == "__main__":
    sys.exit(main())
