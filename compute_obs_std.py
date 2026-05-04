"""Offline helper: compute per-dim observation std for the obs_noise study.

For each env, runs a single-seed rollout with the unperturbed expert for
``--n-steps`` env-steps, collects observations, computes per-dim
``std + epsilon``, and writes the result to ``obs_std/<env>.npy``.

The worker (``run_degradation_v2.py:_load_obs_std``) reads these files at
startup so that the obs_noise sigma is interpreted as a *fraction of
per-dim std* rather than an absolute Gaussian std on raw obs. Sigma
values calibrated against per-dim std (e.g. {0.05, 0.15, 0.30}) become
roughly comparable across envs with very different obs scales.

Mujoco Playground envs (CheetahRun, BarkourJoystick) use a different
env API (no ``env_params`` arg, env_state object instead of (obs, state)
tuple). They are skipped here; the worker's ``_load_obs_std`` falls back
to scalar ``1.0`` for them, in which case sigma is interpreted as
absolute Gaussian std on raw obs and may need per-env retuning.

Usage::

    python compute_obs_std.py                  # default target_gym sweep
    python compute_obs_std.py --envs FourTank  # one env
    python compute_obs_std.py --n-steps 2000   # longer rollout
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# CPU-only is fine here: this is a 1-seed lightweight rollout, not training.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent
OBS_STD_DIR = ROOT / "obs_std"

# Default target_gym envs in the v2 study. Excludes Mujoco Playground
# (CheetahRun, BarkourJoystick) which use a different env API; those use
# the scalar 1.0 fallback in the worker.
DEFAULT_TARGET_GYM_ENVS = [
    "Plane3DCircle",
    "Plane3DFigureEight",
    "FourTank",
    "GlassFurnace",
]


def rollout_obs(env, env_params, expert, n_steps: int, seed: int = 0):
    """Run env for n_steps with the unperturbed expert, return collected obs.

    Mirrors the rollout pattern in ``eval_pid_expert.py``: per-seed scan
    over env steps, freezing state past termination so we don't pull NaNs
    into the std calculation.
    """
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(seed)
    reset_key, scan_key = jax.random.split(key)
    obs0, state0 = env.reset(reset_key, env_params)
    expert_state0 = jax.tree.map(lambda x: x[0], expert.init_state(1))

    def body(carry, step_key):
        obs, state, expert_state, done = carry
        action, new_expert_state = expert(
            jax.tree.map(lambda x: x[None], expert_state), obs[None]
        )
        action = action[0]
        new_expert_state = jax.tree.map(lambda x: x[0], new_expert_state)
        new_obs, new_state, _reward, new_done, _info = env.step(
            step_key, state, action, env_params
        )
        # Freeze post-terminal state so subsequent obs don't drift.
        new_obs = jax.tree.map(lambda a, b: jnp.where(done, a, b), obs, new_obs)
        new_state = jax.tree.map(lambda a, b: jnp.where(done, a, b), state, new_state)
        new_expert_state = jax.tree.map(
            lambda a, b: jnp.where(done, a, b), expert_state, new_expert_state
        )
        return (new_obs, new_state, new_expert_state, done | new_done), new_obs

    sub_steps = getattr(env, "control_period", 1)
    # n_steps is in env-step units; cap at the env's max episode length so we
    # don't run past truncation (we don't reset mid-rollout for std est).
    max_in_episode = int(env_params.max_steps_in_episode) // int(sub_steps)
    n_run = min(int(n_steps), max_in_episode)
    step_keys = jax.random.split(scan_key, n_run)
    init = (obs0, state0, expert_state0, jnp.bool_(False))
    _, obs_seq = jax.lax.scan(body, init, step_keys)
    # Prepend the initial obs so we count it in the std.
    return jnp.concatenate([obs0[None], obs_seq], axis=0)


def compute_one(env_name: str, n_steps: int) -> dict:
    """Return {'env', 'obs_std', 'shape', 'n_used'} for one env."""
    import jax.numpy as jnp
    from envs import make_env_by_name  # type: ignore
    from sac_hyperparam_search import _load_expert_policy  # type: ignore

    env, env_params = make_env_by_name(env_name)
    if env_params is None:
        return {
            "env": env_name,
            "skipped": "no env_params (Playground env); use scalar fallback",
        }
    expert = _load_expert_policy(env_name, env)
    if not hasattr(expert, "init_state"):
        return {
            "env": env_name,
            "skipped": "expert is stateless; cannot reproduce its rollout API here",
        }

    obs_arr = rollout_obs(env, env_params, expert, n_steps=n_steps)
    std = jnp.std(obs_arr, axis=0) + 1e-6  # epsilon to avoid zero-std dims
    return {
        "env": env_name,
        "obs_std": std,
        "shape": tuple(std.shape),
        "n_used": int(obs_arr.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=DEFAULT_TARGET_GYM_ENVS)
    parser.add_argument("--n-steps", type=int, default=1000,
                        help="Env-steps per rollout (capped by env episode length).")
    args = parser.parse_args()

    OBS_STD_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np
    for env_name in args.envs:
        try:
            res = compute_one(env_name, args.n_steps)
        except Exception as e:
            print(f"[obs_std] {env_name}: ERROR ({type(e).__name__}: {e})",
                  file=sys.stderr)
            continue
        if "skipped" in res:
            print(f"[obs_std] {env_name}: SKIPPED ({res['skipped']})")
            continue
        out = OBS_STD_DIR / f"{env_name.lower()}.npy"
        np.save(out, np.asarray(res["obs_std"]))
        std_arr = res["obs_std"]
        print(f"[obs_std] {env_name}: wrote {out} "
              f"(shape={res['shape']}, n_used={res['n_used']}, "
              f"min={float(std_arr.min()):.4g}, max={float(std_arr.max()):.4g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
