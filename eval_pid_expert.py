"""Evaluate the pure PID expert on Plane3DHeading over N seeds."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import jax
import jax.numpy as jnp

from envs import make_env_by_name


def rollout_one(env, env_params, expert, key):
    reset_key, step_key = jax.random.split(key)
    obs, state = env.reset(reset_key, env_params)
    expert_state = jax.tree.map(lambda x: x[0], expert.init_state(1))

    def body(carry, step_key):
        obs, state, expert_state, ret, done = carry
        action, new_expert_state = expert(
            jax.tree.map(lambda x: x[None], expert_state), obs[None]
        )
        action = action[0]
        new_expert_state = jax.tree.map(lambda x: x[0], new_expert_state)
        new_obs, new_state, reward, new_done, _ = env.step(
            step_key, state, action, env_params
        )
        ret = jnp.where(done, ret, ret + reward)
        done_next = jnp.logical_or(done, new_done)
        # Freeze state on termination so post-terminal physics can't generate NaNs
        new_obs = jax.tree.map(lambda a, b: jnp.where(done, a, b), obs, new_obs)
        new_state = jax.tree.map(lambda a, b: jnp.where(done, a, b), state, new_state)
        new_expert_state = jax.tree.map(
            lambda a, b: jnp.where(done, a, b), expert_state, new_expert_state
        )
        return (new_obs, new_state, new_expert_state, ret, done_next), None

    max_steps = int(env_params.max_steps_in_episode)
    init = (obs, state, expert_state, jnp.float32(0.0), jnp.bool_(False))
    (_, _, _, ret, _), _ = jax.lax.scan(
        lambda c, _: body(c, step_key), init, xs=None, length=max_steps
    )
    return ret


def eval_env(env_name: str, n_seeds: int):
    env, env_params = make_env_by_name(env_name)
    expert = env.expert_policy
    keys = jax.random.split(jax.random.PRNGKey(0), n_seeds)
    rollout = jax.jit(lambda k: rollout_one(env, env_params, expert, k))
    returns = jnp.stack([rollout(k) for k in keys])
    returns = jax.device_get(returns)
    print(f"  [debug] per-seed returns: {returns.tolist()}")
    max_steps = int(env_params.max_steps_in_episode)
    return {
        "env": env_name,
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "per_step": float(returns.mean() / max_steps),
        "cv": float(returns.std() / (abs(returns.mean()) + 1e-9)),
        "max_steps": max_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None,
                        help="Single env; if omitted sweeps the default set")
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()

    if args.env is not None:
        envs = [args.env]
    else:
        envs = [
            "Plane", "Plane3DHeading", "Plane3DCircle", "Plane3DFigureEight",
            "CSTR", "FirstOrderSystem", "FourTank", "GlassFurnace", "Reactor",
        ]

    rows = []
    for name in envs:
        try:
            rows.append(eval_env(name, args.n_seeds))
            r = rows[-1]
            print(f"{r['env']:22s} mean={r['mean']:10.2f}  std={r['std']:8.2f}  "
                  f"cv={r['cv']:.3f}  per_step={r['per_step']:.3f}  "
                  f"min={r['min']:.2f}  max={r['max']:.2f}  steps={r['max_steps']}")
        except Exception as e:
            print(f"{name:22s} ERR {type(e).__name__}: {e}")

    if len(rows) > 1:
        print("\n--- ranked by CV (higher CV ⇒ expert more variable ⇒ more RL headroom) ---")
        for r in sorted(rows, key=lambda x: -x["cv"]):
            print(f"  {r['env']:22s} cv={r['cv']:.3f}  mean={r['mean']:.2f}")


if __name__ == "__main__":
    main()
