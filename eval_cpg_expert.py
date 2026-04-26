"""Evaluate a simple sinusoidal CPG 'expert' on mujoco_playground locomotion.

The CPG is open-loop: each actuator is driven by a sine wave. Parameters are
per-actuator amplitude + phase and a single frequency. No feedback.

This is the locomotion analogue of eval_pid_expert.py — a simple hand-coded
baseline used to estimate how much headroom EDGE could exploit.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_disable_hlo_passes=layout_assignment")

import argparse
from typing import NamedTuple
import jax
import jax.numpy as jnp


class CPGParams(NamedTuple):
    frequency: jnp.ndarray
    amplitudes: jnp.ndarray  # (n_actuators,)
    phases: jnp.ndarray  # (n_actuators,) radians


def walker_cpg() -> CPGParams:
    # WalkerRun actuators: hip, knee, ankle × (right, left).
    amp = jnp.array([0.7, 0.9, 0.5, 0.7, 0.9, 0.5])
    phase = jnp.array([0.0, 1.5, 2.5, jnp.pi, jnp.pi + 1.5, jnp.pi + 2.5])
    return CPGParams(frequency=jnp.float32(1.5), amplitudes=amp, phases=phase)


def cheetah_cpg() -> CPGParams:
    # CheetahRun actuators: bthigh, bshin, bfoot, fthigh, fshin, ffoot.
    amp = jnp.array([0.8, 0.8, 0.6, 0.8, 0.8, 0.6])
    phase = jnp.array([0.0, 1.0, 2.0, jnp.pi / 2, jnp.pi / 2 + 1.0, jnp.pi / 2 + 2.0])
    return CPGParams(frequency=jnp.float32(2.5), amplitudes=amp, phases=phase)


def barkour_cpg() -> CPGParams:
    # BarkourJoystick actuators (12): [abduction, hip, knee] x [FL, HL, FR, HR].
    # Trot: FL+HR and HL+FR in anti-phase diagonals. Knee leads hip by pi/4.
    amp_abd, amp_hip, amp_knee = 0.0, 0.3, 0.4
    amps = jnp.array([
        amp_abd, amp_hip, amp_knee,
        amp_abd, amp_hip, amp_knee,
        amp_abd, amp_hip, amp_knee,
        amp_abd, amp_hip, amp_knee,
    ])
    pi = float(jnp.pi)
    knee_lead = pi / 4.0
    phases = jnp.array([
        0.0, 0.0, knee_lead,          # front-left (pair A)
        0.0, pi,  pi + knee_lead,     # hind-left  (pair B, antiphase)
        0.0, pi,  pi + knee_lead,     # front-right (pair B)
        0.0, 0.0, knee_lead,          # hind-right (pair A)
    ])
    return CPGParams(frequency=jnp.float32(2.0), amplitudes=amps, phases=phases)


CPG_REGISTRY = {
    "WalkerRun": walker_cpg,
    "WalkerWalk": walker_cpg,
    "CheetahRun": cheetah_cpg,
    "BarkourJoystick": barkour_cpg,
}


def cpg_action(cpg: CPGParams, t: jnp.ndarray) -> jnp.ndarray:
    return cpg.amplitudes * jnp.sin(2.0 * jnp.pi * cpg.frequency * t + cpg.phases)


def rollout(env, cpg: CPGParams, n_steps: int, key):
    state = env.reset(key)

    def body(carry, step_idx):
        state, ret, done = carry
        t = step_idx * env.dt
        action = cpg_action(cpg, t)
        new_state = env.step(state, action)
        r = new_state.reward
        new_done = new_state.done.astype(jnp.bool_)
        ret = jnp.where(done, ret, ret + r)
        done_next = done | new_done
        # Freeze state on done so post-terminal physics can't generate NaNs
        new_state = jax.tree.map(
            lambda a, b: jnp.where(done, a, b), state, new_state
        )
        return (new_state, ret, done_next), None

    init = (state, jnp.float32(0.0), jnp.bool_(False))
    (_, ret, _), _ = jax.lax.scan(body, init, jnp.arange(n_steps))
    return ret


def eval_env(env_name: str, n_seeds: int, n_steps: int):
    from mujoco_playground import registry
    env = registry.load(env_name, config_overrides={"impl": "jax"})
    cpg = CPG_REGISTRY[env_name]()
    keys = jax.random.split(jax.random.PRNGKey(0), n_seeds)
    jit_rollout = jax.jit(lambda k: rollout(env, cpg, n_steps, k))
    returns = jnp.stack([jit_rollout(k) for k in keys])
    returns = jax.device_get(returns)
    return {
        "env": env_name,
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "per_step": float(returns.mean() / n_steps),
        "cv": float(returns.std() / (abs(returns.mean()) + 1e-9)),
        "n_steps": n_steps,
        "returns": returns.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="WalkerRun", choices=list(CPG_REGISTRY))
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=1000)
    args = parser.parse_args()
    r = eval_env(args.env, args.n_seeds, args.n_steps)
    print(f"env={r['env']} n_seeds={args.n_seeds} n_steps={r['n_steps']}")
    print(f"  mean={r['mean']:.2f} std={r['std']:.2f} cv={r['cv']:.3f} "
          f"per_step={r['per_step']:.3f} min={r['min']:.2f} max={r['max']:.2f}")
    print(f"  returns: {[round(x, 1) for x in r['returns']]}")


if __name__ == "__main__":
    main()
