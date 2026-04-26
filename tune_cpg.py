"""Tune the CheetahRun (or WalkerRun) CPG via scipy differential evolution.

Search space (13 dims):
    frequency ∈ [0.5, 5.0] Hz
    amplitudes[6] ∈ [0, 1]
    phases[6] ∈ [0, 2π]

Objective = mean return over N_SEEDS rollouts of N_STEPS steps. Maximise.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_disable_hlo_passes=layout_assignment")

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution

from eval_cpg_expert import CPGParams, cpg_action, rollout


def unpack(x: np.ndarray, n_act: int) -> CPGParams:
    return CPGParams(
        frequency=jnp.float32(x[0]),
        amplitudes=jnp.asarray(x[1:1 + n_act], dtype=jnp.float32),
        phases=jnp.asarray(x[1 + n_act:1 + 2 * n_act], dtype=jnp.float32),
    )


def build_objective(env, n_seeds: int, n_steps: int):
    keys = jax.random.split(jax.random.PRNGKey(0), n_seeds)

    def mean_return(cpg: CPGParams) -> float:
        rets = jnp.stack([rollout(env, cpg, n_steps, k) for k in keys])
        return float(rets.mean())

    # JIT the rollout once so each eval is fast.
    jit_single = jax.jit(lambda cpg, k: rollout(env, cpg, n_steps, k))

    def mean_return_jit(cpg: CPGParams) -> tuple[float, float]:
        rets = jnp.stack([jit_single(cpg, k) for k in keys])
        rets = jax.device_get(rets)
        return float(rets.mean()), float(rets.std())

    return mean_return_jit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CheetahRun")
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from mujoco_playground import registry
    env = registry.load(args.env, config_overrides={"impl": "jax"})
    n_act = env.action_size

    print(f"Tuning CPG for {args.env} ({n_act} actuators), {args.n_seeds} seeds x {args.n_steps} steps, "
          f"popsize={args.popsize}, maxiter={args.maxiter}")
    objective = build_objective(env, args.n_seeds, args.n_steps)

    bounds = [(0.5, 5.0)]  # frequency
    bounds += [(0.0, 1.0)] * n_act  # amplitudes
    bounds += [(0.0, 2 * np.pi)] * n_act  # phases

    best = {"value": -np.inf, "x": None, "std": None}
    t0 = time.time()
    eval_count = [0]

    def de_obj(x: np.ndarray) -> float:
        cpg = unpack(x, n_act)
        mean_r, std_r = objective(cpg)
        eval_count[0] += 1
        if mean_r > best["value"]:
            best["value"] = mean_r
            best["x"] = x.copy()
            best["std"] = std_r
            print(f"  [{eval_count[0]:4d}] new best mean={mean_r:7.2f} std={std_r:6.2f}  "
                  f"f={x[0]:.2f}  elapsed={time.time()-t0:.1f}s")
        return -mean_r  # minimize

    result = differential_evolution(
        de_obj,
        bounds,
        popsize=args.popsize,
        maxiter=args.maxiter,
        polish=False,
        seed=0,
        tol=1e-3,
        updating="immediate",
        workers=1,
        vectorized=False,
        disp=True,
    )

    print(f"\nDone. evals={eval_count[0]} wall={time.time()-t0:.1f}s")
    x = best["x"]
    print(f"Best mean={best['value']:.2f} std={best['std']:.2f}")
    amps = x[1:1 + n_act]
    phases = x[1 + n_act:1 + 2 * n_act]
    print(f"  frequency = {x[0]:.4f}")
    print(f"  amplitudes = {amps.tolist()}")
    print(f"  phases = {phases.tolist()}")

    payload = {
        "env": args.env,
        "mean": best["value"],
        "std": best["std"],
        "frequency": float(x[0]),
        "amplitudes": amps.tolist(),
        "phases": phases.tolist(),
        "n_seeds_used": args.n_seeds,
        "n_steps_used": args.n_steps,
    }
    out_path = args.out or f"cpg_tuned_{args.env.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
