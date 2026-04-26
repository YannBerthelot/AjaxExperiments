"""Expert-degradation sensitivity study.

Same setup as run_final_eval.py, but the expert is perturbed via
``target_gym.experts.degradation.degrade_expert(expert, sigma, key)``
before training. Writes to ``degradation_results/<env>__<method>__sigmaS.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEGRADATION_RESULTS = ROOT / "degradation_results"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--n-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--degradation-seed", type=int, default=0,
                        help="Seed for expert degradation noise (single draw)")
    parser.add_argument("--n-chunks", type=int, default=1,
                        help="Split n-timesteps into N sequential train() calls "
                             "with checkpoint-resume between. Avoids OOM on the "
                             "[n_seeds, n_steps] per-step metric tensor.")
    args = parser.parse_args()
    if args.n_timesteps % args.n_chunks != 0:
        sys.stderr.write(f"[degradation] --n-timesteps ({args.n_timesteps}) must "
                         f"be divisible by --n-chunks ({args.n_chunks})\n")
        return 5

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        from sac_hyperparam_search import build_agent_and_train  # type: ignore
    except ImportError:
        sys.stderr.write(
            "[degradation] sac_hyperparam_search.build_agent_and_train not found.\n"
            "Expose it with an extra kwarg:\n"
            "    build_agent_and_train(..., expert_override=None)\n"
            "where expert_override is a FunctionalExpertPolicy that replaces\n"
            "the task's default expert.\n"
        )
        return 2

    import jax
    import optuna
    from optuna.trial import TrialState
    from envs import make_env_by_name  # type: ignore
    from target_gym.experts.degradation import degrade_expert

    # Best hparams
    study_name = (
        f"sac_hp_search_{args.env.lower()}" if args.method == "sac"
        else f"{args.method}_hp_search_{args.env.lower()}"
    )
    candidates = [
        ROOT / "hp_results" / args.env.lower() / "optuna.db",
        ROOT / "hp_results" / args.env.lower() / args.method / "optuna.db",
    ]
    study = None
    db_path = None
    for p in candidates:
        if not p.exists():
            continue
        try:
            study = optuna.load_study(
                study_name=study_name, storage=f"sqlite:///{p}"
            )
            db_path = p
            break
        except KeyError:
            continue
    if study is None:
        sys.stderr.write(
            f"[degradation] No study named {study_name!r} in "
            f"{[str(p) for p in candidates if p.exists()]}\n"
        )
        return 3
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not done:
        sys.stderr.write(f"[degradation] No COMPLETE trials\n")
        return 4
    best = max(done, key=lambda t: t.value)
    best_params = best.params

    # Build anchor expert then degrade it
    env, _env_params = make_env_by_name(args.env)
    anchor_expert = env.expert_policy
    degraded = degrade_expert(
        anchor_expert, args.sigma, jax.random.PRNGKey(args.degradation_seed)
    )

    # Measure degraded expert's own return as the x-axis reference
    # (reuse eval_pid_expert if available; else skip and leave blank)
    degraded_return = None
    try:
        from eval_pid_expert import eval_env as _eval_pid_env  # type: ignore
        # NB: eval_pid_expert pulls env.expert_policy; we'd need a variant that
        # accepts a handed-in expert. Skip for now; this is a nice-to-have.
    except Exception:
        pass

    print(f"[degradation] {args.env}/{args.method} sigma={args.sigma}: "
          f"best trial={best.number} score={best.value:.2f}")

    seeds = list(range(args.n_seeds))
    CHECKPOINT_DIR = ROOT / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / (
        f"degrade__{args.env.lower()}__{args.method}__sigma{args.sigma}.pkl"
    )
    chunk_steps = args.n_timesteps // args.n_chunks
    run_ids: list = []
    tb_folder: str = ""
    for chunk in range(args.n_chunks):
        print(f"[degradation] chunk {chunk+1}/{args.n_chunks}: "
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

    out = DEGRADATION_RESULTS / (
        f"{args.env.lower()}__{args.method}__sigma{args.sigma}.json"
    )
    DEGRADATION_RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": args.env,
        "method": args.method,
        "sigma": args.sigma,
        "n_timesteps": args.n_timesteps,
        "n_seeds": args.n_seeds,
        "best_hpo_trial": best.number,
        "degraded_expert_return": degraded_return,
    }
    try:
        from sac_hyperparam_search import read_final_metric  # type: ignore
        final_window = 200_000
        payload["final_metric"] = read_final_metric(
            tb_folder, run_ids, window_steps=final_window
        )
        payload["final_metric_window_steps"] = final_window
    except Exception as e:
        payload["final_metric"] = None
        payload["final_metric_error"] = str(e)
    if args.n_chunks > 1:
        payload["n_chunks"] = args.n_chunks
        payload["chunk_steps"] = chunk_steps
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[degradation] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
