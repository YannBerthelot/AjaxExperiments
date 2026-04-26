"""EDGE design ablations.

Reads the best EDGE hyperparameters from the Optuna HPO for the given env,
overrides one or more SAC kwargs per the ``--variant`` choice, trains at
``--n-timesteps`` steps × ``--n-seeds`` seeds, and writes a JSON result
to ``ablation_results/<env>__<variant>.json``.

Variants correspond to §5.3 ablations in the paper:
    gating_argmax, mc_pretrain, decay_{0.05,0.30,0.50,1.00},
    eps_{0.1,0.25,0.75,0.9}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ABLATION_RESULTS = ROOT / "ablation_results"
CHECKPOINT_DIR = ROOT / "checkpoints"


# variant name -> dict of EDGE method_kwargs overrides applied on top of the
# tuned-EDGE config from the HPO study.
ABLATION_VARIANTS: dict = {
    "gating_argmax":   {"exploration_argmax": True},
    "mc_pretrain":     {"use_mc_critic_pretrain": True},
    "decay_0.05":      {"exploration_decay_frac": 0.05},
    "decay_0.30":      {"exploration_decay_frac": 0.30},
    "decay_0.50":      {"exploration_decay_frac": 0.50},
    "decay_1.00":      {"exploration_decay_frac": 1.00},
    "eps_0.1":         {"fixed_exploration_prob": 0.1},
    "eps_0.25":        {"fixed_exploration_prob": 0.25},
    "eps_0.75":        {"fixed_exploration_prob": 0.75},
    "eps_0.9":         {"fixed_exploration_prob": 0.9},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--variant", required=True, choices=list(ABLATION_VARIANTS))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--n-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-seeds", type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        from sac_hyperparam_search import build_agent_and_train  # type: ignore
    except ImportError:
        sys.stderr.write(
            "[ablation] sac_hyperparam_search.build_agent_and_train not found.\n"
        )
        return 2

    import optuna
    from optuna.trial import TrialState

    # Ablations always start from the tuned EDGE configuration.
    method = "edge"
    study_name = f"edge_hp_search_{args.env.lower()}"
    candidates = [
        ROOT / "hp_results" / args.env.lower() / "optuna.db",
        ROOT / "hp_results" / args.env.lower() / method / "optuna.db",
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
            f"[ablation] No study named {study_name!r} in "
            f"{[str(p) for p in candidates if p.exists()]}\n"
        )
        return 3
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not done:
        sys.stderr.write(
            f"[ablation] No COMPLETE trials in {study_name}\n"
        )
        return 4
    best = max(done, key=lambda t: t.value)

    override = ABLATION_VARIANTS[args.variant]
    print(
        f"[ablation] {args.env}/{args.variant}: best trial={best.number} "
        f"score={best.value:.2f}  override={override}"
    )

    seeds = list(range(args.n_seeds))
    ckpt_path = (
        CHECKPOINT_DIR / f"ablation__{args.env.lower()}__{args.variant}.pkl"
    )
    final_state, run_ids, tb_folder = build_agent_and_train(
        env_name=args.env,
        method=method,
        best_params=best.params,
        seed_list=seeds,
        n_timesteps=args.n_timesteps,
        checkpoint_path=str(ckpt_path),
        method_kwargs_override=override,
    )

    ABLATION_RESULTS.mkdir(parents=True, exist_ok=True)
    out = ABLATION_RESULTS / f"{args.env.lower()}__{args.variant}.json"
    payload = {
        "env": args.env,
        "variant": args.variant,
        "override": override,
        "n_timesteps": args.n_timesteps,
        "n_seeds": args.n_seeds,
        "best_hpo_trial": best.number,
        "best_hpo_score": best.value,
        "best_hpo_params": best.params,
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
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ablation] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
