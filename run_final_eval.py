"""Final evaluation: train with best HPO hyperparameters at extended budget.

Reads the best-found hyperparameters from the Optuna study at
``hp_results/<env>/[<method>/]optuna.db``, trains ``--n-seeds`` seeds for
``--n-timesteps`` environment steps, saves a checkpoint (for later
2M->3M extension via ``ajax.checkpoint``), and writes a results JSON to
``final_results/<env>__<method>.json``.

This script depends on the agent-setup logic in ``sac_hyperparam_search.py``
(specifically the per-method dispatch that decides which SAC flags to set).
Rather than duplicate that logic here, this runner imports a small helper
``build_agent_and_train`` from that file. If the helper is not yet exported,
the run aborts with a clear error describing what needs to be exposed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
FINAL_RESULTS = ROOT / "final_results"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--n-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--n-chunks", type=int, default=1,
                        help="Split n-timesteps into N sequential train() calls "
                             "with checkpoint-resume between. Used to avoid OOM "
                             "on per-step metric tensors of shape [n_seeds, n_steps].")
    args = parser.parse_args()
    if args.n_timesteps % args.n_chunks != 0:
        sys.stderr.write(f"[final_eval] --n-timesteps ({args.n_timesteps}) must be "
                         f"divisible by --n-chunks ({args.n_chunks})\n")
        return 5

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        from sac_hyperparam_search import build_agent_and_train  # type: ignore
    except ImportError:
        sys.stderr.write(
            "[final_eval] sac_hyperparam_search.build_agent_and_train not found.\n"
            "Before run_full_study can dispatch final-eval work, expose a helper\n"
            "in sac_hyperparam_search.py with the signature:\n"
            "    build_agent_and_train(env_name, method, best_params, seed_list,\n"
            "                          n_timesteps, checkpoint_path=None)\n"
            "returning the final agent state. This is the same code the HPO\n"
            "worker (_train_worker) uses, refactored to take explicit params\n"
            "rather than reading them from the CLI.\n"
        )
        return 2

    # Read best hparams from the HPO study
    import optuna
    from optuna.trial import TrialState

    study_name = (
        f"sac_hp_search_{args.env.lower()}" if args.method == "sac"
        else f"{args.method}_hp_search_{args.env.lower()}"
    )
    # Different methods live in different DBs (sac in hp_results/<env>/optuna.db,
    # others in hp_results/<env>/<method>/optuna.db). Probe both and pick the
    # one that actually contains our study; falling back to "first existing"
    # loads the wrong DB and raises KeyError deep inside Optuna.
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
            f"[final_eval] No study named {study_name!r} in "
            f"{[str(p) for p in candidates if p.exists()]}\n"
        )
        return 3
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not done:
        sys.stderr.write(f"[final_eval] No COMPLETE trials in {study_name}\n")
        return 4
    best = max(done, key=lambda t: t.value)
    best_params = best.params

    print(f"[final_eval] {args.env}/{args.method}: best trial={best.number} "
          f"score={best.value:.2f}")

    # Build + train
    from ajax.checkpoint import save_checkpoint

    seeds = list(range(args.n_seeds))
    ckpt_path = CHECKPOINT_DIR / f"{args.env.lower()}__{args.method}.pkl"
    chunk_steps = args.n_timesteps // args.n_chunks
    run_ids: list = []
    tb_folder: str = ""
    for chunk in range(args.n_chunks):
        print(f"[final_eval] chunk {chunk+1}/{args.n_chunks}: "
              f"training {chunk_steps} steps (resume={chunk > 0})")
        final_state, run_ids, tb_folder = build_agent_and_train(
            env_name=args.env,
            method=args.method,
            best_params=best_params,
            seed_list=seeds,
            n_timesteps=chunk_steps,
            checkpoint_path=str(ckpt_path),
        )
        save_checkpoint(final_state, str(ckpt_path))

    # Write a summary JSON (mean / std / per-seed returns over last 200k steps)
    # The helper is expected to include these metrics in the returned state
    # under `final_state.eval_metrics`. If not present, write a stub.
    out = FINAL_RESULTS / f"{args.env.lower()}__{args.method}.json"
    FINAL_RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": args.env,
        "method": args.method,
        "n_timesteps": args.n_timesteps,
        "n_seeds": args.n_seeds,
        "best_hpo_trial": best.number,
        "best_hpo_score": best.value,
        "best_hpo_params": best_params,
    }
    # Read the actual final-eval metric from on-disk TB events. Use the same
    # last-200k window as the HPO ``read_final_metric`` call so the FINAL score
    # is directly comparable to the HPO best score.
    try:
        from sac_hyperparam_search import read_final_metric  # type: ignore
        final_window = 200_000
        final_metric = read_final_metric(
            tb_folder, run_ids, window_steps=final_window
        )
        payload["final_metric"] = final_metric
        payload["final_metric_window_steps"] = final_window
    except Exception as e:
        payload["final_metric"] = None
        payload["final_metric_error"] = str(e)
    if args.n_chunks > 1:
        payload["n_chunks"] = args.n_chunks
        payload["chunk_steps"] = chunk_steps
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[final_eval] wrote {out}")
    return 0


def jax_tree_to_py(tree):
    import numpy as np
    import jax
    return jax.tree.map(lambda x: (np.asarray(x).tolist()
                                   if hasattr(x, "shape") else x),
                        tree)


if __name__ == "__main__":
    sys.exit(main())
