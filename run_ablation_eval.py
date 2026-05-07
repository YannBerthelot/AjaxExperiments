"""EDGE-QA design ablations (paper §5.3).

Reads the best ``sac_quality_aware`` (EDGE-QA / LCB) hyperparameters from
the Optuna HPO for the given env, overrides one or more SAC kwargs per
the ``--variant`` choice, retrains at ``--n-timesteps`` steps ×
``--n-seeds`` seeds, and writes a JSON result to
``ablation_results/<env>__<variant>.json``.

Each variant flips a single design choice of EDGE-QA so we can attribute
the gain to (a) the LCB gate, (b) the obs+expert_state augmentation, or
(c) running observation normalisation. The baseline gating-mechanism
comparison vs ε-greedy is provided by the ``edge`` method's own HPO
in run_full_study.py — no separate ablation variant is needed for it.
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


# Variant name -> dict of SAC method_kwargs overrides applied on top of
# the tuned EDGE-QA config from the sac_quality_aware HPO study. Each
# variant flips exactly one design knob so the ablation cleanly
# attributes the change in score.
ABLATION_VARIANTS: dict = {
    # ──────── Gating mechanism ────────
    # Replace the LCB gate with hard argmax over critic-mean Q. Removes
    # the disagreement-based deferral to expert in unknown regions.
    "gating_argmax": {
        "exploration_lcb": False,
        "exploration_thompson": False,
        "exploration_argmax": True,
    },
    # ──────── Preprocessing ────────
    # Drop the obs+expert_state augmentation. Tests whether the
    # non-Markov fix (V3 critic-fit diagnostic) is the load-bearing
    # piece on stateful-expert tasks.
    "no_state_aug": {
        "augment_obs_with_expert_state": False,
    },
    # Drop the agent-side running observation normalisation. Tests
    # whether the always-on preprocessing is really required when
    # state-aug is on (integrator dims std ~10^4).
    "no_obs_norm": {
        "normalize_obs_running": False,
    },
    # ──────── Critic target / bootstrap ────────
    # Add IBRL-style max(Q_policy, Q_expert) TD target on top of the
    # LCB gate. Decomposes IBRL's contribution from its argmax action
    # selection — pairs with the IBRL method-level baseline.
    "bootstrap_argmax": {
        "ibrl_bootstrap": True,
    },
    # Apply the LCB gate at the value bootstrap step too (not only at
    # action selection): TD target = Q(s', a_lcb_gated'). Tests whether
    # consistent LCB-everywhere is strictly better than LCB-on-action-
    # selection-only. If yes, this becomes part of edge-qa's default.
    "bootstrap_lcb_gated": {
        "lcb_gated_bootstrap": True,
    },
    # ──────── Expert quality ────────
    # Replace the PID expert with a stateful random-action policy that
    # mirrors the PID expert's state shape. Same dim of obs+expert_state,
    # uninformative actions and state. Tests "the expert provides signal,
    # not just structured noise." Marker; expert swap is plumbed via
    # build_agent_and_train.expert_override below.
    "random_expert": {"__expert_override": "random_scrambled"},
    # ──────── Off-policy correctness ────────
    # Store the policy's action in the buffer (instead of the actually-
    # executed action). Misspecified Q learning when the gate selected
    # the expert. Justifies the design choice of storing executed
    # actions.
    "store_policy_action": {
        "store_policy_action": True,
    },
    # ──────── Orthogonal augmentation: tt-prefill ────────
    # Add jsrl-tt-style training-time expert prefill on top of edge-qa
    # (LCB gate stays on). 100k expert transitions per seed are written
    # into the replay buffer before any learner rollout. Tests whether
    # the buffer-prefill mechanism (which makes our jsrl-tt baseline
    # surprisingly competitive on clean experts) is additive on top of
    # the LCB gate, or substitutes for it.
    "expert_prefill": {
        "expert_buffer_n_steps": 100_000,
    },
    # ──────── Compositionality with other expert-using mechanisms ────────
    # JSRL-curriculum on top of EDGE-QA: per-episode handoff at decaying
    # in-episode horizon H_t while EDGE-QA's LCB gate handles the
    # post-handoff segment. Tests whether the curriculum's early-episode
    # expert coverage composes with EDGE-QA's late-episode quality-aware
    # gating, or whether they redundantly compete.
    "edge_qa_with_curriculum": {
        "jsrl_curriculum": True,
        "jsrl_decay_frac": 0.5,
        "__needs_episode_length": True,
    },
    # ──────── Pessimism axis (discrete) ────────
    # Disable pessimism (β = 0) while keeping the softmax gate over
    # min-ensemble Q. The HPO sweep over β ∈ [0.1, 5.0] already shows
    # the optimum is task-dependent; this discrete cell isolates the
    # contribution of β > 0 from the contribution of the gate form itself.
    "no_pessimism": {
        "lcb_beta_init": 0.0,
    },
    # ──────── Ensemble size (one-off) ────────
    # HPO picks K ∈ {2, 4}; this probe pushes K to 10 to test whether the
    # disagreement signal benefits from a larger ensemble. Expensive (K=10
    # roughly 2.5× the per-step critic cost vs K=4); kept as a lowest-
    # priority variant in the queue.
    "large_ensemble": {
        "num_critics": 10,
    },
    # ──────── Bandit view: literal Thompson sampling vs LCB gate ────────
    # Replace EDGE's LCB-softmax gate with literal Gaussian-parametric
    # Thompson sampling on the (μ, σ) of the small critic ensemble:
    # q̃_e ∼ N(μ_e, (T σ_e)^2), q̃_p ∼ N(μ_p, (T σ_p)^2), pick whichever
    # is larger. Tests the paper's claim that the under-dispersion of
    # small ensembles (N ∈ {2, 4}) makes literal TS collapse toward
    # greedy and that EDGE's fixed-bandwidth Gaussian approximation
    # (with τ compensating for under-dispersion) actually helps.
    "literal_thompson": {
        "exploration_lcb": False,
        "exploration_thompson": True,
        "exploration_argmax": False,
    },
    # ──────── 2x2 corner: (argmax gate) x (LCB scoring) ────────
    # Closes the (argmax, LCB) cell of the gating-form x scoring-rule
    # 2x2. Keeps EDGE's LCB scoring (mu - beta*sigma) and replaces the
    # softmax sampling with a deterministic argmax. Together with
    # gating_argmax (argmax + min-Q), no_pessimism (softmax + min-Q,
    # via beta=0), and EDGE default (softmax + LCB), this fills the
    # remaining corner.
    "argmax_lcb": {
        "exploration_argmax_lcb": True,
    },
    # ──────── Mechanistic: literal TS + 10-critic ensemble ────────
    # Literal Gaussian-parametric Thompson sampling, but with K=10
    # critics instead of EDGE's HPO-tuned K in {2,4}. The bandit-view
    # claim is that small ensembles under-disperse so literal TS
    # collapses to greedy; if true, K=10 should narrow or close the
    # gap to EDGE's softmax-LCB gate. Direct mechanistic test of the
    # under-dispersion claim made in §4 / discussed in app:ablation.
    "literal_thompson_k10": {
        "exploration_lcb": False,
        "exploration_thompson": True,
        "exploration_argmax": False,
        "num_critics": 10,
    },
    # ──────── Diagnostic: empty override (EDGE default config) ────────
    # Re-runs EDGE at the tuned hyperparameters with no method_kwarg
    # changes. Used to collect the p_expert_max telemetry (Coverage
    # Lemma diagnostic) without altering the gate. The result IQM
    # should match the existing EDGE proxy bar to within seed noise.
    "edge_default_probe": {},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--variant", required=True, choices=list(ABLATION_VARIANTS))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--n-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-seeds", type=int, default=50)
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

    # Ablations always start from the tuned EDGE-QA (sac_quality_aware)
    # configuration. Each variant flips one knob in method_kwargs.
    method = "sac_quality_aware"
    study_name = f"sac_quality_aware_hp_search_{args.env.lower()}"
    candidates = [
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

    override = dict(ABLATION_VARIANTS[args.variant])
    print(
        f"[ablation] {args.env}/{args.variant}: best trial={best.number} "
        f"score={best.value:.2f}  override={override}"
    )

    # ``edge_qa_with_curriculum`` needs the env's episode length injected
    # at dispatch time (it's not in ABLATION_VARIANTS because it's
    # env-specific). Pop the marker and replace with the actual int.
    if override.pop("__needs_episode_length", False):
        from sac_hyperparam_search import _ENV_EPISODE_LENGTH
        ep_len = _ENV_EPISODE_LENGTH.get(args.env, 1000)
        override["jsrl_episode_length"] = int(ep_len)
        print(f"[ablation] injected jsrl_episode_length={ep_len} from env")

    # ``random_expert`` is special: instead of overriding a method_kwarg,
    # it swaps the expert_policy passed to build_agent_and_train. We keep
    # the marker key in override so the JSON record reflects the variant
    # but pull it out before the kwargs reach SAC.
    expert_override = None
    if override.pop("__expert_override", None) == "random_scrambled":
        from random_expert import RandomScrambledExpert
        from envs import make_env_by_name
        from sac_hyperparam_search import _load_expert_policy
        from ajax.environments.utils import get_action_dim
        _env, _params = make_env_by_name(args.env)
        _template_expert = _load_expert_policy(args.env, _env)
        _action_dim = get_action_dim(_env, _params)
        expert_override = RandomScrambledExpert(_template_expert, _action_dim)
        print(f"[ablation] random_expert: swapped expert with RandomScrambledExpert")

    seeds = list(range(args.n_seeds))
    # Skip the ablation-checkpoint pickle: build_agent_and_train calls
    # ajax.save_checkpoint -> jax.tree.map(np.asarray, state) at the end,
    # which trips a TracerArrayConversionError on CheetahRun (one leaf of
    # the agent state escapes the vmap as a BatchTracer; harmless for the
    # in-process result but unsafe to materialize). The checkpoint is only
    # used for cross-process resume, which the ablation pipeline never
    # does (single 300k-step run), so passing checkpoint_path=None is the
    # least-invasive fix; the per-seed metrics we report come from
    # read_final_metric over TB events, not the checkpoint.
    final_state, run_ids, tb_folder = build_agent_and_train(
        env_name=args.env,
        method=method,
        best_params=best.params,
        seed_list=seeds,
        n_timesteps=args.n_timesteps,
        checkpoint_path=None,
        method_kwargs_override=override,
        expert_override=expert_override,
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
