# EDGE: Expert-Decay-Guided Exploration

This repository contains the experiment code and the LaTeX sources for the
paper *Expert-Decay-Guided Exploration*. The proposed method (`edge-qa`) is a
SAC variant with critic-disagreement-aware (LCB) gating between a learned
policy and a known-good expert; the paper compares it to vanilla SAC and to
the ε-greedy expert-mixing baseline (`edge-eps`) across five continuous
control tasks.

## Layout

| File / dir                  | Purpose                                                                                                                          |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `run_full_study.py`         | Top-level orchestrator. Schedules the per-`(env, method)` HPO + final-evaluation + degradation work units across available GPUs. |
| `sac_hyperparam_search.py`  | Single-method Optuna HPO + training. Invoked per-method by `run_full_study.py`; can also be run standalone.                      |
| `run_final_eval.py`         | Trains a single seed-bank for an `(env, method)` at the HPO-best hyperparameters, used to produce the paper's main-result table. |
| `run_degradation_eval.py`   | Sweeps an expert-noise grid for an `(env, method)` and trains the agent at each level, producing the degradation curves.         |
| `run_ablation_eval.py`      | Trains the §5.3 EDGE ablation variants (gating mode, MC pretrain, decay schedules, ε floors).                                    |
| `envs.py`                   | Single registry of every environment used in the paper, exposing the `(env, env_params, expert, slug)` tuple consumed by every other script. |
| `eval_pid_expert.py`        | Stand-alone evaluation of the PID expert on the TargetGym tasks. Used as the expert-only baseline.                               |
| `eval_cpg_expert.py`        | Stand-alone evaluation of the open-loop CPG expert on the locomotion tasks (`CheetahRun`, `BarkourJoystick`).                    |
| `tune_cpg.py`               | One-off optimisation of the CPG expert's frequency / amplitudes / phases per locomotion task. Run before the locomotion HPO.      |
| `paper/`                    | LaTeX sources (`paper.tex`, `appendix_hyperparam_search.tex`, NeurIPS style file). Build artefacts under `paper/build/` are gitignored. |
| `pyproject.toml`, `poetry.lock` | Pinned Python environment.                                                                                                   |
| `tests/test_env_registry.py` | Sanity-check that every env declared in `envs.py` builds, exposes the expected obs/action shapes, and runs an expert step.       |

The Ajax RL library (the algorithms and training core) lives in a separate
repository (`Ajax/`); this repo only contains the experiment-level code.

## Reproducing the paper

```bash
# 1. Install the pinned Python environment.
poetry install

# 2. (Locomotion only) tune the open-loop CPG expert once per locomotion env.
poetry run python tune_cpg.py --env CheetahRun
poetry run python tune_cpg.py --env BarkourJoystick

# 3. Run the full study (HPO + final eval + degradation across all envs/methods).
#    Distributes work across visible GPUs; resumable.
poetry run python run_full_study.py --gpus 0 1 2

# 4. Inspect status and partial results without re-running.
poetry run python run_full_study.py --status
```

Per-method HPO can also be invoked directly:

```bash
poetry run python sac_hyperparam_search.py \
    --env Plane3DCircle --method sac_quality_aware \
    --gpus 0 --trials 30
```

## Methods compared in the paper

| `--method`            | Label in paper | Description                                                              |
| --------------------- | -------------- | ------------------------------------------------------------------------ |
| `sac`                 | SAC            | Vanilla SAC baseline (no expert).                                        |
| `edge`                | EDGE-ε         | SAC + ε-greedy expert mixing with a decay schedule (gating ablation).    |
| `sac_quality_aware`   | EDGE-QA        | SAC + LCB expert gating with critic-disagreement σ + augmented obs (proposed). |

All methods are run with agent-side running observation normalisation
(treated as a fixed preprocessing step rather than an algorithmic choice;
see paper §5).

## Outputs (gitignored)

Running the scripts populates several local directories which are not
checked in:

- `hp_results/<env>/<method>/` — Optuna study, per-trial logs, TensorBoard.
- `final_results/` — final-eval JSONs.
- `degradation_results/` — degradation-eval JSONs.
- `ablation_results/` — ablation JSONs.
- `study_logs/` — per-process subprocess logs from `run_full_study.py`.
