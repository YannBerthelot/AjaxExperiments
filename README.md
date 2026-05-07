# When (and How) to Trust the Expert

Experiment code and LaTeX sources for the paper *When (and How) to
Trust the Expert: Diagnosing Query-Time Expert-Guided Reinforcement
Learning*. The repository runs five query-time-controller RL methods
on a shared SAC backbone with matched HPO and a common evaluation
protocol, then sweeps each (env, method) cell over expert-degradation
intensities. The paper contributes the harmonized comparison, three
mechanism-specific failure modes (F1-F3) it surfaces, and EDGE, a
softmax-over-ensemble-LCB design point used to demonstrate that both
axes the taxonomy points to (gate form, scoring rule) are individually
exploitable.

The Ajax RL library (algorithms and training core) lives in a separate
repo (`Ajax/`); this repo only contains the experiment-level glue.

## Methods compared

| `--method`         | Label in paper      | Description                                                     |
| ------------------ | ------------------- | --------------------------------------------------------------- |
| `sac`              | SAC                 | Vanilla SAC, no expert. Anchor for the per-task delta.          |
| `ibrl`             | IBRL                | Argmax over actor-vs-expert mean-Q; bootstraps off the argmax.  |
| `jsrl`             | JSRL                | Episode-horizon handoff, fixed schedule (no warm-start buffer). |
| `jsrl_curriculum`  | JSRL-curriculum     | Episode-horizon handoff with a curriculum on the handoff step.  |
| `residual`         | Residual SAC        | Actor outputs an additive correction on top of the expert.      |
| `sac_quality_aware`| EDGE                | Softmax over ensemble lower-confidence-bound critic values.     |
| `pid_policy`       | Expert-only         | Stateful PID / CPG expert run with no learning. Reference line. |

All learning methods share agent-side running observation
normalisation; see paper §5 for the protocol details.

## Environments

Registered in [envs.py](envs.py): `FourTank`, `GlassFurnace`,
`Reactor`, `Plane3DCircle`, `Plane3DFigureEight`, `CheetahRun`,
`BarkourJoystick`. Each entry exposes
`(env, env_params, expert, slug)` consumed by every script below.

## Reproducing the paper

```bash
# 1. Pinned environment.
poetry install

# 2. (Locomotion only) tune the open-loop CPG expert once per env.
poetry run python tune_cpg.py --env CheetahRun
poetry run python tune_cpg.py --env BarkourJoystick

# 3. Full study: HPO -> final eval -> degradation sweep, all
#    (env, method) cells, distributed across visible GPUs, resumable.
./run_safely.sh poetry run python run_full_study.py --gpus 0 1 2

# 4. Status of scheduled / running / completed work units.
poetry run python run_full_study.py --status

# 5. Build the paper tables and figures from the populated results.
poetry run python make_main_table.py        # tab:main_perenv
poetry run python plot_sweep.py --paper     # per-env headline + ablation + degradation
poetry run python run_directional_tests.py  # F1, F2, F3 directional p-values
```

`run_full_study.py` is the canonical entry point; the per-stage
scripts below are only useful for re-running individual cells.

## Per-stage entry points

| Stage              | Script                                | Output                                              |
| ------------------ | ------------------------------------- | --------------------------------------------------- |
| HPO                | `sac_hyperparam_search.py`            | `hp_results/<env>/<method>/optuna.db` + trial JSONs |
| Final eval         | `run_final_eval.py`                   | `final_results/<env>__<method>.json`                |
| Degradation sweep  | `run_degradation_eval.py`             | `degradation_results/<env>__<method>__sigma*.json`  |
| EDGE design ablation | `run_ablation_eval.py`              | `ablation_results/<env>__<variant>.json`            |
| Directional tests  | `run_directional_tests.py`            | F1 / F2 / F3 p-values printed; consumed by paper.tex |

Single-cell HPO example:

```bash
./run_safely.sh poetry run python sac_hyperparam_search.py \
    --env Plane3DCircle --method sac_quality_aware \
    --gpus 0 --trials 30 \
    --search-timesteps 300000 --search-seeds 20 \
    --auto-phase2 --auto-phase3 --top-k 5 --no-prune
```

The exhaustive list of canonical invocations (with the conventions
around `run_safely.sh`, log locations, and how to kill a detached run)
lives in [COMMANDS.md](COMMANDS.md).

## EDGE design ablation

`run_ablation_eval.py` populates the (gate form, scoring rule) 2x2
plus a few mechanistic probes:

| `--variant`              | Gate form | Scoring rule    | Notes                                             |
| ------------------------ | --------- | --------------- | ------------------------------------------------- |
| (default EDGE)           | softmax   | LCB (mu - beta sigma) | Reference; same hparams as the main table.   |
| `gating_argmax`          | argmax    | min-Q           | Old gating-form ablation.                         |
| `argmax_lcb`             | argmax    | LCB             | Closes the (argmax, LCB) corner.                  |
| `no_pessimism`           | softmax   | mean-Q (beta=0) | Pessimism off, gating preserved.                  |
| `literal_thompson`       | TS sample | (mu, sigma)     | Tests under-dispersion claim of small ensembles.  |
| `literal_thompson_k10`   | TS sample | (mu, sigma)     | Same with K=10 critics.                           |
| `large_ensemble`         | softmax   | LCB             | K=10 critics; isolates ensemble size from gate.   |
| `edge_default_probe`     | softmax   | LCB             | Empty override; collects p_expert_max telemetry.  |

## Outputs (gitignored)

- `hp_results/<env>/<method>/` -- Optuna study, per-trial logs,
  TensorBoard. The `.gitignore` whitelists `optuna.db` and the
  `trial_*_params.json` / `trial_*_phase3_result.json` summaries so a
  fresh clone can pick up best-trial hparams without the TB events.
- `final_results/` -- final-eval JSONs.
- `degradation_results/` -- degradation-eval JSONs.
- `ablation_results/` -- ablation JSONs.
- `study_logs/` -- per-process subprocess logs from
  `run_full_study.py`.
- `tensorboard/<run_id>/` -- per-seed TB events; consumed by the
  directional tests and the paper plots.

## Layout

| File / dir                  | Purpose                                                                                                                          |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `run_full_study.py`         | Top-level orchestrator. Schedules the per-`(env, method)` HPO + final-evaluation + degradation work units across available GPUs. |
| `sac_hyperparam_search.py`  | Single-method Optuna HPO + training. Invoked per-method by `run_full_study.py`; can also be run standalone.                      |
| `run_final_eval.py`         | Trains a single seed-bank for an `(env, method)` at the HPO-best hyperparameters; produces the main-result table.                |
| `run_degradation_eval.py`   | Sweeps an expert-noise grid for an `(env, method)` and trains the agent at each level. Writes `per_seed_metric` directly. |
| `run_ablation_eval.py`      | Trains the §5.3 EDGE ablation variants (gate form, scoring rule, ensemble size, etc.).                                           |
| `run_directional_tests.py`  | Permutation-test p-values for F1 (IBRL < SAC on near-ceiling experts), F2 (residual saturation), F3 (warm-start poisoning).     |
| `make_main_table.py`        | Renders `paper/tables/main_perenv.tex` from the populated final-eval JSONs.                                                      |
| `plot_sweep.py`             | Renders all paper figures (per-env headline, degradation bars, ablation bars) under `paper/figs/`.                              |
| `envs.py`                   | Single registry of every environment used in the paper.                                                                          |
| `eval_pid_expert.py`        | Stand-alone evaluation of the PID expert on the TargetGym tasks.                                                                 |
| `eval_cpg_expert.py`        | Stand-alone evaluation of the open-loop CPG expert on the locomotion tasks.                                                      |
| `tune_cpg.py`               | One-off optimisation of the CPG expert's frequency / amplitudes / phases per locomotion task.                                    |
| `run_safely.sh`             | Wraps long-running commands in `setsid nohup` with a timestamped log so they survive SSH logout. See [COMMANDS.md](COMMANDS.md). |
| `paper/`                    | LaTeX sources (`paper.tex`, NeurIPS style file, appendices). Build artefacts under `paper/build/` are gitignored.                |
| `pyproject.toml`, `poetry.lock` | Pinned Python environment.                                                                                                   |
| `tests/test_env_registry.py` | Sanity-check that every env declared in `envs.py` builds, exposes the expected obs/action shapes, and runs an expert step.       |
