"""
Task registry: per-task pipeline configuration.

To support a new environment, add a TaskConfig entry to TASKS and implement
task-specific experiment scripts that accept --exp-index and --list.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskConfig:
    """Pipeline configuration for one task / environment.

    name must match the key used in hyperparams/ajax_sac.yml so that
    load_hyperparams("SAC", cfg.name) resolves correctly.
    """

    # Canonical task name — must match the key in hyperparams/ajax_sac.yml.
    name: str

    # ── Hyperparameter search ──────────────────────────────────────────────
    hp_script: str = "sac_hyperparam_search"
    hp_results_dir: str = "hp_results"
    hp_db_name: str = "optuna.db"
    hp_n_trials: int = 50       # Phase 1 target (matches sac_hyperparam_search --trials default)
    hp_top_k: int = 5           # Phase 2: confirm top-K configs at full budget

    # ── Ablation study ────────────────────────────────────────────────────
    ablation_script: str = "ablation_study"
    ablation_registry: str = "ablation_run_registry.json"
    ablation_n_seeds: int = 100
    ablation_n_timesteps: int = 1_000_000

    # ── Noise / degradation study ─────────────────────────────────────────
    noise_script: str = "noisy_expert_study"
    noise_registry: str = "noisy_expert_run_registry.json"
    noise_n_seeds: int = 100
    noise_n_timesteps: int = 1_000_000

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_script: str = "plot_sweep"
    plot_output_dir: str = "plots"

    @property
    def hp_db_path(self) -> Path:
        return Path(self.hp_results_dir) / self.hp_db_name


# ---------------------------------------------------------------------------
# Registered tasks
# ---------------------------------------------------------------------------

TASKS: dict[str, TaskConfig] = {
    "plane": TaskConfig(name="plane"),
}


def get_task_config(task: str) -> TaskConfig:
    if task not in TASKS:
        available = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{task}'. Available: {available}")
    return TASKS[task]
