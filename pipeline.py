"""
Full research pipeline for one task.

Steps (run in order):
  1. hyperparam_search   Bayesian HP optimisation — Optuna TPE, multi-GPU
  2. ablation_study      Full ablation suite via gpu_launcher
  3. noise_study         Noisy-expert degradation study via gpu_launcher
  4. plots               Generate paper-ready figures

Each step is skipped when already complete.  Individual steps can be
selected with --steps; use --status to inspect progress without running.

Usage:
  python pipeline.py --task plane                          # all steps
  python pipeline.py --task plane --steps ablation plots  # run a subset
  python pipeline.py --task plane --status                 # show status, no run
  python pipeline.py --task plane --gpus 0 1              # restrict GPUs
  python pipeline.py --task plane --no-wandb              # skip W&B logging
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional

from task_configs import TaskConfig, get_task_config

ALL_STEPS = ["hyperparam_search", "ablation_study", "noise_study", "plots"]


# ---------------------------------------------------------------------------
# Completion checks
# ---------------------------------------------------------------------------


def _run_list(script: str) -> str:
    """Run `python <script>.py --list` on CPU and return stdout."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, f"{script}.py", "--list"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    return result.stdout


def _parse_list_output(output: str) -> tuple[int, int]:
    """Parse --list output lines and return (n_done, n_total).

    Both ablation_study and noisy_expert_study print lines of the form:
        [index] DONE  name  ...    (complete)
        [index]       name  ...    (pending, four spaces)
    """
    n_done = n_total = 0
    for line in output.splitlines():
        line = line.strip()
        if not (line.startswith("[") and "]" in line):
            continue
        after = line.split("]", 1)[1].strip()
        tokens = after.split()
        if not tokens:
            continue
        n_total += 1
        if tokens[0] == "DONE":
            n_done += 1
    return n_done, n_total


def check_hp_search(cfg: TaskConfig) -> tuple[str, bool]:
    """Return (status_message, is_complete) for the hyperparam_search step."""
    db = cfg.hp_db_path
    if not db.exists():
        return "not started", False

    try:
        with sqlite3.connect(str(db)) as con:
            n_p1 = con.execute(
                "SELECT COUNT(*) FROM trials WHERE state = 1"
            ).fetchone()[0]
    except Exception:
        return "DB unreadable", False

    if n_p1 < cfg.hp_n_trials:
        return f"phase 1: {n_p1}/{cfg.hp_n_trials} trials", False

    n_p2 = len(list(Path(cfg.hp_results_dir).glob("*_phase2_result.json")))
    if n_p2 < cfg.hp_top_k:
        return f"phase 1 done; phase 2: {n_p2}/{cfg.hp_top_k} configs", False

    return f"done ({n_p1} trials, {n_p2} phase-2 configs)", True


def check_study(script: str) -> tuple[str, bool]:
    """Return (status_message, is_complete) for an ablation/noise study step."""
    output = _run_list(script)
    if not output.strip():
        return "no output from --list (script error?)", False
    n_done, n_total = _parse_list_output(output)
    if n_total == 0:
        return "no experiments found", False
    done = n_done == n_total
    return f"{n_done}/{n_total} experiments done", done


def check_plots(cfg: TaskConfig) -> tuple[str, bool]:
    """Return (status_message, is_complete) for the plots step."""
    out = Path(cfg.plot_output_dir)
    if not out.exists():
        return "not started", False
    figures = list(out.glob("**/*.pdf")) + list(out.glob("**/*.png"))
    if not figures:
        return "output dir exists but no figures found", False
    return f"{len(figures)} figure(s) in {cfg.plot_output_dir}/", True


def step_status(step: str, cfg: TaskConfig) -> tuple[str, bool]:
    if step == "hyperparam_search":
        return check_hp_search(cfg)
    if step == "ablation_study":
        return check_study(cfg.ablation_script)
    if step == "noise_study":
        return check_study(cfg.noise_script)
    if step == "plots":
        return check_plots(cfg)
    raise ValueError(f"Unknown step: {step!r}")


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> int:
    """Run a command with live stdio passthrough and return its exit code."""
    print(f"\n[pipeline] $ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd).returncode


def run_hp_search(cfg: TaskConfig, gpus: Optional[list[int]], no_wandb: bool) -> int:
    gpu_args = ["--gpus"] + [str(g) for g in gpus] if gpus else []
    wandb_args = ["--no-wandb"] if no_wandb else []

    # Determine which phases still need to run.
    n_p1 = 0
    if cfg.hp_db_path.exists():
        try:
            with sqlite3.connect(str(cfg.hp_db_path)) as con:
                n_p1 = con.execute(
                    "SELECT COUNT(*) FROM trials WHERE state = 1"
                ).fetchone()[0]
        except Exception:
            pass

    if n_p1 < cfg.hp_n_trials:
        ret = _run([
            sys.executable, f"{cfg.hp_script}.py",
            "--trials", str(cfg.hp_n_trials),
        ] + gpu_args + wandb_args)
        if ret != 0:
            return ret

    n_p2 = len(list(Path(cfg.hp_results_dir).glob("*_phase2_result.json")))
    if n_p2 < cfg.hp_top_k:
        ret = _run([
            sys.executable, f"{cfg.hp_script}.py",
            "--phase2", "--top-k", str(cfg.hp_top_k),
        ] + gpu_args + wandb_args)
        if ret != 0:
            return ret

    return 0


def run_study(
    script: str,
    gpus: Optional[list[int]],
    no_wandb: bool,
    n_timesteps: int,
) -> int:
    gpu_args = ["--gpus"] + [str(g) for g in gpus] if gpus else []
    wandb_args = ["--no-wandb"] if no_wandb else []
    return _run([
        sys.executable, "gpu_launcher.py",
        "--script", script,
        "--total-timesteps", str(n_timesteps),
    ] + gpu_args + wandb_args)


def run_plots(cfg: TaskConfig) -> int:
    return _run([sys.executable, f"{cfg.plot_script}.py"])


def run_step(step: str, cfg: TaskConfig, gpus: Optional[list[int]], no_wandb: bool) -> int:
    if step == "hyperparam_search":
        return run_hp_search(cfg, gpus, no_wandb)
    if step == "ablation_study":
        return run_study(cfg.ablation_script, gpus, no_wandb, cfg.ablation_n_timesteps)
    if step == "noise_study":
        return run_study(cfg.noise_script, gpus, no_wandb, cfg.noise_n_timesteps)
    if step == "plots":
        return run_plots(cfg)
    raise ValueError(f"Unknown step: {step!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full research pipeline for a given task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task", default="plane",
        help="Task name (default: plane). See task_configs.py for available tasks.",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=ALL_STEPS, default=None, metavar="STEP",
        help=f"Steps to run (default: all). Choices: {', '.join(ALL_STEPS)}",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print completion status for each step and exit without running.",
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=None,
        help="GPU IDs to use (default: all detected by nvidia-smi).",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging (TensorBoard logs are always written).",
    )
    args = parser.parse_args()

    cfg = get_task_config(args.task)
    steps_to_run = args.steps or ALL_STEPS

    # --- Status table (always shown) ---
    print(f"\nPipeline  task={cfg.name}")
    print("─" * 58)
    statuses: dict[str, tuple[str, bool]] = {}
    for step in ALL_STEPS:
        msg, done = step_status(step, cfg)
        statuses[step] = (msg, done)
        tick = "✓" if done else "○"
        sel = "*" if step in steps_to_run else " "
        print(f"  {sel} {tick}  {step:<22}  {msg}")
    print()

    if args.status:
        return

    # --- Run selected steps in order ---
    for step in steps_to_run:
        msg, done = statuses[step]
        if done:
            print(f"[pipeline] {step}: already complete ({msg}) — skipping.")
            continue
        print(f"[pipeline] {step}: {msg} — starting...")
        ret = run_step(step, cfg, args.gpus, args.no_wandb)
        if ret != 0:
            print(f"\n[pipeline] {step} failed (exit {ret}) — aborting pipeline.")
            sys.exit(ret)
        print(f"[pipeline] {step}: complete.")

    print("\n[pipeline] All selected steps complete.")


if __name__ == "__main__":
    main()
