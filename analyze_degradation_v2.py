"""Analyze v2 degradation results (action_bias + obs_noise).

Mirrors the v1 mistune analysis pattern in the paper but for the
deployment-side perturbations. Outputs:
  1. Per-(env, method, deg_type, intensity) `final_metric` table.
  2. Per-method worst-cell gap to per-cell-best (the headline metric).
  3. Comparison: v1 mistune worst-cell gap vs v2 worst-cell gap.
  4. Coverage summary (how many cells per env, what's pending).

Honest about partial coverage: prints which (env, method) pairs are
complete vs partial. Aggregates only over complete pairs unless
``--include-partial`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT / "degradation_results"
V2_DIR = ROOT / "degradation_v2_results"

METHODS = ["sac_quality_aware", "ibrl", "jsrl_curriculum", "jsrl", "residual"]
METHOD_DISPLAY = {
    "sac_quality_aware": "edge-qa",
    "ibrl": "ibrl",
    "jsrl_curriculum": "jsrl-cur",
    "jsrl": "jsrl-tt",
    "residual": "residual",
}
ENVS = ["plane3dcircle", "cheetahrun", "fourtank", "glassfurnace"]
V2_TYPES = ["action_bias", "obs_noise"]
V2_INTENSITIES = {
    "action_bias": [0.10, 0.25, 0.50],
    "obs_noise":   [0.05, 0.15, 0.30],
}
V1_SIGMAS = [0.25, 0.50, 0.75]


def _load_v2(env: str, method: str, dtype: str, intensity: float) -> Optional[float]:
    tag = f"{float(intensity)}"
    p = V2_DIR / f"{env}__{method}__{dtype}__{tag}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    v = d.get("final_metric")
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return float(v)


def _load_v1(env: str, method: str, sigma: float) -> Optional[float]:
    p = V1_DIR / f"{env}__{method}__sigma{sigma}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    v = d.get("final_metric")
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return float(v)


def coverage_report():
    print("=" * 70)
    print("COVERAGE")
    print("=" * 70)
    for env in ENVS:
        line = f"  {env:<22}"
        for method in METHODS:
            v2_done = sum(
                _load_v2(env, method, t, i) is not None
                for t in V2_TYPES for i in V2_INTENSITIES[t]
            )
            v1_done = sum(
                _load_v1(env, method, s) is not None for s in V1_SIGMAS
            )
            tag = "DONE" if v2_done == 6 else (
                "PART" if v2_done > 0 else "----"
            )
            line += f"  {METHOD_DISPLAY[method]:<8}v1={v1_done}/3 v2={v2_done}/6 [{tag}]"
        print(line)
    print()


def per_cell_table():
    """For each (env, method) where v2 is complete, print metric per cell."""
    print("=" * 70)
    print("V2 PER-CELL FINAL METRIC (only envs with at least one method complete)")
    print("=" * 70)
    for env in ENVS:
        any_complete = any(
            all(_load_v2(env, m, t, i) is not None
                for t in V2_TYPES for i in V2_INTENSITIES[t])
            for m in METHODS
        )
        if not any_complete:
            continue
        print(f"\n  {env}")
        # Header
        intensity_labels = []
        for t in V2_TYPES:
            for i in V2_INTENSITIES[t]:
                intensity_labels.append(f"{t[:4]}={i}")
        header = f"    {'method':<10}  " + "  ".join(f"{l:>10}" for l in intensity_labels)
        print(header)
        for method in METHODS:
            row_vals = []
            for t in V2_TYPES:
                for i in V2_INTENSITIES[t]:
                    v = _load_v2(env, method, t, i)
                    row_vals.append(f"{v:>10.1f}" if v is not None else f"{'-':>10}")
            print(f"    {METHOD_DISPLAY[method]:<10}  " + "  ".join(row_vals))


def worst_cell_gap_v2(restrict_envs: Optional[list] = None,
                     restrict_dtype: Optional[str] = None) -> dict:
    """Per-method worst-cell gap to per-cell-best, in % of |best|.

    A "cell" is (env, dtype, intensity). Only includes cells where ALL methods
    have data (so the per-cell-best is defined over the same method set).
    """
    envs = restrict_envs or ENVS
    types = [restrict_dtype] if restrict_dtype else V2_TYPES
    cells = []
    for env in envs:
        for t in types:
            for i in V2_INTENSITIES[t]:
                vals = {m: _load_v2(env, m, t, i) for m in METHODS}
                if any(v is None for v in vals.values()):
                    continue
                cells.append((env, t, i, vals))
    if not cells:
        return {m: None for m in METHODS}

    method_gaps = defaultdict(list)
    for env, t, i, vals in cells:
        best = max(vals.values())
        for m, v in vals.items():
            if best == 0:
                gap = 0.0
            else:
                gap = (best - v) / abs(best) * 100
            method_gaps[m].append(gap)
    return {m: max(gaps) if gaps else None for m, gaps in method_gaps.items()}


def worst_cell_gap_v1(restrict_envs: Optional[list] = None) -> dict:
    """Same metric but for v1 mistune."""
    envs = restrict_envs or ENVS
    cells = []
    for env in envs:
        for sigma in V1_SIGMAS:
            vals = {m: _load_v1(env, m, sigma) for m in METHODS}
            if any(v is None for v in vals.values()):
                continue
            cells.append((env, sigma, vals))
    if not cells:
        return {m: None for m in METHODS}
    method_gaps = defaultdict(list)
    for env, sigma, vals in cells:
        best = max(vals.values())
        for m, v in vals.items():
            if best == 0:
                gap = 0.0
            else:
                gap = (best - v) / abs(best) * 100
            method_gaps[m].append(gap)
    return {m: max(gaps) if gaps else None for m, gaps in method_gaps.items()}


def headline_table():
    print("=" * 70)
    print("HEADLINE: per-method worst-cell gap to per-cell best (% of |best|)")
    print("=" * 70)
    print()

    def _fmt(g):
        return f"{g:>6.1f}%" if g is not None else "   -   "

    # Find which envs are complete enough to include
    v2_envs_complete = []
    for env in ENVS:
        if all(
            all(_load_v2(env, m, t, i) is not None
                for t in V2_TYPES for i in V2_INTENSITIES[t])
            for m in METHODS
        ):
            v2_envs_complete.append(env)

    v1_envs_complete = []
    for env in ENVS:
        if all(
            all(_load_v1(env, m, s) is not None for s in V1_SIGMAS)
            for m in METHODS
        ):
            v1_envs_complete.append(env)

    print(f"  v2 envs (all 5 methods complete): {v2_envs_complete}")
    print(f"  v1 envs (all 5 methods complete): {v1_envs_complete}")
    print()

    print(f"  {'method':<10}  {'v1-mistune':>14}  {'v2-bias':>14}  {'v2-obs':>14}  {'v2-both':>14}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")

    v1_gaps = worst_cell_gap_v1(restrict_envs=v1_envs_complete or None)
    v2_bias = worst_cell_gap_v2(restrict_envs=v2_envs_complete or None,
                                restrict_dtype="action_bias")
    v2_obs = worst_cell_gap_v2(restrict_envs=v2_envs_complete or None,
                               restrict_dtype="obs_noise")
    v2_both = worst_cell_gap_v2(restrict_envs=v2_envs_complete or None)

    for m in METHODS:
        print(f"  {METHOD_DISPLAY[m]:<10}  "
              f"{_fmt(v1_gaps.get(m)):>14}  "
              f"{_fmt(v2_bias.get(m)):>14}  "
              f"{_fmt(v2_obs.get(m)):>14}  "
              f"{_fmt(v2_both.get(m)):>14}")
    print()
    print("Higher = worse (more behind the per-cell best). "
          "0% means this method was best in every cell.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", action="store_true",
                        help="Print per-cell metric table")
    args = parser.parse_args()
    coverage_report()
    headline_table()
    if args.cells:
        per_cell_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
