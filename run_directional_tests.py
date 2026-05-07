"""One-sided permutation tests for the F1/F2/F3 directional predictions.

For each test, we extract per-seed trailing-window mean returns from
TB event files and run a one-sided permutation test of the directional
hypothesis at the front of each F-mechanism.

F1: IBRL < SAC on FourTank (clean phase3).
F2: Residual SAC < JSRL-training-time on Plane3DCircle (clean phase3).
F3: JSRL-training-time < EDGE on Plane3DCircle expert mistuning sigma=0.5.

For F1 and F2 we read the phase3 run_ids from hp_results/<env>/<method>/
trial_*_phase3_result.json and pull TB events from
final_results_tb/<env>/<method>/tensorboard/<run_id>/events*.

For F3, the degradation v1 runs do not have run_ids stored alongside
their final_metric JSON; we therefore re-run a per-seed extraction from
the WandB-mirrored TB events under tensorboard/<run_id> as long as the
degradation eval pipeline left them there. If absent, the test is
skipped with a clear error.
"""

from __future__ import annotations

import glob
import json
import os
import struct
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from tensorboardX.proto import event_pb2

ROOT = Path(__file__).resolve().parent
HP_RESULTS = ROOT / "hp_results"
FINAL_TB = ROOT / "final_results_tb"
DEGRADATION_RESULTS = ROOT / "degradation_results"

EVAL_TAG = "Eval/episodic_mean_reward"
WINDOW_STEPS = 200_000
N_PERMUTATIONS = 100_000
RNG = np.random.default_rng(20260506)


def _scan_events(path: str):
    """Yield (step, value) pairs for the EVAL_TAG scalar in a TB events file."""
    try:
        data = open(path, "rb").read()
    except OSError:
        return
    i = 0
    while i < len(data):
        if len(data) - i < 12:
            break
        (length,) = struct.unpack("<Q", data[i : i + 8])
        i += 12
        if len(data) - i < length + 4:
            break
        payload = data[i : i + length]
        i += length + 4
        try:
            ev = event_pb2.Event.FromString(payload)
        except Exception:
            break
        for v in ev.summary.value:
            if v.tag == EVAL_TAG:
                yield int(ev.step), float(v.simple_value)


def trailing_mean_for_run(run_dir: str, window_steps: int = WINDOW_STEPS) -> Optional[float]:
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.*")))
    if not files:
        return None
    entries = []
    for f in files:
        entries.extend(_scan_events(f))
    if not entries:
        return None
    entries.sort()
    last_step = entries[-1][0]
    window = [v for s, v in entries if s >= last_step - window_steps]
    if not window:
        return None
    return float(np.mean(window))


def per_seed_phase3(env: str, method: str) -> List[float]:
    """Read the phase3 result, get its run_ids, return per-seed trailing means.

    Phase3 run_ids are stored alongside the phase2 TB events:
    `hp_results/<env>/<method>/trial_<X>_phase2_tb/tensorboard/<run_id>/`.
    Phase2_tb holds 100 seeds, of which 50 are referenced by phase2_result and
    50 by phase3_result (disjoint).
    """
    candidates = sorted((HP_RESULTS / env / method).glob("trial_*_phase3_result.json"))
    if not candidates:
        raise FileNotFoundError(f"no phase3 result under hp_results/{env}/{method}")
    p3 = candidates[-1]
    payload = json.loads(p3.read_text())
    run_ids = payload.get("run_ids", [])
    if not run_ids:
        raise ValueError(f"phase3 result {p3} has no run_ids")
    # phase3 run_ids live under the corresponding trial_<X>_phase2_tb folder.
    trial_num = p3.stem.split("_")[1]  # e.g. "031" from "trial_031_phase3_result"
    tb_root = HP_RESULTS / env / method / f"trial_{trial_num}_phase2_tb" / "tensorboard"
    if not tb_root.is_dir():
        raise FileNotFoundError(f"phase2_tb folder missing: {tb_root}")
    means: List[float] = []
    missing = 0
    for rid in run_ids:
        run_dir = tb_root / rid
        if not run_dir.is_dir():
            missing += 1
            continue
        tm = trailing_mean_for_run(str(run_dir))
        if tm is not None:
            means.append(tm)
    if not means:
        raise ValueError(
            f"no per-seed trailing means recoverable for {env}/{method} (missing={missing})"
        )
    if missing:
        print(f"  [warn] {missing}/{len(run_ids)} run_ids missing TB for {env}/{method}", file=sys.stderr)
    return means


def per_seed_degradation_v1(env_lc: str, method: str, sigma: float) -> List[float]:
    """Per-seed trailing means for v1 (mistune) degradation runs.

    Current run_degradation_eval.py writes the per-seed final-window
    means directly into the JSON as ``per_seed_metric``. We use that
    when available; otherwise we fall back to TB-event extraction via
    ``run_ids`` (which a re-run of the cell will produce).
    """
    payload_path = DEGRADATION_RESULTS / f"{env_lc}__{method}__sigma{sigma}.json"
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = json.loads(payload_path.read_text())
    per_seed = payload.get("per_seed_metric")
    if per_seed:
        means = [float(x) for x in per_seed if x is not None]
        if not means:
            raise ValueError(
                f"per_seed_metric in {payload_path} is all None; re-run the cell."
            )
        return means
    run_ids = payload.get("run_ids", [])
    if not run_ids:
        raise ValueError(
            f"degradation v1 result {payload_path} has no per_seed_metric "
            f"and no run_ids; re-run with current run_degradation_eval.py."
        )
    tb_root = ROOT / "tensorboard"
    means: List[float] = []
    missing = 0
    for rid in run_ids:
        run_dir = tb_root / rid
        if not run_dir.is_dir():
            missing += 1
            continue
        tm = trailing_mean_for_run(str(run_dir))
        if tm is not None:
            means.append(tm)
    if not means:
        raise ValueError(
            f"no per-seed trailing means recoverable for degradation {env_lc}/{method}/sigma{sigma} "
            f"(missing={missing})"
        )
    if missing:
        print(f"  [warn] {missing}/{len(run_ids)} TB run_dirs missing for "
              f"{env_lc}/{method}/sigma{sigma}", file=sys.stderr)
    return means


def _scalar_iqm(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    sorted_a = np.sort(arr)
    q1 = int(np.floor(0.25 * len(sorted_a)))
    q3 = int(np.ceil(0.75 * len(sorted_a)))
    return float(np.mean(sorted_a[q1:q3]))


def one_sided_perm_test(
    a: np.ndarray, b: np.ndarray, alternative: str, n_perm: int = N_PERMUTATIONS,
    rng: np.random.Generator = RNG,
) -> dict:
    """One-sided permutation test on IQM(a) vs IQM(b).

    alternative='less' tests H1: IQM(a) < IQM(b).
    alternative='greater' tests H1: IQM(a) > IQM(b).
    """
    obs = _scalar_iqm(a) - _scalar_iqm(b)
    combined = np.concatenate([a, b])
    na = len(a)
    if alternative == "less":
        # count permutations as extreme as observed in the predicted direction
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(combined)
            diff = _scalar_iqm(perm[:na]) - _scalar_iqm(perm[na:])
            if diff <= obs:
                count += 1
    elif alternative == "greater":
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(combined)
            diff = _scalar_iqm(perm[:na]) - _scalar_iqm(perm[na:])
            if diff >= obs:
                count += 1
    else:
        raise ValueError(alternative)
    p = (count + 1) / (n_perm + 1)  # additive smoothing for plausible bound
    return {
        "iqm_a": _scalar_iqm(a),
        "iqm_b": _scalar_iqm(b),
        "delta_iqm": obs,
        "n_a": na,
        "n_b": len(b),
        "n_permutations": n_perm,
        "alternative": alternative,
        "p_value": p,
    }


def report(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  H1: IQM({name.split(':')[1].split('<')[0].strip()}) < "
          f"IQM({name.split('<')[1].split(',')[0].strip()})")
    print(f"  IQM(a) = {result['iqm_a']:.2f}, n={result['n_a']}")
    print(f"  IQM(b) = {result['iqm_b']:.2f}, n={result['n_b']}")
    print(f"  Delta IQM (a - b) = {result['delta_iqm']:.2f}")
    print(f"  one-sided p-value ({result['alternative']}) = {result['p_value']:.5f}")
    print(f"  ({result['n_permutations']:,} permutations, additive smoothing)")


def main() -> int:
    print("Loading per-seed trailing means from TB events...")
    print(f"Window: last {WINDOW_STEPS // 1000}k steps; {N_PERMUTATIONS:,} permutations")

    # F1: IBRL < SAC on FourTank, replicated on GlassFurnace.
    print("\nF1 setup: IBRL < SAC on FourTank")
    try:
        ibrl_ft = np.array(per_seed_phase3("fourtank", "ibrl"))
        sac_ft = np.array(per_seed_phase3("fourtank", "sac"))
        f1 = one_sided_perm_test(ibrl_ft, sac_ft, alternative="less")
        report("F1: IBRL < SAC, FourTank", f1)
    except Exception as e:
        print(f"  [skip] F1: {e}")
        f1 = None

    # F1 replication on GlassFurnace: IBRL < SAC, both at near-ceiling
    # expert. Per reviewer W2: a second F1 anchor.
    print("\nF1 setup: IBRL < SAC on GlassFurnace")
    try:
        ibrl_gf = np.array(per_seed_phase3("glassfurnace", "ibrl"))
        sac_gf = np.array(per_seed_phase3("glassfurnace", "sac"))
        f1_gf = one_sided_perm_test(ibrl_gf, sac_gf, alternative="less")
        report("F1: IBRL < SAC, GlassFurnace", f1_gf)
    except Exception as e:
        print(f"  [skip] F1 GF: {e}")
        f1_gf = None

    # F2: Residual < {JSRL-tt, IBRL, EDGE} on Plane3DCircle.
    # JSRL-tt is the strongest expert-using alternative; IBRL and EDGE
    # confirm the saturation against two more baselines.
    print("\nF2 setup: Residual SAC < {JSRL-tt, IBRL, EDGE} on Plane3DCircle")
    f2 = {}
    try:
        res_p3d = np.array(per_seed_phase3("plane3dcircle", "residual"))
        for opp_method, opp_label in [("jsrl", "JSRL-tt"), ("ibrl", "IBRL"),
                                      ("sac_quality_aware", "EDGE")]:
            opp = np.array(per_seed_phase3("plane3dcircle", opp_method))
            r = one_sided_perm_test(res_p3d, opp, alternative="less")
            report(f"F2: Residual < {opp_label}, P3DC", r)
            f2[opp_label] = r
    except Exception as e:
        print(f"  [skip] F2: {e}")
        f2 = None

    # F3: JSRL-training-time < EDGE on Plane3DCircle mistune sigma=0.5.
    print("\nF3 setup: JSRL-training-time < EDGE on Plane3DCircle mistune sigma=0.5")
    try:
        jsrltt_pert = np.array(per_seed_degradation_v1("plane3dcircle", "jsrl", 0.5))
        edge_pert = np.array(per_seed_degradation_v1("plane3dcircle", "sac_quality_aware", 0.5))
        f3 = one_sided_perm_test(jsrltt_pert, edge_pert, alternative="less")
        report("F3: JSRL-tt < EDGE, P3DC mistune sigma=0.5", f3)
    except Exception as e:
        print(f"  [skip] F3: {e}")
        f3 = None

    # Save summary
    out = {"F1": f1, "F1_GF": f1_gf, "F2": f2, "F3": f3}
    out_path = ROOT / "directional_test_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
