"""Generate Table 2 (per-env asymptotic IQM final return) with dispersion
and SAC-vs-method significance.

For each (env, method): reads per-seed trailing-200k-step mean returns from
TensorBoard via ``plot_sweep._paper_per_seed_final``, then reports
  - IQM (interquartile mean)
  - bootstrap 95% [Q1, Q3] of the IQM
  - Δ% relative to SAC
  - p-value of a two-sided Mann-Whitney U test against SAC

P-values are corrected with Holm-Bonferroni *within each env*, across the
five expert-using methods (i.e.\ SAC itself is the reference and not
counted in the family). Significance markers:
  - ***  p_corr < 0.001
  - **   p_corr < 0.01
  - *    p_corr < 0.05
  - (no marker) p_corr >= 0.05

Output: writes a LaTeX-formatted tabular fragment to
``paper/tables/main_perenv.tex`` that ``paper.tex`` can ``\\input``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "paper" / "tables"

ENVS = [
    ("Plane3DCircle", "P3D-Circle"),
    ("FourTank",      "FourTank"),
    ("GlassFurnace",  "GlassFurnace"),
    ("CheetahRun",    "CheetahRun"),
]
METHODS = [
    ("sac",                "SAC (baseline)"),
    ("ibrl",               "IBRL"),
    ("jsrl_curriculum",    "JSRL (curriculum)"),
    ("jsrl",               "JSRL (training-time)"),
    ("residual",           "Residual SAC"),
    ("sac_quality_aware",  "EDGE (ours)"),
]
PROPOSED = "sac_quality_aware"
REFERENCE = "sac"


def _bootstrap_iqm_qrange(
    values: np.ndarray, n_bootstrap: int = 5000, ci: float = 95
) -> tuple[float, float, float]:
    """IQM and bootstrap [q_lo, q_hi] (default 95%)."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(42)
    n = len(valid)
    lo_idx = int(np.floor(0.25 * n))
    hi_idx = int(np.ceil(0.75 * n))
    center = float(np.sort(valid)[lo_idx:hi_idx].mean())
    boots = rng.choice(valid, size=(n_bootstrap, n), replace=True)
    boot_iqm = np.sort(boots, axis=1)[:, lo_idx:hi_idx].mean(axis=1)
    alpha = (100 - ci) / 2
    return (
        center,
        float(np.percentile(boot_iqm, alpha)),
        float(np.percentile(boot_iqm, 100 - alpha)),
    )


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Order-preserving."""
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p_values[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected.tolist()


def _sig_marker(p: float) -> str:
    if p < 0.001: return "$^{***}$"
    if p < 0.01:  return "$^{**}$"
    if p < 0.05:  return "$^{*}$"
    return ""


def _fmt_iqm(v: float) -> str:
    if not np.isfinite(v):
        return "TBD"
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", "{,}")
    return f"{v:,.1f}".replace(",", "{,}")


def _fmt_delta_pct(d: float) -> str:
    if not np.isfinite(d):
        return "---"
    sign = "+" if d >= 0 else "$-$"
    return f"{sign}{abs(d):.0f}\\%"


def main() -> int:
    from plot_sweep import _paper_per_seed_final

    rows: dict[tuple[str, str], dict] = {}
    for env_name, _ in ENVS:
        # Reference series (SAC). If SAC is missing on this env, all
        # Δ%/p-values for this env are reported as "---".
        ref_arr = _paper_per_seed_final(env_name, REFERENCE)
        for m_key, _ in METHODS:
            arr = _paper_per_seed_final(env_name, m_key)
            if arr is None or arr.size == 0:
                rows[(env_name, m_key)] = {"iqm": np.nan}
                continue
            iqm, q_lo, q_hi = _bootstrap_iqm_qrange(arr)
            rec = {"iqm": iqm, "q_lo": q_lo, "q_hi": q_hi,
                   "raw": arr, "delta_pct": np.nan, "p_raw": np.nan}
            if m_key != REFERENCE and ref_arr is not None and ref_arr.size:
                ref_iqm = float(np.sort(ref_arr)[
                    int(np.floor(0.25 * len(ref_arr))):
                    int(np.ceil(0.75 * len(ref_arr)))
                ].mean())
                if abs(ref_iqm) > 1e-9:
                    rec["delta_pct"] = (iqm - ref_iqm) / abs(ref_iqm) * 100.0
                # Mann-Whitney U two-sided
                try:
                    _, p_raw = mannwhitneyu(arr, ref_arr, alternative="two-sided")
                except ValueError:
                    p_raw = np.nan
                rec["p_raw"] = float(p_raw) if np.isfinite(p_raw) else np.nan
            rows[(env_name, m_key)] = rec

        # Holm-Bonferroni within env, over expert-using methods only.
        family_keys = [m for m, _ in METHODS if m != REFERENCE]
        family_p = [rows[(env_name, m)].get("p_raw", np.nan)
                    for m in family_keys]
        valid = [(k, p) for k, p in zip(family_keys, family_p) if np.isfinite(p)]
        if valid:
            corrected = _holm_bonferroni([p for _, p in valid])
            for (k, _), pc in zip(valid, corrected):
                rows[(env_name, k)]["p_corr"] = pc

    # Emit LaTeX
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "main_perenv.tex"
    lines: list[str] = []
    n_envs = len(ENVS)
    col_spec = "l" + "r" * n_envs
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Method} & "
        + " & ".join(f"\\textbf{{{disp}}}" for _, disp in ENVS)
        + " \\\\"
    )
    lines.append("\\midrule")
    for m_key, m_disp in METHODS:
        cells = []
        for env_name, _ in ENVS:
            r = rows[(env_name, m_key)]
            iqm = r["iqm"]
            if not np.isfinite(iqm):
                cells.append("TBD")
                continue
            iqm_str = _fmt_iqm(iqm)
            if m_key == REFERENCE:
                # Reference (SAC) row: show raw IQM as the absolute anchor.
                disp = iqm_str
            else:
                # Non-reference rows: show only Δ% vs SAC + significance
                # marker. CI brackets and raw IQM dropped to reduce
                # visual clutter.
                d = r.get("delta_pct", np.nan)
                p_corr = r.get("p_corr", np.nan)
                if np.isfinite(d):
                    disp = (
                        f"{_fmt_delta_pct(d)}"
                        f"{_sig_marker(p_corr) if np.isfinite(p_corr) else ''}"
                    )
                else:
                    disp = iqm_str
            cells.append(disp)
        bold = "\\textbf{" + m_disp + "}" if m_key == PROPOSED else m_disp
        if m_key == PROPOSED:
            lines.append("\\midrule")
        lines.append(bold + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out.write_text("\n".join(lines) + "\n")
    print(f"[main_table] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
