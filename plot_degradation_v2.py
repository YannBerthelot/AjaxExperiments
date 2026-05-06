"""Plot v1 mistune + v2 deployment-side degradation as grouped bars.

One PDF per env, three panels per PDF:
  Panel 1: v1 mistune (param perturbation), σ ∈ {0.25, 0.5, 0.75}
  Panel 2: v2 action bias, σ ∈ {0.10, 0.25, 0.50}
  Panel 3: v2 obs noise,   σ ∈ {0.05, 0.15, 0.30}

Each panel: x-groups = σ values, one bar per method, y = IQM final return
across 50 seeds. Per-bar annotations show % change from the method's clean
phase3 reference. Style matches the legacy ``degradation_bars_*.pdf``:
hidden top/right spines, dashed y-grid, framed legend, DeepMind palette.

Output: paper/figs/degradation_v2_bars_{env}.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT / "degradation_results"
V2_DIR = ROOT / "degradation_v2_results"
HP_DIR = ROOT / "hp_results"
FIG_DIR = ROOT / "paper" / "figs"

ENVS = ["plane3dcircle", "cheetahrun", "fourtank", "glassfurnace"]
ENV_DISPLAY = {
    "plane3dcircle": "Plane3DCircle",
    "cheetahrun": "CheetahRun",
    "fourtank": "FourTank",
    "glassfurnace": "GlassFurnace",
}

METHODS = ["sac_quality_aware", "ibrl", "jsrl_curriculum", "jsrl", "residual"]
# Display names + palette: kept in sync with plot_sweep._PAPER_METHODS so a
# given algorithm has the same name and color across every figure in the paper.
METHOD_DISPLAY = {
    "sac_quality_aware": "EDGE",
    "ibrl":              "IBRL",
    "jsrl_curriculum":   "JSRL",
    "jsrl":              "JSRL-tt",
    "residual":          "Residual",
}
METHOD_COLOR = {
    "sac_quality_aware": "#1A9B6C",  # teal green (proposed)
    "ibrl":              "#9B59B6",  # violet
    "jsrl_curriculum":   "#E8A820",  # amber (canonical JSRL curriculum)
    "jsrl":              "#C0392B",  # red (training-time JSRL variant)
    "residual":          "#F4762A",  # orange
}

V1_SIGMAS = [0.25, 0.50, 0.75]
V2_INTENSITIES = {
    "action_bias": [0.10, 0.25, 0.50],
    "obs_noise":   [0.05, 0.15, 0.30],
}

# Per-env J_exp / J_ref for ENA = (J - J_exp) / (J_ref - J_exp).
# Keep in sync with run_full_study.EXPERT_REFERENCE and
# plot_sweep._PAPER_J_STAR.
J_EXP = {
    "plane3dcircle": 4765.8,
    "fourtank":      246.0,
    "glassfurnace":  4315.31,
    "cheetahrun":    291.2,
}
J_REF = {
    "plane3dcircle": 10000.0,
    "fourtank":      500.0,
    "glassfurnace":  6000.0,
    "cheetahrun":    1000.0,
}


def _to_ena(env: str, j: float) -> float:
    e = J_EXP.get(env.lower())
    r = J_REF.get(env.lower())
    if e is None or r is None or (r - e) <= 0:
        return j
    return (j - e) / (r - e)


def _apply_style():
    plt.rcParams.update({
        # Type
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.titleweight": "regular",
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        # Grid (light, minor)
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#dddddd",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        # Ticks
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.major.size": 0,
        "ytick.major.size": 3,
        "ytick.major.width": 0.7,
        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.4,
        "legend.handletextpad": 0.6,
        # Figure
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _load_clean(env: str, method: str) -> Optional[float]:
    cands = sorted((HP_DIR / env / method).glob("trial_*_phase3_result.json"))
    if not cands:
        return None
    try:
        d = json.loads(cands[-1].read_text())
        v = d.get("metric")
        return float(v) if v is not None else None
    except Exception:
        return None


def _load_v1(env: str, method: str, sigma: float) -> Optional[float]:
    p = V1_DIR / f"{env}__{method}__sigma{sigma}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        v = d.get("final_metric")
        if v is None or not np.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def _load_v2(env: str, method: str, dtype: str, intensity: float) -> Optional[float]:
    tag = f"{float(intensity)}"
    p = V2_DIR / f"{env}__{method}__{dtype}__{tag}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        v = d.get("final_metric")
        if v is None or not np.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def _draw_panel(ax, env: str, sigmas: list, loader, title: str, xlabel: str):
    """Render one grouped-bars panel with a clean (σ=0) reference group.

    loader(env, method, sigma) -> Optional[float] for the perturbed cells.
    The σ=0 column is filled from each method's HPO phase3 metric (50 seeds,
    1M steps, no perturbation). Per-bar % annotations on the perturbed
    columns show change vs that method's σ=0 baseline.
    """
    # Prepend σ=0 (clean) as the first group.
    sigmas_with_clean = [0.0] + list(sigmas)
    n_groups = len(sigmas_with_clean)
    n_methods = len(METHODS)
    group_w = 0.82
    bar_w = group_w / max(n_methods, 1)
    x = np.arange(n_groups, dtype=np.float32)

    pct_labels: list[tuple[float, float, str, bool]] = []
    for i, method in enumerate(METHODS):
        is_proposed = method == "sac_quality_aware"
        color = METHOD_COLOR[method]
        clean = _load_clean(env, method)
        offsets = x - group_w / 2 + bar_w / 2 + i * bar_w
        heights = []
        for s in sigmas_with_clean:
            if s == 0.0:
                heights.append(clean if clean is not None else np.nan)
            else:
                v = loader(env, method, s)
                heights.append(v if v is not None else np.nan)
        heights = np.asarray(heights, dtype=np.float32)
        mask = np.isfinite(heights)
        ax.bar(
            offsets[mask], heights[mask], width=bar_w * 0.88,
            color=color,
            alpha=0.95 if is_proposed else 0.85,
            edgecolor="#222222" if is_proposed else "none",
            linewidth=0.8 if is_proposed else 0,
            label=METHOD_DISPLAY[method], zorder=3 if is_proposed else 2,
        )
        # Subtle hatch-style hint for the clean group: slightly higher alpha
        # is already applied uniformly; instead annotate % change for σ>0
        # bars only. (σ=0 is the reference, no annotation.)
        if clean is None or abs(clean) < 1e-6:
            continue
        for xi, s in enumerate(sigmas_with_clean):
            if s == 0.0:
                continue
            h = heights[xi]
            if not np.isfinite(h):
                continue
            pct = (h - clean) / abs(clean) * 100.0
            sign = "+" if pct >= 0 else "−"
            pct_labels.append(
                (offsets[xi], h, f"{sign}{abs(pct):.0f}%", is_proposed)
            )
    ax.axhline(0.0, color="#9a9a9a", linewidth=0.8, linestyle="-",
               alpha=0.7, zorder=1)
    ax.set_xticks(x)
    xlabels = []
    for s in sigmas_with_clean:
        if s == 0.0:
            xlabels.append("clean")
        else:
            xlabels.append(f"$\\sigma\\!=\\!{s:g}$")
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("IQM final return")
    ax.set_title(title, pad=8)
    # Vertical separator between clean and perturbed groups
    sep_x = x[0] + (x[1] - x[0]) / 2
    ax.axvline(sep_x, color="#cccccc", linewidth=0.7, linestyle="--",
               alpha=0.8, zorder=0)
    if pct_labels:
        y_lo, y_hi = ax.get_ylim()
        span = max(y_hi - y_lo, 1.0)
        ax.set_ylim(y_lo - 0.06 * span, y_hi + 0.10 * span)
        offset = 0.014 * span
        for xpos, h, text, is_prop in pct_labels:
            above = h >= 0
            y = h + offset if above else h - offset
            ax.annotate(
                text, xy=(xpos, y),
                ha="center", va="bottom" if above else "top",
                fontsize=6.5,
                fontweight="bold" if is_prop else "regular",
                color="#333333", zorder=5,
            )


def _shade_toward_white(base, t: float) -> tuple:
    """Blend ``base`` color toward white by fraction ``t`` in [0, 1]."""
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(base))
    return tuple(rgb * (1 - t) + np.array([1.0, 1.0, 1.0]) * t)


def _draw_panel_by_method(ax, env: str, sigmas: list, loader,
                          title: str,
                          show_pct_labels: bool = True,
                          metric: str = "raw") -> None:
    """Method-grouped variant: x = methods, bars in group = clean + σ values.

    Sequential shading per method: most-saturated color = clean, lightest =
    highest σ. Per-bar % annotations vs the method's clean baseline,
    staggered vertically when adjacent bars have similar values.
    Each bar also carries its own σ tag underneath (vertical text), so no
    legend is needed.
    """
    sigmas_with_clean = [0.0] + list(sigmas)
    n_bars = len(sigmas_with_clean)
    n_methods = len(METHODS)
    group_w = 0.84
    bar_w = group_w / n_bars
    x = np.arange(n_methods, dtype=np.float32)

    # First pass: draw bars + record (xpos, height, label, is_prop, group_idx, bar_idx)
    bar_records = []
    transform = (lambda v: _to_ena(env, v)) if metric == "ena" else (lambda v: v)
    for m_i, method in enumerate(METHODS):
        is_proposed = method == "sac_quality_aware"
        clean = _load_clean(env, method)
        base_color = METHOD_COLOR[method]
        offsets = x[m_i] - group_w / 2 + bar_w / 2 + np.arange(n_bars) * bar_w
        for s_i, s in enumerate(sigmas_with_clean):
            if s == 0.0:
                raw_h = clean if clean is not None else np.nan
            else:
                v = loader(env, method, s)
                raw_h = v if v is not None else np.nan
            if not np.isfinite(raw_h):
                continue
            h = transform(raw_h)
            t = 0.55 * (s_i / max(n_bars - 1, 1))
            color = _shade_toward_white(base_color, t)
            ax.bar(
                offsets[s_i], h, width=bar_w * 0.90,
                color=color,
                edgecolor=("#222222" if (is_proposed and s_i == 0) else "none"),
                linewidth=0.8 if (is_proposed and s_i == 0) else 0,
                zorder=3 if is_proposed else 2,
            )
            text = ""
            if s != 0.0 and clean is not None and abs(clean) > 1e-6:
                pct = (h - clean) / abs(clean) * 100.0
                sign = "+" if pct >= 0 else "−"
                text = f"{sign}{abs(pct):.0f}%"
            bar_records.append({
                "xpos": offsets[s_i], "h": h, "pct_text": text,
                "is_prop": is_proposed, "m_i": m_i, "s_i": s_i, "sigma": s,
            })

    ax.axhline(0.0, color="#9a9a9a", linewidth=0.8, linestyle="-",
               alpha=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY[m] for m in METHODS],
                       fontweight="medium", fontsize=11,
                       rotation=25, ha="right",
                       rotation_mode="anchor")
    ylab = "ENA" if metric == "ena" else "IQM final return"
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_title(title, pad=8, fontsize=14)
    ax.tick_params(axis="y", labelsize=11)

    # Per-panel σ legend (neutral gray ramp showing the shading meaning)
    # placed below the axes.
    leg_handles = []
    for s_i, s in enumerate(sigmas_with_clean):
        t = 0.55 * (s_i / max(n_bars - 1, 1))
        color = _shade_toward_white("#555555", t)
        label = "clean" if s == 0.0 else f"$\\sigma\\!=\\!{s:g}$"
        leg_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color,
                                         ec="none", label=label))
    ax.legend(
        handles=leg_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8, ncol=2,
        frameon=False, handlelength=1.0, handletextpad=0.3,
        columnspacing=0.8, labelspacing=0.25,
    )

    # Caller is expected to set the y-limits (so all shared-y panels
    # land on the same scale). We just compute a panel-local span here
    # for the label-offset heuristic.
    bars_h = [r["h"] for r in bar_records]
    if bars_h:
        local_span = max(max(bars_h) - min(bars_h), 1.0)
    else:
        local_span = 1.0

    # Vertical (90 deg) %-change labels, all bold. Optional: callers
    # can disable when the figure is space-constrained (the body
    # figure) and refer to the appendix for the annotated version.
    if bar_records and show_pct_labels:
        offset = 0.018 * local_span
        for r in bar_records:
            if not r["pct_text"]:
                continue
            h = r["h"]
            above = h >= 0
            y = (h + offset) if above else (h - offset)
            ax.annotate(
                r["pct_text"], xy=(r["xpos"], y),
                ha="center", va="bottom" if above else "top",
                rotation=90, fontsize=8.5, fontweight="bold",
                color="#333333", zorder=5,
            )
    return bars_h


def _select_panels(env: str):
    """Return list of (sigmas, loader, title, count) for panels that have data."""
    panels = []
    n_v1 = sum(_load_v1(env, m, s) is not None for m in METHODS for s in V1_SIGMAS)
    if n_v1 > 0:
        panels.append((
            V1_SIGMAS,
            lambda e, m, s: _load_v1(e, m, s),
            "Expert mistuning",
            n_v1,
        ))
    n_v2_bias = sum(
        _load_v2(env, m, "action_bias", i) is not None
        for m in METHODS for i in V2_INTENSITIES["action_bias"]
    )
    if n_v2_bias > 0:
        panels.append((
            V2_INTENSITIES["action_bias"],
            lambda e, m, s: _load_v2(e, m, "action_bias", s),
            "Expert action bias",
            n_v2_bias,
        ))
    n_v2_obs = sum(
        _load_v2(env, m, "obs_noise", i) is not None
        for m in METHODS for i in V2_INTENSITIES["obs_noise"]
    )
    if n_v2_obs > 0:
        panels.append((
            V2_INTENSITIES["obs_noise"],
            lambda e, m, s: _load_v2(e, m, "obs_noise", s),
            "Observation noise",
            n_v2_obs,
        ))
    return panels


def plot_env_by_method(env: str, layout: str = "vertical",
                       show_pct_labels: bool = True,
                       suffix_extra: str = "") -> Optional[Path]:
    """Method-grouped figure. Renders only the panels with data.

    ``layout='vertical'``: stack panels in a single column (one row per
    perturbation type). Each panel gets the full figure width so the
    method labels and per-bar % annotations are readable. Best for
    half-column / single-column body figures.

    ``layout='horizontal'``: panels in a single row (legacy).
    """
    panels = _select_panels(env)
    if not panels:
        return None
    _apply_style()
    n_panels = len(panels)
    if layout == "vertical":
        # One panel per row, full width. Sized for a standard text column
        # so the figure reads cleanly inline.
        panel_h = 3.4
        env_x_in = 0.08
        ylabel_in = 0.55
        left_in = 1.10
        fig_w = 7.5
        fig_h = panel_h * n_panels + 0.4
        fig = plt.figure(figsize=(fig_w, fig_h))
        left_frac = left_in / fig_w
        env_x_frac = env_x_in / fig_w
        ylabel_frac = ylabel_in / fig_w
        # Top/bottom margins per panel (use absolute fractions of total height).
        top_pad = 0.4 / fig_h
        bot_pad = 0.7 / fig_h  # legend below each panel
        gs = fig.add_gridspec(
            n_panels, 1,
            left=left_frac, right=0.985,
            top=1 - top_pad, bottom=bot_pad,
            hspace=0.95,
        )
        axes = [fig.add_subplot(gs[i, 0]) for i in range(n_panels)]
        for i, (ax, (sigmas, loader, title, n)) in enumerate(zip(axes, panels)):
            _draw_panel_by_method(ax, env, sigmas, loader, title)
        # Each panel keeps its own y-axis label (full width per panel makes
        # this fine), but pin all y-labels to the same fig-x for alignment.
        for ax in axes:
            ax.yaxis.set_label_coords(ylabel_frac, 0.5,
                                      transform=fig.transFigure)
        fig.text(
            env_x_frac, 0.5, ENV_DISPLAY[env],
            rotation=90, ha="left", va="center",
            fontsize=15, fontweight="bold", color="#222222",
        )
    else:
        # Horizontal layout: vertical env name on the far left, shared
        # y-axis, σ legend (2 cols × 2 rows) below each panel.
        panel_w = 3.0
        env_x_in = 0.12
        ylabel_in = 0.65
        left_in = 1.40
        fig_h = 3.8
        fig_w = panel_w * n_panels + left_in + 0.05
        fig = plt.figure(figsize=(fig_w, fig_h))
        left_frac = left_in / fig_w
        env_x_frac = env_x_in / fig_w
        ylabel_frac = ylabel_in / fig_w
        gs = fig.add_gridspec(1, n_panels, left=left_frac, right=0.99,
                              top=0.92, bottom=0.30, wspace=0.10)
        axes = []
        for i in range(n_panels):
            kwargs = {"sharey": axes[0]} if i > 0 else {}
            axes.append(fig.add_subplot(gs[0, i], **kwargs))
        all_bars = []
        for i, (ax, (sigmas, loader, title, n)) in enumerate(zip(axes, panels)):
            bars_h = _draw_panel_by_method(
                ax, env, sigmas, loader, title,
                show_pct_labels=show_pct_labels,
            )
            all_bars.extend(bars_h)
            if i > 0:
                ax.set_ylabel("")
                plt.setp(ax.get_yticklabels(), visible=False)
        if all_bars:
            y_lo, y_hi = min(all_bars), max(all_bars)
            span = max(y_hi - y_lo, 1.0)
            axes[0].set_ylim(y_lo - 0.18 * span, y_hi + 0.20 * span)
        # Pin the leftmost panel's y-label to a fixed fig-x so the
        # vertical env name has a clean column to its left.
        axes[0].yaxis.set_label_coords(ylabel_frac, 0.5,
                                       transform=fig.transFigure)
        fig.text(
            env_x_frac, 0.5, ENV_DISPLAY[env],
            rotation=90, ha="left", va="center",
            fontsize=15, fontweight="bold", color="#222222",
        )
    suffix = "" if layout == "horizontal" else f"_{layout}"
    out = FIG_DIR / f"degradation_v2_bars_by_method_{env.lower()}{suffix}{suffix_extra}.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all_envs_grid(
    envs: tuple = ("plane3dcircle", "fourtank", "glassfurnace", "cheetahrun"),
    show_pct_labels: bool = False,
    out_name: str = "degradation_v2_bars_by_method_grid.pdf",
    metric: str = "raw",
) -> Optional[Path]:
    """4-row × 3-column grid: rows = envs, cols = perturbation types.

    Layout:
    - Vertical env name on the left of each row
    - Per-row shared y-axis (each env has its own scale)
    - Method names as x-tick labels only on the bottom row
    - Panel titles only on the top row
    - Per-column σ legend below the bottom row
    - Empty (data-pending) cells show a "no data" placeholder
    """
    _apply_style()
    n_rows = len(envs)
    # Three perturbation types in canonical order (matching _select_panels).
    perturb_types = [
        ("Expert mistuning", V1_SIGMAS,
         lambda e, m, s: _load_v1(e, m, s)),
        ("Expert action bias", V2_INTENSITIES["action_bias"],
         lambda e, m, s: _load_v2(e, m, "action_bias", s)),
        ("Observation noise", V2_INTENSITIES["obs_noise"],
         lambda e, m, s: _load_v2(e, m, "obs_noise", s)),
    ]
    n_cols = len(perturb_types)
    panel_w = 2.5
    panel_h = 1.45
    env_x_in = 0.10
    ylabel_in = 0.40
    left_in = 0.95
    bot_in = 0.55   # x-tick labels only (legend is now at the top)
    top_in = 0.95   # title + per-col σ legend
    fig_w = panel_w * n_cols + left_in + 0.05
    fig_h = panel_h * n_rows + bot_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    left_frac = left_in / fig_w
    env_x_frac = env_x_in / fig_w
    ylabel_frac = ylabel_in / fig_w
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=left_frac, right=0.99,
        top=1 - top_in / fig_h, bottom=bot_in / fig_h,
        wspace=0.12, hspace=0.18,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    # Build axes: first column for each row is the y-axis owner; other
    # columns share y within the row.
    for r in range(n_rows):
        for c in range(n_cols):
            sharey_target = axes[r, 0] if c > 0 else None
            ax = fig.add_subplot(
                gs[r, c],
                sharey=sharey_target,
            )
            axes[r, c] = ax

    # Draw each cell
    last_handles_per_col = [None] * n_cols
    for r, env in enumerate(envs):
        row_bars = []
        any_in_row = False
        for c, (pname, sigmas, loader) in enumerate(perturb_types):
            ax = axes[r, c]
            # Check if env has any data for this perturbation type
            has_data = any(
                loader(env, m, s) is not None
                for m in METHODS for s in sigmas
            )
            if not has_data:
                ax.text(
                    0.5, 0.5, "data pending",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=10, color="#aaaaaa", style="italic",
                )
                ax.set_xticks([]); ax.set_yticks([])
                ax.spines[["top", "right", "left", "bottom"]].set_color("#dddddd")
                continue
            any_in_row = True
            bars_h = _draw_panel_by_method(
                ax, env, sigmas, loader,
                title=pname if r == 0 else "",
                show_pct_labels=show_pct_labels,
                metric=metric,
            )
            row_bars.extend(bars_h)
            sigmas_with_clean = [0.0] + list(sigmas)
            n_bars = len(sigmas_with_clean)
            handles = []
            for s_i, s in enumerate(sigmas_with_clean):
                t = 0.55 * (s_i / max(n_bars - 1, 1))
                color = _shade_toward_white("#555555", t)
                label = "clean" if s == 0.0 else f"$\\sigma\\!=\\!{s:g}$"
                handles.append(plt.Rectangle((0, 0), 1, 1, fc=color,
                                             ec="none", label=label))
            last_handles_per_col[c] = handles
            # Remove the per-panel legend that _draw_panel_by_method added
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            ax.set_ylabel("")
            if r == 0:
                # Bold title with extra pad to make room for legend below it.
                ax.set_title(pname, fontsize=14, fontweight="bold", pad=42)
            else:
                ax.set_title("")
            if r != n_rows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xticklabels([METHOD_DISPLAY[m] for m in METHODS],
                                   fontweight="medium", fontsize=12,
                                   rotation=25, ha="right",
                                   rotation_mode="anchor")
            if c > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
        # Set per-row shared y-limits
        if row_bars:
            y_lo, y_hi = min(row_bars), max(row_bars)
            span = max(y_hi - y_lo, 1e-6)
            axes[r, 0].set_ylim(y_lo - 0.12 * span, y_hi + 0.18 * span)
        # Per-row y-label only on first column
        if any_in_row:
            axes[r, 0].set_ylabel(
                "ENA" if metric == "ena" else "IQM final return",
                fontsize=10, labelpad=2)
            axes[r, 0].yaxis.set_label_coords(
                ylabel_frac, (n_rows - 1 - r + 0.5) / n_rows
                * (1 - top_in / fig_h - bot_in / fig_h)
                + bot_in / fig_h,
                transform=fig.transFigure,
            )
        # Vertical env label
        row_y = (n_rows - 1 - r + 0.5) / n_rows * (
            1 - top_in / fig_h - bot_in / fig_h
        ) + bot_in / fig_h
        fig.text(
            env_x_frac, row_y, ENV_DISPLAY[env],
            rotation=90, ha="left", va="center",
            fontsize=12, fontweight="bold", color="#222222",
        )
    # Per-column σ legend ABOVE the top-row panels (just under each
    # panel's bold title). loc='lower center' + bbox at y=1.02 places
    # the bottom of the legend just above the panel area; the title
    # already sits even higher because we set pad=42.
    for c in range(n_cols):
        handles = last_handles_per_col[c]
        if handles is None:
            continue
        ax = axes[0, c]
        ax.legend(
            handles=handles, loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            fontsize=10, ncol=2,
            frameon=False,
            handlelength=1.6, handleheight=1.4,
            handletextpad=0.4,
            columnspacing=1.0, labelspacing=0.3,
        )

    out = FIG_DIR / out_name
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_env(env: str) -> Optional[Path]:
    """σ-grouped variant: groups by σ on the x-axis, methods are colored bars.

    Matches the polish of plot_env_by_method: clean panel titles (no v1/v2
    prefix), single y-label on the leftmost panel, env name as a bold
    vertical label on the far left, top-anchored method legend.
    """
    panel_specs = _select_panels(env)
    if not panel_specs:
        return None
    _apply_style()
    n_panels = len(panel_specs)
    panel_w = 5.3
    # Absolute-inch padding (see plot_env_by_method for layout description).
    env_x_in = 0.08
    ylabel_in = 0.42
    left_in = 1.05
    fig_h = 5.0
    fig_w = panel_w * n_panels + left_in + 0.05
    fig = plt.figure(figsize=(fig_w, fig_h))
    left_frac = left_in / fig_w
    env_x_frac = env_x_in / fig_w
    ylabel_frac = ylabel_in / fig_w
    # Bottom margin reserved for the figure-level method legend.
    gs = fig.add_gridspec(1, n_panels, left=left_frac, right=0.99,
                          top=0.95, bottom=0.20, wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    xlabel_map = {
        "Expert mistuning":   "Mistune $\\sigma$",
        "Expert action bias": "Bias $\\sigma$",
        "Observation noise":  "Obs-noise $\\sigma$",
    }
    for i, (ax, (sigmas, loader, title, n)) in enumerate(zip(axes, panel_specs)):
        _draw_panel(ax, env, sigmas, loader, title,
                    xlabel_map.get(title, "$\\sigma$"))
        if i > 0:
            ax.set_ylabel("")
    # Pin the leftmost panel's y-label to a fixed figure-x so it stays clear
    # of the vertical env name regardless of tick-label width.
    axes[0].yaxis.set_label_coords(ylabel_frac, 0.5, transform=fig.transFigure)

    # Single method legend at the bottom of the figure.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc="lower center", ncol=len(labels),
                   frameon=False,
                   bbox_to_anchor=(0.5, 0.01))
        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    # Env name as bold vertical label on the far left.
    fig.text(
        env_x_frac, 0.5, ENV_DISPLAY[env],
        rotation=90, ha="left", va="center",
        fontsize=13, fontweight="bold", color="#222222",
    )
    out = FIG_DIR / f"degradation_v2_bars_{env.lower()}.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    # Body grid: ENA-normalized, no per-bar % labels (clean for body).
    grid = plot_all_envs_grid(
        show_pct_labels=False, metric="ena",
        out_name="degradation_v2_bars_by_method_grid.pdf",
    )
    if grid:
        print(f"[wrote] {grid}")
    # Appendix annotated companion: ENA-normalized, with % labels.
    grid_lab = plot_all_envs_grid(
        show_pct_labels=True, metric="ena",
        out_name="degradation_v2_bars_by_method_grid_with_labels.pdf",
    )
    if grid_lab:
        print(f"[wrote] {grid_lab}")
    # Appendix raw companion: raw IQM final return for direct inspection.
    grid_raw = plot_all_envs_grid(
        show_pct_labels=True, metric="raw",
        out_name="degradation_v2_bars_by_method_grid_raw.pdf",
    )
    if grid_raw:
        print(f"[wrote] {grid_raw}")
    for env in ENVS:
        out_a = plot_env(env)
        if out_a:
            print(f"[wrote] {out_a}")
        # Horizontal layout WITHOUT % labels — used by the body figure
        # for compactness. Default filename: degradation_v2_bars_by_method_<env>.pdf
        out_b = plot_env_by_method(env, layout="horizontal",
                                   show_pct_labels=False)
        if out_b:
            print(f"[wrote] {out_b}")
        # Horizontal layout WITH % labels — used by the appendix per-env
        # detail figure. Filename: <...>_with_labels.pdf
        out_b2 = plot_env_by_method(env, layout="horizontal",
                                    show_pct_labels=True,
                                    suffix_extra="_with_labels")
        if out_b2:
            print(f"[wrote] {out_b2}")
        # Vertical layout — kept around for completeness.
        out_c = plot_env_by_method(env, layout="vertical")
        if out_c:
            print(f"[wrote] {out_c}")


if __name__ == "__main__":
    main()
