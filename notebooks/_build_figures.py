"""
Generate the figures that the README embeds.

Read the saved result JSONs (experiment_results.json, ablation_results.json,
leakage_results.json) plus the features parquet, and produce matplotlib PNGs
under `notebooks/figures/`. Re-run after any pipeline / ablation / leakage
update.

Usage:
    python notebooks/_build_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import FEATURES_PATH  # noqa: E402

RESULTS = ROOT / "results"
FIGS    = ROOT / "notebooks" / "figures"
FIGS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.facecolor"]  = "white"
plt.rcParams["axes.facecolor"]    = "white"
plt.rcParams["savefig.facecolor"] = "white"


def _save(fig: plt.Figure, name: str) -> None:
    out = FIGS / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ── Hazard ratio forest plot ────────────────────────────────────────────────

def build_hazard_forest() -> None:
    exp = json.loads((RESULTS / "experiment_results.json").read_text())
    hr  = pd.DataFrame(exp["hazard_ratios"])

    llm_set = {"sentiment_score", "frustration_level", "technical_issue",
               "value_complaint", "engagement_dropped", "positive_signal"}
    hr["kind"]  = hr["feature"].apply(lambda f: "LLM" if f in llm_set else "Behavioural")
    hr["color"] = hr["kind"].map({"LLM": "#fb923c", "Behavioural": "#60a5fa"})
    hr = hr.sort_values("HR").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, row in hr.iterrows():
        ax.hlines(i, row["HR_lower"], row["HR_upper"], color=row["color"], linewidth=2.5)
        ax.plot(row["HR"], i, marker="D", color=row["color"], markersize=8,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(1.0, ls="--", color="#6b7280", linewidth=1)
    ax.set_yticks(range(len(hr)))
    ax.set_yticklabels(hr["feature"])
    ax.set_xscale("log")
    ax.set_xlim(0.4, 3.0)
    ax.set_xlabel("Hazard ratio (log scale; 1.0 = no effect)")
    ax.set_title("Augmented Cox model, hazard ratios with 95% CIs")

    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="D", linestyle="-", color="#fb923c", label="LLM-extracted"),
              Line2D([0], [0], marker="D", linestyle="-", color="#60a5fa", label="Behavioural")]
    ax.legend(handles=legend, loc="lower right", frameon=True)

    _save(fig, "hazard_forest.png")


# ── Multi-encoder ablation: isolation rows ──────────────────────────────────

def build_ablation_isolation() -> None:
    abl = json.loads((RESULTS / "ablation_results.json").read_text())
    iso = abl["isolation"]
    labels   = list(iso.keys())
    cv_means = [iso[k]["cv_cindex"]["mean"] for k in labels]
    cv_stds  = [iso[k]["cv_cindex"]["std"]  for k in labels]
    ho       = [iso[k]["holdout_cindex"]    for k in labels]

    palette = ["#6b7280", "#60a5fa", "#a78bfa", "#fb923c"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, cv_means, yerr=cv_stds, color=palette, alpha=0.85,
                  capsize=5, error_kw=dict(color="#374151", lw=1.5))
    for bar, m in zip(bars, cv_means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.015, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10, fontfamily="monospace")
    ax.scatter(x, ho, marker="*", s=180, color="#facc15",
               edgecolor="#1f2937", linewidth=1.0, zorder=5, label="Hold-out (20%)")
    ax.axhline(0.5, ls=":", color="#9ca3af", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("C-index")
    ax.set_title("Each text encoder alone (on top of behavioural floor)")
    ax.legend(loc="lower right")
    _save(fig, "ablation_isolation.png")


# ── Multi-encoder ablation: additive rows ───────────────────────────────────

def build_ablation_additive() -> None:
    abl = json.loads((RESULTS / "ablation_results.json").read_text())
    add = abl["additive"]
    labels   = list(add.keys())
    cv_means = [add[k]["cv_cindex"]["mean"] for k in labels]
    cv_stds  = [add[k]["cv_cindex"]["std"]  for k in labels]
    ho       = [add[k]["holdout_cindex"]    for k in labels]
    base     = cv_means[0]
    deltas   = [v - base for v in cv_means]

    def step_color(d):
        if d == 0:    return "#60a5fa"
        if d > 0.25:  return "#4ade80"
        if d > 0.10:  return "#a78bfa"
        return "#22d3ee"

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, cv_means, yerr=cv_stds, color=[step_color(d) for d in deltas],
                  alpha=0.85, capsize=4, error_kw=dict(color="#374151", lw=1.2))
    for bar, m, d in zip(bars, cv_means, deltas):
        label = f"{m:.3f}" if d == 0 else f"{m:.3f}\n(+{d:.3f})"
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.015, label,
                ha="center", va="bottom", fontsize=9, fontfamily="monospace")
    ax.scatter(x, ho, marker="*", s=180, color="#facc15",
               edgecolor="#1f2937", linewidth=1.0, zorder=5, label="Hold-out (20%)")
    ax.axhline(base, ls="--", color="#9ca3af", linewidth=1, label="baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("C-index")
    ax.set_title("Stacking text encoders, does adding the next one help?")
    ax.legend(loc="lower right")
    _save(fig, "ablation_additive.png")


# ── Leakage decomposition (horizontal stacked bar) ──────────────────────────

def build_leakage_decomp() -> None:
    lk = json.loads((RESULTS / "leakage_results.json").read_text())
    var = lk["variants"]
    base       = var["baseline"]["cv_cindex"]["mean"]
    non_pol_d  = var["non_polarity"]["delta_vs_baseline"]
    full_d     = var["full_augmented"]["delta_vs_baseline"]
    leakage_d  = full_d - non_pol_d

    fig, ax = plt.subplots(figsize=(9, 2.0))
    ax.barh(0, base, color="#4b5563", label=f"Behavioural floor ({base:.3f})")
    ax.barh(0, non_pol_d, left=base, color="#4ade80",
            label=f"Defensible non-polarity lift (+{non_pol_d:.3f})")
    ax.barh(0, leakage_d, left=base + non_pol_d, color="#f87171",
            label=f"Polarity lift (mostly label leakage) (+{leakage_d:.3f})")
    ax.text(base / 2, 0, f"{base:.3f}", ha="center", va="center",
            color="white", fontsize=10, fontfamily="monospace")
    ax.text(base + non_pol_d / 2, 0, f"+{non_pol_d:.3f}", ha="center", va="center",
            color="#0f1117", fontsize=10, fontfamily="monospace")
    ax.text(base + non_pol_d + leakage_d / 2, 0, f"+{leakage_d:.3f}",
            ha="center", va="center", color="#0f1117", fontsize=10, fontfamily="monospace")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("C-index")
    ax.set_title("Where does the augmented C-index actually come from?")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.95), ncol=3, frameon=False, fontsize=9)
    _save(fig, "leakage_decomposition.png")


# ── Kaplan-Meier survival by event class ────────────────────────────────────

def build_km_by_event() -> None:
    from lifelines import KaplanMeierFitter
    df = pd.read_parquet(FEATURES_PATH)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for value, color, label in [(1, "#f87171", "Churned (recommend=False)"),
                                (0, "#4ade80", "Retained (recommend=True)")]:
        kmf = KaplanMeierFitter().fit(
            df.loc[df["event"] == value, "duration"],
            df.loc[df["event"] == value, "event"],
            label=label,
        )
        kmf.plot_survival_function(ax=ax, color=color, ci_alpha=0.15)

    ax.axhline(0.5, ls=":", color="#9ca3af", linewidth=1)
    ax.set_xlabel("Hours played (playtime_forever)")
    ax.set_ylabel("S(t), fraction still 'alive'")
    ax.set_title("Kaplan-Meier survival curves by churn outcome")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    _save(fig, "km_by_event.png")


def main() -> None:
    print(f"writing figures to {FIGS.relative_to(ROOT)}/")
    build_hazard_forest()
    build_ablation_isolation()
    build_ablation_additive()
    build_leakage_decomp()
    build_km_by_event()


if __name__ == "__main__":
    main()
