"""
Paper Figure 8 -- Posterior-temporoparietal substrate of canonical ignitions.

Three panels:
  A. Scalp posterior-anterior gradient in LEMON Q4: per-subject scatter of
     anterior vs posterior SR1 event/baseline ratio with diagonal reference;
     contrast statistics annotated.
  B. sLORETA top regions by median per-vertex SR1 event/baseline ratio
     (LEMON adults, fsaverage ico-5, Desikan parcellation).
  C. Cross-cohort topographic variation: cohort-mean per-channel SR1
     event/baseline ratio rendered as compact bar summaries with the dominant
     region annotated, illustrating that frequency stays at 7.687 Hz across
     cohorts while topography is cohort-specific.

Inputs:
  outputs/schumann/images/coupling/posterior_sr1_crosscohort.csv  (Panels A, C)
  outputs/schumann/images/source/Q4_SR1_label_ranking.csv         (Panel B)

Output:
  papers/schumann_canonical/images/fig_posterior_substrate.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "papers" / "schumann_canonical" / "images" / "fig_posterior_substrate.png"
PA_PATH = ROOT / "outputs" / "schumann" / "images" / "coupling" / "posterior_sr1_crosscohort.csv"
RANK_PATH = ROOT / "outputs" / "schumann" / "images" / "source" / "Q4_SR1_label_ranking.csv"


def panel_a(ax) -> None:
    df = pd.read_csv(PA_PATH)
    df = df[df["cohort"] == "lemon"].copy()
    df = df.dropna(subset=["sr1_ratio_posterior", "sr1_ratio_anterior"])

    p_vals = df["sr1_ratio_posterior"]
    a_vals = df["sr1_ratio_anterior"]
    contrast = df["sr1_contrast"]
    pct_above = 100 * (contrast > 0).mean()
    median_contrast = float(contrast.median())
    try:
        _, wp = wilcoxon(contrast)
    except ValueError:
        wp = float("nan")

    ax.scatter(a_vals, p_vals, s=14, c="#1f77b4", alpha=0.55,
               edgecolor="white", linewidth=0.4)
    lo = 0.0
    hi = float(max(a_vals.max(), p_vals.max())) * 1.05
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Anterior $\mathrm{SR}_1$ event/baseline ratio", fontsize=10)
    ax.set_ylabel(r"Posterior $\mathrm{SR}_1$ event/baseline ratio", fontsize=10)
    ax.set_title("A — Scalp posterior–anterior gradient in LEMON",
                 loc="left", fontweight="bold", fontsize=11)

    if not np.isnan(wp) and wp < 1e-6:
        exp = int(np.floor(np.log10(wp)))
        p_str = f"$p < 10^{{{exp + 1}}}$"
    elif np.isnan(wp):
        p_str = ""
    else:
        p_str = f"$p = {wp:.2g}$"
    ax.text(0.04, 0.96,
            f"$N = {len(df)}$ subjects\n"
            f"posterior > anterior in {pct_above:.0f}%\n"
            f"median contrast = {median_contrast:+.2f}\n"
            f"Wilcoxon {p_str}",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.3"))
    ax.text(0.99, 0.02,
            "All-events aggregation; the $\\mathrm{Q}_4$\n"
            "canonical-quartile contrast is sharper\n"
            "(Section 2.6.4: $+1.17$, 72%, $p = 2.2 \\times 10^{-8}$).",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color="#555555")
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="box")


def panel_b(ax) -> None:
    df = pd.read_csv(RANK_PATH)
    df["label_full"] = df["label"] + "-" + df["hemi"]
    df = df.sort_values("ratio_median", ascending=False).head(15).reset_index(drop=True)
    df = df.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(df))
    ax.barh(y, df["ratio_median"], color="#2a9d8f", alpha=0.85,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label_full"], fontsize=8.5)
    ax.set_xlabel(r"Median per-vertex $\mathrm{SR}_1$ event/baseline ratio",
                  fontsize=10)
    ax.set_title("B — sLORETA top regions (LEMON adults, fsaverage ico-5)",
                 loc="left", fontweight="bold", fontsize=11)
    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlim(left=min(0.95, float(df["ratio_median"].min()) - 0.02))

    for i, row in df.iterrows():
        ax.text(row["ratio_median"] + 0.003, i,
                f"  {row['ratio_median']:.3f}", va="center", fontsize=7.5,
                color="#444444")
    ax.text(0.99, 0.02,
            "Frontal regions consistently lowest-ranked (not shown).\n"
            "Desikan-Killiany parcellation; rank 1–15.",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            color="#555555")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def panel_c(ax) -> None:
    """Cross-cohort topographic variation summary.

    For each cohort, compute the per-subject posterior - anterior log-ratio
    contrast and plot a violin / boxplot summary. This conveys that all
    cohorts express a posterior-leaning substrate while individual cohorts
    differ in magnitude. The text annotates the dominant cortical
    expression noted in the per-cohort coupling reports.
    """
    df = pd.read_csv(PA_PATH)
    df = df.dropna(subset=["sr1_ratio_posterior", "sr1_ratio_anterior"]).copy()
    df["log_contrast"] = np.log10(df["sr1_ratio_posterior"].clip(lower=0.1)) \
        - np.log10(df["sr1_ratio_anterior"].clip(lower=0.1))
    df = df[df["cohort"] != "hbn_all"]

    cohort_order = ["lemon", "tdbrain", "hbn_R1", "hbn_R2",
                    "hbn_R3", "hbn_R4", "hbn_R6"]
    cohort_labels = {
        "lemon": "LEMON\nposterior",
        "tdbrain": "TDBRAIN\ncentral",
        "hbn_R1": "HBN R1\ncentral",
        "hbn_R2": "HBN R2\ncentral",
        "hbn_R3": "HBN R3\ncentral",
        "hbn_R4": "HBN R4\ncentral",
        "hbn_R6": "HBN R6\ncentral",
    }
    data = [df[df["cohort"] == c]["log_contrast"].values for c in cohort_order]
    positions = np.arange(len(cohort_order))

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.4))
    palette = ["#1f77b4", "#9467bd", "#ff7f0e", "#ff7f0e", "#ff7f0e",
               "#ff7f0e", "#ff7f0e"]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    for i, vals in enumerate(data):
        if len(vals):
            jit = (np.random.RandomState(i + 1).rand(len(vals)) - 0.5) * 0.18
            ax.scatter(np.full_like(vals, positions[i], dtype=float) + jit,
                       vals, s=4, alpha=0.35, color="#333333", edgecolor="none")

    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([cohort_labels[c] for c in cohort_order], fontsize=8)
    ax.set_ylabel(r"posterior − anterior $\log_{10}$ ratio",
                  fontsize=10)
    ax.set_title(r"C — Cross-cohort topographic variation; "
                 r"frequency conserved at $7.687 \pm 0.050$ Hz",
                 loc="left", fontweight="bold", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.02,
            "Dominant cortical expression noted under each cohort label.\n"
            "Pediatric central-scalp dominance partly reflects EGI 129-channel reference.",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            color="#555555")


def main() -> None:
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.15],
                          hspace=0.42, wspace=0.32, left=0.07, right=0.98,
                          top=0.93, bottom=0.07)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    fig.suptitle("Posterior–temporoparietal substrate of canonical ignitions",
                 fontsize=13, fontweight="bold", y=0.985)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
