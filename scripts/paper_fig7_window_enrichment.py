"""
Paper Figure 7 -- Within-event spectral restructuring: SR1-convergence signature.

Three panels:
  A. Top FDR-significant window-enrichment metrics with pooled Cohen's d and
     sign-agreement across contributing datasets.
  B. Per-band conservation test: Σ-statistic by band (theta / alpha /
     beta_low / beta_high / gamma), shown raw and partialed on per-subject
     n_peaks change.
  C. SR1-specificity: bands containing SR1 (theta + alpha) receive net positive
     recruitment; bands containing SR2-SR6 show Σ values consistent with zero.

Inputs:
  outputs/sie_window_enrichment_pooled_summary.csv          (Panel A)
  outputs/sie_wenr_followup_conservation.csv                (Panel B raw)
  outputs/sie_wenr_followup_v3_partialed_conservation.csv   (Panel B partialed)

Output:
  papers/schumann_canonical/images/fig_window_enrichment.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "papers" / "schumann_canonical" / "images" / "fig_window_enrichment.png"
SUM_PATH = ROOT / "outputs" / "sie_window_enrichment_pooled_summary.csv"
CONS_RAW_PATH = ROOT / "outputs" / "sie_wenr_followup_conservation.csv"
CONS_PART_PATH = ROOT / "outputs" / "sie_wenr_followup_v3_partialed_conservation.csv"


def load_top_survivors(n_top: int = 12) -> pd.DataFrame:
    df = pd.read_csv(SUM_PATH)
    df = df[df["fdr_survive"]].copy()
    df["abs_d"] = df["mean_d"].abs()
    df = df.sort_values("pooled_p").head(n_top).reset_index(drop=True)
    return df


def panel_a(ax, df: pd.DataFrame) -> None:
    df_sorted = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df_sorted))
    colors = ["#d62728" if d < 0 else "#1f77b4" for d in df_sorted["mean_d"]]
    ax.barh(y, df_sorted["mean_d"], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["metric"].str.replace("_", " "), fontsize=8.5)
    ax.set_xlabel("Pooled Cohen's $d$", fontsize=10)
    ax.set_title("A — Top FDR-significant window-enrichment metrics",
                 loc="left", fontweight="bold", fontsize=11)

    for i, row in df_sorted.iterrows():
        sign_pct = int(round(row["sign_agreement"] * 100))
        ann = f" {sign_pct}%"
        x = row["mean_d"]
        ha = "left" if x >= 0 else "right"
        ax.text(x, i, ann, va="center", ha=ha, fontsize=7.5, color="#333333")

    p_min = df_sorted["pooled_p"].min()
    p_max = df_sorted["pooled_p"].max()
    n_total = int(df_sorted["total_n"].max())
    p_min_exp = int(np.floor(np.log10(p_min)))
    p_max_exp = int(np.floor(np.log10(p_max)))
    ax.text(0.99, 0.02,
            f"$N = {n_total:,}$ subj-cond, 15 datasets\n"
            f"pooled $p$: $10^{{{p_min_exp}}} \\to 10^{{{p_max_exp}}}$",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            color="#555555")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def panel_b(ax) -> None:
    raw = pd.read_csv(CONS_RAW_PATH).set_index("band")
    part = pd.read_csv(CONS_PART_PATH).set_index("band")

    bands = ["theta", "alpha", "beta_low", "beta_high", "gamma"]
    raw_vals = raw.loc[bands, "mean_sum"].values
    raw_se = raw.loc[bands, "se_sum"].values
    raw_p = raw.loc[bands, "p"].values
    part_vals = part.loc[bands, "intercept"].values

    x = np.arange(len(bands))
    width = 0.36
    raw_colors = ["#1f77b4" if v >= 0 else "#d62728" for v in raw_vals]
    part_colors = ["#7faedd" if v >= 0 else "#e6878a" for v in part_vals]

    ax.bar(x - width / 2, raw_vals, width, yerr=raw_se, color=raw_colors,
           edgecolor="black", linewidth=0.5, capsize=3, label="Raw $\\Sigma$",
           alpha=0.95)
    ax.bar(x + width / 2, part_vals, width, color=part_colors,
           edgecolor="black", linewidth=0.5,
           label="Partialed on $\\Delta n_\\mathrm{peaks}$", alpha=0.65,
           hatch="///")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["theta", "alpha", r"$\beta_{\mathrm{low}}$",
                        r"$\beta_{\mathrm{high}}$", "gamma"], fontsize=9.5)
    ax.set_ylabel(r"Within-event $\Sigma$ (sum across positions)", fontsize=10)
    ax.set_title(r"B — Per-band conservation: theta + alpha gain peaks, "
                 r"$\beta_{\mathrm{low}}$ loses peaks",
                 loc="left", fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for idx, (xi, p, raw_v) in enumerate(zip(x, raw_p, raw_vals)):
        if p < 1e-6:
            exp = int(np.floor(np.log10(p)))
            label = f"$p < 10^{{{exp+1}}}$"
        elif p < 0.05:
            label = f"$p = {p:.1e}$"
        else:
            label = "ns"
        ymax = raw_v + raw_se[idx]
        y_text = ymax + 1.5 if raw_v >= 0 else ymax - 3.5
        ax.text(xi - width / 2, y_text, label, ha="center", fontsize=7.5,
                color="#333333")


def panel_c(ax) -> None:
    """SR1-specificity: only the band containing SR1 receives positive Σ.

    Display the same band-Σ pattern but re-coded to highlight which SR
    harmonic falls in which band, and which bands show net positive vs null
    response. SR1 ≈ 7.83 Hz lives in the theta-alpha boundary; SR2 ≈ 14.3 Hz
    in beta_low; SR3 ≈ 20.8 Hz, SR4 ≈ 27.3 Hz in beta_high; SR5 ≈ 33.8 Hz,
    SR6 ≈ 39.0 Hz in gamma.
    """
    raw = pd.read_csv(CONS_RAW_PATH).set_index("band")
    bands = ["theta", "alpha", "beta_low", "beta_high", "gamma"]
    sigma = raw.loc[bands, "mean_sum"].values
    p_vals = raw.loc[bands, "p"].values

    sr_in_band = {
        "theta": r"$\mathrm{SR}_1$",
        "alpha": r"$\mathrm{SR}_1$",
        "beta_low": r"$\mathrm{SR}_2$",
        "beta_high": r"$\mathrm{SR}_3, \mathrm{SR}_4$",
        "gamma": r"$\mathrm{SR}_5, \mathrm{SR}_6$",
    }
    band_labels = {
        "theta": "theta\n(4–8 Hz)",
        "alpha": "alpha\n(8–13 Hz)",
        "beta_low": r"$\beta_{\mathrm{low}}$" + "\n(13–20 Hz)",
        "beta_high": r"$\beta_{\mathrm{high}}$" + "\n(20–32 Hz)",
        "gamma": "gamma\n(33–45 Hz)",
    }
    contains_sr1 = ["theta", "alpha"]

    x = np.arange(len(bands))
    colors = []
    for b, s in zip(bands, sigma):
        if b in contains_sr1:
            colors.append("#1f77b4")
        elif s >= 0:
            colors.append("#bbbbbb")
        else:
            colors.append("#dd7777")

    ax.bar(x, sigma, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([band_labels[b] for b in bands], fontsize=8.5)
    ax.set_ylabel(r"Within-event $\Sigma$", fontsize=10)
    ax.set_title(r"C — Only the band containing $\mathrm{SR}_1$ "
                 "receives net positive recruitment",
                 loc="left", fontweight="bold", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for xi, b, s in zip(x, bands, sigma):
        ax.text(xi, max(s, 0) + 1.2, sr_in_band[b], ha="center",
                fontsize=10, fontweight="bold", color="#222222")

    ax.text(0.99, 0.02,
            "Blue: contains $\\mathrm{SR}_1$ (positive recruitment)\n"
            "Gray/red: contains $\\mathrm{SR}_2$–$\\mathrm{SR}_6$ "
            "(null or negative)",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            family="monospace", color="#555555")


def main() -> None:
    survivors = load_top_survivors(n_top=12)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.45,
                          wspace=0.28, left=0.06, right=0.98, top=0.94,
                          bottom=0.07)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, survivors)
    panel_b(ax_b)
    panel_c(ax_c)

    fig.suptitle(r"$\mathrm{SR}_1$-convergence signature in within-event "
                 r"spectral restructuring",
                 fontsize=13, fontweight="bold", y=0.985)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
