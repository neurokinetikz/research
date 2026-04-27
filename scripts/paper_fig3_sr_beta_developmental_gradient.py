#!/usr/bin/env python3
"""
Paper Figure 3 — SR × β developmental gradient.

Two-panel figure:
  Panel A: per-cohort cross-subject SR × β coupling Spearman ρ, ordered
           by regime: clinical → adult EC → adult EO → pediatric.
           Shows the regime-level gradient (median ρ ≈ 0.16 clinical,
           ~0.31 adult EC, ~0.21 adult EO, ~0.72 pediatric).
  Panel B: pooled within-pediatric age-binned ρ (5-7, 8-10, 11-13,
           14-17 yr) showing the monotonic decline, with the per-release
           HBN ρ values overlaid as small markers at each release's
           mean age.

Sources:
  - Per-cohort ρ values from the wiki cross-cohort meta (hand-coded
    for the four adult/clinical cohorts; computed from beta_peak_iaf_coupling
    CSVs for the 11 HBN releases).
  - outputs/2026-04-25-hbn-within-pediatric-age-gradient.csv (the
    age-binned ρ).

Stdlib + matplotlib only.
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import os
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parent.parent
SCOPE = os.environ.get("SCOPE", "all")  # 'all', 'q4', 'sw'
SUFFIX = {"all": "", "q4": "_q4", "sw": "_sw"}[SCOPE]
_TAG = "" if SCOPE == "all" else f"_{SCOPE}"
AGE_CSV = REPO / f"outputs/2026-04-25-hbn-within-pediatric-age-gradient{_TAG}.csv"
PSD_DIR = REPO / "outputs/schumann/images/psd_timelapse"
OUT_PNG = REPO / f"papers/schumann_canonical/images/fig3_sr_beta_developmental_gradient{_TAG}.png"


def _rho_from_cohort_csv(cohort_tag):
    """Compute Spearman ρ(SR_peak_ratio, β_peak_ratio) for one cohort under SCOPE.
    Returns (rho, n) or (None, 0) on failure."""
    import csv
    p = PSD_DIR / f"{cohort_tag}_composite/beta_peak_iaf_coupling{SUFFIX}.csv"
    if not p.exists():
        return None, 0
    sr_ratio, beta_ratio = [], []
    with p.open() as fh:
        for r in csv.DictReader(fh):
            try:
                s = float(r["sr_peak_ratio"])
                b = float(r["beta_peak_ratio"])
            except (KeyError, ValueError):
                continue
            sr_ratio.append(s)
            beta_ratio.append(b)
    if len(sr_ratio) < 5:
        return None, len(sr_ratio)
    rho, _ = spearmanr(sr_ratio, beta_ratio)
    return float(rho), len(sr_ratio)

# Cohort-level ρ values from finding-composite-v2-cross-cohort.md, table
# at lines 110-127.  Order chosen to display the clinical → adult EC →
# adult EO → pediatric gradient.
# Cohort definitions (label, regime, cohort_tag for CSV lookup, color).
# Spearman ρ and n are computed dynamically from the per-cohort CSV under SCOPE.
COHORT_DEFS = [
    ("tdbrain EC", "clinical",  "tdbrain",     "#e31a1c"),
    ("tdbrain EO", "clinical",  "tdbrain_EO",  "#fb9a99"),
    ("Dortmund",   "adult EC",  "dortmund",    "#1f78b4"),
    ("CHBMP",      "adult EC",  "chbmp",       "#a6cee3"),
    ("LEMON EO",   "adult EO",  "lemon_EO",    "#a6cee3"),
    ("LEMON EC",   "adult EC",  "lemon",       "#1f78b4"),
    ("HBN R9",     "pediatric", "hbn_R9",      "#ff7f00"),
    ("HBN R10",    "pediatric", "hbn_R10",     "#ff7f00"),
    ("HBN R4",     "pediatric", "hbn_R4",      "#ff7f00"),
    ("HBN R3",     "pediatric", "hbn_R3",      "#ff7f00"),
    ("HBN R5",     "pediatric", "hbn_R5",      "#ff7f00"),
    ("HBN R8",     "pediatric", "hbn_R8",      "#ff7f00"),
    ("HBN R6",     "pediatric", "hbn_R6",      "#ff7f00"),
    ("HBN R7",     "pediatric", "hbn_R7",      "#ff7f00"),
    ("HBN R2",     "pediatric", "hbn_R2",      "#ff7f00"),
    ("HBN R1",     "pediatric", "hbn_R1",      "#ff7f00"),
    ("HBN R11",    "pediatric", "hbn_R11",     "#ff7f00"),
]
# Compute ρ and n for each cohort under current SCOPE; drop cohorts with no data.
COHORT_RHO = []
for label, regime, tag, color in COHORT_DEFS:
    rho, n = _rho_from_cohort_csv(tag)
    if rho is not None:
        COHORT_RHO.append((label, regime, rho, n, color))


def load_age_binned_rho():
    """Return list of (age_bin_label, age_mean, n, rho) for pooled bins
    and list of (release_id, n, rho, age_mean) for per-release rows."""
    pooled = []
    per_release = []
    with AGE_CSV.open() as fh:
        section = None
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("# Within-pediatric"):
                section = "header"
                continue
            if line.startswith("# Per-release between-subject"):
                section = "per_release"
                continue
            if line.startswith("# Pooled age-binned"):
                section = "pooled"
                continue
            if line.startswith("# Per-release OLS"):
                section = "ols"
                continue
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split(",")
            if section == "per_release" and parts[0] != "release":
                # release,n,rho_spearman,p_spearman,r_pearson,p_pearson,age_mean,age_sd
                try:
                    per_release.append(
                        (parts[0], int(parts[1]), float(parts[2]), float(parts[6]))
                    )
                except (IndexError, ValueError):
                    pass
            elif section == "pooled" and parts[0] != "age_bin":
                try:
                    pooled.append(
                        (parts[0], float(parts[1]), int(parts[2]), float(parts[3]))
                    )
                except (IndexError, ValueError):
                    pass
    return pooled, per_release


def main():
    pooled, per_release = load_age_binned_rho()
    print(f"Loaded {len(pooled)} pooled age bins and {len(per_release)} releases")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5),
                              gridspec_kw={"wspace": 0.30,
                                            "width_ratios": [1.4, 1.0]})

    # ---------- Panel A: per-cohort bar chart ----------
    ax = axes[0]
    labels = [c[0] for c in COHORT_RHO]
    rhos = [c[2] for c in COHORT_RHO]
    cols = [c[4] for c in COHORT_RHO]
    xs = list(range(len(labels)))
    bars = ax.bar(xs, rhos, color=cols, edgecolor="white", lw=0.8, zorder=2)

    # Add ρ value above each bar
    for x, r in zip(xs, rhos):
        ax.text(x, r + 0.015, f"{r:.2f}", ha="center", va="bottom",
                 fontsize=8, color="#222")

    # Regime separators
    sep_after = [0, 3, 4]
    sep_x = [s + 0.5 for s in sep_after]
    for sx in sep_x:
        ax.axvline(sx, color="#bbbbbb", lw=0.8, ls="-", alpha=0.6, zorder=1)

    # Regime labels
    regime_spans = [
        (0, 0, "Clinical"),
        (1, 3, "Adult EC / EO"),
        (4, 14, "Pediatric (HBN R1–R11)"),
    ]
    for x0, x1, lab in regime_spans:
        ax.text((x0 + x1) / 2, -0.10, lab, ha="center", va="top",
                 fontsize=10, color="#444", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cross-subject SR × β coupling ρ (Spearman)", fontsize=11)
    ped_rhos = [r for _, regime, r, _, _ in COHORT_RHO if regime == "pediatric"]
    adult_rhos = [r for _, regime, r, _, _ in COHORT_RHO if regime in ("adult EC", "adult EO")]
    title_a = (f"A — {len(ped_rhos)}/{len(ped_rhos)} HBN releases: "
               f"pediatric ρ = {min(ped_rhos):.2f}–{max(ped_rhos):.2f} "
               f"vs adult ρ = {min(adult_rhos):.2f}–{max(adult_rhos):.2f}")
    ax.set_title(title_a,
                  loc="left", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, 0.95)
    ax.set_xlim(-0.6, len(labels) - 0.4)

    # ---------- Panel B: age-binned ρ ----------
    ax = axes[1]
    # Filter to pediatric bins (5-7 through 14-17; exclude 18-21 because
    # only 51 subjects across all releases, narrower coverage).
    ped_bins = [b for b in pooled if b[1] < 18]

    if ped_bins:
        bin_x = [b[1] for b in ped_bins]
        bin_y = [b[3] for b in ped_bins]
        bin_n = [b[2] for b in ped_bins]
        bin_lab = [b[0] for b in ped_bins]
        # Bar plot for clarity
        ax.bar(bin_x, bin_y, width=2.4, color="#ff7f00", edgecolor="white",
                lw=1.0, alpha=0.85, zorder=2)
        for x, y, n, lab in zip(bin_x, bin_y, bin_n, bin_lab):
            ax.text(x, y + 0.015, f"ρ={y:.2f}\nn={n}", ha="center",
                     va="bottom", fontsize=8, color="#222")

    # Per-release scatter overlay
    for rel, n, rho, age in per_release:
        ax.plot(age, rho, "o", color="#222", ms=4, mec="white", mew=0.6,
                 zorder=4, alpha=0.85)

    # Adult-EC reference band
    ax.axhspan(0.16, 0.45, color="#1f78b4", alpha=0.10, zorder=1)
    ax.text(15.5, 0.30, "Adult ρ\nrange\n0.16–0.45",
             ha="center", va="center", fontsize=9, color="#1f78b4",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f78b4", lw=0.6))

    ax.set_xlabel("Age (years)", fontsize=11)
    ax.set_ylabel("SR × β coupling ρ (Spearman)", fontsize=11)
    ax.set_title("B — Within-pediatric age gradient: ρ declines monotonically with age",
                  loc="left", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.set_xlim(4, 18)
    ax.grid(alpha=0.25)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#ff7f00", alpha=0.85,
                       label="Pooled age-binned ρ"),
        plt.Line2D([], [], marker="o", color="#222", lw=0, ms=5,
                    label="Per-release HBN ρ (n=11)"),
        plt.Rectangle((0, 0), 1, 1, fc="#1f78b4", alpha=0.10,
                       label="Adult cohort range"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True)

    fig.suptitle("Cross-cohort SR × β coupling: developmental de-coupling gradient",
                  fontsize=12, fontweight="bold", y=1.005)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
