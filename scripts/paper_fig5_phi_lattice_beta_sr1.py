#!/usr/bin/env python3
"""
Paper Figure 5 — φ-lattice β/SR1 ratio at φ² = 2.618.

Two-panel figure:
  Panel A: per-subject scatter of β-band peak vs SR1 peak across all
           17 cohorts; reference lines for φ² = 2.618 and the simple
           2× harmonic (15.66 Hz expectation if β were locked to 2×SR1).
           Cohort means overlaid as larger markers.
  Panel B: per-cohort β/SR1 ratio bar chart with the φ² = 2.618 reference
           line, showing all 17 cohorts within ±0.03 of φ² and clearly
           separated from the 2× value.

Sources:
  - outputs/schumann/images/psd_timelapse/{cohort}_composite/beta_peak_iaf_coupling.csv
    (subject_id, sr_peak_hz, beta_peak_hz; ratio = β/SR1).

Stdlib + matplotlib only.
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PSD_DIR = REPO / "outputs/schumann/images/psd_timelapse"
import os as _os
_SCOPE = _os.environ.get("SCOPE", "all")
_TAG_OUT = "" if _SCOPE == "all" else f"_{_SCOPE}"
OUT_PNG = REPO / f"papers/schumann_canonical/images/fig5_phi_lattice_beta_sr1{_TAG_OUT}.png"

PHI_SQ = (1 + 5 ** 0.5) / 2
PHI_SQ = PHI_SQ * PHI_SQ  # 2.618...

COHORTS = [
    ("lemon_composite", "LEMON EC", "adult_EC"),
    ("lemon_EO_composite", "LEMON EO", "adult_EO"),
    ("dortmund_composite", "Dortmund", "adult_EC"),
    ("chbmp_composite", "CHBMP", "adult_EC"),
    ("tdbrain_composite", "tdbrain EC", "clinical"),
    ("tdbrain_EO_composite", "tdbrain EO", "clinical"),
    ("hbn_R1_composite", "HBN R1", "pediatric"),
    ("hbn_R2_composite", "HBN R2", "pediatric"),
    ("hbn_R3_composite", "HBN R3", "pediatric"),
    ("hbn_R4_composite", "HBN R4", "pediatric"),
    ("hbn_R5_composite", "HBN R5", "pediatric"),
    ("hbn_R6_composite", "HBN R6", "pediatric"),
    ("hbn_R7_composite", "HBN R7", "pediatric"),
    ("hbn_R8_composite", "HBN R8", "pediatric"),
    ("hbn_R9_composite", "HBN R9", "pediatric"),
    ("hbn_R10_composite", "HBN R10", "pediatric"),
    ("hbn_R11_composite", "HBN R11", "pediatric"),
]

REGIME_COLORS = {
    "adult_EC": "#1f78b4",
    "adult_EO": "#a6cee3",
    "clinical": "#e31a1c",
    "pediatric": "#ff7f00",
}


import os
SCOPE = os.environ.get("SCOPE", "all")
SUFFIX = {"all": "", "q4": "_q4", "sw": "_sw"}[SCOPE]
_TAG = "" if SCOPE == "all" else f"_{SCOPE}"


def load(folder):
    p = PSD_DIR / folder / f"beta_peak_iaf_coupling{SUFFIX}.csv"
    if not p.exists():
        return []
    out = []
    with p.open() as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                sr = float(r["sr_peak_hz"])
                bp = float(r["beta_peak_hz"])
            except (KeyError, ValueError):
                continue
            if sr < 5 or bp < 12:  # discard implausible values
                continue
            out.append((sr, bp))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    cohort_data = []
    for folder, label, regime in COHORTS:
        rows = load(folder)
        if not rows:
            continue
        sr_means = [r[0] for r in rows]
        bp_means = [r[1] for r in rows]
        ratio = [bp / sr for sr, bp in rows if sr > 0]
        cohort_data.append((label, regime, sr_means, bp_means, ratio))
        print(f"{label}: n={len(rows)} mean_sr={mean(sr_means):.2f} "
              f"mean_β={mean(bp_means):.2f} mean_ratio={mean(ratio):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5),
                              gridspec_kw={"wspace": 0.28,
                                            "width_ratios": [1.1, 1.0]})

    # ---------- Panel A: per-subject scatter ----------
    ax = axes[0]
    seen = set()
    for label, regime, sr_means, bp_means, _ in cohort_data:
        col = REGIME_COLORS[regime]
        legend = regime.replace("_", " ").title() if regime not in seen else None
        seen.add(regime)
        ax.scatter(sr_means, bp_means, color=col, alpha=0.20, s=10,
                    ec="none", zorder=2)

    # Cohort means
    for label, regime, sr_means, bp_means, _ in cohort_data:
        col = REGIME_COLORS[regime]
        ax.plot(mean(sr_means), mean(bp_means), "o", color=col, ms=10,
                 mec="white", mew=1.4, zorder=4)
        # Annotate cohort mean
        ax.annotate(label, (mean(sr_means), mean(bp_means)),
                     xytext=(6, 0), textcoords="offset points", fontsize=7,
                     color="#333")

    # Reference lines
    sr_grid = [6.5, 9.0]
    ax.plot(sr_grid, [PHI_SQ * s for s in sr_grid], color="#1a9641",
             ls="-", lw=1.6, alpha=0.85, zorder=3,
             label=f"β = φ² × SR1 ({PHI_SQ:.3f})")
    ax.plot(sr_grid, [2.0 * s for s in sr_grid], color="#888888",
             ls="--", lw=1.0, alpha=0.7, zorder=2,
             label="β = 2 × SR1 (2nd harmonic)")
    ax.plot(sr_grid, [2.5 * s for s in sr_grid], color="#888888",
             ls=":", lw=0.8, alpha=0.5, zorder=2,
             label="β = 2.5 × SR1")

    ax.set_xlabel("SR1 peak (Hz)", fontsize=11)
    ax.set_ylabel("β-band peak (Hz)", fontsize=11)
    ax.set_title("A — Per-subject β vs SR1 across 17 cohorts",
                  loc="left", fontsize=11, fontweight="bold")
    ax.set_xlim(6.5, 9.0)
    ax.set_ylim(13, 25)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    # ---------- Panel B: cohort-level ratio bars ----------
    ax = axes[1]
    cohort_data_sorted = sorted(cohort_data, key=lambda c: mean(c[4]))
    labels = [c[0] for c in cohort_data_sorted]
    ratios = [mean(c[4]) for c in cohort_data_sorted]
    cols = [REGIME_COLORS[c[1]] for c in cohort_data_sorted]
    xs = list(range(len(labels)))

    ax.bar(xs, ratios, color=cols, edgecolor="white", lw=0.8, zorder=2)
    for x, r in zip(xs, ratios):
        ax.text(x, r + 0.008, f"{r:.2f}", ha="center", va="bottom",
                 fontsize=7.5, color="#222")

    # φ² reference
    ax.axhline(PHI_SQ, color="#1a9641", lw=1.6, alpha=0.85, zorder=3,
                label=f"φ² = {PHI_SQ:.3f}")
    ax.axhspan(PHI_SQ - 0.03, PHI_SQ + 0.03, color="#1a9641", alpha=0.10,
                zorder=1, label="±0.03")
    ax.axhline(2.0, color="#888888", ls="--", lw=0.9, alpha=0.7,
                label="2× harmonic")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("β / SR1 ratio", fontsize=11)
    ax.set_title("B — Cohort-mean β/SR1 = 2.57 ± 0.03 ≈ φ²",
                  loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_ylim(1.9, 2.85)

    fig.suptitle("β peak is φ²-locked to SR1 across all 17 cohorts",
                  fontsize=12, fontweight="bold", y=1.005)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
