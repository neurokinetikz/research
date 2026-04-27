#!/usr/bin/env python3
"""
Paper Figure 4 — Four cohort regimes (quantitative ordination).

Two-panel figure showing how the 16 cohorts resolve into four regimes
on key axes:

  Panel A: IEI median (s) vs envelope nadir z (event amplitude).
           Distinguishes regime 3 (clinical, fast/sharp) from regime 4
           (pediatric, slow/shallow) and regimes 1-2 (adult, slow/deep).
  Panel B: SR × β coupling ρ vs β-SR temporal lag (s).
           Distinguishes regime ordering — adult EC β LAGS, EO co-peaks,
           clinical β LEADS, pediatric co-peaks-with-strong-coupling.

Source: hand-coded values from finding-composite-v2-cross-cohort.md
        §"Four cohort regimes" table (lines 79-94) and the SR×β coupling
        table (lines 110-127).

Stdlib + matplotlib only.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

import os
REPO = Path(__file__).resolve().parent.parent
SCOPE = os.environ.get("SCOPE", "all")  # 'all' or 'sw'
SUFFIX = {"all": "", "sw": "_sw"}[SCOPE]
_TAG = "" if SCOPE == "all" else f"_{SCOPE}"
OUT_PNG = REPO / f"papers/schumann_canonical/images/fig4_four_cohort_regimes{_TAG}.png"

# Cohort definitions (label, regime, cohort_tag for SW CSV lookup,
# IEI median, env_nadir z, beta_sr_lag).
# IEI/env_nadir/beta_sr_lag are still hardcoded all-events values (TBD: per-
# cohort SW peri-event analysis to update). sr_beta_rho is computed from the
# SW CSVs at SCOPE=sw, falling back to all-events otherwise.
COHORT_DEFS = [
    ("LEMON EC",   "adult_EC",  "lemon",      82.0, -0.65, 12.0),
    ("Dortmund",   "adult_EC",  "dortmund",   61.0, -0.65, 12.0),
    ("CHBMP",      "adult_EC",  "chbmp",      85.0, -0.65, 12.0),
    ("LEMON EO",   "adult_EO",  "lemon_EO",   85.3, -0.64, 0.0),
    ("tdbrain EC", "clinical",  "tdbrain",    44.4, -0.70, -6.0),
    ("tdbrain EO", "clinical",  "tdbrain_EO", 44.3, -0.57, -6.0),
    ("HBN R1",     "pediatric", "hbn_R1",     84.0, -0.40, 0.0),
    ("HBN R2",     "pediatric", "hbn_R2",     84.0, -0.40, 0.0),
    ("HBN R3",     "pediatric", "hbn_R3",     82.0, -0.40, 0.0),
    ("HBN R4",     "pediatric", "hbn_R4",     80.4, -0.40, 0.0),
    ("HBN R5",     "pediatric", "hbn_R5",     82.0, -0.40, 0.0),
    ("HBN R6",     "pediatric", "hbn_R6",     84.0, -0.40, 0.0),
    ("HBN R7",     "pediatric", "hbn_R7",     87.0, -0.40, 0.0),
    ("HBN R8",     "pediatric", "hbn_R8",     82.0, -0.40, 0.0),
    ("HBN R9",     "pediatric", "hbn_R9",     82.0, -0.40, 1.0),
    ("HBN R10",    "pediatric", "hbn_R10",    82.0, -0.40, 0.0),
    ("HBN R11",    "pediatric", "hbn_R11",    82.8, -0.40, 0.0),
]


def _rho_from_csv(tag):
    """Read sr_peak_ratio × beta_peak_ratio Spearman ρ from CSV under SCOPE.
    Returns ρ or None if CSV missing or insufficient data."""
    import csv
    p = REPO / f"outputs/schumann/images/psd_timelapse/{tag}_composite/beta_peak_iaf_coupling{SUFFIX}.csv"
    if not p.exists():
        return None
    sr_ratio, beta_ratio = [], []
    with p.open() as fh:
        for r in csv.DictReader(fh):
            try:
                s = float(r["sr_peak_ratio"]); b = float(r["beta_peak_ratio"])
            except (KeyError, ValueError):
                continue
            sr_ratio.append(s); beta_ratio.append(b)
    if len(sr_ratio) < 5:
        return None
    # Spearman via ranking
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0] * len(xs)
        for r, i in enumerate(order):
            ranks[i] = r + 1
        return ranks
    sx, sy = rank(sr_ratio), rank(beta_ratio)
    n = len(sx); mx = sum(sx)/n; my = sum(sy)/n
    num = sum((sx[i]-mx)*(sy[i]-my) for i in range(n))
    dx = (sum((sx[i]-mx)**2 for i in range(n)))**0.5
    dy = (sum((sy[i]-my)**2 for i in range(n)))**0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else None

# Build COHORTS list with sr_beta_rho computed under SCOPE
COHORTS = []
for label, regime, tag, iei, env_nadir, lag in COHORT_DEFS:
    rho = _rho_from_csv(tag)
    if rho is None:
        continue
    COHORTS.append((label, regime, iei, env_nadir, rho, lag))

REGIME_INFO = {
    "adult_EC":  ("Regime 1 — Adult EC (slow harmonic cascade)", "#1f78b4"),
    "adult_EO":  ("Regime 2 — Adult EO (co-peak β-SR)",            "#a6cee3"),
    "clinical":  ("Regime 3 — Clinical (fast, anticipatory β)",     "#e31a1c"),
    "pediatric": ("Regime 4 — Pediatric (strong SR × β coupling)",  "#ff7f00"),
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5),
                              gridspec_kw={"wspace": 0.28})

    # ---------- Panel A: IEI vs envelope nadir ----------
    ax = axes[0]
    seen_regimes = set()
    for label, regime, iei, env_nadir, _, _ in COHORTS:
        title, col = REGIME_INFO[regime]
        legend = title if regime not in seen_regimes else None
        seen_regimes.add(regime)
        ax.scatter(iei, env_nadir, color=col, s=130, alpha=0.85,
                   ec="white", lw=1.2, label=legend, zorder=3)
        ax.annotate(label, (iei, env_nadir), xytext=(5, 5),
                     textcoords="offset points", fontsize=7.5, color="#333")

    ax.set_xlabel("IEI median (s)", fontsize=11)
    ax.set_ylabel("Envelope nadir z (deeper ↓)", fontsize=11)
    ax.invert_yaxis()
    ax.set_title("A — Event timing × envelope swing",
                  loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_xlim(35, 110)
    ax.set_ylim(-0.30, -0.90)  # explicit ymin/ymax with inverted axis

    # Inline regime annotation INSIDE axes
    ax.text(45, -0.86, "clinical\n(fast, sharp)", fontsize=8.5,
             color="#e31a1c", fontweight="bold", ha="center", va="bottom")
    ax.text(75, -0.78, "adult EC / EO\n(slow, deep)", fontsize=8.5,
             color="#1f78b4", fontweight="bold", ha="center", va="bottom")
    ax.text(99, -0.34, "pediatric\n(slow, shallow)", fontsize=8.5,
             color="#ff7f00", fontweight="bold", ha="center", va="bottom")

    # ---------- Panel B: SR×β ρ vs β-SR lag ----------
    ax = axes[1]
    seen_regimes = set()
    for label, regime, _, _, rho, lag in COHORTS:
        title, col = REGIME_INFO[regime]
        legend = title if regime not in seen_regimes else None
        seen_regimes.add(regime)
        ax.scatter(lag, rho, color=col, s=130, alpha=0.85,
                   ec="white", lw=1.2, label=legend, zorder=3)
        ax.annotate(label, (lag, rho), xytext=(5, 5),
                     textcoords="offset points", fontsize=7.5, color="#333")

    ax.axvline(0, color="#444", lw=0.7, ls=":")
    ax.text(0.2, 0.05, "β co-peaks with SR", fontsize=8, color="#444",
             rotation=0, va="bottom")
    ax.text(-7, 0.05, "β LEADS SR", fontsize=8, color="#444",
             rotation=0, va="bottom", ha="left")
    ax.text(13, 0.05, "β LAGS SR", fontsize=8, color="#444",
             rotation=0, va="bottom", ha="right")

    ax.set_xlabel("β-SR temporal lag (s)", fontsize=11)
    ax.set_ylabel("Cross-subject SR × β coupling ρ (Spearman)", fontsize=11)
    ax.set_title("B — Coupling strength × β-SR temporal ordering",
                  loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_xlim(-9, 15)
    ax.set_ylim(0, 1.0)

    fig.suptitle("Four cohort regimes ordination",
                  fontsize=12, fontweight="bold", y=1.005)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
