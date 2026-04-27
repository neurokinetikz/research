#!/usr/bin/env python3
"""
Paper Figure 1 — Event architecture (four-stream peri-event trajectory).

Plots envelope (E), Kuramoto R, PLV, and MSC streams aligned to the
triple-stream nadir at t = 0 (the joint-dip onset). Six phases are
shaded:
  1 baseline (-10 to -2.1 s)
  2 preparatory desync (-2.1 to -1.45 s)
  3 nadir / onset (-1.45 to -1.10 s)
  4 ignition rise (-1.10 to +0.95 s)
  5 peak (+0.95 to +1.45 s)
  6 decay (+1.45 to +5.6 s)

Source data:
  outputs/schumann/images/multistream/multistream_nadir_aligned.csv

Stdlib + matplotlib only.
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import os
REPO = Path(__file__).resolve().parent.parent
SCOPE = os.environ.get("SCOPE", "all")  # 'all' or 'sw'
_TAG = "" if SCOPE == "all" else f"_{SCOPE}"
# All-events: cohort-aggregate path. SW: per-cohort LEMON EC path (only cohort
# with SW peri-event computed locally so far).
SRC_CSV = (REPO / "outputs/schumann/images/multistream/multistream_nadir_aligned.csv"
           if SCOPE == "all"
           else REPO / "outputs/schumann/images/multistream/lemon_composite/multistream_nadir_aligned_sw.csv")
OUT_PNG = REPO / f"papers/schumann_canonical/images/fig1_event_architecture{_TAG}.png"

# Phase boundaries (seconds, relative to triple-stream nadir at t = 0)
PHASE_BOUNDS = [
    (-10.0, -2.10, "1 baseline", "#dddddd"),
    (-2.10, -1.45, "2 prep desync", "#ffe2b3"),
    (-1.45, -1.10, "3 nadir", "#ff9999"),
    (-1.10, 0.95, "4 rise", "#b8e0b8"),
    (0.95, 1.45, "5 peak", "#3a8a3a"),
    (1.45, 5.60, "6 decay", "#bcd6f0"),
]


def load_streams(path: Path):
    rows = []
    with path.open() as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append(r)
    t = [float(r["t_rel"]) for r in rows]

    def col(name):
        return [float(r[name]) for r in rows]

    streams = {
        "env": (col("env_mean"), col("env_ci_lo"), col("env_ci_hi")),
        "R": (col("R_mean"), col("R_ci_lo"), col("R_ci_hi")),
        "PLV": (col("PLV_mean"), col("PLV_ci_lo"), col("PLV_ci_hi")),
        "MSC": (col("MSC_mean"), col("MSC_ci_lo"), col("MSC_ci_hi")),
    }
    return t, streams


def shade_phases(ax, ymin, ymax, phase_bounds):
    for x0, x1, label, col in phase_bounds:
        ax.axvspan(x0, x1, ymin=0, ymax=1, color=col, alpha=0.32, lw=0)
    ax.set_xlim(-8.0, 6.0)


def derive_phase_bounds(t, env_mean):
    """Compute phase boundaries from the actual env mean trajectory.
    Phases:
      1 baseline: −10 s to (nadir − 0.8 s)
      2 prep desync: (nadir − 0.8 s) to (nadir − 0.2 s)
      3 nadir: (nadir − 0.2 s) to (nadir + 0.2 s)
      4 rise: (nadir + 0.2 s) to (peak − 0.2 s)
      5 peak: (peak − 0.2 s) to (peak + 0.3 s)
      6 decay: (peak + 0.3 s) to +6 s
    """
    import numpy as np
    arr = np.asarray(env_mean)
    tt = np.asarray(t)
    nadir_t = float(tt[int(np.nanargmin(arr))])
    # peak after nadir
    post = tt > nadir_t
    if post.any():
        peak_t = float(tt[post][int(np.nanargmax(arr[post]))])
    else:
        peak_t = nadir_t + 1.5
    return [
        (-10.0,           nadir_t - 0.8,  "1 baseline",     "#dddddd"),
        (nadir_t - 0.8,   nadir_t - 0.2,  "2 prep desync",  "#ffe2b3"),
        (nadir_t - 0.2,   nadir_t + 0.2,  "3 nadir",        "#ff9999"),
        (nadir_t + 0.2,   peak_t  - 0.2,  "4 rise",         "#b8e0b8"),
        (peak_t  - 0.2,   peak_t  + 0.3,  "5 peak",         "#3a8a3a"),
        (peak_t  + 0.3,   6.0,            "6 decay",        "#bcd6f0"),
    ], nadir_t, peak_t


def main():
    t, streams = load_streams(SRC_CSV)

    # Derive phase boundaries from actual env mean (data-driven, not hardcoded)
    phase_bounds, nadir_t, peak_t = derive_phase_bounds(t, streams["env"][0])

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 9.5), sharex=True,
                              gridspec_kw={"hspace": 0.18})

    cfg = [
        ("env", "#e07b1a", "Envelope z (7.83 Hz)"),
        ("R", "#2a8a4a", "Kuramoto R (7.2–8.4 Hz)"),
        ("PLV", "#7a3aa8", "PLV to median (7.2–8.4 Hz)"),
        ("MSC", "#1a5fb4", "MSC to median (7.83 Hz)"),
    ]

    for ax, (key, color, ylab) in zip(axes, cfg):
        m, lo, hi = streams[key]
        ymin, ymax = min(lo) - 0.02 * (max(hi) - min(lo)), max(hi) + 0.02 * (max(hi) - min(lo))
        shade_phases(ax, ymin, ymax, phase_bounds)
        ax.fill_between(t, lo, hi, color=color, alpha=0.22, lw=0)
        ax.plot(t, m, color=color, lw=1.8)
        ax.axvline(nadir_t, color="#222222", lw=0.7, ls=":")
        ax.axvline(peak_t, color="#444444", lw=0.6, ls=(0, (1, 2)))
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_ylim(ymin, ymax)

    ax0 = axes[0]
    handles = []
    for x0, x1, label, col in phase_bounds:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.55, ec="none",
                                       label=label))
    ax0.legend(handles=handles, loc="upper left", ncol=3, fontsize=8,
                frameon=True, fancybox=False)

    title_scope = "shape-weighted" if SCOPE == "sw" else "Q4 canonical"
    ax0.set_title(f"Universal four-stream peri-event architecture ({title_scope}; LEMON EC)",
                   fontsize=11, loc="left", fontweight="bold")
    axes[-1].set_xlabel("time relative to triple-stream nadir (s)", fontsize=10)

    axes[0].annotate(f"nadir\n{nadir_t:+.2f} s",
                      xy=(nadir_t, min(streams["env"][0])),
                      xytext=(nadir_t - 3.0, min(streams["env"][0])),
                      fontsize=8,
                      arrowprops=dict(arrowstyle="->", color="#666", lw=0.7))
    axes[0].annotate(f"peak\n{peak_t:+.2f} s",
                      xy=(peak_t, max(streams["env"][0])),
                      xytext=(peak_t + 1.5, max(streams["env"][0])),
                      fontsize=8,
                      arrowprops=dict(arrowstyle="->", color="#666", lw=0.7))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
