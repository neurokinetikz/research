#!/usr/bin/env python3
"""
A7 — Phase segmentation of the ignition time-course.

Using the grand-average peri-onset traces (A3: envelope z, R, PLV aligned on
t0_net, 192 subj / 914 events), define data-driven phase boundaries and
compute the duration and signature of each phase.

Proposed 6-phase schema:

  Phase 1  Baseline            [t_L, t_pre_desync_start)
     → streams at session baseline, flat
  Phase 2  Preparatory desync  [t_pre_desync_start, t_nadir)
     → monotonic decline below baseline; driver of the dip
  Phase 3  Nadir (onset)       [t_nadir - 0.2, t_rise_start)
     → joint minimum; tightest inter-stream alignment
  Phase 4  Ignition rise       [t_rise_start, t_peak)
     → steep synchronization gain; passes through t0_net
  Phase 5  Peak                [t_peak - 0.2, t_peak + 0.2]
     → maximum coordination
  Phase 6  Decay               [t_peak, t_return)
     → return toward baseline; may slightly overshoot

Boundary rules (applied to the grand-average R(t), which is the cleanest
indicator — envelope z is dominated by the detection threshold):

  t_pre_desync_start : last time R ≥ baseline_median in [-10, t_nadir)
                       (first sustained drop below baseline)
  t_nadir            : argmin R in [-3.0, +0.4] s
  t_rise_start       : max of dR/dt in [t_nadir, t_peak]
  t_peak             : argmax R in [-0.5, +2.5] s
  t_return           : first time after t_peak where R returns to
                       baseline_median ± 0.5 * baseline_mad

Baseline window: [-10, -5] s. Emit a CSV of phase boundaries, a summary
table of phase durations and stream values, and a phase-shaded figure.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'perionset')
CSV_T0 = os.path.join(OUT_DIR, 'perionset_triple_average.csv')

BASELINE_WIN = (-10.0, -5.0)
DIP_WIN = (-3.0, 0.4)
PEAK_WIN = (-0.5, 2.5)


def main():
    df = pd.read_csv(CSV_T0)
    t = df['t_rel'].values
    E = df['env_mean'].values
    R = df['R_mean'].values
    P = df['P_mean'].values

    base_mask = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
    dip_mask = (t >= DIP_WIN[0]) & (t <= DIP_WIN[1])
    peak_mask = (t >= PEAK_WIN[0]) & (t <= PEAK_WIN[1])

    base_R = np.nanmedian(R[base_mask])
    base_E = np.nanmedian(E[base_mask])
    base_P = np.nanmedian(P[base_mask])
    mad_R = np.nanmedian(np.abs(R[base_mask] - base_R)) * 1.4826
    if mad_R < 1e-6: mad_R = np.nanstd(R[base_mask])

    # t_nadir: argmin R in dip window
    R_dip = np.where(dip_mask, R, np.inf)
    i_nadir = int(np.nanargmin(R_dip))
    t_nadir = float(t[i_nadir])
    R_nadir = float(R[i_nadir])

    # t_peak: argmax R in peak window
    R_peak = np.where(peak_mask, R, -np.inf)
    i_peak = int(np.nanargmax(R_peak))
    t_peak = float(t[i_peak])
    R_peak_val = float(R[i_peak])

    # t_rise_start: argmax of dR/dt between t_nadir and t_peak
    dR = np.gradient(R, t)
    rise_mask = (t >= t_nadir) & (t <= t_peak)
    dR_rise = np.where(rise_mask, dR, -np.inf)
    i_rise = int(np.nanargmax(dR_rise))
    t_rise_start = float(t[i_rise])

    # t_pre_desync_start: last time in [-10, t_nadir) where R ≥ base_R
    pre_mask = (t >= -10.0) & (t < t_nadir)
    above = pre_mask & (R >= base_R)
    if np.any(above):
        idx = int(np.where(above)[0][-1])
        t_pre_desync_start = float(t[idx])
    else:
        t_pre_desync_start = -10.0

    # t_return: first time after t_peak where R returns to base_R + 0.5*mad_R
    post_mask = t > t_peak
    returned = post_mask & (R <= base_R + 0.5 * mad_R)
    if np.any(returned):
        idx = int(np.where(returned)[0][0])
        t_return = float(t[idx])
    else:
        t_return = 10.0

    phases = pd.DataFrame([
        {'phase': '1 baseline',          't_start': -10.0,               't_end': t_pre_desync_start},
        {'phase': '2 preparatory desync','t_start': t_pre_desync_start, 't_end': t_nadir},
        {'phase': '3 nadir (onset)',     't_start': t_nadir - 0.2,      't_end': t_rise_start},
        {'phase': '4 ignition rise',     't_start': t_rise_start,       't_end': t_peak},
        {'phase': '5 peak',              't_start': t_peak - 0.2,       't_end': t_peak + 0.2},
        {'phase': '6 decay',             't_start': t_peak,             't_end': t_return},
    ])
    phases['duration'] = phases['t_end'] - phases['t_start']

    # Per-phase mean stream values
    def mean_in(arr, tl, th):
        m = (t >= tl) & (t <= th)
        return float(np.nanmean(arr[m])) if m.any() else np.nan

    phases['env_mean'] = [mean_in(E, r.t_start, r.t_end) for _, r in phases.iterrows()]
    phases['R_mean']   = [mean_in(R, r.t_start, r.t_end) for _, r in phases.iterrows()]
    phases['PLV_mean'] = [mean_in(P, r.t_start, r.t_end) for _, r in phases.iterrows()]

    print("\n=== Ignition phase segmentation ===\n")
    print(phases.to_string(index=False))

    print(f"\nBaseline R  : {base_R:.3f} ± {mad_R:.3f}")
    print(f"Nadir R     : {R_nadir:.3f} at t={t_nadir:+.2f} s")
    print(f"Peak R      : {R_peak_val:.3f} at t={t_peak:+.2f} s")
    print(f"Rise start  : t={t_rise_start:+.2f} s (max dR/dt)")
    print(f"Dip depth   : {base_R - R_nadir:.3f}   Peak height: {R_peak_val - base_R:.3f}")
    print(f"Total event duration (pre-desync → return): "
          f"{t_return - t_pre_desync_start:.2f} s")

    phases.to_csv(os.path.join(OUT_DIR, 'ignition_phase_segmentation.csv'),
                   index=False)

    # --- Figure: stream traces with phase shading ---
    colors = ['#e0e0e0', '#ffd89a', '#ff8c69', '#b6e7b6', '#6bc66b', '#a8c7ff']
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, y, lab, col in [
        (axes[0], E, 'envelope z (7.83 ± 0.6 Hz)', 'darkorange'),
        (axes[1], R, 'Kuramoto R(t) in 7.2–8.4 Hz', 'seagreen'),
        (axes[2], P, 'mean PLV to median',          'purple'),
    ]:
        for pi, r in phases.iterrows():
            ax.axvspan(r.t_start, r.t_end, color=colors[pi], alpha=0.6,
                       label=r.phase if ax is axes[0] else None)
        ax.plot(t, y, color=col, lw=2)
        ax.axvline(0, color='k', ls='--', lw=0.6, alpha=0.5)
        ax.set_ylabel(lab)
        # landmark lines
        for x_l, label_l in [(t_nadir, 'nadir'), (t_rise_start, 'rise'),
                              (t_peak, 'peak')]:
            ax.axvline(x_l, color='k', ls=':', lw=0.6, alpha=0.4)
    axes[0].legend(loc='upper right', fontsize=8, ncol=3)
    axes[0].set_title(f'A7 — Ignition phase segmentation (LEMON EC · 914 events)\n'
                       f'Total duration {t_return - t_pre_desync_start:.1f} s '
                       f'(prep desync {t_pre_desync_start:+.2f}s → return {t_return:+.2f}s)')
    axes[2].set_xlabel('time relative to t₀_net (s)')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'ignition_phase_segmentation.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
