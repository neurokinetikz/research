#!/usr/bin/env python3
"""
A7 re-run on composite v2 detector.

Reads the composite A3 grand-average CSV produced by
sie_perionset_triple_average_composite.py and derives the 6-phase taxonomy.

Usage:
    python scripts/sie_ignition_phase_segmentation_composite.py --cohort lemon
    python scripts/sie_ignition_phase_segmentation_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')

BASELINE_WIN = (-10.0, -5.0)
DIP_WIN = (-3.0, 0.4)
PEAK_WIN = (-0.5, 2.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'perionset', f'{args.cohort}_composite')
    csv_t0 = os.path.join(out_dir, 'perionset_triple_average.csv')
    if not os.path.isfile(csv_t0):
        raise SystemExit(f"Missing A3 CSV for cohort {args.cohort}: {csv_t0}\n"
                         f"Run sie_perionset_triple_average_composite.py first.")

    df = pd.read_csv(csv_t0)
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
    if mad_R < 1e-6:
        mad_R = np.nanstd(R[base_mask])

    R_dip = np.where(dip_mask, R, np.inf)
    i_nadir = int(np.nanargmin(R_dip))
    t_nadir = float(t[i_nadir])
    R_nadir = float(R[i_nadir])

    R_peak = np.where(peak_mask, R, -np.inf)
    i_peak = int(np.nanargmax(R_peak))
    t_peak = float(t[i_peak])
    R_peak_val = float(R[i_peak])

    dR = np.gradient(R, t)
    rise_mask = (t >= t_nadir) & (t <= t_peak)
    dR_rise = np.where(rise_mask, dR, -np.inf)
    i_rise = int(np.nanargmax(dR_rise))
    t_rise_start = float(t[i_rise])

    pre_mask = (t >= -10.0) & (t < t_nadir)
    above = pre_mask & (R >= base_R)
    if np.any(above):
        idx = int(np.where(above)[0][-1])
        t_pre_desync_start = float(t[idx])
    else:
        t_pre_desync_start = -10.0

    post_mask = t > t_peak
    returned = post_mask & (R <= base_R + 0.5 * mad_R)
    if np.any(returned):
        idx = int(np.where(returned)[0][0])
        t_return = float(t[idx])
    else:
        t_return = 10.0

    phases = pd.DataFrame([
        {'phase': '1 baseline',           't_start': -10.0,              't_end': t_pre_desync_start},
        {'phase': '2 preparatory desync', 't_start': t_pre_desync_start, 't_end': t_nadir},
        {'phase': '3 nadir (onset)',      't_start': t_nadir - 0.2,      't_end': t_rise_start},
        {'phase': '4 ignition rise',      't_start': t_rise_start,       't_end': t_peak},
        {'phase': '5 peak',               't_start': t_peak - 0.2,       't_end': t_peak + 0.2},
        {'phase': '6 decay',              't_start': t_peak,             't_end': t_return},
    ])
    phases['duration'] = phases['t_end'] - phases['t_start']

    def mean_in(arr, tl, th):
        m = (t >= tl) & (t <= th)
        return float(np.nanmean(arr[m])) if m.any() else np.nan

    phases['env_mean'] = [mean_in(E, r.t_start, r.t_end) for _, r in phases.iterrows()]
    phases['R_mean']   = [mean_in(R, r.t_start, r.t_end) for _, r in phases.iterrows()]
    phases['PLV_mean'] = [mean_in(P, r.t_start, r.t_end) for _, r in phases.iterrows()]

    print(f"\n=== {args.cohort} composite · ignition phase segmentation ===\n")
    print(phases.to_string(index=False))

    rise_dur = t_peak - t_rise_start
    decay_dur = t_return - t_peak
    print(f"\nBaseline R      : {base_R:.3f} ± {mad_R:.3f}")
    print(f"Nadir R         : {R_nadir:.3f} at t={t_nadir:+.2f} s")
    print(f"Peak R          : {R_peak_val:.3f} at t={t_peak:+.2f} s")
    print(f"Rise start      : t={t_rise_start:+.2f} s (max dR/dt)")
    print(f"Rise duration   : {rise_dur:.2f} s")
    print(f"Decay duration  : {decay_dur:.2f} s")
    print(f"Rise:Decay      : 1 : {decay_dur / rise_dur:.2f}" if rise_dur > 0 else "")
    print(f"Dip depth       : {base_R - R_nadir:.3f}   Peak height: {R_peak_val - base_R:.3f}")
    print(f"Total event dur : {t_return - t_pre_desync_start:.2f} s "
          f"(prep desync {t_pre_desync_start:+.2f}s → return {t_return:+.2f}s)")

    phases.to_csv(os.path.join(out_dir, 'ignition_phase_segmentation.csv'), index=False)

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
        for x_l, label_l in [(t_nadir, 'nadir'), (t_rise_start, 'rise'),
                              (t_peak, 'peak')]:
            ax.axvline(x_l, color='k', ls=':', lw=0.6, alpha=0.4)
    axes[0].legend(loc='upper right', fontsize=8, ncol=3)
    axes[0].set_title(f'A7 · ignition phase segmentation · {args.cohort} composite v2\n'
                       f'Total {t_return - t_pre_desync_start:.1f} s · rise:decay 1:{decay_dur/rise_dur:.2f}')
    axes[2].set_xlabel('time relative to t₀_net (s)')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'ignition_phase_segmentation.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
