#!/usr/bin/env python3
"""
B15 part 2 on composite v2 — trivial re-use of §31 t0_shift CSVs.

Envelope B15 found:
  - Q4 shift IQR ~1 s (tight)
  - Q1 shift distribution ~6 s wide

Under composite, we already have per-event t_shift_s from §31. Merge with
composite template_ρ quartile and tabulate shift-distribution width per
rho_q.

Usage:
    python scripts/sie_b15_shift_by_rho_composite.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')


def summarize(cohort):
    shift_csv = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                              'psd_timelapse', f'{cohort}_composite',
                              'per_event_t0_shift.csv')
    qual_csv = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                             f'per_event_quality_{cohort}_composite.csv')

    shifts = pd.read_csv(shift_csv)
    qual = pd.read_csv(qual_csv).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    # Match on (subject_id, t0_net) rounded
    shifts['key'] = shifts.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
    qual['key']   = qual.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
    merged = shifts.merge(qual[['key', 'template_rho', 'rho_q']], on='key', how='left').dropna(subset=['rho_q'])
    print(f"\n=== {cohort} composite · shift-distribution width by template_ρ quartile ===")
    print(f"(envelope B15: Q4 shift IQR ~1 s tight; Q1 distribution ~6 s wide)")
    print(f"{'rho_q':<6} {'n':<6} {'shift median':<14} {'shift IQR':<18} {'|shift−1.2| median':<20} {'shift std':<12}")
    rows = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = merged[merged['rho_q'] == q]
        shift = sub['t_shift_s'].dropna()
        if len(shift) == 0:
            continue
        iqr = float(shift.quantile(0.75) - shift.quantile(0.25))
        med_dist = float(np.abs(shift - 1.2).median())
        rows.append({
            'rho_q': q, 'n_events': len(shift),
            'shift_median': float(shift.median()),
            'shift_q25': float(shift.quantile(0.25)),
            'shift_q75': float(shift.quantile(0.75)),
            'shift_iqr': iqr,
            'timing_distance_median': med_dist,
            'shift_std': float(shift.std()),
        })
        print(f"{q:<6} {len(shift):<6} {shift.median():+.2f}          "
              f"[{shift.quantile(0.25):+.2f}, {shift.quantile(0.75):+.2f}] = {iqr:.2f}s    "
              f"{med_dist:.2f}s              {shift.std():.2f}s")
    return merged, pd.DataFrame(rows)


def main():
    ec_merged, ec_summary = summarize('lemon')
    eo_merged, eo_summary = summarize('lemon_EO')

    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'psd_timelapse')
    ec_summary.to_csv(os.path.join(out_dir, 'b15_shift_by_rho_lemon.csv'), index=False)
    eo_summary.to_csv(os.path.join(out_dir, 'b15_shift_by_rho_lemon_EO.csv'), index=False)

    # Plot: shift distribution per quartile, both cohorts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (merged, title) in zip(axes, [(ec_merged, 'LEMON EC'), (eo_merged, 'LEMON EO')]):
        colors = {'Q1': '#4575b4', 'Q2': '#91bfdb', 'Q3': '#fdae61', 'Q4': '#d73027'}
        for q in ['Q1', 'Q4']:
            vals = merged[merged['rho_q'] == q]['t_shift_s']
            ax.hist(vals, bins=np.linspace(-5, 5, 40), color=colors[q],
                    alpha=0.55, label=f'{q} (n={len(vals)}, IQR {vals.quantile(.75)-vals.quantile(.25):.2f}s)',
                    edgecolor='k', lw=0.3)
        ax.axvline(1.2, color='k', ls='--', lw=1, label='canonical +1.2 s')
        ax.set_xlabel('t_shift (t0_sr − t0_net)  s')
        ax.set_ylabel('events')
        ax.set_title(f'{title} composite')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('B15 part 2 · shift distribution width by template_ρ quartile · composite v2',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'b15_shift_by_rho_composite.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/b15_shift_by_rho_composite.png")
    print(f"Saved: {out_dir}/b15_shift_by_rho_lemon{{_EO,}}.csv")


if __name__ == '__main__':
    main()
