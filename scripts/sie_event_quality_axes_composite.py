#!/usr/bin/env python3
"""
B6 partial re-run on composite v2 detector.

Envelope B6 scored each event on 5 axes:
  peak_S (detector amplitude), S_fwhm_s (composite duration), template_ρ
  (shape fidelity), spatial_coh (channel nadir simultaneity), baseline_calm
  (pre-event noise floor)

Under composite v2, we don't have a stored composite detector peak_S/S_fwhm_s.
Instead we use the available per-event scalars across analyses:
  - template_rho (composite quality CSV)                     [morphology]
  - sr_score (composite events CSV)                          [detector amplitude proxy]
  - duration_s (composite events CSV)                        [duration]
  - sr1_z_max (composite events CSV)                         [amplitude]
  - HSI (composite events CSV)                               [cavity-mode index]
  - E_depth = yE_peak - yE_dip (§23b A6 dip-rebound CSV)     [envelope z rebound range]
  - R_depth (§23b)                                           [R-stream rebound]
  - P_depth (§23b)                                           [PLV rebound]

Compute Spearman cross-correlation matrix across these axes. B6 envelope
finding: max |ρ| = 0.28 (axes are largely orthogonal). Check whether composite
event quality axes are similarly orthogonal.

Cohort-parameterized.

Usage:
    python scripts/sie_event_quality_axes_composite.py --cohort lemon
    python scripts/sie_event_quality_axes_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), '..')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'quality', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    events_dir = os.path.join(ROOT, 'exports_sie', f'{args.cohort}_composite')
    quality_csv = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                                f'per_event_quality_{args.cohort}_composite.csv')
    dip_csv = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'perionset',
                            f'{args.cohort}_composite', 'dip_rebound_per_event.csv')

    # Load quality (template_rho)
    qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
    print(f"Cohort: {args.cohort} composite · quality events: {len(qual)}")

    # Load per-subject events CSVs, extract scalar fields
    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    event_rows = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if not os.path.isfile(ep):
            continue
        ev = pd.read_csv(ep)
        for _, e in ev.iterrows():
            event_rows.append({
                'subject_id': e.get('subject_id', r['subject_id']),
                't0_net': float(e['t0_net']) if pd.notna(e.get('t0_net')) else np.nan,
                'duration_s': float(e.get('duration_s', np.nan)) if pd.notna(e.get('duration_s')) else np.nan,
                'sr1_z_max': float(e.get('sr1_z_max', np.nan)) if pd.notna(e.get('sr1_z_max')) else np.nan,
                'HSI': float(e.get('HSI', np.nan)) if pd.notna(e.get('HSI')) else np.nan,
                'sr_score': float(e.get('sr_score', np.nan)) if pd.notna(e.get('sr_score')) else np.nan,
            })
    events_df = pd.DataFrame(event_rows).dropna(subset=['t0_net'])
    print(f"  events from CSVs: {len(events_df)}")

    # Merge quality + events
    events_df['key'] = events_df.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
    qual['key'] = qual.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
    merged = events_df.merge(qual[['key', 'template_rho']], on='key', how='left')

    # Load dip-rebound depths if available
    if os.path.isfile(dip_csv):
        dip = pd.read_csv(dip_csv)
        dip['key'] = dip.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        merged = merged.merge(dip[['key', 'E_depth', 'R_depth', 'P_depth']],
                               on='key', how='left')
    else:
        print(f"  (no §23b dip CSV at {dip_csv})")

    merged = merged.drop(columns=['key'])
    print(f"  merged events: {len(merged)}  (with template_rho: {merged['template_rho'].notna().sum()})")

    # Spearman cross-correlation matrix
    axes = ['template_rho', 'sr_score', 'sr1_z_max', 'duration_s', 'HSI',
            'E_depth', 'R_depth', 'P_depth']
    axes = [a for a in axes if a in merged.columns]
    n_axes = len(axes)
    print(f"\n=== {args.cohort} composite · Spearman ρ cross-correlation (n_axes = {n_axes}) ===")
    corr = np.full((n_axes, n_axes), np.nan)
    pvals = np.full((n_axes, n_axes), np.nan)
    for i, a in enumerate(axes):
        for j, b in enumerate(axes):
            if i >= j:
                continue
            sub = merged.dropna(subset=[a, b])
            if len(sub) < 10:
                continue
            r, p = spearmanr(sub[a], sub[b])
            corr[i, j] = r
            corr[j, i] = r
            pvals[i, j] = p
            pvals[j, i] = p
    for i, a in enumerate(axes):
        corr[i, i] = 1.0

    print(f"\n{'axis':<16}", '  '.join(f"{a[:9]:>10}" for a in axes))
    for i, a in enumerate(axes):
        row = '  '.join(f"{corr[i, j]:+.3f}     " if np.isfinite(corr[i, j]) else '  nan     ' for j in range(n_axes))
        print(f"{a:<16} {row}")

    max_abs = np.nanmax(np.abs(corr - np.eye(n_axes)))
    max_pair = None
    for i in range(n_axes):
        for j in range(i+1, n_axes):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) == max_abs:
                max_pair = (axes[i], axes[j], corr[i, j])
                break
        if max_pair:
            break
    print(f"\n  max |ρ| (off-diagonal) = {max_abs:.3f}  pair: {max_pair}")
    print(f"  (envelope B6: max |ρ| = 0.28 — axes largely orthogonal)")

    # Save correlation matrix
    corr_df = pd.DataFrame(corr, index=axes, columns=axes)
    corr_df.to_csv(os.path.join(out_dir, 'b6_cross_axis_correlations.csv'))

    # Multi-axis threshold passing
    # Each axis: "quality" threshold = 75th percentile for positive axes,
    # 25th percentile for "bad" axes (none here)
    thresh_rows = []
    for a in axes:
        vals = merged[a].dropna()
        if len(vals) == 0:
            continue
        q75 = vals.quantile(0.75)
        n_pass = (merged[a] >= q75).sum()
        thresh_rows.append({
            'axis': a, 'q75': float(q75), 'n_pass': int(n_pass),
            'pct_pass': float(n_pass / len(merged) * 100),
        })
    print(f"\n=== Per-axis 75th-percentile threshold ===")
    for r in thresh_rows:
        print(f"  {r['axis']:<16} q75 = {r['q75']:+.3f}  n_pass = {r['n_pass']} ({r['pct_pass']:.1f}%)")

    # Fraction passing all / 4-of-5 / 3-of-5 axes (using 4 core axes: template_rho, sr_score, sr1_z_max, E_depth)
    core = [a for a in ['template_rho', 'sr_score', 'sr1_z_max', 'E_depth'] if a in merged.columns]
    if len(core) >= 3:
        q75s = {a: merged[a].quantile(0.75) for a in core}
        # pass-count per event
        merged['n_pass_core'] = sum((merged[a] >= q75s[a]).astype(int) for a in core)
        print(f"\n=== Core {len(core)} axes: template_rho + sr_score + sr1_z_max + E_depth ===")
        for k in range(len(core), 0, -1):
            n = int((merged['n_pass_core'] >= k).sum())
            print(f"  events passing ≥ {k} of {len(core)} thresholds: {n} ({n/len(merged)*100:.1f}%)")
        all_pass = int((merged['n_pass_core'] == len(core)).sum())
        print(f"  events passing ALL {len(core)}: {all_pass} ({all_pass/len(merged)*100:.1f}%)")
        print(f"  (envelope B6: 6.2% of events pass all 4 quality thresholds)")

    # Figure: heatmap
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    ax.set_xticks(range(n_axes)); ax.set_xticklabels(axes, rotation=45, ha='right')
    ax.set_yticks(range(n_axes)); ax.set_yticklabels(axes)
    for i in range(n_axes):
        for j in range(n_axes):
            if np.isfinite(corr[i, j]):
                ax.text(j, i, f'{corr[i, j]:+.2f}', ha='center', va='center',
                        fontsize=8, color='white' if abs(corr[i, j]) > 0.35 else 'black')
    plt.colorbar(im, ax=ax, label='Spearman ρ')
    ax.set_title(f'B6 · event-quality axis cross-correlation · {args.cohort} composite v2\n'
                 f'max |ρ| = {max_abs:.2f}  (envelope B6: 0.28)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'b6_cross_axis_correlations.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/b6_cross_axis_correlations.png")
    print(f"Saved: {out_dir}/b6_cross_axis_correlations.csv")


if __name__ == '__main__':
    main()
