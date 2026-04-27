#!/usr/bin/env python3
"""
B1 re-run on composite v2 detector (lean version, reusing §23b A6 outputs).

B1 envelope finding: per-event dip-time std ≈ 1 s across all streams; grand
average is NOT a phantom — single events and per-subject means both show
visible dips. We already computed per-event dip times for env/R/PLV under
composite in §23b (scripts/sie_dip_rebound_analysis_composite.py), saved to
dip_rebound_per_event.csv. This script uses those CSVs to:

  1. Confirm per-event dip-time std (covered by §23b)
  2. NEW: Cross-stream correlation of per-event dip times
     (does a given event have all three streams dipping at the same time?)
  3. NEW: Per-subject dip times — confirm variability is at the event level,
     not the subject level (supports "grand mean is not a phantom")

The qualitative overlay plot from the envelope B1 is NOT reproduced here — we
have already shown per-event structure is robust via §23b (per-event std 0.9 s
tightly clustered at −1.30 s) and per-subject structure via §23 A3 peri-onset
(thin per-subject traces behind grand mean show coherent dip/peak structure).

Cohort-parameterized.

Usage:
    python scripts/sie_single_event_inspection_composite.py --cohort lemon
    python scripts/sie_single_event_inspection_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'single_event', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                             'perionset', f'{args.cohort}_composite',
                             'dip_rebound_per_event.csv')
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Missing §23b CSV for {args.cohort}: {csv_path}")

    df = pd.read_csv(csv_path)
    n_events = len(df)
    n_subs = df['subject_id'].nunique()
    print(f"Cohort: {args.cohort} composite · events: {n_events}  subjects: {n_subs}")

    # (1) Per-event dip-time std per stream
    streams = [('tE_dip', 'envelope z'),
                ('tR_dip', 'Kuramoto R'),
                ('tP_dip', 'mean PLV')]
    print(f"\n=== {args.cohort} composite · per-event dip-time std ===")
    print(f"(envelope B1: per-event dip std ~1s across all streams)")
    for col, label in streams:
        vals = df[col].dropna()
        print(f"  {label:12s}: median {vals.median():+.2f}s  IQR [{vals.quantile(.25):+.2f}, {vals.quantile(.75):+.2f}]s  std {vals.std():.2f}s")

    # (2) Cross-stream correlation of per-event dip times
    cols = [c for c, _ in streams]
    corr = df[cols].corr()
    print(f"\n=== Cross-stream correlation of per-event dip times ===")
    print(corr.round(3).to_string())
    corr.to_csv(os.path.join(out_dir, 'dip_time_cross_correlations.csv'))

    # (3) Per-subject dip time stability
    print(f"\n=== Per-subject vs per-event variance ===")
    # For each stream, compute within-subject var vs between-subject var
    for col, label in streams:
        sub_means = df.groupby('subject_id')[col].mean().dropna()
        within_var = np.mean([df[df['subject_id'] == s][col].var(ddof=1)
                               for s in sub_means.index
                               if df[df['subject_id'] == s][col].count() >= 2])
        between_var = sub_means.var(ddof=1)
        icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else np.nan
        print(f"  {label:12s}: within_var {within_var:.3f}s²  between_var {between_var:.3f}s²  ICC {icc:.3f}")
    print(f"  (if ICC near 0, dip-time variability is event-level, not subject-level)")

    # Visualization: histogram of per-event dip times per stream + scatter of cross-stream dip times
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, (col, label) in enumerate(streams):
        ax = axes[0, i]
        vals = df[col].dropna()
        ax.hist(vals, bins=30, color='steelblue', edgecolor='k', lw=0.3, alpha=0.85)
        ax.axvline(vals.median(), color='firebrick', ls='--', lw=1.5,
                    label=f'median {vals.median():+.2f}s')
        ax.axvline(-1.30, color='green', ls=':', lw=1, label='A7 nadir (−1.30s)')
        ax.set_xlabel(f'{label} per-event dip time (s rel t0_net)')
        ax.set_ylabel('events')
        ax.set_title(f'{label} · std {vals.std():.2f}s')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Row 2: pairwise scatters
    pairs = [('tE_dip', 'tR_dip'), ('tE_dip', 'tP_dip'), ('tR_dip', 'tP_dip')]
    labels = {'tE_dip': 'envelope z', 'tR_dip': 'R', 'tP_dip': 'PLV'}
    for i, (a, b) in enumerate(pairs):
        ax = axes[1, i]
        sub = df[[a, b]].dropna()
        r = sub[a].corr(sub[b])
        ax.scatter(sub[a], sub[b], s=3, alpha=0.3, color='slategray')
        # Identity line
        lo, hi = -3, 0.5
        ax.plot([lo, hi], [lo, hi], 'r-', lw=0.8, alpha=0.5, label='identity')
        ax.set_xlabel(f'{labels[a]} dip time (s)')
        ax.set_ylabel(f'{labels[b]} dip time (s)')
        ax.set_title(f'{labels[a]} vs {labels[b]} · r = {r:+.3f}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    plt.suptitle(f'B1 · single-event dip-time visibility · {args.cohort} composite v2\n'
                 f'{n_events} events · {n_subs} subjects', y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'single_event_dip_times.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/single_event_dip_times.png")
    print(f"Saved: {out_dir}/dip_time_cross_correlations.csv")


if __name__ == '__main__':
    main()
