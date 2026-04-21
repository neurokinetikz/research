#!/usr/bin/env python3
"""Summary stats per composite-extracted cohort.

Per-cohort: subject count, total events, median per subject, median
f0 (sr1), sr_score distribution, HSI distribution. Tests whether
composite extraction produces comparable rates across cohorts, and
whether cohort-level f0 aligns with B59's age-adjusted conclusion.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kruskal

EVENTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
os.makedirs(OUT_DIR, exist_ok=True)

COHORTS = sorted([d for d in os.listdir(EVENTS_ROOT)
                    if d.endswith('_composite')])

print(f"Composite cohorts: {len(COHORTS)}")
for c in COHORTS:
    print(f"  {c}")

print(f"\n{'cohort':<40}{'n_subj':>8}{'n_ev':>8}{'ev/subj':>10}"
      f"{'f0_med':>10}{'HSI_med':>10}{'sr_score_med':>15}")
print('-' * 90)
rows = []
per_sub_f0 = {}
for c in COHORTS:
    d = os.path.join(EVENTS_ROOT, c)
    files = sorted(glob.glob(os.path.join(d, 'sub-*_sie_events.csv'))
                    + glob.glob(os.path.join(d, 'S*_sie_events.csv'))
                    + glob.glob(os.path.join(d, '*_sie_events.csv')))
    files = [f for f in files if not f.endswith('extraction_summary.csv')]
    files = list(set(files))
    all_ev = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df = df.dropna(subset=['sr1']) if 'sr1' in df.columns else df
        all_ev.append(df)
    if not all_ev:
        continue
    ev = pd.concat(all_ev, ignore_index=True)
    n_sub = ev['subject_id'].nunique() if 'subject_id' in ev.columns else len(files)
    n_ev = len(ev)
    sr1_med = ev['sr1'].median() if 'sr1' in ev.columns else np.nan
    hsi_med = ev['HSI'].median() if 'HSI' in ev.columns else np.nan
    sr_med = ev['sr_score'].median() if 'sr_score' in ev.columns else np.nan
    rows.append({'cohort': c, 'n_subjects': n_sub, 'n_events': n_ev,
                 'events_per_subj': n_ev / max(1, n_sub),
                 'f0_median': sr1_med,
                 'HSI_median': hsi_med,
                 'sr_score_median': sr_med})
    print(f"{c:<40}{n_sub:>8}{n_ev:>8}{n_ev/max(1,n_sub):>10.1f}"
          f"{sr1_med:>10.3f}{hsi_med:>10.3f}{sr_med:>15.3f}")
    # per-subject medians for KW
    if 'sr1' in ev.columns:
        ps = ev.groupby('subject_id')['sr1'].median()
        per_sub_f0[c] = ps.values

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, 'composite_cohort_summary.csv'), index=False)
print(f"\nSaved: composite_cohort_summary.csv ({len(df)} cohorts)")

# Cross-cohort f0 KW test
groups = [v for v in per_sub_f0.values() if len(v) >= 10]
if len(groups) >= 3:
    stat, p = kruskal(*groups)
    print(f"\nKruskal-Wallis across cohorts (per-subject f0 medians): "
          f"H={stat:.2f}  p={p:.3g}")
    for c, v in per_sub_f0.items():
        if len(v) < 10: continue
        print(f"  {c:<40}  n={len(v)}  med={np.median(v):.3f}  "
              f"IQR [{np.percentile(v,25):.3f}, {np.percentile(v,75):.3f}]")
