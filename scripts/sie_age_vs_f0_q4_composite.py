#!/usr/bin/env python3
"""B59 re-run on composite-v2 events with proper Q4 filtering.

The published "76% age-explained" cohort-invariance claim was computed by
sie_age_vs_f0.py reading from non-composite events with no Q4 filter. The
paper text says the analysis is on "≥3 Q4 events" but that wasn't actually
implemented. This script does it properly:

  1. Per cohort, load the composite events CSV per subject
  2. Load per_event_quality CSV (with template_rho per event)
  3. Per subject, take Q4 events (top quartile of template_rho)
  4. Per subject, compute median sr1 frequency from Q4 events only
  5. Merge with cohort-specific age tables
  6. Kruskal-Wallis on per-cohort medians (raw + age-residualized)
  7. Report: % age-explained = (1 - H_resid / H_raw) * 100

Compare to published claim: H raw = 36.85, H resid = 8.83, 76% age-explained.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

ROOT = '/Users/neurokinetikz/Code/research'
EVENTS_ROOT = os.path.join(ROOT, 'exports_sie')
QUALITY_ROOT = os.path.join(ROOT, 'outputs/schumann/images/quality')

# Cohort -> (events dir suffix, quality csv name, age table loader)
def lemon_ages():
    p = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
    if not os.path.isfile(p):
        return None
    lm = pd.read_csv(p)
    lm.columns = [c.strip() for c in lm.columns]
    def midpoint(a):
        if pd.isna(a): return np.nan
        a = str(a).strip()
        if '-' in a:
            try: return np.mean([float(x) for x in a.split('-')])
            except: return np.nan
        try: return float(a)
        except: return np.nan
    lm['age'] = lm['Age'].apply(midpoint)
    return lm[['ID','age']].rename(columns={'ID':'subject_id'})

def hbn_ages(rel):
    p = f'/Volumes/T9/hbn_data/cmi_bids_{rel}/participants.tsv'
    if not os.path.isfile(p):
        return None
    m = pd.read_csv(p, sep='\t').rename(columns={'participant_id':'subject_id'})
    m['age'] = pd.to_numeric(m['age'], errors='coerce')
    return m[['subject_id','age']]

def tdbrain_ages():
    p = os.path.expanduser('~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')
    if not os.path.isfile(p):
        return None
    m = pd.read_csv(p, sep='\t')
    m.columns = [c.strip() for c in m.columns]
    m = m.rename(columns={'participants_ID':'subject_id'})
    m['age'] = (m['age'].astype(str).str.replace(',','.',regex=False)
                       .replace('nan',np.nan))
    m['age'] = pd.to_numeric(m['age'], errors='coerce')
    return m[['subject_id','age']]

# Map of cohort label -> (events_dir, quality_csv, age_loader_fn)
COHORTS = [
    ('LEMON',    'lemon_composite',     'per_event_quality_lemon_composite.csv',     lambda: lemon_ages()),
    ('HBN_R1',   'hbn_R1_composite',    'per_event_quality_hbn_R1_composite.csv',    lambda: hbn_ages('R1')),
    ('HBN_R2',   'hbn_R2_composite',    'per_event_quality_hbn_R2_composite.csv',    lambda: hbn_ages('R2')),
    ('HBN_R3',   'hbn_R3_composite',    'per_event_quality_hbn_R3_composite.csv',    lambda: hbn_ages('R3')),
    ('HBN_R4',   'hbn_R4_composite',    'per_event_quality_hbn_R4_composite.csv',    lambda: hbn_ages('R4')),
    ('HBN_R6',   'hbn_R6_composite',    'per_event_quality_hbn_R6_composite.csv',    lambda: hbn_ages('R6')),
    ('TDBRAIN',  'tdbrain_composite',   'per_event_quality_tdbrain_composite.csv',   lambda: tdbrain_ages()),
]


def _weighted_median(values, weights):
    """Weighted median: smallest x such that cumulative weight up to x ≥ 0.5 * total."""
    if len(values) == 0 or sum(weights) <= 0:
        return float('nan')
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    cum = 0
    for v, w in pairs:
        cum += w
        if cum >= total / 2:
            return v
    return pairs[-1][0]


def per_subject_median_f0(cohort_label, events_dir, quality_csv_name, mode):
    """Return per-subject (weighted) median sr1 frequency.
    mode: 'all' = unweighted median; 'q4' = Q4-only median; 'sw' = template_rho-weighted median.
    """
    rows = []
    quality_csv = os.path.join(QUALITY_ROOT, quality_csv_name)
    has_quality = os.path.isfile(quality_csv)
    q4_t0_set = {}
    rho_lookup = {}  # (sid, t0r) -> template_rho
    if mode in ('q4', 'sw') and has_quality:
        qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
        if mode == 'q4':
            def _qcut(g):
                if g.nunique() < 4:
                    return pd.Series(['NA'] * len(g), index=g.index)
                return pd.qcut(g, 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
            qual['rho_q'] = qual.groupby('subject_id')['template_rho'].transform(_qcut)
            q4 = qual[qual['rho_q'] == 'Q4']
            for sid, g in q4.groupby('subject_id'):
                q4_t0_set[sid] = set(np.round(g['t0_net'].values, 3))
        elif mode == 'sw':
            qual['t0r'] = np.round(qual['t0_net'], 3)
            for _, r in qual.iterrows():
                rho_lookup[(r['subject_id'], r['t0r'])] = r['template_rho']
    for f in glob.glob(os.path.join(EVENTS_ROOT, events_dir, 'sub-*_sie_events.csv')):
        d = pd.read_csv(f)
        if 'sr1' not in d.columns:
            continue
        sid = os.path.basename(f).replace('_sie_events.csv', '')
        d = d.dropna(subset=['sr1'])
        d = d[(d['sr1'] >= 7) & (d['sr1'] <= 8.3)]
        if mode == 'q4':
            if sid not in q4_t0_set:
                continue
            d['t0r'] = np.round(d['t0_net'], 3)
            d = d[d['t0r'].isin(q4_t0_set[sid])]
            if len(d) >= 1:
                f_med = float(d['sr1'].median())
                rows.append(dict(cohort=cohort_label, subject_id=sid,
                                 f_median=f_med, n=int(len(d))))
        elif mode == 'sw':
            d['t0r'] = np.round(d['t0_net'], 3)
            d['rho'] = d['t0r'].apply(lambda t: rho_lookup.get((sid, t), 0.0))
            weights = np.maximum(d['rho'].values, 0.0)
            if len(d) >= 1 and weights.sum() > 0:
                f_med = _weighted_median(d['sr1'].values, weights)
                rows.append(dict(cohort=cohort_label, subject_id=sid,
                                 f_median=f_med, n=int(len(d))))
        else:  # all
            if len(d) >= 1:
                rows.append(dict(cohort=cohort_label, subject_id=sid,
                                 f_median=float(d['sr1'].median()),
                                 n=int(len(d))))
    return pd.DataFrame(rows)


def run(mode):
    label = {'all': 'all events', 'q4': 'Q4 only', 'sw': 'shape-weighted'}[mode]
    print('=' * 78)
    print(f'COHORT-INVARIANCE B59 RE-RUN: {label}, composite-v2 events')
    print('=' * 78)
    all_dfs = []
    for cohort_label, events_dir, quality_csv_name, age_fn in COHORTS:
        df_c = per_subject_median_f0(cohort_label, events_dir, quality_csv_name, mode)
        if len(df_c) == 0:
            print(f'  {cohort_label:<10}: 0 subjects')
            continue
        ages = age_fn()
        if ages is None:
            print(f'  {cohort_label:<10}: {len(df_c)} subjects, NO age table')
            continue
        df_c = df_c.merge(ages, on='subject_id', how='left')
        df_c = df_c.dropna(subset=['age', 'f_median'])
        n_age = len(df_c)
        if n_age == 0:
            print(f'  {cohort_label:<10}: {len(df_c)} subjects but 0 with age')
            continue
        print(f'  {cohort_label:<10}: {n_age} subjects with age, '
              f'median f0 {df_c["f_median"].median():.3f}')
        all_dfs.append(df_c)
    if not all_dfs:
        print('No data. Skipping.')
        return
    pool = pd.concat(all_dfs, ignore_index=True)
    pool = pool.dropna(subset=['age', 'f_median'])
    print(f'\nTotal subjects with age: {len(pool)} '
          f'({pool["cohort"].value_counts().to_dict()})')

    # Continuous age × f_median
    rho, p = spearmanr(pool['age'], pool['f_median'])
    slope, intercept = np.polyfit(pool['age'], pool['f_median'], 1)
    print(f'\nContinuous age × f_median: ρ = {rho:+.3f}, p = {p:.2g}')
    print(f'  OLS slope: {slope:+.4f} Hz/year, intercept = {intercept:.3f}')

    # Raw Kruskal-Wallis
    groups = [g['f_median'].values for _, g in pool.groupby('cohort')]
    H_raw, p_raw = kruskal(*groups)
    print(f'\nRaw Kruskal-Wallis (cohort effect): H = {H_raw:.2f}, p = {p_raw:.2g}')

    # Age-residualized
    pool['f_resid'] = pool['f_median'] - (slope * pool['age'] + intercept)
    groups_resid = [g['f_resid'].values for _, g in pool.groupby('cohort')]
    H_resid, p_resid = kruskal(*groups_resid)
    print(f'Residualized   Kruskal-Wallis: H = {H_resid:.2f}, p = {p_resid:.2g}')

    pct_explained = 100 * (1 - H_resid / H_raw) if H_raw > 0 else float('nan')
    print(f'\n>>> Age-explained variance: {pct_explained:.1f}%')
    print(f'    (Published claim: 76% via H_raw=36.85, H_resid=8.83)')

    # Per-cohort medians
    print()
    print('Per-cohort medians (raw / age-adjusted):')
    for c, g in pool.groupby('cohort'):
        gm = pool['f_median'].mean()
        dev = g['f_median'].mean() - gm
        gr = pool['f_resid'].mean()
        dev_adj = g['f_resid'].mean() - gr
        print(f'  {c:<10}: n={len(g):>4d}  median={g["f_median"].median():.3f}  '
              f'mean={g["f_median"].mean():.3f}  '
              f'dev={dev:+.4f}  dev_adj={dev_adj:+.4f}')


if __name__ == '__main__':
    run('all')
    print('\n')
    run('q4')
    print('\n')
    run('sw')
