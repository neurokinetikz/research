#!/usr/bin/env python3
"""
SIE Replication Analysis -- Phase 2 (Age Trajectory) + Phase 3 (Correlates)
============================================================================
Developmental trajectory, cognitive correlates, and clinical associations.

Usage:
    python scripts/analyze_sie_phase2_3.py
"""

import os
import sys
import warnings
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SIE_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

PHI = (1 + np.sqrt(5)) / 2


# =========================================================================
# DATA LOADERS
# =========================================================================

def load_sie_summaries():
    """Load all per-subject SIE summary CSVs."""
    dfs = []
    for dirname in sorted(os.listdir(SIE_BASE)):
        dirpath = os.path.join(SIE_BASE, dirname)
        if not os.path.isdir(dirpath):
            continue
        for f in glob(os.path.join(dirpath, '*_sie_summary.csv')):
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    df['_source_dir'] = dirname
                    dfs.append(df)
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_sie_events():
    """Load all per-event SIE CSVs."""
    dfs = []
    for dirname in sorted(os.listdir(SIE_BASE)):
        dirpath = os.path.join(SIE_BASE, dirname)
        if not os.path.isdir(dirpath):
            continue
        for f in glob(os.path.join(dirpath, '*_sie_events.csv')):
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    df['_source_dir'] = dirname
                    dfs.append(df)
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_hbn_demographics():
    """Load HBN participants.tsv from all releases, return age + psychopathology."""
    dfs = []
    for release in ['R1', 'R2', 'R3', 'R4', 'R6']:
        path = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if os.path.isfile(path):
            df = pd.read_csv(path, sep='\t')
            df['release'] = release
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    demo = pd.concat(dfs, ignore_index=True)
    demo = demo.rename(columns={'participant_id': 'subject_id'})
    demo['age'] = pd.to_numeric(demo['age'], errors='coerce')
    for col in ['p_factor', 'attention', 'internalizing', 'externalizing']:
        if col in demo.columns:
            demo[col] = pd.to_numeric(demo[col], errors='coerce')
    return demo


def load_dortmund_demographics():
    """Load Dortmund participants.tsv."""
    path = '/Volumes/T9/dortmund_data_dl/participants.tsv'
    if not os.path.isfile(path):
        return pd.DataFrame()
    demo = pd.read_csv(path, sep='\t')
    demo = demo.rename(columns={'participant_id': 'subject_id'})
    demo['age'] = pd.to_numeric(demo['age'], errors='coerce')
    return demo


def load_tdbrain_demographics():
    """Load TDBRAIN participants V2."""
    path = os.path.expanduser('~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')
    if not os.path.isfile(path):
        return pd.DataFrame()
    demo = pd.read_csv(path, sep='\t')
    demo = demo.rename(columns={'participants_ID': 'subject_id'})
    # Fix comma decimal separator in age
    demo['age'] = demo['age'].astype(str).str.replace(',', '.', regex=False)
    demo['age'] = pd.to_numeric(demo['age'], errors='coerce')
    demo['gender'] = pd.to_numeric(demo['gender'], errors='coerce')
    return demo


def load_lemon_demographics_and_cognitive():
    """Load LEMON demographics + cognitive scores using lemon_utils."""
    from lib.lemon_utils import load_demographics, load_cognitive_data, build_master_table
    demo = load_demographics()
    cog = load_cognitive_data()
    master = build_master_table(demo, cog)
    return master


# =========================================================================
# PHASE 2: DEVELOPMENTAL TRAJECTORY
# =========================================================================

def phase2_age_trajectory_hbn(summaries):
    """Age trajectory of SIE characteristics in HBN (ages 5-21)."""
    print("\n" + "=" * 80)
    print("PHASE 2C: DEVELOPMENTAL TRAJECTORY -- HBN (ages 5-21)")
    print("=" * 80)

    hbn_demo = load_hbn_demographics()
    if hbn_demo.empty:
        print("  HBN demographics not available")
        return

    # Match SIE summaries to HBN demographics
    hbn_sie = summaries[summaries['_source_dir'].str.startswith('hbn_')].copy()
    hbn_sie = hbn_sie.merge(hbn_demo[['subject_id', 'age', 'sex', 'p_factor',
                                       'attention', 'internalizing', 'externalizing']],
                            on='subject_id', how='left')
    has_age = hbn_sie['age'].notna()
    print(f"  HBN subjects with SIE data + age: {has_age.sum()}")

    hbn_age = hbn_sie[has_age].copy()
    if len(hbn_age) < 20:
        print("  Too few subjects with age data")
        return

    # Correlate SIE metrics with age
    metrics = ['n_events', 'event_rate_per_min', 'median_sr1_z_max',
               'median_HSI', 'median_sr_score']

    print(f"\n  SIE metric correlations with age (N={len(hbn_age)}):")
    print(f"  {'Metric':30s} {'Spearman ρ':>12s} {'p-value':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for m in metrics:
        if m not in hbn_age.columns:
            continue
        vals = hbn_age[[m, 'age']].dropna()
        if len(vals) < 20:
            continue
        rho, p = spearmanr(vals['age'], vals[m])
        sig = '*' if p < 0.05 else ''
        print(f"  {m:30s} {rho:+12.3f} {p:12.4e} {sig}")

    # Age bins
    bins = [5, 8, 11, 14, 17, 22]
    labels = ['5-7', '8-10', '11-13', '14-16', '17-21']
    hbn_age['age_bin'] = pd.cut(hbn_age['age'], bins=bins, labels=labels, right=False)

    print(f"\n  Event rate by age group:")
    print(f"  {'Age':>8s} {'N':>6s} {'Rate/min':>12s} {'Events/subj':>14s}")
    for lbl in labels:
        sub = hbn_age[hbn_age['age_bin'] == lbl]
        if len(sub) > 0:
            rate = sub['event_rate_per_min'].mean()
            evts = sub['n_events'].mean()
            print(f"  {lbl:>8s} {len(sub):>6d} {rate:>12.3f} {evts:>14.2f}")


def phase2_age_trajectory_tdbrain(summaries):
    """Age trajectory in TDBRAIN (ages 5-88)."""
    print("\n" + "=" * 80)
    print("PHASE 2D: LIFESPAN TRAJECTORY -- TDBRAIN (ages 5-88)")
    print("=" * 80)

    tdb_demo = load_tdbrain_demographics()
    if tdb_demo.empty:
        print("  TDBRAIN demographics not available")
        return

    tdb_sie = summaries[summaries['_source_dir'] == 'tdbrain'].copy()
    tdb_sie = tdb_sie.merge(tdb_demo[['subject_id', 'age', 'gender', 'indication']],
                            on='subject_id', how='left')
    has_age = tdb_sie['age'].notna()
    print(f"  TDBRAIN subjects with SIE data + age: {has_age.sum()}")

    tdb_age = tdb_sie[has_age].copy()
    if len(tdb_age) < 20:
        print("  Too few subjects")
        return

    metrics = ['n_events', 'event_rate_per_min', 'median_sr1_z_max',
               'median_HSI', 'median_sr_score']

    print(f"\n  SIE metric correlations with age (N={len(tdb_age)}):")
    print(f"  {'Metric':30s} {'Spearman ρ':>12s} {'p-value':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for m in metrics:
        if m not in tdb_age.columns:
            continue
        vals = tdb_age[[m, 'age']].dropna()
        if len(vals) < 20:
            continue
        rho, p = spearmanr(vals['age'], vals[m])
        sig = '*' if p < 0.05 else ''
        print(f"  {m:30s} {rho:+12.3f} {p:12.4e} {sig}")

    # Age decade bins
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    labels = ['<10', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80+']
    tdb_age['age_bin'] = pd.cut(tdb_age['age'], bins=bins, labels=labels, right=False)

    print(f"\n  Event rate by age decade:")
    print(f"  {'Age':>8s} {'N':>6s} {'Rate/min':>12s} {'Events/subj':>14s}")
    for lbl in labels:
        sub = tdb_age[tdb_age['age_bin'] == lbl]
        if len(sub) > 0:
            rate = sub['event_rate_per_min'].mean()
            evts = sub['n_events'].mean()
            print(f"  {lbl:>8s} {len(sub):>6d} {rate:>12.3f} {evts:>14.2f}")


def phase2_age_trajectory_dortmund(summaries):
    """Age trajectory in Dortmund (ages 18-85)."""
    print("\n" + "=" * 80)
    print("PHASE 2E: ADULT TRAJECTORY -- Dortmund (ages 18-85)")
    print("=" * 80)

    dort_demo = load_dortmund_demographics()
    if dort_demo.empty:
        print("  Dortmund demographics not available")
        return

    dort_sie = summaries[summaries['_source_dir'] == 'dortmund'].copy()
    dort_sie = dort_sie.merge(dort_demo[['subject_id', 'age', 'sex']],
                              on='subject_id', how='left')
    has_age = dort_sie['age'].notna()
    print(f"  Dortmund subjects with SIE data + age: {has_age.sum()}")

    dort_age = dort_sie[has_age].copy()
    if len(dort_age) < 20:
        print("  Too few subjects")
        return

    metrics = ['n_events', 'event_rate_per_min', 'median_sr1_z_max',
               'median_HSI', 'median_sr_score']

    print(f"\n  SIE metric correlations with age (N={len(dort_age)}):")
    print(f"  {'Metric':30s} {'Spearman ρ':>12s} {'p-value':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for m in metrics:
        if m not in dort_age.columns:
            continue
        vals = dort_age[[m, 'age']].dropna()
        if len(vals) < 20:
            continue
        rho, p = spearmanr(vals['age'], vals[m])
        sig = '*' if p < 0.05 else ''
        print(f"  {m:30s} {rho:+12.3f} {p:12.4e} {sig}")


# =========================================================================
# PHASE 3: CORRELATES
# =========================================================================

def phase3_lemon_cognitive(summaries):
    """SIE metrics vs LEMON cognitive scores."""
    print("\n" + "=" * 80)
    print("PHASE 3A: COGNITIVE CORRELATES -- LEMON")
    print("=" * 80)

    lemon_master = load_lemon_demographics_and_cognitive()
    if lemon_master.empty:
        print("  LEMON cognitive data not available")
        return

    lemon_sie = summaries[summaries['_source_dir'] == 'lemon'].copy()
    lemon_sie = lemon_sie.merge(lemon_master, on='subject_id', how='left')
    print(f"  LEMON subjects with SIE + cognitive data: {len(lemon_sie)}")

    sie_metrics = ['n_events', 'event_rate_per_min', 'median_sr1_z_max',
                   'median_HSI', 'median_sr_score']

    cog_tests = ['CVLT', 'TMT_A', 'TMT_B', 'TAP_Alert', 'TAP_WM',
                 'TAP_Incompat', 'LPS', 'WST']

    # Use log-transformed RT measures where available
    cog_cols = []
    for t in cog_tests:
        if f'log_{t}' in lemon_sie.columns:
            cog_cols.append((t, f'log_{t}'))
        elif t in lemon_sie.columns:
            cog_cols.append((t, t))

    if not cog_cols:
        print("  No cognitive columns found")
        return

    print(f"\n  Spearman correlations (SIE metric × cognitive test):")
    print(f"  {'':30s}", end='')
    for name, _ in cog_cols:
        print(f" {name:>10s}", end='')
    print()

    n_tests = 0
    n_sig = 0
    sig_results = []

    for sie_m in sie_metrics:
        if sie_m not in lemon_sie.columns:
            continue
        print(f"  {sie_m:30s}", end='')
        for cog_name, cog_col in cog_cols:
            pair = lemon_sie[[sie_m, cog_col, 'age_midpoint']].dropna()
            if len(pair) < 20:
                print(f" {'---':>10s}", end='')
                continue
            rho, p = spearmanr(pair[sie_m], pair[cog_col])
            n_tests += 1
            marker = ''
            if p < 0.05:
                n_sig += 1
                marker = '*'
                sig_results.append((sie_m, cog_name, rho, p, len(pair)))
            print(f" {rho:+.3f}{marker:1s}   ", end='')
        print()

    # Age-partialed correlations for significant results
    if sig_results:
        print(f"\n  Significant results (p < 0.05): {n_sig}/{n_tests}")
        print(f"\n  Age-partialed Spearman for significant results:")
        print(f"  {'SIE metric':25s} {'Cog test':>10s} {'ρ':>8s} {'p':>10s} {'ρ_partial':>10s} {'p_partial':>10s} {'N':>5s}")
        for sie_m, cog_name, rho, p, n in sig_results:
            cog_col = [c for name, c in cog_cols if name == cog_name][0]
            pair = lemon_sie[[sie_m, cog_col, 'age_midpoint']].dropna()
            # Partial correlation: regress out age from both
            from scipy.stats import linregress
            res_sie = pair[sie_m] - linregress(pair['age_midpoint'], pair[sie_m]).slope * pair['age_midpoint']
            res_cog = pair[cog_col] - linregress(pair['age_midpoint'], pair[cog_col]).slope * pair['age_midpoint']
            rho_p, p_p = spearmanr(res_sie, res_cog)
            print(f"  {sie_m:25s} {cog_name:>10s} {rho:+8.3f} {p:10.4e} {rho_p:+10.3f} {p_p:10.4e} {n:>5d}")


def phase3_hbn_psychopathology(summaries):
    """SIE metrics vs HBN CBCL psychopathology."""
    print("\n" + "=" * 80)
    print("PHASE 3B: PSYCHOPATHOLOGY CORRELATES -- HBN")
    print("=" * 80)

    hbn_demo = load_hbn_demographics()
    if hbn_demo.empty:
        print("  HBN demographics not available")
        return

    hbn_sie = summaries[summaries['_source_dir'].str.startswith('hbn_')].copy()
    hbn_sie = hbn_sie.merge(hbn_demo[['subject_id', 'age', 'p_factor',
                                       'attention', 'internalizing', 'externalizing']],
                            on='subject_id', how='left')

    psych_cols = ['p_factor', 'attention', 'internalizing', 'externalizing']
    sie_metrics = ['n_events', 'event_rate_per_min', 'median_sr1_z_max',
                   'median_HSI', 'median_sr_score']

    print(f"\n  Spearman correlations (age-partialed):")
    print(f"  {'':30s}", end='')
    for pc in psych_cols:
        print(f" {pc:>14s}", end='')
    print()

    for sie_m in sie_metrics:
        if sie_m not in hbn_sie.columns:
            continue
        print(f"  {sie_m:30s}", end='')
        for pc in psych_cols:
            triplet = hbn_sie[[sie_m, pc, 'age']].dropna()
            if len(triplet) < 20:
                print(f" {'---':>14s}", end='')
                continue
            # Age-partial
            from scipy.stats import linregress
            res_sie = triplet[sie_m] - linregress(triplet['age'], triplet[sie_m]).slope * triplet['age']
            res_psych = triplet[pc] - linregress(triplet['age'], triplet[pc]).slope * triplet['age']
            rho, p = spearmanr(res_sie, res_psych)
            marker = '*' if p < 0.05 else ' '
            print(f" {rho:+.3f} {marker:1s}(N={len(triplet):3d})", end='')
        print()


def phase3_tdbrain_clinical(summaries):
    """SIE metrics by TDBRAIN diagnostic group."""
    print("\n" + "=" * 80)
    print("PHASE 3C: CLINICAL ASSOCIATIONS -- TDBRAIN")
    print("=" * 80)

    tdb_demo = load_tdbrain_demographics()
    if tdb_demo.empty:
        print("  TDBRAIN demographics not available")
        return

    tdb_sie = summaries[summaries['_source_dir'] == 'tdbrain'].copy()
    tdb_sie = tdb_sie.merge(tdb_demo[['subject_id', 'age', 'gender', 'indication']],
                            on='subject_id', how='left')

    # Clean indications
    tdb_sie['dx'] = tdb_sie['indication'].fillna('UNKNOWN')

    # Major diagnostic groups
    dx_groups = ['HEALTHY', 'ADHD', 'MDD', 'OCD', 'SMC', 'BURNOUT']
    print(f"\n  SIE metrics by diagnostic group (age-controlled):")

    for m in ['event_rate_per_min', 'median_sr_score', 'median_HSI']:
        if m not in tdb_sie.columns:
            continue
        print(f"\n  {m}:")
        print(f"  {'Diagnosis':>12s} {'N':>6s} {'Mean':>10s} {'SD':>10s}")
        group_data = {}
        for dx in dx_groups:
            sub = tdb_sie[tdb_sie['dx'].str.contains(dx, case=False, na=False)]
            vals = sub[m].dropna()
            if len(vals) >= 5:
                print(f"  {dx:>12s} {len(vals):>6d} {vals.mean():>10.3f} {vals.std():>10.3f}")
                group_data[dx] = vals.values

        # ANOVA across groups with sufficient N
        groups_for_anova = [v for v in group_data.values() if len(v) >= 10]
        if len(groups_for_anova) >= 2:
            F, p = stats.f_oneway(*groups_for_anova)
            print(f"  ANOVA: F = {F:.2f}, p = {p:.4e}")

        # Key pairwise: ADHD vs HEALTHY, MDD vs HEALTHY
        for dx in ['ADHD', 'MDD']:
            if dx in group_data and 'HEALTHY' in group_data:
                t, p = stats.ttest_ind(group_data[dx], group_data['HEALTHY'])
                d = (group_data[dx].mean() - group_data['HEALTHY'].mean()) / np.sqrt(
                    (group_data[dx].std() ** 2 + group_data['HEALTHY'].std() ** 2) / 2)
                print(f"  {dx} vs HEALTHY: t = {t:.2f}, p = {p:.4f}, d = {d:+.3f}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("Loading SIE data...")
    summaries = load_sie_summaries()
    events = load_sie_events()
    print(f"  Summaries: {len(summaries)}, Events: {len(events)}")

    # Phase 2: Age trajectories
    phase2_age_trajectory_hbn(summaries)
    phase2_age_trajectory_tdbrain(summaries)
    phase2_age_trajectory_dortmund(summaries)

    # Phase 3: Correlates
    phase3_lemon_cognitive(summaries)
    phase3_hbn_psychopathology(summaries)
    phase3_tdbrain_clinical(summaries)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
