#!/usr/bin/env python3
"""
TDBRAIN α/β Trough Analysis: ADHD vs MDD
==========================================

Tests the spectral differentiation paper's prediction:
  - ADHD (externalizing, GABA deficit) → shallower α/β trough
  - MDD (internalizing) → deeper α/β trough

Uses TDBRAIN dataset (N=1,227 discovery, van Dijk et al. 2022):
  - ADHD adults: N ≈ 120
  - MDD adults: N ≈ 347
  - Healthy adults: N ≈ 37

This is a genuine out-of-sample test: independent site (Netherlands),
different recording system (26-channel), formal DSM-5 diagnoses.

Usage:
    python scripts/tdbrain_trough_analysis.py

Requires: TDBRAIN peaks extracted to exports_adaptive_v4/tdbrain/
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'tdbrain_analysis')
PEAK_DIR = os.path.join(BASE_DIR, 'exports_adaptive_v4', 'tdbrain')
PARTICIPANTS_PATH = os.path.expanduser(
    '~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')

# Trough measurement parameters (matching paper exactly)
KNOWN_TROUGHS_HZ = np.array([5.08, 7.81, 13.42, 25.30, 35.04])
SHORT_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']
LOG_HALF_WINDOW = 0.06
LOG_FLANK_OFFSET = 0.15
MIN_POWER_PCT = 50


def per_subject_trough_depth(freqs, trough_hz):
    """Exact copy of paper's method from trough_depth_covariance.py."""
    log_freqs = np.log(freqs)
    log_trough = np.log(trough_hz)

    trough_mask = np.abs(log_freqs - log_trough) < LOG_HALF_WINDOW
    trough_count = trough_mask.sum()

    left_center = log_trough - LOG_FLANK_OFFSET
    left_mask = np.abs(log_freqs - left_center) < LOG_HALF_WINDOW
    left_count = left_mask.sum()

    right_center = log_trough + LOG_FLANK_OFFSET
    right_mask = np.abs(log_freqs - right_center) < LOG_HALF_WINDOW
    right_count = right_mask.sum()

    mean_flank = (left_count + right_count) / 2
    if mean_flank > 0:
        return trough_count / mean_flank
    return np.nan


def load_participants():
    """Load TDBRAIN participants with diagnoses."""
    df = pd.read_csv(PARTICIPANTS_PATH, sep='\t')
    df['age_float'] = df['age'].str.replace(',', '.').astype(float)

    # Classify into diagnostic groups
    df['dx_group'] = 'OTHER'
    df.loc[df['indication'] == 'HEALTHY', 'dx_group'] = 'HEALTHY'
    df.loc[df['indication'].str.contains('ADHD', na=False) &
           ~df['indication'].str.contains('MDD', na=False), 'dx_group'] = 'ADHD'
    df.loc[df['indication'].str.contains('MDD', na=False) &
           ~df['indication'].str.contains('ADHD', na=False), 'dx_group'] = 'MDD'
    df.loc[df['indication'].str.contains('OCD', na=False), 'dx_group'] = 'OCD'

    # Only discovery set
    df = df[df['DISC/REP'] == 'DISCOVERY'].copy()

    return df


def compute_all_trough_depths():
    """Compute per-subject trough depths from extracted peaks."""
    if not os.path.isdir(PEAK_DIR):
        print(f"  ERROR: Peak directory not found: {PEAK_DIR}")
        print(f"  Run extraction first: python scripts/run_f0_760_extraction.py --dataset tdbrain")
        return None

    files = sorted(glob.glob(os.path.join(PEAK_DIR, '*_peaks.csv')))
    if not files:
        print(f"  ERROR: No peak files found in {PEAK_DIR}")
        return None

    print(f"  Found {len(files)} peak files")

    rows = []
    for f in files:
        subj_id = os.path.basename(f).replace('_peaks.csv', '')
        try:
            df = pd.read_csv(f, usecols=['freq', 'power', 'phi_octave'])
        except Exception:
            continue

        # Power filter: top 50% per band
        filtered = []
        for octave in df['phi_octave'].unique():
            bp = df[df.phi_octave == octave]
            if len(bp) == 0:
                continue
            thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
            filtered.append(bp[bp['power'] >= thresh])
        if not filtered:
            continue
        df = pd.concat(filtered, ignore_index=True)
        freqs = df['freq'].values

        if len(freqs) < 100:
            continue

        row = {'subject': subj_id, 'n_peaks': len(freqs)}
        for trough_hz, label in zip(KNOWN_TROUGHS_HZ, SHORT_LABELS):
            row[f'depth_{label}'] = per_subject_trough_depth(freqs, trough_hz)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TDBRAIN α/β Trough Analysis: ADHD vs MDD")
    print("=" * 70)

    # Load participants
    print("\n--- Loading Participants ---")
    participants = load_participants()
    print(f"  Discovery set: {len(participants)} subjects")
    print(f"  Groups: {participants['dx_group'].value_counts().to_dict()}")

    # Compute trough depths
    print("\n--- Computing Trough Depths ---")
    depths = compute_all_trough_depths()
    if depths is None:
        return

    # Merge with demographics
    merged = depths.merge(
        participants[['participants_ID', 'age_float', 'gender', 'dx_group', 'indication']],
        left_on='subject', right_on='participants_ID', how='inner')

    print(f"\n  Merged: {len(merged)} subjects with peaks + diagnosis")
    print(f"  Groups after merge: {merged['dx_group'].value_counts().to_dict()}")

    # Adults only
    adults = merged[merged['age_float'] >= 18].copy()
    print(f"\n  Adults (≥18): {len(adults)}")
    for group in ['ADHD', 'MDD', 'HEALTHY', 'OCD']:
        sub = adults[adults.dx_group == group]
        if len(sub) > 0:
            print(f"    {group}: N={len(sub)}, age={sub['age_float'].mean():.1f}±{sub['age_float'].std():.1f}")

    # --- PRIMARY TEST: ADHD vs MDD α/β trough depth ---
    print("\n" + "=" * 70)
    print("PRIMARY TEST: ADHD vs MDD at α/β Trough")
    print("=" * 70)

    adhd = adults[adults.dx_group == 'ADHD']['depth_α/β'].dropna()
    mdd = adults[adults.dx_group == 'MDD']['depth_α/β'].dropna()

    if len(adhd) < 10 or len(mdd) < 10:
        print(f"  Insufficient subjects: ADHD={len(adhd)}, MDD={len(mdd)}")
        return

    print(f"\n  ADHD (N={len(adhd)}): depth_α/β = {adhd.median():.3f} (IQR [{adhd.quantile(0.25):.3f}, {adhd.quantile(0.75):.3f}])")
    print(f"  MDD  (N={len(mdd)}):  depth_α/β = {mdd.median():.3f} (IQR [{mdd.quantile(0.25):.3f}, {mdd.quantile(0.75):.3f}])")

    # Higher depth ratio = shallower trough = less inhibitory carving
    # Prediction: ADHD > MDD (shallower)
    u_stat, p_mann = stats.mannwhitneyu(adhd, mdd, alternative='greater')
    cohens_d = (adhd.mean() - mdd.mean()) / np.sqrt((adhd.std()**2 + mdd.std()**2) / 2)

    print(f"\n  Mann-Whitney U (ADHD > MDD): U={u_stat:.0f}, p={p_mann:.4f}")
    print(f"  Cohen's d: {cohens_d:+.3f}")
    print(f"  Direction: {'ADHD shallower (✓ predicted)' if adhd.median() > mdd.median() else 'MDD shallower (✗ opposite to prediction)'}")

    # Age-controlled: ANCOVA-like using rank regression
    print("\n  --- Age-Controlled ---")
    adhd_full = adults[adults.dx_group == 'ADHD'][['age_float', 'depth_α/β']].dropna()
    mdd_full = adults[adults.dx_group == 'MDD'][['age_float', 'depth_α/β']].dropna()
    combined = pd.concat([
        adhd_full.assign(group='ADHD'),
        mdd_full.assign(group='MDD')
    ])
    # Partial correlation: depth ~ group controlling for age
    from scipy.stats import rankdata
    age_rank = rankdata(combined['age_float'])
    depth_rank = rankdata(combined['depth_α/β'])
    group_code = (combined['group'] == 'ADHD').astype(float).values

    # Residualize depth on age
    slope_age, intercept, _, _, _ = stats.linregress(age_rank, depth_rank)
    depth_resid = depth_rank - (slope_age * age_rank + intercept)

    # Compare residualized depth between groups
    adhd_resid = depth_resid[group_code == 1]
    mdd_resid = depth_resid[group_code == 0]
    u_age, p_age = stats.mannwhitneyu(adhd_resid, mdd_resid, alternative='greater')
    print(f"  Age-controlled Mann-Whitney U: U={u_age:.0f}, p={p_age:.4f}")

    # --- ALL TROUGHS ---
    print("\n" + "=" * 70)
    print("ALL TROUGHS: ADHD vs MDD")
    print("=" * 70)

    for label in SHORT_LABELS:
        col = f'depth_{label}'
        a = adults[adults.dx_group == 'ADHD'][col].dropna()
        m = adults[adults.dx_group == 'MDD'][col].dropna()
        if len(a) > 10 and len(m) > 10:
            u, p = stats.mannwhitneyu(a, m, alternative='two-sided')
            d = (a.mean() - m.mean()) / np.sqrt((a.std()**2 + m.std()**2) / 2)
            direction = 'ADHD shallower' if a.median() > m.median() else 'MDD shallower'
            sig = '*' if p < 0.05 else ' '
            print(f"  {label}: ADHD={a.median():.3f}, MDD={m.median():.3f}, "
                  f"d={d:+.3f}, p={p:.4f}{sig} ({direction})")

    # --- HEALTHY COMPARISON ---
    print("\n" + "=" * 70)
    print("HEALTHY vs ADHD vs MDD")
    print("=" * 70)

    healthy = adults[adults.dx_group == 'HEALTHY']['depth_α/β'].dropna()
    if len(healthy) >= 5:
        print(f"  HEALTHY (N={len(healthy)}): depth_α/β = {healthy.median():.3f}")
        print(f"  ADHD   (N={len(adhd)}):    depth_α/β = {adhd.median():.3f}")
        print(f"  MDD    (N={len(mdd)}):     depth_α/β = {mdd.median():.3f}")

        # Kruskal-Wallis
        h_stat, p_kw = stats.kruskal(healthy, adhd, mdd)
        print(f"\n  Kruskal-Wallis H={h_stat:.2f}, p={p_kw:.4f}")

        # Prediction: ADHD > HEALTHY > MDD (shallowest to deepest)
        order = sorted([(healthy.median(), 'HEALTHY'), (adhd.median(), 'ADHD'),
                        (mdd.median(), 'MDD')], reverse=True)
        print(f"  Order (shallowest→deepest): {' > '.join(name for _, name in order)}")
        predicted = 'ADHD > HEALTHY > MDD'
        observed = ' > '.join(name for _, name in order)
        print(f"  Predicted: {predicted}")
        print(f"  Observed:  {observed}")
        print(f"  Match: {'✓' if predicted == observed else '✗'}")

    # Save
    merged.to_csv(os.path.join(OUT_DIR, 'tdbrain_per_subject_trough_depths.csv'), index=False)

    summary = pd.DataFrame([
        {'test': 'ADHD_vs_MDD_alpha_beta', 'stat': u_stat, 'p': p_mann,
         'cohens_d': cohens_d, 'n_adhd': len(adhd), 'n_mdd': len(mdd)},
    ])
    summary.to_csv(os.path.join(OUT_DIR, 'tdbrain_primary_test.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
