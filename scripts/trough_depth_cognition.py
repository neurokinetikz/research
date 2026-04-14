#!/usr/bin/env python3
"""
Analysis 5: Per-Subject Trough Depth vs Cognition (LEMON)
=========================================================
Tests whether individual variation in trough depth predicts cognitive
performance, independently of the within-band enrichment features
already in the paper.

If spectral differentiation reflects inhibitory circuit integrity,
and trough depth is the boundary-level expression of that integrity,
then deeper troughs should predict better cognition.

Uses LEMON (N≈203) with 8 cognitive tests.

Usage:
    python scripts/trough_depth_cognition.py [--plot]
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age')
MIN_POWER_PCT = 50

KNOWN_TROUGHS_HZ = np.array([5.08, 7.81, 13.42, 25.30, 35.04])
TROUGH_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']

LOG_HALF_WINDOW = 0.06
LOG_FLANK_OFFSET = 0.15

# LEMON cognitive tests
COG_BASE = ('/Volumes/T9/lemon_data/behavioral/'
            'Behavioural_Data_MPILMBB_LEMON/Cognitive_Test_Battery_LEMON')
COG_TESTS = {
    'LPS': ('LPS/LPS.csv', 'LPS_1'),
    'RWT': ('RWT/RWT.csv', 'RWT_1'),
    'TMT': ('TMT/TMT.csv', 'TMT_1'),
    'CVLT': ('CVLT /CVLT.csv', 'CVLT_1'),
    'WST': ('WST/WST.csv', 'WST_1'),
    'TAP_Alert': ('TAP_Alertness/TAP-Alertness.csv', 'TAP_A_1'),
    'TAP_Incompat': ('TAP_Incompatibility/TAP-Incompatibility.csv', 'TAP_I_1'),
    'TAP_WM': ('TAP_Working_Memory/TAP-Working Memory.csv', 'TAP_WM_1'),
}

LEMON_META = ('/Volumes/T9/lemon_data/behavioral/'
              'Behavioural_Data_MPILMBB_LEMON/'
              'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
LEMON_AGE_MAP = {
    '20-25': 22.5, '25-30': 27.5, '30-35': 32.5, '35-40': 37.5,
    '55-60': 57.5, '60-65': 62.5, '65-70': 67.5, '70-75': 72.5, '75-80': 77.5,
}


def per_subject_trough_depth(freqs, trough_hz):
    """Windowed count depth ratio for a single subject at one trough."""
    log_freqs = np.log(freqs)
    log_trough = np.log(trough_hz)
    trough_count = np.sum(np.abs(log_freqs - log_trough) < LOG_HALF_WINDOW)
    left_count = np.sum(np.abs(log_freqs - (log_trough - LOG_FLANK_OFFSET)) < LOG_HALF_WINDOW)
    right_count = np.sum(np.abs(log_freqs - (log_trough + LOG_FLANK_OFFSET)) < LOG_HALF_WINDOW)
    mean_flank = (left_count + right_count) / 2
    return trough_count / mean_flank if mean_flank > 0 else np.nan


def load_lemon_depths_and_cog():
    """Load LEMON subjects with trough depths, age, and cognitive scores."""
    # Load LEMON peaks
    lemon_path = os.path.join(PEAK_BASE, 'lemon')
    files = sorted(glob.glob(os.path.join(lemon_path, '*_peaks.csv')))
    first = pd.read_csv(files[0], nrows=1)
    has_power = 'power' in first.columns

    rows = []
    for f in files:
        subj_id = os.path.basename(f).replace('_peaks.csv', '')
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        df = pd.read_csv(f, usecols=cols)
        if has_power and MIN_POWER_PCT > 0:
            filtered = []
            for octave in df['phi_octave'].unique():
                bp = df[df.phi_octave == octave]
                thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                filtered.append(bp[bp['power'] >= thresh])
            df = pd.concat(filtered, ignore_index=True)
        freqs = df['freq'].values
        if len(freqs) < 100:
            continue
        row = {'subject': subj_id, 'n_peaks': len(freqs)}
        for trough_hz, label in zip(KNOWN_TROUGHS_HZ, TROUGH_LABELS):
            row[f'depth_{label}'] = per_subject_trough_depth(freqs, trough_hz)
        rows.append(row)

    lemon_df = pd.DataFrame(rows)
    print(f"  LEMON peaks: {len(lemon_df)} subjects")

    # Load age
    if os.path.exists(LEMON_META):
        meta = pd.read_csv(LEMON_META)
        age_map = {}
        for _, r in meta.iterrows():
            mid = LEMON_AGE_MAP.get(str(r.get('Age', '')), np.nan)
            if pd.notna(mid):
                age_map[r['ID']] = mid
        lemon_df['age'] = lemon_df['subject'].map(age_map)

    # Load cognitive scores
    for test_name, (subpath, col) in COG_TESTS.items():
        path = os.path.join(COG_BASE, subpath)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        cog_df = pd.read_csv(path)
        cog_df[col] = pd.to_numeric(cog_df[col], errors='coerce')
        cog_map = dict(zip(cog_df['ID'], cog_df[col]))
        lemon_df[f'cog_{test_name}'] = lemon_df['subject'].map(cog_map)

    n_cog = sum(1 for c in lemon_df.columns if c.startswith('cog_'))
    n_with_any_cog = lemon_df[[c for c in lemon_df.columns if c.startswith('cog_')]].notna().any(axis=1).sum()
    print(f"  Cognitive tests loaded: {n_cog}")
    print(f"  Subjects with any cognitive data: {n_with_any_cog}")

    return lemon_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading LEMON data...")
    df = load_lemon_depths_and_cog()

    depth_cols = [f'depth_{l}' for l in TROUGH_LABELS]
    cog_cols = [c for c in df.columns if c.startswith('cog_')]

    # ===================================================================
    # Part 1: Raw Spearman correlations (trough depth vs cognition)
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Raw correlations (trough depth vs cognition)")
    print("=" * 70)
    print(f"\n  Note: depth_ratio < 1.0 = depletion (deeper trough)")
    print(f"  Negative ρ = deeper trough → higher score (expected if inhibition helps)")
    print(f"  For TMT: higher score = worse (it's a time-based test), so positive ρ expected\n")

    results_rows = []
    print(f"  {'Test':>15s}", end='')
    for label in TROUGH_LABELS:
        print(f"  {label:>10s}", end='')
    print(f"  {'N':>5s}")

    for cog_col in sorted(cog_cols):
        test_name = cog_col.replace('cog_', '')
        print(f"  {test_name:>15s}", end='')
        for label in TROUGH_LABELS:
            depth_col = f'depth_{label}'
            valid = df[[depth_col, cog_col]].dropna()
            if len(valid) < 30:
                print(f"  {'--':>10s}", end='')
                continue
            rho, p = spearmanr(valid[depth_col].values, valid[cog_col].values)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {rho:>6.3f}{sig:>4s}", end='')
            results_rows.append({
                'test': test_name, 'trough': label,
                'rho': rho, 'p': p, 'n': len(valid),
                'type': 'raw',
            })
        n = df[[cog_col]].dropna().shape[0]
        print(f"  {n:>5d}")

    # ===================================================================
    # Part 2: Age-partialed correlations
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 2: Age-partialed correlations")
    print("=" * 70)

    df_valid = df.dropna(subset=['age'])
    print(f"\n  N with age: {len(df_valid)}")

    # Residualize depths on age (quadratic)
    for col in depth_cols:
        v = df_valid[['age', col]].dropna()
        if len(v) > 50:
            coeffs = np.polyfit(v['age'].values, v[col].values, 2)
            df_valid.loc[v.index, f'{col}_resid'] = v[col].values - np.polyval(coeffs, v['age'].values)

    # Residualize cognitive scores on age
    for col in cog_cols:
        v = df_valid[['age', col]].dropna()
        if len(v) > 50:
            coeffs = np.polyfit(v['age'].values, v[col].values, 2)
            df_valid.loc[v.index, f'{col}_resid'] = v[col].values - np.polyval(coeffs, v['age'].values)

    print(f"\n  {'Test':>15s}", end='')
    for label in TROUGH_LABELS:
        print(f"  {label:>10s}", end='')
    print()

    for cog_col in sorted(cog_cols):
        test_name = cog_col.replace('cog_', '')
        cog_resid = f'{cog_col}_resid'
        if cog_resid not in df_valid.columns:
            continue
        print(f"  {test_name:>15s}", end='')
        for label in TROUGH_LABELS:
            depth_resid = f'depth_{label}_resid'
            if depth_resid not in df_valid.columns:
                print(f"  {'--':>10s}", end='')
                continue
            valid = df_valid[[depth_resid, cog_resid]].dropna()
            if len(valid) < 30:
                print(f"  {'--':>10s}", end='')
                continue
            rho, p = spearmanr(valid[depth_resid].values, valid[cog_resid].values)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {rho:>6.3f}{sig:>4s}", end='')
            results_rows.append({
                'test': test_name, 'trough': label,
                'rho': rho, 'p': p, 'n': len(valid),
                'type': 'age_partialed',
            })
        print()

    # ===================================================================
    # Part 3: FDR correction
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 3: FDR correction (Benjamini-Hochberg)")
    print("=" * 70)

    results_df = pd.DataFrame(results_rows)

    for analysis_type in ['raw', 'age_partialed']:
        sub = results_df[results_df.type == analysis_type].copy()
        if len(sub) == 0:
            continue
        # BH correction
        sub = sub.sort_values('p')
        n_tests = len(sub)
        sub['rank'] = range(1, n_tests + 1)
        sub['fdr_threshold'] = sub['rank'] / n_tests * 0.05
        sub['fdr_sig'] = sub['p'] <= sub['fdr_threshold']

        n_sig = sub['fdr_sig'].sum()
        print(f"\n  {analysis_type}: {n_sig}/{n_tests} FDR survivors (q < 0.05)")

        if n_sig > 0:
            survivors = sub[sub.fdr_sig].sort_values('p')
            for _, r in survivors.iterrows():
                print(f"    {r['test']:>15s} × {r['trough']:>6s}: "
                      f"ρ = {r['rho']:+.3f}, p = {r['p']:.4f}, N = {int(r['n'])}")

        # Also show top 5 regardless
        print(f"\n  Top 5 ({analysis_type}):")
        for _, r in sub.head(5).iterrows():
            fdr = '✓' if r['fdr_sig'] else ''
            print(f"    {r['test']:>15s} × {r['trough']:>6s}: "
                  f"ρ = {r['rho']:+.3f}, p = {r['p']:.4f} {fdr}")

    results_df.to_csv(os.path.join(OUT_DIR, 'trough_depth_cognition.csv'), index=False)
    print(f"\nResults saved to {OUT_DIR}/trough_depth_cognition.csv")

    # ===================================================================
    # Part 4: Compare to within-band enrichment effect sizes
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 4: Effect size comparison")
    print("=" * 70)

    raw = results_df[results_df.type == 'raw']
    if len(raw) > 0:
        peak_rho = raw.loc[raw['rho'].abs().idxmax()]
        print(f"\n  Peak trough-depth × cognition (raw): "
              f"|ρ| = {abs(peak_rho['rho']):.3f} "
              f"({peak_rho['test']} × {peak_rho['trough']})")
        print(f"  Paper's peak enrichment × cognition (raw): |ρ| ≈ 0.27")
        print(f"  Ratio: {abs(peak_rho['rho']) / 0.27:.2f}×")

    partial = results_df[results_df.type == 'age_partialed']
    if len(partial) > 0:
        peak_rho_p = partial.loc[partial['rho'].abs().idxmax()]
        print(f"\n  Peak trough-depth × cognition (age-partialed): "
              f"|ρ| = {abs(peak_rho_p['rho']):.3f} "
              f"({peak_rho_p['test']} × {peak_rho_p['trough']})")
        print(f"  Paper's peak enrichment × cognition (age-partialed): |ρ| ≈ 0.15")


if __name__ == '__main__':
    main()
