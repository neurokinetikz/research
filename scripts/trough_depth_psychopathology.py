#!/usr/bin/env python3
"""
Analysis 6: Trough Depth × Psychopathology (HBN)
=================================================
Tests whether externalizing psychopathology predicts shallower troughs
(GABA deficit model) and internalizing predicts deeper α/β trough
specifically (GABAergic enhancement model).

Uses HBN (N≈906) with CBCL dimensional scores.

Usage:
    python scripts/trough_depth_psychopathology.py [--plot]
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

HBN_RELEASES = ['R1', 'R2', 'R3', 'R4', 'R6']
HBN_DEMO_TEMPLATE = '/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
PSY_VARS = ['externalizing', 'internalizing', 'p_factor', 'attention']


def per_subject_trough_depth(freqs, trough_hz):
    log_freqs = np.log(freqs)
    log_trough = np.log(trough_hz)
    trough_count = np.sum(np.abs(log_freqs - log_trough) < LOG_HALF_WINDOW)
    left_count = np.sum(np.abs(log_freqs - (log_trough - LOG_FLANK_OFFSET)) < LOG_HALF_WINDOW)
    right_count = np.sum(np.abs(log_freqs - (log_trough + LOG_FLANK_OFFSET)) < LOG_HALF_WINDOW)
    mean_flank = (left_count + right_count) / 2
    return trough_count / mean_flank if mean_flank > 0 else np.nan


def load_hbn_data():
    """Load HBN subjects with trough depths, age, and psychopathology scores."""
    # Load demographics + psychopathology
    demo_rows = []
    for release in HBN_RELEASES:
        tsv = HBN_DEMO_TEMPLATE.format(release=release)
        if not os.path.exists(tsv):
            continue
        d = pd.read_csv(tsv, sep='\t')
        d['release'] = release
        demo_rows.append(d)
    demo = pd.concat(demo_rows, ignore_index=True)
    print(f"  HBN demographics: {len(demo)} subjects")

    # Build lookup
    demo_map = {}
    for _, row in demo.iterrows():
        pid = row['participant_id']
        demo_map[pid] = {
            'age': float(row['age']) if pd.notna(row.get('age')) else np.nan,
        }
        for psy in PSY_VARS:
            demo_map[pid][psy] = pd.to_numeric(row.get(psy, np.nan), errors='coerce')

    # Load HBN peaks
    rows = []
    for release in HBN_RELEASES:
        subdir = f'hbn_{release}'
        path = os.path.join(PEAK_BASE, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        for f in files:
            subj_id = os.path.basename(f).replace('_peaks.csv', '')
            if subj_id not in demo_map:
                continue
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
            row = {
                'subject': subj_id,
                'n_peaks': len(freqs),
                **demo_map[subj_id],
            }
            for trough_hz, label in zip(KNOWN_TROUGHS_HZ, TROUGH_LABELS):
                row[f'depth_{label}'] = per_subject_trough_depth(freqs, trough_hz)
            rows.append(row)

    hbn_df = pd.DataFrame(rows)
    print(f"  HBN with peaks + demographics: {len(hbn_df)}")
    for psy in PSY_VARS:
        n_valid = hbn_df[psy].notna().sum()
        print(f"    {psy}: {n_valid} subjects with data")

    return hbn_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading HBN data...")
    df = load_hbn_data()

    depth_cols = [f'depth_{l}' for l in TROUGH_LABELS]

    # ===================================================================
    # Part 1: Raw correlations (trough depth vs psychopathology)
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Raw correlations (trough depth vs psychopathology)")
    print("=" * 70)
    print(f"\n  Note: depth_ratio < 1.0 = deeper trough")
    print(f"  Positive ρ = higher symptoms → shallower trough (GABA deficit)")
    print(f"  Negative ρ = higher symptoms → deeper trough\n")

    results_rows = []
    print(f"  {'Variable':>15s}", end='')
    for label in TROUGH_LABELS:
        print(f"  {label:>10s}", end='')
    print(f"  {'N':>5s}")

    for psy in PSY_VARS:
        print(f"  {psy:>15s}", end='')
        for label in TROUGH_LABELS:
            depth_col = f'depth_{label}'
            valid = df[[depth_col, psy]].dropna()
            if len(valid) < 50:
                print(f"  {'--':>10s}", end='')
                continue
            rho, p = spearmanr(valid[depth_col].values, valid[psy].values)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {rho:>6.3f}{sig:>4s}", end='')
            results_rows.append({
                'variable': psy, 'trough': label,
                'rho': rho, 'p': p, 'n': len(valid), 'type': 'raw',
            })
        n = df[psy].notna().sum()
        print(f"  {n:>5d}")

    # ===================================================================
    # Part 2: Age-partialed correlations
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 2: Age-partialed correlations")
    print("=" * 70)

    df_p = df.dropna(subset=['age']).copy()

    # Residualize on age (quadratic)
    for col in depth_cols + PSY_VARS:
        v = df_p[['age', col]].dropna()
        if len(v) > 50:
            coeffs = np.polyfit(v['age'].values, v[col].values, 2)
            df_p.loc[v.index, f'{col}_resid'] = v[col].values - np.polyval(coeffs, v['age'].values)

    print(f"\n  {'Variable':>15s}", end='')
    for label in TROUGH_LABELS:
        print(f"  {label:>10s}", end='')
    print()

    for psy in PSY_VARS:
        psy_resid = f'{psy}_resid'
        if psy_resid not in df_p.columns:
            continue
        print(f"  {psy:>15s}", end='')
        for label in TROUGH_LABELS:
            depth_resid = f'depth_{label}_resid'
            if depth_resid not in df_p.columns:
                print(f"  {'--':>10s}", end='')
                continue
            valid = df_p[[depth_resid, psy_resid]].dropna()
            if len(valid) < 50:
                print(f"  {'--':>10s}", end='')
                continue
            rho, p = spearmanr(valid[depth_resid].values, valid[psy_resid].values)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {rho:>6.3f}{sig:>4s}", end='')
            results_rows.append({
                'variable': psy, 'trough': label,
                'rho': rho, 'p': p, 'n': len(valid), 'type': 'age_partialed',
            })
        print()

    # ===================================================================
    # Part 3: FDR correction
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 3: FDR correction")
    print("=" * 70)

    results_df = pd.DataFrame(results_rows)

    for analysis_type in ['raw', 'age_partialed']:
        sub = results_df[results_df.type == analysis_type].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values('p')
        n_tests = len(sub)
        sub['rank'] = range(1, n_tests + 1)
        sub['fdr_threshold'] = sub['rank'] / n_tests * 0.05
        sub['fdr_sig'] = sub['p'] <= sub['fdr_threshold']

        n_sig = sub['fdr_sig'].sum()
        print(f"\n  {analysis_type}: {n_sig}/{n_tests} FDR survivors")

        if n_sig > 0:
            survivors = sub[sub.fdr_sig].sort_values('p')
            for _, r in survivors.iterrows():
                direction = 'shallower' if r['rho'] > 0 else 'deeper'
                print(f"    {r['variable']:>15s} × {r['trough']:>6s}: "
                      f"ρ = {r['rho']:+.3f}, p = {r['p']:.4f} → "
                      f"higher {r['variable']} = {direction} trough")

        print(f"\n  Top 5 ({analysis_type}):")
        for _, r in sub.head(5).iterrows():
            fdr = '✓' if r['fdr_sig'] else ''
            print(f"    {r['variable']:>15s} × {r['trough']:>6s}: "
                  f"ρ = {r['rho']:+.3f}, p = {r['p']:.4f} {fdr}")

    # ===================================================================
    # Part 4: Externalizing vs internalizing direction test
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 4: Externalizing vs internalizing direction comparison")
    print("=" * 70)

    raw = results_df[results_df.type == 'raw']
    for label in TROUGH_LABELS:
        ext = raw[(raw.variable == 'externalizing') & (raw.trough == label)]
        int_ = raw[(raw.variable == 'internalizing') & (raw.trough == label)]
        if len(ext) > 0 and len(int_) > 0:
            ext_rho = ext.iloc[0]['rho']
            int_rho = int_.iloc[0]['rho']
            diff = ext_rho - int_rho
            same_dir = (ext_rho > 0) == (int_rho > 0)
            print(f"  {label:>6s}: ext ρ = {ext_rho:+.3f}, int ρ = {int_rho:+.3f}, "
                  f"diff = {diff:+.3f}  {'SAME direction' if same_dir else 'DISSOCIATED'}")

    results_df.to_csv(os.path.join(OUT_DIR, 'trough_depth_psychopathology.csv'), index=False)
    print(f"\nResults saved to {OUT_DIR}/trough_depth_psychopathology.csv")


if __name__ == '__main__':
    main()
