#!/usr/bin/env python3
"""
Per-Subject Voronoi Enrichment × Emotion/Personality (LEMON)
=============================================================

Correlates per-subject enrichment at each position × band with
LEMON's 23 emotion and personality instruments (~50 subscales).

Usage:
    python scripts/per_subject_voronoi_personality.py
    python scripts/per_subject_voronoi_personality.py --condition EO
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

PHI_INV = 1.0 / PHI

POS_LIST = [
    ('boundary', 0.000), ('noble_6', round(PHI_INV**6, 6)),
    ('noble_5', round(PHI_INV**5, 6)), ('noble_4', round(PHI_INV**4, 6)),
    ('noble_3', round(PHI_INV**3, 6)), ('inv_noble_1', round(PHI_INV**2, 6)),
    ('attractor', 0.5), ('noble_1', round(PHI_INV, 6)),
    ('inv_noble_3', round(1 - PHI_INV**3, 6)), ('inv_noble_4', round(1 - PHI_INV**4, 6)),
    ('inv_noble_5', round(1 - PHI_INV**5, 6)), ('inv_noble_6', round(1 - PHI_INV**6, 6)),
]
POS_NAMES = [p[0] for p in POS_LIST]
POS_VALS = np.array([p[1] for p in POS_LIST])
N_POS = len(POS_VALS)

VORONOI_BINS = []
for i in range(N_POS):
    d_prev = (POS_VALS[i] - POS_VALS[(i - 1) % N_POS]) % 1.0
    d_next = (POS_VALS[(i + 1) % N_POS] - POS_VALS[i]) % 1.0
    VORONOI_BINS.append(d_prev / 2 + d_next / 2)

OCTAVE_BAND = {'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
               'n+2': 'beta_high', 'n+3': 'gamma'}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

EMO_DIR = ('/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/'
           'Emotion_and_Personality_Test_Battery_LEMON')


def lattice_coord(freqs, f0=F0):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def per_subject_enrichment(peaks_df, min_peaks=30):
    results = {}
    for octave, band in OCTAVE_BAND.items():
        bp = peaks_df[peaks_df.phi_octave == octave]['freq'].values
        n = len(bp)
        if n < min_peaks:
            continue
        u = lattice_coord(bp)
        dists = np.abs(u[:, None] - POS_VALS[None, :])
        dists = np.minimum(dists, 1 - dists)
        assignments = np.argmin(dists, axis=1)
        for i in range(N_POS):
            count = (assignments == i).sum()
            expected = VORONOI_BINS[i] * n
            results[f'{band}_{POS_NAMES[i]}'] = (count / expected - 1) * 100 if expected > 0 else np.nan
        bnd = results.get(f'{band}_boundary', np.nan)
        n1 = results.get(f'{band}_noble_1', np.nan)
        if not np.isnan(bnd) and not np.isnan(n1):
            results[f'{band}_mountain'] = n1 - bnd
        if band == 'beta_low':
            center = np.nanmean([results.get(f'{band}_{p}', np.nan)
                                 for p in ['noble_5', 'noble_4', 'noble_3']])
            results[f'{band}_ushape'] = bnd - center if not np.isnan(bnd) else np.nan
        results[f'{band}_n_peaks'] = n
    return results


def load_personality():
    """Load all emotion/personality scores."""
    scores = {}  # sub_id -> {instrument_subscale: value}

    csvs = sorted(glob.glob(os.path.join(EMO_DIR, '*.csv')))
    instrument_cols = {}

    for f in csvs:
        name = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f)
        if 'ID' not in df.columns:
            continue

        cols = [c for c in df.columns if c != 'ID']
        # Convert all to numeric
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Use all numeric columns with sufficient non-null values
        for c in cols:
            if df[c].notna().sum() < 100:
                continue
            key = f"{name}_{c}" if not c.startswith(name) else c
            instrument_cols[key] = (f, c)

            for _, row in df.iterrows():
                sub_id = row['ID']
                if sub_id not in scores:
                    scores[sub_id] = {}
                if pd.notna(row[c]):
                    scores[sub_id][key] = row[c]

    return scores, list(instrument_cols.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='EC', choices=['EC', 'EO'])
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    suffix = '_EO' if args.condition == 'EO' else ''
    peak_dir = f'exports_adaptive/lemon{suffix}'
    peak_files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    print(f"Loading {len(peak_files)} subjects from {peak_dir}")

    scores, score_names = load_personality()
    print(f"Personality instruments: {len(score_names)} subscales from {len(set(n.split('_')[0] for n in score_names))} instruments")

    # Compute per-subject enrichment
    rows = []
    for f in peak_files:
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
        enrich = per_subject_enrichment(peaks, min_peaks=30)
        enrich['subject'] = sub_id

        if sub_id in scores:
            for key, val in scores[sub_id].items():
                enrich[f'pers_{key}'] = val
        rows.append(enrich)

    df = pd.DataFrame(rows)
    print(f"Subjects: {len(df)}")

    enrich_cols = [c for c in df.columns if any(c.startswith(b + '_') for b in BAND_ORDER)
                   and not c.endswith('_n_peaks')]
    pers_cols = [c for c in df.columns if c.startswith('pers_')]
    print(f"Enrichment features: {len(enrich_cols)}")
    print(f"Personality subscales: {len(pers_cols)}")

    # Correlations
    results = []
    for pers in pers_cols:
        pers_name = pers.replace('pers_', '')
        for enrich in enrich_cols:
            valid = df[[enrich, pers]].dropna()
            n = len(valid)
            if n < 20:
                continue
            rho, p = stats.spearmanr(valid[enrich], valid[pers])
            results.append({
                'personality': pers_name, 'enrichment': enrich,
                'rho': rho, 'p': p, 'n': n, 'abs_rho': abs(rho),
            })

    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("No valid correlations.")
        return

    reject, p_fdr, _, _ = multipletests(results_df['p'].values, method='fdr_bh', alpha=0.05)
    results_df['p_fdr'] = p_fdr
    results_df['significant'] = reject

    n_tests = len(results_df)
    n_sig = results_df['significant'].sum()
    n_unc = (results_df['p'] < 0.05).sum()
    expected = n_tests * 0.05

    print(f"\n{'='*80}")
    print(f"  PERSONALITY × ENRICHMENT ({args.condition})")
    print(f"{'='*80}")
    print(f"\n  Total tests: {n_tests}")
    print(f"  FDR survivors (q=0.05): {n_sig}")
    print(f"  Uncorrected p<0.05: {n_unc} (expected by chance: {expected:.0f}, ratio: {n_unc/expected:.2f}x)")
    print(f"  Largest |rho|: {results_df['abs_rho'].max():.3f}")

    # Top results
    top = results_df.nlargest(25, 'abs_rho')
    print(f"\n  Top 25 by |rho|:")
    print(f"  {'Personality':<30} {'Enrichment':<30} {'rho':>6} {'p':>10} {'p_FDR':>10} {'N':>5} {'Sig'}")
    print(f"  {'-'*95}")
    for _, row in top.iterrows():
        sig = '***' if row['p_fdr'] < 0.001 else ('**' if row['p_fdr'] < 0.01 else ('*' if row['p_fdr'] < 0.05 else ''))
        print(f"  {row['personality']:<30} {row['enrichment']:<30} {row['rho']:>+.3f} {row['p']:>10.4f} {row['p_fdr']:>10.4f} {row['n']:>5d} {sig}")

    # Per-instrument summary
    print(f"\n  Per-instrument summary (best |rho|):")
    instruments = sorted(set(r.split('_')[0] for r in results_df['personality'].unique()))
    for inst in instruments:
        inst_df = results_df[results_df['personality'].str.startswith(inst)]
        if len(inst_df) == 0:
            continue
        n_sig_inst = inst_df['significant'].sum()
        best = inst_df.loc[inst_df['abs_rho'].idxmax()]
        if best['abs_rho'] > 0.15:
            print(f"    {inst:<20}: max |rho|={best['abs_rho']:.3f} ({best['personality']} × {best['enrichment']})"
                  f" {'*' if n_sig_inst > 0 else ''}")

    # Per-band summary
    print(f"\n  Per-band summary:")
    for band in BAND_ORDER:
        band_df = results_df[results_df['enrichment'].str.startswith(band + '_')]
        if len(band_df) == 0:
            continue
        n_sig_band = band_df['significant'].sum()
        best = band_df.loc[band_df['abs_rho'].idxmax()]
        print(f"    {band:<12}: {n_sig_band} FDR sig, best |rho|={best['abs_rho']:.3f}"
              f" ({best['enrichment']} × {best['personality']})")

    # Save
    out = args.output or f'outputs/lemon_{args.condition}_voronoi_personality.csv'
    results_df.to_csv(out, index=False)
    print(f"\n  Saved to {out}")


if __name__ == '__main__':
    main()
