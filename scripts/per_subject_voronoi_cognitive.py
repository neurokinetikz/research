#!/usr/bin/env python3
"""
Per-Subject Voronoi Enrichment × Cognitive Correlation
=======================================================

Computes per-subject enrichment at each position × band using Voronoi bins,
then correlates with LEMON cognitive test battery (8 tests).

Usage:
    python scripts/per_subject_voronoi_cognitive.py
    python scripts/per_subject_voronoi_cognitive.py --min-peaks 50
    python scripts/per_subject_voronoi_cognitive.py --condition EO
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

PHI_INV = 1.0 / PHI

# Degree-6 positions
POS_LIST = [
    ('boundary',    0.000),
    ('noble_6',     round(PHI_INV ** 6, 6)),
    ('noble_5',     round(PHI_INV ** 5, 6)),
    ('noble_4',     round(PHI_INV ** 4, 6)),
    ('noble_3',     round(PHI_INV ** 3, 6)),
    ('inv_noble_1', round(PHI_INV ** 2, 6)),
    ('attractor',   0.5),
    ('noble_1',     round(PHI_INV, 6)),
    ('inv_noble_3', round(1 - PHI_INV ** 3, 6)),
    ('inv_noble_4', round(1 - PHI_INV ** 4, 6)),
    ('inv_noble_5', round(1 - PHI_INV ** 5, 6)),
    ('inv_noble_6', round(1 - PHI_INV ** 6, 6)),
]

POS_NAMES = [p[0] for p in POS_LIST]
POS_VALS = np.array([p[1] for p in POS_LIST])
N_POS = len(POS_VALS)

# Voronoi cell sizes
VORONOI_BINS = []
for i in range(N_POS):
    d_prev = (POS_VALS[i] - POS_VALS[(i - 1) % N_POS]) % 1.0
    d_next = (POS_VALS[(i + 1) % N_POS] - POS_VALS[i]) % 1.0
    VORONOI_BINS.append(d_prev / 2 + d_next / 2)

OCTAVE_BAND = {'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low', 'n+2': 'beta_high', 'n+3': 'gamma'}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

# Cognitive tests
COG_DIR = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/Cognitive_Test_Battery_LEMON'
META_PATH = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'

# Summary scores per test (first or most representative column)
COG_TESTS = {
    'CVLT': ('CVLT /CVLT.csv', 'CVLT_1'),       # trial 1 recall
    'LPS': ('LPS/LPS.csv', 'LPS_1'),             # performance score
    'RWT': ('RWT/RWT.csv', 'RWT_1'),             # verbal fluency
    'TAP_Alert': ('TAP_Alertness/TAP-Alertness.csv', 'TAP_A_1'),  # alertness RT
    'TAP_Incompat': ('TAP_Incompatibility/TAP-Incompatibility.csv', 'TAP_I_1'),  # incompatibility RT
    'TAP_WM': ('TAP_Working_Memory/TAP-Working Memory.csv', 'TAP_WM_1'),  # working memory
    'TMT': ('TMT/TMT.csv', 'TMT_1'),             # trail making time
    'WST': ('WST/WST.csv', 'WST_1'),             # vocabulary
}


def lattice_coord(freqs, f0=F0):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def per_subject_enrichment(peaks_df, min_peaks=30):
    """Compute enrichment at each position for a single subject."""
    results = {}
    for octave, band in OCTAVE_BAND.items():
        bp = peaks_df[peaks_df.phi_octave == octave]['freq'].values
        n = len(bp)
        if n < min_peaks:
            continue

        u = lattice_coord(bp)
        pos_arr = POS_VALS
        dists = np.abs(u[:, None] - pos_arr[None, :])
        dists = np.minimum(dists, 1 - dists)
        assignments = np.argmin(dists, axis=1)

        for i in range(N_POS):
            count = (assignments == i).sum()
            expected = VORONOI_BINS[i] * n
            enrich = (count / expected - 1) * 100 if expected > 0 else np.nan
            results[f'{band}_{POS_NAMES[i]}'] = enrich

        # Also compute summary metrics
        # Mountain height: Noble1 - boundary average
        bnd_enrich = results.get(f'{band}_boundary', np.nan)
        n1_enrich = results.get(f'{band}_noble_1', np.nan)
        if not np.isnan(bnd_enrich) and not np.isnan(n1_enrich):
            results[f'{band}_mountain'] = n1_enrich - bnd_enrich

        # U-shape depth: boundary average - center average
        if band == 'beta_low':
            bnd_avg = np.nanmean([results.get(f'{band}_boundary', np.nan)])
            center_avg = np.nanmean([results.get(f'{band}_{p}', np.nan)
                                     for p in ['noble_5', 'noble_4', 'noble_3']])
            results[f'{band}_ushape'] = bnd_avg - center_avg

        results[f'{band}_n_peaks'] = n

    return results


def load_cognitive():
    """Load all cognitive test scores."""
    cog_data = {}
    for test_name, (filename, col) in COG_TESTS.items():
        path = os.path.join(COG_DIR, filename)
        df = pd.read_csv(path)
        # Convert to numeric, coerce errors
        df[col] = pd.to_numeric(df[col], errors='coerce')
        for _, row in df.iterrows():
            sub_id = row['ID']
            if sub_id not in cog_data:
                cog_data[sub_id] = {}
            cog_data[sub_id][test_name] = row[col]
    return cog_data


def load_demographics():
    """Load age and sex from META file."""
    meta = pd.read_csv(META_PATH)
    demo = {}
    for _, row in meta.iterrows():
        sub_id = row['ID']
        demo[sub_id] = {
            'age_bin': row['Age'],
            'sex': row['Gender_ 1=female_2=male'],
        }
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-peaks', type=int, default=30,
                        help='Minimum peaks per band for per-subject enrichment')
    parser.add_argument('--condition', type=str, default='EC',
                        choices=['EC', 'EO'], help='LEMON condition')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')
    args = parser.parse_args()

    # Load peaks
    suffix = '_EO' if args.condition == 'EO' else ''
    peak_dir = f'exports_adaptive/lemon{suffix}'
    peak_files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    print(f"Loading {len(peak_files)} subjects from {peak_dir}")

    # Load cognitive and demographic data
    cog_data = load_cognitive()
    demo_data = load_demographics()

    # Compute per-subject enrichment
    rows = []
    for f in peak_files:
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
        enrich = per_subject_enrichment(peaks, min_peaks=args.min_peaks)
        enrich['subject'] = sub_id

        # Add cognitive scores
        if sub_id in cog_data:
            for test, score in cog_data[sub_id].items():
                enrich[f'cog_{test}'] = score

        # Add demographics
        if sub_id in demo_data:
            enrich['age_bin'] = demo_data[sub_id]['age_bin']
            enrich['sex'] = demo_data[sub_id]['sex']

        rows.append(enrich)

    df = pd.DataFrame(rows)
    print(f"Per-subject enrichment computed: {len(df)} subjects")

    # Identify enrichment and cognitive columns
    enrich_cols = [c for c in df.columns if any(c.startswith(b + '_') for b in BAND_ORDER)
                   and not c.endswith('_n_peaks')]
    cog_cols = [c for c in df.columns if c.startswith('cog_')]

    print(f"Enrichment features: {len(enrich_cols)}")
    print(f"Cognitive tests: {len(cog_cols)}")

    # Correlations
    print(f"\n{'='*80}")
    print(f"  PER-SUBJECT VORONOI ENRICHMENT × COGNITIVE CORRELATIONS")
    print(f"  (Spearman rank, N varies by band due to min_peaks={args.min_peaks})")
    print(f"{'='*80}")

    results = []
    for cog in cog_cols:
        cog_name = cog.replace('cog_', '')
        for enrich in enrich_cols:
            valid = df[[enrich, cog]].dropna()
            n = len(valid)
            if n < 20:
                continue
            rho, p = stats.spearmanr(valid[enrich], valid[cog])
            results.append({
                'cognitive': cog_name,
                'enrichment': enrich,
                'rho': rho,
                'p': p,
                'n': n,
                'abs_rho': abs(rho),
            })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No valid correlations computed.")
        return

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    reject, p_fdr, _, _ = multipletests(results_df['p'].values, method='fdr_bh', alpha=0.05)
    results_df['p_fdr'] = p_fdr
    results_df['significant'] = reject

    n_tests = len(results_df)
    n_sig = results_df['significant'].sum()
    print(f"\n  Total tests: {n_tests}")
    print(f"  FDR survivors (q=0.05): {n_sig}")
    print(f"  Largest |rho|: {results_df['abs_rho'].max():.3f}")

    # Show top results
    top = results_df.nlargest(20, 'abs_rho')
    print(f"\n  Top 20 by |rho|:")
    print(f"  {'Cognitive':<15} {'Enrichment':<30} {'rho':>6} {'p':>10} {'p_FDR':>10} {'N':>5} {'Sig':>5}")
    print(f"  {'-'*80}")
    for _, row in top.iterrows():
        sig = '***' if row['p_fdr'] < 0.001 else ('**' if row['p_fdr'] < 0.01 else ('*' if row['p_fdr'] < 0.05 else ''))
        print(f"  {row['cognitive']:<15} {row['enrichment']:<30} {row['rho']:>+.3f} {row['p']:>10.4f} {row['p_fdr']:>10.4f} {row['n']:>5d} {sig:>5}")

    # Per-band summary
    print(f"\n  Per-band summary (max |rho| across all cognitive tests):")
    for band in BAND_ORDER:
        band_results = results_df[results_df['enrichment'].str.startswith(band + '_')]
        if len(band_results) == 0:
            continue
        best = band_results.loc[band_results['abs_rho'].idxmax()]
        n_sig_band = band_results['significant'].sum()
        print(f"  {band:<12}: max |rho|={best['abs_rho']:.3f} ({best['enrichment']} × {best['cognitive']}), "
              f"{n_sig_band}/{len(band_results)} FDR significant")

    # Age correlation with enrichment
    print(f"\n\n{'='*80}")
    print(f"  ENRICHMENT × AGE")
    print(f"{'='*80}")

    # Convert age bins to numeric midpoints
    age_map = {'20-25': 22.5, '25-30': 27.5, '30-35': 32.5, '35-40': 37.5,
               '55-60': 57.5, '60-65': 62.5, '65-70': 67.5, '70-75': 72.5, '75-80': 77.5}
    df['age_numeric'] = df['age_bin'].map(age_map)

    age_results = []
    for enrich in enrich_cols:
        valid = df[[enrich, 'age_numeric']].dropna()
        n = len(valid)
        if n < 20:
            continue
        rho, p = stats.spearmanr(valid[enrich], valid['age_numeric'])
        age_results.append({'enrichment': enrich, 'rho': rho, 'p': p, 'n': n, 'abs_rho': abs(rho)})

    age_df = pd.DataFrame(age_results)
    if len(age_df) > 0:
        reject_age, p_fdr_age, _, _ = multipletests(age_df['p'].values, method='fdr_bh', alpha=0.05)
        age_df['p_fdr'] = p_fdr_age
        age_df['significant'] = reject_age

        n_sig_age = age_df['significant'].sum()
        print(f"\n  Total tests: {len(age_df)}")
        print(f"  FDR survivors: {n_sig_age}")

        top_age = age_df.nlargest(10, 'abs_rho')
        print(f"\n  Top 10 by |rho|:")
        print(f"  {'Enrichment':<30} {'rho':>6} {'p':>10} {'p_FDR':>10} {'N':>5} {'Sig':>5}")
        print(f"  {'-'*65}")
        for _, row in top_age.iterrows():
            sig = '*' if row['p_fdr'] < 0.05 else ''
            print(f"  {row['enrichment']:<30} {row['rho']:>+.3f} {row['p']:>10.4f} {row['p_fdr']:>10.4f} {row['n']:>5d} {sig:>5}")

    # Save full results
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
    else:
        out_path = f'outputs/lemon_{args.condition}_voronoi_cognitive.csv'
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
