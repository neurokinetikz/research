#!/usr/bin/env python3
"""
Voronoi Enrichment Lifespan Trajectory
========================================

Per-subject enrichment × age across three datasets spanning ages 5-77:
  - HBN (5-21, N=927, pediatric development)
  - Dortmund (20-70, N=608, adult aging)
  - LEMON (20-77, N=167, adult aging, bimodal age distribution)

Computes Spearman correlations with FDR correction per dataset,
then compares developmental vs aging trajectories.

Usage:
    python scripts/voronoi_lifespan_trajectory.py
    python scripts/voronoi_lifespan_trajectory.py --dataset dortmund
    python scripts/voronoi_lifespan_trajectory.py --dataset lemon
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


def lattice_coord(freqs, f0=F0):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def per_subject_enrichment(peaks_df, min_peaks=30):
    """Compute per-subject enrichment at each position × band."""
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


def run_age_analysis(peak_dir, demo_loader, label, output_csv=None):
    """Run age × enrichment analysis for a single dataset."""
    peak_files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    if not peak_files:
        print(f"  No peak files in {peak_dir}")
        return None

    demo = demo_loader()

    rows = []
    for f in peak_files:
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
        enrich = per_subject_enrichment(peaks, min_peaks=30)
        enrich['subject'] = sub_id
        if sub_id in demo:
            enrich['age'] = demo[sub_id]
        rows.append(enrich)

    df = pd.DataFrame(rows)
    n_with_age = df['age'].notna().sum()
    enrich_cols = [c for c in df.columns if any(c.startswith(b + '_') for b in BAND_ORDER)
                   and not c.endswith('_n_peaks')]

    print(f"\n  {label}: {n_with_age} subjects with age, range {df['age'].min():.1f}-{df['age'].max():.1f}")

    age_results = []
    for col in enrich_cols:
        valid = df[['age', col]].dropna()
        if len(valid) < 20:
            continue
        rho, p = stats.spearmanr(valid['age'], valid[col])
        age_results.append({'feature': col, 'rho': rho, 'p': p, 'n': len(valid),
                           'abs_rho': abs(rho)})

    age_df = pd.DataFrame(age_results)
    if len(age_df) == 0:
        return None

    reject, p_fdr, _, _ = multipletests(age_df['p'].values, method='fdr_bh', alpha=0.05)
    age_df['p_fdr'] = p_fdr
    age_df['significant'] = reject

    print(f"  Tests: {len(age_df)}, FDR survivors: {age_df['significant'].sum()}")

    top = age_df.nlargest(15, 'abs_rho')
    print(f"\n  Top 15 by |rho|:")
    print(f"  {'Feature':<35} {'rho':>7} {'p':>12} {'p_FDR':>10} {'N':>5} {'Sig'}")
    print(f"  {'-'*75}")
    for _, r in top.iterrows():
        sig = '***' if r['p_fdr'] < 0.001 else ('**' if r['p_fdr'] < 0.01 else ('*' if r['p_fdr'] < 0.05 else ''))
        print(f"  {r['feature']:<35} {r['rho']:>+.3f} {r['p']:>12.2e} {r['p_fdr']:>10.4f} {r['n']:>5d} {sig}")

    # Per-band summary
    print(f"\n  Per-band:")
    for band in BAND_ORDER:
        band_df = age_df[age_df['feature'].str.startswith(band + '_')]
        if len(band_df) == 0:
            continue
        n_sig = band_df['significant'].sum()
        best = band_df.loc[band_df['abs_rho'].idxmax()]
        print(f"    {band:<12}: {n_sig}/{len(band_df)} FDR sig, best={best['feature']} rho={best['rho']:+.3f}")

    if output_csv:
        age_df.to_csv(output_csv, index=False)
        print(f"\n  Saved to {output_csv}")

    return age_df


def load_hbn_demo():
    """Load HBN demographics across all releases."""
    demo = {}
    for release in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']:
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if os.path.exists(tsv):
            df = pd.read_csv(tsv, sep='\t')
            for _, row in df.iterrows():
                demo[row['participant_id']] = row['age']
    return demo


def load_dortmund_demo():
    """Load Dortmund demographics."""
    tsv = '/Volumes/T9/dortmund_data/participants.tsv'
    df = pd.read_csv(tsv, sep='\t')
    return {row['participant_id']: row['age'] for _, row in df.iterrows()}


def load_lemon_demo():
    """Load LEMON demographics (age bins → midpoints)."""
    meta = pd.read_csv('/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/'
                       'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
    age_map = {'20-25': 22.5, '25-30': 27.5, '30-35': 32.5, '35-40': 37.5,
               '55-60': 57.5, '60-65': 62.5, '65-70': 67.5, '70-75': 72.5, '75-80': 77.5}
    return {row['ID']: age_map.get(row['Age'], np.nan) for _, row in meta.iterrows()
            if pd.notna(age_map.get(row['Age']))}


def compare_trajectories(results_dict):
    """Compare age effect directions across datasets."""
    labels = list(results_dict.keys())
    if len(labels) < 2:
        return

    print(f"\n{'='*80}")
    print(f"  CROSS-DATASET AGE TRAJECTORY COMPARISON")
    print(f"{'='*80}")

    # Merge all
    merged = results_dict[labels[0]][['feature', 'rho', 'significant']].copy()
    merged = merged.rename(columns={'rho': f'rho_{labels[0]}', 'significant': f'sig_{labels[0]}'})

    for label in labels[1:]:
        other = results_dict[label][['feature', 'rho', 'significant']].copy()
        other = other.rename(columns={'rho': f'rho_{label}', 'significant': f'sig_{label}'})
        merged = pd.merge(merged, other, on='feature', how='outer')

    # Pairwise correlations
    print(f"\n  Pairwise correlation of age rhos:")
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1:]:
            c1, c2 = f'rho_{l1}', f'rho_{l2}'
            valid = merged[[c1, c2]].dropna()
            if len(valid) > 3:
                r, p = stats.pearsonr(valid[c1], valid[c2])
                print(f"    {l1} vs {l2}: r={r:.3f} (p={p:.4f})")

    # Classify patterns
    if len(labels) >= 2:
        dev_label = [l for l in labels if 'HBN' in l]
        age_labels = [l for l in labels if 'HBN' not in l]

        if dev_label and age_labels:
            dev = dev_label[0]
            age = age_labels[0]

            inv_u = []
            u_shape = []
            mono_up = []
            mono_down = []

            dev_col = f'rho_{dev}'
            age_col = f'rho_{age}'
            dev_sig = f'sig_{dev}'
            age_sig = f'sig_{age}'

            for _, r in merged.iterrows():
                if pd.isna(r.get(dev_col)) or pd.isna(r.get(age_col)):
                    continue
                if r.get(dev_sig, False) and r.get(age_sig, False):
                    if (r[dev_col] > 0) != (r[age_col] > 0):
                        if r[dev_col] > 0:
                            inv_u.append(r['feature'])
                        else:
                            u_shape.append(r['feature'])
                    else:
                        if r[dev_col] > 0:
                            mono_up.append(r['feature'])
                        else:
                            mono_down.append(r['feature'])

            print(f"\n  Lifespan patterns ({dev} vs {age}):")
            print(f"    Inverted-U (peaks mid-life): {len(inv_u)}")
            print(f"    U-shape (dips mid-life): {len(u_shape)}")
            print(f"    Monotonic ↑: {len(mono_up)}")
            print(f"    Monotonic ↓: {len(mono_down)}")


def main():
    parser = argparse.ArgumentParser(description='Voronoi lifespan trajectory')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['hbn', 'dortmund', 'lemon', 'all'])
    args = parser.parse_args()

    results = {}

    configs = {
        'hbn': {
            'dirs': [f'exports_adaptive_v4/hbn_{r}' for r in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']],
            'demo_loader': load_hbn_demo,
            'label': 'HBN (5-21)',
            'output': 'outputs/hbn_age_enrichment.csv',
        },
        'dortmund': {
            'dirs': ['exports_adaptive_v4/dortmund'],
            'demo_loader': load_dortmund_demo,
            'label': 'Dortmund (20-70)',
            'output': 'outputs/dortmund_age_enrichment.csv',
        },
        'lemon': {
            'dirs': ['exports_adaptive_v4/lemon'],
            'demo_loader': load_lemon_demo,
            'label': 'LEMON (20-77)',
            'output': 'outputs/lemon_age_enrichment.csv',
        },
    }

    datasets_to_run = [args.dataset] if args.dataset != 'all' else ['hbn', 'dortmund', 'lemon']

    for ds in datasets_to_run:
        cfg = configs[ds]
        # For HBN, combine all release directories
        if ds == 'hbn':
            # Run with combined peak files from all releases
            all_peaks_dirs = [d for d in cfg['dirs'] if os.path.exists(d)]
            if not all_peaks_dirs:
                continue
            # Create temporary combined view
            peak_files = []
            for d in all_peaks_dirs:
                peak_files.extend(glob.glob(os.path.join(d, '*_peaks.csv')))

            demo = cfg['demo_loader']()
            rows = []
            for f in peak_files:
                sub_id = os.path.basename(f).replace('_peaks.csv', '')
                peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
                enrich = per_subject_enrichment(peaks, min_peaks=30)
                enrich['subject'] = sub_id
                if sub_id in demo:
                    enrich['age'] = demo[sub_id]
                rows.append(enrich)

            df = pd.DataFrame(rows)
            n_with_age = df['age'].notna().sum()
            enrich_cols = [c for c in df.columns
                          if any(c.startswith(b + '_') for b in BAND_ORDER)
                          and not c.endswith('_n_peaks')]

            print(f"\n  {cfg['label']}: {n_with_age} subjects, ages {df['age'].min():.1f}-{df['age'].max():.1f}")

            age_results = []
            for col in enrich_cols:
                valid = df[['age', col]].dropna()
                if len(valid) < 50:
                    continue
                rho, p = stats.spearmanr(valid['age'], valid[col])
                age_results.append({'feature': col, 'rho': rho, 'p': p,
                                   'n': len(valid), 'abs_rho': abs(rho)})

            age_df = pd.DataFrame(age_results)
            reject, p_fdr, _, _ = multipletests(age_df['p'].values, method='fdr_bh', alpha=0.05)
            age_df['p_fdr'] = p_fdr
            age_df['significant'] = reject

            print(f"  Tests: {len(age_df)}, FDR survivors: {age_df['significant'].sum()}")
            age_df.to_csv(cfg['output'], index=False)
            results[cfg['label']] = age_df
        else:
            r = run_age_analysis(cfg['dirs'][0], cfg['demo_loader'],
                                cfg['label'], cfg['output'])
            if r is not None:
                results[cfg['label']] = r

    if len(results) > 1:
        compare_trajectories(results)


if __name__ == '__main__':
    main()
