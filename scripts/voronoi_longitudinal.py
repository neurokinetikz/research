#!/usr/bin/env python3
"""
Voronoi Enrichment 5-Year Longitudinal Analysis (Dortmund)
============================================================

Compares per-subject enrichment between ses-1 and ses-2 (~5 years apart)
for N=208 returning subjects. Tests:
  1. ICC(2,1) for each enrichment feature (vs Paper 3's -0.25 to -0.36)
  2. Group-level stability (does the population profile change?)
  3. Does baseline enrichment predict 5-year change?

Usage:
    python scripts/voronoi_longitudinal.py
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy import stats

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


def icc_21(x, y):
    """Compute ICC(2,1) for two matched measurements."""
    n = len(x)
    if n < 3:
        return np.nan
    ms_between = 2 * np.sum(((x + y) / 2 - np.mean(np.concatenate([x, y]))) ** 2) / (n - 1)
    ms_within = np.sum((x - y) ** 2) / (2 * n)
    if (ms_between + ms_within) == 0:
        return np.nan
    return (ms_between - ms_within) / (ms_between + ms_within)


def main():
    ses1_dir = 'exports_adaptive/dortmund'
    ses2_dir = 'exports_adaptive/dortmund_EC_pre_ses2'

    if not os.path.exists(ses2_dir):
        print(f"ERROR: {ses2_dir} not found. Run extraction first:")
        print(f"  python scripts/run_adaptive_resolution_extraction.py --dataset dortmund --condition EC-pre --session 2")
        return

    # Match subjects
    ses1_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                  for f in glob.glob(os.path.join(ses1_dir, '*_peaks.csv'))}
    ses2_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                  for f in glob.glob(os.path.join(ses2_dir, '*_peaks.csv'))}
    matched = sorted(set(ses1_files.keys()) & set(ses2_files.keys()))
    print(f"Matched subjects: {len(matched)} (ses-1: {len(ses1_files)}, ses-2: {len(ses2_files)})")

    # Load demographics for age at ses-1
    demo = pd.read_csv('/Volumes/T9/dortmund_data/participants.tsv', sep='\t')
    demo_dict = {r['participant_id']: r['age'] for _, r in demo.iterrows()}

    # Compute per-subject enrichment for both sessions
    data_s1 = {}
    data_s2 = {}
    ages = {}
    for sub_id in matched:
        p1 = pd.read_csv(ses1_files[sub_id], usecols=['freq', 'phi_octave'])
        p2 = pd.read_csv(ses2_files[sub_id], usecols=['freq', 'phi_octave'])
        data_s1[sub_id] = per_subject_enrichment(p1)
        data_s2[sub_id] = per_subject_enrichment(p2)
        ages[sub_id] = demo_dict.get(sub_id, np.nan)

    # Get all enrichment features
    all_features = set()
    for d in list(data_s1.values()) + list(data_s2.values()):
        all_features.update(k for k in d.keys() if not k.endswith('_n_peaks'))

    # === 1. ICC and test-retest ===
    print(f"\n{'='*80}")
    print(f"  5-YEAR TEST-RETEST RELIABILITY (N={len(matched)})")
    print(f"  Compare to Paper 3 dominant-peak ICC: -0.25 to -0.36")
    print(f"{'='*80}")

    results = []
    for feat in sorted(all_features):
        vals_s1, vals_s2 = [], []
        for sub_id in matched:
            v1 = data_s1[sub_id].get(feat, np.nan)
            v2 = data_s2[sub_id].get(feat, np.nan)
            if not np.isnan(v1) and not np.isnan(v2):
                vals_s1.append(v1)
                vals_s2.append(v2)

        if len(vals_s1) < 20:
            continue

        x, y = np.array(vals_s1), np.array(vals_s2)
        r_val, p_r = stats.pearsonr(x, y)
        rho, p_s = stats.spearmanr(x, y)
        icc = icc_21(x, y)
        mean_s1 = np.mean(x)
        mean_s2 = np.mean(y)
        mean_diff = np.mean(y - x)
        pooled_sd = np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)
        d = mean_diff / pooled_sd if pooled_sd > 0 else 0

        results.append({
            'feature': feat, 'n': len(vals_s1),
            'mean_s1': mean_s1, 'mean_s2': mean_s2,
            'pearson_r': r_val, 'spearman_rho': rho,
            'icc': icc, 'cohen_d': d,
        })

    rdf = pd.DataFrame(results)

    print(f"\n  {'Feature':<30} {'N':>5} {'S1':>7} {'S2':>7} {'r':>6} {'ICC':>6} {'d':>6}")
    print(f"  {'-'*70}")

    for band in BAND_ORDER:
        band_df = rdf[rdf['feature'].str.startswith(band + '_')]
        if len(band_df) == 0:
            continue
        for _, row in band_df.sort_values('feature').iterrows():
            print(f"  {row['feature']:<30} {row['n']:>5} {row['mean_s1']:>+6.0f}% {row['mean_s2']:>+6.0f}%"
                  f" {row['pearson_r']:>+.3f} {row['icc']:>+.3f} {row['cohen_d']:>+.3f}")

    # Per-band summary
    print(f"\n  Per-band 5-year ICC summary:")
    print(f"  {'Band':<12} {'median ICC':>10} {'range':>20} {'median r':>10} {'Paper 3 ICC':>12}")
    for band in BAND_ORDER:
        band_df = rdf[rdf['feature'].str.startswith(band + '_')]
        if len(band_df) == 0:
            continue
        print(f"  {band:<12} {band_df['icc'].median():>+10.3f}"
              f" [{band_df['icc'].min():>+.3f}, {band_df['icc'].max():>+.3f}]"
              f" {band_df['pearson_r'].median():>+10.3f}"
              f" {'−0.25 to −0.36':>12}")

    overall_icc = rdf['icc'].median()
    overall_r = rdf['pearson_r'].median()
    print(f"\n  Overall: median ICC = {overall_icc:+.3f}, median r = {overall_r:+.3f}")
    print(f"  Paper 3 dominant-peak: ICC = -0.25 to -0.36")
    if overall_icc > 0:
        print(f"  → Voronoi enrichment is MORE stable than dominant-peak alignment")
    else:
        print(f"  → Voronoi enrichment is ALSO unstable across 5 years")

    # === 2. Group-level stability ===
    print(f"\n\n{'='*80}")
    print(f"  GROUP-LEVEL STABILITY")
    print(f"{'='*80}")

    print(f"\n  Mean enrichment ses-1 vs ses-2 (paired t-test):")
    sig_changes = []
    for _, row in rdf.iterrows():
        if abs(row['cohen_d']) > 0.2:
            sig_changes.append(row)

    if sig_changes:
        print(f"  Features with |d| > 0.2:")
        for r in sig_changes:
            print(f"    {r['feature']:<30} S1={r['mean_s1']:>+.0f}% S2={r['mean_s2']:>+.0f}% d={r['cohen_d']:>+.3f}")
    else:
        print(f"  No features with |d| > 0.2 — group profile is stable")

    # === 3. Does ses-1 age predict 5-year change? ===
    print(f"\n\n{'='*80}")
    print(f"  DOES BASELINE AGE PREDICT 5-YEAR ENRICHMENT CHANGE?")
    print(f"{'='*80}")

    from statsmodels.stats.multitest import multipletests

    change_age_results = []
    for feat in sorted(all_features):
        deltas, ages_list = [], []
        for sub_id in matched:
            v1 = data_s1[sub_id].get(feat, np.nan)
            v2 = data_s2[sub_id].get(feat, np.nan)
            age = ages.get(sub_id, np.nan)
            if not np.isnan(v1) and not np.isnan(v2) and not np.isnan(age):
                deltas.append(v2 - v1)
                ages_list.append(age)

        if len(deltas) < 20:
            continue

        rho, p = stats.spearmanr(ages_list, deltas)
        change_age_results.append({
            'feature': feat, 'rho': rho, 'p': p,
            'n': len(deltas), 'abs_rho': abs(rho),
        })

    ca_df = pd.DataFrame(change_age_results)
    if len(ca_df) > 0:
        rej, pfdr, _, _ = multipletests(ca_df['p'].values, method='fdr_bh', alpha=0.05)
        ca_df['p_fdr'] = pfdr
        ca_df['significant'] = rej

        n_sig = ca_df['significant'].sum()
        print(f"\n  Tests: {len(ca_df)}, FDR survivors: {n_sig}")

        top = ca_df.nlargest(10, 'abs_rho')
        print(f"\n  Top 10 (age × 5-year Δenrichment):")
        print(f"  {'Feature':<30} {'rho':>7} {'p':>10} {'p_FDR':>8} {'N':>5} {'Sig'}")
        print(f"  {'-'*65}")
        for _, r in top.iterrows():
            sig = '*' if r['p_fdr'] < 0.05 else ''
            print(f"  {r['feature']:<30} {r['rho']:>+.3f} {r['p']:>10.4f} {r['p_fdr']:>8.4f} {r['n']:>5d} {sig}")

    # Save
    rdf.to_csv('outputs/dortmund_longitudinal_icc.csv', index=False)
    print(f"\nSaved to outputs/dortmund_longitudinal_icc.csv")


if __name__ == '__main__':
    main()
