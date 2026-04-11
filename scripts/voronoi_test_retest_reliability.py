#!/usr/bin/env python3
"""
Per-Subject Voronoi Enrichment Test-Retest Reliability
=======================================================

Computes within-subject reliability of per-subject enrichment using
matched condition pairs:
  - Dortmund EC-pre vs EC-post (same session, ~2 hours apart)
  - Dortmund EO-pre vs EO-post (same session, ~2 hours apart)
  - Dortmund EC-pre vs EO-pre (different condition, same timepoint)

Reports Pearson r, ICC(2,1), and Cohen's d for each enrichment feature.

Usage:
    python scripts/voronoi_test_retest_reliability.py
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
    return results


def icc_21(x, y):
    """Compute ICC(2,1) for two matched measurements."""
    n = len(x)
    if n < 3:
        return np.nan
    grand_mean = (np.mean(x) + np.mean(y)) / 2
    # Between-subjects variance
    subject_means = (x + y) / 2
    ms_between = np.sum((subject_means - grand_mean) ** 2) / (n - 1) * 2
    # Within-subjects variance
    diffs = x - y
    ms_within = np.sum(diffs ** 2) / (2 * n)
    # Measurement variance (systematic)
    ms_measure = n * (np.mean(x) - np.mean(y)) ** 2
    # ICC(2,1)
    if (ms_between + ms_within) == 0:
        return np.nan
    icc = (ms_between - ms_within) / (ms_between + ms_within)
    return icc


def compute_reliability(dir_a, dir_b, label):
    """Compute test-retest reliability between two condition directories."""
    files_a = {os.path.basename(f).replace('_peaks.csv', ''): f
               for f in glob.glob(os.path.join(dir_a, '*_peaks.csv'))}
    files_b = {os.path.basename(f).replace('_peaks.csv', ''): f
               for f in glob.glob(os.path.join(dir_b, '*_peaks.csv'))}
    matched = sorted(set(files_a.keys()) & set(files_b.keys()))
    print(f"\n  {label}: {len(matched)} matched subjects")

    # Compute per-subject enrichment for both conditions
    data_a = {}
    data_b = {}
    for sub_id in matched:
        peaks_a = pd.read_csv(files_a[sub_id], usecols=['freq', 'phi_octave'])
        peaks_b = pd.read_csv(files_b[sub_id], usecols=['freq', 'phi_octave'])
        data_a[sub_id] = per_subject_enrichment(peaks_a)
        data_b[sub_id] = per_subject_enrichment(peaks_b)

    # Get all enrichment features
    all_features = set()
    for d in list(data_a.values()) + list(data_b.values()):
        all_features.update(k for k in d.keys() if not k.endswith('_n_peaks'))

    results = []
    for feat in sorted(all_features):
        vals_a = []
        vals_b = []
        for sub_id in matched:
            va = data_a[sub_id].get(feat, np.nan)
            vb = data_b[sub_id].get(feat, np.nan)
            if not np.isnan(va) and not np.isnan(vb):
                vals_a.append(va)
                vals_b.append(vb)

        if len(vals_a) < 20:
            continue

        x = np.array(vals_a)
        y = np.array(vals_b)

        r, p = stats.pearsonr(x, y)
        rho, p_s = stats.spearmanr(x, y)
        icc = icc_21(x, y)
        mean_diff = np.mean(y - x)
        sd_diff = np.std(y - x, ddof=1)
        pooled_sd = np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)
        d = mean_diff / pooled_sd if pooled_sd > 0 else 0

        results.append({
            'feature': feat,
            'n': len(vals_a),
            'pearson_r': r,
            'spearman_rho': rho,
            'icc': icc,
            'mean_diff': mean_diff,
            'cohen_d': d,
        })

    rdf = pd.DataFrame(results)

    # Summary
    print(f"\n  {'Feature':<30} {'N':>5} {'r':>6} {'rho':>6} {'ICC':>6} {'d':>6}")
    print(f"  {'-'*65}")

    for band in BAND_ORDER:
        band_df = rdf[rdf['feature'].str.startswith(band + '_')]
        if len(band_df) == 0:
            continue
        for _, row in band_df.sort_values('feature').iterrows():
            print(f"  {row['feature']:<30} {row['n']:>5} {row['pearson_r']:>+.3f}"
                  f" {row['spearman_rho']:>+.3f} {row['icc']:>+.3f} {row['cohen_d']:>+.3f}")

    # Per-band summary
    print(f"\n  Per-band ICC summary:")
    for band in BAND_ORDER:
        band_df = rdf[rdf['feature'].str.startswith(band + '_')]
        if len(band_df) == 0:
            continue
        print(f"    {band:<12}: median ICC = {band_df['icc'].median():+.3f}"
              f"  range [{band_df['icc'].min():+.3f}, {band_df['icc'].max():+.3f}]"
              f"  median r = {band_df['pearson_r'].median():+.3f}")

    # Overall
    print(f"\n  Overall: median ICC = {rdf['icc'].median():+.3f}"
          f"  median r = {rdf['pearson_r'].median():+.3f}"
          f"  median |d| = {rdf['cohen_d'].abs().median():.3f}")

    return rdf


def main():
    pairs = [
        ('exports_adaptive/dortmund', 'exports_adaptive/dortmund_EC_post',
         'EC-pre vs EC-post (same-condition, ~2hr apart)'),
        ('exports_adaptive/dortmund_EO_pre', 'exports_adaptive/dortmund_EO_post',
         'EO-pre vs EO-post (same-condition, ~2hr apart)'),
        ('exports_adaptive/dortmund', 'exports_adaptive/dortmund_EO_pre',
         'EC-pre vs EO-pre (different-condition, same timepoint)'),
    ]

    for dir_a, dir_b, label in pairs:
        if not os.path.exists(dir_a) or not os.path.exists(dir_b):
            print(f"\n  SKIP {label}: missing directory")
            continue
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        rdf = compute_reliability(dir_a, dir_b, label)


if __name__ == '__main__':
    main()
