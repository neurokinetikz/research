#!/usr/bin/env python3
"""
Cross-Band Coupling Signatures
================================

Tests whether per-subject enrichment metrics correlate across bands.
Key question: do subjects with stronger alpha mountain also have
deeper beta-low U-shape?

Uses per-subject enrichment from LEMON (EC), Dortmund (EC-pre),
and HBN (RestingState) independently to test replication.

Usage:
    python scripts/voronoi_cross_band_coupling.py
"""

import os
import sys
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

# Summary metrics per band
SUMMARY_METRICS = {
    'theta_boundary': 'Theta boundary enrichment (f₀ convergence)',
    'theta_mountain': 'Theta Noble1-boundary gap',
    'alpha_mountain': 'Alpha mountain height (Noble1 - boundary)',
    'alpha_noble_1': 'Alpha Noble1 enrichment',
    'alpha_attractor': 'Alpha attractor enrichment',
    'alpha_boundary': 'Alpha boundary depletion',
    'beta_low_ushape': 'Beta-low U-shape depth',
    'beta_low_boundary': 'Beta-low boundary enrichment',
    'beta_low_attractor': 'Beta-low attractor depletion',
    'beta_low_mountain': 'Beta-low Noble1-boundary gap',
    'beta_high_mountain': 'Beta-high Noble1-boundary gap',
    'beta_high_inv_noble_4': 'Beta-high inv_noble_4 enrichment',
    'gamma_mountain': 'Gamma Noble1-boundary gap',
    'gamma_inv_noble_3': 'Gamma inv_noble_3 (ramp start)',
    'gamma_inv_noble_4': 'Gamma inv_noble_4 (ramp middle)',
    'gamma_inv_noble_5': 'Gamma inv_noble_5 (ramp peak)',
}


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


def load_dataset(peak_dirs, label):
    """Load per-subject enrichment from one or more peak directories."""
    rows = []
    for peak_dir in peak_dirs:
        if not os.path.exists(peak_dir):
            continue
        for f in sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv'))):
            sub_id = os.path.basename(f).replace('_peaks.csv', '')
            peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
            enrich = per_subject_enrichment(peaks)
            enrich['subject'] = sub_id
            rows.append(enrich)

    df = pd.DataFrame(rows)
    print(f"  {label}: {len(df)} subjects")
    return df


def analyze_cross_band(df, label):
    """Correlate summary metrics across bands."""
    metrics = [m for m in SUMMARY_METRICS.keys() if m in df.columns]
    print(f"\n  Available summary metrics: {len(metrics)}")

    # All pairwise correlations between DIFFERENT bands
    results = []
    for i, m1 in enumerate(metrics):
        band1 = m1.split('_')[0] if not m1.startswith('beta') else '_'.join(m1.split('_')[:2])
        for m2 in metrics[i + 1:]:
            band2 = m2.split('_')[0] if not m2.startswith('beta') else '_'.join(m2.split('_')[:2])
            if band1 == band2:
                continue  # skip within-band

            valid = df[[m1, m2]].dropna()
            if len(valid) < 30:
                continue
            rho, p = stats.spearmanr(valid[m1], valid[m2])
            results.append({
                'metric_1': m1, 'metric_2': m2,
                'band_1': band1, 'band_2': band2,
                'rho': rho, 'p': p, 'n': len(valid),
                'abs_rho': abs(rho),
            })

    rdf = pd.DataFrame(results)
    if len(rdf) == 0:
        print("  No valid cross-band pairs")
        return None

    reject, pfdr, _, _ = multipletests(rdf['p'].values, method='fdr_bh', alpha=0.05)
    rdf['p_fdr'] = pfdr
    rdf['significant'] = reject

    n_sig = rdf['significant'].sum()
    print(f"  Cross-band pairs: {len(rdf)}, FDR survivors: {n_sig}")

    # Top results
    top = rdf.nlargest(20, 'abs_rho')
    print(f"\n  Top 20 cross-band correlations:")
    print(f"  {'Metric 1':<25} {'Metric 2':<25} {'rho':>6} {'p':>10} {'p_FDR':>8} {'N':>5} {'Sig'}")
    print(f"  {'-'*85}")
    for _, r in top.iterrows():
        sig = '***' if r['p_fdr'] < 0.001 else ('**' if r['p_fdr'] < 0.01 else ('*' if r['p_fdr'] < 0.05 else ''))
        print(f"  {r['metric_1']:<25} {r['metric_2']:<25} {r['rho']:>+.3f} {r['p']:>10.2e} {r['p_fdr']:>8.4f} {r['n']:>5d} {sig}")

    # Key hypothesis tests
    print(f"\n  KEY HYPOTHESES:")

    hyps = [
        ('alpha_mountain', 'beta_low_ushape', 'Taller alpha mountain ↔ deeper beta-low U-shape?'),
        ('alpha_noble_1', 'beta_low_boundary', 'Higher alpha Noble1 ↔ stronger beta-low boundary?'),
        ('alpha_mountain', 'gamma_inv_noble_4', 'Taller alpha mountain ↔ stronger gamma ramp?'),
        ('beta_low_ushape', 'gamma_inv_noble_4', 'Deeper beta-low U-shape ↔ stronger gamma ramp?'),
        ('alpha_boundary', 'beta_low_boundary', 'Alpha boundary depletion ↔ beta-low boundary enrichment?'),
        ('theta_boundary', 'alpha_boundary', 'Theta boundary ↔ alpha boundary?'),
        ('theta_boundary', 'beta_low_boundary', 'Theta boundary ↔ beta-low boundary?'),
        ('alpha_mountain', 'beta_high_mountain', 'Alpha mountain ↔ beta-high mountain?'),
    ]

    for m1, m2, question in hyps:
        row = rdf[(rdf['metric_1'] == m1) & (rdf['metric_2'] == m2) |
                  (rdf['metric_1'] == m2) & (rdf['metric_2'] == m1)]
        if len(row) == 0:
            print(f"    {question}")
            print(f"      Not available (insufficient data)")
            continue
        r = row.iloc[0]
        sig = '*' if r['p_fdr'] < 0.05 else ''
        print(f"    {question}")
        print(f"      rho = {r['rho']:+.3f}, p = {r['p']:.2e}, p_FDR = {r['p_fdr']:.4f}, N = {r['n']} {sig}")

    return rdf


def main():
    datasets = {
        'LEMON EC': ['exports_adaptive/lemon'],
        'Dortmund EC-pre': ['exports_adaptive/dortmund'],
        'HBN (all releases)': [f'exports_adaptive/hbn_{r}' for r in ['R1', 'R2', 'R3', 'R4', 'R6']],
    }

    all_results = {}
    for label, dirs in datasets.items():
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")
        df = load_dataset(dirs, label)
        rdf = analyze_cross_band(df, label)
        if rdf is not None:
            all_results[label] = rdf

    # Cross-dataset replication of cross-band correlations
    if len(all_results) >= 2:
        print(f"\n\n{'='*80}")
        print(f"  CROSS-DATASET REPLICATION OF CROSS-BAND CORRELATIONS")
        print(f"{'='*80}")

        labels = list(all_results.keys())
        for i, l1 in enumerate(labels):
            for l2 in labels[i + 1:]:
                merged = pd.merge(
                    all_results[l1][['metric_1', 'metric_2', 'rho']],
                    all_results[l2][['metric_1', 'metric_2', 'rho']],
                    on=['metric_1', 'metric_2'], suffixes=(f'_{l1[:4]}', f'_{l2[:4]}'))
                if len(merged) > 3:
                    r_col1 = [c for c in merged.columns if c.startswith('rho_')][0]
                    r_col2 = [c for c in merged.columns if c.startswith('rho_')][1]
                    r, p = stats.pearsonr(merged[r_col1], merged[r_col2])
                    print(f"\n  {l1} vs {l2}: r = {r:.3f} (p = {p:.4f}), N = {len(merged)} pairs")

                    # Show replicated strong correlations
                    merged['avg_rho'] = (merged[r_col1].abs() + merged[r_col2].abs()) / 2
                    strong = merged[merged['avg_rho'] > 0.15].nlargest(10, 'avg_rho')
                    if len(strong) > 0:
                        print(f"  Top replicated (avg |rho| > 0.15):")
                        for _, row in strong.iterrows():
                            same = '✓' if (row[r_col1] > 0) == (row[r_col2] > 0) else '✗'
                            print(f"    {row['metric_1']:<25} × {row['metric_2']:<25}"
                                  f" {row[r_col1]:>+.3f} / {row[r_col2]:>+.3f} {same}")


if __name__ == '__main__':
    main()
