#!/usr/bin/env python3
"""
Sex × Age Interaction in Enrichment Trajectories
==================================================

Tests whether the developmental (HBN) and aging (Dortmund) enrichment
trajectories differ between males and females.

Usage:
    python scripts/voronoi_sex_age_interaction.py
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


def load_hbn():
    """Load HBN per-subject enrichment with age and sex."""
    rows = []
    for release in ['R1', 'R2', 'R3', 'R4', 'R6']:
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if not os.path.exists(tsv):
            continue
        demo = pd.read_csv(tsv, sep='\t')
        demo_dict = {r['participant_id']: {'age': r['age'], 'sex': r['sex']}
                     for _, r in demo.iterrows()}

        peak_dir = f'exports_adaptive/hbn_{release}'
        for f in sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv'))):
            sub_id = os.path.basename(f).replace('_peaks.csv', '')
            peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
            enrich = per_subject_enrichment(peaks)
            enrich['subject'] = sub_id
            if sub_id in demo_dict:
                enrich['age'] = demo_dict[sub_id]['age']
                enrich['sex'] = demo_dict[sub_id]['sex']
            rows.append(enrich)

    return pd.DataFrame(rows)


def load_dortmund():
    """Load Dortmund per-subject enrichment with age and sex."""
    demo = pd.read_csv('/Volumes/T9/dortmund_data/participants.tsv', sep='\t')
    demo_dict = {r['participant_id']: {'age': r['age'], 'sex': r['sex']}
                 for _, r in demo.iterrows()}

    rows = []
    for f in sorted(glob.glob('exports_adaptive/dortmund/*_peaks.csv')):
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
        enrich = per_subject_enrichment(peaks)
        enrich['subject'] = sub_id
        if sub_id in demo_dict:
            enrich['age'] = demo_dict[sub_id]['age']
            enrich['sex'] = demo_dict[sub_id]['sex']
        rows.append(enrich)

    return pd.DataFrame(rows)


def analyze_sex_age(df, label):
    """Test sex × age interaction for each enrichment feature."""
    enrich_cols = [c for c in df.columns if any(c.startswith(b + '_') for b in BAND_ORDER)
                   and not c.endswith('_n_peaks')]

    df_valid = df[df['age'].notna() & df['sex'].notna()].copy()
    males = df_valid[df_valid['sex'] == 'M']
    females = df_valid[df_valid['sex'] == 'F']
    print(f"\n  {label}: {len(males)} M, {len(females)} F")
    print(f"  Age range M: {males['age'].min():.1f}-{males['age'].max():.1f} (mean {males['age'].mean():.1f})")
    print(f"  Age range F: {females['age'].min():.1f}-{females['age'].max():.1f} (mean {females['age'].mean():.1f})")

    results = []
    for feat in enrich_cols:
        m_valid = males[['age', feat]].dropna()
        f_valid = females[['age', feat]].dropna()
        if len(m_valid) < 20 or len(f_valid) < 20:
            continue

        rho_m, p_m = stats.spearmanr(m_valid['age'], m_valid[feat])
        rho_f, p_f = stats.spearmanr(f_valid['age'], f_valid[feat])

        # Fisher z-test for difference between correlations
        z_m = np.arctanh(rho_m)
        z_f = np.arctanh(rho_f)
        se = np.sqrt(1 / (len(m_valid) - 3) + 1 / (len(f_valid) - 3))
        z_diff = (z_m - z_f) / se
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

        results.append({
            'feature': feat,
            'rho_M': rho_m, 'p_M': p_m, 'n_M': len(m_valid),
            'rho_F': rho_f, 'p_F': p_f, 'n_F': len(f_valid),
            'delta_rho': rho_m - rho_f,
            'z_diff': z_diff, 'p_diff': p_diff,
            'abs_delta': abs(rho_m - rho_f),
        })

    rdf = pd.DataFrame(results)
    if len(rdf) == 0:
        return None

    reject, pfdr, _, _ = multipletests(rdf['p_diff'].values, method='fdr_bh', alpha=0.05)
    rdf['p_fdr'] = pfdr
    rdf['significant'] = reject

    n_sig = rdf['significant'].sum()
    print(f"\n  Sex × age interaction tests: {len(rdf)}")
    print(f"  FDR survivors: {n_sig}")

    # Top by interaction strength
    top = rdf.nlargest(15, 'abs_delta')
    print(f"\n  Top 15 by |Δrho| (M-F difference in age trajectory):")
    print(f"  {'Feature':<30} {'rho_M':>6} {'rho_F':>6} {'Δrho':>6} {'p_diff':>10} {'p_FDR':>8} {'Sig'}")
    print(f"  {'-'*75}")
    for _, r in top.iterrows():
        sig = '*' if r['p_fdr'] < 0.05 else ''
        print(f"  {r['feature']:<30} {r['rho_M']:>+.3f} {r['rho_F']:>+.3f} {r['delta_rho']:>+.3f}"
              f" {r['p_diff']:>10.4f} {r['p_fdr']:>8.4f} {sig}")

    # Per-band summary
    print(f"\n  Per-band interaction summary:")
    for band in BAND_ORDER:
        band_df = rdf[rdf['feature'].str.startswith(band + '_')]
        if len(band_df) == 0:
            continue
        n_sig_band = band_df['significant'].sum()
        best = band_df.loc[band_df['abs_delta'].idxmax()]
        print(f"    {band:<12}: {n_sig_band} FDR sig, max |Δrho|={best['abs_delta']:.3f}"
              f" ({best['feature']}: M={best['rho_M']:+.3f}, F={best['rho_F']:>+.3f})")

    # Show features where M and F trajectories go in OPPOSITE directions
    opposite = rdf[(rdf['rho_M'] * rdf['rho_F'] < 0) & (rdf['abs_delta'] > 0.1)]
    if len(opposite) > 0:
        print(f"\n  Features with OPPOSITE sex trajectories (M↑ F↓ or M↓ F↑):")
        for _, r in opposite.sort_values('abs_delta', ascending=False).iterrows():
            sig = '*' if r['p_fdr'] < 0.05 else ''
            print(f"    {r['feature']:<30} M={r['rho_M']:>+.3f} F={r['rho_F']:>+.3f} {sig}")

    return rdf


def main():
    print(f"{'='*80}")
    print(f"  SEX × AGE INTERACTION IN ENRICHMENT TRAJECTORIES")
    print(f"{'='*80}")

    # HBN (development)
    print(f"\n{'='*80}")
    print(f"  HBN (ages 5-21, DEVELOPMENT)")
    print(f"{'='*80}")
    hbn = load_hbn()
    hbn_results = analyze_sex_age(hbn, 'HBN')

    # Dortmund (aging)
    print(f"\n\n{'='*80}")
    print(f"  DORTMUND (ages 20-70, AGING)")
    print(f"{'='*80}")
    dort = load_dortmund()
    dort_results = analyze_sex_age(dort, 'Dortmund')

    # Cross-dataset comparison
    if hbn_results is not None and dort_results is not None:
        print(f"\n\n{'='*80}")
        print(f"  CROSS-DATASET: Do the same features show sex differences in both?")
        print(f"{'='*80}")

        merged = pd.merge(
            hbn_results[['feature', 'delta_rho', 'significant']].rename(
                columns={'delta_rho': 'delta_hbn', 'significant': 'sig_hbn'}),
            dort_results[['feature', 'delta_rho', 'significant']].rename(
                columns={'delta_rho': 'delta_dort', 'significant': 'sig_dort'}),
            on='feature')

        r, p = stats.pearsonr(merged['delta_hbn'], merged['delta_dort'])
        print(f"\n  Correlation of sex×age interactions: r = {r:.3f} (p = {p:.4f})")
        same_dir = ((merged['delta_hbn'] > 0) == (merged['delta_dort'] > 0)).sum()
        print(f"  Same direction: {same_dir}/{len(merged)} ({100*same_dir/len(merged):.0f}%)")

        # Features with large sex differences in BOTH datasets
        both_large = merged[(merged['delta_hbn'].abs() > 0.08) & (merged['delta_dort'].abs() > 0.08)]
        if len(both_large) > 0:
            print(f"\n  Features with |Δrho| > 0.08 in BOTH datasets:")
            for _, r in both_large.sort_values('delta_hbn', key=abs, ascending=False).iterrows():
                same = '✓' if (r['delta_hbn'] > 0) == (r['delta_dort'] > 0) else '✗'
                print(f"    {r['feature']:<30} HBN:{r['delta_hbn']:>+.3f} Dort:{r['delta_dort']:>+.3f} {same}")


if __name__ == '__main__':
    main()
