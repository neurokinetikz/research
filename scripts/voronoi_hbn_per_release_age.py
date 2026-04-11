#!/usr/bin/env python3
"""
HBN Per-Release Age Trajectory Replication
============================================

Tests whether the 43 developmental age correlations replicate
independently within each HBN release (R1=136, R2=150, R3=184,
R4=322, R6=135).

Usage:
    python scripts/voronoi_hbn_per_release_age.py
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
RELEASES = ['R1', 'R2', 'R3', 'R4', 'R6']


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


def main():
    all_release_results = {}

    for release in RELEASES:
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if not os.path.exists(tsv):
            continue
        demo = pd.read_csv(tsv, sep='\t')
        demo_dict = {r['participant_id']: r['age'] for _, r in demo.iterrows()}

        peak_dir = f'exports_adaptive/hbn_{release}'
        rows = []
        for f in sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv'))):
            sub_id = os.path.basename(f).replace('_peaks.csv', '')
            peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
            enrich = per_subject_enrichment(peaks)
            enrich['subject'] = sub_id
            enrich['age'] = demo_dict.get(sub_id, np.nan)
            rows.append(enrich)

        df = pd.DataFrame(rows)
        n_valid = df['age'].notna().sum()

        enrich_cols = [c for c in df.columns
                      if any(c.startswith(b + '_') for b in BAND_ORDER)
                      and not c.endswith('_n_peaks')]

        age_results = []
        for col in enrich_cols:
            valid = df[['age', col]].dropna()
            if len(valid) < 20:
                continue
            rho, p = stats.spearmanr(valid['age'], valid[col])
            age_results.append({'feature': col, 'rho': rho, 'p': p,
                               'n': len(valid), 'abs_rho': abs(rho)})

        adf = pd.DataFrame(age_results)
        if len(adf) > 0:
            rej, pfdr, _, _ = multipletests(adf['p'].values, method='fdr_bh', alpha=0.05)
            adf['p_fdr'] = pfdr
            adf['significant'] = rej
            all_release_results[release] = adf

            n_sig = adf['significant'].sum()
            print(f"{release}: N={n_valid}, ages {df['age'].min():.1f}-{df['age'].max():.1f}, "
                  f"tests={len(adf)}, FDR sig={n_sig}")

    # Cross-release correlation
    print(f"\nPairwise correlation of age rhos:")
    rels = list(all_release_results.keys())
    for i, r1 in enumerate(rels):
        for r2 in rels[i + 1:]:
            m = pd.merge(all_release_results[r1][['feature', 'rho']],
                        all_release_results[r2][['feature', 'rho']],
                        on='feature', suffixes=(f'_{r1}', f'_{r2}'))
            if len(m) > 5:
                r, p = stats.pearsonr(m[f'rho_{r1}'], m[f'rho_{r2}'])
                print(f"  {r1} vs {r2}: r={r:.3f} (p={p:.4f})")

    # Features replicated across releases
    all_feats = set(all_release_results[rels[0]]['feature'])
    for rel in rels[1:]:
        all_feats &= set(all_release_results[rel]['feature'])

    print(f"\nFeatures sig in ≥3/5 releases, same direction:")
    for feat in sorted(all_feats):
        rhos = []
        n_sig_count = 0
        for rel in rels:
            row = all_release_results[rel][all_release_results[rel]['feature'] == feat]
            if len(row) == 0:
                break
            rhos.append(row.iloc[0]['rho'])
            if row.iloc[0]['significant']:
                n_sig_count += 1

        if len(rhos) == len(rels) and n_sig_count >= 3:
            if all(r > 0 for r in rhos) or all(r < 0 for r in rhos):
                rho_str = '/'.join(f'{r:+.2f}' for r in rhos)
                print(f"  {feat:<30} {rho_str}  sig:{n_sig_count}/5")

    # Per-band summary
    print(f"\nPer-band replication (≥3/5 releases, same direction):")
    for band in BAND_ORDER:
        count = 0
        for feat in sorted(all_feats):
            if not feat.startswith(band + '_'):
                continue
            rhos = []
            n_sig_count = 0
            for rel in rels:
                row = all_release_results[rel][all_release_results[rel]['feature'] == feat]
                if len(row) == 0:
                    break
                rhos.append(row.iloc[0]['rho'])
                if row.iloc[0]['significant']:
                    n_sig_count += 1
            if len(rhos) == len(rels) and n_sig_count >= 3:
                if all(r > 0 for r in rhos) or all(r < 0 for r in rhos):
                    count += 1
        print(f"  {band:<12}: {count}")


if __name__ == '__main__':
    main()
