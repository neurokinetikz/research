#!/usr/bin/env python3
"""
State Sensitivity × Age
========================

Tests whether the magnitude or direction of EC→EO enrichment change
correlates with age. Uses matched within-subject EC/EO pairs.

Usage:
    python scripts/voronoi_state_sensitivity_age.py
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
    return results


def analyze_dataset(ec_dir, eo_dir, demo, label):
    ec_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                for f in glob.glob(os.path.join(ec_dir, '*_peaks.csv'))}
    eo_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                for f in glob.glob(os.path.join(eo_dir, '*_peaks.csv'))}
    matched = sorted(set(ec_files.keys()) & set(eo_files.keys()))

    rows = []
    for sub_id in matched:
        ec = per_subject_enrichment(pd.read_csv(ec_files[sub_id], usecols=['freq', 'phi_octave']))
        eo = per_subject_enrichment(pd.read_csv(eo_files[sub_id], usecols=['freq', 'phi_octave']))
        delta = {'subject': sub_id, 'age': demo.get(sub_id, np.nan)}
        for k in set(list(ec.keys()) + list(eo.keys())):
            ec_v, eo_v = ec.get(k, np.nan), eo.get(k, np.nan)
            if not np.isnan(ec_v) and not np.isnan(eo_v):
                delta[f'delta_{k}'] = eo_v - ec_v
                delta[f'abs_delta_{k}'] = abs(eo_v - ec_v)
        rows.append(delta)

    df = pd.DataFrame(rows)
    n = df['age'].notna().sum()
    print(f"\n  {label}: {len(matched)} matched, {n} with age, range {df['age'].min():.1f}-{df['age'].max():.1f}")

    for analysis, prefix in [('Direction', 'delta_'), ('Magnitude', 'abs_delta_')]:
        cols = [c for c in df.columns if c.startswith(prefix) and not c.startswith('abs_delta_') if prefix == 'delta_']\
               if prefix == 'delta_' else [c for c in df.columns if c.startswith(prefix)]
        if prefix == 'delta_':
            cols = [c for c in df.columns if c.startswith('delta_') and not c.startswith('abs_')]

        results = []
        for col in cols:
            valid = df[['age', col]].dropna()
            if len(valid) < 20:
                continue
            rho, p = stats.spearmanr(valid['age'], valid[col])
            results.append({'feature': col.replace(prefix, ''), 'rho': rho, 'p': p,
                           'n': len(valid), 'abs_rho': abs(rho)})
        rdf = pd.DataFrame(results)
        if len(rdf) == 0:
            continue
        rej, pfdr, _, _ = multipletests(rdf['p'].values, method='fdr_bh', alpha=0.05)
        rdf['p_fdr'] = pfdr
        rdf['significant'] = rej
        n_sig = rdf['significant'].sum()
        print(f"\n  {analysis} × age: {len(rdf)} tests, {n_sig} FDR survivors")
        top = rdf.nlargest(5, 'abs_rho')
        for _, r in top.iterrows():
            sig = '*' if r['p_fdr'] < 0.05 else ''
            print(f"    {r['feature']:<30} rho={r['rho']:>+.3f} p_FDR={r['p_fdr']:.4f} {sig}")


def main():
    # LEMON
    meta = pd.read_csv('/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/'
                       'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
    age_map = {'20-25': 22.5, '25-30': 27.5, '30-35': 32.5, '35-40': 37.5,
               '55-60': 57.5, '60-65': 62.5, '65-70': 67.5, '70-75': 72.5, '75-80': 77.5}
    lemon_demo = {row['ID']: age_map.get(row['Age'], np.nan) for _, row in meta.iterrows()
                  if pd.notna(age_map.get(row['Age']))}
    analyze_dataset('exports_adaptive/lemon', 'exports_adaptive/lemon_EO', lemon_demo, 'LEMON')

    # Dortmund
    dort_df = pd.read_csv('/Volumes/T9/dortmund_data/participants.tsv', sep='\t')
    dort_demo = {row['participant_id']: row['age'] for _, row in dort_df.iterrows()}
    analyze_dataset('exports_adaptive/dortmund', 'exports_adaptive/dortmund_EO_pre', dort_demo, 'Dortmund')


if __name__ == '__main__':
    main()
