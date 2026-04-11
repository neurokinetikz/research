#!/usr/bin/env python3
"""
Medical/Metabolic and Handedness × Enrichment
===============================================

Tests whether medical data (BMI, blood pressure, blood biomarkers)
or handedness predict per-subject enrichment profiles.

Usage:
    python scripts/voronoi_medical_handedness.py --analysis medical
    python scripts/voronoi_medical_handedness.py --analysis handedness
    python scripts/voronoi_medical_handedness.py --analysis all
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


def run_medical():
    """LEMON medical data × enrichment."""
    MED_DIR = ('/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/'
               'Medical_LEMON')

    anthro = pd.read_csv(os.path.join(MED_DIR, 'Anthropometry/Anthropometry_LEMON.csv'))
    anthro['BMI'] = anthro['Weight_kg'] / (anthro['Height_cm'] / 100) ** 2
    anthro['WHR'] = anthro['Waist_cm'] / anthro['Hip_cm']

    bp = pd.read_csv(os.path.join(MED_DIR, 'Blood Pressure/Blood_Pressure_LEMON.csv'))
    bp['systole_avg'] = bp[['BP1_left_systole', 'BP1_right_systole']].mean(axis=1)
    bp['diastole_avg'] = bp[['BP1_left_diastole', 'BP1_right_diastole']].mean(axis=1)
    bp['pulse'] = pd.to_numeric(bp['pulse1_left'], errors='coerce')

    blood = pd.read_csv(os.path.join(MED_DIR, 'Blood Sample/Blood_Results_LEMON.csv'))
    blood_cols = [c for c in blood.columns if c != 'ID' and 'Reference' not in c
                  and 'reference' not in c and 'Date' not in c
                  and 'IND' not in c and 'INR' not in c and 'PTS' not in c and 'PTR' not in c]
    for c in blood_cols:
        blood[c] = pd.to_numeric(blood[c], errors='coerce')

    medical = {}
    for _, r in anthro.iterrows():
        medical.setdefault(r['ID'], {}).update({'BMI': r['BMI'], 'WHR': r['WHR']})
    for _, r in bp.iterrows():
        medical.setdefault(r['ID'], {}).update({
            'Systole': r['systole_avg'], 'Diastole': r['diastole_avg'], 'Pulse': r['pulse']})
    for _, r in blood.iterrows():
        medical.setdefault(r['ID'], {})
        for c in blood_cols:
            if pd.notna(r[c]):
                medical[r['ID']][c] = r[c]

    rows = []
    for f in sorted(glob.glob('exports_adaptive/lemon/*_peaks.csv')):
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
        enrich = per_subject_enrichment(peaks)
        enrich['subject'] = sub_id
        if sub_id in medical:
            for k, v in medical[sub_id].items():
                enrich[f'med_{k}'] = v
        rows.append(enrich)

    df = pd.DataFrame(rows)
    enrich_cols = [c for c in df.columns if any(c.startswith(b + '_') for b in BAND_ORDER)
                   and not c.endswith('_n_peaks')]
    med_cols = sorted([c for c in df.columns if c.startswith('med_') and df[c].notna().sum() > 50])

    print(f"\n  LEMON Medical × Enrichment: {len(df)} subjects, {len(med_cols)} medical variables")

    results = []
    for med in med_cols:
        for enrich in enrich_cols:
            valid = df[[enrich, med]].dropna()
            if len(valid) < 50:
                continue
            rho, p = stats.spearmanr(valid[enrich], valid[med])
            results.append({'medical': med.replace('med_', ''), 'enrichment': enrich,
                           'rho': rho, 'p': p, 'n': len(valid), 'abs_rho': abs(rho)})

    rdf = pd.DataFrame(results)
    rej, pfdr, _, _ = multipletests(rdf['p'].values, method='fdr_bh', alpha=0.05)
    rdf['p_fdr'] = pfdr
    rdf['significant'] = rej

    n_sig = rdf['significant'].sum()
    print(f"  Tests: {len(rdf)}, FDR survivors: {n_sig}")

    top = rdf.nlargest(10, 'abs_rho')
    print(f"\n  Top 10:")
    for _, r in top.iterrows():
        sig = '*' if r['p_fdr'] < 0.05 else ''
        print(f"    {r['medical']:<20} {r['enrichment']:<28} rho={r['rho']:>+.3f} p_FDR={r['p_fdr']:.4f} {sig}")


def run_handedness():
    """HBN + Dortmund handedness × enrichment."""
    # HBN
    print(f"\n  HBN Handedness (EHQ continuous)")
    hbn_rows = []
    for release in ['R1', 'R2', 'R3', 'R4', 'R6']:
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if not os.path.exists(tsv):
            continue
        demo = pd.read_csv(tsv, sep='\t')
        demo_dict = {r['participant_id']: pd.to_numeric(r.get('ehq_total', np.nan), errors='coerce')
                     for _, r in demo.iterrows()}
        for f in sorted(glob.glob(f'exports_adaptive/hbn_{release}/*_peaks.csv')):
            sub_id = os.path.basename(f).replace('_peaks.csv', '')
            peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
            enrich = per_subject_enrichment(peaks)
            enrich['subject'] = sub_id
            enrich['ehq'] = demo_dict.get(sub_id, np.nan)
            hbn_rows.append(enrich)

    hbn_df = pd.DataFrame(hbn_rows)
    hbn_valid = hbn_df[hbn_df['ehq'].notna()]
    enrich_cols = [c for c in hbn_df.columns if any(c.startswith(b + '_') for b in BAND_ORDER)
                   and not c.endswith('_n_peaks')]

    print(f"  N={len(hbn_valid)}, EHQ range: {hbn_valid['ehq'].min():.0f} to {hbn_valid['ehq'].max():.0f}")

    results = []
    for col in enrich_cols:
        valid = hbn_valid[['ehq', col]].dropna()
        if len(valid) < 50:
            continue
        rho, p = stats.spearmanr(valid['ehq'], valid[col])
        results.append({'feature': col, 'rho': rho, 'p': p, 'n': len(valid), 'abs_rho': abs(rho)})

    rdf = pd.DataFrame(results)
    rej, pfdr, _, _ = multipletests(rdf['p'].values, method='fdr_bh', alpha=0.05)
    rdf['p_fdr'] = pfdr
    rdf['significant'] = rej
    print(f"  Tests: {len(rdf)}, FDR survivors: {rdf['significant'].sum()}")

    # Dortmund
    print(f"\n  Dortmund Handedness (left vs right)")
    dort_demo = pd.read_csv('/Volumes/T9/dortmund_data/participants.tsv', sep='\t')
    dort_rows = []
    for f in sorted(glob.glob('exports_adaptive/dortmund/*_peaks.csv')):
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
        enrich = per_subject_enrichment(peaks)
        enrich['subject'] = sub_id
        match = dort_demo[dort_demo['participant_id'] == sub_id]
        if len(match) > 0:
            enrich['handedness'] = match.iloc[0]['handedness']
        dort_rows.append(enrich)

    dort_df = pd.DataFrame(dort_rows)
    right = dort_df[dort_df['handedness'] == 'right']
    left = dort_df[dort_df['handedness'] == 'left']
    print(f"  Right: {len(right)}, Left: {len(left)}")

    if len(left) >= 15:
        d_results = []
        for col in enrich_cols:
            l_v = left[col].dropna()
            r_v = right[col].dropna()
            if len(l_v) < 10 or len(r_v) < 50:
                continue
            _, p = stats.mannwhitneyu(l_v, r_v, alternative='two-sided')
            d = (l_v.mean() - r_v.mean()) / np.sqrt((l_v.std() ** 2 + r_v.std() ** 2) / 2)
            d_results.append({'feature': col, 'd': d, 'p': p, 'abs_d': abs(d)})

        ddf = pd.DataFrame(d_results)
        rej, pfdr, _, _ = multipletests(ddf['p'].values, method='fdr_bh', alpha=0.05)
        ddf['p_fdr'] = pfdr
        ddf['significant'] = rej
        print(f"  Tests: {len(ddf)}, FDR survivors: {ddf['significant'].sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', default='all', choices=['medical', 'handedness', 'all'])
    args = parser.parse_args()

    if args.analysis in ('medical', 'all'):
        run_medical()
    if args.analysis in ('handedness', 'all'):
        run_handedness()


if __name__ == '__main__':
    main()
