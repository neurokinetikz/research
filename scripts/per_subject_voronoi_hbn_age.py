#!/usr/bin/env python3
"""
Per-Subject Voronoi Enrichment × Age/Sex/Psychopathology (HBN)
===============================================================

Computes per-subject enrichment at each position × band, then
correlates with age (continuous 5-21), sex, and psychopathology
(p_factor, attention, internalizing, externalizing) across all
HBN releases.

Usage:
    python scripts/per_subject_voronoi_hbn_age.py
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
    ('inv_noble_3', round(1-PHI_INV**3, 6)), ('inv_noble_4', round(1-PHI_INV**4, 6)),
    ('inv_noble_5', round(1-PHI_INV**5, 6)), ('inv_noble_6', round(1-PHI_INV**6, 6)),
]
POS_NAMES = [p[0] for p in POS_LIST]
POS_VALS = np.array([p[1] for p in POS_LIST])
N_POS = len(POS_VALS)

VORONOI_BINS = []
for i in range(N_POS):
    d_prev = (POS_VALS[i] - POS_VALS[(i-1) % N_POS]) % 1.0
    d_next = (POS_VALS[(i+1) % N_POS] - POS_VALS[i]) % 1.0
    VORONOI_BINS.append(d_prev/2 + d_next/2)

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
    # Load demographics from all releases
    demo_rows = []
    for release in RELEASES:
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if not os.path.exists(tsv):
            continue
        df = pd.read_csv(tsv, sep='\t')
        df['release'] = release
        demo_rows.append(df)
    demo = pd.concat(demo_rows, ignore_index=True)
    print(f"Demographics loaded: {len(demo)} subjects across {demo['release'].nunique()} releases")
    print(f"Age range: {demo['age'].min():.1f} - {demo['age'].max():.1f}")

    # Load per-subject peaks and compute enrichment
    rows = []
    for release in RELEASES:
        peak_dir = f'exports_adaptive/hbn_{release}'
        peak_files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
        for f in peak_files:
            sub_id = os.path.basename(f).replace('_peaks.csv', '')
            peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
            enrich = per_subject_enrichment(peaks, min_peaks=30)
            enrich['subject'] = sub_id
            enrich['release'] = release

            # Match demographics
            match = demo[demo['participant_id'] == sub_id]
            if len(match) > 0:
                m = match.iloc[0]
                enrich['age'] = m['age']
                enrich['sex'] = m['sex']
                enrich['ehq_total'] = m.get('ehq_total', np.nan)
                for psy in ['p_factor', 'attention', 'internalizing', 'externalizing']:
                    enrich[psy] = pd.to_numeric(m.get(psy, np.nan), errors='coerce')
            rows.append(enrich)

    df = pd.DataFrame(rows)
    n_with_age = df['age'].notna().sum()
    print(f"Per-subject enrichment: {len(df)} subjects, {n_with_age} with age data")

    enrich_cols = [c for c in df.columns if any(c.startswith(b + '_') for b in BAND_ORDER)
                   and not c.endswith('_n_peaks')]

    # === AGE CORRELATIONS ===
    print(f"\n{'='*80}")
    print(f"  AGE × PER-BAND ENRICHMENT (N={n_with_age}, ages {df['age'].min():.1f}-{df['age'].max():.1f})")
    print(f"{'='*80}")

    age_results = []
    for col in enrich_cols:
        valid = df[['age', col]].dropna()
        if len(valid) < 50:
            continue
        rho, p = stats.spearmanr(valid['age'], valid[col])
        age_results.append({'feature': col, 'rho': rho, 'p': p, 'n': len(valid),
                           'abs_rho': abs(rho)})

    age_df = pd.DataFrame(age_results)
    from statsmodels.stats.multitest import multipletests
    reject, p_fdr, _, _ = multipletests(age_df['p'].values, method='fdr_bh', alpha=0.05)
    age_df['p_fdr'] = p_fdr
    age_df['significant'] = reject

    print(f"\n  Total tests: {len(age_df)}")
    print(f"  FDR survivors: {age_df['significant'].sum()}")

    top = age_df.nlargest(20, 'abs_rho')
    print(f"\n  Top 20 by |rho|:")
    print(f"  {'Feature':<35} {'rho':>6} {'p':>10} {'p_FDR':>10} {'N':>5} {'Sig'}")
    print(f"  {'-'*72}")
    for _, r in top.iterrows():
        sig = '***' if r['p_fdr'] < 0.001 else ('**' if r['p_fdr'] < 0.01 else ('*' if r['p_fdr'] < 0.05 else ''))
        print(f"  {r['feature']:<35} {r['rho']:>+.3f} {r['p']:>10.2e} {r['p_fdr']:>10.4f} {r['n']:>5d} {sig}")

    # Per-band summary
    print(f"\n  Per-band age summary:")
    for band in BAND_ORDER:
        band_df = age_df[age_df['feature'].str.startswith(band + '_')]
        if len(band_df) == 0:
            continue
        n_sig = band_df['significant'].sum()
        best = band_df.loc[band_df['abs_rho'].idxmax()]
        print(f"    {band:<12}: {n_sig}/{len(band_df)} FDR sig, best |rho|={best['abs_rho']:.3f} ({best['feature']})")

    # === PSYCHOPATHOLOGY ===
    psy_vars = ['p_factor', 'attention', 'internalizing', 'externalizing']
    psy_available = [p for p in psy_vars if df[p].notna().sum() > 50]

    if psy_available:
        print(f"\n\n{'='*80}")
        print(f"  PSYCHOPATHOLOGY × PER-BAND ENRICHMENT")
        print(f"{'='*80}")

        for psy in psy_available:
            n_valid = df[psy].notna().sum()
            print(f"\n  {psy} (N={n_valid}):")

            psy_results = []
            for col in enrich_cols:
                valid = df[[psy, col]].dropna()
                if len(valid) < 50:
                    continue
                rho, p = stats.spearmanr(valid[psy], valid[col])
                psy_results.append({'feature': col, 'rho': rho, 'p': p, 'n': len(valid),
                                   'abs_rho': abs(rho)})

            if not psy_results:
                print(f"    Insufficient data")
                continue

            psy_df = pd.DataFrame(psy_results)
            rej, pfdr, _, _ = multipletests(psy_df['p'].values, method='fdr_bh', alpha=0.05)
            psy_df['p_fdr'] = pfdr
            psy_df['significant'] = rej

            n_sig = psy_df['significant'].sum()
            print(f"    Tests: {len(psy_df)}, FDR survivors: {n_sig}")

            if n_sig > 0 or True:  # show top regardless
                top_psy = psy_df.nlargest(5, 'abs_rho')
                for _, r in top_psy.iterrows():
                    sig = '*' if r['p_fdr'] < 0.05 else ''
                    print(f"      {r['feature']:<35} rho={r['rho']:>+.3f} p={r['p']:.2e} p_FDR={r['p_fdr']:.4f} {sig}")

    # === SEX DIFFERENCES ===
    print(f"\n\n{'='*80}")
    print(f"  SEX DIFFERENCES")
    print(f"{'='*80}")

    males = df[df['sex'] == 'M']
    females = df[df['sex'] == 'F']
    print(f"\n  Males: {len(males)}, Females: {len(females)}")

    sex_results = []
    for col in enrich_cols:
        m_vals = males[col].dropna()
        f_vals = females[col].dropna()
        if len(m_vals) < 30 or len(f_vals) < 30:
            continue
        u_stat, p = stats.mannwhitneyu(m_vals, f_vals, alternative='two-sided')
        d = (m_vals.mean() - f_vals.mean()) / np.sqrt((m_vals.std()**2 + f_vals.std()**2) / 2)
        sex_results.append({'feature': col, 'd': d, 'p': p, 'n_m': len(m_vals),
                           'n_f': len(f_vals), 'abs_d': abs(d)})

    sex_df = pd.DataFrame(sex_results)
    if len(sex_df) > 0:
        rej, pfdr, _, _ = multipletests(sex_df['p'].values, method='fdr_bh', alpha=0.05)
        sex_df['p_fdr'] = pfdr
        sex_df['significant'] = rej
        n_sig = sex_df['significant'].sum()
        print(f"  Tests: {len(sex_df)}, FDR survivors: {n_sig}")

        top_sex = sex_df.nlargest(10, 'abs_d')
        print(f"\n  Top 10 by |d|:")
        for _, r in top_sex.iterrows():
            sig = '*' if r['p_fdr'] < 0.05 else ''
            print(f"    {r['feature']:<35} d={r['d']:>+.3f} p={r['p']:.2e} p_FDR={r['p_fdr']:.4f} {sig}")

    # Save
    age_df.to_csv('outputs/hbn_age_enrichment.csv', index=False)
    print(f"\nAge results saved to outputs/hbn_age_enrichment.csv")


if __name__ == '__main__':
    main()
