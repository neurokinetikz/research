#!/usr/bin/env python3
"""
SRM Spectral Differentiation Analysis
======================================

Complete spectral differentiation analysis for the SRM resting-state EEG dataset
(Oslo, BioSemi 64ch, 111 subjects, eyes-closed, test-retest subset N=42).

Analyses:
1. Voronoi enrichment (FOOOF + IRASA)
2. Cognitive correlations (RAVLT, digit span, TMT, Stroop, verbal fluency)
3. Age correlations
4. Test-retest reliability (ICC for ses-t1 vs ses-t2)
5. FOOOF vs IRASA method comparison

Usage:
    python scripts/srm_spectral_differentiation.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FOOOF_DIR = os.path.join(BASE_DIR, 'exports_adaptive_v4', 'srm')
IRASA_DIR = os.path.join(BASE_DIR, 'exports_irasa_v4', 'srm')
FOOOF_T2_DIR = os.path.join(BASE_DIR, 'exports_adaptive_v4', 'srm_t2')
IRASA_T2_DIR = os.path.join(BASE_DIR, 'exports_irasa_v4', 'srm_t2')
PARTICIPANTS = os.path.join(BASE_DIR, 'exports_adaptive_v4', 'srm', 'participants.tsv')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'srm_spectral_differentiation')
os.makedirs(OUT_DIR, exist_ok=True)

# Try local participants.tsv first, then GCS copy
if not os.path.exists(PARTICIPANTS):
    PARTICIPANTS = os.path.join(BASE_DIR, 'data', 'srm_participants.tsv')

PHI_INV = 1.0 / PHI

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

BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
OCTAVE_BAND = {'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
               'n+2': 'beta_high', 'n+3': 'gamma'}

MIN_POWER_PCT = 50
MIN_PEAKS_PER_BAND = 30


def lattice_coord(freqs, f0=F0):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def assign_voronoi(u_vals):
    u = np.asarray(u_vals, dtype=float) % 1.0
    dists = np.abs(u[:, None] - POS_VALS[None, :])
    dists = np.minimum(dists, 1 - dists)
    return np.argmin(dists, axis=1)


def load_peaks(directory, min_power_pct=MIN_POWER_PCT):
    files = sorted(glob.glob(os.path.join(directory, '*_peaks.csv')))
    if not files:
        return None, 0
    first = pd.read_csv(files[0], nrows=1)
    has_power = 'power' in first.columns
    cols = ['freq', 'phi_octave'] + (['power'] if has_power else [])
    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=cols)
        dfs.append(df)
    peaks = pd.concat(dfs, ignore_index=True)
    if has_power and min_power_pct > 0:
        filtered = []
        for octave in peaks['phi_octave'].unique():
            bp = peaks[peaks.phi_octave == octave]
            thresh = bp['power'].quantile(min_power_pct / 100)
            filtered.append(bp[bp['power'] >= thresh])
        peaks = pd.concat(filtered, ignore_index=True)
    return peaks, len(files)


def compute_enrichment(peaks):
    rows = []
    for octave, band in OCTAVE_BAND.items():
        bp = peaks[peaks['phi_octave'] == octave]
        if len(bp) < 10:
            continue
        u = lattice_coord(bp['freq'].values)
        assignments = assign_voronoi(u)
        counts = np.bincount(assignments, minlength=N_POS)
        expected = len(bp) / N_POS
        for i, pos_name in enumerate(POS_NAMES):
            enr = (counts[i] / expected - 1) * 100 if expected > 0 else 0
            rows.append({'band': band, 'position': pos_name, 'enrichment_pct': enr,
                        'count': counts[i], 'expected': expected})
    return pd.DataFrame(rows)


def per_subject_enrichment(peaks_df):
    result = {}
    for octave, band in OCTAVE_BAND.items():
        bp = peaks_df[peaks_df['phi_octave'] == octave]
        n = len(bp)
        result[f'{band}_n_peaks'] = n
        if n < MIN_PEAKS_PER_BAND:
            continue
        u = lattice_coord(bp['freq'].values)
        assignments = assign_voronoi(u)
        counts = np.bincount(assignments, minlength=N_POS)
        expected = n / N_POS
        for i, pos_name in enumerate(POS_NAMES):
            enr = (counts[i] / expected - 1) * 100 if expected > 0 else 0
            result[f'{band}_{pos_name}'] = enr
    return result


def load_subject_enrichments(peak_dir):
    files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    rows = []
    for f in files:
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        df = pd.read_csv(f)
        if 'power' in df.columns and MIN_POWER_PCT > 0:
            filtered = []
            for octave in df['phi_octave'].unique():
                bp = df[df.phi_octave == octave]
                thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                filtered.append(bp[bp['power'] >= thresh])
            df = pd.concat(filtered, ignore_index=True)
        enr = per_subject_enrichment(df)
        enr['subject'] = sub_id
        rows.append(enr)
    return pd.DataFrame(rows)


def icc_21(x, y):
    n = len(x)
    grand_mean = (np.mean(x) + np.mean(y)) / 2
    ss_between = sum((((x[i] + y[i]) / 2 - grand_mean) ** 2) for i in range(n)) * 2
    ss_within = sum((x[i] - y[i]) ** 2 for i in range(n)) / 2
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / n
    icc = (ms_between - ms_within) / (ms_between + ms_within) if (ms_between + ms_within) > 0 else 0
    return icc


# =========================================================================
# ANALYSES
# =========================================================================

def analysis_1_enrichment():
    print("=" * 70)
    print("  1. VORONOI ENRICHMENT (SRM, FOOOF vs IRASA)")
    print("=" * 70)

    for method, peak_dir in [('FOOOF', FOOOF_DIR), ('IRASA', IRASA_DIR)]:
        peaks, n_sub = load_peaks(peak_dir)
        if peaks is None:
            print(f"  {method}: NO DATA")
            continue
        enr = compute_enrichment(peaks)
        print(f"\n  {method}: {n_sub} subjects, {len(peaks):,} peaks")
        print(f"  {'Band':<12s} {'Position':<16s} {'Enrichment':>10s}")
        print(f"  {'-'*12} {'-'*16} {'-'*10}")
        for band in BAND_ORDER:
            for _, row in enr[enr['band'] == band].iterrows():
                print(f"  {band:<12s} {row['position']:<16s} {row['enrichment_pct']:>+9.0f}%")
            print()
        enr.to_csv(os.path.join(OUT_DIR, f'enrichment_{method.lower()}.csv'), index=False)


def analysis_2_cognitive():
    from statsmodels.stats.multitest import multipletests

    print("=" * 70)
    print("  2. COGNITIVE CORRELATIONS (SRM)")
    print("=" * 70)

    # Load per-subject enrichments
    df = load_subject_enrichments(FOOOF_DIR)
    print(f"  Subjects with enrichment: {len(df)}")

    # Load participants.tsv
    if not os.path.exists(PARTICIPANTS):
        print(f"  participants.tsv not found at {PARTICIPANTS}")
        # Try to fetch from GCS
        import subprocess
        local_path = os.path.join(OUT_DIR, 'participants.tsv')
        subprocess.run(['gsutil', 'cp',
                       'gs://eeg-extraction-data/srm_resting_eeg/participants.tsv',
                       local_path], capture_output=True)
        if os.path.exists(local_path):
            participants = pd.read_csv(local_path, sep='\t')
        else:
            print("  Cannot load participants data. Skipping cognitive analysis.")
            return
    else:
        participants = pd.read_csv(PARTICIPANTS, sep='\t')

    print(f"  Participants data: {len(participants)} rows")
    print(f"  Columns: {list(participants.columns)}")

    # Merge
    df = df.merge(participants, left_on='subject', right_on='participant_id', how='left')

    # Define cognitive tests
    cog_cols = [c for c in participants.columns
                if c not in ('participant_id', 'age', 'sex')]
    for c in cog_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    enrich_cols = [c for c in df.columns
                   if any(c.startswith(b + '_') for b in BAND_ORDER)
                   and not c.endswith('_n_peaks')]

    print(f"  Enrichment features: {len(enrich_cols)}")
    print(f"  Cognitive measures: {len(cog_cols)}: {cog_cols}")

    # Run correlations
    results = []
    for feat in enrich_cols:
        for cog in cog_cols:
            valid = df[[feat, cog]].dropna()
            if len(valid) < 20:
                continue
            rho, p = stats.spearmanr(valid[feat], valid[cog])
            results.append({'feature': feat, 'cognitive': cog,
                           'rho': rho, 'p': p, 'n': len(valid)})

    rdf = pd.DataFrame(results)
    if len(rdf) > 0:
        rdf['p_fdr'] = multipletests(rdf['p'], method='fdr_bh')[1]
        sig = rdf[rdf['p_fdr'] < 0.05].sort_values('p_fdr')
        print(f"\n  Total tests: {len(rdf)}")
        print(f"  FDR-significant (q<0.05): {len(sig)}")
        if len(sig) > 0:
            print(f"\n  {'Feature':<35s} {'Cognitive':<12s} {'rho':>6s} {'p_fdr':>8s} {'N':>4s}")
            for _, r in sig.head(20).iterrows():
                print(f"  {r['feature']:<35s} {r['cognitive']:<12s} {r['rho']:>+.3f} {r['p_fdr']:>8.4f} {r['n']:>4.0f}")
        rdf.to_csv(os.path.join(OUT_DIR, 'cognitive_correlations.csv'), index=False)

    # Age correlations
    if 'age' in df.columns:
        print(f"\n  --- Age Correlations ---")
        age_results = []
        for feat in enrich_cols:
            valid = df[[feat, 'age']].dropna()
            if len(valid) < 20:
                continue
            rho, p = stats.spearmanr(valid[feat], valid['age'])
            age_results.append({'feature': feat, 'rho': rho, 'p': p, 'n': len(valid)})
        adf = pd.DataFrame(age_results)
        if len(adf) > 0:
            adf['p_fdr'] = multipletests(adf['p'], method='fdr_bh')[1]
            sig_age = adf[adf['p_fdr'] < 0.05].sort_values('p_fdr')
            print(f"  FDR-significant age correlations: {len(sig_age)}")
            if len(sig_age) > 0:
                for _, r in sig_age.head(15).iterrows():
                    print(f"    {r['feature']:<35s} rho={r['rho']:>+.3f} q={r['p_fdr']:.4f}")
            adf.to_csv(os.path.join(OUT_DIR, 'age_correlations.csv'), index=False)


def analysis_3_reliability():
    print("=" * 70)
    print("  3. TEST-RETEST RELIABILITY (SRM ses-t1 vs ses-t2)")
    print("=" * 70)

    for method, t1_dir, t2_dir in [('FOOOF', FOOOF_DIR, FOOOF_T2_DIR),
                                     ('IRASA', IRASA_DIR, IRASA_T2_DIR)]:
        if not os.path.exists(t2_dir):
            print(f"  {method}: ses-t2 data not found at {t2_dir}")
            continue

        t1_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                    for f in glob.glob(os.path.join(t1_dir, '*_peaks.csv'))}
        t2_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                    for f in glob.glob(os.path.join(t2_dir, '*_peaks.csv'))}
        matched = sorted(set(t1_files.keys()) & set(t2_files.keys()))
        print(f"\n  {method}: {len(matched)} matched subjects")

        if len(matched) < 10:
            print(f"  Too few matched subjects for reliability analysis")
            continue

        data_t1, data_t2 = {}, {}
        for sub_id in matched:
            p1 = pd.read_csv(t1_files[sub_id])
            p2 = pd.read_csv(t2_files[sub_id])
            if 'power' in p1.columns and MIN_POWER_PCT > 0:
                for octave in p1['phi_octave'].unique():
                    bp = p1[p1.phi_octave == octave]
                    thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                    p1 = pd.concat([p1[p1.phi_octave != octave], bp[bp['power'] >= thresh]])
                for octave in p2['phi_octave'].unique():
                    bp = p2[p2.phi_octave == octave]
                    thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                    p2 = pd.concat([p2[p2.phi_octave != octave], bp[bp['power'] >= thresh]])
            data_t1[sub_id] = per_subject_enrichment(p1)
            data_t2[sub_id] = per_subject_enrichment(p2)

        all_features = set()
        for d in list(data_t1.values()) + list(data_t2.values()):
            all_features.update(k for k in d.keys() if not k.endswith('_n_peaks'))

        results = []
        for feat in sorted(all_features):
            vals_t1, vals_t2 = [], []
            for sub_id in matched:
                v1 = data_t1[sub_id].get(feat, np.nan)
                v2 = data_t2[sub_id].get(feat, np.nan)
                if not np.isnan(v1) and not np.isnan(v2):
                    vals_t1.append(v1); vals_t2.append(v2)
            if len(vals_t1) < 10:
                continue
            x, y = np.array(vals_t1), np.array(vals_t2)
            r_val, _ = stats.pearsonr(x, y)
            icc = icc_21(x, y)
            results.append({'feature': feat, 'n': len(vals_t1),
                           'pearson_r': r_val, 'icc': icc})

        rdf = pd.DataFrame(results)
        if len(rdf) > 0:
            print(f"  Mean ICC: {rdf['icc'].mean():.3f}")
            print(f"  Mean Pearson r: {rdf['pearson_r'].mean():.3f}")
            print(f"\n  {'Feature':<35s} {'ICC':>6s} {'r':>6s} {'N':>4s}")
            for _, r in rdf.sort_values('icc', ascending=False).head(15).iterrows():
                print(f"  {r['feature']:<35s} {r['icc']:>+.3f} {r['pearson_r']:>+.3f} {r['n']:>4.0f}")
            rdf.to_csv(os.path.join(OUT_DIR, f'reliability_{method.lower()}.csv'), index=False)

            # Summary by band
            print(f"\n  ICC by band:")
            for band in BAND_ORDER:
                band_feats = rdf[rdf['feature'].str.startswith(band + '_')]
                if len(band_feats) > 0:
                    print(f"    {band:<12s} mean ICC={band_feats['icc'].mean():.3f} "
                          f"(range: {band_feats['icc'].min():.3f} to {band_feats['icc'].max():.3f})")


def analysis_4_method_comparison():
    print("=" * 70)
    print("  4. FOOOF vs IRASA METHOD COMPARISON")
    print("=" * 70)

    fooof_peaks, n_f = load_peaks(FOOOF_DIR)
    irasa_peaks, n_i = load_peaks(IRASA_DIR)

    if fooof_peaks is None or irasa_peaks is None:
        print("  Missing data for comparison")
        return

    fooof_enr = compute_enrichment(fooof_peaks)
    irasa_enr = compute_enrichment(irasa_peaks)

    print(f"  FOOOF: {n_f} subjects, {len(fooof_peaks):,} peaks")
    print(f"  IRASA: {n_i} subjects, {len(irasa_peaks):,} peaks")

    # Merge and compare
    merged = fooof_enr.merge(irasa_enr, on=['band', 'position'],
                              suffixes=('_fooof', '_irasa'))
    merged['diff'] = merged['enrichment_pct_fooof'] - merged['enrichment_pct_irasa']
    merged['agree'] = np.sign(merged['enrichment_pct_fooof']) == np.sign(merged['enrichment_pct_irasa'])

    n_agree = merged['agree'].sum()
    n_total = len(merged)
    r, p = stats.pearsonr(merged['enrichment_pct_fooof'], merged['enrichment_pct_irasa'])
    print(f"\n  Sign agreement: {n_agree}/{n_total} ({100*n_agree/n_total:.0f}%)")
    print(f"  Pearson r: {r:.3f} (p={p:.2e})")
    print(f"\n  {'Band':<12s} {'Position':<16s} {'FOOOF':>8s} {'IRASA':>8s} {'Agree':>6s}")
    for _, row in merged.iterrows():
        agree = '  Y' if row['agree'] else '  N'
        print(f"  {row['band']:<12s} {row['position']:<16s} "
              f"{row['enrichment_pct_fooof']:>+7.0f}% {row['enrichment_pct_irasa']:>+7.0f}% {agree}")

    merged.to_csv(os.path.join(OUT_DIR, 'method_comparison.csv'), index=False)


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    analysis_1_enrichment()
    print()
    analysis_2_cognitive()
    print()
    analysis_3_reliability()
    print()
    analysis_4_method_comparison()

    print("\n" + "=" * 70)
    print(f"  All results saved to: {OUT_DIR}")
    print("=" * 70)
