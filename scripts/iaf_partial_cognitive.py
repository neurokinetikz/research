#!/usr/bin/env python3
"""
IAF-Partialed Cognitive Correlations
======================================

Derives IAF (Individual Alpha Frequency) from peak CSVs, then tests
whether spectral differentiation × cognition correlations survive
after partialing out IAF (not just age).

If the cognitive signal survives IAF-partialing, it's not driven by
boundary misalignment from individual f₀ variation.

Usage:
    python scripts/iaf_partial_cognitive.py
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
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'f0_760_reanalysis')
MIN_POWER_PCT = 50

PHI_INV = 1.0 / PHI
OCTAVE_BAND = {
    'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
    'n+2': 'beta_high', 'n+3': 'gamma',
}

# Import enrichment machinery from main analysis script
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

COG_DIR = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/Cognitive_Test_Battery_LEMON'
META_PATH = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'

COG_TESTS = {
    'LPS': ('LPS/LPS.csv', 'LPS_1'),
    'TAP_Incompat': ('TAP_Incompatibility/TAP-Incompatibility.csv', 'TAP_I_1'),
    'RWT': ('RWT/RWT.csv', 'RWT_1'),
    'TMT': ('TMT/TMT.csv', 'TMT_1'),
}

# Voronoi machinery (minimal copy from run_all_f0_760_analyses.py)
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

_VORONOI_EDGES = []
for i in range(N_POS):
    if i == 0:
        u_left = (POS_VALS[-1] + POS_VALS[0] + 1) / 2 % 1.0
        u_right = (POS_VALS[0] + POS_VALS[1]) / 2
    elif i == N_POS - 1:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[0] + 1) / 2
    else:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[i + 1]) / 2
    _VORONOI_EDGES.append((u_left, u_right))

HZ_FRACS = []
for i in range(N_POS):
    u_left, u_right = _VORONOI_EDGES[i]
    if i == 0:
        hz_frac = (PHI ** 1.0 - PHI ** u_left + PHI ** u_right - PHI ** 0.0) / (PHI - 1)
    else:
        hz_frac = (PHI ** u_right - PHI ** u_left) / (PHI - 1)
    HZ_FRACS.append(hz_frac)


def lattice_coord(freqs, f0=F0):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def assign_voronoi(u_vals):
    u = np.asarray(u_vals, dtype=float) % 1.0
    dists = np.abs(u[:, None] - POS_VALS[None, :])
    dists = np.minimum(dists, 1 - dists)
    return np.argmin(dists, axis=1)


def per_subject_enrichment(peaks_df, min_peaks=30):
    results = {}
    for octave, band in OCTAVE_BAND.items():
        bp = peaks_df[peaks_df.phi_octave == octave]['freq'].values
        n = len(bp)
        if n < min_peaks:
            for pname in POS_NAMES:
                results[f'{band}_{pname}'] = np.nan
            results[f'{band}_mountain'] = np.nan
            results[f'{band}_ushape'] = np.nan
            results[f'{band}_peak_height'] = np.nan
            results[f'{band}_ramp_depth'] = np.nan
            results[f'{band}_center_depletion'] = np.nan
            results[f'{band}_asymmetry'] = np.nan
            results[f'{band}_n_peaks'] = n
            continue
        u = lattice_coord(bp)
        assignments = assign_voronoi(u)
        for i, pname in enumerate(POS_NAMES):
            count = int((assignments == i).sum())
            expected = HZ_FRACS[i] * n
            results[f'{band}_{pname}'] = (count / expected - 1) * 100 if expected > 0 else 0
        results[f'{band}_mountain'] = results[f'{band}_noble_1'] - results[f'{band}_boundary']
        results[f'{band}_ushape'] = (results[f'{band}_boundary'] + results.get(f'{band}_inv_noble_6', 0)) / 2 - results[f'{band}_attractor']
        results[f'{band}_peak_height'] = results[f'{band}_noble_1'] - results[f'{band}_attractor']
        results[f'{band}_ramp_depth'] = results[f'{band}_inv_noble_4'] - results[f'{band}_noble_4']
        center = np.mean([results[f'{band}_{p}'] for p in ['noble_5', 'noble_4', 'noble_3']])
        results[f'{band}_center_depletion'] = results[f'{band}_attractor'] - center
        upper = np.mean([results[f'{band}_{p}'] for p in ['inv_noble_3', 'inv_noble_4', 'inv_noble_5']])
        lower = np.mean([results[f'{band}_{p}'] for p in ['noble_5', 'noble_4', 'noble_3']])
        results[f'{band}_asymmetry'] = upper - lower
        results[f'{band}_n_peaks'] = n
    return results


def compute_iaf(peaks_df):
    """Compute IAF as power-weighted mean of alpha peaks."""
    alpha = peaks_df[peaks_df.phi_octave == 'n+0']
    if 'power' not in alpha.columns or len(alpha) < 5:
        return np.nan
    # Center-of-gravity IAF (more robust than peak)
    freqs = alpha['freq'].values
    powers = alpha['power'].values
    mask = (freqs >= 7.5) & (freqs <= 13.0) & (powers > 0)
    if mask.sum() < 3:
        return np.nan
    return np.average(freqs[mask], weights=powers[mask])


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 20:
        return np.nan, np.nan, 0
    x, y, z = x[valid], y[valid], z[valid]
    # Residualize on z using linear regression
    x_resid = x - np.polyval(np.polyfit(z, x, 1), z)
    y_resid = y - np.polyval(np.polyfit(z, y, 1), z)
    rho, p = stats.spearmanr(x_resid, y_resid)
    return rho, p, valid.sum()


def main():
    peak_dir = os.path.join(PEAK_BASE, 'lemon')
    files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    print(f"Loading {len(files)} subjects from LEMON...")

    rows = []
    for f in files:
        subj = os.path.basename(f).replace('_peaks.csv', '')
        df = pd.read_csv(f)
        has_power = 'power' in df.columns
        # Power filter
        if has_power and MIN_POWER_PCT > 0:
            filtered = []
            for octave in df['phi_octave'].unique():
                bp = df[df.phi_octave == octave]
                thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                filtered.append(bp[bp['power'] >= thresh])
            df = pd.concat(filtered, ignore_index=True)

        enr = per_subject_enrichment(df)
        enr['subject'] = subj
        enr['iaf'] = compute_iaf(df)
        rows.append(enr)

    subjects = pd.DataFrame(rows)
    print(f"  Subjects with IAF: {subjects['iaf'].notna().sum()}/{len(subjects)}")
    print(f"  IAF range: {subjects['iaf'].min():.2f} - {subjects['iaf'].max():.2f} Hz")
    print(f"  IAF mean: {subjects['iaf'].mean():.2f} ± {subjects['iaf'].std():.2f} Hz")

    # Load cognitive scores
    for test_name, (filename, col) in COG_TESTS.items():
        path = os.path.join(COG_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        cog_df = pd.read_csv(path)
        cog_df[col] = pd.to_numeric(cog_df[col], errors='coerce')
        cog_map = dict(zip(cog_df['ID'], cog_df[col]))
        subjects[f'cog_{test_name}'] = subjects['subject'].map(cog_map)

    # Load age
    if os.path.exists(META_PATH):
        meta = pd.read_csv(META_PATH)
        def age_bin_to_mid(s):
            try:
                lo, hi = s.split('-')
                return (float(lo) + float(hi)) / 2
            except Exception:
                return np.nan
        meta['age_mid'] = meta['Age'].apply(age_bin_to_mid)
        age_map = dict(zip(meta['ID'], meta['age_mid']))
        subjects['age'] = subjects['subject'].map(age_map)

    enrich_cols = [c for c in subjects.columns if any(
        c.startswith(f'{b}_') and not c.endswith('_n_peaks')
        for b in ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
    )]

    # Key comparison: raw vs age-partialed vs IAF-partialed vs both-partialed
    print(f"\n{'=' * 80}")
    print(f"  IAF-PARTIALED COGNITIVE CORRELATIONS")
    print(f"  (Does spectral differentiation × cognition survive IAF control?)")
    print(f"{'=' * 80}")

    results = []
    for test_name in COG_TESTS:
        cog_col = f'cog_{test_name}'
        if cog_col not in subjects.columns:
            continue
        for feat in enrich_cols:
            vals = subjects[[feat, cog_col, 'iaf', 'age']].dropna()
            if len(vals) < 30:
                continue

            x = vals[feat].values
            y = vals[cog_col].values
            iaf = vals['iaf'].values
            age = vals['age'].values

            # Raw
            rho_raw, p_raw = stats.spearmanr(x, y)

            # Age-partialed
            rho_age, p_age, n_age = partial_spearman(x, y, age)

            # IAF-partialed
            rho_iaf, p_iaf, n_iaf = partial_spearman(x, y, iaf)

            # Both partialed (age + IAF)
            # Residualize on both: x ~ age + iaf, y ~ age + iaf
            X_cov = np.column_stack([age, iaf])
            from numpy.linalg import lstsq
            beta_x, _, _, _ = lstsq(
                np.column_stack([X_cov, np.ones(len(X_cov))]), x, rcond=None)
            beta_y, _, _, _ = lstsq(
                np.column_stack([X_cov, np.ones(len(X_cov))]), y, rcond=None)
            x_resid = x - np.column_stack([X_cov, np.ones(len(X_cov))]) @ beta_x
            y_resid = y - np.column_stack([X_cov, np.ones(len(X_cov))]) @ beta_y
            rho_both, p_both = stats.spearmanr(x_resid, y_resid)

            results.append({
                'test': test_name,
                'feature': feat,
                'n': len(vals),
                'rho_raw': round(rho_raw, 4),
                'p_raw': round(p_raw, 6),
                'rho_age_partial': round(rho_age, 4) if not np.isnan(rho_age) else np.nan,
                'p_age_partial': round(p_age, 6) if not np.isnan(p_age) else np.nan,
                'rho_iaf_partial': round(rho_iaf, 4) if not np.isnan(rho_iaf) else np.nan,
                'p_iaf_partial': round(p_iaf, 6) if not np.isnan(p_iaf) else np.nan,
                'rho_both_partial': round(rho_both, 4),
                'p_both_partial': round(p_both, 6),
            })

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(OUT_DIR, 'iaf_partialed_cognitive.csv'), index=False)

    # Print top results comparison
    print(f"\n  Top 15 features by |raw rho| with LPS:")
    lps = rdf[rdf.test == 'LPS'].copy()
    lps['abs_raw'] = lps['rho_raw'].abs()
    top = lps.nlargest(15, 'abs_raw')
    print(f"\n  {'Feature':<35s} {'Raw':>8s} {'Age-p':>8s} {'IAF-p':>8s} {'Both-p':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, r in top.iterrows():
        print(f"  {r['feature']:<35s} {r['rho_raw']:+.3f}  {r['rho_age_partial']:+.3f}  "
              f"{r['rho_iaf_partial']:+.3f}  {r['rho_both_partial']:+.3f}")

    # Summary across all tests
    print(f"\n\n  Summary across all tests (features with |raw rho| > 0.20):")
    strong = rdf[rdf['rho_raw'].abs() > 0.20].copy()
    print(f"  N features: {len(strong)}")
    if len(strong) > 0:
        print(f"  Mean |raw rho|:        {strong['rho_raw'].abs().mean():.3f}")
        print(f"  Mean |age-partialed|:  {strong['rho_age_partial'].abs().mean():.3f}")
        print(f"  Mean |IAF-partialed|:  {strong['rho_iaf_partial'].abs().mean():.3f}")
        print(f"  Mean |both-partialed|: {strong['rho_both_partial'].abs().mean():.3f}")
        surv_iaf = (strong['p_iaf_partial'] < 0.05).sum()
        surv_both = (strong['p_both_partial'] < 0.05).sum()
        print(f"  Survive p<0.05 (IAF-partial): {surv_iaf}/{len(strong)}")
        print(f"  Survive p<0.05 (both-partial): {surv_both}/{len(strong)}")

    print(f"\n  Full results: {os.path.join(OUT_DIR, 'iaf_partialed_cognitive.csv')}")


if __name__ == '__main__':
    main()
