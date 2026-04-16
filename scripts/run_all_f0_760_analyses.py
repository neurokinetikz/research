#!/usr/bin/env python3
"""
Run All Analyses on f0=7.60 Extraction & Compile Comparison Report
===================================================================

Master script that:
1. Runs voronoi enrichment on all 9 EC datasets (f0=7.60 extraction)
2. Runs EC/EO comparisons (LEMON + Dortmund)
3. Runs Dortmund 2x2 condition analysis
4. Runs per-subject cognitive correlations (LEMON)
5. Runs per-subject HBN age/sex/psychopathology
6. Runs Dortmund adult aging trajectory
7. Runs personality correlations (LEMON)
8. Runs Dortmund ses-2 test-retest reliability
9. Produces side-by-side comparison with f0=7.83 results

Outputs everything to outputs/f0_760_reanalysis/

Usage:
    python scripts/run_all_f0_760_analyses.py --step enrichment
    python scripts/run_all_f0_760_analyses.py --step cognitive
    python scripts/run_all_f0_760_analyses.py --step hbn_age
    python scripts/run_all_f0_760_analyses.py --step dortmund_age
    python scripts/run_all_f0_760_analyses.py --step eceo
    python scripts/run_all_f0_760_analyses.py --step dortmund_2x2
    python scripts/run_all_f0_760_analyses.py --step personality
    python scripts/run_all_f0_760_analyses.py --step reliability
    python scripts/run_all_f0_760_analyses.py --step compare
    python scripts/run_all_f0_760_analyses.py --step all
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

PHI_INV = 1.0 / PHI
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OLD_PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive')
NEW_PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'f0_760_reanalysis')

# Global power filter: keep top N% of peaks by power within each band
# 0 = no filter, 50 = keep top half (default), 75 = keep top quarter
MIN_POWER_PCT = 50

# Minimum peaks per band per subject for per-subject enrichment profiles
MIN_PEAKS_PER_BAND = 30

# =========================================================================
# SHARED VORONOI MACHINERY
# =========================================================================

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

# Voronoi bin edges in u-space (midpoints between adjacent positions, circular)
_VORONOI_EDGES = []  # list of (u_left, u_right) for each position
for i in range(N_POS):
    if i == 0:
        # Boundary straddles u=0/1: left edge wraps from previous position
        u_left = (POS_VALS[-1] + POS_VALS[0] + 1) / 2 % 1.0  # midpoint wrapping
        u_right = (POS_VALS[0] + POS_VALS[1]) / 2
    elif i == N_POS - 1:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[0] + 1) / 2  # wrap to boundary
    else:
        u_left = (POS_VALS[i - 1] + POS_VALS[i]) / 2
        u_right = (POS_VALS[i] + POS_VALS[i + 1]) / 2
    _VORONOI_EDGES.append((u_left, u_right))

# Hz-weighted bin fractions: (phi^u_right - phi^u_left) / (phi - 1)
# This is the fraction of the band's Hz range covered by each bin.
# Under a Hz-uniform null, expected_count = HZ_FRAC[i] * total_peaks.
HZ_FRACS = []
for i in range(N_POS):
    u_left, u_right = _VORONOI_EDGES[i]
    if i == 0:
        # Boundary wraps: [u_left..1.0] + [0.0..u_right]
        hz_frac = (PHI ** 1.0 - PHI ** u_left + PHI ** u_right - PHI ** 0.0) / (PHI - 1)
    else:
        hz_frac = (PHI ** u_right - PHI ** u_left) / (PHI - 1)
    HZ_FRACS.append(hz_frac)

BOUNDARY_HW = POS_VALS[1] / 2
# Hz-weighted half-boundary fractions for the split boundary
BOUNDARY_LO_HZ_FRAC = (PHI ** BOUNDARY_HW - PHI ** 0.0) / (PHI - 1)
BOUNDARY_HI_HZ_FRAC = (PHI ** 1.0 - PHI ** (1 - BOUNDARY_HW)) / (PHI - 1)

OCTAVE_BAND = {
    'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
    'n+2': 'beta_high', 'n+3': 'gamma',
}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
BAND_HZ = {
    'theta': (4.70, 7.60), 'alpha': (7.60, 12.30),
    'beta_low': (12.30, 19.90), 'beta_high': (19.90, 32.19),
    'gamma': (32.19, 52.09),
}

EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp', 'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3', 'hbn_R4': 'hbn_R4', 'hbn_R5': 'hbn_R5', 'hbn_R6': 'hbn_R6',
    'hbn_R7': 'hbn_R7', 'hbn_R8': 'hbn_R8', 'hbn_R11': 'hbn_R11',
    'tdbrain': 'tdbrain', 'srm': 'srm',
}

SHORT_NAMES = {
    'eegmmidb': 'EEGM', 'lemon': 'LEM', 'dortmund': 'Dort',
    'chbmp': 'CHBMP', 'hbn_R1': 'R1', 'hbn_R2': 'R2',
    'hbn_R3': 'R3', 'hbn_R4': 'R4', 'hbn_R5': 'R5', 'hbn_R6': 'R6',
    'hbn_R7': 'R7n', 'hbn_R8': 'R8', 'hbn_R11': 'R11', 'tdbrain': 'TDB', 'srm': 'SRM',
}


def lattice_coord(freqs, f0=F0):
    return (np.log(freqs / f0) / np.log(PHI)) % 1.0


def assign_voronoi(u_vals):
    u = np.asarray(u_vals, dtype=float) % 1.0
    dists = np.abs(u[:, None] - POS_VALS[None, :])
    dists = np.minimum(dists, 1 - dists)
    return np.argmin(dists, axis=1)


def load_peaks(directory, min_power_pct=None):
    """Load peaks, optionally filtering to top peaks by power per band.

    min_power_pct: keep only peaks above this percentile of power within
    each band. 0 = keep all, 50 = keep top half, 75 = keep top quarter.
    """
    files = sorted(glob.glob(os.path.join(directory, '*_peaks.csv')))
    if not files:
        return None, 0
    # Check if power column exists
    first = pd.read_csv(files[0], nrows=1)
    has_power = 'power' in first.columns
    if min_power_pct is None:
        min_power_pct = MIN_POWER_PCT
    cols = ['freq', 'phi_octave'] + (['power'] if has_power else [])
    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=cols)
        dfs.append(df)
    peaks = pd.concat(dfs, ignore_index=True)

    if has_power and min_power_pct > 0:
        # Filter per band to keep top peaks by power
        filtered = []
        for octave in peaks['phi_octave'].unique():
            bp = peaks[peaks.phi_octave == octave]
            thresh = bp['power'].quantile(min_power_pct / 100)
            filtered.append(bp[bp['power'] >= thresh])
        peaks = pd.concat(filtered, ignore_index=True)

    return peaks, len(files)


def compute_enrichment(peaks_df):
    """Compute per-band Voronoi enrichment with Hz-weighted expected counts."""
    rows = []
    for octave, band_name in OCTAVE_BAND.items():
        band_peaks = peaks_df[peaks_df.phi_octave == octave]
        freqs = band_peaks['freq'].values
        n = len(freqs)
        f_lo, f_hi = BAND_HZ[band_name]
        if n < 10:
            for p_name in POS_NAMES + ['boundary_hi']:
                rows.append({'band': band_name, 'position': p_name,
                             'n_peaks': n, 'enrichment_pct': np.nan})
            continue
        u = lattice_coord(freqs)
        assignments = assign_voronoi(u)
        enrichments, counts = [], []
        for i in range(N_POS):
            count = int((assignments == i).sum())
            expected = HZ_FRACS[i] * n
            e = round((count / expected - 1) * 100) if expected > 0 else 0
            enrichments.append(e); counts.append(count)
        # Split boundary
        lower_count = int((u < BOUNDARY_HW).sum())
        upper_count = int((u >= (1 - BOUNDARY_HW)).sum())
        exp_lo = BOUNDARY_LO_HZ_FRAC * n
        exp_hi = BOUNDARY_HI_HZ_FRAC * n
        enr_lower = round((lower_count / exp_lo - 1) * 100) if exp_lo > 0 else 0
        enr_upper = round((upper_count / exp_hi - 1) * 100) if exp_hi > 0 else 0
        rows.append({'band': band_name, 'position': 'boundary', 'n_peaks': lower_count, 'enrichment_pct': enr_lower})
        for i in range(1, N_POS):
            rows.append({'band': band_name, 'position': POS_NAMES[i], 'n_peaks': counts[i], 'enrichment_pct': enrichments[i]})
        rows.append({'band': band_name, 'position': 'boundary_hi', 'n_peaks': upper_count, 'enrichment_pct': enr_upper})
    return pd.DataFrame(rows)


def per_subject_enrichment(peaks_df, min_peaks=30):
    """Per-subject enrichment with Hz-weighted expected counts."""
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
        # Derived metrics (boundary-based, legacy)
        results[f'{band}_mountain'] = results[f'{band}_noble_1'] - results[f'{band}_boundary']
        results[f'{band}_ushape'] = (results[f'{band}_boundary'] + results.get(f'{band}_inv_noble_6', 0)) / 2 - results[f'{band}_attractor']
        # Derived metrics (interior-only, robust)
        results[f'{band}_peak_height'] = results[f'{band}_noble_1'] - results[f'{band}_attractor']
        results[f'{band}_ramp_depth'] = results[f'{band}_inv_noble_4'] - results[f'{band}_noble_4']
        center = np.mean([results[f'{band}_{p}'] for p in ['noble_5', 'noble_4', 'noble_3']])
        results[f'{band}_center_depletion'] = results[f'{band}_attractor'] - center
        upper = np.mean([results[f'{band}_{p}'] for p in ['inv_noble_3', 'inv_noble_4', 'inv_noble_5']])
        lower = np.mean([results[f'{band}_{p}'] for p in ['noble_5', 'noble_4', 'noble_3']])
        results[f'{band}_asymmetry'] = upper - lower
        results[f'{band}_n_peaks'] = n
    return results


# =========================================================================
# STEP 1: ENRICHMENT (all 9 EC datasets)
# =========================================================================

def run_enrichment():
    print("=" * 70)
    print("  STEP 1: Voronoi Enrichment (9 EC datasets, f0=7.60 extraction)")
    print("=" * 70)

    all_results = {}
    for name, subdir in EC_DATASETS.items():
        path = os.path.join(NEW_PEAK_BASE, subdir)
        peaks, n_sub = load_peaks(path)
        if peaks is None:
            print(f"  {name}: NO DATA at {path}")
            continue
        enr = compute_enrichment(peaks)
        all_results[name] = enr
        n_total = peaks.shape[0]
        print(f"  {name}: {n_sub} subjects, {n_total:,} peaks")

    if not all_results:
        print("  No datasets found!")
        return

    # Precompute old enrichment (once per dataset, not per position)
    old_results = {}
    for name, subdir in EC_DATASETS.items():
        old_path = os.path.join(OLD_PEAK_BASE, subdir)
        old_peaks, _ = load_peaks(old_path)
        if old_peaks is not None:
            old_results[name] = compute_enrichment(old_peaks)

    # Build comparison CSV: new vs old
    all_positions = POS_NAMES + ['boundary_hi']
    csv_rows = []

    for band in BAND_ORDER:
        for pos in all_positions:
            row = {'band': band, 'position': pos}
            for name in EC_DATASETS:
                if name in all_results:
                    df = all_results[name]
                    r = df[(df['band'] == band) & (df['position'] == pos)]
                    v = int(r.iloc[0]['enrichment_pct']) if (not r.empty and not np.isnan(r.iloc[0]['enrichment_pct'])) else None
                    row[f'{SHORT_NAMES[name]}_new'] = v

                if name in old_results:
                    old_enr = old_results[name]
                    r = old_enr[(old_enr['band'] == band) & (old_enr['position'] == pos)]
                    v = int(r.iloc[0]['enrichment_pct']) if (not r.empty and not np.isnan(r.iloc[0]['enrichment_pct'])) else None
                    row[f'{SHORT_NAMES[name]}_old'] = v

            csv_rows.append(row)

    df_out = pd.DataFrame(csv_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df_out.to_csv(os.path.join(OUT_DIR, 'enrichment_9ds_comparison.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/enrichment_9ds_comparison.csv")

    # Print summary table
    print(f"\n  CROSS-DATASET SUMMARY (new f0=7.60 extraction)")
    for band in BAND_ORDER:
        print(f"\n  {band.upper()}")
        print(f"  {'Position':<16s}", end='')
        for name in EC_DATASETS:
            if name in all_results:
                print(f" {SHORT_NAMES[name]:>6s}", end='')
        print(f" {'Mean':>6s}")

        n_check = 0; n_tilde = 0; n_cross = 0
        for pos in all_positions:
            pos_disp = 'bnd_hi' if pos == 'boundary_hi' else pos
            print(f"  {pos_disp:<16s}", end='')
            vals = []
            for name in EC_DATASETS:
                if name in all_results:
                    df = all_results[name]
                    r = df[(df['band'] == band) & (df['position'] == pos)]
                    v = int(r.iloc[0]['enrichment_pct']) if (not r.empty and not np.isnan(r.iloc[0]['enrichment_pct'])) else None
                    print(f" {v:>+5d}%" if v is not None else "    —", end='')
                    if v is not None:
                        vals.append(v)
            mean = np.mean(vals) if vals else 0
            # Consistency marker
            if len(vals) >= 2:
                signs = [1 if v > 5 else (-1 if v < -5 else 0) for v in vals]
                pos_count = sum(1 for s in signs if s > 0)
                neg_count = sum(1 for s in signs if s < 0)
                if pos_count == len(vals) or neg_count == len(vals):
                    mark = '✓'; n_check += 1
                elif pos_count >= len(vals) - 1 or neg_count >= len(vals) - 1:
                    mark = '~'; n_tilde += 1
                else:
                    mark = '✗'; n_cross += 1
            else:
                mark = '—'; n_cross += 1
            print(f" {mean:>+5.0f}% {mark}")
        consistent = n_check + n_tilde
        print(f"  ✓={n_check} ~={n_tilde} ✗={n_cross} Consistent={consistent}/13")


# =========================================================================
# SHARED: Spearman correlation with FDR
# =========================================================================

def run_correlations(df, feature_cols, target_cols, target_prefix='', min_n=20):
    """Run Spearman correlations between feature_cols and target_cols, return DataFrame with FDR."""
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    results = []
    for tgt in target_cols:
        tgt_name = tgt.replace(target_prefix, '') if target_prefix else tgt
        for feat in feature_cols:
            valid = df[[feat, tgt]].dropna()
            if len(valid) < min_n:
                continue
            rho, p = stats.spearmanr(valid[feat], valid[tgt])
            results.append({'target': tgt_name, 'feature': feat, 'rho': rho, 'p': p,
                           'n': len(valid), 'abs_rho': abs(rho)})

    rdf = pd.DataFrame(results)
    if len(rdf) == 0:
        return rdf
    reject, p_fdr, _, _ = multipletests(rdf['p'].values, method='fdr_bh', alpha=0.05)
    rdf['p_fdr'] = p_fdr
    rdf['significant'] = reject
    return rdf


def print_correlation_summary(rdf, title, top_n=20):
    """Print standard correlation summary."""
    n_tests = len(rdf)
    n_sig = rdf['significant'].sum() if len(rdf) > 0 else 0
    n_unc = (rdf['p'] < 0.05).sum() if len(rdf) > 0 else 0
    expected = n_tests * 0.05

    print(f"\n  Total tests: {n_tests}")
    print(f"  FDR survivors (q=0.05): {n_sig}")
    if expected > 0:
        print(f"  Uncorrected p<0.05: {n_unc} (expected: {expected:.0f}, ratio: {n_unc/expected:.2f}x)")
    if len(rdf) > 0:
        print(f"  Largest |rho|: {rdf['abs_rho'].max():.3f}")

    if len(rdf) > 0:
        top = rdf.nlargest(top_n, 'abs_rho')
        print(f"\n  Top {top_n} by |rho|:")
        print(f"  {'Target':<20} {'Feature':<30} {'rho':>6} {'p':>10} {'p_FDR':>10} {'N':>5} {'Sig'}")
        print(f"  {'-'*85}")
        for _, row in top.iterrows():
            sig = '***' if row['p_fdr'] < 0.001 else ('**' if row['p_fdr'] < 0.01 else ('*' if row['p_fdr'] < 0.05 else ''))
            print(f"  {row['target']:<20} {row['feature']:<30} {row['rho']:>+.3f} {row['p']:>10.2e} {row['p_fdr']:>10.4f} {row['n']:>5d} {sig}")


def load_subject_enrichments(peak_dir, min_peaks=None, min_power_pct=None):
    """Load per-subject enrichment from a peak directory."""
    peak_files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    if not peak_files:
        return pd.DataFrame()
    if min_peaks is None:
        min_peaks = MIN_PEAKS_PER_BAND
    # Check if power column exists
    if min_power_pct is None:
        min_power_pct = MIN_POWER_PCT
    first = pd.read_csv(peak_files[0], nrows=1)
    has_power = 'power' in first.columns
    cols = ['freq', 'phi_octave'] + (['power'] if has_power else [])
    rows = []
    for f in peak_files:
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        peaks = pd.read_csv(f, usecols=cols)
        # Per-band power filtering for this subject
        if has_power and min_power_pct > 0:
            filtered = []
            for octave in peaks['phi_octave'].unique():
                bp = peaks[peaks.phi_octave == octave]
                if len(bp) >= 2:
                    thresh = bp['power'].quantile(min_power_pct / 100)
                    filtered.append(bp[bp['power'] >= thresh])
                else:
                    filtered.append(bp)
            peaks = pd.concat(filtered, ignore_index=True) if filtered else peaks
        enrich = per_subject_enrichment(peaks, min_peaks=min_peaks)
        enrich['subject'] = sub_id
        rows.append(enrich)
    return pd.DataFrame(rows)


def get_enrich_cols(df):
    """Get enrichment feature columns from a per-subject DataFrame."""
    return [c for c in df.columns if any(c.startswith(b + '_') for b in BAND_ORDER)
            and not c.endswith('_n_peaks') and c != 'subject']


# =========================================================================
# STEP 2: COGNITIVE (LEMON per-subject × cognitive battery)
# =========================================================================

COG_DIR = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/Cognitive_Test_Battery_LEMON'
META_PATH = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
COG_TESTS = {
    'CVLT': ('CVLT /CVLT.csv', 'CVLT_1'),
    'LPS': ('LPS/LPS.csv', 'LPS_1'),
    'RWT': ('RWT/RWT.csv', 'RWT_1'),
    'TAP_Alert': ('TAP_Alertness/TAP-Alertness.csv', 'TAP_A_1'),
    'TAP_Incompat': ('TAP_Incompatibility/TAP-Incompatibility.csv', 'TAP_I_1'),
    'TAP_WM': ('TAP_Working_Memory/TAP-Working Memory.csv', 'TAP_WM_1'),
    'TMT': ('TMT/TMT.csv', 'TMT_1'),
    'WST': ('WST/WST.csv', 'WST_1'),
}


def run_cognitive():
    print("=" * 70)
    print("  STEP 2: Per-Subject Cognitive Correlations (LEMON EC)")
    print("=" * 70)

    peak_dir = os.path.join(NEW_PEAK_BASE, 'lemon')
    df = load_subject_enrichments(peak_dir)
    print(f"  Subjects: {len(df)}")

    # Load cognitive scores
    for test_name, (filename, col) in COG_TESTS.items():
        path = os.path.join(COG_DIR, filename)
        if not os.path.exists(path):
            continue
        cog_df = pd.read_csv(path)
        cog_df[col] = pd.to_numeric(cog_df[col], errors='coerce')
        cog_map = dict(zip(cog_df['ID'], cog_df[col]))
        df[f'cog_{test_name}'] = df['subject'].map(cog_map)

    # Load demographics
    if os.path.exists(META_PATH):
        meta = pd.read_csv(META_PATH)
        # Convert age bin string ("20-25") to numeric midpoint
        def age_bin_to_midpoint(s):
            try:
                lo, hi = s.split('-')
                return (float(lo) + float(hi)) / 2
            except Exception:
                return np.nan
        meta['age_mid'] = meta['Age'].apply(age_bin_to_midpoint)
        age_map = dict(zip(meta['ID'], meta['age_mid']))
        sex_map = dict(zip(meta['ID'], meta['Gender_ 1=female_2=male']))
        df['age_bin'] = df['subject'].map(age_map)
        df['sex'] = df['subject'].map(sex_map)

    enrich_cols = get_enrich_cols(df)
    cog_cols = [c for c in df.columns if c.startswith('cog_')]
    print(f"  Enrichment features: {len(enrich_cols)}, Cognitive tests: {len(cog_cols)}")

    # Cognitive correlations
    rdf = run_correlations(df, enrich_cols, cog_cols, target_prefix='cog_')
    print_correlation_summary(rdf, 'Cognitive')
    rdf.to_csv(os.path.join(OUT_DIR, 'cognitive_correlations.csv'), index=False)

    # Age-partialed cognitive correlations (LPS only, the key finding)
    if 'age_bin' in df.columns and 'cog_LPS' in df.columns:
        from scipy import stats
        print(f"\n  --- Age-Partialed LPS Correlations ---")
        for feat in enrich_cols:
            valid = df[[feat, 'cog_LPS', 'age_bin']].dropna()
            if len(valid) < 30:
                continue
            # Partial correlation: residualize both on age
            _, _, r_feat_age = stats.linregress(valid['age_bin'], valid[feat])[:3]
            _, _, r_lps_age = stats.linregress(valid['age_bin'], valid['cog_LPS'])[:3]
            feat_resid = valid[feat] - np.polyval(np.polyfit(valid['age_bin'], valid[feat], 1), valid['age_bin'])
            lps_resid = valid['cog_LPS'] - np.polyval(np.polyfit(valid['age_bin'], valid['cog_LPS'], 1), valid['age_bin'])
            rho, p = stats.spearmanr(feat_resid, lps_resid)
            if abs(rho) > 0.15:
                print(f"    {feat:<30s} partial_rho={rho:+.3f} p={p:.4f} N={len(valid)}")

    # Age correlations
    if 'age_bin' in df.columns:
        print(f"\n  --- Age × Enrichment ---")
        age_rdf = run_correlations(df, enrich_cols, ['age_bin'])
        print_correlation_summary(age_rdf, 'Age', top_n=10)
        age_rdf.to_csv(os.path.join(OUT_DIR, 'lemon_age_correlations.csv'), index=False)


# =========================================================================
# STEP 3: HBN AGE/SEX/PSYCHOPATHOLOGY
# =========================================================================

HBN_RELEASES = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R11']


def run_hbn_age():
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    print("=" * 70)
    print("  STEP 3: HBN Developmental Trajectory")
    print("=" * 70)

    # Load demographics
    demo_rows = []
    for release in HBN_RELEASES:
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if not os.path.exists(tsv):
            continue
        d = pd.read_csv(tsv, sep='\t')
        d['release'] = release
        demo_rows.append(d)
    demo = pd.concat(demo_rows, ignore_index=True) if demo_rows else pd.DataFrame()
    print(f"  Demographics: {len(demo)} subjects")

    # Load per-subject enrichment
    rows = []
    for release in HBN_RELEASES:
        peak_dir = os.path.join(NEW_PEAK_BASE, f'hbn_{release}')
        peak_files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
        for f in peak_files:
            sub_id = os.path.basename(f).replace('_peaks.csv', '')
            peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
            enrich = per_subject_enrichment(peaks)
            enrich['subject'] = sub_id
            enrich['release'] = release
            match = demo[demo['participant_id'] == sub_id]
            if len(match) > 0:
                m = match.iloc[0]
                enrich['age'] = m['age']
                enrich['sex'] = m.get('sex', np.nan)
                for psy in ['p_factor', 'attention', 'internalizing', 'externalizing']:
                    enrich[psy] = pd.to_numeric(m.get(psy, np.nan), errors='coerce')
            rows.append(enrich)

    df = pd.DataFrame(rows)
    n_age = df['age'].notna().sum()
    print(f"  Per-subject enrichment: {len(df)} subjects, {n_age} with age")

    enrich_cols = get_enrich_cols(df)

    # Age correlations
    age_rdf = run_correlations(df, enrich_cols, ['age'])
    print(f"\n  --- AGE × ENRICHMENT (N={n_age}) ---")
    print_correlation_summary(age_rdf, 'HBN Age', top_n=15)
    age_rdf.to_csv(os.path.join(OUT_DIR, 'hbn_age_correlations.csv'), index=False)

    # Psychopathology
    for psy in ['p_factor', 'attention', 'internalizing', 'externalizing']:
        n_valid = df[psy].notna().sum()
        if n_valid < 50:
            continue
        print(f"\n  --- {psy.upper()} (N={n_valid}) ---")
        psy_rdf = run_correlations(df, enrich_cols, [psy])
        n_sig = psy_rdf['significant'].sum() if len(psy_rdf) > 0 else 0
        print(f"  FDR survivors: {n_sig}")
        if len(psy_rdf) > 0:
            top = psy_rdf.nlargest(5, 'abs_rho')
            for _, r in top.iterrows():
                sig = '*' if r['p_fdr'] < 0.05 else ''
                print(f"    {r['feature']:<35} rho={r['rho']:>+.3f} p_FDR={r['p_fdr']:.4f} {sig}")
        psy_rdf.to_csv(os.path.join(OUT_DIR, f'hbn_{psy}_correlations.csv'), index=False)

    # Sex differences
    males = df[df['sex'] == 'M']
    females = df[df['sex'] == 'F']
    print(f"\n  --- SEX DIFFERENCES (M={len(males)}, F={len(females)}) ---")
    sex_results = []
    for col in enrich_cols:
        m_vals = males[col].dropna()
        f_vals = females[col].dropna()
        if len(m_vals) < 30 or len(f_vals) < 30:
            continue
        u_stat, p = stats.mannwhitneyu(m_vals, f_vals, alternative='two-sided')
        pooled = np.sqrt((m_vals.std()**2 + f_vals.std()**2) / 2)
        d = (m_vals.mean() - f_vals.mean()) / pooled if pooled > 0 else 0
        sex_results.append({'feature': col, 'd': d, 'p': p, 'abs_d': abs(d)})
    sex_df = pd.DataFrame(sex_results)
    if len(sex_df) > 0:
        rej, pfdr, _, _ = multipletests(sex_df['p'].values, method='fdr_bh', alpha=0.05)
        sex_df['p_fdr'] = pfdr
        sex_df['significant'] = rej
        print(f"  FDR survivors: {sex_df['significant'].sum()}")
        sex_df.to_csv(os.path.join(OUT_DIR, 'hbn_sex_differences.csv'), index=False)


# =========================================================================
# STEP 3b: TDBRAIN (Age, Diagnosis, Psychopathology, Personality)
# =========================================================================

TDBRAIN_PARTICIPANTS = os.path.expanduser(
    '~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')


def run_tdbrain():
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    print("=" * 70)
    print("  STEP 3b: TDBRAIN (Age, Diagnosis, Psychopathology, Personality)")
    print("=" * 70)

    # Load demographics
    if not os.path.exists(TDBRAIN_PARTICIPANTS):
        print(f"  ERROR: {TDBRAIN_PARTICIPANTS} not found")
        return
    demo = pd.read_csv(TDBRAIN_PARTICIPANTS, sep='\t')
    demo['age_float'] = demo['age'].str.replace(',', '.').astype(float)
    demo['dx_group'] = 'OTHER'
    demo.loc[demo['indication'] == 'HEALTHY', 'dx_group'] = 'HEALTHY'
    demo.loc[demo['indication'].str.contains('ADHD', na=False) &
             ~demo['indication'].str.contains('MDD', na=False), 'dx_group'] = 'ADHD'
    demo.loc[demo['indication'].str.contains('MDD', na=False) &
             ~demo['indication'].str.contains('ADHD', na=False), 'dx_group'] = 'MDD'
    demo.loc[demo['indication'].str.contains('OCD', na=False), 'dx_group'] = 'OCD'

    # Only discovery set
    demo = demo[demo['DISC/REP'] == 'DISCOVERY']
    print(f"  Demographics: {len(demo)} discovery subjects")

    # Load per-subject enrichment
    peak_dir = os.path.join(NEW_PEAK_BASE, 'tdbrain')
    df = load_subject_enrichments(peak_dir)
    print(f"  Per-subject enrichment: {len(df)} subjects")

    # Merge with demographics (TDBRAIN subjects use sub-{numeric_id})
    # The participants file uses numeric IDs, peaks use sub-{id}
    # participants_ID already has 'sub-' prefix in the V2 file
    pid_str = demo['participants_ID'].astype(str)
    demo['subject_key'] = pid_str.where(pid_str.str.startswith('sub-'), 'sub-' + pid_str)
    df = df.merge(demo[['subject_key', 'age_float', 'gender', 'dx_group',
                         'indication', 'formal_status']],
                  left_on='subject', right_on='subject_key', how='inner')
    df['age'] = df['age_float']
    df['sex'] = df['gender'].map({0: 'M', 1: 'F'})

    n_age = df['age'].notna().sum()
    print(f"  Merged: {len(df)} subjects, {n_age} with age")
    print(f"  Groups: {df['dx_group'].value_counts().to_dict()}")

    enrich_cols = get_enrich_cols(df)

    # --- 1. Age correlations ---
    age_rdf = run_correlations(df, enrich_cols, ['age'])
    print(f"\n  --- TDBRAIN AGE × ENRICHMENT (N={n_age}, ages {df['age'].min():.0f}-{df['age'].max():.0f}) ---")
    print_correlation_summary(age_rdf, 'TDBRAIN Age', top_n=15)
    age_rdf.to_csv(os.path.join(OUT_DIR, 'tdbrain_age_correlations.csv'), index=False)

    # --- 2. Diagnosis group comparisons ---
    print(f"\n  --- DIAGNOSTIC GROUP COMPARISONS ---")
    adults = df[df['age'] >= 18]
    for dx_pair in [('ADHD', 'MDD'), ('ADHD', 'HEALTHY'), ('MDD', 'HEALTHY')]:
        g1 = adults[adults.dx_group == dx_pair[0]]
        g2 = adults[adults.dx_group == dx_pair[1]]
        if len(g1) < 10 or len(g2) < 10:
            print(f"  {dx_pair[0]} vs {dx_pair[1]}: insufficient N ({len(g1)} vs {len(g2)})")
            continue
        print(f"\n  {dx_pair[0]} (N={len(g1)}, age={g1['age'].mean():.1f}) vs "
              f"{dx_pair[1]} (N={len(g2)}, age={g2['age'].mean():.1f}):")
        dx_results = []
        for col in enrich_cols:
            v1 = g1[col].dropna()
            v2 = g2[col].dropna()
            if len(v1) < 10 or len(v2) < 10:
                continue
            u, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            pooled_sd = np.sqrt((v1.std()**2 + v2.std()**2) / 2)
            d = (v1.mean() - v2.mean()) / pooled_sd if pooled_sd > 0 else 0
            dx_results.append({'feature': col, 'd': d, 'p': p, 'abs_d': abs(d),
                                'group1': dx_pair[0], 'group2': dx_pair[1]})
        dx_df = pd.DataFrame(dx_results)
        if len(dx_df) > 0:
            rej, pfdr, _, _ = multipletests(dx_df['p'].values, method='fdr_bh', alpha=0.05)
            dx_df['p_fdr'] = pfdr
            dx_df['significant'] = rej
            n_sig = dx_df['significant'].sum()
            print(f"    FDR survivors: {n_sig}/{len(dx_df)}")
            top = dx_df.nlargest(5, 'abs_d')
            for _, r in top.iterrows():
                sig = '*' if r['p_fdr'] < 0.05 else ''
                print(f"      {r['feature']:<35} d={r['d']:>+.3f} p_FDR={r['p_fdr']:.4f} {sig}")
            dx_df.to_csv(os.path.join(OUT_DIR, f'tdbrain_{dx_pair[0]}_vs_{dx_pair[1]}.csv'), index=False)

    # --- 3. NEO-FFI personality correlations ---
    # NEO-FFI items are neoFFI_q1-q60 in participants file
    # Compute Big Five from items: N=1-12, E=13-24, O=25-36, A=37-48, C=49-60
    # (reverse-scored items would need the scoring key -- use raw sum for now)
    demo_full = pd.read_csv(TDBRAIN_PARTICIPANTS, sep='\t')
    demo_full = demo_full[demo_full['DISC/REP'] == 'DISCOVERY']
    neo_cols = [f'neoFFI_q{i}' for i in range(1, 61)]
    has_neo = all(c in demo_full.columns for c in neo_cols[:5])
    if has_neo:
        # Convert to numeric (may have 'REPLICATION' strings)
        for c in neo_cols:
            demo_full[c] = pd.to_numeric(demo_full[c], errors='coerce')
        # Simple sum scores (approximate, ignoring reverse scoring)
        demo_full['NEO_N'] = demo_full[[f'neoFFI_q{i}' for i in range(1, 13)]].sum(axis=1, min_count=6)
        demo_full['NEO_E'] = demo_full[[f'neoFFI_q{i}' for i in range(13, 25)]].sum(axis=1, min_count=6)
        demo_full['NEO_O'] = demo_full[[f'neoFFI_q{i}' for i in range(25, 37)]].sum(axis=1, min_count=6)
        demo_full['NEO_A'] = demo_full[[f'neoFFI_q{i}' for i in range(37, 49)]].sum(axis=1, min_count=6)
        demo_full['NEO_C'] = demo_full[[f'neoFFI_q{i}' for i in range(49, 61)]].sum(axis=1, min_count=6)

        demo_full['subject_key'] = 'sub-' + demo_full['participants_ID'].astype(str)
        personality_cols = ['NEO_N', 'NEO_E', 'NEO_O', 'NEO_A', 'NEO_C']
        for col in personality_cols:
            df[col] = df['subject'].map(dict(zip(demo_full['subject_key'], demo_full[col])))

        valid_personality = df[personality_cols].notna().all(axis=1).sum()
        if valid_personality > 50:
            print(f"\n  --- NEO-FFI PERSONALITY × ENRICHMENT (N={valid_personality}) ---")
            pers_rdf = run_correlations(df, enrich_cols, personality_cols)
            n_sig = pers_rdf['significant'].sum() if len(pers_rdf) > 0 else 0
            n_total = len(pers_rdf)
            print(f"  FDR survivors: {n_sig}/{n_total}")
            pers_rdf.to_csv(os.path.join(OUT_DIR, 'tdbrain_personality_correlations.csv'), index=False)

    # --- 4. Sex differences ---
    males = df[df['sex'] == 'M']
    females = df[df['sex'] == 'F']
    if len(males) > 30 and len(females) > 30:
        print(f"\n  --- SEX DIFFERENCES (M={len(males)}, F={len(females)}) ---")
        sex_results = []
        for col in enrich_cols:
            m_vals = males[col].dropna()
            f_vals = females[col].dropna()
            if len(m_vals) < 30 or len(f_vals) < 30:
                continue
            u, p = stats.mannwhitneyu(m_vals, f_vals, alternative='two-sided')
            pooled_sd = np.sqrt((m_vals.std()**2 + f_vals.std()**2) / 2)
            d = (m_vals.mean() - f_vals.mean()) / pooled_sd if pooled_sd > 0 else 0
            sex_results.append({'feature': col, 'd': d, 'p': p, 'abs_d': abs(d)})
        sex_df = pd.DataFrame(sex_results)
        if len(sex_df) > 0:
            rej, pfdr, _, _ = multipletests(sex_df['p'].values, method='fdr_bh', alpha=0.05)
            sex_df['p_fdr'] = pfdr
            sex_df['significant'] = rej
            print(f"  FDR survivors: {sex_df['significant'].sum()}")
            sex_df.to_csv(os.path.join(OUT_DIR, 'tdbrain_sex_differences.csv'), index=False)

    # Save full per-subject data
    df.to_csv(os.path.join(OUT_DIR, 'tdbrain_per_subject_enrichment.csv'), index=False)


# =========================================================================
# STEP 4: DORTMUND ADULT AGING
# =========================================================================

def run_dortmund_age():
    print("=" * 70)
    print("  STEP 4: Dortmund Adult Aging Trajectory")
    print("=" * 70)

    peak_dir = os.path.join(NEW_PEAK_BASE, 'dortmund')
    df = load_subject_enrichments(peak_dir)
    print(f"  Subjects: {len(df)}")

    # Load demographics
    demo_path = '/Volumes/T9/dortmund_data/participants.tsv'
    if os.path.exists(demo_path):
        demo = pd.read_csv(demo_path, sep='\t')
        age_map = dict(zip(demo['participant_id'], demo['age']))
        sex_map = dict(zip(demo['participant_id'], demo.get('sex', pd.Series(dtype=str))))
        df['age'] = df['subject'].map(age_map)
        df['sex'] = df['subject'].map(sex_map)

    enrich_cols = get_enrich_cols(df)
    n_age = df['age'].notna().sum()
    print(f"  With age data: {n_age}")

    age_rdf = run_correlations(df, enrich_cols, ['age'])
    print(f"\n  --- DORTMUND AGE × ENRICHMENT (N={n_age}) ---")
    print_correlation_summary(age_rdf, 'Dortmund Age', top_n=15)
    age_rdf.to_csv(os.path.join(OUT_DIR, 'dortmund_age_correlations.csv'), index=False)


# =========================================================================
# STEP 5: EC/EO COMPARISON (LEMON + Dortmund)
# =========================================================================

def run_eceo():
    print("=" * 70)
    print("  STEP 5: EC vs EO Comparison (LEMON + Dortmund)")
    print("=" * 70)

    comparisons = [
        ('LEMON', 'lemon', 'lemon_EO'),
        ('Dortmund', 'dortmund', 'dortmund_EO_pre'),
    ]

    all_positions = POS_NAMES + ['boundary_hi']
    csv_rows = []

    for label, ec_dir, eo_dir in comparisons:
        ec_path = os.path.join(NEW_PEAK_BASE, ec_dir)
        eo_path = os.path.join(NEW_PEAK_BASE, eo_dir)
        ec_peaks, n_ec = load_peaks(ec_path)
        eo_peaks, n_eo = load_peaks(eo_path)

        if ec_peaks is None or eo_peaks is None:
            print(f"  {label}: missing data (EC={ec_path}, EO={eo_path})")
            continue

        ec_enr = compute_enrichment(ec_peaks)
        eo_enr = compute_enrichment(eo_peaks)
        print(f"\n  {label}: EC={n_ec} subjects, EO={n_eo} subjects")

        for band in BAND_ORDER:
            print(f"\n    {band.upper()}")
            print(f"    {'Position':<16s} {'EC':>7s} {'EO':>7s} {'Δ':>6s}")
            print(f"    {'-'*38}")
            for pos in all_positions:
                ec_row = ec_enr[(ec_enr['band'] == band) & (ec_enr['position'] == pos)]
                eo_row = eo_enr[(eo_enr['band'] == band) & (eo_enr['position'] == pos)]
                v_ec = int(ec_row.iloc[0]['enrichment_pct']) if (not ec_row.empty and not np.isnan(ec_row.iloc[0]['enrichment_pct'])) else None
                v_eo = int(eo_row.iloc[0]['enrichment_pct']) if (not eo_row.empty and not np.isnan(eo_row.iloc[0]['enrichment_pct'])) else None
                delta = (v_eo - v_ec) if (v_ec is not None and v_eo is not None) else None
                pos_d = 'bnd_hi' if pos == 'boundary_hi' else pos
                s_ec = f'{v_ec:+d}%' if v_ec is not None else '—'
                s_eo = f'{v_eo:+d}%' if v_eo is not None else '—'
                s_d = f'{delta:+d}' if delta is not None else '—'
                print(f"    {pos_d:<16s} {s_ec:>7s} {s_eo:>7s} {s_d:>6s}")
                csv_rows.append({'dataset': label, 'band': band, 'position': pos,
                                'ec': v_ec, 'eo': v_eo, 'delta': delta})

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, 'eceo_comparison.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/eceo_comparison.csv")


# =========================================================================
# STEP 6: DORTMUND 2×2 (EC/EO × pre/post)
# =========================================================================

def run_dortmund_2x2():
    print("=" * 70)
    print("  STEP 6: Dortmund 2×2 (EC/EO × pre/post)")
    print("=" * 70)

    conditions = {
        'EC-pre':  'dortmund',
        'EO-pre':  'dortmund_EO_pre',
        'EC-post': 'dortmund_EC_post',
        'EO-post': 'dortmund_EO_post',
    }

    results = {}
    for cond, subdir in conditions.items():
        path = os.path.join(NEW_PEAK_BASE, subdir)
        peaks, n_sub = load_peaks(path)
        if peaks is None:
            print(f"  {cond}: NO DATA")
            continue
        results[cond] = compute_enrichment(peaks)
        print(f"  {cond}: {n_sub} subjects, {peaks.shape[0]:,} peaks")

    if len(results) < 4:
        print("  Need all 4 conditions. Skipping.")
        return

    all_positions = POS_NAMES + ['boundary_hi']
    csv_rows = []

    for band in BAND_ORDER:
        print(f"\n  {band.upper()}")
        print(f"  {'Position':<16s} {'EC-pre':>7s} {'EC-post':>8s} {'EO-pre':>7s} {'EO-post':>8s}  {'max|Δ|':>7s}")
        print(f"  {'-'*60}")
        for pos in all_positions:
            vals = {}
            for cond in conditions:
                if cond not in results:
                    continue
                r = results[cond]
                row = r[(r['band'] == band) & (r['position'] == pos)]
                vals[cond] = int(row.iloc[0]['enrichment_pct']) if (not row.empty and not np.isnan(row.iloc[0]['enrichment_pct'])) else None

            pos_d = 'bnd_hi' if pos == 'boundary_hi' else pos
            valid_vals = [v for v in vals.values() if v is not None]
            max_delta = max(valid_vals) - min(valid_vals) if len(valid_vals) >= 2 else 0
            parts = []
            for cond in conditions:
                v = vals.get(cond)
                parts.append(f'{v:>+5d}%' if v is not None else '    —')
            print(f"  {pos_d:<16s} {'  '.join(parts)}  {max_delta:>5d}")
            csv_rows.append({'band': band, 'position': pos, **vals, 'max_delta': max_delta})

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, 'dortmund_2x2.csv'), index=False)


# =========================================================================
# STEP 7: PERSONALITY (LEMON)
# =========================================================================

EMO_DIR = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/Emotion_and_Personality_Test_Battery_LEMON'


def run_personality():
    print("=" * 70)
    print("  STEP 7: Personality/Emotion Correlations (LEMON EC)")
    print("=" * 70)

    peak_dir = os.path.join(NEW_PEAK_BASE, 'lemon')
    df = load_subject_enrichments(peak_dir)
    print(f"  Subjects: {len(df)}")

    # Load personality scores
    csvs = sorted(glob.glob(os.path.join(EMO_DIR, '*.csv')))
    for fpath in csvs:
        name = os.path.basename(fpath).replace('.csv', '')
        try:
            pdf = pd.read_csv(fpath)
        except Exception:
            continue
        if 'ID' not in pdf.columns:
            continue
        cols = [c for c in pdf.columns if c != 'ID']
        for c in cols:
            pdf[c] = pd.to_numeric(pdf[c], errors='coerce')
            if pdf[c].notna().sum() < 100:
                continue
            key = f"pers_{name}_{c}" if not c.startswith(name) else f"pers_{c}"
            score_map = dict(zip(pdf['ID'], pdf[c]))
            df[key] = df['subject'].map(score_map)

    enrich_cols = get_enrich_cols(df)
    pers_cols = [c for c in df.columns if c.startswith('pers_')]
    print(f"  Enrichment features: {len(enrich_cols)}, Personality subscales: {len(pers_cols)}")

    rdf = run_correlations(df, enrich_cols, pers_cols, target_prefix='pers_')
    print_correlation_summary(rdf, 'Personality', top_n=25)
    rdf.to_csv(os.path.join(OUT_DIR, 'personality_correlations.csv'), index=False)


# =========================================================================
# STEP 8: TEST-RETEST RELIABILITY (Dortmund ses-1 vs ses-2)
# =========================================================================

def icc_21(x, y):
    """ICC(2,1) for two-session data."""
    n = len(x)
    if n < 3:
        return np.nan
    grand_mean = np.mean(np.concatenate([x, y]))
    row_means = (x + y) / 2
    col_means = np.array([np.mean(x), np.mean(y)])
    ms_between = 2 * np.sum((row_means - grand_mean) ** 2) / (n - 1)
    ms_within = np.sum((x - row_means) ** 2 + (y - row_means) ** 2) / n
    if (ms_between + ms_within) == 0:
        return np.nan
    return (ms_between - ms_within) / (ms_between + ms_within)


def run_reliability():
    from scipy import stats

    print("=" * 70)
    print("  STEP 8: Test-Retest Reliability (Dortmund ses-1 vs ses-2)")
    print("=" * 70)

    ses1_dir = os.path.join(NEW_PEAK_BASE, 'dortmund')
    ses2_dir = os.path.join(NEW_PEAK_BASE, 'dortmund_EC_pre_ses2')

    if not os.path.exists(ses2_dir):
        print(f"  ses-2 data not found at {ses2_dir}")
        return

    ses1_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                  for f in glob.glob(os.path.join(ses1_dir, '*_peaks.csv'))}
    ses2_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                  for f in glob.glob(os.path.join(ses2_dir, '*_peaks.csv'))}
    matched = sorted(set(ses1_files.keys()) & set(ses2_files.keys()))
    print(f"  Matched subjects: {len(matched)}")

    data_s1, data_s2 = {}, {}
    for sub_id in matched:
        p1 = pd.read_csv(ses1_files[sub_id], usecols=['freq', 'phi_octave'])
        p2 = pd.read_csv(ses2_files[sub_id], usecols=['freq', 'phi_octave'])
        data_s1[sub_id] = per_subject_enrichment(p1)
        data_s2[sub_id] = per_subject_enrichment(p2)

    all_features = set()
    for d in list(data_s1.values()) + list(data_s2.values()):
        all_features.update(k for k in d.keys() if not k.endswith('_n_peaks'))

    results = []
    for feat in sorted(all_features):
        vals_s1, vals_s2 = [], []
        for sub_id in matched:
            v1 = data_s1[sub_id].get(feat, np.nan)
            v2 = data_s2[sub_id].get(feat, np.nan)
            if not np.isnan(v1) and not np.isnan(v2):
                vals_s1.append(v1); vals_s2.append(v2)
        if len(vals_s1) < 20:
            continue
        x, y = np.array(vals_s1), np.array(vals_s2)
        r_val, _ = stats.pearsonr(x, y)
        icc = icc_21(x, y)
        pooled_sd = np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
        d = np.mean(y - x) / pooled_sd if pooled_sd > 0 else 0
        results.append({'feature': feat, 'n': len(vals_s1), 'mean_s1': np.mean(x),
                       'mean_s2': np.mean(y), 'pearson_r': r_val, 'icc': icc, 'cohen_d': d})

    rdf = pd.DataFrame(results)
    print(f"\n  {'Feature':<30} {'N':>5} {'S1':>7} {'S2':>7} {'r':>6} {'ICC':>6} {'d':>6}")
    print(f"  {'-'*70}")
    for band in BAND_ORDER:
        band_df = rdf[rdf['feature'].str.startswith(band + '_')]
        for _, row in band_df.sort_values('feature').iterrows():
            print(f"  {row['feature']:<30} {row['n']:>5} {row['mean_s1']:>+6.0f}% {row['mean_s2']:>+6.0f}%"
                  f" {row['pearson_r']:>+.3f} {row['icc']:>+.3f} {row['cohen_d']:>+.3f}")

    print(f"\n  Per-band ICC summary:")
    for band in BAND_ORDER:
        band_df = rdf[rdf['feature'].str.startswith(band + '_')]
        if len(band_df) == 0:
            continue
        print(f"    {band:<12}: median ICC = {band_df['icc'].median():+.3f}  median r = {band_df['pearson_r'].median():+.3f}")

    print(f"\n  Overall: median ICC = {rdf['icc'].median():+.3f}, median r = {rdf['pearson_r'].median():+.3f}")
    rdf.to_csv(os.path.join(OUT_DIR, 'test_retest_reliability.csv'), index=False)

    # Group profile stability: ses-1 vs ses-2 enrichment correlation per band
    print(f"\n  Group profile stability (ses-1 vs ses-2):")
    all_positions = POS_NAMES + ['boundary_hi']
    ses1_peaks, _ = load_peaks(ses1_dir)
    ses2_peaks, _ = load_peaks(ses2_dir)
    if ses1_peaks is not None and ses2_peaks is not None:
        s1_enr = compute_enrichment(ses1_peaks)
        s2_enr = compute_enrichment(ses2_peaks)
        for band in BAND_ORDER:
            v1 = [int(r['enrichment_pct']) for _, r in s1_enr[s1_enr['band']==band].iterrows() if not np.isnan(r['enrichment_pct'])]
            v2 = [int(r['enrichment_pct']) for _, r in s2_enr[s2_enr['band']==band].iterrows() if not np.isnan(r['enrichment_pct'])]
            if len(v1) == len(v2) and len(v1) >= 3:
                r = np.corrcoef(v1, v2)[0, 1]
                print(f"    {band:<12s}: r={r:.3f}")

    # Ses-2 2×2 replication
    ses2_conditions = {
        'EC-pre':  'dortmund_EC_pre_ses2',
        'EO-pre':  'dortmund_EO_pre_ses2',
        'EC-post': 'dortmund_EC_post_ses2',
        'EO-post': 'dortmund_EO_post_ses2',
    }
    ses2_results = {}
    for cond, subdir in ses2_conditions.items():
        path = os.path.join(NEW_PEAK_BASE, subdir)
        peaks, n_sub = load_peaks(path)
        if peaks is not None:
            ses2_results[cond] = compute_enrichment(peaks)

    if len(ses2_results) >= 4:
        print(f"\n  Ses-2 2×2 replication (cross-session profile correlations):")
        ses1_conds = {'EC-pre': 'dortmund', 'EO-pre': 'dortmund_EO_pre',
                      'EC-post': 'dortmund_EC_post', 'EO-post': 'dortmund_EO_post'}
        ses1_results = {}
        for cond, subdir in ses1_conds.items():
            path = os.path.join(NEW_PEAK_BASE, subdir)
            peaks, _ = load_peaks(path)
            if peaks is not None:
                ses1_results[cond] = compute_enrichment(peaks)

        print(f"  {'Band':<12s}", end='')
        for cond in ses2_conditions:
            print(f" {cond:>8s}", end='')
        print()
        for band in BAND_ORDER:
            print(f"  {band:<12s}", end='')
            for cond in ses2_conditions:
                if cond in ses2_results and cond in ses1_results:
                    v1 = [int(r['enrichment_pct']) for _, r in ses1_results[cond][ses1_results[cond]['band']==band].iterrows() if not np.isnan(r['enrichment_pct'])]
                    v2 = [int(r['enrichment_pct']) for _, r in ses2_results[cond][ses2_results[cond]['band']==band].iterrows() if not np.isnan(r['enrichment_pct'])]
                    if len(v1) == len(v2) and len(v1) >= 3:
                        r = np.corrcoef(v1, v2)[0, 1]
                        print(f" {r:>8.3f}", end='')
                    else:
                        print(f"      —", end='')
                else:
                    print(f"      —", end='')
            print()

        # EC-EO delta pattern stability across sessions
        print(f"\n  EC→EO delta pattern correlation (ses-1 vs ses-2):")
        for band in BAND_ORDER:
            if all(c in ses1_results for c in ['EC-pre', 'EO-pre']) and all(c in ses2_results for c in ['EC-pre', 'EO-pre']):
                s1_ec = [int(r['enrichment_pct']) for _, r in ses1_results['EC-pre'][ses1_results['EC-pre']['band']==band].iterrows() if not np.isnan(r['enrichment_pct'])]
                s1_eo = [int(r['enrichment_pct']) for _, r in ses1_results['EO-pre'][ses1_results['EO-pre']['band']==band].iterrows() if not np.isnan(r['enrichment_pct'])]
                s2_ec = [int(r['enrichment_pct']) for _, r in ses2_results['EC-pre'][ses2_results['EC-pre']['band']==band].iterrows() if not np.isnan(r['enrichment_pct'])]
                s2_eo = [int(r['enrichment_pct']) for _, r in ses2_results['EO-pre'][ses2_results['EO-pre']['band']==band].iterrows() if not np.isnan(r['enrichment_pct'])]
                if len(s1_ec) == len(s1_eo) == len(s2_ec) == len(s2_eo) and len(s1_ec) >= 3:
                    delta1 = [a - b for a, b in zip(s1_ec, s1_eo)]
                    delta2 = [a - b for a, b in zip(s2_ec, s2_eo)]
                    r = np.corrcoef(delta1, delta2)[0, 1]
                    print(f"    {band:<12s}: r={r:.3f}")

    # Age does NOT predict 5-year change
    demo_path = '/Volumes/T9/dortmund_data/participants.tsv'
    if os.path.exists(demo_path):
        from scipy import stats as sp_stats
        demo = pd.read_csv(demo_path, sep='\t')
        age_map = dict(zip(demo['participant_id'], demo['age']))
        change_results = []
        for feat in sorted(all_features):
            ages_f, deltas_f = [], []
            for sub_id in matched:
                v1 = data_s1[sub_id].get(feat, np.nan)
                v2 = data_s2[sub_id].get(feat, np.nan)
                age = age_map.get(sub_id, np.nan)
                if not np.isnan(v1) and not np.isnan(v2) and not np.isnan(age):
                    ages_f.append(age)
                    deltas_f.append(v2 - v1)
            if len(ages_f) < 30:
                continue
            rho, p = sp_stats.spearmanr(ages_f, deltas_f)
            change_results.append({'feature': feat, 'rho': rho, 'p': p})
        if change_results:
            from statsmodels.stats.multitest import multipletests as mt
            cdf = pd.DataFrame(change_results)
            rej, pfdr, _, _ = mt(cdf['p'].values, method='fdr_bh', alpha=0.05)
            cdf['p_fdr'] = pfdr
            n_sig = sum(rej)
            print(f"\n  Age predicts 5-year change? {n_sig} FDR survivors across {len(cdf)} tests")


# =========================================================================
# STEP 9: COMPARE (side-by-side old vs new enrichment)
# =========================================================================

def run_compare():
    """Run enrichment on BOTH old and new extractions and produce comparison."""
    print("=" * 70)
    print("  STEP 9: COMPARISON old vs v2 (all 9 EC datasets)")
    print("=" * 70)

    all_positions = POS_NAMES + ['boundary_hi']
    csv_rows = []

    for name, subdir in EC_DATASETS.items():
        new_path = os.path.join(NEW_PEAK_BASE, subdir)
        old_path = os.path.join(OLD_PEAK_BASE, subdir)
        new_peaks, n_new = load_peaks(new_path)
        old_peaks, n_old = load_peaks(old_path)

        if new_peaks is None:
            print(f"  {name}: no new data")
            continue

        new_enr = compute_enrichment(new_peaks)
        old_enr = compute_enrichment(old_peaks) if old_peaks is not None else None

        for band in BAND_ORDER:
            for pos in all_positions:
                r_new = new_enr[(new_enr['band'] == band) & (new_enr['position'] == pos)]
                v_new = int(r_new.iloc[0]['enrichment_pct']) if (not r_new.empty and not np.isnan(r_new.iloc[0]['enrichment_pct'])) else None
                v_old = None
                if old_enr is not None:
                    r_old = old_enr[(old_enr['band'] == band) & (old_enr['position'] == pos)]
                    v_old = int(r_old.iloc[0]['enrichment_pct']) if (not r_old.empty and not np.isnan(r_old.iloc[0]['enrichment_pct'])) else None

                csv_rows.append({
                    'dataset': name, 'band': band, 'position': pos,
                    'old_f0_783': v_old, 'new_v2': v_new,
                    'delta': (v_new - v_old) if (v_new is not None and v_old is not None) else None,
                })

    df = pd.DataFrame(csv_rows)
    df.to_csv(os.path.join(OUT_DIR, 'enrichment_comparison_full.csv'), index=False)

    flips = df[((df['old_f0_783'] > 5) & (df['new_v2'] < -5)) |
               ((df['old_f0_783'] < -5) & (df['new_v2'] > 5))].dropna(subset=['delta'])
    print(f"\n  Sign flips (>±5%): {len(flips)}")
    for _, r in flips.iterrows():
        print(f"    {r['dataset']:<12s} {r['band']:<12s} {r['position']:<16s} {r['old_f0_783']:+d}% → {r['new_v2']:+d}%")

    print(f"\n  Per-band mean profile correlation:")
    for band in BAND_ORDER:
        bdf = df[df['band'] == band].dropna(subset=['delta'])
        means = bdf.groupby('position')[['old_f0_783', 'new_v2']].mean().dropna()
        if len(means) >= 3:
            r = np.corrcoef(means['old_f0_783'], means['new_v2'])[0, 1]
            mae = (means['new_v2'] - means['old_f0_783']).abs().mean()
            print(f"    {band:<12s}  r={r:.3f}  MAE={mae:.1f}pp")

    print(f"\n  Saved: {OUT_DIR}/enrichment_comparison_full.csv")


# =========================================================================
# STEP 10: HBN CROSS-RELEASE CONSISTENCY
# =========================================================================

def run_hbn_cross_release():
    print("=" * 70)
    print("  STEP 10: HBN Cross-Release Consistency")
    print("=" * 70)

    all_positions = POS_NAMES + ['boundary_hi']
    release_enr = {}

    for release in HBN_RELEASES:
        path = os.path.join(NEW_PEAK_BASE, f'hbn_{release}')
        peaks, n_sub = load_peaks(path)
        if peaks is None:
            print(f"  {release}: NO DATA")
            continue
        release_enr[release] = compute_enrichment(peaks)
        print(f"  {release}: {n_sub} subjects")

    if len(release_enr) < 2:
        print("  Need >=2 releases. Skipping.")
        return

    csv_rows = []
    for band in BAND_ORDER:
        print(f"\n  {band.upper()}")
        n_agree = 0; n_conflict = 0
        for pos in all_positions:
            vals = {}
            for release in HBN_RELEASES:
                if release not in release_enr:
                    continue
                r = release_enr[release]
                row = r[(r['band'] == band) & (r['position'] == pos)]
                if not row.empty and not np.isnan(row.iloc[0]['enrichment_pct']):
                    vals[release] = int(row.iloc[0]['enrichment_pct'])

            if len(vals) >= 2:
                signs = [1 if v > 5 else (-1 if v < -5 else 0) for v in vals.values()]
                agree = all(s >= 0 for s in signs) or all(s <= 0 for s in signs)
                if agree:
                    n_agree += 1
                else:
                    n_conflict += 1
                sd = np.std(list(vals.values()), ddof=0)
                pos_d = 'bnd_hi' if pos == 'boundary_hi' else pos
                v_strs = [f'{vals.get(r, 0):+d}' for r in HBN_RELEASES if r in vals]
                print(f"    {pos_d:<16s} {', '.join(v_strs):>30s}  SD={sd:.1f}  {'✓' if agree else '✗'}")

            csv_rows.append({'band': band, 'position': pos, **vals})

        print(f"    Agree: {n_agree}, Conflict: {n_conflict}")

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, 'hbn_cross_release.csv'), index=False)


# =========================================================================
# STEP 11: EO COGNITIVE REPLICATION
# =========================================================================

def run_cognitive_eo():
    print("=" * 70)
    print("  STEP 11: Per-Subject Cognitive Correlations (LEMON EO)")
    print("=" * 70)

    peak_dir = os.path.join(NEW_PEAK_BASE, 'lemon_EO')
    df = load_subject_enrichments(peak_dir)
    print(f"  Subjects: {len(df)}")

    for test_name, (filename, col) in COG_TESTS.items():
        path = os.path.join(COG_DIR, filename)
        if not os.path.exists(path):
            continue
        cog_df = pd.read_csv(path)
        cog_df[col] = pd.to_numeric(cog_df[col], errors='coerce')
        cog_map = dict(zip(cog_df['ID'], cog_df[col]))
        df[f'cog_{test_name}'] = df['subject'].map(cog_map)

    enrich_cols = get_enrich_cols(df)
    cog_cols = [c for c in df.columns if c.startswith('cog_')]
    print(f"  Enrichment features: {len(enrich_cols)}, Cognitive tests: {len(cog_cols)}")

    rdf = run_correlations(df, enrich_cols, cog_cols, target_prefix='cog_')
    print_correlation_summary(rdf, 'Cognitive EO')
    rdf.to_csv(os.path.join(OUT_DIR, 'cognitive_correlations_eo.csv'), index=False)


# =========================================================================
# STEP 12: ADULT vs PEDIATRIC COMPARISON
# =========================================================================

def run_adult_vs_pediatric():
    print("=" * 70)
    print("  STEP 12: Adult vs Pediatric Comparison")
    print("=" * 70)

    adult_datasets = ['eegmmidb', 'lemon', 'dortmund', 'chbmp']
    pediatric_datasets = [f'hbn_{r}' for r in HBN_RELEASES]
    all_positions = POS_NAMES + ['boundary_hi']

    # Load and pool
    adult_peaks_list = []
    for name in adult_datasets:
        path = os.path.join(NEW_PEAK_BASE, name)
        peaks, n = load_peaks(path)
        if peaks is not None:
            adult_peaks_list.append(peaks)
            print(f"  Adult: {name} ({n} subjects)")

    ped_peaks_list = []
    for name in pediatric_datasets:
        path = os.path.join(NEW_PEAK_BASE, name)
        peaks, n = load_peaks(path)
        if peaks is not None:
            ped_peaks_list.append(peaks)
            print(f"  Pediatric: {name} ({n} subjects)")

    if not adult_peaks_list or not ped_peaks_list:
        print("  Missing data. Skipping.")
        return

    adult_peaks = pd.concat(adult_peaks_list, ignore_index=True)
    ped_peaks = pd.concat(ped_peaks_list, ignore_index=True)

    adult_enr = compute_enrichment(adult_peaks)
    ped_enr = compute_enrichment(ped_peaks)

    print(f"\n  Adult: {len(adult_peaks):,} peaks, Pediatric: {len(ped_peaks):,} peaks")

    csv_rows = []
    for band in BAND_ORDER:
        ba = adult_enr[adult_enr['band'] == band]
        bp = ped_enr[ped_enr['band'] == band]

        vals_a, vals_p = [], []
        print(f"\n  {band.upper()}")
        print(f"  {'Position':<16s} {'Adult':>7s} {'Pedi':>7s} {'Δ':>6s}")
        print(f"  {'-'*38}")
        for pos in all_positions:
            ra = ba[ba['position'] == pos]
            rp = bp[bp['position'] == pos]
            va = int(ra.iloc[0]['enrichment_pct']) if not ra.empty and not np.isnan(ra.iloc[0]['enrichment_pct']) else None
            vp = int(rp.iloc[0]['enrichment_pct']) if not rp.empty and not np.isnan(rp.iloc[0]['enrichment_pct']) else None
            delta = (vp - va) if va is not None and vp is not None else None
            pos_d = 'bnd_hi' if pos == 'boundary_hi' else pos
            print(f"  {pos_d:<16s} {va:>+6d}% {vp:>+6d}% {delta:>+5d}" if va is not None and vp is not None else f"  {pos_d:<16s}     —      —     —")
            if va is not None: vals_a.append(va)
            if vp is not None: vals_p.append(vp)
            csv_rows.append({'band': band, 'position': pos, 'adult': va, 'pediatric': vp, 'delta': delta})

        if len(vals_a) == len(vals_p) and len(vals_a) >= 3:
            r = np.corrcoef(vals_a, vals_p)[0, 1]
            print(f"  Profile correlation: r={r:.3f}")

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, 'adult_vs_pediatric.csv'), index=False)


# =========================================================================
# STEP 13: HBN PER-RELEASE REPLICATION OF AGE EFFECTS
# =========================================================================

def run_hbn_per_release():
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    print("=" * 70)
    print("  STEP 13: HBN Per-Release Replication of Age Effects")
    print("=" * 70)

    demo_rows = []
    for release in HBN_RELEASES:
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if os.path.exists(tsv):
            d = pd.read_csv(tsv, sep='\t')
            d['release'] = release
            demo_rows.append(d)
    demo = pd.concat(demo_rows, ignore_index=True) if demo_rows else pd.DataFrame()

    all_release_rhos = {}
    for release in HBN_RELEASES:
        peak_dir = os.path.join(NEW_PEAK_BASE, f'hbn_{release}')
        peak_files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
        if not peak_files:
            continue

        rows = []
        for f in peak_files:
            sub_id = os.path.basename(f).replace('_peaks.csv', '')
            peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
            enrich = per_subject_enrichment(peaks)
            enrich['subject'] = sub_id
            match = demo[demo['participant_id'] == sub_id]
            if len(match) > 0:
                enrich['age'] = match.iloc[0]['age']
            rows.append(enrich)

        df = pd.DataFrame(rows)
        enrich_cols = get_enrich_cols(df)
        age_valid = df['age'].notna().sum()

        rhos = {}
        for col in enrich_cols:
            valid = df[['age', col]].dropna()
            if len(valid) < 30:
                continue
            rho, p = stats.spearmanr(valid['age'], valid[col])
            rhos[col] = rho

        all_release_rhos[release] = rhos
        n_sig = 0
        if rhos:
            rho_df = pd.DataFrame([{'feature': k, 'rho': v} for k, v in rhos.items()])
            # Quick FDR
            ps = [stats.spearmanr(df[['age', f]].dropna()['age'], df[['age', f]].dropna()[f])[1]
                  for f in rhos.keys() if len(df[['age', f]].dropna()) >= 30]
            if ps:
                rej, _, _, _ = multipletests(ps, method='fdr_bh', alpha=0.05)
                n_sig = sum(rej)
        print(f"  {release}: N={age_valid}, FDR survivors: {n_sig}")

    # Cross-release correlation of rhos
    if len(all_release_rhos) >= 2:
        releases = list(all_release_rhos.keys())
        print(f"\n  Cross-release correlation of age rhos:")
        for i in range(len(releases)):
            for j in range(i + 1, len(releases)):
                r1, r2 = releases[i], releases[j]
                shared = set(all_release_rhos[r1].keys()) & set(all_release_rhos[r2].keys())
                if len(shared) < 5:
                    continue
                v1 = [all_release_rhos[r1][k] for k in shared]
                v2 = [all_release_rhos[r2][k] for k in shared]
                r = np.corrcoef(v1, v2)[0, 1]
                print(f"    {r1} vs {r2}: r={r:.3f} (N features={len(shared)})")

    # Save
    rows = []
    for release, rhos in all_release_rhos.items():
        for feat, rho in rhos.items():
            rows.append({'release': release, 'feature': feat, 'rho': rho})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'hbn_per_release_age_rhos.csv'), index=False)


# =========================================================================
# STEP 14: LIFESPAN TRAJECTORY (HBN + Dortmund)
# =========================================================================

def run_lifespan():
    from scipy import stats

    print("=" * 70)
    print("  STEP 14: Lifespan Trajectory (HBN development + Dortmund aging)")
    print("=" * 70)

    # Load HBN age rhos
    hbn_path = os.path.join(OUT_DIR, 'hbn_age_correlations.csv')
    dort_path = os.path.join(OUT_DIR, 'dortmund_age_correlations.csv')
    if not os.path.exists(hbn_path) or not os.path.exists(dort_path):
        print("  Run hbn_age and dortmund_age first.")
        return

    hbn = pd.read_csv(hbn_path)
    dort = pd.read_csv(dort_path)

    # Match features
    hbn_dict = dict(zip(hbn['feature'], hbn['rho']))
    dort_dict = dict(zip(dort['feature'], dort['rho']))
    shared = sorted(set(hbn_dict.keys()) & set(dort_dict.keys()))

    hbn_rhos = [hbn_dict[f] for f in shared]
    dort_rhos = [dort_dict[f] for f in shared]

    r_cross = np.corrcoef(hbn_rhos, dort_rhos)[0, 1]
    print(f"  Cross-dataset age-rho correlation: r={r_cross:.3f} (N features={len(shared)})")

    # Classify patterns
    hbn_sig = set(hbn[hbn['significant']]['feature'].values) if 'significant' in hbn.columns else set()
    dort_sig = set(dort[dort['significant']]['feature'].values) if 'significant' in dort.columns else set()
    both_sig = hbn_sig & dort_sig

    n_inverted_u = 0; n_u_shape = 0; n_mono = 0; n_same = 0
    rows = []
    for f in shared:
        h, d = hbn_dict[f], dort_dict[f]
        if f in both_sig:
            if h > 0 and d < 0:
                pattern = 'inverted-U'
                n_inverted_u += 1
            elif h < 0 and d > 0:
                pattern = 'U-shape'
                n_u_shape += 1
            elif h > 0 and d > 0:
                pattern = 'monotonic-up'
                n_mono += 1
            else:
                pattern = 'monotonic-down'
                n_mono += 1
        else:
            pattern = 'not both significant'
        rows.append({'feature': f, 'hbn_rho': h, 'dort_rho': d, 'pattern': pattern})

    print(f"\n  Jointly significant features: {len(both_sig)}")
    print(f"    Inverted-U (↑dev, ↓aging): {n_inverted_u}")
    print(f"    U-shape (↓dev, ↑aging):    {n_u_shape}")
    print(f"    Monotonic:                  {n_mono}")

    # Opposite direction count
    n_opposite = sum(1 for f in both_sig if hbn_dict[f] * dort_dict[f] < 0)
    print(f"    Opposite direction: {n_opposite}/{len(both_sig)}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'lifespan_trajectory.csv'), index=False)

    # Three-dataset validation: include LEMON age rhos
    lemon_path = os.path.join(OUT_DIR, 'lemon_age_correlations.csv')
    if os.path.exists(lemon_path):
        lemon = pd.read_csv(lemon_path)
        lemon_dict = dict(zip(lemon['feature'], lemon['rho']))
        lemon_sig = set(lemon[lemon['significant']]['feature'].values) if 'significant' in lemon.columns else set()

        # Cross-dataset correlations of age rhos
        print(f"\n  Three-Dataset Validation:")
        for name_a, dict_a, name_b, dict_b in [
            ('HBN', hbn_dict, 'Dortmund', dort_dict),
            ('LEMON', lemon_dict, 'Dortmund', dort_dict),
            ('LEMON', lemon_dict, 'HBN', hbn_dict),
        ]:
            s = sorted(set(dict_a.keys()) & set(dict_b.keys()))
            if len(s) >= 5:
                r = np.corrcoef([dict_a[f] for f in s], [dict_b[f] for f in s])[0, 1]
                print(f"    {name_a} vs {name_b}: r={r:+.3f} (N={len(s)} features)")

        # Features significant in all 3 datasets
        three_sig = hbn_sig & dort_sig & lemon_sig
        print(f"    Features FDR-significant in ALL 3 datasets: {len(three_sig)}")
        for f in sorted(three_sig):
            print(f"      {f:<30s}  HBN={hbn_dict.get(f,0):+.3f}  LEMON={lemon_dict.get(f,0):+.3f}  Dort={dort_dict.get(f,0):+.3f}")

    # Cognition × Age × Enrichment Triangle
    cog_path = os.path.join(OUT_DIR, 'cognitive_correlations.csv')
    if os.path.exists(cog_path):
        cog = pd.read_csv(cog_path)
        lps = cog[cog['target'] == 'LPS']
        if len(lps) > 0:
            print(f"\n  Cognition × Age Triangle (LPS features that are also age-significant):")
            lps_dict = dict(zip(lps['feature'], lps['rho']))
            lps_sig = set(lps[lps['significant']]['feature'].values) if 'significant' in lps.columns else set()
            age_feats = hbn_sig | dort_sig | (lemon_sig if os.path.exists(lemon_path) else set())
            triangle = lps_sig & age_feats
            if triangle:
                print(f"    {'Feature':<30s} {'LPS':>6s} {'HBN':>6s} {'Dort':>6s} {'Opposite?'}")
                for f in sorted(triangle):
                    lps_r = lps_dict.get(f, 0)
                    hbn_r = hbn_dict.get(f, 0)
                    dort_r = dort_dict.get(f, 0)
                    opposite = 'YES' if (lps_r * dort_r < 0) else 'no'
                    print(f"    {f:<30s} {lps_r:>+.3f} {hbn_r:>+.3f} {dort_r:>+.3f} {opposite}")


# =========================================================================
# STEP 15: CROSS-BAND COUPLING
# =========================================================================

def run_cross_band_coupling():
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    print("=" * 70)
    print("  STEP 15: Cross-Band Coupling (individual differences)")
    print("=" * 70)

    datasets = [
        ('LEMON', os.path.join(NEW_PEAK_BASE, 'lemon')),
        ('Dortmund', os.path.join(NEW_PEAK_BASE, 'dortmund')),
        ('HBN', None),  # special: pool releases
    ]

    for label, peak_dir in datasets:
        if label == 'HBN':
            dfs = []
            for release in HBN_RELEASES:
                path = os.path.join(NEW_PEAK_BASE, f'hbn_{release}')
                d = load_subject_enrichments(path)
                if len(d) > 0:
                    dfs.append(d)
            if not dfs:
                continue
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = load_subject_enrichments(peak_dir)

        enrich_cols = get_enrich_cols(df)
        print(f"\n  {label}: {len(df)} subjects")

        # Comprehensive cross-band coupling: all band-pair combinations of key metrics
        key_metrics = ['boundary', 'attractor', 'noble_1', 'mountain', 'ushape']
        cross_results = []
        for band_a in BAND_ORDER:
            for band_b in BAND_ORDER:
                if band_a >= band_b:
                    continue  # upper triangle only
                for met_a in key_metrics:
                    for met_b in key_metrics:
                        col_a = f'{band_a}_{met_a}'
                        col_b = f'{band_b}_{met_b}'
                        if col_a not in df.columns or col_b not in df.columns:
                            continue
                        valid = df[[col_a, col_b]].dropna()
                        if len(valid) < 30:
                            continue
                        rho, p = stats.spearmanr(valid[col_a], valid[col_b])
                        cross_results.append({'band_a': band_a, 'metric_a': met_a,
                                            'band_b': band_b, 'metric_b': met_b,
                                            'rho': rho, 'p': p, 'n': len(valid),
                                            'abs_rho': abs(rho)})

        if cross_results:
            cdf = pd.DataFrame(cross_results)
            rej, pfdr, _, _ = multipletests(cdf['p'].values, method='fdr_bh', alpha=0.05)
            cdf['p_fdr'] = pfdr
            cdf['significant'] = rej
            n_sig = cdf['significant'].sum()
            print(f"    Total cross-band tests: {len(cdf)}, FDR survivors: {n_sig}")

            top = cdf.nlargest(10, 'abs_rho')
            for _, r in top.iterrows():
                sig = '*' if r['p_fdr'] < 0.05 else ''
                print(f"    {r['band_a']}_{r['metric_a']:<12s} × {r['band_b']}_{r['metric_b']:<12s}  rho={r['rho']:+.3f}  p_FDR={r['p_fdr']:.4f} {sig}")

            cdf.to_csv(os.path.join(OUT_DIR, f'cross_band_coupling_{label.lower()}.csv'), index=False)

    # Coupling stability: test-retest of coupling within Dortmund
    print(f"\n  --- Coupling Stability (Dortmund EC-pre vs EC-post) ---")
    pre_sub = load_subject_enrichments(os.path.join(NEW_PEAK_BASE, 'dortmund'))
    post_sub = load_subject_enrichments(os.path.join(NEW_PEAK_BASE, 'dortmund_EC_post'))

    if len(pre_sub) > 0 and len(post_sub) > 0:
        from scipy import stats as sp_stats
        matched_c = sorted(set(pre_sub['subject']) & set(post_sub['subject']))
        pre_idx = pre_sub.set_index('subject')
        post_idx = post_sub.set_index('subject')

        coupling_pairs = [('alpha_boundary', 'beta_low_attractor'),
                         ('alpha_noble_1', 'beta_low_ushape')]
        for col_a, col_b in coupling_pairs:
            if col_a not in pre_idx.columns or col_b not in pre_idx.columns:
                continue
            pre_products, post_products = [], []
            for sub_id in matched_c:
                va1 = pre_idx.at[sub_id, col_a]; vb1 = pre_idx.at[sub_id, col_b]
                va2 = post_idx.at[sub_id, col_a]; vb2 = post_idx.at[sub_id, col_b]
                if not any(np.isnan(x) for x in [va1, vb1, va2, vb2]):
                    pre_products.append(va1 * vb1)
                    post_products.append(va2 * vb2)
            if len(pre_products) >= 30:
                r, p = sp_stats.pearsonr(pre_products, post_products)
                print(f"    {col_a} × {col_b}: coupling r(pre,post)={r:+.3f} p={p:.2e} N={len(pre_products)}")

    # 5-year coupling stability (ses-1 vs ses-2)
    ses2_dir = os.path.join(NEW_PEAK_BASE, 'dortmund_EC_pre_ses2')
    if os.path.exists(ses2_dir):
        print(f"\n  --- Coupling Stability (Dortmund ses-1 vs ses-2, ~5 years) ---")
        ses1_sub = load_subject_enrichments(os.path.join(NEW_PEAK_BASE, 'dortmund'))
        ses2_sub = load_subject_enrichments(ses2_dir)

        if len(ses1_sub) > 0 and len(ses2_sub) > 0:
            from scipy import stats as sp_stats
            matched_5y = sorted(set(ses1_sub['subject']) & set(ses2_sub['subject']))
            s1_idx = ses1_sub.set_index('subject')
            s2_idx = ses2_sub.set_index('subject')

            coupling_pairs_5y = [('alpha_boundary', 'beta_low_attractor'),
                                ('alpha_noble_1', 'beta_low_ushape'),
                                ('alpha_noble_1', 'beta_low_ramp_depth')]
            for col_a, col_b in coupling_pairs_5y:
                if col_a not in s1_idx.columns or col_b not in s1_idx.columns:
                    continue
                s1_products, s2_products = [], []
                for sub_id in matched_5y:
                    va1 = s1_idx.at[sub_id, col_a]; vb1 = s1_idx.at[sub_id, col_b]
                    va2 = s2_idx.at[sub_id, col_a]; vb2 = s2_idx.at[sub_id, col_b]
                    if not any(np.isnan(x) for x in [va1, vb1, va2, vb2]):
                        s1_products.append(va1 * vb1)
                        s2_products.append(va2 * vb2)
                if len(s1_products) >= 30:
                    r, p = sp_stats.pearsonr(s1_products, s2_products)
                    print(f"    {col_a} × {col_b}: coupling r(ses1,ses2)={r:+.3f} p={p:.2e} N={len(s1_products)}")


# =========================================================================
# STEP 16: WITHIN-SESSION RELIABILITY (Dortmund EC-pre vs EC-post)
# =========================================================================

def run_within_session():
    from scipy import stats

    print("=" * 70)
    print("  STEP 16: Within-Session Reliability (EC-pre vs EC-post)")
    print("=" * 70)

    pre_dir = os.path.join(NEW_PEAK_BASE, 'dortmund')
    post_dir = os.path.join(NEW_PEAK_BASE, 'dortmund_EC_post')

    pre_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                 for f in glob.glob(os.path.join(pre_dir, '*_peaks.csv'))}
    post_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                  for f in glob.glob(os.path.join(post_dir, '*_peaks.csv'))}
    matched = sorted(set(pre_files.keys()) & set(post_files.keys()))
    print(f"  Matched subjects: {len(matched)}")

    data_pre, data_post = {}, {}
    for sub_id in matched:
        p1 = pd.read_csv(pre_files[sub_id], usecols=['freq', 'phi_octave'])
        p2 = pd.read_csv(post_files[sub_id], usecols=['freq', 'phi_octave'])
        data_pre[sub_id] = per_subject_enrichment(p1)
        data_post[sub_id] = per_subject_enrichment(p2)

    all_features = set()
    for d in list(data_pre.values()) + list(data_post.values()):
        all_features.update(k for k in d.keys() if not k.endswith('_n_peaks'))

    results = []
    for feat in sorted(all_features):
        v_pre, v_post = [], []
        for sub_id in matched:
            a = data_pre[sub_id].get(feat, np.nan)
            b = data_post[sub_id].get(feat, np.nan)
            if not np.isnan(a) and not np.isnan(b):
                v_pre.append(a); v_post.append(b)
        if len(v_pre) < 20:
            continue
        x, y = np.array(v_pre), np.array(v_post)
        r_val, _ = stats.pearsonr(x, y)
        icc = icc_21(x, y)
        results.append({'feature': feat, 'n': len(v_pre), 'pearson_r': r_val, 'icc': icc})

    rdf = pd.DataFrame(results)
    print(f"\n  Per-band within-session ICC:")
    for band in BAND_ORDER:
        bdf = rdf[rdf['feature'].str.startswith(band + '_')]
        if len(bdf) == 0:
            continue
        print(f"    {band:<12}: median ICC = {bdf['icc'].median():+.3f}  median r = {bdf['pearson_r'].median():+.3f}")
    print(f"\n  Overall: median ICC = {rdf['icc'].median():+.3f}")
    rdf.to_csv(os.path.join(OUT_DIR, 'within_session_reliability.csv'), index=False)

    # Cross-condition reliability (EC vs EO, same timepoint)
    eo_dir = os.path.join(NEW_PEAK_BASE, 'dortmund_EO_pre')
    if os.path.exists(eo_dir):
        eo_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                    for f in glob.glob(os.path.join(eo_dir, '*_peaks.csv'))}
        matched_eo = sorted(set(pre_files.keys()) & set(eo_files.keys()))
        print(f"\n  Cross-condition (EC vs EO, same timepoint): {len(matched_eo)} matched")

        data_eo = {}
        for sub_id in matched_eo:
            p = pd.read_csv(eo_files[sub_id], usecols=['freq', 'phi_octave'])
            data_eo[sub_id] = per_subject_enrichment(p)

        cc_results = []
        for feat in sorted(all_features):
            v_ec, v_eo = [], []
            for sub_id in matched_eo:
                a = data_pre[sub_id].get(feat, np.nan)
                b = data_eo[sub_id].get(feat, np.nan)
                if not np.isnan(a) and not np.isnan(b):
                    v_ec.append(a); v_eo.append(b)
            if len(v_ec) < 20:
                continue
            x, y = np.array(v_ec), np.array(v_eo)
            r_val, _ = stats.pearsonr(x, y)
            icc = icc_21(x, y)
            cc_results.append({'feature': feat, 'n': len(v_ec), 'pearson_r': r_val, 'icc': icc})

        cc_df = pd.DataFrame(cc_results)
        print(f"  Cross-condition median ICC = {cc_df['icc'].median():+.3f}, median r = {cc_df['pearson_r'].median():+.3f}")
        cc_df.to_csv(os.path.join(OUT_DIR, 'cross_condition_reliability.csv'), index=False)


# =========================================================================
# STEP 17: MEDICAL/METABOLIC MARKERS (LEMON)
# =========================================================================

def run_medical():
    print("=" * 70)
    print("  STEP 17: Medical/Metabolic Markers (LEMON)")
    print("=" * 70)

    peak_dir = os.path.join(NEW_PEAK_BASE, 'lemon')
    df = load_subject_enrichments(peak_dir)

    # Load medical data
    med_dir = '/Volumes/T9/lemon_data/behavioral/Behavioural_Data_MPILMBB_LEMON/Medical_LEMON'
    csvs = sorted(glob.glob(os.path.join(med_dir, '*.csv')))
    for fpath in csvs:
        name = os.path.basename(fpath).replace('.csv', '')
        try:
            mdf = pd.read_csv(fpath)
        except Exception:
            continue
        if 'ID' not in mdf.columns:
            continue
        cols = [c for c in mdf.columns if c != 'ID']
        for c in cols:
            mdf[c] = pd.to_numeric(mdf[c], errors='coerce')
            if mdf[c].notna().sum() < 50:
                continue
            key = f'med_{name}_{c}'
            df[key] = df['subject'].map(dict(zip(mdf['ID'], mdf[c])))

    enrich_cols = get_enrich_cols(df)
    med_cols = [c for c in df.columns if c.startswith('med_')]
    print(f"  Subjects: {len(df)}, Medical variables: {len(med_cols)}")

    if not med_cols:
        print("  No medical data found.")
        return

    rdf = run_correlations(df, enrich_cols, med_cols, target_prefix='med_')
    print_correlation_summary(rdf, 'Medical', top_n=15)
    rdf.to_csv(os.path.join(OUT_DIR, 'medical_correlations.csv'), index=False)


# =========================================================================
# STEP 18: HANDEDNESS
# =========================================================================

def run_handedness():
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    print("=" * 70)
    print("  STEP 18: Handedness × Enrichment")
    print("=" * 70)

    # HBN
    demo_rows = []
    for release in HBN_RELEASES:
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if os.path.exists(tsv):
            d = pd.read_csv(tsv, sep='\t')
            d['release'] = release
            demo_rows.append(d)

    if demo_rows:
        demo = pd.concat(demo_rows, ignore_index=True)
        rows = []
        for release in HBN_RELEASES:
            peak_dir = os.path.join(NEW_PEAK_BASE, f'hbn_{release}')
            for f in sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv'))):
                sub_id = os.path.basename(f).replace('_peaks.csv', '')
                peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
                enrich = per_subject_enrichment(peaks)
                enrich['subject'] = sub_id
                match = demo[demo['participant_id'] == sub_id]
                if len(match) > 0:
                    enrich['ehq_total'] = pd.to_numeric(match.iloc[0].get('ehq_total', np.nan), errors='coerce')
                rows.append(enrich)

        df = pd.DataFrame(rows)
        enrich_cols = get_enrich_cols(df)
        n_ehq = df['ehq_total'].notna().sum()
        print(f"  HBN: {len(df)} subjects, {n_ehq} with EHQ")

        if n_ehq > 50:
            rdf = run_correlations(df, enrich_cols, ['ehq_total'])
            n_sig = rdf['significant'].sum() if len(rdf) > 0 else 0
            print(f"  HBN EHQ continuous × enrichment: {len(rdf)} tests, {n_sig} FDR survivors")
            rdf.to_csv(os.path.join(OUT_DIR, 'hbn_handedness.csv'), index=False)

            # Dichotomized: strong left vs strong right
            strong_left = df[df['ehq_total'] < -50]
            strong_right = df[df['ehq_total'] > 50]
            print(f"  HBN strong left (EHQ<-50): {len(strong_left)}, strong right (EHQ>50): {len(strong_right)}")
            if len(strong_left) >= 20 and len(strong_right) >= 20:
                group_results = []
                for col in enrich_cols:
                    l_vals = strong_left[col].dropna()
                    r_vals = strong_right[col].dropna()
                    if len(l_vals) < 20 or len(r_vals) < 20:
                        continue
                    _, p = stats.mannwhitneyu(l_vals, r_vals, alternative='two-sided')
                    group_results.append({'feature': col, 'p': p})
                if group_results:
                    gdf = pd.DataFrame(group_results)
                    rej, pfdr, _, _ = multipletests(gdf['p'].values, method='fdr_bh', alpha=0.05)
                    gdf['p_fdr'] = pfdr
                    print(f"  Strong L vs R: {sum(rej)} FDR survivors across {len(gdf)} tests")

    # Dortmund
    dort_demo_path = '/Volumes/T9/dortmund_data/participants.tsv'
    if os.path.exists(dort_demo_path):
        dort_demo = pd.read_csv(dort_demo_path, sep='\t')
        df_dort = load_subject_enrichments(os.path.join(NEW_PEAK_BASE, 'dortmund'))
        if 'handedness' in dort_demo.columns:
            hand_map = dict(zip(dort_demo['participant_id'], dort_demo['handedness']))
            df_dort['handedness'] = df_dort['subject'].map(hand_map)
            left = df_dort[df_dort['handedness'] == 'left']
            right = df_dort[df_dort['handedness'] == 'right']
            print(f"  Dortmund: left={len(left)}, right={len(right)}")

            enrich_cols = get_enrich_cols(df_dort)
            sex_results = []
            for col in enrich_cols:
                l_vals = left[col].dropna()
                r_vals = right[col].dropna()
                if len(l_vals) < 20 or len(r_vals) < 20:
                    continue
                _, p = stats.mannwhitneyu(l_vals, r_vals, alternative='two-sided')
                sex_results.append({'feature': col, 'p': p})

            if sex_results:
                hdf = pd.DataFrame(sex_results)
                rej, pfdr, _, _ = multipletests(hdf['p'].values, method='fdr_bh', alpha=0.05)
                hdf['p_fdr'] = pfdr
                hdf['significant'] = rej
                print(f"  Dortmund L vs R: {len(hdf)} tests, {hdf['significant'].sum()} FDR survivors")


# =========================================================================
# STEP 19: SEX × AGE INTERACTION
# =========================================================================

def run_sex_age_interaction():
    from scipy import stats

    print("=" * 70)
    print("  STEP 19: Sex × Age Interaction")
    print("=" * 70)

    for label, load_fn in [('HBN', None), ('Dortmund', None)]:
        if label == 'HBN':
            demo_rows = []
            for release in HBN_RELEASES:
                tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
                if os.path.exists(tsv):
                    d = pd.read_csv(tsv, sep='\t')
                    d['release'] = release
                    demo_rows.append(d)
            if not demo_rows:
                continue
            demo = pd.concat(demo_rows, ignore_index=True)
            rows = []
            for release in HBN_RELEASES:
                peak_dir = os.path.join(NEW_PEAK_BASE, f'hbn_{release}')
                for f in sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv'))):
                    sub_id = os.path.basename(f).replace('_peaks.csv', '')
                    peaks = pd.read_csv(f, usecols=['freq', 'phi_octave'])
                    enrich = per_subject_enrichment(peaks)
                    enrich['subject'] = sub_id
                    match = demo[demo['participant_id'] == sub_id]
                    if len(match) > 0:
                        enrich['age'] = match.iloc[0]['age']
                        enrich['sex'] = match.iloc[0].get('sex', np.nan)
                    rows.append(enrich)
            df = pd.DataFrame(rows)
        else:
            df = load_subject_enrichments(os.path.join(NEW_PEAK_BASE, 'dortmund'))
            demo_path = '/Volumes/T9/dortmund_data/participants.tsv'
            if os.path.exists(demo_path):
                demo = pd.read_csv(demo_path, sep='\t')
                df['age'] = df['subject'].map(dict(zip(demo['participant_id'], demo['age'])))
                df['sex'] = df['subject'].map(dict(zip(demo['participant_id'], demo.get('sex', pd.Series(dtype=str)))))

        males = df[(df['sex'] == 'M') & df['age'].notna()]
        females = df[(df['sex'] == 'F') & df['age'].notna()]
        print(f"\n  {label}: M={len(males)}, F={len(females)}")

        enrich_cols = get_enrich_cols(df)
        n_sig = 0
        for col in enrich_cols:
            m_valid = males[['age', col]].dropna()
            f_valid = females[['age', col]].dropna()
            if len(m_valid) < 30 or len(f_valid) < 30:
                continue
            rho_m, _ = stats.spearmanr(m_valid['age'], m_valid[col])
            rho_f, _ = stats.spearmanr(f_valid['age'], f_valid[col])
            # Fisher z-test
            zm = np.arctanh(rho_m)
            zf = np.arctanh(rho_f)
            se = np.sqrt(1 / (len(m_valid) - 3) + 1 / (len(f_valid) - 3))
            z = (zm - zf) / se if se > 0 else 0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            if p < 0.05 / len(enrich_cols):
                n_sig += 1

        print(f"  Bonferroni-significant sex×age interactions: {n_sig}/{len(enrich_cols)}")


# =========================================================================
# STEP 20: STATE SENSITIVITY × AGE
# =========================================================================

def run_state_sensitivity_age():
    from scipy import stats

    print("=" * 70)
    print("  STEP 20: State Sensitivity × Age")
    print("=" * 70)

    for label, ec_dir, eo_dir, demo_path in [
        ('LEMON', 'lemon', 'lemon_EO', META_PATH),
        ('Dortmund', 'dortmund', 'dortmund_EO_pre', '/Volumes/T9/dortmund_data/participants.tsv'),
    ]:
        ec_sub = load_subject_enrichments(os.path.join(NEW_PEAK_BASE, ec_dir))
        eo_sub = load_subject_enrichments(os.path.join(NEW_PEAK_BASE, eo_dir))

        # Match subjects
        matched = sorted(set(ec_sub['subject']) & set(eo_sub['subject']))
        if len(matched) < 30:
            print(f"  {label}: only {len(matched)} matched subjects")
            continue

        ec_idx = ec_sub.set_index('subject')
        eo_idx = eo_sub.set_index('subject')
        enrich_cols = get_enrich_cols(ec_sub)

        # Load age
        if label == 'LEMON' and os.path.exists(demo_path):
            meta = pd.read_csv(demo_path)
            def _age_mid(s):
                try:
                    lo, hi = s.split('-')
                    return (float(lo) + float(hi)) / 2
                except Exception:
                    return np.nan
            meta['age_mid'] = meta['Age'].apply(_age_mid)
            age_map = dict(zip(meta['ID'], meta['age_mid']))
        elif label == 'Dortmund' and os.path.exists(demo_path):
            demo = pd.read_csv(demo_path, sep='\t')
            age_map = {k: float(v) if pd.notna(v) else np.nan
                       for k, v in zip(demo['participant_id'], demo['age'])}
        else:
            age_map = {}

        # Compute delta enrichment and correlate with age
        n_sig = 0
        total_tests = 0
        for col in enrich_cols:
            ages, deltas = [], []
            for sub in matched:
                if sub not in age_map:
                    continue
                age_val = age_map[sub]
                ec_val = ec_idx.at[sub, col] if sub in ec_idx.index and col in ec_idx.columns else np.nan
                eo_val = eo_idx.at[sub, col] if sub in eo_idx.index and col in eo_idx.columns else np.nan
                try:
                    if np.isnan(ec_val) or np.isnan(eo_val) or np.isnan(age_val):
                        continue
                except (TypeError, ValueError):
                    continue
                ages.append(age_map[sub])
                deltas.append(eo_val - ec_val)

            if len(ages) < 30:
                continue
            total_tests += 1
            rho, p = stats.spearmanr(ages, deltas)
            if p < 0.05 / max(1, len(enrich_cols)):
                n_sig += 1

        print(f"  {label}: {len(matched)} matched, {total_tests} tests, {n_sig} Bonferroni-sig state×age interactions")


# =========================================================================
# STEP 21: POWER SENSITIVITY SWEEP
# =========================================================================

def run_power_sensitivity():
    """Run enrichment at multiple power thresholds to test robustness."""
    print("=" * 70)
    print("  STEP 21: Power Sensitivity Sweep")
    print("=" * 70)

    thresholds = [0, 25, 50, 75]
    key_positions = {
        'alpha': ['boundary', 'attractor', 'noble_1', 'inv_noble_4', 'boundary_hi'],
        'beta_low': ['boundary', 'noble_4', 'attractor', 'noble_1', 'inv_noble_4', 'inv_noble_6', 'boundary_hi'],
        'theta': ['boundary', 'attractor', 'noble_1', 'inv_noble_3', 'boundary_hi'],
        'gamma': ['boundary', 'noble_3', 'noble_1', 'inv_noble_3', 'inv_noble_5'],
        'beta_high': ['boundary', 'attractor', 'noble_1', 'inv_noble_4', 'boundary_hi'],
    }

    # Use Dortmund (largest dataset) for the sweep
    print(f"\n  Dataset: Dortmund (N=608)")

    # Load ALL peaks once (no power filter)
    files = sorted(glob.glob(os.path.join(NEW_PEAK_BASE, 'dortmund', '*_peaks.csv')))
    first = pd.read_csv(files[0], nrows=1)
    has_power = 'power' in first.columns
    if not has_power:
        print("  No power column in peaks. Skipping.")
        return

    all_peaks = pd.concat([pd.read_csv(f, usecols=['freq', 'phi_octave', 'power']) for f in files], ignore_index=True)
    print(f"  Total peaks: {len(all_peaks):,}")

    csv_rows = []

    # Pooled enrichment at each threshold
    print(f"\n  --- Pooled Enrichment Across Thresholds ---")
    for band in BAND_ORDER:
        print(f"\n  {band.upper()}")
        positions = key_positions.get(band, ['attractor', 'noble_1'])
        header = f"  {'Position':<16s}" + "".join(f" {'p>'+str(t)+'%':>8s}" for t in thresholds)
        print(header)
        print(f"  {'-'*( 16 + 9*len(thresholds))}")

        for pos in positions:
            line = f"  {pos:<16s}"
            for t in thresholds:
                if t == 0:
                    filtered = all_peaks
                else:
                    filtered_parts = []
                    for octave in all_peaks['phi_octave'].unique():
                        bp = all_peaks[all_peaks.phi_octave == octave]
                        thresh = bp['power'].quantile(t / 100)
                        filtered_parts.append(bp[bp['power'] >= thresh])
                    filtered = pd.concat(filtered_parts, ignore_index=True)

                enr = compute_enrichment(filtered)
                row = enr[(enr['band'] == band) & (enr['position'] == pos)]
                v = int(row.iloc[0]['enrichment_pct']) if not row.empty and not np.isnan(row.iloc[0]['enrichment_pct']) else None
                line += f" {v:>+7d}%" if v is not None else "       —"
                csv_rows.append({'band': band, 'position': pos, 'threshold': t, 'enrichment': v})
            print(line)

    # Per-subject cognitive sensitivity (LEMON)
    print(f"\n  --- Cognitive (LPS) Sensitivity ---")
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    lemon_files = sorted(glob.glob(os.path.join(NEW_PEAK_BASE, 'lemon', '*_peaks.csv')))
    if lemon_files:
        # Load cognitive
        cog_scores = {}
        lps_path = os.path.join(COG_DIR, 'LPS/LPS.csv')
        if os.path.exists(lps_path):
            lps_df = pd.read_csv(lps_path)
            lps_df['LPS_1'] = pd.to_numeric(lps_df['LPS_1'], errors='coerce')
            cog_scores = dict(zip(lps_df['ID'], lps_df['LPS_1']))

        key_features = ['beta_low_mountain', 'beta_low_attractor', 'beta_low_ramp_depth',
                       'beta_low_asymmetry', 'alpha_peak_height', 'alpha_mountain']

        print(f"  {'Feature':<30s}" + "".join(f" {'p>'+str(t)+'%':>10s}" for t in thresholds))
        print(f"  {'-'*(30 + 11*len(thresholds))}")

        for t in thresholds:
            # Build per-subject enrichment at this threshold
            rows = []
            for f in lemon_files:
                sub_id = os.path.basename(f).replace('_peaks.csv', '')
                peaks = pd.read_csv(f, usecols=['freq', 'phi_octave', 'power'])
                # Filter by power
                if t > 0:
                    parts = []
                    for octave in peaks['phi_octave'].unique():
                        bp = peaks[peaks.phi_octave == octave]
                        if len(bp) >= 2:
                            thresh = bp['power'].quantile(t / 100)
                            parts.append(bp[bp['power'] >= thresh])
                        else:
                            parts.append(bp)
                    peaks = pd.concat(parts, ignore_index=True) if parts else peaks
                enrich = per_subject_enrichment(peaks)
                enrich['subject'] = sub_id
                if sub_id in cog_scores:
                    enrich['lps'] = cog_scores[sub_id]
                rows.append(enrich)

            df = pd.DataFrame(rows)
            # Store rhos for this threshold
            if t == thresholds[0]:
                feature_rhos = {feat: [] for feat in key_features}
            for feat in key_features:
                valid = df[['lps', feat]].dropna()
                if len(valid) >= 20:
                    rho, p = stats.spearmanr(valid['lps'], valid[feat])
                    feature_rhos[feat].append(f"{rho:+.3f}")
                else:
                    feature_rhos[feat].append("     —")

        for feat in key_features:
            print(f"  {feat:<30s}" + "".join(f" {r:>10s}" for r in feature_rhos[feat]))

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, 'power_sensitivity.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/power_sensitivity.csv")


# =========================================================================
# STEP 22: REPORT COMPARISON (old report numbers vs new results)
# =========================================================================

def run_report_comparison():
    """Compare key numbers from the old enrichment reanalysis report against new v2 results."""
    print("=" * 70)
    print("  STEP 21: Old Report vs New Results Comparison")
    print("=" * 70)

    results = []

    def check(section, claim, old_val, new_val, unit=''):
        delta = new_val - old_val if (old_val is not None and new_val is not None) else None
        results.append({'section': section, 'claim': claim, 'old': old_val, 'new': new_val,
                       'delta': delta, 'unit': unit})
        s_old = f'{old_val:+.0f}{unit}' if old_val is not None else '—'
        s_new = f'{new_val:+.0f}{unit}' if new_val is not None else '—'
        s_delta = f'{delta:+.0f}' if delta is not None else '—'
        changed = '***' if (delta is not None and abs(delta) > 20) else ''
        print(f"  {claim:<45s}  old={s_old:>8s}  new={s_new:>8s}  Δ={s_delta:>6s} {changed}")

    # --- ENRICHMENT (9-dataset means) ---
    enr_path = os.path.join(OUT_DIR, 'enrichment_9ds_comparison.csv')
    if os.path.exists(enr_path):
        enr = pd.read_csv(enr_path)
        # Compute new means across datasets
        new_cols = [c for c in enr.columns if c.endswith('_new')]
        old_cols = [c for c in enr.columns if c.endswith('_old')]

        def get_mean(df, band, pos, cols):
            row = df[(df['band'] == band) & (df['position'] == pos)]
            if row.empty:
                return None
            vals = [row.iloc[0][c] for c in cols if pd.notna(row.iloc[0].get(c))]
            return np.mean(vals) if vals else None

        print(f"\n  --- Enrichment (9-dataset mean) ---")
        # Old report key values (from impact report)
        old_enrichment = {
            ('beta_low', 'boundary'): +101, ('beta_low', 'noble_5'): -59,
            ('beta_low', 'attractor'): -21, ('beta_low', 'noble_1'): +2,
            ('beta_low', 'inv_noble_4'): +56, ('beta_low', 'boundary_hi'): +74,
            ('alpha', 'boundary'): -37, ('alpha', 'attractor'): +24,
            ('alpha', 'noble_1'): +25, ('alpha', 'boundary_hi'): -32,
            ('theta', 'boundary'): +47, ('theta', 'boundary_hi'): +38,
            ('beta_high', 'attractor'): -12, ('beta_high', 'inv_noble_4'): +12,
            ('gamma', 'inv_noble_3'): +27, ('gamma', 'inv_noble_5'): +61,
        }
        for (band, pos), old_val in old_enrichment.items():
            new_val = get_mean(enr, band, pos, new_cols)
            check('Enrichment', f'{band} {pos}', old_val, new_val, '%')

    # --- COGNITIVE ---
    cog_path = os.path.join(OUT_DIR, 'cognitive_correlations.csv')
    if os.path.exists(cog_path):
        print(f"\n  --- Cognitive (LEMON EC × LPS) ---")
        cog = pd.read_csv(cog_path)
        lps = cog[cog['target'] == 'LPS']

        old_cognitive = {
            'beta_low_mountain': -0.314,
            'beta_low_boundary': +0.312,
            'beta_low_inv_noble_1': -0.294,
            'beta_low_attractor': -0.284,
        }
        for feat, old_rho in old_cognitive.items():
            row = lps[lps['feature'] == feat]
            new_rho = row.iloc[0]['rho'] if not row.empty else None
            new_fdr = row.iloc[0]['p_fdr'] if not row.empty else None
            check('Cognitive', f'LPS × {feat} rho', old_rho * 1000, (new_rho * 1000) if new_rho else None)
            if new_fdr is not None:
                sig = 'YES' if new_fdr < 0.05 else 'no'
                print(f"    p_FDR={new_fdr:.4f} ({sig})")

        n_fdr = lps['significant'].sum() if 'significant' in lps.columns and len(lps) > 0 else 0
        check('Cognitive', 'Total FDR survivors (was 4)', 4, n_fdr)

    # --- HBN AGE ---
    hbn_age_path = os.path.join(OUT_DIR, 'hbn_age_correlations.csv')
    if os.path.exists(hbn_age_path):
        print(f"\n  --- HBN Developmental Trajectory ---")
        hbn_age = pd.read_csv(hbn_age_path)
        n_fdr = hbn_age['significant'].sum() if 'significant' in hbn_age.columns else 0
        check('HBN Age', 'FDR survivors (was 43/66)', 43, n_fdr)

        old_hbn_age = {
            'alpha_inv_noble_3': +0.302,
            'beta_low_noble_3': -0.230,
            'beta_low_ushape': +0.166,
            'gamma_noble_3': -0.196,
        }
        for feat, old_rho in old_hbn_age.items():
            row = hbn_age[hbn_age['feature'] == feat]
            new_rho = row.iloc[0]['rho'] if not row.empty else None
            check('HBN Age', f'{feat} rho', old_rho * 1000, (new_rho * 1000) if new_rho else None)

    # --- DORTMUND AGE ---
    dort_age_path = os.path.join(OUT_DIR, 'dortmund_age_correlations.csv')
    if os.path.exists(dort_age_path):
        print(f"\n  --- Dortmund Adult Aging ---")
        dort_age = pd.read_csv(dort_age_path)
        n_fdr = dort_age['significant'].sum() if 'significant' in dort_age.columns else 0
        check('Dort Age', 'FDR survivors (was 40/66)', 40, n_fdr)

    # --- PERSONALITY ---
    pers_path = os.path.join(OUT_DIR, 'personality_correlations.csv')
    if os.path.exists(pers_path):
        print(f"\n  --- Personality ---")
        pers = pd.read_csv(pers_path)
        n_fdr = pers['significant'].sum() if 'significant' in pers.columns else 0
        check('Personality', 'FDR survivors (was 0)', 0, n_fdr)

    # --- RELIABILITY ---
    rel_path = os.path.join(OUT_DIR, 'test_retest_reliability.csv')
    if os.path.exists(rel_path):
        print(f"\n  --- 5-Year Test-Retest ---")
        rel = pd.read_csv(rel_path)
        med_icc = rel['icc'].median()
        med_r = rel['pearson_r'].median()
        check('Reliability', 'Median ICC (was +0.42)', 420, int(med_icc * 1000))
        check('Reliability', 'Median r (was +0.42)', 420, int(med_r * 1000))

    # --- WITHIN SESSION ---
    ws_path = os.path.join(OUT_DIR, 'within_session_reliability.csv')
    if os.path.exists(ws_path):
        print(f"\n  --- Within-Session Reliability ---")
        ws = pd.read_csv(ws_path)
        med_icc = ws['icc'].median()
        check('Within-Ses', 'Median ICC (was +0.40)', 400, int(med_icc * 1000))

    # --- CROSS-BAND COUPLING ---
    cb_lemon = os.path.join(OUT_DIR, 'cross_band_coupling_lemon.csv')
    if os.path.exists(cb_lemon):
        print(f"\n  --- Cross-Band Coupling (LEMON) ---")
        cb = pd.read_csv(cb_lemon)
        row = cb[(cb['band_a'] == 'alpha') & (cb['metric_a'] == 'boundary') &
                 (cb['band_b'] == 'beta_low') & (cb['metric_b'] == 'attractor')]
        if not row.empty:
            new_rho = row.iloc[0]['rho']
            check('Coupling', 'alpha_bnd × beta_low_att rho (was -0.41)', -410, int(new_rho * 1000))

    # --- LIFESPAN ---
    life_path = os.path.join(OUT_DIR, 'lifespan_trajectory.csv')
    if os.path.exists(life_path):
        print(f"\n  --- Lifespan Trajectory ---")
        life = pd.read_csv(life_path)
        n_opposite = len(life[(life['pattern'].isin(['inverted-U', 'U-shape']))])
        both_sig = life[life['pattern'] != 'not both significant']
        n_both = len(both_sig)
        if n_both > 0:
            n_opp = len(both_sig[both_sig.apply(lambda r: r['hbn_rho'] * r['dort_rho'] < 0, axis=1)])
            check('Lifespan', f'Jointly sig features (was 28)', 28, n_both)
            check('Lifespan', f'Opposite direction (was 24/28)', 24, n_opp)

    # --- HBN PSYCHOPATHOLOGY ---
    for psy in ['externalizing', 'internalizing']:
        psy_path = os.path.join(OUT_DIR, f'hbn_{psy}_correlations.csv')
        if os.path.exists(psy_path):
            psy_df = pd.read_csv(psy_path)
            n_fdr = psy_df['significant'].sum() if 'significant' in psy_df.columns else 0
            old_n = 10 if psy == 'externalizing' else 4
            check('HBN Psych', f'{psy} FDR survivors (was {old_n})', old_n, n_fdr)

    # --- MEDICAL ---
    med_path = os.path.join(OUT_DIR, 'medical_correlations.csv')
    if os.path.exists(med_path):
        print(f"\n  --- Medical ---")
        med = pd.read_csv(med_path)
        n_fdr = med['significant'].sum() if 'significant' in med.columns else 0
        check('Medical', 'FDR survivors (was 0)', 0, n_fdr)

    # Save comparison
    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(OUT_DIR, 'report_comparison.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/report_comparison.csv")

    # Summary
    print(f"\n  {'='*50}")
    print(f"  SUMMARY OF CHANGES")
    print(f"  {'='*50}")
    large_changes = rdf[rdf['delta'].abs() > 20].dropna(subset=['delta'])
    if len(large_changes) > 0:
        print(f"  Large changes (|Δ|>20):")
        for _, r in large_changes.iterrows():
            print(f"    {r['claim']:<45s}  {r['old']:>+.0f} → {r['new']:>+.0f} (Δ={r['delta']:+.0f})")
    else:
        print(f"  No large changes detected.")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Run all f0=7.60 v2 analyses')
    parser.add_argument('--step', required=True,
                        choices=['enrichment', 'compare', 'cognitive', 'cognitive_eo',
                                 'hbn_age', 'hbn_cross_release', 'hbn_per_release',
                                 'dortmund_age', 'eceo', 'dortmund_2x2',
                                 'personality', 'reliability', 'within_session',
                                 'adult_pediatric', 'lifespan', 'cross_band',
                                 'medical', 'handedness', 'sex_age', 'state_age',
                                 'power_sensitivity', 'report_comparison',
                                 'tdbrain', 'all'])
    parser.add_argument('--min-power-pct', type=int, default=50,
                        help='Keep top N%% of peaks by power per band (0=all, 50=top half)')
    parser.add_argument('--peak-base', type=str, default=None,
                        help='Override peak CSV directory (default: exports_adaptive_v3). '
                             'Use exports_irasa_v4 for IRASA-extracted peaks.')
    parser.add_argument('--min-peaks', type=int, default=30,
                        help='Minimum peaks per band per subject for per-subject analyses (default: 30)')
    args = parser.parse_args()

    global MIN_POWER_PCT, NEW_PEAK_BASE, MIN_PEAKS_PER_BAND
    MIN_POWER_PCT = args.min_power_pct
    MIN_PEAKS_PER_BAND = args.min_peaks
    if args.peak_base is not None:
        NEW_PEAK_BASE = os.path.join(BASE_DIR, args.peak_base)

    os.makedirs(OUT_DIR, exist_ok=True)

    steps = {
        'enrichment': run_enrichment,
        'cognitive': run_cognitive,
        'hbn_age': run_hbn_age,
        'dortmund_age': run_dortmund_age,
        'eceo': run_eceo,
        'dortmund_2x2': run_dortmund_2x2,
        'personality': run_personality,
        'reliability': run_reliability,
        'compare': run_compare,
        'hbn_cross_release': run_hbn_cross_release,
        'cognitive_eo': run_cognitive_eo,
        'adult_pediatric': run_adult_vs_pediatric,
        'hbn_per_release': run_hbn_per_release,
        'lifespan': run_lifespan,
        'tdbrain': run_tdbrain,
        'cross_band': run_cross_band_coupling,
        'within_session': run_within_session,
        'medical': run_medical,
        'handedness': run_handedness,
        'sex_age': run_sex_age_interaction,
        'state_age': run_state_sensitivity_age,
        'power_sensitivity': run_power_sensitivity,
        'report_comparison': run_report_comparison,
    }

    if args.step == 'all':
        for name, fn in steps.items():
            try:
                fn()
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                import traceback; traceback.print_exc()
    else:
        steps[args.step]()


if __name__ == '__main__':
    main()
