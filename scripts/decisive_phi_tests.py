#!/usr/bin/env python
"""
Decisive Tests: Schumann-Anchored vs IAF Geometric Accident
============================================================

Five tests to discriminate whether phi-lattice alignment at f₀=7.83 Hz
is caused by Schumann resonance or is a geometric accident of IAF placement.

Test 1: Alpha-excluded cross-base comparison
Test 2: IAF-stratified non-alpha alignment
Test 3: Per-subject optimal f₀ sweep
Test 4: Theta boundary frequency analysis
Test 5: Generative neuroscience null

Usage:
    python scripts/decisive_phi_tests.py
"""
import sys, os, time, io
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from phi_replication import (
    F0, PHI, BANDS, BASES,
    POSITIONS_DEG2, PHI_POSITIONS, POSITIONS_14,
    lattice_coord, circ_dist, min_lattice_dist, positions_for_base,
)

# ═══════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════
DATASETS = {
    'EEGMMIDB EC': 'exports_eegmmidb/replication/combined/per_subject_dominant_peaks.csv',
    'LEMON EC':    'exports_lemon/replication/EC/per_subject_dominant_peaks.csv',
    'LEMON EO':    'exports_lemon/replication/EO/per_subject_dominant_peaks.csv',
    'HBN EC':      'exports_hbn/EC/per_subject_dominant_peaks.csv',
    'Dortmund EC': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_dominant_peaks.csv',
    'Dortmund EO': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesOpen_pre/per_subject_dominant_peaks.csv',
}

ALL_BANDS = list(BANDS.keys())  # delta, theta, alpha, beta_low, beta_high, gamma
NON_ALPHA_BANDS = [b for b in ALL_BANDS if b != 'alpha']
THETA_GAMMA_ONLY = ['theta', 'gamma']
N_PERM = 5000
RNG_SEED = 42


def load_datasets():
    """Load all available datasets."""
    loaded = {}
    for name, path in DATASETS.items():
        if os.path.isfile(path):
            df = pd.read_csv(path)
            loaded[name] = df
            print(f"  {name}: N={len(df)} subjects")
        else:
            print(f"  {name}: NOT FOUND ({path})")
    return loaded


def compute_mean_d_for_bands(row, bands, f0=F0, base=PHI, positions=POSITIONS_DEG2):
    """Compute mean lattice distance for specified bands at given f0/base."""
    ds = []
    for b in bands:
        freq = row.get(f'{b}_freq', np.nan)
        if pd.isna(freq) or freq <= 0:
            continue
        u = lattice_coord(freq, f0=f0, base=base)
        d = min_lattice_dist(u, positions)
        ds.append(d)
    return np.mean(ds) if ds else np.nan


def compute_null_expected(positions, n_samples=100_000, seed=42):
    """Expected mean_d under uniform null for given positions."""
    rng = np.random.RandomState(seed)
    us = rng.uniform(0, 1, n_samples)
    ds = np.array([min_lattice_dist(u, positions) for u in us])
    return ds.mean()


def cross_base_comparison(df, bands, degree=3, n_perm=N_PERM, seed=RNG_SEED):
    """Run cross-base comparison (9 bases) for given band subset.

    Returns dict: base_name → {mean_d, z_score, n_positions}
    """
    # Collect per-subject frequencies for the bands
    subject_freqs = []
    for _, row in df.iterrows():
        freqs = {}
        for b in bands:
            freq = row.get(f'{b}_freq', np.nan)
            if not pd.isna(freq) and freq > 0:
                freqs[b] = freq
        if freqs:
            subject_freqs.append(freqs)

    if len(subject_freqs) < 10:
        return {}

    # Band-level frequency arrays for permutation
    band_freq_arrays = {}
    for b in bands:
        vals = [sf[b] for sf in subject_freqs if b in sf]
        if vals:
            band_freq_arrays[b] = np.array(vals)

    base_results = {}
    for base_name, base_val in BASES.items():
        positions = positions_for_base(base_val, degree=degree)

        # Observed: per-subject mean_d
        obs_ds = []
        for sf in subject_freqs:
            band_ds = []
            for b, freq in sf.items():
                u = lattice_coord(freq, f0=F0, base=base_val)
                d = min_lattice_dist(u, positions)
                band_ds.append(d)
            obs_ds.append(np.mean(band_ds))
        obs_ds = np.array(obs_ds)
        obs_mean = obs_ds.mean()

        # Permutation null
        rng = np.random.RandomState(seed)
        null_means = np.empty(n_perm)
        for pi in range(n_perm):
            perm_ds = []
            for b, freqs in band_freq_arrays.items():
                shuffled_u = rng.uniform(0, 1, len(freqs))
                dists = np.array([min_lattice_dist(u, positions) for u in shuffled_u])
                perm_ds.append(dists.mean())
            null_means[pi] = np.mean(perm_ds) if perm_ds else np.nan

        null_mean = np.nanmean(null_means)
        null_sd = np.nanstd(null_means)
        z_score = (null_mean - obs_mean) / null_sd if null_sd > 0 else 0.0

        base_results[base_name] = {
            'mean_d': obs_mean,
            'z_score': z_score,
            'n_positions': len(positions),
        }

    return base_results


def phi_rank_from_results(base_results):
    """Get phi's rank by z-score (1 = best)."""
    ranking = sorted(base_results.items(), key=lambda x: -x[1]['z_score'])
    for i, (name, _) in enumerate(ranking):
        if name == 'phi':
            return i + 1
    return -1


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Alpha-Excluded Cross-Base Comparison
# ═══════════════════════════════════════════════════════════════════
def test1_alpha_excluded(datasets):
    """Cross-base comparison excluding alpha band."""
    print("\n" + "=" * 90)
    print("TEST 1: ALPHA-EXCLUDED CROSS-BASE COMPARISON")
    print("=" * 90)
    print("If phi stays rank #1-2/9 without alpha, IAF cannot explain alignment.\n")

    results = {}
    for ds_name, df in datasets.items():
        print(f"  {ds_name} (N={len(df)}):")

        # Non-alpha (5 bands)
        br_5b = cross_base_comparison(df, NON_ALPHA_BANDS, degree=3)
        rank_5b = phi_rank_from_results(br_5b)
        phi_z_5b = br_5b['phi']['z_score'] if 'phi' in br_5b else 0

        # Theta+gamma only (2 bands, maximally distant from IAF)
        br_2b = cross_base_comparison(df, THETA_GAMMA_ONLY, degree=3)
        rank_2b = phi_rank_from_results(br_2b)
        phi_z_2b = br_2b['phi']['z_score'] if 'phi' in br_2b else 0

        # All 6 bands (reference)
        br_6b = cross_base_comparison(df, ALL_BANDS, degree=3)
        rank_6b = phi_rank_from_results(br_6b)
        phi_z_6b = br_6b['phi']['z_score'] if 'phi' in br_6b else 0

        print(f"    All 6 bands:     phi rank {rank_6b}/9  (z={phi_z_6b:+.2f})")
        print(f"    No-alpha (5b):   phi rank {rank_5b}/9  (z={phi_z_5b:+.2f})")
        print(f"    Theta+gamma (2b): phi rank {rank_2b}/9  (z={phi_z_2b:+.2f})")

        # Top 3 for non-alpha
        ranking_5b = sorted(br_5b.items(), key=lambda x: -x[1]['z_score'])[:3]
        top3_str = ', '.join(f"{n}:{r['z_score']:+.1f}" for n, r in ranking_5b)
        print(f"    No-alpha top-3:  {top3_str}")

        results[ds_name] = {
            'rank_6b': rank_6b, 'z_6b': phi_z_6b,
            'rank_5b': rank_5b, 'z_5b': phi_z_5b,
            'rank_2b': rank_2b, 'z_2b': phi_z_2b,
            'base_results_5b': br_5b,
        }

    # Summary
    print(f"\n  SUMMARY:")
    print(f"  {'Dataset':<16s}  {'6b rank':>8s}  {'5b rank':>8s}  {'2b rank':>8s}")
    print(f"  {'-'*50}")
    for ds_name, r in results.items():
        print(f"  {ds_name:<16s}  {r['rank_6b']:>5d}/9   {r['rank_5b']:>5d}/9   {r['rank_2b']:>5d}/9")

    return results


# ═══════════════════════════════════════════════════════════════════
# TEST 2: IAF-Stratified Non-Alpha Alignment
# ═══════════════════════════════════════════════════════════════════
def test2_iaf_stratified(datasets):
    """Non-alpha alignment stratified by IAF quartile."""
    print("\n" + "=" * 90)
    print("TEST 2: IAF-STRATIFIED NON-ALPHA ALIGNMENT")
    print("=" * 90)
    print("If non-alpha Cohen's d is IAF-invariant across quartiles → Schumann.\n")

    # Pool all datasets for IAF analysis
    all_rows = []
    for ds_name, df in datasets.items():
        df_copy = df.copy()
        df_copy['dataset'] = ds_name
        all_rows.append(df_copy)
    pooled = pd.concat(all_rows, ignore_index=True)

    # Use alpha_freq as IAF proxy
    pooled['iaf'] = pooled['alpha_freq']
    valid = pooled[pooled['iaf'].notna() & (pooled['iaf'] > 0)].copy()

    # Compute non-alpha mean_d for each subject
    null_expected = compute_null_expected(POSITIONS_DEG2)
    non_alpha_ds = []
    for _, row in valid.iterrows():
        md = compute_mean_d_for_bands(row, NON_ALPHA_BANDS, f0=F0, base=PHI, positions=POSITIONS_DEG2)
        non_alpha_ds.append(md)
    valid = valid.copy()
    valid['non_alpha_mean_d'] = non_alpha_ds
    valid = valid[valid['non_alpha_mean_d'].notna()].copy()

    print(f"  Pooled subjects with valid IAF and non-alpha mean_d: N={len(valid)}")
    print(f"  Null expected mean_d (uniform, deg-2): {null_expected:.4f}")
    print(f"  IAF range: {valid['iaf'].min():.1f} – {valid['iaf'].max():.1f} Hz")

    # IAF quartiles
    valid['iaf_quartile'] = pd.qcut(valid['iaf'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])

    print(f"\n  {'Quartile':<14s}  {'IAF range':>16s}  {'N':>4s}  {'mean_d':>8s}  {'Cohen d':>8s}  {'p-value':>10s}")
    print(f"  {'-'*70}")

    quartile_results = {}
    for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
        qdf = valid[valid['iaf_quartile'] == q]
        iaf_lo, iaf_hi = qdf['iaf'].min(), qdf['iaf'].max()
        ds_arr = qdf['non_alpha_mean_d'].values
        obs_mean = ds_arr.mean()
        obs_sd = ds_arr.std()
        cohen_d = (null_expected - obs_mean) / obs_sd if obs_sd > 0 else 0.0
        _, p_val = stats.ttest_1samp(ds_arr, null_expected)
        print(f"  {q:<14s}  {iaf_lo:6.1f}–{iaf_hi:5.1f} Hz  {len(qdf):>4d}  {obs_mean:.4f}  {cohen_d:+.3f}  {p_val:.2e}")
        quartile_results[q] = {'n': len(qdf), 'mean_d': obs_mean, 'cohen_d': cohen_d, 'p': p_val}

    # Correlation: non_alpha_mean_d ~ IAF
    r, p = stats.pearsonr(valid['iaf'], valid['non_alpha_mean_d'])
    print(f"\n  Correlation (non-alpha mean_d ~ IAF): r={r:+.3f}, p={p:.3e}")
    print(f"  Interpretation: {'IAF-DEPENDENT (accident)' if p < 0.05 else 'IAF-INVARIANT (Schumann-consistent)'}")

    # Cross-base per quartile (just for Q1 and Q4 for speed)
    for q in ['Q1 (low)', 'Q4 (high)']:
        qdf = valid[valid['iaf_quartile'] == q]
        br = cross_base_comparison(qdf, NON_ALPHA_BANDS, degree=3, n_perm=2000)
        rank = phi_rank_from_results(br)
        phi_z = br['phi']['z_score'] if 'phi' in br else 0
        print(f"  {q} cross-base: phi rank {rank}/9 (z={phi_z:+.1f})")

    return quartile_results, r, p


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Per-Subject Optimal f₀ Sweep
# ═══════════════════════════════════════════════════════════════════
def test3_f0_sweep(datasets):
    """Sweep f₀ per subject, find optimal, test vs IAF."""
    print("\n" + "=" * 90)
    print("TEST 3: PER-SUBJECT OPTIMAL f₀ SWEEP")
    print("=" * 90)
    print("If optimal f₀ clusters at 7.83 and is uncorrelated with IAF → Schumann.\n")

    f0_range = np.arange(6.0, 10.05, 0.05)
    positions = POSITIONS_DEG2

    # Use EEGMMIDB and LEMON EC (largest local datasets)
    test_datasets = {k: v for k, v in datasets.items() if k in ['EEGMMIDB EC', 'LEMON EC', 'HBN EC']}
    if not test_datasets:
        test_datasets = dict(list(datasets.items())[:2])

    all_opt_f0 = []
    all_opt_f0_nonalpha = []
    all_iaf = []
    all_ds_names = []

    for ds_name, df in test_datasets.items():
        print(f"  {ds_name} (N={len(df)}):")
        opt_f0s = []
        opt_f0s_na = []
        iafs = []

        for _, row in df.iterrows():
            # Collect frequencies for all bands
            freqs_all = {}
            freqs_na = {}
            for b in ALL_BANDS:
                freq = row.get(f'{b}_freq', np.nan)
                if not pd.isna(freq) and freq > 0:
                    freqs_all[b] = freq
                    if b != 'alpha':
                        freqs_na[b] = freq

            if len(freqs_all) < 3:
                continue

            # Sweep f₀
            best_d_all = 1.0
            best_f0_all = 7.83
            best_d_na = 1.0
            best_f0_na = 7.83

            for f0 in f0_range:
                # All bands
                ds_vals = []
                for freq in freqs_all.values():
                    u = lattice_coord(freq, f0=f0, base=PHI)
                    d = min_lattice_dist(u, positions)
                    ds_vals.append(d)
                md = np.mean(ds_vals)
                if md < best_d_all:
                    best_d_all = md
                    best_f0_all = f0

                # Non-alpha
                if freqs_na:
                    ds_na = []
                    for freq in freqs_na.values():
                        u = lattice_coord(freq, f0=f0, base=PHI)
                        d = min_lattice_dist(u, positions)
                        ds_na.append(d)
                    md_na = np.mean(ds_na)
                    if md_na < best_d_na:
                        best_d_na = md_na
                        best_f0_na = f0

            opt_f0s.append(best_f0_all)
            opt_f0s_na.append(best_f0_na)

            iaf = row.get('alpha_freq', np.nan)
            iafs.append(iaf)

        opt_f0s = np.array(opt_f0s)
        opt_f0s_na = np.array(opt_f0s_na)
        iafs = np.array(iafs)

        # Distribution stats
        print(f"    Optimal f₀ (all bands):  mean={np.mean(opt_f0s):.2f}, median={np.median(opt_f0s):.2f}, SD={np.std(opt_f0s):.2f}")
        print(f"    Optimal f₀ (non-alpha):  mean={np.mean(opt_f0s_na):.2f}, median={np.median(opt_f0s_na):.2f}, SD={np.std(opt_f0s_na):.2f}")

        # Fraction within ±0.5 of 7.83
        near_783_all = np.mean(np.abs(opt_f0s - 7.83) < 0.5)
        near_783_na = np.mean(np.abs(opt_f0s_na - 7.83) < 0.5)
        print(f"    Fraction within ±0.5 Hz of 7.83: all={near_783_all:.1%}, non-alpha={near_783_na:.1%}")

        # Correlation with IAF
        valid_mask = np.isfinite(iafs)
        if valid_mask.sum() > 10:
            r_all, p_all = stats.pearsonr(iafs[valid_mask], opt_f0s[valid_mask])
            r_na, p_na = stats.pearsonr(iafs[valid_mask], opt_f0s_na[valid_mask])
            print(f"    r(f₀*, IAF):     all-bands r={r_all:+.3f} p={p_all:.3e}")
            print(f"    r(f₀*_na, IAF):  non-alpha r={r_na:+.3f} p={p_na:.3e}")
        else:
            r_all = r_na = p_all = p_na = np.nan

        all_opt_f0.extend(opt_f0s)
        all_opt_f0_nonalpha.extend(opt_f0s_na)
        all_iaf.extend(iafs)
        all_ds_names.extend([ds_name] * len(opt_f0s))

    # Pooled
    all_opt_f0 = np.array(all_opt_f0)
    all_opt_f0_na = np.array(all_opt_f0_nonalpha)
    all_iaf = np.array(all_iaf)

    print(f"\n  POOLED (N={len(all_opt_f0)}):")
    print(f"    Optimal f₀ (all):       mean={np.mean(all_opt_f0):.2f}, median={np.median(all_opt_f0):.2f}")
    print(f"    Optimal f₀ (non-alpha): mean={np.mean(all_opt_f0_na):.2f}, median={np.median(all_opt_f0_na):.2f}")

    # Histogram of optimal f₀ (binned)
    bins = np.arange(6.0, 10.2, 0.2)
    hist_all, _ = np.histogram(all_opt_f0, bins=bins)
    hist_na, _ = np.histogram(all_opt_f0_na, bins=bins)
    print(f"\n  Distribution of optimal f₀ (all bands):")
    for i in range(len(hist_all)):
        bar = '#' * (hist_all[i] * 60 // max(hist_all.max(), 1))
        mark = ' <-- 7.83' if bins[i] <= 7.83 < bins[i+1] else ''
        print(f"    {bins[i]:5.1f}-{bins[i+1]:5.1f}: {hist_all[i]:>4d} {bar}{mark}")

    print(f"\n  Distribution of optimal f₀ (non-alpha):")
    for i in range(len(hist_na)):
        bar = '#' * (hist_na[i] * 60 // max(hist_na.max(), 1))
        mark = ' <-- 7.83' if bins[i] <= 7.83 < bins[i+1] else ''
        print(f"    {bins[i]:5.1f}-{bins[i+1]:5.1f}: {hist_na[i]:>4d} {bar}{mark}")

    valid_mask = np.isfinite(all_iaf)
    if valid_mask.sum() > 10:
        r, p = stats.pearsonr(all_iaf[valid_mask], all_opt_f0[valid_mask])
        r_na, p_na = stats.pearsonr(all_iaf[valid_mask], all_opt_f0_na[valid_mask])
        print(f"\n  Pooled correlation:")
        print(f"    r(f₀*, IAF):     r={r:+.3f}, p={p:.3e}")
        print(f"    r(f₀*_na, IAF):  r={r_na:+.3f}, p={p_na:.3e}")

    return all_opt_f0, all_opt_f0_na, all_iaf


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Theta Boundary Frequency Analysis
# ═══════════════════════════════════════════════════════════════════
def test4_theta_boundary(datasets):
    """Analyze actual frequencies of theta peaks near boundary."""
    print("\n" + "=" * 90)
    print("TEST 4: THETA BOUNDARY FREQUENCY ANALYSIS")
    print("=" * 90)
    print("If theta peaks cluster at 7.83 Hz specifically (not IAF/2) → Schumann.\n")

    all_theta_freqs = []
    all_iafs = []

    for ds_name, df in datasets.items():
        theta_f = df['theta_freq'].dropna().values
        theta_f = theta_f[theta_f > 0]
        iaf = df['alpha_freq'].dropna().values
        iaf = iaf[iaf > 0]

        if len(theta_f) == 0:
            continue

        # Fraction near 7.83
        near_783 = np.mean(np.abs(theta_f - 7.83) < 0.5)
        near_6 = np.mean(np.abs(theta_f - 6.0) < 0.5)

        # Lattice coordinate
        theta_u = np.array([lattice_coord(f, f0=F0, base=PHI) for f in theta_f])
        near_boundary = np.mean(np.array([circ_dist(u, 0.0) for u in theta_u]) < 0.05)

        print(f"  {ds_name} (N={len(theta_f)}):")
        print(f"    Theta freq:  mean={np.mean(theta_f):.2f}, median={np.median(theta_f):.2f}, SD={np.std(theta_f):.2f}")
        print(f"    Near 7.83 Hz (±0.5): {near_783:.1%}  |  Near 6.0 Hz (±0.5): {near_6:.1%}")
        print(f"    Near boundary (|u|<0.05): {near_boundary:.1%}")

        all_theta_freqs.extend(theta_f)
        all_iafs.extend(iaf[:len(theta_f)])  # may differ in length

    all_theta_freqs = np.array(all_theta_freqs)

    # Histogram
    bins = np.arange(4.0, 8.2, 0.25)
    hist, _ = np.histogram(all_theta_freqs, bins=bins)
    print(f"\n  Pooled theta frequency histogram:")
    for i in range(len(hist)):
        bar = '#' * (hist[i] * 50 // max(hist.max(), 1))
        mark = ''
        if bins[i] <= 7.83 < bins[i+1]:
            mark = ' <-- f₀=7.83'
        elif bins[i] <= 6.0 < bins[i+1]:
            mark = ' <-- typical theta'
        print(f"    {bins[i]:5.2f}-{bins[i+1]:5.2f}: {hist[i]:>4d} {bar}{mark}")

    # Under uniform-in-Hz null within theta band [4,8]:
    # P(within ±0.5 of 7.83) = 1.0/4.0 = 0.25 (since 7.33-7.83 is within [4,8])
    # Actually: |f-7.83| < 0.5 → f ∈ [7.33, 8.0] (capped at band edge) = 0.67 Hz out of 4 Hz = 16.75%
    expected_near_783 = 0.67 / 4.0  # 16.75%
    observed_near_783 = np.mean(np.abs(all_theta_freqs - 7.83) < 0.5)
    ratio = observed_near_783 / expected_near_783
    print(f"\n  Observed near 7.83 Hz: {observed_near_783:.1%}")
    print(f"  Expected (uniform in [4,8]): {expected_near_783:.1%}")
    print(f"  Enrichment ratio: {ratio:.2f}x")

    # Correlation: theta_freq ~ IAF (test IAF/2 subharmonic)
    # Match subjects that have both theta and alpha
    all_results = []
    for ds_name, df in datasets.items():
        valid = df[df['theta_freq'].notna() & df['alpha_freq'].notna() &
                    (df['theta_freq'] > 0) & (df['alpha_freq'] > 0)].copy()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid['alpha_freq'], valid['theta_freq'])
            # Also test theta ~ IAF/2
            r2, p2 = stats.pearsonr(valid['alpha_freq'] / 2, valid['theta_freq'])
            print(f"\n  {ds_name}: r(theta, IAF)={r:+.3f} p={p:.3e}")
            print(f"  {ds_name}: r(theta, IAF/2)={r2:+.3f} p={p2:.3e}")
            # Mean |theta - IAF/2|
            iaf_half = valid['alpha_freq'] / 2
            mean_dist_to_half = np.mean(np.abs(valid['theta_freq'] - iaf_half))
            mean_dist_to_783 = np.mean(np.abs(valid['theta_freq'] - 7.83))
            print(f"  {ds_name}: mean |theta - IAF/2| = {mean_dist_to_half:.2f} Hz, mean |theta - 7.83| = {mean_dist_to_783:.2f} Hz")
            all_results.append({
                'dataset': ds_name, 'n': len(valid),
                'r_iaf': r, 'p_iaf': p,
                'dist_iaf_half': mean_dist_to_half,
                'dist_783': mean_dist_to_783,
            })

    return all_results


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Generative Neuroscience Null
# ═══════════════════════════════════════════════════════════════════
def test5_generative_null(datasets):
    """Generate synthetic subjects from known neuroscience distributions."""
    print("\n" + "=" * 90)
    print("TEST 5: GENERATIVE NEUROSCIENCE NULL")
    print("=" * 90)
    print("If synthetic brains show comparable alignment → geometric accident.\n")

    # Known neuroscience peak frequency distributions (mean, sd)
    NEURO_DISTS = {
        'delta':     (2.5, 0.5),
        'theta':     (6.0, 0.8),
        'alpha':     (10.0, 1.2),
        'beta_low':  (15.0, 1.5),
        'beta_high': (22.0, 2.0),
        'gamma':     (38.0, 3.0),
    }

    # First, estimate actual distributions from real data
    print("  Observed peak frequency distributions (pooled across datasets):")
    real_dists = {}
    for b in ALL_BANDS:
        all_f = []
        for ds_name, df in datasets.items():
            vals = df[f'{b}_freq'].dropna().values
            vals = vals[vals > 0]
            all_f.extend(vals)
        all_f = np.array(all_f)
        if len(all_f) > 0:
            real_dists[b] = (np.mean(all_f), np.std(all_f))
            print(f"    {b:>10s}: mean={np.mean(all_f):.2f}, SD={np.std(all_f):.2f}, N={len(all_f)}")

    n_synth = 5000
    rng = np.random.RandomState(42)

    # --- Method A: Standard textbook distributions ---
    synth_mean_ds_textbook = []
    for _ in range(n_synth):
        ds_vals = []
        for b, (mu, sd) in NEURO_DISTS.items():
            freq = rng.normal(mu, sd)
            lo, hi = BANDS[b]
            freq = np.clip(freq, lo + 0.1, hi - 0.1)
            u = lattice_coord(freq, f0=F0, base=PHI)
            d = min_lattice_dist(u, POSITIONS_DEG2)
            ds_vals.append(d)
        synth_mean_ds_textbook.append(np.mean(ds_vals))
    synth_mean_ds_textbook = np.array(synth_mean_ds_textbook)

    # --- Method B: Empirical distributions (fit from real data) ---
    synth_mean_ds_empirical = []
    for _ in range(n_synth):
        ds_vals = []
        for b, (mu, sd) in real_dists.items():
            freq = rng.normal(mu, sd)
            lo, hi = BANDS[b]
            freq = np.clip(freq, lo + 0.1, hi - 0.1)
            u = lattice_coord(freq, f0=F0, base=PHI)
            d = min_lattice_dist(u, POSITIONS_DEG2)
            ds_vals.append(d)
        synth_mean_ds_empirical.append(np.mean(ds_vals))
    synth_mean_ds_empirical = np.array(synth_mean_ds_empirical)

    # Real data mean_d
    real_mean_ds = []
    for ds_name, df in datasets.items():
        vals = df['mean_d'].dropna().values
        real_mean_ds.extend(vals)
    real_mean_ds = np.array(real_mean_ds)

    null_expected = compute_null_expected(POSITIONS_DEG2)

    # Cohen's d for each
    real_d = (null_expected - np.mean(real_mean_ds)) / np.std(real_mean_ds) if np.std(real_mean_ds) > 0 else 0
    text_d = (null_expected - np.mean(synth_mean_ds_textbook)) / np.std(synth_mean_ds_textbook) if np.std(synth_mean_ds_textbook) > 0 else 0
    emp_d = (null_expected - np.mean(synth_mean_ds_empirical)) / np.std(synth_mean_ds_empirical) if np.std(synth_mean_ds_empirical) > 0 else 0

    print(f"\n  Results:")
    print(f"    {'Source':<25s}  {'mean_d':>8s}  {'SD':>6s}  {'Cohen d':>8s}")
    print(f"    {'-'*55}")
    print(f"    {'Uniform null':<25s}  {null_expected:.4f}  {'—':>6s}  {'—':>8s}")
    print(f"    {'Real subjects (pooled)':<25s}  {np.mean(real_mean_ds):.4f}  {np.std(real_mean_ds):.4f}  {real_d:+.3f}")
    print(f"    {'Synth (textbook dists)':<25s}  {np.mean(synth_mean_ds_textbook):.4f}  {np.std(synth_mean_ds_textbook):.4f}  {text_d:+.3f}")
    print(f"    {'Synth (empirical dists)':<25s}  {np.mean(synth_mean_ds_empirical):.4f}  {np.std(synth_mean_ds_empirical):.4f}  {emp_d:+.3f}")

    # Mann-Whitney: real vs synthetic
    u_stat_text, p_mw_text = stats.mannwhitneyu(real_mean_ds, synth_mean_ds_textbook, alternative='less')
    u_stat_emp, p_mw_emp = stats.mannwhitneyu(real_mean_ds, synth_mean_ds_empirical, alternative='less')
    print(f"\n    Mann-Whitney (real < textbook): U={u_stat_text:.0f}, p={p_mw_text:.3e}")
    print(f"    Mann-Whitney (real < empirical): U={u_stat_emp:.0f}, p={p_mw_emp:.3e}")

    # --- f₀ sweep on synthetic data ---
    print(f"\n  f₀ sweep on synthetic (empirical dists, N={n_synth}):")
    f0_range = np.arange(6.0, 10.05, 0.1)
    synth_d_by_f0 = []
    for f0 in f0_range:
        ds_vals = []
        for b, (mu, sd) in real_dists.items():
            freq = mu  # Use mean frequency (population level)
            u = lattice_coord(freq, f0=f0, base=PHI)
            d = min_lattice_dist(u, POSITIONS_DEG2)
            ds_vals.append(d)
        synth_d_by_f0.append(np.mean(ds_vals))
    synth_d_by_f0 = np.array(synth_d_by_f0)
    best_idx = np.argmin(synth_d_by_f0)
    best_f0_synth = f0_range[best_idx]
    print(f"    Optimal f₀ for synthetic (empirical means): {best_f0_synth:.2f} Hz")
    print(f"    mean_d at f₀=7.83: {synth_d_by_f0[np.argmin(np.abs(f0_range - 7.83))]:.4f}")
    print(f"    mean_d at optimal f₀={best_f0_synth}: {synth_d_by_f0[best_idx]:.4f}")

    # Also sweep for textbook distributions
    synth_d_text = []
    for f0 in f0_range:
        ds_vals = []
        for b, (mu, sd) in NEURO_DISTS.items():
            u = lattice_coord(mu, f0=f0, base=PHI)
            d = min_lattice_dist(u, POSITIONS_DEG2)
            ds_vals.append(d)
        synth_d_text.append(np.mean(ds_vals))
    synth_d_text = np.array(synth_d_text)
    best_idx_text = np.argmin(synth_d_text)
    best_f0_text = f0_range[best_idx_text]
    print(f"    Optimal f₀ for textbook means: {best_f0_text:.2f} Hz")

    # Cross-base for synthetic (9 bases)
    print(f"\n  Cross-base comparison for synthetic subjects (empirical dists):")
    synth_df_rows = []
    for _ in range(n_synth):
        row = {'subject': f'synth_{_}'}
        for b, (mu, sd) in real_dists.items():
            freq = rng.normal(mu, sd)
            lo, hi = BANDS[b]
            freq = np.clip(freq, lo + 0.1, hi - 0.1)
            row[f'{b}_freq'] = freq
        synth_df_rows.append(row)
    synth_df = pd.DataFrame(synth_df_rows)

    br_synth = cross_base_comparison(synth_df, ALL_BANDS, degree=3, n_perm=2000, seed=99)
    rank_synth = phi_rank_from_results(br_synth)
    ranking = sorted(br_synth.items(), key=lambda x: -x[1]['z_score'])
    print(f"    Phi rank: {rank_synth}/9")
    for name, r in ranking:
        mark = ' <--' if name == 'phi' else ''
        print(f"      {name:>6s}: z={r['z_score']:+.2f}, mean_d={r['mean_d']:.4f}{mark}")

    return {
        'real_d': real_d,
        'textbook_d': text_d,
        'empirical_d': emp_d,
        'synth_rank': rank_synth,
        'best_f0_synth': best_f0_synth,
        'best_f0_text': best_f0_text,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    output = io.StringIO()

    class Tee:
        """Write to both stdout and StringIO."""
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    tee = Tee(sys.stdout, output)
    old_stdout = sys.stdout
    sys.stdout = tee

    print("Decisive Tests: Schumann-Anchored vs IAF Geometric Accident")
    print("=" * 90)
    print(f"f₀ = {F0} Hz, base = φ = {PHI:.6f}")
    print(f"Positions (degree-2): {list(POSITIONS_DEG2.keys())}")
    print(f"Bands: {ALL_BANDS}")
    print(f"Non-alpha bands: {NON_ALPHA_BANDS}")
    print(f"Permutations: {N_PERM}")
    print(f"\nLoading datasets:")

    datasets = load_datasets()
    if not datasets:
        print("ERROR: No datasets found!")
        sys.exit(1)

    # Run all 5 tests
    r1 = test1_alpha_excluded(datasets)
    r2 = test2_iaf_stratified(datasets)
    r3 = test3_f0_sweep(datasets)
    r4 = test4_theta_boundary(datasets)
    r5 = test5_generative_null(datasets)

    # ── Final verdict ──
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    # Collect key numbers
    test1_ranks = [r1[ds]['rank_5b'] for ds in r1]
    test1_verdict = "SCHUMANN" if np.mean(test1_ranks) <= 3 else "ACCIDENT"

    _, r_iaf, p_iaf = r2
    test2_verdict = "SCHUMANN" if p_iaf > 0.05 else "ACCIDENT"

    test4_results = r4
    # Check if theta is closer to 7.83 than IAF/2
    if test4_results:
        closer_to_783 = sum(1 for r in test4_results if r['dist_783'] < r['dist_iaf_half'])
        test4_verdict = "SCHUMANN" if closer_to_783 > len(test4_results) / 2 else "ACCIDENT"
    else:
        test4_verdict = "UNCLEAR"

    test5_verdict = "ACCIDENT" if r5['synth_rank'] <= 3 else "SCHUMANN"

    print(f"\n  Test 1 (alpha-excluded rank):    {test1_verdict}")
    print(f"    Non-alpha phi ranks: {test1_ranks}")
    print(f"\n  Test 2 (IAF-stratified):         {test2_verdict}")
    print(f"    r(non-alpha mean_d, IAF) = {r_iaf:+.3f}, p = {p_iaf:.3e}")
    print(f"\n  Test 3 (f₀ sweep):               (see distributions above)")
    print(f"\n  Test 4 (theta boundary):         {test4_verdict}")
    if test4_results:
        for r in test4_results:
            print(f"    {r['dataset']}: |θ-7.83|={r['dist_783']:.2f}, |θ-IAF/2|={r['dist_iaf_half']:.2f}")
    print(f"\n  Test 5 (generative null):        {test5_verdict}")
    print(f"    Real Cohen's d: {r5['real_d']:+.3f}")
    print(f"    Synthetic Cohen's d (empirical): {r5['empirical_d']:+.3f}")
    print(f"    Synthetic phi rank: {r5['synth_rank']}/9")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    # Save output
    sys.stdout = old_stdout
    with open('decisive_phi_tests_results.txt', 'w') as f:
        f.write(output.getvalue())
    print(f"\nResults saved to decisive_phi_tests_results.txt")
