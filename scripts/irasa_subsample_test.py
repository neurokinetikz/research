#!/usr/bin/env python3
"""
IRASA Peak-Yield Subsample Test
================================
Tests whether the drop from 31 to 8 cognitive FDR survivors (FOOOF→IRASA)
is explained by IRASA's lower per-subject peak yield, or whether IRASA
genuinely finds different/weaker enrichment patterns.

Approach:
1. Load FOOOF LEMON peaks → compute enrichment → correlate with cognition → count FDR
2. Subsample FOOOF peaks to match IRASA per-subject yield → same pipeline → count FDR
3. Repeat subsampling 200 times → distribution of FDR counts under matched yield
4. Compare: if subsample FDR ≈ 8, yield explains the gap; if ≈ 31, IRASA differs

Usage:
    python scripts/irasa_subsample_test.py [--n-iter 200]
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FOOOF_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3', 'lemon')
IRASA_BASE = os.path.join(BASE_DIR, 'exports_irasa_v4', 'lemon')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'irasa_subsample_test')

MIN_POWER_PCT = 50
MIN_PEAKS_PER_BAND = 30

# Frequency bands (same as run_all_f0_760_analyses.py)
OCTAVE_BAND = {'n-1': 'theta', 'n+0': 'alpha', 'n+1': 'beta_low',
               'n+2': 'beta_high', 'n+3': 'gamma'}

# Voronoi positions
N_POS = 12
POS_OFFSETS = np.array([0.0, 0.069, 0.131, 0.186, 0.236, 0.382,
                         0.500, 0.618, 0.764, 0.814, 0.869, 0.931])
POS_NAMES = ['boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3',
             'inv_noble_1', 'attractor', 'noble_1', 'inv_noble_3',
             'inv_noble_4', 'inv_noble_5', 'inv_noble_6']

# Hz-weighted expected fractions (pre-computed for these positions)
# Each Voronoi cell's expected fraction under uniform log-frequency
BOUNDARIES = np.zeros(N_POS + 1)
for i in range(N_POS):
    if i < N_POS - 1:
        BOUNDARIES[i + 1] = (POS_OFFSETS[i] + POS_OFFSETS[i + 1]) / 2
    else:
        BOUNDARIES[i + 1] = (POS_OFFSETS[i] + 1.0) / 2
BOUNDARIES[0] = (POS_OFFSETS[0] + POS_OFFSETS[-1] - 1.0) / 2

# Hz-weighted fractions
def compute_hz_fracs():
    fracs = np.zeros(N_POS)
    for i in range(N_POS):
        lo = BOUNDARIES[i] % 1.0
        hi = BOUNDARIES[i + 1] % 1.0
        if hi <= lo:
            hi += 1.0
            lo_part = 1.0 - lo
            hi_part = hi - 1.0 if hi > 1.0 else 0
        else:
            lo_part = hi - lo
            hi_part = 0
        # Hz-weight: integral of PHI^u from lo to hi
        frac = (PHI**hi - PHI**lo) / (PHI**1.0 - PHI**0.0)
        fracs[i] = frac
    return fracs / fracs.sum()

HZ_FRACS = compute_hz_fracs()


def lattice_coord(freqs):
    log_ratio = np.log(freqs / F0) / np.log(PHI)
    return log_ratio % 1.0


def assign_voronoi(u):
    dists = np.abs(u[:, None] - POS_OFFSETS[None, :])
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
            for m in ['mountain', 'ushape', 'peak_height', 'ramp_depth',
                      'center_depletion', 'asymmetry']:
                results[f'{band}_{m}'] = np.nan
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


# Cognitive test loading
COG_BASE = ('/Volumes/T9/lemon_data/behavioral/'
            'Behavioural_Data_MPILMBB_LEMON/Cognitive_Test_Battery_LEMON')
COG_TESTS = {
    'LPS': ('LPS/LPS.csv', 'LPS_1'),
    'RWT': ('RWT/RWT.csv', 'RWT_1'),
    'TMT': ('TMT/TMT.csv', 'TMT_1'),
    'CVLT': ('CVLT /CVLT.csv', 'CVLT_1'),
    'WST': ('WST/WST.csv', 'WST_1'),
    'TAP_Alert': ('TAP_Alertness/TAP-Alertness.csv', 'TAP_A_1'),
    'TAP_Incompat': ('TAP_Incompatibility/TAP-Incompatibility.csv', 'TAP_I_1'),
    'TAP_WM': ('TAP_Working_Memory/TAP-Working Memory.csv', 'TAP_WM_1'),
}


def load_cognitive_scores():
    cog = {}
    for test_name, (subpath, col) in COG_TESTS.items():
        path = os.path.join(COG_BASE, subpath)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        cog[test_name] = dict(zip(df['ID'], df[col]))
    return cog


def load_subject_peaks(peak_dir):
    """Load per-subject peak DataFrames (after power filtering).
    Returns dict of {subject_id: filtered_peaks_df}.
    """
    files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    subjects = {}
    for f in files:
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        df = pd.read_csv(f, usecols=['freq', 'power', 'phi_octave'])
        # Power filter per band
        filtered = []
        for octave in df['phi_octave'].unique():
            bp = df[df.phi_octave == octave]
            if len(bp) >= 2:
                thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                filtered.append(bp[bp['power'] >= thresh])
            else:
                filtered.append(bp)
        subjects[sub_id] = pd.concat(filtered, ignore_index=True) if filtered else df
    return subjects


def compute_enrichment_and_correlate(subjects_peaks, cog_scores):
    """Compute per-subject enrichment and correlate with cognitive scores.
    Returns: n_fdr_survivors, correlation_df
    """
    # Per-subject enrichment
    rows = []
    for sub_id, peaks_df in subjects_peaks.items():
        enrich = per_subject_enrichment(peaks_df)
        enrich['subject'] = sub_id
        rows.append(enrich)
    df = pd.DataFrame(rows)

    # Add cognitive scores
    cog_cols = []
    for test_name, scores in cog_scores.items():
        col = f'cog_{test_name}'
        df[col] = df['subject'].map(scores)
        cog_cols.append(col)

    # Feature columns (exclude n_peaks and subject)
    feature_cols = [c for c in df.columns if c != 'subject' and not c.startswith('cog_')
                    and not c.endswith('_n_peaks')]

    # Run correlations
    results = []
    for tgt in cog_cols:
        for feat in feature_cols:
            valid = df[[feat, tgt]].dropna()
            if len(valid) < 20:
                continue
            rho, p = spearmanr(valid[feat].values, valid[tgt].values)
            results.append({'target': tgt, 'feature': feat, 'rho': rho,
                           'p': p, 'n': len(valid)})

    rdf = pd.DataFrame(results)
    if len(rdf) == 0:
        return 0, rdf

    reject, p_fdr, _, _ = multipletests(rdf['p'].values, method='fdr_bh', alpha=0.05)
    rdf['significant'] = reject
    return int(reject.sum()), rdf


def subsample_peaks(subjects_peaks, target_counts):
    """Subsample each subject's peaks to match target counts per band.
    target_counts: dict of {subject_id: {octave: target_n}}
    """
    rng = np.random.default_rng()
    subsampled = {}
    for sub_id, peaks_df in subjects_peaks.items():
        if sub_id not in target_counts:
            subsampled[sub_id] = peaks_df
            continue
        parts = []
        for octave in peaks_df['phi_octave'].unique():
            bp = peaks_df[peaks_df.phi_octave == octave]
            target = target_counts[sub_id].get(octave, len(bp))
            if target < len(bp):
                idx = rng.choice(len(bp), size=target, replace=False)
                parts.append(bp.iloc[idx])
            else:
                parts.append(bp)
        subsampled[sub_id] = pd.concat(parts, ignore_index=True) if parts else peaks_df
    return subsampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-iter', type=int, default=200)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    fooof_peaks = load_subject_peaks(FOOOF_BASE)
    irasa_peaks = load_subject_peaks(IRASA_BASE)
    cog_scores = load_cognitive_scores()
    print(f"  FOOOF subjects: {len(fooof_peaks)}")
    print(f"  IRASA subjects: {len(irasa_peaks)}")
    print(f"  Cognitive tests: {len(cog_scores)}")

    # Compute per-subject, per-band peak counts for both methods
    fooof_counts = {}
    for sub_id, df in fooof_peaks.items():
        fooof_counts[sub_id] = {}
        for octave in df['phi_octave'].unique():
            fooof_counts[sub_id][octave] = int((df.phi_octave == octave).sum())

    irasa_counts = {}
    for sub_id, df in irasa_peaks.items():
        irasa_counts[sub_id] = {}
        for octave in df['phi_octave'].unique():
            irasa_counts[sub_id][octave] = int((df.phi_octave == octave).sum())

    # Compute yield ratio per band
    print("\n  Per-band yield comparison (matched subjects):")
    common = set(fooof_counts.keys()) & set(irasa_counts.keys())
    print(f"  Common subjects: {len(common)}")
    for octave in ['n-1', 'n+0', 'n+1', 'n+2', 'n+3']:
        f_counts = [fooof_counts[s].get(octave, 0) for s in common]
        i_counts = [irasa_counts[s].get(octave, 0) for s in common]
        ratio = np.mean(i_counts) / np.mean(f_counts) if np.mean(f_counts) > 0 else 0
        print(f"    {octave} ({OCTAVE_BAND.get(octave, '?')}): "
              f"FOOOF={np.mean(f_counts):.0f}, IRASA={np.mean(i_counts):.0f}, "
              f"ratio={ratio:.3f}")

    # ===================================================================
    # Step 1: Baseline FOOOF cognitive FDR count
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STEP 1: Baseline FOOOF")
    print("=" * 70)

    n_fdr_fooof, corr_fooof = compute_enrichment_and_correlate(fooof_peaks, cog_scores)
    n_tests_fooof = len(corr_fooof)
    print(f"  FDR survivors: {n_fdr_fooof}/{n_tests_fooof}")
    print(f"  Peak |ρ|: {corr_fooof['rho'].abs().max():.3f}")

    # ===================================================================
    # Step 2: Baseline IRASA cognitive FDR count
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Baseline IRASA")
    print("=" * 70)

    n_fdr_irasa, corr_irasa = compute_enrichment_and_correlate(irasa_peaks, cog_scores)
    n_tests_irasa = len(corr_irasa)
    print(f"  FDR survivors: {n_fdr_irasa}/{n_tests_irasa}")
    print(f"  Peak |ρ|: {corr_irasa['rho'].abs().max():.3f}")

    # ===================================================================
    # Step 3: Compute per-subject IRASA:FOOOF yield ratio
    # ===================================================================
    # For each FOOOF subject, compute target counts that match the
    # IRASA yield (per band)
    global_ratio = {}
    for octave in ['n-1', 'n+0', 'n+1', 'n+2', 'n+3']:
        f_total = sum(fooof_counts[s].get(octave, 0) for s in common)
        i_total = sum(irasa_counts[s].get(octave, 0) for s in common)
        global_ratio[octave] = i_total / f_total if f_total > 0 else 1.0

    print(f"\n  Global IRASA:FOOOF yield ratios:")
    for octave, ratio in global_ratio.items():
        print(f"    {octave} ({OCTAVE_BAND.get(octave, '?')}): {ratio:.3f}")

    # Build per-subject target counts
    target_counts = {}
    for sub_id in fooof_peaks:
        target_counts[sub_id] = {}
        for octave in ['n-1', 'n+0', 'n+1', 'n+2', 'n+3']:
            orig = fooof_counts[sub_id].get(octave, 0)
            target_counts[sub_id][octave] = max(1, int(orig * global_ratio.get(octave, 1.0)))

    # ===================================================================
    # Step 4: Subsample FOOOF to IRASA yield, repeat N times
    # ===================================================================
    print(f"\n" + "=" * 70)
    print(f"  STEP 4: Subsampling FOOOF to IRASA yield ({args.n_iter} iterations)")
    print("=" * 70)

    subsample_fdr_counts = []
    t0 = time.time()
    for i in range(args.n_iter):
        subsampled = subsample_peaks(fooof_peaks, target_counts)
        n_fdr, _ = compute_enrichment_and_correlate(subsampled, cog_scores)
        subsample_fdr_counts.append(n_fdr)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.n_iter - i - 1) / rate
            print(f"    Iteration {i+1}/{args.n_iter}: "
                  f"mean FDR={np.mean(subsample_fdr_counts):.1f}, "
                  f"last={n_fdr} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    subsample_fdr = np.array(subsample_fdr_counts)
    elapsed = time.time() - t0

    # ===================================================================
    # Step 5: Results
    # ===================================================================
    print(f"\n" + "=" * 70)
    print(f"  RESULTS")
    print("=" * 70)

    print(f"\n  FOOOF baseline: {n_fdr_fooof} FDR survivors")
    print(f"  IRASA baseline: {n_fdr_irasa} FDR survivors")
    print(f"\n  Subsampled FOOOF (matched to IRASA yield, {args.n_iter} iterations):")
    print(f"    Mean FDR survivors: {subsample_fdr.mean():.1f}")
    print(f"    Median: {np.median(subsample_fdr):.0f}")
    print(f"    95% CI: [{np.percentile(subsample_fdr, 2.5):.0f}, "
          f"{np.percentile(subsample_fdr, 97.5):.0f}]")
    print(f"    Min: {subsample_fdr.min()}, Max: {subsample_fdr.max()}")
    print(f"    SD: {subsample_fdr.std():.1f}")

    # Does the subsample distribution contain the IRASA value?
    p_below = (subsample_fdr <= n_fdr_irasa).mean()
    print(f"\n  P(subsampled FDR ≤ IRASA FDR={n_fdr_irasa}): {p_below:.4f}")

    if subsample_fdr.mean() > n_fdr_irasa * 2:
        print(f"\n  CONCLUSION: Yield reduction does NOT explain the FOOOF→IRASA FDR gap.")
        print(f"    Subsampled FOOOF ({subsample_fdr.mean():.0f} FDR) >> IRASA ({n_fdr_irasa})")
        print(f"    IRASA produces genuinely different enrichment patterns.")
    elif p_below > 0.05:
        print(f"\n  CONCLUSION: Yield reduction PARTIALLY explains the gap.")
        print(f"    {p_below*100:.1f}% of subsamples have ≤{n_fdr_irasa} FDR survivors.")
    else:
        print(f"\n  CONCLUSION: Yield reduction FULLY explains the gap.")
        print(f"    Subsampled FOOOF and IRASA produce comparable FDR counts.")

    # Save
    pd.DataFrame({
        'iteration': range(args.n_iter),
        'n_fdr_survivors': subsample_fdr_counts,
    }).to_csv(os.path.join(OUT_DIR, 'subsample_fdr_distribution.csv'), index=False)

    summary = {
        'fooof_fdr': n_fdr_fooof,
        'irasa_fdr': n_fdr_irasa,
        'subsample_mean': subsample_fdr.mean(),
        'subsample_median': np.median(subsample_fdr),
        'subsample_ci_lo': np.percentile(subsample_fdr, 2.5),
        'subsample_ci_hi': np.percentile(subsample_fdr, 97.5),
        'p_below_irasa': p_below,
        'n_iterations': args.n_iter,
        'elapsed_seconds': elapsed,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, 'subsample_summary.csv'), index=False)
    print(f"\n  Results saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
