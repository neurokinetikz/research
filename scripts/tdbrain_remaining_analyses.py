#!/usr/bin/env python3
"""
TDBRAIN Remaining Analyses
===========================

Runs all analyses not yet completed for TDBRAIN as dataset 10:
  1. NEO-FFI personality correlations
  2. Cognitive correlations (oddball, working memory RT)
  3. Trough detection and φ-scaling
  4. IRASA enrichment comparison
  5. Cross-band coupling

Usage:
    python scripts/tdbrain_remaining_analyses.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'tdbrain_analysis')
FOOOF_DIR = os.path.join(BASE_DIR, 'exports_adaptive_v4', 'tdbrain')
IRASA_DIR = os.path.join(BASE_DIR, 'exports_irasa_v4', 'tdbrain')
PARTICIPANTS_PATH = os.path.expanduser(
    '~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')

sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
from phi_frequency_model import PHI, F0

MIN_POWER_PCT = 50
FOOOF_TROUGHS = np.array([5.0274, 7.8227, 13.5949, 24.7516, 34.3834])


def load_all_freqs(peak_dir):
    """Load all peak frequencies from a directory."""
    all_freqs = []
    files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    for f in files:
        try:
            df = pd.read_csv(f, usecols=['freq', 'power', 'phi_octave'])
        except Exception:
            continue
        filtered = []
        for octave in df['phi_octave'].unique():
            bp = df[df.phi_octave == octave]
            if len(bp) == 0:
                continue
            thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
            filtered.append(bp[bp['power'] >= thresh])
        if filtered:
            df = pd.concat(filtered, ignore_index=True)
            all_freqs.extend(df['freq'].values)
    return np.array(all_freqs), len(files)


def find_troughs(freqs, n_hist=1000, sigma=8, f_range=(3, 55)):
    """Find density troughs."""
    log_freqs = np.log(freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)
    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)
    median_val = np.median(smoothed[smoothed > 0])
    trough_idx, _ = find_peaks(-smoothed, prominence=median_val * 0.08,
                                distance=n_hist // 25)
    return hz_centers[trough_idx]


def load_participants():
    """Load TDBRAIN participants."""
    df = pd.read_csv(PARTICIPANTS_PATH, sep='\t')
    df['age_float'] = df['age'].str.replace(',', '.').astype(float)
    df = df[df['DISC/REP'] == 'DISCOVERY']
    return df


# =============================================================
# 1. NEO-FFI Personality Correlations
# =============================================================
def analysis_personality():
    from statsmodels.stats.multitest import multipletests

    print("\n" + "=" * 70)
    print("1. NEO-FFI PERSONALITY × ENRICHMENT")
    print("=" * 70)

    demo = load_participants()
    neo_cols = [f'neoFFI_q{i}' for i in range(1, 61)]
    for c in neo_cols:
        demo[c] = pd.to_numeric(demo[c], errors='coerce')

    # Compute Big Five (approximate sum scores, no reverse scoring)
    demo['NEO_N'] = demo[[f'neoFFI_q{i}' for i in range(1, 13)]].sum(axis=1, min_count=6)
    demo['NEO_E'] = demo[[f'neoFFI_q{i}' for i in range(13, 25)]].sum(axis=1, min_count=6)
    demo['NEO_O'] = demo[[f'neoFFI_q{i}' for i in range(25, 37)]].sum(axis=1, min_count=6)
    demo['NEO_A'] = demo[[f'neoFFI_q{i}' for i in range(37, 49)]].sum(axis=1, min_count=6)
    demo['NEO_C'] = demo[[f'neoFFI_q{i}' for i in range(49, 61)]].sum(axis=1, min_count=6)

    # Load per-subject enrichment
    enrich_df = pd.read_csv(os.path.join(OUT_DIR.replace('tdbrain_analysis', 'f0_760_reanalysis'),
                                          'tdbrain_per_subject_enrichment.csv'))

    personality_cols = ['NEO_N', 'NEO_E', 'NEO_O', 'NEO_A', 'NEO_C']
    pid_str = demo['participants_ID'].astype(str)
    demo['subject_key'] = pid_str.where(pid_str.str.startswith('sub-'), 'sub-' + pid_str)

    for col in personality_cols:
        enrich_df[col] = enrich_df['subject'].map(
            dict(zip(demo['subject_key'], demo[col])))

    enrich_cols = [c for c in enrich_df.columns
                   if any(c.startswith(b) for b in ['theta_', 'alpha_', 'beta_low_', 'beta_high_', 'gamma_'])
                   and 'n_peaks' not in c]

    n_total = 0
    n_sig = 0
    results = []
    for trait in personality_cols:
        valid = enrich_df.dropna(subset=[trait])
        print(f"\n  {trait}: N = {len(valid)} with scores")
        for col in enrich_cols:
            v = valid.dropna(subset=[col])
            if len(v) < 30:
                continue
            rho, p = stats.spearmanr(v[trait], v[col])
            results.append({'trait': trait, 'feature': col, 'rho': rho, 'p': p,
                           'n': len(v), 'abs_rho': abs(rho)})
            n_total += 1

    df_r = pd.DataFrame(results)
    if len(df_r) > 0:
        rej, pfdr, _, _ = multipletests(df_r['p'].values, method='fdr_bh', alpha=0.05)
        df_r['p_fdr'] = pfdr
        df_r['significant'] = rej
        n_sig = rej.sum()

    print(f"\n  Total tests: {n_total}")
    print(f"  FDR survivors: {n_sig}")
    print(f"  (Paper prediction: 0 personality FDR survivors — spectral differentiation")
    print(f"   should be silent for personality traits)")

    if n_sig > 0:
        print(f"  Top FDR survivors:")
        for _, r in df_r[df_r.significant].nlargest(5, 'abs_rho').iterrows():
            print(f"    {r['trait']} × {r['feature']}: ρ={r['rho']:+.3f}, p_fdr={r['p_fdr']:.4f}")

    df_r.to_csv(os.path.join(OUT_DIR, 'tdbrain_personality_correlations.csv'), index=False)


# =============================================================
# 2. Cognitive Correlations (Oddball, Working Memory)
# =============================================================
def analysis_cognitive():
    from statsmodels.stats.multitest import multipletests

    print("\n" + "=" * 70)
    print("2. COGNITIVE × ENRICHMENT (Oddball, Working Memory)")
    print("=" * 70)

    demo = load_participants()

    # Cognitive columns from TDBRAIN
    cog_cols = {
        'avg_rt_oddb_CP': 'Oddball correct positive RT',
        'avg_rt_wm_CP': 'Working memory correct positive RT',
        'n_oddb_CP': 'Oddball correct positives (count)',
        'n_oddb_FN': 'Oddball false negatives (count)',
        'n_wm_CP': 'WM correct positives (count)',
        'n_wm_FN': 'WM false negatives (count)',
    }

    for col in cog_cols:
        demo[col] = pd.to_numeric(demo[col], errors='coerce')

    enrich_df = pd.read_csv(os.path.join(OUT_DIR.replace('tdbrain_analysis', 'f0_760_reanalysis'),
                                          'tdbrain_per_subject_enrichment.csv'))

    pid_str = demo['participants_ID'].astype(str)
    demo['subject_key'] = pid_str.where(pid_str.str.startswith('sub-'), 'sub-' + pid_str)

    for col in cog_cols:
        enrich_df[col] = enrich_df['subject'].map(
            dict(zip(demo['subject_key'], demo[col])))

    enrich_cols = [c for c in enrich_df.columns
                   if any(c.startswith(b) for b in ['theta_', 'alpha_', 'beta_low_', 'beta_high_', 'gamma_'])
                   and 'n_peaks' not in c]

    for cog_col, cog_name in cog_cols.items():
        valid = enrich_df.dropna(subset=[cog_col])
        if len(valid) < 50:
            print(f"\n  {cog_name}: N = {len(valid)} (skipping, < 50)")
            continue

        results = []
        for col in enrich_cols:
            v = valid.dropna(subset=[col])
            if len(v) < 30:
                continue
            rho, p = stats.spearmanr(v[cog_col], v[col])
            results.append({'cognitive': cog_col, 'feature': col, 'rho': rho, 'p': p,
                           'n': len(v), 'abs_rho': abs(rho)})

        df_r = pd.DataFrame(results)
        if len(df_r) > 0:
            rej, pfdr, _, _ = multipletests(df_r['p'].values, method='fdr_bh', alpha=0.05)
            df_r['p_fdr'] = pfdr
            df_r['significant'] = rej
            n_sig = rej.sum()
            print(f"\n  {cog_name}: N={len(valid)}, {len(df_r)} tests, {n_sig} FDR survivors")
            if n_sig > 0:
                for _, r in df_r[df_r.significant].nlargest(3, 'abs_rho').iterrows():
                    print(f"    {r['feature']}: ρ={r['rho']:+.3f}, p_fdr={r['p_fdr']:.4f}")
            df_r.to_csv(os.path.join(OUT_DIR, f'tdbrain_cognitive_{cog_col}.csv'), index=False)


# =============================================================
# 3. Trough Detection and φ-Scaling
# =============================================================
def analysis_troughs():
    print("\n" + "=" * 70)
    print("3. TROUGH DETECTION AND φ-SCALING (TDBRAIN as 10th dataset)")
    print("=" * 70)

    print("\n  --- FOOOF ---")
    fooof_freqs, n_fooof = load_all_freqs(FOOOF_DIR)
    print(f"  {n_fooof} subjects, {len(fooof_freqs):,} peaks")
    fooof_troughs = find_troughs(fooof_freqs)
    fooof_troughs = fooof_troughs[(fooof_troughs > 4) & (fooof_troughs < 50)]
    print(f"  Troughs: {np.round(fooof_troughs, 2)} Hz")
    if len(fooof_troughs) >= 2:
        ratios = fooof_troughs[1:] / fooof_troughs[:-1]
        geo = np.exp(np.mean(np.log(ratios)))
        print(f"  Ratios: {[f'{r:.4f}' for r in ratios]}")
        print(f"  Geometric mean: {geo:.4f} (φ = {PHI:.4f})")

    # Compare to paper's troughs
    print(f"\n  Comparison to paper's pooled troughs:")
    for i, (paper_t, label) in enumerate(zip(FOOOF_TROUGHS, ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ'])):
        dists = np.abs(fooof_troughs - paper_t)
        if len(dists) > 0 and dists.min() < paper_t * 0.3:
            nearest = fooof_troughs[np.argmin(dists)]
            delta = nearest - paper_t
            print(f"    {label}: paper={paper_t:.2f}, TDBRAIN={nearest:.2f}, Δ={delta:+.2f} Hz")
        else:
            print(f"    {label}: paper={paper_t:.2f}, TDBRAIN=not found")

    print(f"\n  --- IRASA ---")
    irasa_freqs, n_irasa = load_all_freqs(IRASA_DIR)
    print(f"  {n_irasa} subjects, {len(irasa_freqs):,} peaks")
    irasa_troughs = find_troughs(irasa_freqs)
    irasa_troughs = irasa_troughs[(irasa_troughs > 4) & (irasa_troughs < 50)]
    print(f"  Troughs: {np.round(irasa_troughs, 2)} Hz")
    if len(irasa_troughs) >= 2:
        ratios_i = irasa_troughs[1:] / irasa_troughs[:-1]
        geo_i = np.exp(np.mean(np.log(ratios_i)))
        print(f"  Ratios: {[f'{r:.4f}' for r in ratios_i]}")
        print(f"  Geometric mean: {geo_i:.4f} (φ = {PHI:.4f})")

    results = pd.DataFrame({
        'method': ['FOOOF', 'IRASA'],
        'n_subjects': [n_fooof, n_irasa],
        'n_peaks': [len(fooof_freqs), len(irasa_freqs)],
        'troughs': [str(np.round(fooof_troughs, 2).tolist()),
                    str(np.round(irasa_troughs, 2).tolist())],
        'geo_mean_ratio': [geo if len(fooof_troughs) >= 2 else np.nan,
                           geo_i if len(irasa_troughs) >= 2 else np.nan],
    })
    results.to_csv(os.path.join(OUT_DIR, 'tdbrain_trough_detection.csv'), index=False)


# =============================================================
# 4. IRASA Enrichment Comparison
# =============================================================
def analysis_irasa_comparison():
    print("\n" + "=" * 70)
    print("4. IRASA vs FOOOF ENRICHMENT COMPARISON")
    print("=" * 70)

    # Load IRASA enrichment for TDBRAIN
    sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

    # Compute enrichment from IRASA peaks using same Voronoi machinery
    from run_all_f0_760_analyses import compute_enrichment, load_peaks, OCTAVE_BAND, BAND_ORDER

    print("  Loading FOOOF peaks...")
    fooof_peaks, n_f = load_peaks(FOOOF_DIR)
    print(f"  FOOOF: {n_f} subjects, {len(fooof_peaks):,} peaks")

    print("  Loading IRASA peaks...")
    irasa_peaks, n_i = load_peaks(IRASA_DIR)
    print(f"  IRASA: {n_i} subjects, {len(irasa_peaks):,} peaks")

    fooof_enrich = compute_enrichment(fooof_peaks)
    irasa_enrich = compute_enrichment(irasa_peaks)

    # Side by side
    print(f"\n  {'Band':<12} {'Position':<15} {'FOOOF':>8} {'IRASA':>8} {'Agree?':>8}")
    print("  " + "-" * 55)

    agree = 0
    total = 0
    for band in BAND_ORDER:
        f_band = fooof_enrich[fooof_enrich.band == band]
        i_band = irasa_enrich[irasa_enrich.band == band]
        for _, f_row in f_band.iterrows():
            pos = f_row['position']
            i_row = i_band[i_band.position == pos]
            if len(i_row) > 0:
                f_val = f_row['enrichment_pct']
                i_val = i_row.iloc[0]['enrichment_pct']
                if pd.notna(f_val) and pd.notna(i_val):
                    same = np.sign(f_val) == np.sign(i_val) or abs(f_val) < 5 or abs(i_val) < 5
                    agree += int(same)
                    total += 1
                    mark = '✓' if same else '✗'
                    print(f"  {band:<12} {pos:<15} {f_val:>+8.0f}% {i_val:>+8.0f}% {mark:>8}")

    print(f"\n  Sign agreement: {agree}/{total} ({agree/total*100:.0f}%)")

    merged = fooof_enrich.merge(irasa_enrich, on=['band', 'position'],
                                 suffixes=('_fooof', '_irasa'))
    merged.to_csv(os.path.join(OUT_DIR, 'tdbrain_fooof_irasa_enrichment.csv'), index=False)


# =============================================================
# 5. Cross-Band Coupling
# =============================================================
def analysis_cross_band():
    print("\n" + "=" * 70)
    print("5. CROSS-BAND COUPLING")
    print("=" * 70)

    enrich_df = pd.read_csv(os.path.join(OUT_DIR.replace('tdbrain_analysis', 'f0_760_reanalysis'),
                                          'tdbrain_per_subject_enrichment.csv'))

    # Key cross-band test: does alpha mountain predict beta-low U-shape?
    pairs = [
        ('alpha_mountain', 'beta_low_ushape', 'Alpha mountain → Beta-low U-shape'),
        ('alpha_mountain', 'beta_low_ramp_depth', 'Alpha mountain → Beta-low ramp'),
        ('alpha_attractor', 'beta_low_center_depletion', 'Alpha attractor → Beta-low center'),
        ('alpha_asymmetry', 'beta_high_asymmetry', 'Alpha asymmetry → Beta-high asymmetry'),
        ('theta_noble_1', 'alpha_noble_1', 'Theta noble_1 → Alpha noble_1'),
    ]

    print(f"\n  {'Pair':<50} {'ρ':>8} {'p':>10} {'N':>6}")
    print("  " + "-" * 80)

    for col1, col2, desc in pairs:
        if col1 in enrich_df.columns and col2 in enrich_df.columns:
            valid = enrich_df.dropna(subset=[col1, col2])
            if len(valid) > 30:
                rho, p = stats.spearmanr(valid[col1], valid[col2])
                sig = '*' if p < 0.05 else ' '
                print(f"  {desc:<50} {rho:>+8.3f} {p:>9.4f}{sig} {len(valid):>6}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TDBRAIN REMAINING ANALYSES")
    print("=" * 70)

    analysis_personality()
    analysis_cognitive()
    analysis_troughs()
    analysis_irasa_comparison()
    analysis_cross_band()

    print(f"\n\nAll results saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
