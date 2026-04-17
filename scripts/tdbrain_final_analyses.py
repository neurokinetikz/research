#!/usr/bin/env python3
"""
TDBRAIN Final Analyses
======================

Runs remaining analyses:
  1. IRASA age/diagnostic/personality correlations
  2. Age-partialed cognitive correlations
  3. Power sensitivity analysis
  4. IRASA regional and trough analyses

Usage:
    python scripts/tdbrain_final_analyses.py
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

sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from run_all_f0_760_analyses import (
    per_subject_enrichment, load_subject_enrichments, get_enrich_cols,
    run_correlations, print_correlation_summary, MIN_POWER_PCT
)

MIN_POWER_PCT_DEFAULT = 50


def load_participants():
    df = pd.read_csv(PARTICIPANTS_PATH, sep='\t')
    df['age_float'] = df['age'].str.replace(',', '.').astype(float)
    pid_str = df['participants_ID'].astype(str)
    df['subject_key'] = pid_str.where(pid_str.str.startswith('sub-'), 'sub-' + pid_str)
    df = df[df['DISC/REP'] == 'DISCOVERY']

    df['dx_group'] = 'OTHER'
    df.loc[df['indication'] == 'HEALTHY', 'dx_group'] = 'HEALTHY'
    df.loc[df['indication'].str.contains('ADHD', na=False) &
           ~df['indication'].str.contains('MDD', na=False), 'dx_group'] = 'ADHD'
    df.loc[df['indication'].str.contains('MDD', na=False) &
           ~df['indication'].str.contains('ADHD', na=False), 'dx_group'] = 'MDD'

    # Cognitive
    for col in ['avg_rt_oddb_CP', 'avg_rt_wm_CP', 'n_oddb_CP', 'n_oddb_FN',
                'n_wm_CP', 'n_wm_FN']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # NEO-FFI
    neo_cols = [f'neoFFI_q{i}' for i in range(1, 61)]
    for c in neo_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['NEO_N'] = df[[f'neoFFI_q{i}' for i in range(1, 13)]].sum(axis=1, min_count=6)
    df['NEO_E'] = df[[f'neoFFI_q{i}' for i in range(13, 25)]].sum(axis=1, min_count=6)
    df['NEO_O'] = df[[f'neoFFI_q{i}' for i in range(25, 37)]].sum(axis=1, min_count=6)
    df['NEO_A'] = df[[f'neoFFI_q{i}' for i in range(37, 49)]].sum(axis=1, min_count=6)
    df['NEO_C'] = df[[f'neoFFI_q{i}' for i in range(49, 61)]].sum(axis=1, min_count=6)

    return df


def merge_enrichment_with_demo(peak_dir, demo):
    """Load per-subject enrichment and merge with demographics."""
    df = load_subject_enrichments(peak_dir)
    df = df.merge(demo[['subject_key', 'age_float', 'gender', 'dx_group',
                         'n_oddb_CP', 'n_oddb_FN', 'n_wm_CP', 'n_wm_FN',
                         'NEO_N', 'NEO_E', 'NEO_O', 'NEO_A', 'NEO_C']],
                  left_on='subject', right_on='subject_key', how='inner')
    df['age'] = df['age_float']
    return df


# =============================================================
# 1. IRASA Age/Diagnostic/Personality
# =============================================================
def analysis_irasa_full():
    from statsmodels.stats.multitest import multipletests

    print("\n" + "=" * 70)
    print("1. IRASA FULL ANALYSIS (Age, Diagnostic, Personality)")
    print("=" * 70)

    demo = load_participants()
    df = merge_enrichment_with_demo(IRASA_DIR, demo)
    print(f"  IRASA subjects: {len(df)}")

    enrich_cols = get_enrich_cols(df)

    # Age
    n_age = df['age'].notna().sum()
    age_rdf = run_correlations(df, enrich_cols, ['age'])
    n_sig = age_rdf['significant'].sum() if len(age_rdf) > 0 else 0
    print(f"\n  IRASA Age × enrichment: {n_sig}/{len(age_rdf)} FDR survivors (N={n_age})")
    age_rdf.to_csv(os.path.join(OUT_DIR, 'tdbrain_irasa_age_correlations.csv'), index=False)

    # Compare FOOOF vs IRASA age correlations
    fooof_age = pd.read_csv(os.path.join(OUT_DIR.replace('tdbrain_analysis', 'f0_760_reanalysis'),
                                          'tdbrain_age_correlations.csv'))
    print(f"  FOOOF Age FDR survivors: {fooof_age['significant'].sum()}")
    print(f"  IRASA Age FDR survivors: {n_sig}")

    # Merge and compare signs
    merged = fooof_age[['feature', 'rho']].merge(
        age_rdf[['feature', 'rho']], on='feature', suffixes=('_fooof', '_irasa'))
    if len(merged) > 0:
        same_sign = (np.sign(merged['rho_fooof']) == np.sign(merged['rho_irasa'])).sum()
        print(f"  Sign agreement: {same_sign}/{len(merged)} ({same_sign/len(merged)*100:.0f}%)")
        r, p = stats.pearsonr(merged['rho_fooof'], merged['rho_irasa'])
        print(f"  FOOOF-IRASA age rho correlation: r = {r:.3f} (p = {p:.1e})")

    # Diagnostic (ADHD vs MDD)
    adults = df[df['age'] >= 18]
    adhd = adults[adults.dx_group == 'ADHD']
    mdd = adults[adults.dx_group == 'MDD']
    if len(adhd) > 10 and len(mdd) > 10:
        print(f"\n  IRASA ADHD (N={len(adhd)}) vs MDD (N={len(mdd)}):")
        dx_results = []
        for col in enrich_cols:
            v1 = adhd[col].dropna()
            v2 = mdd[col].dropna()
            if len(v1) > 10 and len(v2) > 10:
                u, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                pooled_sd = np.sqrt((v1.std()**2 + v2.std()**2) / 2)
                d = (v1.mean() - v2.mean()) / pooled_sd if pooled_sd > 0 else 0
                dx_results.append({'feature': col, 'd': d, 'p': p})
        dx_df = pd.DataFrame(dx_results)
        if len(dx_df) > 0:
            rej, pfdr, _, _ = multipletests(dx_df['p'].values, method='fdr_bh', alpha=0.05)
            dx_df['p_fdr'] = pfdr
            dx_df['significant'] = rej
            print(f"    FDR survivors: {rej.sum()}/{len(dx_df)}")
            dx_df.to_csv(os.path.join(OUT_DIR, 'tdbrain_irasa_adhd_vs_mdd.csv'), index=False)

    # Personality
    personality_cols = ['NEO_N', 'NEO_E', 'NEO_O', 'NEO_A', 'NEO_C']
    valid_pers = df[personality_cols].notna().all(axis=1).sum()
    if valid_pers > 50:
        pers_rdf = run_correlations(df, enrich_cols, personality_cols)
        n_pers_sig = pers_rdf['significant'].sum() if len(pers_rdf) > 0 else 0
        print(f"\n  IRASA Personality × enrichment: {n_pers_sig}/{len(pers_rdf)} FDR survivors (N={valid_pers})")
        pers_rdf.to_csv(os.path.join(OUT_DIR, 'tdbrain_irasa_personality_correlations.csv'), index=False)


# =============================================================
# 2. Age-Partialed Cognitive Correlations
# =============================================================
def analysis_cognitive_age_partialed():
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import rankdata

    print("\n" + "=" * 70)
    print("2. AGE-PARTIALED COGNITIVE CORRELATIONS")
    print("=" * 70)

    demo = load_participants()
    df = merge_enrichment_with_demo(FOOOF_DIR, demo)
    enrich_cols = get_enrich_cols(df)

    cog_cols = {'n_oddb_CP': 'Oddball correct', 'n_oddb_FN': 'Oddball false neg',
                'n_wm_CP': 'WM correct', 'n_wm_FN': 'WM false neg'}

    for cog_col, cog_name in cog_cols.items():
        valid = df.dropna(subset=[cog_col, 'age']).copy()
        if len(valid) < 50:
            continue

        # Raw correlations
        raw_results = []
        for col in enrich_cols:
            v = valid.dropna(subset=[col])
            if len(v) < 30:
                continue
            rho, p = stats.spearmanr(v[cog_col], v[col])
            raw_results.append({'feature': col, 'rho_raw': rho, 'p_raw': p})
        raw_df = pd.DataFrame(raw_results)

        # Age-partialed: residualize both on age
        partial_results = []
        age_rank = rankdata(valid['age'].values)
        for col in enrich_cols:
            v = valid.dropna(subset=[col])
            if len(v) < 30:
                continue
            feat_rank = rankdata(v[col].values)
            cog_vals = v[cog_col].values
            age_r = rankdata(v['age'].values)

            # Residualize feature on age
            sl_f, int_f, _, _, _ = stats.linregress(age_r, feat_rank)
            feat_resid = feat_rank - (sl_f * age_r + int_f)
            # Residualize cognitive on age
            sl_c, int_c, _, _, _ = stats.linregress(age_r, cog_vals)
            cog_resid = cog_vals - (sl_c * age_r + int_c)

            rho_partial, p_partial = stats.spearmanr(cog_resid, feat_resid)
            partial_results.append({'feature': col, 'rho_partial': rho_partial, 'p_partial': p_partial})

        partial_df = pd.DataFrame(partial_results)

        # Merge
        if len(raw_df) > 0 and len(partial_df) > 0:
            merged = raw_df.merge(partial_df, on='feature')
            rej_raw, pfdr_raw, _, _ = multipletests(merged['p_raw'].values, method='fdr_bh', alpha=0.05)
            rej_par, pfdr_par, _, _ = multipletests(merged['p_partial'].values, method='fdr_bh', alpha=0.05)
            merged['fdr_raw'] = pfdr_raw
            merged['sig_raw'] = rej_raw
            merged['fdr_partial'] = pfdr_par
            merged['sig_partial'] = rej_par

            n_raw = rej_raw.sum()
            n_partial = rej_par.sum()
            attenuation = (1 - n_partial / max(n_raw, 1)) * 100 if n_raw > 0 else 0

            print(f"\n  {cog_name} (N={len(valid)}):")
            print(f"    Raw FDR survivors: {n_raw}")
            print(f"    Age-partialed FDR survivors: {n_partial}")
            print(f"    Attenuation: {attenuation:.0f}%")

            if n_partial > 0:
                sig_rows = merged[merged.sig_partial].copy()
                sig_rows['abs_partial'] = sig_rows['rho_partial'].abs()
                for _, r in sig_rows.nlargest(3, 'abs_partial').iterrows():
                    print(f"      {r['feature']}: raw ρ={r['rho_raw']:+.3f}, partial ρ={r['rho_partial']:+.3f}")

            merged.to_csv(os.path.join(OUT_DIR, f'tdbrain_cognitive_age_partialed_{cog_col}.csv'), index=False)


# =============================================================
# 3. Power Sensitivity Analysis
# =============================================================
def analysis_power_sensitivity():
    print("\n" + "=" * 70)
    print("3. POWER SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Test enrichment stability across power filter thresholds
    from run_all_f0_760_analyses import load_peaks, compute_enrichment

    thresholds = [0, 25, 50, 75]
    results = []

    for thresh in thresholds:
        peaks, n = load_peaks(FOOOF_DIR, min_power_pct=thresh)
        if peaks is None:
            continue
        enrich = compute_enrichment(peaks)
        n_peaks = len(peaks)
        print(f"  Power filter {thresh}%: {n_peaks:,} peaks from {n} subjects")

        for _, row in enrich.iterrows():
            results.append({
                'power_threshold': thresh, 'band': row['band'],
                'position': row['position'], 'enrichment_pct': row['enrichment_pct'],
                'n_peaks': n_peaks,
            })

    df = pd.DataFrame(results)

    # Compute cross-threshold correlation for each band
    print(f"\n  Enrichment profile stability across power thresholds:")
    for band in ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']:
        profiles = {}
        for thresh in thresholds:
            sub = df[(df.band == band) & (df.power_threshold == thresh)]
            if len(sub) > 0:
                profiles[thresh] = sub.set_index('position')['enrichment_pct'].dropna()

        if 0 in profiles and 50 in profiles:
            common = profiles[0].index.intersection(profiles[50].index)
            if len(common) > 5:
                r, p = stats.pearsonr(profiles[0][common], profiles[50][common])
                print(f"    {band}: r(0% vs 50%) = {r:.3f}")

    df.to_csv(os.path.join(OUT_DIR, 'tdbrain_power_sensitivity.csv'), index=False)


# =============================================================
# 4. IRASA Regional and Trough
# =============================================================
def analysis_irasa_trough_regional():
    print("\n" + "=" * 70)
    print("4. IRASA TROUGH DETECTION AND REGIONAL")
    print("=" * 70)

    REGIONS = {
        'frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4'],
        'central': ['C3', 'Cz', 'C4'],
        'temporal': ['T7', 'T8'],
        'parietal': ['CP3', 'CPz', 'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8'],
        'occipital': ['O1', 'Oz', 'O2'],
    }

    # Load IRASA peaks with channel info
    all_freqs = []
    region_freqs = {r: [] for r in REGIONS}
    files = sorted(glob.glob(os.path.join(IRASA_DIR, '*_peaks.csv')))

    for f in files:
        try:
            df = pd.read_csv(f, usecols=['channel', 'freq', 'power', 'phi_octave'])
        except Exception:
            continue
        filtered = []
        for octave in df['phi_octave'].unique():
            bp = df[df.phi_octave == octave]
            if len(bp) == 0:
                continue
            thresh = bp['power'].quantile(0.5)
            filtered.append(bp[bp['power'] >= thresh])
        if filtered:
            df = pd.concat(filtered, ignore_index=True)
            all_freqs.extend(df['freq'].values)
            for region, channels in REGIONS.items():
                rdf = df[df['channel'].isin(channels)]
                region_freqs[region].extend(rdf['freq'].values)

    all_freqs = np.array(all_freqs)
    print(f"  IRASA total peaks: {len(all_freqs):,}")

    # Pooled troughs
    log_freqs = np.log(all_freqs)
    log_edges = np.linspace(np.log(3), np.log(55), 1001)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)
    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=8)
    median_val = np.median(smoothed[smoothed > 0])
    trough_idx, _ = find_peaks(-smoothed, prominence=median_val * 0.05, distance=1000 // 30)
    troughs = hz_centers[trough_idx]
    troughs = troughs[(troughs > 4) & (troughs < 50)]
    print(f"  IRASA pooled troughs: {np.round(troughs, 2)} Hz")

    if len(troughs) >= 2:
        ratios = troughs[1:] / troughs[:-1]
        geo = np.exp(np.mean(np.log(ratios)))
        print(f"  Geo mean ratio: {geo:.4f} (φ = {PHI:.4f})")

    # Regional troughs
    print(f"\n  IRASA troughs by region:")
    for region in ['frontal', 'central', 'temporal', 'parietal', 'occipital']:
        rf = np.array(region_freqs[region])
        if len(rf) < 1000:
            continue
        log_rf = np.log(rf)
        counts_r, _ = np.histogram(log_rf, bins=log_edges)
        smoothed_r = gaussian_filter1d(counts_r.astype(float), sigma=8)
        med_r = np.median(smoothed_r[smoothed_r > 0])
        tidx, _ = find_peaks(-smoothed_r, prominence=med_r * 0.05, distance=1000 // 30)
        t_r = hz_centers[tidx]
        t_r = t_r[(t_r > 4) & (t_r < 50)]
        print(f"    {region}: {np.round(t_r, 2)} Hz ({len(rf):,} peaks)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TDBRAIN FINAL ANALYSES")
    print("=" * 70)

    analysis_irasa_full()
    analysis_cognitive_age_partialed()
    analysis_power_sensitivity()
    analysis_irasa_trough_regional()

    print(f"\n\nAll results saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
