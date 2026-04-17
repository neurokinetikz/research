#!/usr/bin/env python3
"""
SRM Hz-Weighted Spectral Differentiation Analysis
===================================================
Uses the paper's Hz-weighted Voronoi machinery from run_all_f0_760_analyses.py
for per-subject enrichment, cognitive correlations, age correlations, and
test-retest reliability. This replaces the incorrect u-space analysis.
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

# Import Hz-weighted machinery from the main analysis script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all_f0_760_analyses import (
    load_subject_enrichments, get_enrich_cols, per_subject_enrichment,
    run_correlations, print_correlation_summary, icc_21,
    BAND_ORDER, POS_NAMES, MIN_POWER_PCT, MIN_PEAKS_PER_BAND,
)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'srm_spectral_differentiation')
os.makedirs(OUT_DIR, exist_ok=True)


def run_cognitive_srm():
    print("=" * 70)
    print("  SRM COGNITIVE CORRELATIONS (Hz-weighted)")
    print("=" * 70)

    peak_dir = os.path.join(BASE_DIR, 'exports_adaptive_v4', 'srm')
    df = load_subject_enrichments(peak_dir)
    print(f"  Subjects: {len(df)}")

    # Load SRM participants
    ptcp_path = os.path.join(peak_dir, 'participants.tsv')
    if not os.path.exists(ptcp_path):
        print(f"  participants.tsv not found")
        return
    ptcp = pd.read_csv(ptcp_path, sep='\t')
    df = df.merge(ptcp, left_on='subject', right_on='participant_id', how='left')

    cog_cols = [c for c in ptcp.columns if c not in ('participant_id', 'age', 'sex')]
    for c in cog_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    enrich_cols = get_enrich_cols(df)
    print(f"  Enrichment features: {len(enrich_cols)}")
    print(f"  Cognitive measures: {len(cog_cols)}")

    # Cognitive correlations
    rdf = run_correlations(df, enrich_cols, cog_cols)
    print_correlation_summary(rdf, 'SRM Cognitive')
    rdf.to_csv(os.path.join(OUT_DIR, 'cognitive_hz_weighted.csv'), index=False)

    # Age correlations
    if 'age' in df.columns:
        print(f"\n  --- Age Correlations ---")
        adf = run_correlations(df, enrich_cols, ['age'])
        print_correlation_summary(adf, 'SRM Age', top_n=15)
        adf.to_csv(os.path.join(OUT_DIR, 'age_hz_weighted.csv'), index=False)

        # Age-partialed cognitive (for any with |rho| > 0.20)
        print(f"\n  --- Age-Partialed Cognitive (|rho|>0.20) ---")
        if len(rdf) > 0:
            candidates = rdf[rdf['abs_rho'] > 0.20]
            n_partial = 0
            for _, row in candidates.iterrows():
                feat = row['feature']
                cog = row['target']
                valid = df[[feat, cog, 'age']].dropna()
                if len(valid) < 30:
                    continue
                feat_resid = valid[feat] - np.polyval(
                    np.polyfit(valid['age'], valid[feat], 1), valid['age'])
                cog_resid = valid[cog] - np.polyval(
                    np.polyfit(valid['age'], valid[cog], 1), valid['age'])
                rho_p, p_p = stats.spearmanr(feat_resid, cog_resid)
                if abs(rho_p) > 0.15:
                    print(f"    {feat:<35s} × {cog:<12s} "
                          f"raw={row['rho']:>+.3f} partial={rho_p:>+.3f} p={p_p:.4f}")
                    n_partial += 1
            print(f"  Age-partialed with |rho|>0.15: {n_partial}")

    return df


def run_reliability_srm():
    print("\n" + "=" * 70)
    print("  SRM TEST-RETEST RELIABILITY (Hz-weighted)")
    print("=" * 70)

    for method, label in [('exports_adaptive_v4', 'FOOOF'),
                          ('exports_irasa_v4', 'IRASA')]:
        t1_dir = os.path.join(BASE_DIR, method, 'srm')
        t2_dir = os.path.join(BASE_DIR, method, 'srm_t2')

        if not os.path.exists(t2_dir):
            print(f"\n  {label}: ses-t2 not found at {t2_dir}")
            continue

        t1_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                    for f in glob.glob(os.path.join(t1_dir, '*_peaks.csv'))}
        t2_files = {os.path.basename(f).replace('_peaks.csv', ''): f
                    for f in glob.glob(os.path.join(t2_dir, '*_peaks.csv'))}
        matched = sorted(set(t1_files.keys()) & set(t2_files.keys()))
        print(f"\n  {label}: {len(matched)} matched subjects")

        if len(matched) < 10:
            continue

        data_t1, data_t2 = {}, {}
        for sub_id in matched:
            p1 = pd.read_csv(t1_files[sub_id])
            p2 = pd.read_csv(t2_files[sub_id])
            # Apply power filter
            if 'power' in p1.columns and MIN_POWER_PCT > 0:
                for df_p in [p1, p2]:
                    filtered = []
                    for octave in df_p['phi_octave'].unique():
                        bp = df_p[df_p.phi_octave == octave]
                        if len(bp) >= 2:
                            thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                            filtered.append(bp[bp['power'] >= thresh])
                        else:
                            filtered.append(bp)
                    if octave == p1['phi_octave'].unique()[0]:
                        p1 = pd.concat(filtered, ignore_index=True)
                    else:
                        p2 = pd.concat(filtered, ignore_index=True)
            data_t1[sub_id] = per_subject_enrichment(p1)
            data_t2[sub_id] = per_subject_enrichment(p2)

        all_features = set()
        for d in list(data_t1.values()) + list(data_t2.values()):
            all_features.update(k for k in d.keys()
                              if not k.endswith('_n_peaks'))

        results = []
        for feat in sorted(all_features):
            vals_t1, vals_t2 = [], []
            for sub_id in matched:
                v1 = data_t1[sub_id].get(feat, np.nan)
                v2 = data_t2[sub_id].get(feat, np.nan)
                if not np.isnan(v1) and not np.isnan(v2):
                    vals_t1.append(v1)
                    vals_t2.append(v2)
            if len(vals_t1) < 10:
                continue
            x, y = np.array(vals_t1), np.array(vals_t2)
            r_val, _ = stats.pearsonr(x, y)
            icc = icc_21(x, y)
            results.append({'feature': feat, 'n': len(vals_t1),
                           'pearson_r': r_val, 'icc': icc})

        rdf = pd.DataFrame(results)
        if len(rdf) > 0:
            print(f"  Features computed: {len(rdf)}")
            print(f"  Mean ICC: {rdf['icc'].mean():.3f}")
            print(f"  Mean Pearson r: {rdf['pearson_r'].mean():.3f}")

            # Summary by band
            print(f"\n  ICC by band:")
            for band in BAND_ORDER:
                band_feats = rdf[rdf['feature'].str.startswith(band + '_')]
                if len(band_feats) > 0:
                    print(f"    {band:<12s} mean={band_feats['icc'].mean():.3f} "
                          f"range=[{band_feats['icc'].min():.3f}, {band_feats['icc'].max():.3f}] "
                          f"N_feat={len(band_feats)}")

            # Derived metrics
            derived = ['mountain', 'ushape', 'peak_height', 'ramp_depth',
                       'center_depletion', 'asymmetry']
            derived_rows = rdf[rdf['feature'].apply(
                lambda x: any(x.endswith('_' + d) for d in derived))]
            if len(derived_rows) > 0:
                print(f"\n  Derived metric ICCs:")
                for _, r in derived_rows.sort_values('icc', ascending=False).iterrows():
                    print(f"    {r['feature']:<35s} ICC={r['icc']:>+.3f} r={r['pearson_r']:>+.3f}")

            # Top 15
            print(f"\n  Top 15 features:")
            for _, r in rdf.sort_values('icc', ascending=False).head(15).iterrows():
                print(f"    {r['feature']:<35s} ICC={r['icc']:>+.3f} r={r['pearson_r']:>+.3f} N={r['n']}")

            rdf.to_csv(os.path.join(OUT_DIR, f'reliability_hz_{label.lower()}.csv'), index=False)


if __name__ == '__main__':
    df = run_cognitive_srm()
    run_reliability_srm()
    print(f"\n  Results saved to: {OUT_DIR}")
