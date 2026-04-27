#!/usr/bin/env python3
"""
Build Part V additions to the Dryad data package.

Consumes cached outputs in outputs/iaf_anchored/ and produces:
    - dryad/iaf_estimates.csv
    - dryad/enrichment_per_subject_iaf_anchored.csv.gz
    - dryad/iaf_anchored_fdr_cognitive_lemon.csv
    - dryad/iaf_anchored_fdr_age_hbn.csv
    - dryad/iaf_anchored_fdr_age_dortmund.csv
    - dryad/iaf_anchored_power_matched_subsamples.csv
    - dryad/iaf_anchored_power_matched_summary.csv
    - dryad/iaf_anchored_reassignment_detail.csv
    - dryad/iaf_anchored_sanity_check.txt
    - dryad/iaf_anchored_tertile_stratified.csv
    - dryad/iaf_anchored_trimmed_pool.csv
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'outputs', 'iaf_anchored')
DST = os.path.join(BASE, 'dryad')
os.makedirs(DST, exist_ok=True)

# Demographic loaders for age attachment (for tertile stratification)
sys.path.insert(0, os.path.join(BASE, 'scripts'))
from iaf_anchored_full_pool import load_hbn_age
from iaf_anchored_enrichment import feature_base_names

DATASETS = (
    ['lemon', 'dortmund', 'eegmmidb', 'chbmp', 'tdbrain']
    + [f'hbn_R{i}' for i in range(1, 12)]
)

TARGETS = [
    'alpha_boundary',
    'gamma_noble_4',
    'gamma_asymmetry',
    'beta_high_inv_noble_6',
    'alpha_asymmetry',  # negative control
]


def spearman_safe(x, y):
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 20:
        return np.nan
    rho, _ = stats.spearmanr(x[v], y[v])
    return float(rho)


def build_iaf_estimates_and_enrichment():
    """Extract per-subject IAF and concatenate per-subject enrichment."""
    iaf_rows = []
    enr_parts = []
    for ds in DATASETS:
        src = os.path.join(SRC, f'{ds}_per_subject.csv')
        if not os.path.exists(src):
            print(f"  skipping {ds} (no cache)")
            continue
        df = pd.read_csv(src)
        if len(df) == 0:
            continue
        df = df.copy()
        df['dataset'] = ds

        # IAF standalone: subject, dataset, iaf, f0_i
        iaf_rows.append(df[['subject', 'dataset', 'iaf', 'f0_i',
                            'n_peaks_total']].copy())
        # Full per-subject enrichment: keep all columns, ensure subject/dataset
        # columns come first
        cols = ['subject', 'dataset', 'iaf', 'f0_i', 'n_peaks_total']
        rest = [c for c in df.columns if c not in cols]
        enr_parts.append(df[cols + rest])

    iaf = pd.concat(iaf_rows, ignore_index=True)
    iaf.to_csv(os.path.join(DST, 'iaf_estimates.csv'), index=False)
    print(f"  iaf_estimates.csv: {len(iaf)} subjects")

    enr = pd.concat(enr_parts, ignore_index=True, sort=False)
    enr_path = os.path.join(DST, 'enrichment_per_subject_iaf_anchored.csv.gz')
    enr.to_csv(enr_path, index=False, compression='gzip')
    size_mb = os.path.getsize(enr_path) / 1e6
    print(f"  enrichment_per_subject_iaf_anchored.csv.gz: "
          f"{len(enr)} rows, {size_mb:.1f} MB")


def copy_fdr_and_power_files():
    """Simple copies with renaming to the iaf_anchored_ prefix."""
    mapping = {
        'full_cognitive_fdr.csv': 'iaf_anchored_fdr_cognitive_lemon.csv',
        'hbn_pool_age_fdr.csv':   'iaf_anchored_fdr_age_hbn.csv',
        'dortmund_age_fdr.csv':   'iaf_anchored_fdr_age_dortmund.csv',
        'power_matched_hbn_subsamples.csv': 'iaf_anchored_power_matched_subsamples.csv',
        'power_matched_summary.csv':        'iaf_anchored_power_matched_summary.csv',
        'hbn_reassignment_detail.csv':      'iaf_anchored_reassignment_detail.csv',
        'sanity_check.txt':                 'iaf_anchored_sanity_check.txt',
    }
    for src_name, dst_name in mapping.items():
        src_path = os.path.join(SRC, src_name)
        if not os.path.exists(src_path):
            print(f"  MISSING: {src_name}")
            continue
        shutil.copy(src_path, os.path.join(DST, dst_name))
        size_kb = os.path.getsize(src_path) / 1024
        print(f"  {dst_name}: {size_kb:.1f} KB")


def build_tertile_and_trimmed_csvs():
    """Generate per-feature tertile-stratified and trimmed-pool CSVs
    for the 5 sanity-check target features.
    """
    # Rebuild HBN pool
    parts = []
    for i in range(1, 12):
        f = os.path.join(SRC, f'hbn_R{i}_per_subject.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df['dataset'] = f'hbn_R{i}'
        parts.append(df)
    pool = pd.concat(parts, ignore_index=True)
    hbn_age = load_hbn_age()
    pool['age'] = pool['subject'].map(hbn_age)
    pool = pool.dropna(subset=['age']).copy()

    q33, q67 = np.quantile(pool['iaf'], [1/3, 2/3])

    # Tertile-stratified table
    tert_rows = []
    for feat in TARGETS:
        for label, mask in [('low', pool['iaf'] <= q33),
                            ('mid', (pool['iaf'] > q33) & (pool['iaf'] < q67)),
                            ('high', pool['iaf'] >= q67)]:
            sub = pool[mask]
            iaf_range = f"{sub['iaf'].min():.2f}-{sub['iaf'].max():.2f}"
            rp = spearman_safe(sub[f'pop_{feat}'].values, sub['age'].values)
            ri = spearman_safe(sub[f'iaf_{feat}'].values, sub['age'].values)
            tert_rows.append({
                'feature': feat,
                'iaf_tertile': label,
                'iaf_range_hz': iaf_range,
                'n_subjects': int(len(sub)),
                'rho_age_pop': round(rp, 4) if np.isfinite(rp) else np.nan,
                'rho_age_iaf': round(ri, 4) if np.isfinite(ri) else np.nan,
                'amplification': round(abs(ri) - abs(rp), 4)
                                 if np.isfinite(rp) and np.isfinite(ri) else np.nan,
            })
    tert = pd.DataFrame(tert_rows)
    tert.to_csv(os.path.join(DST, 'iaf_anchored_tertile_stratified.csv'),
                index=False)
    print(f"  iaf_anchored_tertile_stratified.csv: {len(tert)} rows")

    # Trimmed-pool comparison (5% each tail)
    lo_iaf, hi_iaf = np.percentile(pool['iaf'], [5, 95])
    trim = pool[(pool['iaf'] >= lo_iaf) & (pool['iaf'] <= hi_iaf)]
    trim_rows = []
    for feat in TARGETS:
        rp_full = spearman_safe(pool[f'pop_{feat}'].values, pool['age'].values)
        ri_full = spearman_safe(pool[f'iaf_{feat}'].values, pool['age'].values)
        rp_trim = spearman_safe(trim[f'pop_{feat}'].values, trim['age'].values)
        ri_trim = spearman_safe(trim[f'iaf_{feat}'].values, trim['age'].values)
        trim_rows.append({
            'feature': feat,
            'n_full_pool': int(len(pool)),
            'rho_age_pop_full': round(rp_full, 4),
            'rho_age_iaf_full': round(ri_full, 4),
            'amplification_full': round(abs(ri_full) - abs(rp_full), 4),
            'n_trimmed': int(len(trim)),
            'iaf_range_trimmed_hz': f"{lo_iaf:.2f}-{hi_iaf:.2f}",
            'rho_age_pop_trimmed': round(rp_trim, 4),
            'rho_age_iaf_trimmed': round(ri_trim, 4),
            'amplification_trimmed': round(abs(ri_trim) - abs(rp_trim), 4),
        })
    tr = pd.DataFrame(trim_rows)
    tr.to_csv(os.path.join(DST, 'iaf_anchored_trimmed_pool.csv'), index=False)
    print(f"  iaf_anchored_trimmed_pool.csv: {len(tr)} rows")


def main():
    print("Building Part V additions to dryad/...")
    print("\n-- iaf_estimates + per-subject enrichment --")
    build_iaf_estimates_and_enrichment()
    print("\n-- FDR tables, power-matched, reassignment --")
    copy_fdr_and_power_files()
    print("\n-- Tertile-stratified + trimmed sanity CSVs --")
    build_tertile_and_trimmed_csvs()
    print("\nDone.")


if __name__ == '__main__':
    main()
