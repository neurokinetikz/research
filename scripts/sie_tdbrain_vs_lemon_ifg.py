#!/usr/bin/env python3
"""LEMON healthy EO vs TDBRAIN ADHD/MDD/OTHER EO: right-IFG (parsopercularis-rh) and
related region ratio comparisons.

Tests §12.22 hypothesis: right-IFG elevated in clinical cohort relative to healthy.
Uses LEMON EO (n=155 community-sample healthy adults) as healthy comparator.

Caveat: LEMON uses 59-ch cap, TDBRAIN uses 26-ch cap. Cross-cohort differences
could reflect sensor density/spatial sampling rather than clinical status.
Interpret cautiously.

Outputs to outputs/2026-04-24-tdbrain-clinical-stratification/:
  - tdbrain_vs_lemon_ifg.csv
  - tdbrain_vs_lemon_ifg.png
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import mne

ROOT = Path(__file__).parent.parent
SOURCE_DIR = ROOT / 'outputs' / 'schumann' / 'images' / 'source'
TDBRAIN_META = ROOT / 'outputs' / 'tdbrain_analysis' / 'tdbrain_per_subject_trough_depths.csv'
LEMON_META = Path('/tmp/cohort_meta/lemon_meta.csv')
OUT = ROOT / 'outputs' / '2026-04-24-tdbrain-clinical-stratification'
OUT.mkdir(exist_ok=True)


def _get_label_masks():
    subjects_dir = os.path.dirname(mne.datasets.fetch_fsaverage(verbose=False))
    labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                         subjects_dir=subjects_dir, verbose=False)
    return {L.name: L for L in labels}


def _per_subject_region_ratios(cohort_composite, regions_hemis, masks):
    stc_dir = SOURCE_DIR / cohort_composite / 'stcs'
    if not stc_dir.is_dir():
        return pd.DataFrame()
    rows = []
    for f in sorted(stc_dir.glob('*_Q4_SR1_ratio-lh.stc')):
        sub_id = f.name.replace('_Q4_SR1_ratio-lh.stc', '')
        try:
            stc = mne.read_source_estimate(str(f).replace('-lh.stc', ''), 'fsaverage')
        except Exception:
            continue
        row = {'subject_id': sub_id}
        for region, hemi in regions_hemis:
            lbl = masks.get(f'{region}-{hemi}')
            if lbl is None:
                continue
            try:
                row[f'{region}-{hemi}'] = float(np.mean(stc.in_label(lbl).data[:, 0]))
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    masks = _get_label_masks()
    regions_hemis = [
        ('parsopercularis', 'rh'),
        ('parsopercularis', 'lh'),
        ('parahippocampal', 'rh'),
        ('parahippocampal', 'lh'),
        ('bankssts', 'lh'),
        ('bankssts', 'rh'),
        ('caudalanteriorcingulate', 'lh'),
        ('posteriorcingulate', 'rh'),
        ('lingual', 'lh'),
        ('medialorbitofrontal', 'rh'),
        ('temporalpole', 'rh'),
    ]

    print('Loading LEMON EO...')
    lemon_eo = _per_subject_region_ratios('lemon_EO_composite', regions_hemis, masks)
    lemon_eo['cohort'] = 'LEMON_EO'
    lemon_eo['dx_group'] = 'HEALTHY'  # LEMON community sample ~= healthy
    print(f'  n={len(lemon_eo)}')

    print('Loading TDBRAIN EO...')
    tdbrain_eo = _per_subject_region_ratios('tdbrain_EO_composite', regions_hemis, masks)
    tdbrain_eo['cohort'] = 'TDBRAIN_EO'
    # Merge with clinical metadata
    meta = pd.read_csv(TDBRAIN_META)[['subject', 'dx_group', 'age_float']].rename(
        columns={'subject': 'subject_id'})
    tdbrain_eo = tdbrain_eo.merge(meta, on='subject_id', how='left')
    print(f'  n={len(tdbrain_eo)}, with dx: {tdbrain_eo["dx_group"].notna().sum()}')
    print('  dx distribution:')
    print(tdbrain_eo['dx_group'].value_counts())

    # Load LEMON ages
    lemon_meta_df = pd.read_csv(LEMON_META)
    lemon_meta_df['age_bin'] = lemon_meta_df.get('Age', None)
    def _age_mid(v):
        if pd.isna(v):
            return np.nan
        s = str(v)
        if '-' in s:
            try:
                lo, hi = s.split('-')
                return (float(lo) + float(hi)) / 2
            except Exception:
                return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    lemon_meta_df['age_float'] = lemon_meta_df['Age'].apply(_age_mid)
    lemon_meta_df['subject_id'] = lemon_meta_df['ID'].astype(str).apply(
        lambda x: x if x.startswith('sub-') else f'sub-{x}')
    lemon_eo = lemon_eo.merge(lemon_meta_df[['subject_id', 'age_float']], on='subject_id', how='left')

    print(f'\nLEMON EO age: n={lemon_eo["age_float"].notna().sum()}, mean={lemon_eo["age_float"].mean():.1f}, range {lemon_eo["age_float"].min():.0f}-{lemon_eo["age_float"].max():.0f}')
    print(f'TDBRAIN EO age: n={tdbrain_eo["age_float"].notna().sum()}, mean={tdbrain_eo["age_float"].mean():.1f}, range {tdbrain_eo["age_float"].min():.0f}-{tdbrain_eo["age_float"].max():.0f}')

    # ==================== Comparisons ====================
    region_names = [f'{r}-{h}' for r, h in regions_hemis]
    results = []

    for reg in region_names:
        if reg not in tdbrain_eo.columns or reg not in lemon_eo.columns:
            continue
        lemon_vals = lemon_eo[reg].dropna().values
        # Compare against each TDBRAIN subgroup
        for grp in ['ADHD', 'MDD', 'OTHER']:
            clinical_vals = tdbrain_eo[tdbrain_eo['dx_group'] == grp][reg].dropna().values
            if len(clinical_vals) < 3:
                continue
            u, p = mannwhitneyu(clinical_vals, lemon_vals, alternative='two-sided')
            n1, n2 = len(clinical_vals), len(lemon_vals)
            rank_biserial = 1 - (2 * u) / (n1 * n2)
            results.append({
                'region_hemi': reg,
                'comparison': f'TDBRAIN_{grp}_vs_LEMON_EO',
                'n_clinical': n1, 'n_lemon': n2,
                'median_clinical': np.median(clinical_vals),
                'median_lemon': np.median(lemon_vals),
                'p': p, 'rank_biserial_r': rank_biserial,
                'direction': 'clinical>LEMON' if np.median(clinical_vals) > np.median(lemon_vals) else 'clinical<LEMON'
            })
        # Combined clinical (ADHD+MDD+OTHER) vs LEMON
        all_clinical = tdbrain_eo[tdbrain_eo['dx_group'].isin(['ADHD','MDD','OTHER'])][reg].dropna().values
        if len(all_clinical) >= 3:
            u, p = mannwhitneyu(all_clinical, lemon_vals, alternative='two-sided')
            n1, n2 = len(all_clinical), len(lemon_vals)
            rank_biserial = 1 - (2 * u) / (n1 * n2)
            results.append({
                'region_hemi': reg,
                'comparison': 'TDBRAIN_ALL_vs_LEMON_EO',
                'n_clinical': n1, 'n_lemon': n2,
                'median_clinical': np.median(all_clinical),
                'median_lemon': np.median(lemon_vals),
                'p': p, 'rank_biserial_r': rank_biserial,
                'direction': 'clinical>LEMON' if np.median(all_clinical) > np.median(lemon_vals) else 'clinical<LEMON'
            })

    df = pd.DataFrame(results)
    df.to_csv(OUT / 'tdbrain_vs_lemon_ifg.csv', index=False)

    # Sort by p for display
    print(f'\n=== TDBRAIN clinical EO vs LEMON EO (healthy) per region ===')
    df_sorted = df.sort_values('p')
    for _, row in df_sorted.iterrows():
        sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
        print(f"  {row['region_hemi']:25s}  {row['comparison']:30s}  "
              f"n_clin={row['n_clinical']:2d} vs n_lemon={row['n_lemon']:3d}  "
              f"med_clin={row['median_clinical']:.3f} vs med_lemon={row['median_lemon']:.3f}  "
              f"p={row['p']:.4g} {sig}  r={row['rank_biserial_r']:+.2f}")

    # ==================== Plot ====================
    key_regions = ['parsopercularis-rh', 'parahippocampal-lh', 'bankssts-lh',
                   'caudalanteriorcingulate-lh', 'medialorbitofrontal-rh',
                   'parsopercularis-lh']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, reg in zip(axes.flat, key_regions):
        if reg not in lemon_eo.columns:
            ax.axis('off')
            continue
        groups = []
        labels = []
        for grp, color in [('HEALTHY (LEMON)', 'green'), ('OTHER', 'grey'),
                           ('MDD', 'coral'), ('ADHD', 'crimson')]:
            if grp == 'HEALTHY (LEMON)':
                vals = lemon_eo[reg].dropna().values
            else:
                vals = tdbrain_eo[tdbrain_eo['dx_group'] == grp][reg].dropna().values
            if len(vals) < 1:
                continue
            groups.append(vals)
            labels.append(f'{grp}\n(n={len(vals)})')
        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.6,
                         showmeans=True, meanline=True,
                         meanprops={'color': 'black', 'linewidth': 1.5, 'linestyle': '--'})
        colors = ['lightgreen', 'lightgrey', 'lightcoral', 'red']
        for patch, c in zip(bp['boxes'], colors[:len(groups)]):
            patch.set_facecolor(c)
        ax.axhline(1.0, color='grey', lw=0.8, alpha=0.5)
        ax.set_title(reg)
        ax.set_ylabel('Per-subject ratio')
        ax.tick_params(axis='x', labelsize=8)
    plt.suptitle('TDBRAIN clinical (EO) vs LEMON EO (healthy comparator)\nCAVEAT: 26-ch vs 59-ch cap — protocol confound', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / 'tdbrain_vs_lemon_ifg.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\nSaved {OUT / "tdbrain_vs_lemon_ifg.png"}')


if __name__ == '__main__':
    main()
