#!/usr/bin/env python3
"""HBN clinical dimension × Q4 SR1 source-space region ratio.

Tests §12.55's clinical-suppression hypothesis in HBN pediatric cohort using
dimensional clinical scores (p_factor, attention, internalizing, externalizing)
from participants.tsv.

Prediction from §12.55:
- Higher clinical severity → LOWER Q4 ignition ratios (global suppression)
- Right-IFG relatively preserved (weaker correlation than DMN regions)

Inputs:
  /tmp/hbn_participants/cmi_bids_R*/participants.tsv (clinical dimensions)
  outputs/schumann/images/source/hbn_R*_composite/stcs/  (per-subject STCs)

Outputs:
  outputs/2026-04-24-tdbrain-clinical-stratification/hbn_clinical_dimensions.csv
  outputs/2026-04-24-tdbrain-clinical-stratification/hbn_clinical_dimensions.png
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import mne

ROOT = Path(__file__).parent.parent
SOURCE_DIR = ROOT / 'outputs' / 'schumann' / 'images' / 'source'
HBN_META = Path('/tmp/hbn_participants')
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


def _load_hbn_meta():
    rows = []
    for rel_dir in sorted(HBN_META.glob('cmi_bids_R*')):
        p = rel_dir / 'participants.tsv'
        if not p.exists():
            continue
        m = pd.read_csv(p, sep='\t')
        m = m.rename(columns={'participant_id': 'subject_id'})
        rel = rel_dir.name.replace('cmi_bids_', '')
        m['release'] = rel
        keep_cols = ['subject_id', 'release', 'age', 'sex',
                     'p_factor', 'attention', 'internalizing', 'externalizing']
        available = [c for c in keep_cols if c in m.columns]
        rows.append(m[available])
    return pd.concat(rows, ignore_index=True)


def _partial_spearman(x, y, z):
    """Partial Spearman correlation of x with y controlling for z."""
    from scipy.stats import spearmanr
    df = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(df) < 10:
        return np.nan, np.nan
    # Rank-transform all
    ranks = df.rank()
    # Residualize x and y against z
    from numpy.linalg import lstsq
    z_ranks = ranks['z'].values.reshape(-1, 1)
    z_aug = np.hstack([z_ranks, np.ones_like(z_ranks)])
    bx, *_ = lstsq(z_aug, ranks['x'].values, rcond=None)
    by, *_ = lstsq(z_aug, ranks['y'].values, rcond=None)
    x_res = ranks['x'].values - z_aug @ bx
    y_res = ranks['y'].values - z_aug @ by
    rho, p = spearmanr(x_res, y_res)
    return rho, p


def main():
    masks = _get_label_masks()
    regions_hemis = [
        ('parsopercularis', 'rh'),    # right IFG (preservation hypothesis)
        ('parsopercularis', 'lh'),
        ('parahippocampal', 'rh'),    # MTL (suppression hypothesis)
        ('parahippocampal', 'lh'),
        ('posteriorcingulate', 'rh'), # PCC (HBN top-1)
        ('posteriorcingulate', 'lh'),
        ('caudalanteriorcingulate', 'lh'),  # ACC (suppression hypothesis)
        ('precuneus', 'rh'),           # DMN
        ('lingual', 'lh'),             # visual
    ]

    # Load per-subject region ratios for available HBN releases
    print('Loading HBN per-subject region ratios...')
    cohorts = ['hbn_R1', 'hbn_R4', 'hbn_R11']
    all_dfs = []
    for c in cohorts:
        df = _per_subject_region_ratios(f'{c}_composite', regions_hemis, masks)
        if df.empty:
            continue
        df['release'] = c.replace('hbn_', '')
        all_dfs.append(df)
    ratios = pd.concat(all_dfs, ignore_index=True)
    print(f'  Total per-subject region ratios: {len(ratios)}')

    # Load clinical metadata
    meta = _load_hbn_meta()
    meta['p_factor'] = pd.to_numeric(meta.get('p_factor'), errors='coerce')
    meta['attention'] = pd.to_numeric(meta.get('attention'), errors='coerce')
    meta['internalizing'] = pd.to_numeric(meta.get('internalizing'), errors='coerce')
    meta['externalizing'] = pd.to_numeric(meta.get('externalizing'), errors='coerce')
    meta['age'] = pd.to_numeric(meta['age'], errors='coerce')
    print(f'  Metadata: {len(meta)} HBN subjects, clinical coverage:')
    for col in ['p_factor', 'attention', 'internalizing', 'externalizing']:
        print(f'    {col}: {meta[col].notna().sum()} non-null')

    # Merge (only match on subject_id)
    merged = ratios.merge(
        meta[['subject_id', 'age', 'sex', 'p_factor', 'attention',
              'internalizing', 'externalizing']],
        on='subject_id', how='inner')
    print(f'  Merged ratios + clinical: {len(merged)} subjects')
    merged.to_csv(OUT / 'hbn_clinical_dimensions_per_subject.csv', index=False)

    region_names = [f'{r}-{h}' for r, h in regions_hemis if f'{r}-{h}' in merged.columns]
    clinical_dims = ['p_factor', 'attention', 'internalizing', 'externalizing']

    # ==================== Raw Spearman ====================
    print('\n=== HBN clinical dimension × region ratio (Spearman, pooled) ===')
    results = []
    for reg in region_names:
        for dim in clinical_dims:
            v = merged[[reg, dim]].dropna()
            # Filter extreme outliers
            v = v[(v[reg] > 0.1) & (v[reg] < 10)]
            if len(v) < 30:
                continue
            rho, p = spearmanr(v[reg], v[dim])
            # Partial correlation controlling for age
            merged_sub = merged[[reg, dim, 'age']].dropna()
            merged_sub = merged_sub[(merged_sub[reg] > 0.1) & (merged_sub[reg] < 10)]
            rho_p, p_p = _partial_spearman(
                merged_sub[reg].values, merged_sub[dim].values, merged_sub['age'].values)
            results.append({
                'region_hemi': reg, 'clinical_dim': dim, 'n': len(v),
                'spearman_rho': rho, 'spearman_p': p,
                'partial_rho_age_ctrl': rho_p, 'partial_p_age_ctrl': p_p,
            })
    df = pd.DataFrame(results)
    df.to_csv(OUT / 'hbn_clinical_dimensions.csv', index=False)

    print('\nSorted by |spearman_rho|:')
    df_sorted = df.copy()
    df_sorted['abs_rho'] = df_sorted['spearman_rho'].abs()
    df_sorted = df_sorted.sort_values('abs_rho', ascending=False)
    for _, row in df_sorted.iterrows():
        sig = '***' if row['spearman_p'] < 0.001 else '**' if row['spearman_p'] < 0.01 else '*' if row['spearman_p'] < 0.05 else ''
        print(f"  {row['region_hemi']:25s} × {row['clinical_dim']:15s}  n={row['n']:3d}  "
              f"ρ={row['spearman_rho']:+.3f} p={row['spearman_p']:.3g} {sig}  "
              f"partial_ρ(age)={row['partial_rho_age_ctrl']:+.3f}")

    # ==================== Summary ====================
    sig = df[df['spearman_p'] < 0.05]
    print(f'\n=== Summary ===')
    print(f'Total tests: {len(df)} (9 regions × 4 dimensions)')
    print(f'Nominally significant (p<0.05): {len(sig)}')
    # Direction of effect
    if len(sig) > 0:
        negative = (sig['spearman_rho'] < 0).sum()
        print(f'  of these, {negative} are NEGATIVE (higher clinical → lower ratio)')
        print(f'  and {len(sig) - negative} are POSITIVE')
    # Bonferroni at n_tests
    bon_thresh = 0.05 / len(df)
    sig_bon = df[df['spearman_p'] < bon_thresh]
    print(f'Bonferroni-surviving (p<{bon_thresh:.4f}): {len(sig_bon)}')

    # Test the preservation hypothesis: is parsopercularis-rh (IFG) LESS correlated than DMN regions?
    dmn_regs = ['posteriorcingulate-rh', 'posteriorcingulate-lh', 'caudalanteriorcingulate-lh',
                'parahippocampal-rh', 'parahippocampal-lh']
    df['abs_rho'] = df['spearman_rho'].abs()
    ifg_rs = df[df['region_hemi']=='parsopercularis-rh']['abs_rho'].values
    dmn_rs = df[df['region_hemi'].isin(dmn_regs)]['abs_rho'].values
    print(f'\nMean |rho|: IFG-rh = {ifg_rs.mean():.3f}, DMN regions = {dmn_rs.mean():.3f}')

    # ==================== Plot ====================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_regions = ['posteriorcingulate-rh', 'posteriorcingulate-lh',
                    'parahippocampal-rh', 'caudalanteriorcingulate-lh',
                    'parsopercularis-rh']
    for ax, reg in zip(axes.flat, plot_regions):
        # Plot vs p_factor
        v = merged[[reg, 'p_factor']].dropna()
        v = v[(v[reg] > 0.1) & (v[reg] < 10)]
        if len(v) < 30:
            ax.axis('off')
            continue
        ax.scatter(v['p_factor'], v[reg], s=10, alpha=0.4, color='steelblue')
        rho, p = spearmanr(v['p_factor'], v[reg])
        # Fit
        from scipy.stats import linregress
        lr = linregress(v['p_factor'], v[reg])
        x = np.array([v['p_factor'].min(), v['p_factor'].max()])
        y = lr.slope * x + lr.intercept
        ax.plot(x, y, 'r--', alpha=0.7, label=f'slope={lr.slope:.3f}')
        ax.axhline(1.0, color='grey', lw=0.8, alpha=0.5)
        ax.set_title(f'{reg}\nn={len(v)}, ρ={rho:+.3f}, p={p:.3g}')
        ax.set_xlabel('p_factor (clinical severity)')
        ax.set_ylabel('Ratio')
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    plt.suptitle('HBN pediatric: Q4 source ratio × p_factor (clinical severity)')
    plt.tight_layout()
    plt.savefig(OUT / 'hbn_clinical_dimensions.png', dpi=120)
    plt.close()
    print(f'\nSaved plot to {OUT / "hbn_clinical_dimensions.png"}')


if __name__ == '__main__':
    main()
