#!/usr/bin/env python3
"""TDBRAIN clinical stratification of source-space region ratios.

Tests whether right-IFG (parsopercularis-rh), MTL (parahippocampal-lh, bankssts-lh),
and other regional ratios differ between ADHD, MDD, HEALTHY, and other clinical groups
in TDBRAIN EC and EO.

Tests §12.22's hypothesis that right-IFG is elevated in clinical (ADHD/MDD) cohorts.

Inputs:
  outputs/schumann/images/source/tdbrain_composite/stcs/  (per-subject STCs, EC)
  outputs/schumann/images/source/tdbrain_EO_composite/stcs/  (EO)
  outputs/tdbrain_analysis/tdbrain_per_subject_trough_depths.csv  (dx_group, indication)

Outputs:
  outputs/2026-04-24-tdbrain-clinical-stratification/*.csv + *.png
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
import mne

ROOT = Path(__file__).parent.parent
SOURCE_DIR = ROOT / 'outputs' / 'schumann' / 'images' / 'source'
META_CSV = ROOT / 'outputs' / 'tdbrain_analysis' / 'tdbrain_per_subject_trough_depths.csv'
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
        ('parsopercularis', 'rh'),   # right IFG (clinical hypothesis)
        ('parsopercularis', 'lh'),
        ('parahippocampal', 'rh'),   # MTL
        ('parahippocampal', 'lh'),
        ('posteriorcingulate', 'rh'),
        ('posteriorcingulate', 'lh'),
        ('caudalanteriorcingulate', 'lh'),
        ('bankssts', 'lh'),           # STS (TDBRAIN EC top-1)
        ('temporalpole', 'rh'),       # TDBRAIN EO #2
        ('medialorbitofrontal', 'rh'),
        ('lingual', 'lh'),
    ]

    # Load per-subject region ratios for EC and EO
    print('Loading TDBRAIN EC per-subject source ratios...')
    ec = _per_subject_region_ratios('tdbrain_composite', regions_hemis, masks)
    ec['condition'] = 'EC'
    print(f'  n={len(ec)}')
    print('Loading TDBRAIN EO per-subject source ratios...')
    eo = _per_subject_region_ratios('tdbrain_EO_composite', regions_hemis, masks)
    eo['condition'] = 'EO'
    print(f'  n={len(eo)}')
    combined = pd.concat([ec, eo], ignore_index=True)

    # Load clinical metadata
    meta = pd.read_csv(META_CSV)
    meta = meta[['subject', 'participants_ID', 'age_float', 'gender',
                 'dx_group', 'indication']].rename(
        columns={'subject': 'subject_id'})
    print(f'\nLoaded metadata for {len(meta)} TDBRAIN subjects')
    print('dx_group counts:', meta['dx_group'].value_counts().to_dict())

    # Merge
    m = combined.merge(meta, on='subject_id', how='inner')
    print(f'\nMerged n={len(m)} (EC={sum(m["condition"]=="EC")}, EO={sum(m["condition"]=="EO")})')
    print('dx_group in matched n:')
    print(m.groupby(['condition', 'dx_group']).size().unstack(fill_value=0))
    m.to_csv(OUT / 'tdbrain_clinical_per_subject.csv', index=False)

    # ==================== Test: dx_group × region ratio ====================
    print('\n=== Kruskal-Wallis test: dx_group × region ratio (per condition) ===')
    region_names = [f'{r}-{h}' for r, h in regions_hemis if f'{r}-{h}' in m.columns]
    results = []
    for cond in ['EC', 'EO']:
        sub = m[m['condition'] == cond]
        for reg in region_names:
            # Focus on ADHD/MDD/HEALTHY as 3-group comparison
            groups = ['ADHD', 'MDD', 'HEALTHY']
            data_by_group = [sub[sub['dx_group'] == g][reg].dropna().values for g in groups]
            ns = [len(g) for g in data_by_group]
            if min(ns) < 3:
                print(f'  [{cond}] {reg:30s} skip: group n={ns}')
                continue
            stat, p = kruskal(*data_by_group)
            medians = {g: np.median(d) if len(d) else np.nan
                       for g, d in zip(groups, data_by_group)}
            print(f'  [{cond}] {reg:30s} n={ns}  H={stat:.2f}  p={p:.3g}  '
                  f'medians ADHD={medians["ADHD"]:.3f} MDD={medians["MDD"]:.3f} HEALTHY={medians["HEALTHY"]:.3f}')
            results.append({'condition': cond, 'region_hemi': reg,
                            'n_ADHD': ns[0], 'n_MDD': ns[1], 'n_HEALTHY': ns[2],
                            'H_stat': stat, 'p_value': p,
                            'median_ADHD': medians['ADHD'],
                            'median_MDD': medians['MDD'],
                            'median_HEALTHY': medians['HEALTHY']})
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / 'tdbrain_clinical_kruskal.csv', index=False)

    # ==================== Pairwise Mann-Whitney for significant regions ====================
    print('\n=== Pairwise (ADHD vs HEALTHY, MDD vs HEALTHY, ADHD vs MDD) ===')
    pair_rows = []
    for cond in ['EC', 'EO']:
        sub = m[m['condition'] == cond]
        for reg in region_names:
            for g1, g2 in [('ADHD', 'HEALTHY'), ('MDD', 'HEALTHY'), ('ADHD', 'MDD')]:
                d1 = sub[sub['dx_group'] == g1][reg].dropna().values
                d2 = sub[sub['dx_group'] == g2][reg].dropna().values
                if len(d1) < 3 or len(d2) < 3:
                    continue
                u, p = mannwhitneyu(d1, d2, alternative='two-sided')
                # Effect size (Cohen's d on ranks / rank-biserial correlation)
                n1, n2 = len(d1), len(d2)
                rank_biserial = 1 - (2 * u) / (n1 * n2)
                pair_rows.append({
                    'condition': cond, 'region_hemi': reg,
                    'group1': g1, 'group2': g2,
                    'n1': n1, 'n2': n2,
                    'median_g1': np.median(d1), 'median_g2': np.median(d2),
                    'U': u, 'p': p, 'rank_biserial_r': rank_biserial,
                })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT / 'tdbrain_clinical_pairwise.csv', index=False)
    # Show nominally significant results
    sig = pair_df[pair_df['p'] < 0.05].sort_values('p')
    print(f"\nNominally significant (uncorrected p<0.05): {len(sig)}/{len(pair_df)}")
    if len(sig):
        print(sig[['condition', 'region_hemi', 'group1', 'group2', 'n1', 'n2',
                    'median_g1', 'median_g2', 'p', 'rank_biserial_r']].to_string(index=False))

    # ==================== Plot: right-IFG by group × condition ====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, reg in zip(axes, ['parsopercularis-rh', 'parahippocampal-lh', 'bankssts-lh']):
        plot_data = []
        plot_labels = []
        for cond in ['EC', 'EO']:
            for g in ['ADHD', 'MDD', 'HEALTHY', 'OCD']:
                d = m[(m['condition'] == cond) & (m['dx_group'] == g)][reg].dropna().values
                if len(d) < 2:
                    continue
                plot_data.append(d)
                plot_labels.append(f'{cond}\n{g}\n(n={len(d)})')
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.axhline(1.0, color='grey', lw=0.8, alpha=0.5)
        ax.set_title(reg)
        ax.set_ylabel('Ratio (median over label)')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    plt.tight_layout()
    plt.savefig(OUT / 'tdbrain_clinical_region_by_group.png', dpi=120)
    plt.close()

    print(f'\nSaved outputs to {OUT}')


if __name__ == '__main__':
    main()
