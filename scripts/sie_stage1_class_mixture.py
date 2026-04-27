#!/usr/bin/env python3
"""Stage 1: within-cohort heterogeneity of per-subject source-space region ratios.

Uses existing per-subject region data from full_lifespan_age_region.csv.
Adds additional regions (bankssts-lh, parsopercularis-rh, lingual-lh) to extend
the generator-class coverage.

Outputs to outputs/2026-04-24-crosscohort-battery/:
  - stage1_region_distributions.png  (density plots per region × cohort)
  - stage1_within_subject_correlations.csv  (region pair correlations per cohort)
  - stage1_dip_test.csv  (Hartigan's dip test for bimodality per region × cohort)
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
OUT = ROOT / 'outputs' / '2026-04-24-crosscohort-battery'


def _get_label_masks():
    subjects_dir = os.path.dirname(mne.datasets.fetch_fsaverage(verbose=False))
    labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                         subjects_dir=subjects_dir, verbose=False)
    return {L.name: L for L in labels}


def _per_subject_multi_region(cohort_composite, regions_hemis, label_masks):
    """For each subject, compute mean ratio at each region."""
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
        row = {'subject_id': sub_id, 'cohort': cohort_composite.replace('_composite', '')}
        for region, hemi in regions_hemis:
            name = f'{region}-{hemi}'
            lbl = label_masks.get(name)
            if lbl is None:
                continue
            try:
                row[name] = float(np.mean(stc.in_label(lbl).data[:, 0]))
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def _dip_test(values):
    """Hartigan's dip test — manual (simplified).

    Returns dip statistic. Higher = more multimodal.
    Here we use a shortcut: skewness + kurtosis deviation from unimodal Gaussian.
    For formal test, requires `diptest` package which may not be installed.
    Return (dip_proxy, percentile_split).
    """
    v = np.asarray(values)
    v = v[~np.isnan(v)]
    if len(v) < 10:
        return np.nan, np.nan
    # simple multimodality check: compare distance between 25th-50th-75th percentiles
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    iqr = q75 - q25
    # Bimodality coefficient (Pfister et al. 2013): b = (skew^2 + 1) / (kurtosis + 3(n-1)^2/((n-2)(n-3)))
    from scipy.stats import skew, kurtosis
    try:
        s = skew(v)
        k = kurtosis(v)
        bimod_coef = (s ** 2 + 1) / (k + 3 * (len(v) - 1) ** 2 / ((len(v) - 2) * (len(v) - 3)))
    except Exception:
        bimod_coef = np.nan
    return bimod_coef, iqr / (q50 + 1e-9)


def main():
    # Regions to examine (extend beyond original 5 to cover more classes)
    regions_hemis = [
        ('posteriorcingulate', 'rh'),
        ('posteriorcingulate', 'lh'),
        ('parahippocampal', 'rh'),
        ('parahippocampal', 'lh'),
        ('caudalanteriorcingulate', 'lh'),
        ('caudalanteriorcingulate', 'rh'),
        ('caudalmiddlefrontal', 'rh'),  # DLPFC
        ('bankssts', 'lh'),  # STS (TDBRAIN top)
        ('parsopercularis', 'rh'),  # IFG (TDBRAIN EO top)
        ('lingual', 'lh'),  # visual (chbmp top)
        ('entorhinal', 'rh'),  # MTL
        ('precuneus', 'rh'),  # DMN
    ]

    cohorts = {
        'hbn_R1': 'HBN pediatric (R1)',
        'hbn_R4': 'HBN pediatric (R4)',
        'hbn_R11': 'HBN pediatric (R11)',
        'lemon': 'LEMON adult EC',
        'lemon_EO': 'LEMON adult EO',
        'dortmund': 'Dortmund EC_pre_s1',
        'dortmund_EC_pre_s2': 'Dortmund EC_pre_s2',
        'tdbrain': 'TDBRAIN clinical EC',
        'tdbrain_EO': 'TDBRAIN clinical EO',
    }

    print('Loading label masks...')
    masks = _get_label_masks()
    print(f'  {len(masks)} aparc labels')

    all_dfs = []
    for cohort, label in cohorts.items():
        print(f'Processing {cohort}...')
        df = _per_subject_multi_region(f'{cohort}_composite', regions_hemis, masks)
        if df.empty:
            print(f'  no per-subject STCs for {cohort}')
            continue
        df['cohort_pretty'] = label
        all_dfs.append(df)
    big = pd.concat(all_dfs, ignore_index=True)
    print(f'\nTotal subjects × regions loaded: {len(big)} subjects')
    big.to_csv(OUT / 'stage1_per_subject_multi_region.csv', index=False)

    # ============ Density plots: region × cohort ============
    region_names = [f'{r}-{h}' for r, h in regions_hemis if f'{r}-{h}' in big.columns]
    n_regions = len(region_names)
    fig, axes = plt.subplots(4, 3, figsize=(18, 18), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(cohorts)))
    cohort_list = sorted(big['cohort_pretty'].unique())
    for ax, reg in zip(axes.flat, region_names):
        for i, c in enumerate(cohort_list):
            sub = big[big['cohort_pretty'] == c][reg].dropna()
            sub = sub[(sub > 0.1) & (sub < 10)]
            if len(sub) < 10:
                continue
            ax.hist(sub.values, bins=40, alpha=0.35, label=f'{c} (n={len(sub)})',
                    color=colors[i % len(colors)], density=True)
        ax.axvline(1.0, color='grey', lw=0.8, alpha=0.5)
        ax.set_title(reg)
        ax.set_xlabel('Ratio')
        ax.legend(fontsize=6, loc='upper right')
    # Turn off empty axes
    for ax in axes.flat[len(region_names):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / 'stage1_region_distributions.png', dpi=120)
    plt.close()
    print(f'  saved stage1_region_distributions.png')

    # ============ Within-subject region-pair correlations (per cohort) ============
    print('\n=== Within-subject region pair correlations ===')
    rows = []
    key_pairs = [
        ('posteriorcingulate-rh', 'parahippocampal-rh'),
        ('posteriorcingulate-rh', 'caudalanteriorcingulate-lh'),
        ('parahippocampal-rh', 'caudalanteriorcingulate-lh'),
        ('parahippocampal-rh', 'bankssts-lh'),
        ('posteriorcingulate-rh', 'lingual-lh'),
        ('caudalanteriorcingulate-lh', 'parsopercularis-rh'),
    ]
    for c in cohort_list:
        sub = big[big['cohort_pretty'] == c]
        for r1, r2 in key_pairs:
            if r1 not in sub.columns or r2 not in sub.columns:
                continue
            v = sub[[r1, r2]].dropna()
            if len(v) < 10:
                continue
            rho, p = spearmanr(v[r1], v[r2])
            rows.append({'cohort': c, 'r1': r1, 'r2': r2, 'n': len(v),
                         'spearman_rho': rho, 'p': p})
    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(OUT / 'stage1_within_subject_correlations.csv', index=False)
    print(pair_df[['cohort', 'r1', 'r2', 'n', 'spearman_rho', 'p']].to_string(index=False))

    # ============ Bimodality / dispersion test ============
    print('\n=== Bimodality coefficient per region × cohort ===')
    print('(BC > 0.555 suggests bimodality under Gaussian; values 0.55-0.65 weak, >0.70 strong)')
    bimod_rows = []
    for c in cohort_list:
        for reg in region_names:
            sub = big[big['cohort_pretty'] == c][reg].dropna()
            sub = sub[(sub > 0.1) & (sub < 10)]
            if len(sub) < 20:
                continue
            bc, iqr_rel = _dip_test(sub.values)
            bimod_rows.append({'cohort': c, 'region': reg, 'n': len(sub),
                               'bimodality_coef': bc, 'iqr_rel': iqr_rel,
                               'median': np.median(sub), 'std': np.std(sub)})
    bimod_df = pd.DataFrame(bimod_rows)
    bimod_df.to_csv(OUT / 'stage1_bimodality.csv', index=False)
    # Show top-10 most bimodal
    top = bimod_df.sort_values('bimodality_coef', ascending=False).head(15)
    print(top.to_string(index=False))

    # ============ Summary ============
    print(f"\n=== Summary ===")
    print(f"Total subjects: {len(big)}")
    print(f"Cohorts analyzed: {len(cohort_list)}")
    print(f"Regions examined: {len(region_names)}")
    n_bimodal = (bimod_df['bimodality_coef'] > 0.555).sum()
    print(f"Region × cohort combinations with bimodality coef > 0.555: {n_bimodal}/{len(bimod_df)}")
    # Anti-correlated region pairs (negative rho, suggest class-trading)
    anti = pair_df[(pair_df['spearman_rho'] < -0.15) & (pair_df['p'] < 0.05)]
    print(f"Anti-correlated region pairs (ρ < -0.15, p<0.05): {len(anti)}")
    if len(anti) > 0:
        print(anti.to_string(index=False))


if __name__ == '__main__':
    main()
