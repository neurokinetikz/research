#!/usr/bin/env python3
"""HBN severity-dichotomized validation of §12.55 clinical-suppression claim.

Dichotomize HBN p_factor at extremes (top vs bottom decile / quintile / tertile)
and test whether "severe" HBN subjects show suppressed Q4 source ratios relative
to "healthy-like" HBN subjects.

If effect emerges at dichotomized extremes but not in continuous correlations,
supports Hypothesis B (§12.55 real, dimensional measures too insensitive).
If null even dichotomized, supports Hypothesis A (§12.55 TDBRAIN-specific / artifactual).

Uses n=465 HBN (R1 + R4 + R11 pooled).

Outputs:
  outputs/2026-04-24-tdbrain-clinical-stratification/hbn_severity_dichotomized.csv
  outputs/2026-04-24-tdbrain-clinical-stratification/hbn_severity_dichotomized.png
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).parent.parent
IN_CSV = ROOT / 'outputs' / '2026-04-24-tdbrain-clinical-stratification' / 'hbn_clinical_dimensions_per_subject.csv'
OUT = IN_CSV.parent


def main():
    m = pd.read_csv(IN_CSV)
    print(f'Loaded {len(m)} subject × region rows')

    regions = [
        'posteriorcingulate-rh', 'posteriorcingulate-lh',
        'parahippocampal-rh', 'parahippocampal-lh',
        'caudalanteriorcingulate-lh',
        'parsopercularis-rh', 'parsopercularis-lh',
        'precuneus-rh', 'lingual-lh',
    ]
    regions = [r for r in regions if r in m.columns]

    results = []
    for clinical_dim in ['p_factor', 'attention', 'externalizing', 'internalizing']:
        for split_pct in [10, 20, 33]:
            sub = m.dropna(subset=[clinical_dim])
            lo = sub[clinical_dim].quantile(split_pct / 100)
            hi = sub[clinical_dim].quantile(1 - split_pct / 100)
            low_sev = sub[sub[clinical_dim] <= lo]
            hi_sev = sub[sub[clinical_dim] >= hi]
            for reg in regions:
                l_vals = low_sev[reg].dropna().values
                h_vals = hi_sev[reg].dropna().values
                # Filter outliers
                l_vals = l_vals[(l_vals > 0.1) & (l_vals < 10)]
                h_vals = h_vals[(h_vals > 0.1) & (h_vals < 10)]
                if len(l_vals) < 10 or len(h_vals) < 10:
                    continue
                u, p = mannwhitneyu(l_vals, h_vals, alternative='two-sided')
                rank_biserial = 1 - (2 * u) / (len(l_vals) * len(h_vals))
                results.append({
                    'clinical_dim': clinical_dim,
                    'split_pct': split_pct,
                    'region_hemi': reg,
                    'n_low_severity': len(l_vals),
                    'n_high_severity': len(h_vals),
                    'median_low': np.median(l_vals),
                    'median_high': np.median(h_vals),
                    'diff_low_minus_high': np.median(l_vals) - np.median(h_vals),
                    'p': p,
                    'rank_biserial_r': rank_biserial,
                })
    df = pd.DataFrame(results)
    df.to_csv(OUT / 'hbn_severity_dichotomized.csv', index=False)

    # Summary: how many tests are significant?
    print(f'\n=== Summary by split pct ===')
    for split in [10, 20, 33]:
        sub = df[df['split_pct'] == split]
        nsig = (sub['p'] < 0.05).sum()
        print(f"  split top/bottom {split}%: {len(sub)} tests, {nsig} nominally p<0.05 ({nsig/len(sub)*100:.0f}%)")

    print(f'\n=== Top 10 most-significant results ===')
    top = df.sort_values('p').head(15)
    for _, row in top.iterrows():
        sig = '***' if row['p']<0.001 else '**' if row['p']<0.01 else '*' if row['p']<0.05 else ''
        print(f"  {row['clinical_dim']:14s} split{row['split_pct']}% {row['region_hemi']:26s}  "
              f"n_lo={row['n_low_severity']:3d} vs n_hi={row['n_high_severity']:3d}  "
              f"med_lo={row['median_low']:.3f} vs med_hi={row['median_high']:.3f}  "
              f"p={row['p']:.4g} {sig}  r={row['rank_biserial_r']:+.2f}")

    # Bonferroni threshold for each split
    for split in [10, 20, 33]:
        sub = df[df['split_pct'] == split]
        bon_thresh = 0.05 / len(sub)
        bon_sig = (sub['p'] < bon_thresh).sum()
        print(f"\nBonferroni split{split}% (α={bon_thresh:.4g}): {bon_sig}/{len(sub)} survive")

    # Plot: compare medians
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, reg in zip(axes.flat, ['posteriorcingulate-rh', 'posteriorcingulate-lh',
                                     'parahippocampal-rh', 'caudalanteriorcingulate-lh',
                                     'parsopercularis-rh', 'precuneus-rh']):
        d = df[(df['region_hemi'] == reg) & (df['clinical_dim'] == 'p_factor')]
        if len(d) == 0:
            continue
        x = d['split_pct'].values
        ax.plot(x, d['median_low'], 'go-', label='Low severity')
        ax.plot(x, d['median_high'], 'rs-', label='High severity')
        ax.axhline(1.0, color='grey', lw=0.8, alpha=0.5)
        ax.set_xlabel('Top/bottom split %')
        ax.set_ylabel('Median ratio')
        ax.set_title(f'{reg}\np_factor dichotomized')
        ax.legend(fontsize=8)
        for i, row in d.iterrows():
            sig = '*' if row['p']<0.05 else ''
            ax.annotate(f"p={row['p']:.3g}{sig}",
                        (row['split_pct'], (row['median_low']+row['median_high'])/2),
                        fontsize=7, ha='center')
    plt.suptitle('HBN severity dichotomized (low vs high p_factor): median source ratios')
    plt.tight_layout()
    plt.savefig(OUT / 'hbn_severity_dichotomized.png', dpi=120)
    plt.close()
    print(f'\nSaved {OUT / "hbn_severity_dichotomized.png"}')


if __name__ == '__main__':
    main()
