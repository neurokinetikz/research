#!/usr/bin/env python3
"""Full-lifespan age × source-space region ratio regression.

Per-subject PCC/PHG/ACC ratios (Desikan-Killiany aparc labels) × age
for cohorts with per-subject STCs + age metadata.

Cohorts: HBN R1/R4/R11 (pediatric), LEMON EC/EO (adult), Dortmund EC_pre_s1,
         Dortmund EC_pre_s2.

Outputs to outputs/2026-04-24-crosscohort-battery/:
  - full_lifespan_age_region.csv  (per-subject × region × age)
  - full_lifespan_regressions.csv
  - full_lifespan_regressions.png  (6-panel scatter with LOESS)
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress
import mne

ROOT = Path(__file__).parent.parent
SOURCE_DIR = ROOT / 'outputs' / 'schumann' / 'images' / 'source'
HBN_META = Path('/tmp/hbn_participants')
COHORT_META = Path('/tmp/cohort_meta')
OUT = ROOT / 'outputs' / '2026-04-24-crosscohort-battery'


def _load_hbn_age():
    rows = []
    for rel_dir in sorted(HBN_META.glob('cmi_bids_R*')):
        p = rel_dir / 'participants.tsv'
        if not p.exists():
            continue
        m = pd.read_csv(p, sep='\t')
        m = m.rename(columns={'participant_id': 'subject_id'})
        m['cohort'] = 'HBN_' + rel_dir.name.replace('cmi_bids_', '')
        rows.append(m[['subject_id', 'cohort', 'age', 'sex']])
    df = pd.concat(rows, ignore_index=True)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    return df.dropna(subset=['age'])


def _load_lemon_age():
    """LEMON metadata with Age column + subject IDs like sub-010002."""
    p = COHORT_META / 'lemon_meta.csv'
    m = pd.read_csv(p)
    # Column names: ID, Gender, Age, ...
    # LEMON subject IDs in events are sub-010002 etc; metadata IDs may be plain (010002)
    m = m.rename(columns={'ID': 'subject_id_raw', 'Age': 'age_bin'})
    # LEMON age is binned ("22-27", "28-32", ...); midpoint it
    def _age_midpoint(v):
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
    m['age'] = m['age_bin'].apply(_age_midpoint)
    # Subject ID format: raw may be "sub-010002" or "010002"
    m['subject_id'] = m['subject_id_raw'].astype(str).apply(
        lambda x: x if x.startswith('sub-') else f'sub-{x}')
    m['cohort'] = 'LEMON'
    return m[['subject_id', 'cohort', 'age']].dropna(subset=['age'])


def _load_dortmund_age():
    p = COHORT_META / 'dortmund_participants.tsv'
    m = pd.read_csv(p, sep='\t')
    m = m.rename(columns={'participant_id': 'subject_id'})
    m['age'] = pd.to_numeric(m['age'], errors='coerce')
    m['cohort'] = 'Dortmund'
    return m[['subject_id', 'cohort', 'age']].dropna(subset=['age'])


def _per_subject_region_ratio(cohort_composite, region, hemi):
    """Compute per-subject mean ratio for a Desikan-Killiany aparc label."""
    subjects_dir = os.path.dirname(mne.datasets.fetch_fsaverage(verbose=False))
    labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                         subjects_dir=subjects_dir, verbose=False)
    name = f"{region}-{hemi}"
    tgt = next((L for L in labels if L.name == name), None)
    if tgt is None:
        return {}
    stc_dir = SOURCE_DIR / cohort_composite / 'stcs'
    if not stc_dir.is_dir():
        return {}
    results = {}
    for f in sorted(stc_dir.glob('*_Q4_SR1_ratio-lh.stc')):
        sub_id = f.name.replace('_Q4_SR1_ratio-lh.stc', '')
        try:
            stc = mne.read_source_estimate(str(f).replace('-lh.stc', ''), 'fsaverage')
            vals = stc.in_label(tgt).data[:, 0]
            results[sub_id] = float(np.mean(vals))
        except Exception:
            continue
    return results


def main():
    # ==================== Pull ages ====================
    print('Loading age metadata...')
    hbn = _load_hbn_age()
    lemon = _load_lemon_age()
    dort = _load_dortmund_age()
    print(f'  HBN: {len(hbn)} subjects with age')
    print(f'  LEMON: {len(lemon)} subjects with age (binned midpoints)')
    print(f'  Dortmund: {len(dort)} subjects with age')
    ages = pd.concat([hbn, lemon, dort], ignore_index=True)
    print(f'  Total: {len(ages)}')
    print(f'  Age range: {ages["age"].min():.1f} - {ages["age"].max():.1f}')
    print()

    # ==================== Available cohorts with per-subject STCs ====================
    cohort_map = {
        'hbn_R1': 'HBN_R1',
        'hbn_R4': 'HBN_R4',
        'hbn_R11': 'HBN_R11',
        'lemon': 'LEMON',       # EC
        'lemon_EO': 'LEMON',    # EO (same subjects)
        'dortmund': 'Dortmund',  # EC_pre_s1
        'dortmund_EC_pre_s2': 'Dortmund',
    }
    regions = [
        ('posteriorcingulate', 'rh'),
        ('posteriorcingulate', 'lh'),
        ('parahippocampal', 'rh'),
        ('parahippocampal', 'lh'),
        ('caudalanteriorcingulate', 'lh'),
    ]
    all_rows = []
    for cohort_name, meta_cohort in cohort_map.items():
        cohort_comp = f'{cohort_name}_composite'
        stc_dir = SOURCE_DIR / cohort_comp / 'stcs'
        if not stc_dir.is_dir() or not any(stc_dir.iterdir()):
            print(f'  skipping {cohort_name}: no per-subject STCs')
            continue
        print(f'  computing per-subject region ratios for {cohort_name}...')
        for region, hemi in regions:
            ratios = _per_subject_region_ratio(cohort_comp, region, hemi)
            for sub, r in ratios.items():
                all_rows.append({
                    'subject_id': sub, 'cohort_source': cohort_name,
                    'cohort_meta': meta_cohort,
                    'region': region, 'hemi': hemi,
                    'region_hemi': f'{region}-{hemi}',
                    'ratio': r,
                })
    df = pd.DataFrame(all_rows)
    print(f'\nTotal per-subject × region rows: {len(df)}')

    # Merge with age (join on subject_id AND meta_cohort to handle same-subject-diff-cohort)
    m = df.merge(ages[['subject_id', 'cohort', 'age']],
                 left_on=['subject_id', 'cohort_meta'],
                 right_on=['subject_id', 'cohort'],
                 how='inner').drop(columns=['cohort'])
    print(f'  with age merged: {len(m)}')
    # Filter extreme outliers per region (|ratio| > 10 — unphysiological for event/baseline)
    n_before = len(m)
    m = m[(m['ratio'] > 0.1) & (m['ratio'] < 10)].copy()
    print(f'  after outlier filter (0.1 < ratio < 10): {len(m)} (dropped {n_before - len(m)})')
    m.to_csv(OUT / 'full_lifespan_age_region.csv', index=False)

    # ==================== Per-cohort + pooled regressions ====================
    print(f'\n=== Per-cohort age × region ratio (Spearman) ===')
    results = []
    for meta_c in m['cohort_meta'].unique():
        for rh, v in m[m['cohort_meta'] == meta_c].groupby('region_hemi'):
            v = v.dropna(subset=['age', 'ratio'])
            if len(v) < 20:
                continue
            rho, p = spearmanr(v['age'], v['ratio'])
            print(f'  {meta_c:15s} {rh:30s} n={len(v):4d} ρ={rho:+.3f} p={p:.3g}')
            results.append({
                'cohort': meta_c, 'region_hemi': rh, 'n': len(v),
                'spearman_rho': rho, 'spearman_p': p,
                'age_min': v['age'].min(), 'age_max': v['age'].max(),
            })

    print(f'\n=== Pooled FULL LIFESPAN age × region (Spearman) ===')
    for rh, v in m.groupby('region_hemi'):
        v = v.dropna(subset=['age', 'ratio'])
        if len(v) < 20:
            continue
        rho, p = spearmanr(v['age'], v['ratio'])
        r_p, p_p = pearsonr(v['age'], v['ratio'])
        print(f'  {rh:30s} n={len(v):4d} ρ={rho:+.3f} p={p:.3g}  Pearson r={r_p:+.3f} p={p_p:.3g}')
        results.append({
            'cohort': 'ALL_POOLED', 'region_hemi': rh, 'n': len(v),
            'spearman_rho': rho, 'spearman_p': p,
            'pearson_r': r_p, 'pearson_p': p_p,
            'age_min': v['age'].min(), 'age_max': v['age'].max(),
        })
    pd.DataFrame(results).to_csv(OUT / 'full_lifespan_regressions.csv', index=False)

    # ==================== Plot per-region scatter across lifespan ====================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_regions = ['posteriorcingulate-rh', 'posteriorcingulate-lh',
                    'parahippocampal-rh', 'parahippocampal-lh',
                    'caudalanteriorcingulate-lh']
    colors = {'HBN_R1': 'tab:blue', 'HBN_R4': 'tab:cyan', 'HBN_R11': 'tab:purple',
              'LEMON': 'tab:green', 'Dortmund': 'tab:orange'}
    for ax, rh in zip(axes.flat, plot_regions):
        v = m[m['region_hemi'] == rh].dropna(subset=['age', 'ratio'])
        if len(v) < 20:
            ax.set_title(f'{rh}\n(no data)')
            continue
        for c_name in v['cohort_meta'].unique():
            sub = v[v['cohort_meta'] == c_name]
            ax.scatter(sub['age'], sub['ratio'], s=8, alpha=0.5,
                       color=colors.get(c_name, 'grey'), label=f'{c_name} n={len(sub)}')
        # Pooled linear fit
        lr = linregress(v['age'], v['ratio'])
        x = np.array([v['age'].min(), v['age'].max()])
        y = lr.slope * x + lr.intercept
        ax.plot(x, y, 'k--', alpha=0.7,
                 label=f'pooled slope={lr.slope:.4f}\np={lr.pvalue:.3g}')
        rho, p_s = spearmanr(v['age'], v['ratio'])
        ax.set_title(f'{rh}\npooled n={len(v)}, ρ={rho:+.3f}, p={p_s:.3g}')
        ax.axhline(1.0, color='grey', lw=0.8, alpha=0.5)
        ax.set_xlabel('Age (yr)')
        ax.set_ylabel('Ratio')
        ax.legend(fontsize=7, loc='best')
    axes.flat[-1].axis('off')
    plt.tight_layout()
    plt.savefig(OUT / 'full_lifespan_regressions.png', dpi=120)
    plt.close()
    print(f'\nSaved {OUT / "full_lifespan_regressions.png"}')


if __name__ == '__main__':
    main()
