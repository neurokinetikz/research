#!/usr/bin/env python3
"""HBN per-subject age × source-ratio regression (across releases, pooled).

Merges per-release participants.tsv age with per_subject_source_summary.csv
per cohort, then tests age × PCC_rh / PCC_lh / PHG_rh ratios via Spearman/Pearson.

Input:
  /tmp/hbn_participants/cmi_bids_R{1-11}/participants.tsv  (merged from VM)
  outputs/schumann/images/source/hbn_R*_composite/per_subject_source_summary.csv

Output:
  outputs/2026-04-24-crosscohort-battery/hbn_age_regression_proper.csv
  outputs/2026-04-24-crosscohort-battery/hbn_age_regression_proper.png
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
META_DIR = Path('/tmp/hbn_participants')
OUT = ROOT / 'outputs' / '2026-04-24-crosscohort-battery'
OUT.mkdir(exist_ok=True)


def _load_metadata():
    """Concatenate all release participants.tsv files."""
    rows = []
    for rel_dir in sorted(META_DIR.glob('cmi_bids_R*')):
        rel = rel_dir.name.replace('cmi_bids_', '')
        p = rel_dir / 'participants.tsv'
        if not p.exists():
            continue
        m = pd.read_csv(p, sep='\t')
        m = m.rename(columns={'participant_id': 'subject_id'})
        if 'release' not in m.columns:
            m['release'] = rel
        rows.append(m[['subject_id', 'release', 'sex', 'age']])
    return pd.concat(rows, ignore_index=True)


def _get_subject_stc_ratio(cohort_dir, sub_id, region_label_lh, region_label_rh):
    """Load per-subject STC and extract mean ratio over region via aparc labels.

    Returns dict {region: mean_ratio}.
    """
    stc_path = cohort_dir / 'stcs' / f'{sub_id}_Q4_SR1_ratio'
    try:
        stc = mne.read_source_estimate(str(stc_path), 'fsaverage')
    except Exception:
        return None
    # We don't have easy access to label masks at this script level without re-loading aparc.
    # Easier: use the group-level label ranking to get per-subject values isn't direct.
    # Use per-subject summary file which has median/p90/p99/max only.
    return None


def _load_per_subject_ratios(cohort):
    """Load per-subject STC-level stats from per_subject_source_summary.csv."""
    p = SOURCE_DIR / f'{cohort}_composite' / 'per_subject_source_summary.csv'
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df['cohort'] = cohort
    return df


def _per_subject_region_ratio(cohort, region='posteriorcingulate', hemi='rh'):
    """Compute per-subject mean ratio over a Desikan-Killiany label using
    per-subject STC files in stcs/ folder.
    """
    # Load aparc label mask once
    subjects_dir = os.path.join(os.path.dirname(mne.datasets.fetch_fsaverage(verbose=False)))
    labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                         subjects_dir=subjects_dir, verbose=False)
    target_name = f"{region}-{hemi}"
    tgt_label = next((L for L in labels if L.name == target_name), None)
    if tgt_label is None:
        return {}
    # For each subject, load STC, extract mean over label vertices
    stc_dir = SOURCE_DIR / f'{cohort}_composite' / 'stcs'
    if not stc_dir.is_dir():
        return {}
    # Source-space metadata: fsaverage ico-5 has 10242 verts per hem
    results = {}
    for f in sorted(stc_dir.glob('*_Q4_SR1_ratio-lh.stc')):
        sub_id = f.name.replace('_Q4_SR1_ratio-lh.stc', '')
        stc = mne.read_source_estimate(str(f).replace('-lh.stc', ''), 'fsaverage')
        # Extract mean over label
        # Use stc.in_label
        try:
            vals = stc.in_label(tgt_label).data[:, 0]
            results[sub_id] = float(np.mean(vals))
        except Exception:
            continue
    return results


def main():
    meta = _load_metadata()
    print(f"Total HBN participants across releases: {len(meta)}")
    meta['age'] = pd.to_numeric(meta['age'], errors='coerce')
    meta = meta.dropna(subset=['age'])
    print(f"  with numeric age: {len(meta)}")
    print(f"  age: min {meta['age'].min():.1f}, max {meta['age'].max():.1f}, "
          f"mean {meta['age'].mean():.1f}, median {meta['age'].median():.1f}")
    print()

    # Find which HBN cohorts have complete source-space data locally
    available = []
    for c in ['R1', 'R2', 'R3', 'R4', 'R10', 'R11']:
        p = SOURCE_DIR / f'hbn_{c}_composite' / 'stcs'
        if p.is_dir() and len(list(p.glob('*_Q4_SR1_ratio-lh.stc'))) > 0:
            available.append(c)
    print(f"Available HBN cohorts with per-subject STC files: {available}")
    print()

    # Compute per-subject PCC-rh, PCC-lh, PHG-rh ratios
    all_rows = []
    for region, hemi in [('posteriorcingulate', 'rh'),
                          ('posteriorcingulate', 'lh'),
                          ('parahippocampal', 'rh'),
                          ('parahippocampal', 'lh'),
                          ('caudalanteriorcingulate', 'lh')]:
        print(f"Computing {region}-{hemi} per subject...")
        for c in available:
            ratios = _per_subject_region_ratio(f'hbn_{c}', region=region, hemi=hemi)
            for sub, r in ratios.items():
                all_rows.append({'subject_id': sub, 'release': c,
                                 'region': region, 'hemi': hemi,
                                 'region_hemi': f'{region}-{hemi}',
                                 'ratio': r})
    df = pd.DataFrame(all_rows)
    print(f"\nTotal per-subject region ratios: {len(df)}")

    # Merge with age
    m = df.merge(meta[['subject_id', 'age', 'sex']], on='subject_id', how='inner')
    print(f"  with age merged: {len(m)}")
    m.to_csv(OUT / 'hbn_per_subject_age_region.csv', index=False)
    print(f"  age: min {m['age'].min():.1f}, max {m['age'].max():.1f}, mean {m['age'].mean():.1f}")
    print()

    # Per-region age correlations (pooled across releases)
    print("=== Age × region ratio (Spearman, pooled across releases) ===")
    result_rows = []
    for rh_name, rh_df in m.groupby('region_hemi'):
        v = rh_df.dropna(subset=['age', 'ratio'])
        if len(v) < 20:
            continue
        rho, p_s = spearmanr(v['age'], v['ratio'])
        r_p, p_p = pearsonr(v['age'], v['ratio'])
        print(f"  {rh_name:30s}  n={len(v):4d}  ρ={rho:+.3f} p={p_s:.3g}  "
              f"Pearson r={r_p:+.3f} p={p_p:.3g}")
        result_rows.append({'region_hemi': rh_name, 'n': len(v),
                            'spearman_rho': rho, 'spearman_p': p_s,
                            'pearson_r': r_p, 'pearson_p': p_p})
    pd.DataFrame(result_rows).to_csv(OUT / 'hbn_age_regression_proper.csv', index=False)

    # Plot key regions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    key = ['posteriorcingulate-rh', 'posteriorcingulate-lh',
           'parahippocampal-rh', 'parahippocampal-lh',
           'caudalanteriorcingulate-lh']
    for ax, rh_name in zip(axes.flat, key):
        v = m[m['region_hemi'] == rh_name].dropna(subset=['age', 'ratio'])
        if len(v) < 20:
            ax.set_title(f'{rh_name}\n(no data)')
            continue
        ax.scatter(v['age'], v['ratio'], s=10, alpha=0.4, color='steelblue')
        lr = linregress(v['age'], v['ratio'])
        x = np.array([v['age'].min(), v['age'].max()])
        y = lr.slope * x + lr.intercept
        ax.plot(x, y, 'r--', label=f'slope={lr.slope:.4f}, p={lr.pvalue:.3g}')
        rho, p_s = spearmanr(v['age'], v['ratio'])
        ax.set_title(f'{rh_name}\nn={len(v)}, ρ={rho:+.3f}, p={p_s:.3g}')
        ax.axhline(1.0, color='grey', lw=0.8, alpha=0.5)
        ax.set_xlabel('Age (yr)')
        ax.set_ylabel('Ratio')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / 'hbn_age_regression_proper.png', dpi=120)
    plt.close()
    print(f"\nSaved plot to {OUT / 'hbn_age_regression_proper.png'}")


if __name__ == '__main__':
    main()
