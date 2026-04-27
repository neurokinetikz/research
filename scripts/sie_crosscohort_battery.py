#!/usr/bin/env python3
"""Cross-cohort fast-battery: summary table + STC similarity + laterality + HBN regression.

Reads completed source-space outputs in outputs/schumann/images/source/<cohort>_composite/
and events CSVs in exports_sie/<cohort>_composite/.

Outputs to outputs/2026-04-24-crosscohort-battery/:
  - cohort_summary.csv, cohort_summary.md
  - pattern_similarity_matrix.csv, .png
  - laterality_indices.csv
  - hbn_developmental_regression.csv, .png
  - top12_jaccard_matrix.csv, .png
  - event_feature_distributions.csv, .png
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).parent.parent
SOURCE_DIR = ROOT / 'outputs' / 'schumann' / 'images' / 'source'
EVENTS_DIR = ROOT / 'exports_sie'
OUT = ROOT / 'outputs' / '2026-04-24-crosscohort-battery'
OUT.mkdir(exist_ok=True)

# Discover cohorts with complete outputs
def _list_completed_cohorts():
    cohorts = []
    for d in sorted(SOURCE_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith('_composite'):
            continue
        if not (d / 'Q4_SR1_label_ranking.csv').exists():
            continue
        if not (d / 'group_Q4_SR1_ratio-lh.stc').exists():
            continue
        cohorts.append(d.name)
    return cohorts


def _cohort_label(name):
    """Pretty cohort label for plots."""
    m = name.replace('_composite', '')
    return m


def _stc_concat(cohort):
    """Return concatenated per-vertex array (lh+rh) from group STC."""
    p = SOURCE_DIR / cohort / 'group_Q4_SR1_ratio'
    stc = mne.read_source_estimate(str(p), 'fsaverage')
    return np.concatenate([stc.lh_data[:, 0], stc.rh_data[:, 0]])


# ============ A. Cohort summary ============
def build_cohort_summary(cohorts):
    rows = []
    for c in cohorts:
        rk = pd.read_csv(SOURCE_DIR / c / 'Q4_SR1_label_ranking.csv')
        # Per-subject summary for n
        per_sub = SOURCE_DIR / c / 'per_subject_source_summary.csv'
        n = len(pd.read_csv(per_sub)) if per_sub.exists() else None
        stc_vals = _stc_concat(c)
        rows.append({
            'cohort': _cohort_label(c),
            'n': n,
            'top1_label': rk.iloc[0]['label'],  # already has -lh/-rh suffix
            'top1_ratio': round(rk.iloc[0]['ratio_median'], 3),
            'top2_label': rk.iloc[1]['label'],
            'top2_ratio': round(rk.iloc[1]['ratio_median'], 3),
            'vertex_median': round(np.median(stc_vals), 3),
            'vertex_p90': round(np.percentile(stc_vals, 90), 3),
            'vertex_p99': round(np.percentile(stc_vals, 99), 3),
            'vertex_max': round(np.max(stc_vals), 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'cohort_summary.csv', index=False)
    # Markdown version (manual, no tabulate dep)
    with open(OUT / 'cohort_summary.md', 'w') as f:
        f.write('| ' + ' | '.join(df.columns) + ' |\n')
        f.write('|' + '|'.join('---' for _ in df.columns) + '|\n')
        for _, row in df.iterrows():
            f.write('| ' + ' | '.join(str(v) for v in row) + ' |\n')
    return df


# ============ B. Pattern similarity matrix ============
def build_similarity_matrix(cohorts):
    labels = [_cohort_label(c) for c in cohorts]
    X = np.stack([_stc_concat(c) for c in cohorts])  # (n_cohorts, n_vertices)
    corr = np.corrcoef(X)
    df = pd.DataFrame(corr, index=labels, columns=labels)
    df.to_csv(OUT / 'pattern_similarity_matrix.csv')
    # Plot
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Per-vertex ratio correlation')
    ax.set_title('Cross-cohort Q4 SR1 source-space pattern similarity')
    plt.tight_layout()
    plt.savefig(OUT / 'pattern_similarity_matrix.png', dpi=120)
    plt.close()
    return df


# ============ C. Laterality indices ============
def build_laterality(cohorts):
    """For each cohort, compute (L-R)/(L+R) for each region that has L+R pair."""
    rows = []
    for c in cohorts:
        rk = pd.read_csv(SOURCE_DIR / c / 'Q4_SR1_label_ranking.csv')
        # The 'label' column has '-lh'/'-rh' suffix. Strip it to get base region.
        rk = rk.copy()
        rk['region'] = rk['label'].str.replace(r'-(lh|rh)$', '', regex=True)
        rk_piv = rk.pivot_table(index='region', columns='hemi', values='ratio_median',
                                 aggfunc='first')
        if 'lh' not in rk_piv.columns or 'rh' not in rk_piv.columns:
            continue
        rk_piv = rk_piv.dropna(subset=['lh', 'rh'], how='any')
        for region, r in rk_piv.iterrows():
            li = (r['lh'] - r['rh']) / (r['lh'] + r['rh'])
            rows.append({
                'cohort': _cohort_label(c),
                'region': region,
                'ratio_lh': round(r['lh'], 3),
                'ratio_rh': round(r['rh'], 3),
                'laterality_index': round(li, 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'laterality_indices.csv', index=False)
    # Focus: key regions laterality across cohorts
    key = df[df['region'].isin(['parahippocampal', 'posteriorcingulate',
                                 'caudalanteriorcingulate', 'parsopercularis'])]
    key.to_csv(OUT / 'laterality_key_regions.csv', index=False)
    return df, key


# ============ D. HBN developmental regression ============
def hbn_developmental(cohorts):
    """Fit PCC-rh ratio vs HBN release number."""
    rows = []
    for c in cohorts:
        if not c.startswith('hbn_R'):
            continue
        release = int(c.replace('hbn_R', '').replace('_composite', ''))
        rk = pd.read_csv(SOURCE_DIR / c / 'Q4_SR1_label_ranking.csv')
        pcc_rh = rk[rk['label'] == 'posteriorcingulate-rh']
        pcc_lh = rk[rk['label'] == 'posteriorcingulate-lh']
        phg_rh = rk[rk['label'] == 'parahippocampal-rh']
        per_sub = SOURCE_DIR / c / 'per_subject_source_summary.csv'
        n = len(pd.read_csv(per_sub)) if per_sub.exists() else None
        rows.append({
            'cohort': c.replace('_composite', ''),
            'release': release,
            'n': n,
            'pcc_rh': pcc_rh['ratio_median'].values[0] if len(pcc_rh) else None,
            'pcc_lh': pcc_lh['ratio_median'].values[0] if len(pcc_lh) else None,
            'phg_rh': phg_rh['ratio_median'].values[0] if len(phg_rh) else None,
        })
    df = pd.DataFrame(rows).sort_values('release').reset_index(drop=True)
    df.to_csv(OUT / 'hbn_developmental_regression.csv', index=False)
    # Fit + plot
    if len(df) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, col, title in [(axes[0], 'pcc_rh', 'PCC-rh ratio'),
                                (axes[1], 'pcc_lh', 'PCC-lh ratio')]:
            ax.scatter(df['release'], df[col], s=80, color='steelblue')
            for _, r in df.iterrows():
                ax.annotate(f"R{r['release']}\n(n={r['n']})",
                            (r['release'], r[col]),
                            textcoords='offset points', xytext=(5, 5), fontsize=8)
            # Linear fit
            valid = df.dropna(subset=[col])
            if len(valid) >= 3:
                from scipy.stats import linregress
                lr = linregress(valid['release'], valid[col])
                x = np.array([valid['release'].min(), valid['release'].max()])
                y = lr.slope * x + lr.intercept
                ax.plot(x, y, 'r--', alpha=0.7,
                         label=f"slope={lr.slope:.3f}, p={lr.pvalue:.3g}, R²={lr.rvalue**2:.3f}")
                ax.legend()
            ax.axhline(1.0, color='grey', alpha=0.5, lw=0.8)
            ax.set_xlabel('HBN release number (proxy for age/cohort)')
            ax.set_ylabel('Ratio median')
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(OUT / 'hbn_developmental_regression.png', dpi=120)
        plt.close()
    return df


# ============ E. Top-12 Jaccard matrix ============
def build_jaccard(cohorts):
    """Jaccard overlap of top-12 labels (with hemi) across cohorts."""
    top12_sets = {}
    for c in cohorts:
        rk = pd.read_csv(SOURCE_DIR / c / 'Q4_SR1_label_ranking.csv').head(12)
        top12_sets[_cohort_label(c)] = set(rk['label'].values)  # 'label' has hemi suffix
    labels = list(top12_sets.keys())
    J = np.zeros((len(labels), len(labels)))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            sa, sb = top12_sets[a], top12_sets[b]
            if not sa and not sb:
                J[i, j] = 0
            else:
                J[i, j] = len(sa & sb) / len(sa | sb)
    df = pd.DataFrame(J, index=labels, columns=labels)
    df.to_csv(OUT / 'top12_jaccard_matrix.csv')
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(J, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Jaccard similarity of top-12 labels')
    ax.set_title('Top-12 label overlap (Jaccard index) across cohorts')
    plt.tight_layout()
    plt.savefig(OUT / 'top12_jaccard_matrix.png', dpi=120)
    plt.close()
    return df


# ============ F-I. Event-feature distributions ============
def event_features(cohorts):
    """Per-cohort event-feature summary from sie_events.csv files.

    Features:
    - n_events per subject
    - median HSI, FSI, sr_score per subject
    - sr3/sr1 and sr5/sr3 harmonic ratios
    """
    FEATURES = ['HSI', 'HSI_canonical', 'FSI', 'FSI_canonical',
                'sr_score', 'sr_score_canonical',
                'sr3/sr1', 'sr5/sr3', 'sr6/sr4']
    rows = []
    for c in cohorts:
        ev_dir = EVENTS_DIR / c
        if not ev_dir.is_dir():
            continue
        ev_files = sorted(ev_dir.glob('*_sie_events.csv'))
        if not ev_files:
            continue
        subject_stats = []
        for f in ev_files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if len(df) == 0:
                continue
            sub_id = f.name.replace('_sie_events.csv', '')
            stats = {'subject_id': sub_id, 'cohort': _cohort_label(c), 'n_events': len(df)}
            for feat in FEATURES:
                if feat in df.columns:
                    vals = pd.to_numeric(df[feat], errors='coerce').dropna()
                    if len(vals) > 0:
                        stats[feat + '_median'] = vals.median()
            subject_stats.append(stats)
        if subject_stats:
            rows.extend(subject_stats)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'event_features_per_subject.csv', index=False)
    # Cohort-level summary
    if len(df):
        num_cols = [c for c in df.columns if c not in ('subject_id', 'cohort')]
        agg = df.groupby('cohort')[num_cols].agg(['median', 'mean', 'std']).round(3)
        agg.to_csv(OUT / 'event_features_cohort_summary.csv')
        # Plot distributions for key features
        key_feats = ['n_events', 'HSI_canonical_median', 'sr_score_canonical_median']
        fig, axes = plt.subplots(1, len(key_feats), figsize=(16, 5))
        for i, feat in enumerate(key_feats):
            if feat not in df.columns:
                continue
            cohort_list = df['cohort'].unique().tolist()
            data = [df[df['cohort'] == c][feat].dropna().values for c in cohort_list]
            axes[i].boxplot(data, labels=cohort_list)
            axes[i].set_title(feat)
            axes[i].tick_params(axis='x', rotation=60, labelsize=8)
            axes[i].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / 'event_features_distributions.png', dpi=120)
        plt.close()
    return df


def main():
    cohorts = _list_completed_cohorts()
    print(f"Completed cohorts: {len(cohorts)}")
    for c in cohorts:
        print(f"  {c}")
    print()

    print("A. Cohort summary...")
    summary = build_cohort_summary(cohorts)
    print(summary.to_string(index=False))
    print()

    print("B. Pattern similarity matrix...")
    sim = build_similarity_matrix(cohorts)
    print(f"  saved pattern_similarity_matrix.{{csv,png}}")
    print()

    print("C. Laterality indices...")
    lat, key = build_laterality(cohorts)
    print(f"  full laterality: {len(lat)} rows")
    print(f"  key-region laterality (PHG/PCC/ACC/IFG):")
    print(key[['cohort', 'region', 'ratio_lh', 'ratio_rh', 'laterality_index']].to_string(index=False))
    print()

    print("D. HBN developmental regression...")
    hbn = hbn_developmental(cohorts)
    if len(hbn):
        print(hbn.to_string(index=False))
    print()

    print("E. Top-12 Jaccard matrix...")
    J = build_jaccard(cohorts)
    print(f"  saved top12_jaccard_matrix.{{csv,png}}")
    print()

    print("F-I. Event features...")
    ev = event_features(cohorts)
    print(f"  {len(ev)} subjects with event features")
    print()

    print(f"All outputs in {OUT}")


if __name__ == '__main__':
    main()
