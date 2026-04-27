#!/usr/bin/env python3
"""
SIE Window Enrichment — Pooled Analysis (14 dataset × conditions, N~1200)
=========================================================================

For each subject, the window-enrichment CSV holds per-band × per-position
occupancy and shape metrics computed on three concatenated conditions:
pre (20 s pre-baseline), ignition (±5 s around t0), post (20 s post-baseline).

We want to answer: does ignition produce a reproducible restructuring of
peak distributions across the φ-lattice Voronoi partition?

Test per metric: ignition − mean(pre, post) ≠ 0 (paired t-test at subject
level, one-sample). FDR across all band × position/shape metrics within
each dataset. Pool all datasets (inverse-variance weighted) for a grand
effect size table. Cross-dataset consistency = metric survives FDR in N
of K datasets.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
OUT_DIR = os.path.abspath(OUT_DIR)

# =========================================================================
# LOAD ALL
# =========================================================================

def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean mask of survivors."""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    thresh = alpha * ranks / n
    sorted_p = p[order]
    survive = sorted_p <= thresh
    # find largest k with p_k <= thresh_k
    if survive.any():
        k_max = np.where(survive)[0].max()
        survive_mask = np.zeros(n, dtype=bool)
        survive_mask[order[:k_max + 1]] = True
        return survive_mask
    return np.zeros(n, dtype=bool)


def load_all(include_ses2=False):
    """Load per-dataset window enrichment CSVs under SCOPE env var.

    SCOPE='all' (default): load sie_window_enrichment_<dataset>.csv
    SCOPE='sw': load sie_window_enrichment_sw_<dataset>.csv

    Skips derived analysis outputs and ses-2 files by default.
    """
    scope = os.environ.get('SCOPE', 'all')
    if scope == 'sw':
        pattern = 'sie_window_enrichment_sw_*.csv'
        prefix = 'sie_window_enrichment_sw_'
    else:
        # All-events: match files NOT starting with sw_, q4_
        pattern = 'sie_window_enrichment_*.csv'
        prefix = 'sie_window_enrichment_'
    files = sorted(glob.glob(os.path.join(OUT_DIR, pattern)))
    skip_bases = {'stats', 'pooled_summary', 'pooled_replication'}
    dfs = {}
    for f in files:
        base = os.path.basename(f).replace(prefix, '').replace('.csv', '')
        # For all-events scope, also skip files that are sw_ or q4_ prefixed
        if scope == 'all' and (base.startswith('sw_') or base.startswith('q4_')):
            continue
        if base in skip_bases:
            continue
        if not include_ses2 and base.endswith('_ses2'):
            continue
        df = pd.read_csv(f)
        if 'status' in df.columns:
            df = df[df['status'] == 'ok'].copy()
        df['dataset'] = base
        dfs[base] = df
    return dfs


def identify_metrics(df):
    """Find all (pre, ignition, post) triplets in the schema.

    Returns list of (metric_key, pre_col, ignition_col, post_col).
    """
    pre_cols = [c for c in df.columns if c.startswith('pre_')]
    triplets = []
    for pc in pre_cols:
        key = pc[len('pre_'):]
        ic = f'ignition_{key}'
        poc = f'post_{key}'
        if ic in df.columns and poc in df.columns:
            triplets.append((key, pc, ic, poc))
    return triplets


def compute_effects(df, triplets):
    """Per subject, per metric: ignition_effect = ignition − (pre + post)/2.

    Returns DataFrame of (subject, metric_key, effect).
    """
    rows = []
    for _, s in df.iterrows():
        for key, pc, ic, poc in triplets:
            pre = s[pc]; ign = s[ic]; post = s[poc]
            if any(pd.isna([pre, ign, post])):
                continue
            base = (pre + post) / 2.0
            eff = ign - base
            rows.append({'subject_id': s['subject_id'], 'metric': key, 'effect': eff})
    return pd.DataFrame(rows)


# =========================================================================
# PER-DATASET PER-METRIC T-TEST
# =========================================================================

def per_dataset_stats(eff_df):
    """Group by metric: mean, SE, t, p, Cohen's d."""
    rows = []
    for m, grp in eff_df.groupby('metric'):
        vals = grp['effect'].values
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n < 5:
            continue
        mean = vals.mean()
        sd = vals.std(ddof=1)
        se = sd / np.sqrt(n)
        if sd > 0:
            t = mean / se
            d = mean / sd
        else:
            t = 0.0; d = 0.0
        p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1)) if n > 1 else 1.0
        rows.append({'metric': m, 'n': n, 'mean_effect': mean,
                     'se': se, 't': t, 'p': p, 'd': d})
    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df['fdr_survive'] = fdr_bh(stats_df['p'].values, alpha=0.05)
    return stats_df


# =========================================================================
# POOLED ACROSS DATASETS (inverse-variance weighted)
# =========================================================================

def pool_effects(per_ds_stats):
    """Inverse-variance weighted pooling per metric across datasets.

    per_ds_stats: dict[dataset] -> stats DataFrame from per_dataset_stats.
    """
    # Collect per-metric series across datasets
    all_metrics = set()
    for sd in per_ds_stats.values():
        all_metrics.update(sd['metric'].tolist())

    rows = []
    for m in sorted(all_metrics):
        means, ses, ns, sds = [], [], [], []
        for ds_name, sd in per_ds_stats.items():
            row = sd[sd['metric'] == m]
            if row.empty:
                continue
            means.append(row['mean_effect'].iloc[0])
            ses.append(row['se'].iloc[0])
            ns.append(row['n'].iloc[0])
            sds.append(row['d'].iloc[0])
        if len(means) < 3:
            continue
        means = np.array(means); ses = np.array(ses); ns = np.array(ns); sds = np.array(sds)
        # Inverse-variance weights
        w = 1.0 / (ses ** 2 + 1e-12)
        pooled_mean = (w * means).sum() / w.sum()
        pooled_se = 1.0 / np.sqrt(w.sum())
        pooled_z = pooled_mean / pooled_se
        pooled_p = 2 * (1 - stats.norm.cdf(abs(pooled_z)))
        mean_d = sds.mean()
        median_d = np.median(sds)
        # Cross-dataset consistency: fraction of datasets with d in same direction as grand
        sign_agree = (np.sign(sds) == np.sign(pooled_mean)).mean()
        rows.append({
            'metric': m, 'n_datasets': len(means), 'total_n': int(ns.sum()),
            'pooled_mean': pooled_mean, 'pooled_se': pooled_se,
            'pooled_z': pooled_z, 'pooled_p': pooled_p,
            'mean_d': mean_d, 'median_d': median_d,
            'sign_agreement': sign_agree,
        })
    pooled = pd.DataFrame(rows)
    if not pooled.empty:
        pooled['fdr_survive'] = fdr_bh(pooled['pooled_p'].values, alpha=0.05)
    return pooled


# =========================================================================
# REPLICATION COUNT
# =========================================================================

def replication_table(per_ds_stats):
    """Metric × dataset matrix: 1 if FDR-survive in that dataset."""
    all_metrics = set()
    for sd in per_ds_stats.values():
        all_metrics.update(sd['metric'].tolist())
    rows = []
    for m in sorted(all_metrics):
        row = {'metric': m}
        row['n_replications'] = 0
        for ds, sd in per_ds_stats.items():
            mrow = sd[sd['metric'] == m]
            if mrow.empty:
                row[ds] = np.nan
            else:
                surv = bool(mrow['fdr_survive'].iloc[0])
                # encode direction + significance
                d = mrow['d'].iloc[0]
                row[ds] = f"{d:+.2f}{'*' if surv else ''}"
                if surv:
                    row['n_replications'] += 1
        rows.append(row)
    return pd.DataFrame(rows).sort_values('n_replications', ascending=False)


# =========================================================================
# MAIN
# =========================================================================

def main():
    print('Loading all datasets...')
    dfs = load_all()
    print(f'  {len(dfs)} datasets loaded')
    for name, df in dfs.items():
        print(f'    {name:25s}  N={len(df)}')

    total_subjects = sum(len(d) for d in dfs.values())
    print(f'  Total subject-condition rows: {total_subjects}')

    # Find metrics (use first dataset schema)
    first_df = list(dfs.values())[0]
    triplets = identify_metrics(first_df)
    print(f'  Identified {len(triplets)} metric triplets (pre/ignition/post)')

    # Per-dataset stats
    per_ds_stats = {}
    for name, df in dfs.items():
        print(f'  Computing {name}...')
        eff = compute_effects(df, triplets)
        per_ds_stats[name] = per_dataset_stats(eff)

    # Replication table
    rep = replication_table(per_ds_stats)
    _scope = os.environ.get('SCOPE', 'all')
    _tag = '' if _scope == 'all' else f'_{_scope}'
    rep_path = os.path.join(OUT_DIR, f'sie_window_enrichment_pooled_replication{_tag}.csv')
    rep.to_csv(rep_path, index=False)
    print(f'\nWrote replication table → {rep_path}')

    # Pooled analysis
    pooled = pool_effects(per_ds_stats)
    pooled_path = os.path.join(OUT_DIR, f'sie_window_enrichment_pooled_summary{_tag}.csv')
    pooled.sort_values('pooled_p').to_csv(pooled_path, index=False)
    print(f'Wrote pooled summary   → {pooled_path}')

    # Top survivors
    print('\n' + '=' * 80)
    print('TOP 15 POOLED (by |Cohen d|, FDR-surviving only):')
    print('=' * 80)
    survivors = pooled[pooled['fdr_survive']].copy()
    survivors['abs_d'] = survivors['mean_d'].abs()
    survivors = survivors.sort_values('abs_d', ascending=False).head(15)
    for _, r in survivors.iterrows():
        print(f"  d={r['mean_d']:+.3f}  {r['metric']:45s}  "
              f"agree={r['sign_agreement']:.2f}  p={r['pooled_p']:.2e}  "
              f"N_ds={r['n_datasets']}")

    print('\nReplication table top-10 (by replication count):')
    print('=' * 80)
    top_rep = rep.head(10)
    for _, r in top_rep.iterrows():
        print(f"  {r['metric']:45s}  replicates in {r['n_replications']} / {len(dfs)} datasets")


if __name__ == '__main__':
    main()
