#!/usr/bin/env python3
"""
Power-matched comparison: HBN at Dortmund's N vs Dortmund.

Tests whether the asymmetry between pediatric development (73/90 IAF-FDR
survivors in HBN) and adult aging (13/90 in Dortmund) reflects a genuine
biological difference in the decomposition, or simply the 5.5x difference
in sample size (N = 2,856 vs N = 516).

Procedure:
    1. Load the cached per-subject HBN pool and Dortmund tables.
    2. For each of 100 random subsamples of N_HBN_sub = 516 subjects from HBN
       (stratified within release), compute the 90 age-feature correlations
       under both pop and IAF anchoring, apply BH-FDR q<0.05, count survivors.
    3. Report mean/median and 95% CI on the HBN-subsampled FDR-survivor count
       under each anchor, compared against Dortmund's 36 (pop) and 13 (IAF).

If HBN's subsampled IAF-FDR count is well above Dortmund's 13 (e.g., 40-60),
the asymmetry is biological. If HBN's subsampled IAF-FDR count is close to
Dortmund's 13, it was mostly statistical power.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'scripts'))
from iaf_anchored_full_pool import (
    load_hbn_age, load_dortmund_age,
    run_or_load_dataset,
    fdr_bh, spearman_safe,
)
from iaf_anchored_enrichment import feature_base_names

OUT_DIR = os.path.join(BASE, 'outputs', 'iaf_anchored')
N_SUBSAMPLE = 100
N_TARGET = 516  # Dortmund's N with valid IAF
SEED = 42


def features_and_age_correlations(pool, feats):
    """Return list of (rho, p) for each feature under pop and IAF anchors."""
    age = pool['age'].values
    out_pop = []
    out_iaf = []
    for feat in feats:
        pop = pool[f'pop_{feat}'].values
        iaf = pool[f'iaf_{feat}'].values
        rp, pp, _ = spearman_safe(pop, age)
        ri, pi, _ = spearman_safe(iaf, age)
        out_pop.append((rp, pp))
        out_iaf.append((ri, pi))
    return out_pop, out_iaf


def count_fdr(pvals, q=0.05):
    sig, _ = fdr_bh(np.array([p for _, p in pvals]), q=q)
    return int(sig.sum())


def main():
    # --- Build HBN pool from cached per-subject CSVs ---
    hbn_age = load_hbn_age()
    hbn_parts = []
    for i in range(1, 12):
        cache = os.path.join(OUT_DIR, f'hbn_R{i}_per_subject.csv')
        if not os.path.exists(cache):
            continue
        df = pd.read_csv(cache)
        df['dataset'] = f'hbn_R{i}'
        hbn_parts.append(df)
    hbn = pd.concat(hbn_parts, ignore_index=True)
    hbn['age'] = hbn['subject'].map(hbn_age)
    hbn = hbn.dropna(subset=['age']).copy()
    print(f"HBN pool: N = {len(hbn)}")

    # --- Load Dortmund ---
    dort_age = load_dortmund_age()
    dort = pd.read_csv(os.path.join(OUT_DIR, 'dortmund_per_subject.csv'))
    dort['age'] = dort['subject'].map(dort_age)
    dort = dort.dropna(subset=['age']).copy()
    print(f"Dortmund: N = {len(dort)}")

    feats = feature_base_names(hbn.columns, 'pop_')
    print(f"Features per anchor: {len(feats)}")

    # --- Reference: full HBN and full Dortmund ---
    pop_full, iaf_full = features_and_age_correlations(hbn, feats)
    hbn_full_pop = count_fdr(pop_full)
    hbn_full_iaf = count_fdr(iaf_full)
    pop_dort, iaf_dort = features_and_age_correlations(dort, feats)
    dort_pop = count_fdr(pop_dort)
    dort_iaf = count_fdr(iaf_dort)
    print(f"\nReference counts:")
    print(f"  HBN full (N = {len(hbn)}):    pop = {hbn_full_pop}/90, IAF = {hbn_full_iaf}/90")
    print(f"  Dortmund (N = {len(dort)}):   pop = {dort_pop}/90,   IAF = {dort_iaf}/90")

    # --- HBN subsamples at Dortmund N, stratified by release ---
    rng = np.random.default_rng(SEED)
    releases = hbn['dataset'].unique()
    # Maintain release composition
    rel_counts = hbn['dataset'].value_counts()
    total = rel_counts.sum()
    # Proportional allocation to N_TARGET
    alloc = {r: max(1, int(round(rel_counts[r] / total * N_TARGET)))
             for r in releases}
    # Adjust so the sum equals N_TARGET
    drift = N_TARGET - sum(alloc.values())
    if drift != 0:
        # Adjust largest release(s) to match
        sorted_rels = sorted(releases, key=lambda r: -rel_counts[r])
        i = 0
        while drift != 0:
            r = sorted_rels[i % len(sorted_rels)]
            if drift > 0:
                alloc[r] += 1
                drift -= 1
            elif alloc[r] > 1:
                alloc[r] -= 1
                drift += 1
            i += 1
    print(f"\nHBN subsample allocation by release (total {sum(alloc.values())}):")
    for r in sorted(releases):
        print(f"  {r}: {alloc[r]} of {rel_counts[r]}")

    idx_by_rel = {r: hbn.index[hbn['dataset'] == r].values for r in releases}

    sub_pop_counts = np.zeros(N_SUBSAMPLE, dtype=int)
    sub_iaf_counts = np.zeros(N_SUBSAMPLE, dtype=int)
    for k in range(N_SUBSAMPLE):
        picks = []
        for r in releases:
            picks.append(rng.choice(idx_by_rel[r], size=alloc[r], replace=False))
        sub = hbn.loc[np.concatenate(picks)]
        pop_k, iaf_k = features_and_age_correlations(sub, feats)
        sub_pop_counts[k] = count_fdr(pop_k)
        sub_iaf_counts[k] = count_fdr(iaf_k)

    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    print(f"\nHBN subsamples ({N_SUBSAMPLE} iterations, N = {N_TARGET} each):")
    print(f"  pop FDR survivors: median = {np.median(sub_pop_counts):.1f}, "
          f"mean = {sub_pop_counts.mean():.1f}, "
          f"95% CI = [{ci(sub_pop_counts)[0]:.1f}, {ci(sub_pop_counts)[1]:.1f}]")
    print(f"  IAF FDR survivors: median = {np.median(sub_iaf_counts):.1f}, "
          f"mean = {sub_iaf_counts.mean():.1f}, "
          f"95% CI = [{ci(sub_iaf_counts)[0]:.1f}, {ci(sub_iaf_counts)[1]:.1f}]")

    # How often does HBN-subsampled IAF count fall below Dortmund's 13?
    pct_below = (sub_iaf_counts <= dort_iaf).mean() * 100
    print(f"\nFraction of HBN subsamples with IAF-FDR <= {dort_iaf}: {pct_below:.1f}%")
    print(f"Minimum HBN-subsample IAF-FDR count: {sub_iaf_counts.min()}")
    print(f"Maximum HBN-subsample IAF-FDR count: {sub_iaf_counts.max()}")

    # Save results
    out = pd.DataFrame({
        'iteration': np.arange(N_SUBSAMPLE),
        'hbn_sub_pop_fdr': sub_pop_counts,
        'hbn_sub_iaf_fdr': sub_iaf_counts,
    })
    out.to_csv(os.path.join(OUT_DIR, 'power_matched_hbn_subsamples.csv'),
               index=False)

    summary = {
        'dortmund_pop': dort_pop,
        'dortmund_iaf': dort_iaf,
        'hbn_full_pop': hbn_full_pop,
        'hbn_full_iaf': hbn_full_iaf,
        'hbn_sub_pop_median': float(np.median(sub_pop_counts)),
        'hbn_sub_iaf_median': float(np.median(sub_iaf_counts)),
        'hbn_sub_pop_ci_lo': ci(sub_pop_counts)[0],
        'hbn_sub_pop_ci_hi': ci(sub_pop_counts)[1],
        'hbn_sub_iaf_ci_lo': ci(sub_iaf_counts)[0],
        'hbn_sub_iaf_ci_hi': ci(sub_iaf_counts)[1],
        'pct_hbn_sub_at_or_below_dort_iaf': pct_below,
    }
    pd.Series(summary).to_csv(os.path.join(OUT_DIR, 'power_matched_summary.csv'))
    print(f"\nSaved: {os.path.join(OUT_DIR, 'power_matched_summary.csv')}")


if __name__ == '__main__':
    main()
