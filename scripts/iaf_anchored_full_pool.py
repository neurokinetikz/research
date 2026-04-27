#!/usr/bin/env python3
"""
IAF-Anchored Enrichment: Full 6-Dataset Pool with FDR + Bootstrap
==================================================================

Drives iaf_anchored_enrichment across all 6 datasets + 11 HBN releases, loads
demographics, pools for developmental and cognitive analyses, and adds:

    - Benjamini-Hochberg FDR correction within analysis families
    - Subject-level stratified bootstrap CIs on key statistics (1000 iters)
    - Cross-anchor attenuation statistics for the developmental trajectory

Families:
    - Cognitive (LEMON only, N ~200): 90 features x 4 tests = 360 tests
    - Developmental HBN pooled (N ~2880): 90 features x 1 age = 90 tests
    - Aging Dortmund (N ~608): 90 features x 1 age = 90 tests

Key headline statistics bootstrapped:
    1. LPS x beta_low_center_depletion under each anchor (raw + age-partialed)
    2. Mean |rho_age| across all enrichment features (pop vs IAF)
    3. HBN alpha_asymmetry x age (strongest developmental feature in v3 paper)

Usage:
    python scripts/iaf_anchored_full_pool.py [--n-boot 1000] [--seed 42]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))

from iaf_anchored_enrichment import (
    load_dataset,
    load_lemon_behavioural,
    COG_TESTS,
    feature_base_names,
    partial_spearman,
    F0,
    SQRT_PHI,
)

OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'iaf_anchored')
os.makedirs(OUT_DIR, exist_ok=True)

ALL_DATASETS = ['lemon', 'dortmund', 'eegmmidb', 'chbmp', 'tdbrain'] + \
               [f'hbn_R{i}' for i in range(1, 12)]


# ---------------------------------------------------------------------------
# Demographic loaders (copied pattern from voronoi_lifespan_trajectory.py)
# ---------------------------------------------------------------------------

def load_hbn_age():
    """Load HBN ages across all 11 releases."""
    demo = {}
    for release in [f'R{i}' for i in range(1, 12)]:
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
        if not os.path.exists(tsv):
            continue
        df = pd.read_csv(tsv, sep='\t')
        for _, row in df.iterrows():
            demo[row['participant_id']] = row.get('age', np.nan)
    return demo


def load_dortmund_age():
    tsv = '/Volumes/T9/dortmund_data/participants.tsv'
    if not os.path.exists(tsv):
        return {}
    df = pd.read_csv(tsv, sep='\t')
    return {row['participant_id']: row.get('age', np.nan) for _, row in df.iterrows()}


def load_lemon_age():
    """LEMON age bins -> midpoints."""
    meta_path = ('/Volumes/T9/lemon_data/behavioral/'
                 'Behavioural_Data_MPILMBB_LEMON/'
                 'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
    if not os.path.exists(meta_path):
        return {}
    meta = pd.read_csv(meta_path)
    def age_mid(s):
        try:
            lo, hi = s.split('-')
            return (float(lo) + float(hi)) / 2
        except Exception:
            return np.nan
    return {row['ID']: age_mid(row['Age']) for _, row in meta.iterrows()
            if pd.notna(age_mid(row['Age']))}


def load_tdbrain_age():
    """TDBRAIN demographics."""
    candidates = [
        '/Volumes/T9/tdbrain_data/participants.tsv',
        '/Volumes/T9/brainclinics/participants.tsv',
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            return {row['participant_id']: row.get('age', np.nan)
                    for _, row in df.iterrows()}
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_or_load_dataset(dataset, cache_dir=OUT_DIR, force=False):
    """Run iaf_anchored_enrichment for a dataset, or load cached per-subject."""
    cache = os.path.join(cache_dir, f'{dataset}_per_subject.csv')
    if os.path.exists(cache) and not force:
        return pd.read_csv(cache)
    df = load_dataset(dataset)
    if len(df) == 0:
        return df
    df.to_csv(cache, index=False)
    return df


def fdr_bh(pvals, q=0.05):
    """Benjamini-Hochberg FDR. Returns boolean mask of survivors and q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([])
    valid = np.isfinite(p)
    order = np.argsort(np.where(valid, p, np.inf))
    ranked = p[order]
    bh = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the end
    bh_adj = np.minimum.accumulate(bh[::-1])[::-1]
    q_vals = np.empty(n)
    q_vals[order] = bh_adj
    q_vals[~valid] = np.nan
    return (q_vals < q) & valid, q_vals


def spearman_safe(x, y):
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 10:
        return np.nan, np.nan, 0
    rho, p = stats.spearmanr(x[v], y[v])
    return float(rho), float(p), int(v.sum())


def bootstrap_stratified(pool, strata_col, stat_fn, n_boot=1000, seed=42):
    """Subject-level bootstrap, stratified by strata_col. stat_fn(pool_resampled) -> scalar or tuple."""
    rng = np.random.default_rng(seed)
    out = []
    strata = pool[strata_col].values
    uniq = np.unique(strata)
    # Pre-compute indices per stratum
    idx_by_stratum = {s: np.where(strata == s)[0] for s in uniq}
    for _ in range(n_boot):
        chosen = []
        for s in uniq:
            idx = idx_by_stratum[s]
            pick = rng.choice(idx, size=len(idx), replace=True)
            chosen.append(pick)
        resamp = pool.iloc[np.concatenate(chosen)]
        out.append(stat_fn(resamp))
    return np.asarray(out)


def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def build_all_subjects():
    """Run/load every dataset, attach demographics, return dict dataset -> DataFrame."""
    out = {}
    hbn_age = load_hbn_age()
    dort_age = load_dortmund_age()
    lemon_age = load_lemon_age()
    tdb_age = load_tdbrain_age()

    for ds in ALL_DATASETS:
        print(f"\n[{ds}]")
        df = run_or_load_dataset(ds)
        if len(df) == 0:
            print(f"  empty, skipping")
            continue
        df = df.copy()
        df['dataset'] = ds
        if ds.startswith('hbn_'):
            df['age'] = df['subject'].map(hbn_age)
        elif ds == 'dortmund':
            df['age'] = df['subject'].map(dort_age)
        elif ds == 'lemon':
            df['age'] = df['subject'].map(lemon_age)
        elif ds == 'tdbrain':
            df['age'] = df['subject'].map(tdb_age)
        else:
            df['age'] = np.nan
        age_n = df['age'].notna().sum()
        print(f"  N = {len(df)}, with age = {age_n}, "
              f"IAF = {df['iaf'].mean():.2f} +/- {df['iaf'].std():.2f}")
        out[ds] = df
    return out


def cognitive_family_fdr(subjects_lemon, n_boot=1000, seed=42):
    """Pop vs IAF cognition with FDR + bootstrap on the anchor test."""
    subj = load_lemon_behavioural(subjects_lemon.copy())
    feats = feature_base_names(subj.columns, 'pop_')

    rows = []
    for feat in feats:
        pop = subj[f'pop_{feat}'].values
        iaf = subj[f'iaf_{feat}'].values
        age = subj['age'].values if 'age' in subj.columns else np.full(len(subj), np.nan)

        for test in COG_TESTS:
            col = f'cog_{test}'
            if col not in subj.columns:
                continue
            y = subj[col].values

            rho_pop, p_pop, n_pop = spearman_safe(pop, y)
            rho_iaf, p_iaf, n_iaf = spearman_safe(iaf, y)
            rho_pop_ap, p_pop_ap = partial_spearman(pop, y, age)
            rho_iaf_ap, p_iaf_ap = partial_spearman(iaf, y, age)

            rows.append({
                'feature': feat, 'test': test,
                'n_pop': n_pop, 'n_iaf': n_iaf,
                'rho_pop': rho_pop, 'p_pop': p_pop,
                'rho_iaf': rho_iaf, 'p_iaf': p_iaf,
                'rho_pop_agept': rho_pop_ap, 'p_pop_agept': p_pop_ap,
                'rho_iaf_agept': rho_iaf_ap, 'p_iaf_agept': p_iaf_ap,
            })

    rdf = pd.DataFrame(rows)
    # BH FDR within each column family (across 90 features x 4 tests = 360)
    for col_p, col_sig in [('p_pop', 'sig_pop'),
                           ('p_iaf', 'sig_iaf'),
                           ('p_pop_agept', 'sig_pop_agept'),
                           ('p_iaf_agept', 'sig_iaf_agept')]:
        sig, q = fdr_bh(rdf[col_p].values, q=0.05)
        rdf[col_sig] = sig
        rdf[col_p.replace('p_', 'q_')] = q
    out_csv = os.path.join(OUT_DIR, 'full_cognitive_fdr.csv')
    rdf.to_csv(out_csv, index=False)

    # Bootstrap the beta_low_center_depletion x LPS anchor (both anchors)
    anchor = subj.dropna(subset=['pop_beta_low_center_depletion',
                                 'iaf_beta_low_center_depletion',
                                 'cog_LPS', 'age']).copy()
    if len(anchor) < 30:
        anchor_boot = None
    else:
        def stat_fn(pool):
            pop_r, _ = stats.spearmanr(pool['pop_beta_low_center_depletion'], pool['cog_LPS'])
            iaf_r, _ = stats.spearmanr(pool['iaf_beta_low_center_depletion'], pool['cog_LPS'])
            pop_r_ap, _ = partial_spearman(
                pool['pop_beta_low_center_depletion'].values,
                pool['cog_LPS'].values,
                pool['age'].values)
            iaf_r_ap, _ = partial_spearman(
                pool['iaf_beta_low_center_depletion'].values,
                pool['cog_LPS'].values,
                pool['age'].values)
            return (pop_r, iaf_r, pop_r_ap, iaf_r_ap)

        anchor['_stratum'] = 'lemon'
        boot = bootstrap_stratified(anchor, '_stratum', stat_fn,
                                    n_boot=n_boot, seed=seed)
        anchor_boot = {
            'pop_raw': boot[:, 0],
            'iaf_raw': boot[:, 1],
            'pop_agept': boot[:, 2],
            'iaf_agept': boot[:, 3],
        }

    return rdf, anchor_boot


def developmental_family_fdr(pool_df, family_label, n_boot=1000, seed=42,
                             stratify_by='dataset'):
    """Age correlations for all 90 enrichment features under both anchors,
    with FDR + bootstrap on the set-level statistics.

    pool_df should have columns: age, pop_<feat>, iaf_<feat>, <stratify_by>.
    """
    if 'age' not in pool_df.columns or pool_df['age'].notna().sum() < 30:
        return None, None
    pool = pool_df.dropna(subset=['age']).copy()
    feats = feature_base_names(pool.columns, 'pop_')

    rows = []
    for feat in feats:
        pop = pool[f'pop_{feat}'].values
        iaf = pool[f'iaf_{feat}'].values
        age = pool['age'].values
        rho_pop, p_pop, n_pop = spearman_safe(pop, age)
        rho_iaf, p_iaf, n_iaf = spearman_safe(iaf, age)
        att = (1 - abs(rho_iaf) / abs(rho_pop)) if abs(rho_pop) > 1e-6 else np.nan
        rows.append({
            'feature': feat,
            'n': n_pop,
            'rho_age_pop': rho_pop, 'p_age_pop': p_pop,
            'rho_age_iaf': rho_iaf, 'p_age_iaf': p_iaf,
            'attenuation': att,
        })
    rdf = pd.DataFrame(rows)
    sig_pop, q_pop = fdr_bh(rdf['p_age_pop'].values, q=0.05)
    sig_iaf, q_iaf = fdr_bh(rdf['p_age_iaf'].values, q=0.05)
    rdf['sig_pop'] = sig_pop
    rdf['sig_iaf'] = sig_iaf
    rdf['q_pop'] = q_pop
    rdf['q_iaf'] = q_iaf
    rdf.to_csv(os.path.join(OUT_DIR, f'{family_label}_age_fdr.csv'), index=False)

    # Bootstrap: mean |rho_age| and # FDR survivors under each anchor
    def stat_fn(sub):
        r_pop = []
        r_iaf = []
        for feat in feats:
            pop = sub[f'pop_{feat}'].values
            iaf = sub[f'iaf_{feat}'].values
            age = sub['age'].values
            rp, _, _ = spearman_safe(pop, age)
            ri, _, _ = spearman_safe(iaf, age)
            r_pop.append(rp)
            r_iaf.append(ri)
        r_pop = np.array(r_pop)
        r_iaf = np.array(r_iaf)
        return (np.nanmean(np.abs(r_pop)),
                np.nanmean(np.abs(r_iaf)),
                np.nanmean(1 - np.abs(r_iaf) / np.where(np.abs(r_pop) > 1e-6, np.abs(r_pop), np.nan)))

    if stratify_by not in pool.columns:
        pool['_stratum'] = family_label
        stratify_by = '_stratum'
    boot = bootstrap_stratified(pool, stratify_by, stat_fn,
                                n_boot=n_boot, seed=seed)

    return rdf, {
        'mean_abs_rho_pop': boot[:, 0],
        'mean_abs_rho_iaf': boot[:, 1],
        'mean_attenuation': boot[:, 2],
    }


def write_summary(subjects_all, cog_rdf, cog_boot, dev_rdf, dev_boot,
                  adult_rdf, adult_boot, out_md):
    L = []
    L.append("# IAF-Anchored Spectral Differentiation: Full-Pool Analysis\n")
    L.append(f"Population anchor: f0 = {F0:.2f} Hz  ")
    L.append(f"Per-subject anchor: f0_i = IAF_i / sqrt(phi), with sqrt(phi) = {SQRT_PHI:.4f}  ")
    L.append(f"IAF maps exactly to u = 0.5 (alpha attractor) in each subject's lattice.\n")

    # Dataset inventory
    L.append("## Dataset inventory\n")
    L.append("| Dataset | N (IAF) | N (age) | Mean IAF (Hz) |")
    L.append("|---------|--------:|--------:|--------------:|")
    for ds, df in subjects_all.items():
        if len(df) == 0:
            continue
        age_n = df['age'].notna().sum() if 'age' in df.columns else 0
        L.append(f"| {ds} | {len(df)} | {age_n} | {df['iaf'].mean():.2f} |")
    L.append("")

    # ---- 1. Cognitive ----
    L.append("## 1. Cognitive correlations (LEMON)\n")
    if cog_rdf is not None:
        n_pop_fdr = cog_rdf['sig_pop'].sum()
        n_iaf_fdr = cog_rdf['sig_iaf'].sum()
        n_pop_ap_fdr = cog_rdf['sig_pop_agept'].sum()
        n_iaf_ap_fdr = cog_rdf['sig_iaf_agept'].sum()
        L.append(f"| Comparison | Pop-anchored | IAF-anchored |")
        L.append(f"|-----------|-------------:|-------------:|")
        L.append(f"| FDR survivors (raw)          | {n_pop_fdr} | {n_iaf_fdr} |")
        L.append(f"| FDR survivors (age-partial)  | {n_pop_ap_fdr} | {n_iaf_ap_fdr} |")
        L.append("")

        # Anchor test with bootstrap CIs
        anchor = cog_rdf[(cog_rdf.feature == 'beta_low_center_depletion') &
                         (cog_rdf.test == 'LPS')]
        if len(anchor) > 0 and cog_boot is not None:
            r = anchor.iloc[0]
            L.append("### Anchor test: beta_low_center_depletion x LPS (bootstrap 95% CIs, 1000 iter)\n")
            L.append("| Anchor | Raw rho [95% CI] | Age-partialed rho [95% CI] |")
            L.append("|--------|------------------|----------------------------|")
            pop_raw_ci = ci95(cog_boot['pop_raw'])
            iaf_raw_ci = ci95(cog_boot['iaf_raw'])
            pop_ap_ci = ci95(cog_boot['pop_agept'])
            iaf_ap_ci = ci95(cog_boot['iaf_agept'])
            L.append(
                f"| Pop | {r['rho_pop']:+.3f} [{pop_raw_ci[0]:+.3f}, {pop_raw_ci[1]:+.3f}] | "
                f"{r['rho_pop_agept']:+.3f} [{pop_ap_ci[0]:+.3f}, {pop_ap_ci[1]:+.3f}] |"
            )
            L.append(
                f"| IAF | {r['rho_iaf']:+.3f} [{iaf_raw_ci[0]:+.3f}, {iaf_raw_ci[1]:+.3f}] | "
                f"{r['rho_iaf_agept']:+.3f} [{iaf_ap_ci[0]:+.3f}, {iaf_ap_ci[1]:+.3f}] |"
            )
            # Paired bootstrap: does IAF outperform pop?
            diff_raw = cog_boot['iaf_raw'] - cog_boot['pop_raw']
            diff_ap = cog_boot['iaf_agept'] - cog_boot['pop_agept']
            diff_raw_ci = ci95(np.abs(cog_boot['iaf_raw']) - np.abs(cog_boot['pop_raw']))
            diff_ap_ci = ci95(np.abs(cog_boot['iaf_agept']) - np.abs(cog_boot['pop_agept']))
            L.append("")
            L.append(f"Paired |rho| improvement (IAF - Pop):  ")
            L.append(f"- Raw:          {diff_raw_ci[0]:+.3f} to {diff_raw_ci[1]:+.3f}  ")
            L.append(f"- Age-partialed: {diff_ap_ci[0]:+.3f} to {diff_ap_ci[1]:+.3f}  ")
        L.append("")

    # ---- 2. Developmental (HBN) ----
    L.append("## 2. Developmental trajectory (HBN pool, N ~2880)\n")
    if dev_rdf is not None:
        n_pop = dev_rdf['sig_pop'].sum()
        n_iaf = dev_rdf['sig_iaf'].sum()
        mean_rho_pop = dev_rdf.dropna(subset=['rho_age_pop'])['rho_age_pop'].abs().mean()
        mean_rho_iaf = dev_rdf.dropna(subset=['rho_age_iaf'])['rho_age_iaf'].abs().mean()
        L.append(f"| Metric | Pop-anchored | IAF-anchored |")
        L.append(f"|--------|-------------:|-------------:|")
        L.append(f"| FDR survivors (age) | {n_pop}/90 | {n_iaf}/90 |")
        L.append(f"| Mean \\|rho_age\\|   | {mean_rho_pop:.3f} | {mean_rho_iaf:.3f} |")
        if dev_boot is not None:
            pop_ci = ci95(dev_boot['mean_abs_rho_pop'])
            iaf_ci = ci95(dev_boot['mean_abs_rho_iaf'])
            att_ci = ci95(dev_boot['mean_attenuation'])
            L.append(f"| Bootstrap 95% CI (pop) | [{pop_ci[0]:.3f}, {pop_ci[1]:.3f}] | |")
            L.append(f"| Bootstrap 95% CI (IAF) | | [{iaf_ci[0]:.3f}, {iaf_ci[1]:.3f}] |")
            L.append(f"| Mean attenuation       | \\multicolumn{{2}}{{c}}{{{dev_boot['mean_attenuation'].mean():.1%} [{att_ci[0]:.1%}, {att_ci[1]:.1%}]}} |")
        L.append("")

        # Top 10 features most vs least attenuated
        valid = dev_rdf.dropna(subset=['attenuation', 'rho_age_pop']).copy()
        valid = valid[valid['rho_age_pop'].abs() > 0.05]
        valid['abs_pop'] = valid['rho_age_pop'].abs()
        top = valid.nlargest(10, 'abs_pop')
        L.append("### Top 10 age-correlated features (pop-anchored)\n")
        L.append("| Feature | Pop rho | IAF rho | Attenuation |")
        L.append("|---------|--------:|--------:|------------:|")
        for _, r in top.iterrows():
            L.append(f"| {r['feature']} | {r['rho_age_pop']:+.3f} | "
                     f"{r['rho_age_iaf']:+.3f} | {r['attenuation']:+.1%} |")
        L.append("")

        # IAF-robust features: smallest attenuation among strong-effect features
        strong = valid[valid['abs_pop'] > 0.1].copy()
        if len(strong) > 0:
            robust = strong.nsmallest(10, 'attenuation')
            L.append("### Most IAF-robust features (smallest attenuation, |rho_pop| > 0.1)\n")
            L.append("| Feature | Pop rho | IAF rho | Attenuation |")
            L.append("|---------|--------:|--------:|------------:|")
            for _, r in robust.iterrows():
                L.append(f"| {r['feature']} | {r['rho_age_pop']:+.3f} | "
                         f"{r['rho_age_iaf']:+.3f} | {r['attenuation']:+.1%} |")
            L.append("")

    # ---- 3. Aging (Dortmund) ----
    L.append("## 3. Aging trajectory (Dortmund, N ~608)\n")
    if adult_rdf is not None:
        n_pop = adult_rdf['sig_pop'].sum()
        n_iaf = adult_rdf['sig_iaf'].sum()
        L.append(f"| Metric | Pop-anchored | IAF-anchored |")
        L.append(f"|--------|-------------:|-------------:|")
        L.append(f"| FDR survivors (age) | {n_pop}/90 | {n_iaf}/90 |")
        if adult_boot is not None:
            pop_ci = ci95(adult_boot['mean_abs_rho_pop'])
            iaf_ci = ci95(adult_boot['mean_abs_rho_iaf'])
            L.append(f"| Mean \\|rho_age\\|   | pop CI [{pop_ci[0]:.3f}, {pop_ci[1]:.3f}] | "
                     f"IAF CI [{iaf_ci[0]:.3f}, {iaf_ci[1]:.3f}] |")
        L.append("")

    # ---- 4. Interpretation ----
    L.append("## 4. Interpretation\n")
    L.append("- If **cognitive FDR survivor count is roughly preserved** under IAF-anchoring, "
             "spectral differentiation captures cognitive variance that is largely independent "
             "of boundary misalignment.\n")
    L.append("- If **developmental FDR survivor count drops sharply** under IAF-anchoring, "
             "much of the v3 developmental signal was tracking IAF passage through a fixed "
             "lattice. Any features that survive IAF-anchoring capture genuine within-band "
             "reorganisation beyond IAF maturation.\n")
    L.append("- The **attenuation distribution** across features identifies which enrichment "
             "metrics are IAF-driven vs IAF-independent, providing the basis for a reduced "
             "'IAF-adjusted' feature set for future biomarker work.\n")

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # ---- Build all per-subject tables ----
    all_subjects = build_all_subjects()

    # ---- 1. Cognitive: LEMON only ----
    print("\n=== Cognitive family (LEMON) ===")
    cog_rdf, cog_boot = (None, None)
    if 'lemon' in all_subjects:
        cog_rdf, cog_boot = cognitive_family_fdr(
            all_subjects['lemon'], n_boot=args.n_boot, seed=args.seed)
        print(f"  FDR survivors: pop raw = {cog_rdf['sig_pop'].sum()}, "
              f"IAF raw = {cog_rdf['sig_iaf'].sum()}, "
              f"pop age-p = {cog_rdf['sig_pop_agept'].sum()}, "
              f"IAF age-p = {cog_rdf['sig_iaf_agept'].sum()}")

    # ---- 2. Developmental: HBN pool ----
    print("\n=== Developmental (HBN pool) ===")
    hbn_parts = [all_subjects[f'hbn_R{i}'] for i in range(1, 12)
                 if f'hbn_R{i}' in all_subjects]
    dev_rdf, dev_boot = (None, None)
    if hbn_parts:
        hbn_pool = pd.concat(hbn_parts, ignore_index=True)
        hbn_pool = hbn_pool.dropna(subset=['age']).copy()
        print(f"  HBN pool N = {len(hbn_pool)}, age range {hbn_pool['age'].min():.1f}-"
              f"{hbn_pool['age'].max():.1f}")
        dev_rdf, dev_boot = developmental_family_fdr(
            hbn_pool, 'hbn_pool',
            n_boot=args.n_boot, seed=args.seed, stratify_by='dataset')
        if dev_rdf is not None:
            print(f"  FDR survivors (age): pop = {dev_rdf['sig_pop'].sum()}, "
                  f"IAF = {dev_rdf['sig_iaf'].sum()}")

    # ---- 3. Aging: Dortmund ----
    print("\n=== Aging (Dortmund) ===")
    adult_rdf, adult_boot = (None, None)
    if 'dortmund' in all_subjects:
        dort = all_subjects['dortmund'].dropna(subset=['age']).copy()
        if len(dort) > 30:
            adult_rdf, adult_boot = developmental_family_fdr(
                dort, 'dortmund', n_boot=args.n_boot, seed=args.seed,
                stratify_by='dataset')
            if adult_rdf is not None:
                print(f"  FDR survivors (age): pop = {adult_rdf['sig_pop'].sum()}, "
                      f"IAF = {adult_rdf['sig_iaf'].sum()}")

    # ---- Write summary ----
    out_md = os.path.join(OUT_DIR, 'full_pool_summary.md')
    write_summary(all_subjects, cog_rdf, cog_boot,
                  dev_rdf, dev_boot, adult_rdf, adult_boot, out_md)
    print(f"\n=== Done. Summary: {out_md}")


if __name__ == '__main__':
    main()
