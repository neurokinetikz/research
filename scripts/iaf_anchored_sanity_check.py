#!/usr/bin/env python3
"""
Sanity checks on the IAF-robust feature amplification.

For each target feature (alpha_boundary, gamma_noble_4, gamma_asymmetry,
beta_high_inv_noble_6):

  1. Stratified by IAF tertile: is the amplification concentrated in the
     low or high tertile, or is it broadly distributed?

  2. Trimmed pool: does the amplification survive when we exclude the
     extreme 5% IAF tails?

  3. By-subject IAF vs residual: correlate (iaf_age_resid) with IAF itself
     to see if extreme IAF subjects drive the effect.

Targets the HBN developmental pool (N ~2856) where the rebalancing was found.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'outputs', 'iaf_anchored')
OUT_DIR = CACHE_DIR

TARGETS = [
    'alpha_boundary',          # pop -0.109 -> iaf -0.180 (65% amp)
    'gamma_noble_4',           # pop -0.105 -> iaf -0.172 (64% amp)
    'gamma_asymmetry',         # pop +0.126 -> iaf +0.188 (50% amp)
    'beta_high_inv_noble_6',   # pop -0.105 -> iaf -0.155 (48% amp)
    # Contrast: a feature that attenuates
    'alpha_asymmetry',         # pop +0.331 -> iaf +0.220 (34% att)
]


def load_hbn_pool():
    """Load all HBN per-subject files and merge into a single pool."""
    parts = []
    for i in range(1, 12):
        path = os.path.join(CACHE_DIR, f'hbn_R{i}_per_subject.csv')
        if not os.path.exists(path):
            continue
        parts.append(pd.read_csv(path).assign(dataset=f'hbn_R{i}'))
    pool = pd.concat(parts, ignore_index=True)

    # Ages from HBN participants.tsv
    demo = {}
    for i in range(1, 12):
        tsv = f'/Volumes/T9/hbn_data/cmi_bids_R{i}/participants.tsv'
        if not os.path.exists(tsv):
            continue
        df = pd.read_csv(tsv, sep='\t')
        for _, row in df.iterrows():
            demo[row['participant_id']] = row.get('age', np.nan)
    pool['age'] = pool['subject'].map(demo)
    return pool.dropna(subset=['age']).copy()


def spearman_safe(x, y):
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 20:
        return np.nan, np.nan, 0
    r, p = stats.spearmanr(x[v], y[v])
    return float(r), float(p), int(v.sum())


def tertile_analysis(pool, feature):
    iaf = pool['iaf'].values
    q33, q67 = np.quantile(iaf, [1/3, 2/3])
    tier = np.where(iaf <= q33, 'low',
                    np.where(iaf >= q67, 'high', 'mid'))
    print(f"\n{feature}")
    print(f"  IAF tertile cuts: low <= {q33:.2f}, mid, high >= {q67:.2f} Hz")
    print(f"  {'Tertile':<6s} {'N':>5s} {'IAF_range':<14s} {'pop rho':>9s} "
          f"{'iaf rho':>9s} {'diff':>8s}")
    for t in ['low', 'mid', 'high']:
        mask = (tier == t)
        sub = pool[mask]
        pop_col = f'pop_{feature}'
        iaf_col = f'iaf_{feature}'
        rp, _, _ = spearman_safe(sub[pop_col].values, sub['age'].values)
        ri, _, _ = spearman_safe(sub[iaf_col].values, sub['age'].values)
        iaf_lo = sub['iaf'].min()
        iaf_hi = sub['iaf'].max()
        print(f"  {t:<6s} {len(sub):>5d} {iaf_lo:.2f}-{iaf_hi:.2f} "
              f"{rp:>+9.3f} {ri:>+9.3f} {abs(ri)-abs(rp):>+8.3f}")


def trimmed_analysis(pool, feature, trim_pct=5):
    iaf = pool['iaf'].values
    lo, hi = np.percentile(iaf, [trim_pct, 100 - trim_pct])
    mask = (iaf >= lo) & (iaf <= hi)
    sub = pool[mask]
    print(f"\n{feature} (trimmed {trim_pct}% each tail, N={len(sub)}, "
          f"IAF range {lo:.2f}-{hi:.2f})")
    rp_full, _, _ = spearman_safe(pool[f'pop_{feature}'].values, pool['age'].values)
    ri_full, _, _ = spearman_safe(pool[f'iaf_{feature}'].values, pool['age'].values)
    rp_trim, _, _ = spearman_safe(sub[f'pop_{feature}'].values, sub['age'].values)
    ri_trim, _, _ = spearman_safe(sub[f'iaf_{feature}'].values, sub['age'].values)
    print(f"  Full pool   pop={rp_full:+.3f}  iaf={ri_full:+.3f}  amp={(abs(ri_full)-abs(rp_full)):+.3f}")
    print(f"  Trimmed     pop={rp_trim:+.3f}  iaf={ri_trim:+.3f}  amp={(abs(ri_trim)-abs(rp_trim)):+.3f}")


def iaf_bin_sweep(pool, feature, n_bins=5):
    """Fine-grained IAF-bin sweep: is amplification monotonic in IAF or concentrated?"""
    iaf = pool['iaf'].values
    edges = np.quantile(iaf, np.linspace(0, 1, n_bins + 1))
    print(f"\n{feature} by IAF quintile")
    print(f"  {'Bin':<5s} {'IAF range':<14s} {'N':>5s} {'pop rho':>9s} {'iaf rho':>9s}")
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (iaf >= lo) & (iaf <= hi)
        sub = pool[mask]
        rp, _, _ = spearman_safe(sub[f'pop_{feature}'].values, sub['age'].values)
        ri, _, _ = spearman_safe(sub[f'iaf_{feature}'].values, sub['age'].values)
        print(f"  Q{i+1:<3d} {lo:.2f}-{hi:.2f}  {len(sub):>5d} {rp:>+9.3f} {ri:>+9.3f}")


def main():
    pool = load_hbn_pool()
    print(f"Loaded HBN pool: N = {len(pool)}")
    print(f"  IAF: {pool['iaf'].mean():.2f} +/- {pool['iaf'].std():.2f} "
          f"(range {pool['iaf'].min():.2f}-{pool['iaf'].max():.2f})")
    print(f"  Age: {pool['age'].mean():.1f} +/- {pool['age'].std():.1f} "
          f"(range {pool['age'].min():.1f}-{pool['age'].max():.1f})")

    # -- IAF vs age correlation (is IAF itself age-correlated? Yes by design.)
    r, p, _ = spearman_safe(pool['iaf'].values, pool['age'].values)
    print(f"  IAF x age: rho = {r:+.3f} (p={p:.3g})")

    print("\n" + "="*74)
    print("TERTILE ANALYSIS: amplification by IAF tertile")
    print("="*74)
    for feat in TARGETS:
        tertile_analysis(pool, feat)

    print("\n" + "="*74)
    print("TRIMMED POOL: amplification with extreme 5% IAF tails removed")
    print("="*74)
    for feat in TARGETS:
        trimmed_analysis(pool, feat, trim_pct=5)

    print("\n" + "="*74)
    print("IAF QUINTILE SWEEP: is amplification monotonic or concentrated?")
    print("="*74)
    for feat in TARGETS:
        iaf_bin_sweep(pool, feat)


if __name__ == '__main__':
    main()
