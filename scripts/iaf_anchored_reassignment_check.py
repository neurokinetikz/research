#!/usr/bin/env python3
"""
Construct validation for Mechanism 1 (within-subject signal recovery).

For each subject in the HBN pool, count how many peaks change band assignment
between pop-anchored (f0 = 7.60 Hz) and IAF-anchored (f0_i = IAF_i / sqrt(phi))
coordinate systems. Then test:

    Is the rate of band-reassignment correlated with |IAF - mean(IAF)|?

If Mechanism 1 is real (signal recovery via misclassification correction),
peaks should reassign MORE often in subjects whose IAF deviates from the
population mean. If this correlation is weak or absent, Mechanism 1 lacks
mechanistic support.

Also breaks out reassignment rate by target band (which bands gain peaks
most from IAF-anchoring).
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
from phi_frequency_model import PHI, F0
from iaf_anchored_enrichment import assign_bands, compute_iaf, SQRT_PHI, BAND_BY_N

PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v4')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'iaf_anchored')


def process_subject(df):
    """Return (iaf, n_peaks, n_reassigned, per_band_gain)."""
    iaf = compute_iaf(df)
    if not np.isfinite(iaf):
        return None
    f0_i = iaf / SQRT_PHI
    freqs = df['freq'].values

    n_pop = assign_bands(freqs, F0)
    n_iaf = assign_bands(freqs, f0_i)
    reassigned = (n_pop != n_iaf)

    # Per-band gain: among peaks whose IAF-anchor band is B, how many weren't B under pop?
    per_band_gain = {}
    for n_val, band in BAND_BY_N.items():
        in_iaf = (n_iaf == n_val)
        in_pop = (n_pop == n_val)
        per_band_gain[f'gain_{band}'] = int(in_iaf.sum() - in_pop.sum())
        per_band_gain[f'reassigned_to_{band}'] = int((in_iaf & ~in_pop).sum())
        per_band_gain[f'reassigned_from_{band}'] = int((in_pop & ~in_iaf).sum())

    return {
        'iaf': iaf,
        'f0_i': f0_i,
        'n_peaks_total': len(freqs),
        'n_reassigned': int(reassigned.sum()),
        'reassignment_rate': float(reassigned.mean()),
        **per_band_gain,
    }


def main():
    # Work on HBN pool (where the rebalancing was found)
    rows = []
    for release in [f'R{i}' for i in range(1, 12)]:
        ddir = os.path.join(PEAK_BASE, f'hbn_{release}')
        files = sorted(glob.glob(os.path.join(ddir, '*_peaks.csv')))
        for f in files:
            subj = os.path.basename(f).replace('_peaks.csv', '')
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if len(df) == 0 or 'freq' not in df.columns:
                continue
            res = process_subject(df)
            if res is None:
                continue
            res['subject'] = subj
            res['release'] = release
            rows.append(res)

    pool = pd.DataFrame(rows)
    if len(pool) == 0:
        print("No subjects loaded.")
        return
    pool.to_csv(os.path.join(OUT_DIR, 'hbn_reassignment_detail.csv'), index=False)

    mean_iaf = pool['iaf'].mean()
    pool['iaf_dev'] = np.abs(pool['iaf'] - mean_iaf)

    print(f"HBN pool: N = {len(pool)}")
    print(f"  IAF: {mean_iaf:.3f} +/- {pool['iaf'].std():.3f}")
    print(f"  Mean peaks/subject: {pool['n_peaks_total'].mean():.1f}")
    print(f"  Mean reassignment rate: {pool['reassignment_rate'].mean():.3f} "
          f"(i.e., {pool['reassignment_rate'].mean()*100:.1f}% of peaks change band)")

    # Core test: reassignment rate vs |IAF deviation|
    rho, p = stats.spearmanr(pool['iaf_dev'], pool['reassignment_rate'])
    print(f"\nMechanism 1 core test:")
    print(f"  rho(|IAF - mean|, reassignment_rate) = {rho:+.3f}  (p = {p:.2e})")

    # Tertile breakdown
    q33, q67 = np.quantile(pool['iaf'], [1/3, 2/3])
    print(f"\nBy IAF tertile (mean reassignment rate):")
    for label, mask in [('low IAF', pool['iaf'] <= q33),
                        ('mid IAF', (pool['iaf'] > q33) & (pool['iaf'] < q67)),
                        ('high IAF', pool['iaf'] >= q67)]:
        sub = pool[mask]
        print(f"  {label:<10s} N={len(sub):>4d}  "
              f"rate = {sub['reassignment_rate'].mean():.3f}  "
              f"peaks reassigned = {sub['n_reassigned'].mean():.1f}")

    # Per-band gain
    print(f"\nPer-band net gain under IAF-anchoring (mean per subject):")
    for band in BAND_BY_N.values():
        gain = pool[f'gain_{band}'].mean()
        to_b = pool[f'reassigned_to_{band}'].mean()
        from_b = pool[f'reassigned_from_{band}'].mean()
        print(f"  {band:<10s} net {gain:+.2f}  (in: +{to_b:.2f}, out: -{from_b:.2f})")

    # Does reassignment concentrate in IAF-deviation tails?
    # For alpha_boundary: we expect peaks near f0_pop * phi = 12.30 Hz to reassign
    # in high-IAF subjects (peaks above 12.30 that were mis-assigned to beta_low
    # get correctly reassigned to alpha under IAF-anchoring, i.e., reassigned_from_beta_low).
    print("\nHigh-IAF subjects (Q3) should show increased reassigned_from_beta_low "
          "(peaks crossing the 12.30 Hz boundary):")
    r, p = stats.spearmanr(pool['iaf'], pool['reassigned_from_beta_low'])
    print(f"  rho(IAF, reassigned_from_beta_low) = {r:+.3f}  (p = {p:.2e})")

    # Low-IAF subjects should show increased reassigned_from_alpha (peaks below
    # f0_pop = 7.60 Hz that were mis-assigned to theta under pop-anchor get
    # correctly reassigned to alpha when their IAF puts f0_i lower).
    r, p = stats.spearmanr(pool['iaf'], pool['reassigned_from_theta'])
    print(f"  rho(IAF, reassigned_from_theta) = {r:+.3f}  (p = {p:.2e})")

    # Save summary
    summary_path = os.path.join(OUT_DIR, 'reassignment_validation.txt')
    with open(summary_path, 'w') as f:
        f.write(f"HBN pool construct validation: N = {len(pool)}\n")
        f.write(f"Mean reassignment rate: {pool['reassignment_rate'].mean():.3f}\n")
        f.write(f"rho(|IAF - mean|, reassignment_rate) = "
                f"{stats.spearmanr(pool['iaf_dev'], pool['reassignment_rate'])[0]:+.3f}\n")
        for label, mask in [('low IAF', pool['iaf'] <= q33),
                            ('mid IAF', (pool['iaf'] > q33) & (pool['iaf'] < q67)),
                            ('high IAF', pool['iaf'] >= q67)]:
            sub = pool[mask]
            f.write(f"{label}: rate = {sub['reassignment_rate'].mean():.3f}\n")
    print(f"\nSaved: {summary_path}")


if __name__ == '__main__':
    main()
