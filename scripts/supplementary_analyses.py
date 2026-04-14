#!/usr/bin/env python3
"""
Supplementary analyses for spectral differentiation paper.

1. Dortmund-only quadratic vertex estimate
2. Steiger's z-test for externalizing/internalizing dissociation

Usage:
    python scripts/supplementary_analyses.py --analysis vertex
    python scripts/supplementary_analyses.py --analysis steiger
    python scripts/supplementary_analyses.py --analysis all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from run_all_f0_760_analyses import load_subject_enrichments, NEW_PEAK_BASE


def steiger_z(r1, r2, r12, n):
    """Test whether two dependent correlations differ (Steiger, 1980).

    r1 = corr(X, Y1), r2 = corr(X, Y2), r12 = corr(Y1, Y2), n = sample size.
    Returns z-statistic and two-tailed p-value.
    """
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    r_bar = (r1 + r2) / 2
    f = (1 - r12) / (2 * (1 - r_bar**2))
    h = (1 - f * r_bar**2) / (1 - r_bar**2)
    se = np.sqrt(2 * (1 - r12) / ((n - 3) * (1 + r_bar**2 * (1 - r12)
                                               / (2 * (1 - r_bar**2)**2))))
    z_stat = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p


def run_vertex_analysis():
    """Dortmund-only quadratic vertex estimate for inverted-U trajectory."""
    print("=" * 60)
    print("  Dortmund-Only Quadratic Vertex Estimate")
    print("=" * 60)

    peak_dir = os.path.join(NEW_PEAK_BASE, 'dortmund')
    df = load_subject_enrichments(peak_dir, min_peaks=30)

    demo = pd.read_csv('/Volumes/T9/dortmund_data/participants.tsv', sep='\t')
    df = df.merge(demo[['participant_id', 'age']],
                  left_on='subject', right_on='participant_id', how='inner')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    print(f"  Dortmund: N={len(df)}, ages {df.age.min():.0f}-{df.age.max():.0f}")

    features = ['alpha_asymmetry', 'alpha_inv_noble_4', 'beta_low_ushape']
    for feat in features:
        valid = df[['age', feat]].dropna()
        if len(valid) < 50:
            print(f"  {feat}: N={len(valid)}, too few")
            continue

        x = valid['age'].values.astype(float)
        y = valid[feat].values.astype(float)

        # Linear fit
        slope, intercept, r_lin, p_lin, _ = stats.linregress(x, y)
        resid_lin = y - (intercept + slope * x)
        bic_lin = len(x) * np.log(np.mean(resid_lin**2)) + 2 * np.log(len(x))

        # Quadratic fit
        coeffs = np.polyfit(x, y, 2)
        resid_quad = y - np.polyval(coeffs, x)
        bic_quad = len(x) * np.log(np.mean(resid_quad**2)) + 3 * np.log(len(x))

        delta_bic = bic_lin - bic_quad  # positive = quadratic better
        a, b, c = coeffs
        vertex = -b / (2 * a) if a != 0 else float('nan')

        # Bootstrap CI on vertex
        vertices = []
        for _ in range(1000):
            idx = np.random.choice(len(x), len(x), replace=True)
            cb = np.polyfit(x[idx], y[idx], 2)
            if cb[0] != 0:
                v = -cb[1] / (2 * cb[0])
                if 0 < v < 100:
                    vertices.append(v)
        ci_lo, ci_hi = (np.percentile(vertices, [2.5, 97.5])
                        if vertices else (float('nan'), float('nan')))

        shape = 'inverted-U' if a < 0 else 'monotone/U-shaped'
        winner = 'quadratic' if delta_bic > 0 else 'linear'

        print(f"\n  {feat} (N={len(valid)}):")
        print(f"    Linear: r={r_lin:.3f}, p={p_lin:.1e}")
        print(f"    ΔBIC (quad - lin): {delta_bic:.1f} → {winner} wins")
        print(f"    Vertex: {vertex:.1f} years, CI=[{ci_lo:.1f}, {ci_hi:.1f}]")
        print(f"    Shape: {shape}")


def run_steiger_analysis():
    """Steiger's z-test for externalizing/internalizing dissociation."""
    print("=" * 60)
    print("  Steiger's Z-Test: Ext/Int Dissociation")
    print("=" * 60)

    # Load HBN enrichments
    dfs = []
    for release in ['hbn_R1', 'hbn_R2', 'hbn_R3', 'hbn_R4', 'hbn_R6']:
        peak_dir = os.path.join(NEW_PEAK_BASE, release)
        rdf = load_subject_enrichments(peak_dir, min_peaks=30)
        if len(rdf) > 0:
            dfs.append(rdf)
    df = pd.concat(dfs, ignore_index=True)

    # Load demographics
    demos = []
    for r in ['R1', 'R2', 'R3', 'R4', 'R6']:
        d = pd.read_csv(f'/Volumes/T9/hbn_data/cmi_bids_{r}/participants.tsv',
                        sep='\t')
        demos.append(d)
    demo = pd.concat(demos, ignore_index=True)

    df = df.merge(demo[['participant_id', 'externalizing', 'internalizing']],
                  left_on='subject', right_on='participant_id', how='inner')
    df['externalizing'] = pd.to_numeric(df['externalizing'], errors='coerce')
    df['internalizing'] = pd.to_numeric(df['internalizing'], errors='coerce')

    valid_ei = df[['externalizing', 'internalizing']].dropna()
    r_ext_int = stats.spearmanr(valid_ei['externalizing'],
                                valid_ei['internalizing'])[0]
    print(f"  Ext-Int correlation: r={r_ext_int:.3f}, N={len(valid_ei)}")

    features = [
        'gamma_inv_noble_4',
        'alpha_inv_noble_4',
        'beta_low_inv_noble_1',
        'gamma_ramp_depth',
        'gamma_ushape',
    ]

    print(f"\n  {'Feature':<25s}  {'Ext ρ':>8s}  {'Int ρ':>8s}  "
          f"{'Steiger z':>10s}  {'p':>8s}  {'Dir':>8s}  {'N':>5s}")
    print("  " + "-" * 80)

    for feat in features:
        v = df[['externalizing', 'internalizing', feat]].dropna()
        if len(v) < 50:
            continue
        r_ext, p_ext = stats.spearmanr(v[feat], v['externalizing'])
        r_int, p_int = stats.spearmanr(v[feat], v['internalizing'])

        z, p_diff = steiger_z(r_ext, r_int, r_ext_int, len(v))

        direction = 'OPPOSITE' if (r_ext > 0) != (r_int > 0) else 'same'
        print(f"  {feat:<25s}  {r_ext:>+.3f}    {r_int:>+.3f}    "
              f"{z:>+.2f}        {p_diff:>.4f}  {direction:>8s}  {len(v):>5d}")


def main():
    parser = argparse.ArgumentParser(
        description='Supplementary analyses for spectral differentiation paper')
    parser.add_argument('--analysis', required=True,
                        choices=['vertex', 'steiger', 'all'])
    args = parser.parse_args()

    if args.analysis in ('vertex', 'all'):
        run_vertex_analysis()
        print()
    if args.analysis in ('steiger', 'all'):
        run_steiger_analysis()


if __name__ == '__main__':
    main()
