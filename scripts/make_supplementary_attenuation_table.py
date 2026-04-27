#!/usr/bin/env python3
"""
Generate supplementary LaTeX table: per-feature attenuation distribution
in the HBN developmental pool under IAF-anchoring.

Reads outputs/iaf_anchored/hbn_pool_age_fdr.csv and writes a LaTeX longtable
to papers/spectral_differentiation_v3/supplementary_iaf_attenuation.tex.
"""

import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'outputs', 'iaf_anchored', 'hbn_pool_age_fdr.csv')
OUT = os.path.join(BASE, 'papers', 'spectral_differentiation_v3',
                   'supplementary_iaf_attenuation.tex')


BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']


def band_of(feat):
    for b in BAND_ORDER:
        if feat.startswith(b + '_'):
            return b
    return 'other'


def band_rank(feat):
    try:
        return BAND_ORDER.index(band_of(feat))
    except ValueError:
        return 99


def main():
    df = pd.read_csv(SRC)
    # Sort by band then by |rho_age_pop| desc
    df['_band_rank'] = df['feature'].apply(band_rank)
    df['_abs_pop'] = df['rho_age_pop'].abs()
    df = df.sort_values(by=['_band_rank', '_abs_pop'], ascending=[True, False])

    # Compute percent attenuation: (|pop| - |iaf|) / |pop| x 100
    # Positive = attenuation; negative = amplification.
    abs_pop = df['rho_age_pop'].abs()
    abs_iaf = df['rho_age_iaf'].abs()
    atten = np.where(abs_pop > 1e-6,
                     (abs_pop - abs_iaf) / abs_pop * 100,
                     np.nan)

    # Pre-attach attenuation so we can filter per-band
    df = df.assign(attenuation=atten)

    band_labels = {
        'theta': (r'\theta', 'theta', r'tab:iaf_attenuation_theta'),
        'alpha': (r'\alpha', 'alpha', r'tab:iaf_attenuation_alpha'),
        'beta_low': (r'\beta_L', r'\beta_L', r'tab:iaf_attenuation_beta_low'),
        'beta_high': (r'\beta_H', r'\beta_H', r'tab:iaf_attenuation_beta_high'),
        'gamma': (r'\gamma', 'gamma', r'tab:iaf_attenuation_gamma'),
    }

    lines = []
    for band in BAND_ORDER:
        math_name, word_name, label = band_labels[band]
        sub = df[df['feature'].apply(band_of) == band]
        if len(sub) == 0:
            continue

        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(r'\caption{Per-feature attenuation of age correlations '
                     r'under IAF-anchoring: $' + math_name + r'$ band '
                     r'(HBN developmental pool, $N = 2{,}856$). '
                     r'Positive attenuation: feature weakens under IAF-anchor '
                     r'(IAF-coupled layer). Negative attenuation: feature '
                     r'preserves or amplifies (IAF-independent layer). '
                     r'FDR: BH-FDR $q < 0.05$ indicator under each anchor.}')
        lines.append(r'\label{' + label + r'}')
        lines.append(r'\begin{tabular}{l r r r c c}')
        lines.append(r'\toprule')
        lines.append(r'Feature & $\rho_\text{pop}$ & $\rho_\text{IAF}$ '
                     r'& Attenuation & Pop FDR & IAF FDR \\')
        lines.append(r'\midrule')
        for row in sub.itertuples(index=False):
            feat = row.feature.replace('_', r'\_')
            rp = f'{row.rho_age_pop:+.3f}' if np.isfinite(row.rho_age_pop) else '--'
            ri = f'{row.rho_age_iaf:+.3f}' if np.isfinite(row.rho_age_iaf) else '--'
            a = row.attenuation
            att = f'{a:+.1f}\\%' if np.isfinite(a) else '--'
            sig_p = r'\checkmark' if row.sig_pop else ''
            sig_i = r'\checkmark' if row.sig_iaf else ''
            lines.append(f'{feat} & {rp} & {ri} & {att} & {sig_p} & {sig_i} \\\\')
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')
        lines.append('')  # blank line between tables

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote {OUT} ({len(df)} rows)')


if __name__ == '__main__':
    main()
