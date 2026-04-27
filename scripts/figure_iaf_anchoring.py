#!/usr/bin/env python3
"""
Figure 11 for Part V: IAF-anchored coordinate system.

Panel A: Cognitive anchor triple-validation
    Three horizontal CI bars for beta_low_center_depletion x LPS in LEMON:
      - Population anchor (pop rho from full_pool_summary.md: -0.248 [-0.384, -0.102])
      - IRASA replication (v3 paper: rho = -0.294, FDR q < 0.02)
      - IAF-anchored (rho = -0.228 [-0.360, -0.073])

Panel B: Developmental signal decomposition
    Stacked bar showing the 77 HBN-pool FDR-significant features under pop-anchor,
    colored by attenuation sign under IAF-anchor (attenuate = IAF-coupled layer;
    preserve or amplify = IAF-independent layer). Parallel bar for the 73
    IAF-anchored FDR survivors to demonstrate preservation with composition shift.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE_DIR, 'outputs', 'iaf_anchored')
OUT_PNG = os.path.join(BASE_DIR, 'papers', 'spectral_differentiation_v3',
                      'images', 'fig11_iaf_anchoring.png')


def band_of(feature):
    """Return band name from a feature like 'alpha_inv_noble_3'."""
    for band in ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']:
        if feature.startswith(band + '_'):
            return band
    return 'other'


def panel_a_triple_validation(ax):
    """Horizontal CI bars for the cognitive anchor under three schemes."""
    schemes = [
        ('Population anchor\n(FOOOF)',  -0.248, (-0.384, -0.102), '#4A90A4'),
        ('IRASA replication',            -0.294, (-0.430, -0.158), '#7B6888'),
        ('IAF-anchored\n(Part V)',       -0.228, (-0.360, -0.073), '#C77B58'),
    ]

    y_positions = np.arange(len(schemes))
    for y, (label, rho, (lo, hi), color) in zip(y_positions, schemes):
        ax.errorbar(rho, y,
                    xerr=[[rho - lo], [hi - rho]],
                    fmt='o', markersize=9, color=color, ecolor=color,
                    elinewidth=2.5, capsize=6, capthick=2)
        ax.text(rho, y + 0.18, f'$\\rho = {rho:+.3f}$',
                ha='center', fontsize=9)

    ax.axvline(0, color='grey', linestyle=':', linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in schemes], fontsize=10)
    ax.set_xlabel(r'Spearman $\rho$ (LEMON, $N = 196$)', fontsize=11)
    ax.set_title('A. Cognitive anchor: triple validation\n'
                 r'$\beta_L$ center depletion $\times$ LPS reasoning', fontsize=11)
    ax.set_xlim(-0.55, 0.1)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)


def panel_b_decomposition(ax):
    """Stacked bars: HBN 77 pop FDR survivors vs 73 IAF FDR survivors,
    composition by attenuation sign."""
    df_path = os.path.join(CACHE, 'hbn_pool_age_fdr.csv')
    df = pd.read_csv(df_path)

    pop_sig = df[df['sig_pop']]
    iaf_sig = df[df['sig_iaf']]

    def count_by_layer(sub):
        # IAF-coupled layer: features that attenuate under IAF (|iaf| < |pop|)
        # IAF-independent layer: features that preserve or amplify (|iaf| >= |pop|)
        attenuate = (sub['rho_age_iaf'].abs() < sub['rho_age_pop'].abs())
        return {
            'alpha_atten':    ((sub['feature'].apply(band_of) == 'alpha') & attenuate).sum(),
            'nonalpha_atten': ((sub['feature'].apply(band_of) != 'alpha') & attenuate).sum(),
            'alpha_ampl':     ((sub['feature'].apply(band_of) == 'alpha') & ~attenuate).sum(),
            'nonalpha_ampl':  ((sub['feature'].apply(band_of) != 'alpha') & ~attenuate).sum(),
        }

    pop_c = count_by_layer(pop_sig)
    iaf_c = count_by_layer(iaf_sig)

    labels = [f'Pop FDR\n({len(pop_sig)} features)', f'IAF FDR\n({len(iaf_sig)} features)']
    x = np.arange(2)
    width = 0.5

    # Layer colours: alpha-mountain/attenuate = warm (IAF-coupled layer),
    # non-alpha/amplify = cool (IAF-independent layer).
    colors = {
        'alpha_atten':    '#C77B58',   # alpha attenuating (IAF-coupled)
        'nonalpha_atten': '#E8A56C',   # non-alpha attenuating
        'alpha_ampl':     '#6B9BB3',   # alpha amplifying (rare)
        'nonalpha_ampl':  '#4A7A90',   # non-alpha amplifying (IAF-independent)
    }

    bottoms = np.zeros(2)
    layer_order = ['alpha_atten', 'nonalpha_atten', 'alpha_ampl', 'nonalpha_ampl']
    layer_labels = {
        'alpha_atten': r'$\alpha$, attenuates (IAF-coupled)',
        'nonalpha_atten': r'non-$\alpha$, attenuates',
        'alpha_ampl': r'$\alpha$, preserves/amplifies',
        'nonalpha_ampl': r'non-$\alpha$, amplifies (IAF-independent)',
    }

    for layer in layer_order:
        heights = np.array([pop_c[layer], iaf_c[layer]])
        if heights.sum() == 0:
            continue
        ax.bar(x, heights, width, bottom=bottoms,
               color=colors[layer], label=layer_labels[layer],
               edgecolor='white', linewidth=1.5)
        for xi, h, b in zip(x, heights, bottoms):
            if h > 0:
                # Larger labels for tall segments, smaller for thin ones
                fs = 16 if h >= 10 else 11
                ax.text(xi, b + h / 2, str(int(h)),
                        ha='center', va='center', fontsize=fs,
                        color='white', fontweight='bold')
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('FDR-significant features', fontsize=11)
    ax.set_title('B. Developmental signal decomposition\n'
                 r'HBN pool ($N = 2{,}856$), BH-FDR $q < 0.05$',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(bottoms) * 1.15)


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))
    panel_a_triple_validation(axA)
    panel_b_decomposition(axB)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")
    plt.close(fig)


if __name__ == '__main__':
    main()
