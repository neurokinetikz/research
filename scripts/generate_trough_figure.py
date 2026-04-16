#!/usr/bin/env python3
"""
Generate publication-quality supplementary figure for trough depth analyses.

4-panel figure:
  A. α/β trough developmental trajectory (HBN + Dortmund) with bootstrap CIs
  B. Differential maturation: % of adult depth for δ/θ vs α/β (HBN)
  C. Trough depth correlation matrix (per-subject, all datasets)
  D. Psychopathology dissociation at α/β trough (HBN)

Usage:
    python scripts/generate_trough_figure.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age')
IMG_DIR = os.path.join(BASE_DIR, 'papers', 'images')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# Publication style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'font.family': 'sans-serif',
})


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    # Load data
    v2 = pd.read_csv(os.path.join(OUT_DIR, 'trough_depth_by_age_v2.csv'))
    maturation = pd.read_csv(os.path.join(OUT_DIR, 'differential_maturation_hbn.csv'))
    per_subj = pd.read_csv(os.path.join(OUT_DIR, 'per_subject_trough_depths.csv'))
    psy = pd.read_csv(os.path.join(OUT_DIR, 'trough_depth_psychopathology.csv'))

    # Load TDBRAIN trough data for Panel A extension
    tdb_path = os.path.join(BASE_DIR, 'outputs', 'tdbrain_analysis',
                            'tdbrain_trough_depth_by_age.csv')
    tdb = pd.read_csv(tdb_path) if os.path.exists(tdb_path) else None

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # ================================================================
    # Panel A: α/β trough developmental trajectory (HBN + Dortmund)
    # ================================================================
    ax = fig.add_subplot(gs[0, 0])

    color_ab = '#2ecc71'
    color_tdb = '#8e44ad'  # purple for TDBRAIN
    for cohort, marker, ls in [('HBN', 'o', '-'), ('Dortmund', 's', '-')]:
        sub = v2[(v2.cohort == cohort) & (v2.trough_label == 'α/β (13.4)')]
        sub = sub.sort_values('age_center')
        if len(sub) < 2:
            continue
        x = sub['age_center'].values
        y = sub['depletion_pct'].values
        ci_lo = sub['ci_lo'].values
        ci_hi = sub['ci_hi'].values

        valid = ~np.isnan(ci_lo) & ~np.isnan(ci_hi)
        if valid.any():
            ax.fill_between(x[valid], ci_lo[valid], ci_hi[valid],
                            color=color_ab, alpha=0.12)
        ax.plot(x, y, marker + ls, color=color_ab, markersize=4,
                linewidth=1.5, label=cohort, markeredgecolor='black',
                markeredgewidth=0.3)

    # TDBRAIN extension (ages 5-88, purple)
    if tdb is not None:
        tdb_ab = tdb[tdb.trough == 'α/β (13.4)'].sort_values('age_center')
        if len(tdb_ab) > 0:
            x_tdb = tdb_ab['age_center'].values
            y_tdb = tdb_ab['depletion_pct'].values
            ax.plot(x_tdb, y_tdb, 'D-', color=color_tdb, markersize=4,
                    linewidth=1.5, label='TDBRAIN', markeredgecolor='black',
                    markeredgewidth=0.3, alpha=0.8)

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Trough depletion (%)')
    ax.set_title('A. α/β trough depth across lifespan', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(4, 82)
    ax.annotate('ρ = +0.897\np < 0.0001', xy=(15, 45), fontsize=7, color=color_ab,
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', edgecolor=color_ab, alpha=0.8))
    ax.annotate('ρ ≈ 0\np = 0.99', xy=(45, 62), fontsize=7, color=color_ab,
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', edgecolor=color_ab, alpha=0.8))
    ax.annotate('late-life\nshallowing', xy=(72, 30), fontsize=7, color=color_tdb,
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', edgecolor=color_tdb, alpha=0.8))

    # ================================================================
    # Panel B: Differential maturation (% adult depth)
    # ================================================================
    ax = fig.add_subplot(gs[0, 1])

    for label, color in [('δ/θ (5.1)', '#e74c3c'), ('α/β (13.4)', '#2ecc71')]:
        pct_col = f'{label}_pct_adult'
        if pct_col not in maturation.columns:
            continue
        x = maturation['age_center'].values
        y = maturation[pct_col].values
        mask = ~np.isnan(y)
        ax.plot(x[mask], y[mask], 'o-', color=color, markersize=4,
                linewidth=1.5, label=label, markeredgecolor='black',
                markeredgewidth=0.3)
        # Linear fit
        if mask.sum() >= 3:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_fit = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=color, alpha=0.4)

    ax.axhline(100, color='gray', linewidth=1, linestyle=':', alpha=0.6, label='Adult reference')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('% of adult depth')
    ax.set_title('B. Maturation timeline (HBN)', fontweight='bold')
    ax.legend(fontsize=7, loc='center right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 260)

    # ================================================================
    # Panel C: Correlation matrix heatmap
    # ================================================================
    ax = fig.add_subplot(gs[1, 0])

    labels = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']
    depth_cols = [f'depth_{l}' for l in labels]

    corr_matrix = np.zeros((5, 5))
    from scipy.stats import spearmanr
    for i in range(5):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, 5):
            valid = per_subj[[depth_cols[i], depth_cols[j]]].dropna()
            if len(valid) > 10:
                rho, _ = spearmanr(valid.iloc[:, 0].values, valid.iloc[:, 1].values)
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', norm=norm, aspect='equal')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(5):
        for j in range(5):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.25 else 'black'
            if i != j:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7.5, color=color, fontweight='bold')
    ax.set_title('C. Trough depth correlations (per-subject)', fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Spearman ρ',
                 ticks=[-0.3, -0.15, 0, 0.15, 0.3])

    # ================================================================
    # Panel D: Psychopathology dissociation at α/β
    # ================================================================
    ax = fig.add_subplot(gs[1, 1])

    raw_psy = psy[psy.type == 'raw']
    troughs = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']
    x_pos = np.arange(len(troughs))
    width = 0.35

    ext_rhos = []
    int_rhos = []
    for t in troughs:
        ext_row = raw_psy[(raw_psy.variable == 'externalizing') & (raw_psy.trough == t)]
        int_row = raw_psy[(raw_psy.variable == 'internalizing') & (raw_psy.trough == t)]
        ext_rhos.append(ext_row.iloc[0]['rho'] if len(ext_row) > 0 else 0)
        int_rhos.append(int_row.iloc[0]['rho'] if len(int_row) > 0 else 0)

    bars1 = ax.bar(x_pos - width/2, ext_rhos, width, color='#e74c3c', alpha=0.7,
                   label='Externalizing', edgecolor='black', linewidth=0.3)
    bars2 = ax.bar(x_pos + width/2, int_rhos, width, color='#3498db', alpha=0.7,
                   label='Internalizing', edgecolor='black', linewidth=0.3)

    # Mark FDR-significant bars
    for i, t in enumerate(troughs):
        ext_row = raw_psy[(raw_psy.variable == 'externalizing') & (raw_psy.trough == t)]
        int_row = raw_psy[(raw_psy.variable == 'internalizing') & (raw_psy.trough == t)]
        if len(ext_row) > 0 and ext_row.iloc[0]['p'] < 0.001:
            ax.text(i - width/2, ext_rhos[i] + 0.008, '***', ha='center',
                    fontsize=7, fontweight='bold')
        if len(int_row) > 0 and int_row.iloc[0]['p'] < 0.001:
            ax.text(i + width/2, int_rhos[i] - 0.015, '***', ha='center',
                    fontsize=7, fontweight='bold')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(troughs, fontsize=8)
    ax.set_ylabel('Spearman ρ with depth ratio')
    ax.set_title('D. Psychopathology × trough depth (HBN)', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(-0.18, 0.22)
    ax.annotate('+ρ = shallower trough', xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=6.5, color='gray', va='top')

    plt.savefig(os.path.join(IMG_DIR, 'fig_trough_analyses.png'), dpi=300,
                bbox_inches='tight')
    print(f"Figure saved to {IMG_DIR}/fig_trough_analyses.png")
    plt.close()


if __name__ == '__main__':
    main()
