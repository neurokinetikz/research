#!/usr/bin/env python3
"""
Paper Figure 3 — Ignition events engage spatially distinct cortical networks.

Two-row × three-column topography figure with shared vmin/vmax per band.
Clean layout with generous spacing.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mne

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
CSV_PATH = os.path.join(OUT_DIR, 'topography_q4_vs_q1.csv')

BANDS = ['SR1', 'β16', 'SR3']
BAND_TITLES = {
    'SR1': 'SR1  (7.82 Hz)',
    'β16': 'β16  (14.5–17.5 Hz)',
    'SR3': 'SR3  (19.95 Hz)',
}
BAND_SUBTITLES = {
    'SR1': 'occipital α',
    'β16': 'centro-parietal β-low',
    'SR3': 'central β-high',
}


def main():
    df = pd.read_csv(CSV_PATH)
    piv = df.pivot_table(index='channel', columns=['quartile', 'band'],
                          values='ratio')
    common_chs = list(piv.index)

    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=common_chs, sfreq=250, ch_types='eeg')
    info.set_montage(montage, match_case=False, on_missing='ignore')

    data = {}
    vrange = {}
    for b in BANDS:
        q4 = np.array([piv.loc[ch, ('Q4', b)] for ch in common_chs])
        q1 = np.array([piv.loc[ch, ('Q1', b)] for ch in common_chs])
        data[(4, b)] = q4; data[(1, b)] = q1
        combined = np.concatenate([q4, q1])
        vrange[b] = (float(np.nanmin(combined)), float(np.nanmax(combined)))

    rhos = {}
    for b in BANDS:
        rho, p = spearmanr(data[(4, b)], data[(1, b)])
        rhos[b] = (rho, p)

    # Figure with generous layout
    fig = plt.figure(figsize=(14, 10))
    # Manual gridspec:
    #   [title area]
    #   [col headers]
    #   [row1: 3 topomap cols] + row-label left
    #   [row2: 3 topomap cols] + row-label left
    #   [3 colorbars]
    #   [caption]

    # Parameters - compact layout
    title_y = 0.95
    col_hdr_y = 0.89
    row1_y = 0.56
    row2_y = 0.24
    topomap_w = 0.22
    topomap_h = 0.27
    col_xs = [0.15, 0.41, 0.67]    # left edges of each column
    cbar_y = 0.12
    cbar_h = 0.018

    # Figure-level text
    fig.text(0.5, title_y,
              'Figure 3 — Ignition events engage spatially distinct cortical networks',
              ha='center', va='center', fontsize=14, fontweight='bold')

    for i, b in enumerate(BANDS):
        fig.text(col_xs[i] + topomap_w / 2, col_hdr_y,
                  BAND_TITLES[b],
                  ha='center', va='center', fontsize=12, fontweight='bold')
        fig.text(col_xs[i] + topomap_w / 2, col_hdr_y - 0.030,
                  BAND_SUBTITLES[b],
                  ha='center', va='center', fontsize=10, style='italic',
                  color='#444')

    # Row labels (left margin)
    fig.text(0.055, row1_y + topomap_h / 2,
              'Q4\ncanonical\nignition',
              ha='center', va='center', fontsize=11, fontweight='bold',
              color='#8c1a1a')
    fig.text(0.055, row2_y + topomap_h / 2,
              'Q1\nnoise-like\nevent',
              ha='center', va='center', fontsize=11, fontweight='bold',
              color='#4575b4')

    # Topomaps
    for row_idx, (q_val, q_lab, row_y) in enumerate([
        (4, 'Q4', row1_y), (1, 'Q1', row2_y)
    ]):
        for col_i, b in enumerate(BANDS):
            ax = fig.add_axes([col_xs[col_i], row_y, topomap_w, topomap_h])
            arr = data[(q_val, b)]
            vmin, vmax = vrange[b]
            im, _ = mne.viz.plot_topomap(arr, info, axes=ax, show=False,
                                          cmap='viridis', contours=5,
                                          vlim=(vmin, vmax))
            peak_ch = common_chs[int(np.nanargmax(arr))]
            # Peak annotation beneath topomap
            ax.text(0.5, -0.08, f'{arr.max():.2f}× @ {peak_ch}',
                     transform=ax.transAxes, ha='center', va='top',
                     fontsize=9, color='#333')

    # Colorbars at bottom, one per column. Put ρ label ABOVE the bar so
    # there's no overlap with default tick labels below the bar.
    for col_i, b in enumerate(BANDS):
        cax = fig.add_axes([col_xs[col_i] + 0.015, cbar_y,
                             topomap_w - 0.03, cbar_h])
        vmin, vmax = vrange[b]
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                    norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=8)
        rho, p = rhos[b]
        col_color = '#8c1a1a' if rho < -0.2 else ('#1a9641' if rho > 0.5 else '#666')
        # Title above: "event/baseline (×)   ρ(Q4,Q1) = ±.##, p = ..."
        fig.text(col_xs[col_i] + topomap_w / 2, cbar_y + cbar_h + 0.015,
                  f'event / baseline (×)',
                  ha='center', va='bottom', fontsize=9, color='#333')
        fig.text(col_xs[col_i] + topomap_w / 2, cbar_y + cbar_h + 0.035,
                  f'ρ(Q4, Q1) = {rho:+.2f}    p = {p:.1g}',
                  ha='center', va='bottom', fontsize=9.5, color=col_color,
                  fontweight='bold')

    # Caption
    caption = (
        'N = 89 LEMON subjects with both Q4 and Q1 events. Shared colorscale '
        'within each band (column) for Q4-vs-Q1 amplitude comparison; separate '
        'scales across bands. '
        'SR1 is posterior-occipital with the same topography in Q4 and Q1 '
        '(ρ = +0.88, Q4 2.4× stronger); this posterior-α pattern IS '
        'individually reliable (B45: 55% of subjects match group with ρ > 0.5). '
        'The β-band aggregate topographies appear anti-correlated between '
        'quartiles (ρ = −0.53 for β16, −0.39 for SR3), but these are '
        'COHORT-LEVEL aggregates that do NOT replicate within individual '
        'subjects (B45: only 1-6% of subjects match the group β-band topos; '
        'within-subject Q4×Q1 correlation is null for β16). The β-band patterns '
        'reflect between-subject variability in β generators averaged '
        'asymmetrically across quartiles, not a within-subject network-identity '
        'flip. The robust individual-level finding is the SR1 posterior α.'
    )
    fig.text(0.5, 0.04, caption, ha='center', va='top',
              fontsize=9, style='italic', wrap=True)

    plt.savefig(os.path.join(OUT_DIR, 'paper_figure3_networks.png'),
                 dpi=180, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'paper_figure3_networks.pdf'),
                 bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR}/paper_figure3_networks.png")
    print(f"Saved: {OUT_DIR}/paper_figure3_networks.pdf")

    print("\n=== Figure 3 summary ===")
    for b in BANDS:
        q4 = data[(4, b)]; q1 = data[(1, b)]
        peak_q4 = common_chs[int(np.nanargmax(q4))]
        peak_q1 = common_chs[int(np.nanargmax(q1))]
        rho, p = rhos[b]
        print(f"  {b}: Q4 {q4.max():.2f}× @ {peak_q4}   "
              f"Q1 {q1.max():.2f}× @ {peak_q1}   "
              f"ρ(Q4,Q1)={rho:+.2f}")


if __name__ == '__main__':
    main()
