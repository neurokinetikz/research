#!/usr/bin/env python3
"""
Generate label-free striking images for eLife submission.

Requirements: landscape, >= 1800x900 px, no labels/text, 1-2 panels,
strong visual impact, not AI-generated.

Generates 3 candidates:
  1. Peak density KDE curve with golden-ratio boundaries
  2. Boundary sweep heatmap
  3. Enrichment landscape heatmap (5 bands x 13 positions)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
IMG_DIR = os.path.join(BASE_DIR, 'papers', 'images')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
MIN_POWER_PCT = 50

C_PHI = '#D4A017'

EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp', 'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3', 'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
}

BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
ENRICHMENT = {
    'theta':     [-52, -58, -62, -58, -58, -46, -31, -3, +35, +65, +86, +98, +126],
    'alpha':     [-35, -24, -15, -2, +21, +39, +43, +29, -7, -40, -57, -69, -74],
    'beta_low':  [-33, -38, -49, -56, -54, -38, -21, +4, +29, +52, +74, +82, +78],
    'beta_high': [+86, +58, +48, +29, +11, -3, -11, -13, -13, -13, -16, -14, -20],
    'gamma':     [-4, -18, -24, -27, -29, -24, -14, +0, +18, +42, +13, +52, +33],
}


def load_all_freqs():
    all_freqs = []
    for name, subdir in EC_DATASETS.items():
        path = os.path.join(PEAK_BASE, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        dfs = [pd.read_csv(f, usecols=cols) for f in files]
        peaks = pd.concat(dfs, ignore_index=True)
        if has_power and MIN_POWER_PCT > 0:
            filtered = []
            for octave in peaks['phi_octave'].unique():
                bp = peaks[peaks.phi_octave == octave]
                thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                filtered.append(bp[bp['power'] >= thresh])
            peaks = pd.concat(filtered, ignore_index=True)
        all_freqs.append(peaks['freq'].values)
        print(f"  {name}: {len(peaks):,} peaks")
    return np.concatenate(all_freqs)


def striking_peak_density(all_freqs):
    """Peak density KDE with golden-ratio boundary bands -- no text."""
    # Target: 2400x1200 at 300 DPI = 8x4 inches
    fig, ax = plt.subplots(figsize=(8, 4))

    f_range = (3, 55)
    log_freqs = np.log(all_freqs)
    log_grid = np.linspace(np.log(f_range[0]), np.log(f_range[1]), 5000)
    hz_grid = np.exp(log_grid)

    kde = stats.gaussian_kde(log_freqs, bw_method=0.02)
    density = kde(log_grid)
    smoothed = gaussian_filter1d(density, sigma=40)

    # Color-fill each band region with a distinct hue
    band_colors = ['#9b59b6', '#3498db', '#e67e22', '#1abc9c', '#e91e63']
    band_alphas = [0.25, 0.25, 0.25, 0.25, 0.25]
    phi_bnds = [F0 / PHI, F0, F0 * PHI, F0 * PHI**2, F0 * PHI**3, F0 * PHI**4]

    for i in range(5):
        lo, hi = phi_bnds[i], phi_bnds[i + 1]
        mask = (hz_grid >= lo) & (hz_grid <= hi)
        ax.fill_between(hz_grid[mask], 0, smoothed[mask],
                        color=band_colors[i], alpha=band_alphas[i])

    # Main density curve
    ax.plot(hz_grid, smoothed, color='#2c3e50', linewidth=2.5)

    # Golden boundary lines -- subtle, no labels
    for pb in phi_bnds:
        if f_range[0] < pb < f_range[1]:
            ax.axvline(pb, color=C_PHI, linewidth=2, alpha=0.7, linestyle='-')

    ax.set_xscale('log')
    ax.set_xlim(f_range)
    ax.set_ylim(bottom=0)

    # Remove ALL text, ticks, spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    path = os.path.join(IMG_DIR, 'striking_peak_density.png')
    fig.savefig(path, dpi=300, facecolor='white', edgecolor='none')
    print(f"  Saved {path}")
    plt.close(fig)


def striking_boundary_sweep():
    """Boundary sweep heatmap -- no text, just the color field with golden star."""
    csv_path = os.path.join(OUT_DIR, 'boundary_sweep', 'sweep_results.csv')
    if not os.path.exists(csv_path):
        print("  SKIP striking_boundary_sweep: sweep_results.csv not found")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 4))

    pivot = df.pivot(index='ratio', columns='f0', values='simplicity_best')
    f0_vals = np.sort(df['f0'].unique())
    ratio_vals = np.sort(df['ratio'].unique())

    im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='YlGnBu',
                   extent=[f0_vals[0], f0_vals[-1], ratio_vals[0], ratio_vals[-1]],
                   interpolation='bilinear')

    # Golden star at phi-lattice position -- no label
    ax.plot(7.60, PHI, '*', color=C_PHI, markersize=20,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    # Remove ALL text, ticks, spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    path = os.path.join(IMG_DIR, 'striking_boundary_sweep.png')
    fig.savefig(path, dpi=300, facecolor='white', edgecolor='none')
    print(f"  Saved {path}")
    plt.close(fig)


def striking_enrichment_landscape():
    """Enrichment landscape heatmap -- no text, pure color grid."""
    fig, ax = plt.subplots(figsize=(8, 4))

    data = np.array([ENRICHMENT[b] for b in BAND_ORDER])
    vmax = 130

    # Use a diverging colormap with smooth interpolation
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                   interpolation='bilinear')

    # Remove ALL text, ticks, spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    path = os.path.join(IMG_DIR, 'striking_enrichment_landscape.png')
    fig.savefig(path, dpi=300, facecolor='white', edgecolor='none')
    print(f"  Saved {path}")
    plt.close(fig)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Loading peak data...")
    all_freqs = load_all_freqs()
    print(f"Total: {len(all_freqs):,} peaks\n")

    print("Generating striking images...")
    striking_peak_density(all_freqs)
    striking_boundary_sweep()
    striking_enrichment_landscape()

    # Verify dimensions
    try:
        from PIL import Image
        for name in ['striking_peak_density', 'striking_boundary_sweep',
                     'striking_enrichment_landscape']:
            path = os.path.join(IMG_DIR, f'{name}.png')
            if os.path.exists(path):
                img = Image.open(path)
                w, h = img.size
                ok = "OK" if w >= 1800 and h >= 900 else "TOO SMALL"
                print(f"  {name}: {w}x{h} [{ok}]")
    except ImportError:
        print("  (install Pillow to verify dimensions)")

    print("\nDone.")


if __name__ == '__main__':
    main()
