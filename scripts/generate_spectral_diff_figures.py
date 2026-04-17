#!/usr/bin/env python3
"""
Generate Publication Figures for Spectral Differentiation Paper
================================================================

Produces 8 figures for the paper:
  Fig 1: Peak density distribution with model comparison
  Fig 2: Aperiodic null test
  Fig 3: Boundary sweep heatmap with named systems
  Fig 4: Enrichment landscape (5 bands × 13 positions)
  Fig 5: Cross-boundary architecture (cliff, void, bridge, weak)
  Fig 6: Cognitive correlations (top FDR survivors)
  Fig 7: Inverted-U lifespan trajectory
  Fig 8: Reliability (5-year ICC + group stability)

Usage:
    python scripts/generate_spectral_diff_figures.py

Outputs to: papers/images/fig{1-8}_*.pdf
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

# =========================================================================
# STYLE
# =========================================================================

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.5,
})

# Color palette
C_PHI = '#D4A017'       # gold for phi
C_POS = '#2ecc71'       # green for enriched
C_NEG = '#e74c3c'       # red for depleted
C_NULL = '#95a5a6'      # gray for null/neutral
C_ALPHA = '#3498db'     # blue for alpha band
C_BETA_L = '#e67e22'    # orange for beta-low
C_THETA = '#9b59b6'     # purple for theta
C_BETA_H = '#1abc9c'    # teal for beta-high
C_GAMMA = '#e91e63'     # pink for gamma
BAND_COLORS = {
    'theta': C_THETA, 'alpha': C_ALPHA, 'beta_low': C_BETA_L,
    'beta_high': C_BETA_H, 'gamma': C_GAMMA,
}
BAND_LABELS = {
    'theta': 'Theta', 'alpha': 'Alpha', 'beta_low': 'Beta-low',
    'beta_high': 'Beta-high', 'gamma': 'Gamma',
}

PHI_INV = 1.0 / PHI
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v4')
IMG_DIR = os.path.join(BASE_DIR, 'papers', 'images')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
ANALYSIS_DIR = os.path.join(OUT_DIR, 'f0_760_reanalysis')
MIN_POWER_PCT = 50

# 6-dataset convention: HBN merged across 11 releases, TDBRAIN included
EC_DATASETS = {
    'eegmmidb': 'eegmmidb',
    'lemon': 'lemon',
    'dortmund': 'dortmund',
    'chbmp': 'chbmp',
    'hbn': [f'hbn_R{i}' for i in range(1, 12)],
    'tdbrain': 'tdbrain',
}

BAND_HZ = {
    'theta': (F0 / PHI, F0),
    'alpha': (F0, F0 * PHI),
    'beta_low': (F0 * PHI, F0 * PHI ** 2),
    'beta_high': (F0 * PHI ** 2, F0 * PHI ** 3),
    'gamma': (F0 * PHI ** 3, F0 * PHI ** 4),
}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

POS_LIST = [
    ('boundary',    0.000),
    ('noble_6',     round(PHI_INV ** 6, 6)),
    ('noble_5',     round(PHI_INV ** 5, 6)),
    ('noble_4',     round(PHI_INV ** 4, 6)),
    ('noble_3',     round(PHI_INV ** 3, 6)),
    ('inv_noble_1', round(PHI_INV ** 2, 6)),
    ('attractor',   0.5),
    ('noble_1',     round(PHI_INV, 6)),
    ('inv_noble_3', round(1 - PHI_INV ** 3, 6)),
    ('inv_noble_4', round(1 - PHI_INV ** 4, 6)),
    ('inv_noble_5', round(1 - PHI_INV ** 5, 6)),
    ('inv_noble_6', round(1 - PHI_INV ** 6, 6)),
    ('boundary_hi', 1.000),
]
POS_NAMES = [p[0] for p in POS_LIST]
POS_SHORT = ['bnd', 'n6', 'n5', 'n4', 'n3', 'in1', 'att', 'n1',
             'in3', 'in4', 'in5', 'in6', 'bhi']

# Enrichment values — loaded from analysis outputs, with static fallback
_ENRICHMENT_FALLBACK = {
    # 6-dataset means (EEGMMIDB, LEMON, Dortmund, CHBMP, HBN merged, TDBRAIN)
    'theta':     [-64, -59, -57, -56, -50, -39, -22, +2, +36, +58, +74, +80, +94],
    'alpha':     [-6, -2, +3, +4, +13, +14, +16, +15, +1, -18, -31, -43, -52],
    'beta_low':  [-34, -44, -52, -56, -55, -41, -17, +10, +34, +54, +66, +69, +72],
    'beta_high': [+44, +37, +32, +18, +7, -5, -9, -8, -6, -5, -6, -9, -12],
    'gamma':     [-12, -16, -16, -20, -18, -9, -1, +2, +12, +50, -19, +14, -8],
}


def _load_enrichment():
    """Load enrichment from 6-dataset merged CSV, fall back to static values."""
    csv_path = os.path.join(ANALYSIS_DIR, 'enrichment_6dataset_merged.csv')
    if not os.path.exists(csv_path):
        # Try legacy CSV
        csv_path = os.path.join(ANALYSIS_DIR, 'enrichment_comparison_full.csv')
    if not os.path.exists(csv_path):
        print(f"  WARNING: no enrichment CSV found, using fallback enrichment values")
        return _ENRICHMENT_FALLBACK
    df = pd.read_csv(csv_path)
    # enrichment_6dataset_merged.csv uses 'enrichment_pct'; legacy CSV uses 'new_v2'
    value_col = 'enrichment_pct' if 'enrichment_pct' in df.columns else 'new_v2'
    result = {}
    for band in BAND_ORDER:
        bdf = df[df['band'] == band]
        means = bdf.groupby('position')[value_col].mean()
        # Preserve position order
        result[band] = [round(means.get(p, 0)) for p in POS_NAMES]
    return result


ENRICHMENT = _load_enrichment()


def load_all_freqs():
    """Load and power-filter peaks, return pooled frequency array."""
    all_freqs = []
    for name, subdir in EC_DATASETS.items():
        # Handle merged datasets (list of subdirs)
        subdirs = subdir if isinstance(subdir, list) else [subdir]
        files = []
        for sd in subdirs:
            path = os.path.join(PEAK_BASE, sd)
            files.extend(sorted(glob.glob(os.path.join(path, '*_peaks.csv'))))
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


def savefig(fig, name):
    """Save figure as PDF to papers/images/."""
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, f'{name}.png')
    fig.savefig(path, format='png')
    print(f"  Saved {path}")
    plt.close(fig)


# =========================================================================
# FIGURE 1: Peak density + model comparison
# =========================================================================

def fig1_peak_density(all_freqs):
    """Peak density in log-Hz space with phi boundaries and model comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), height_ratios=[2, 1])

    f_range = (3, 55)
    log_freqs = np.log(all_freqs)
    log_grid = np.linspace(np.log(f_range[0]), np.log(f_range[1]), 3000)
    hz_grid = np.exp(log_grid)

    kde = stats.gaussian_kde(log_freqs, bw_method=0.02)
    density = kde(log_grid)
    smoothed = gaussian_filter1d(density, sigma=30)

    # Panel A: density with phi boundaries
    ax = axes[0]
    ax.fill_between(hz_grid, 0, density, alpha=0.15, color='steelblue')
    ax.plot(hz_grid, smoothed, color='steelblue', linewidth=2)

    phi_bnds = [F0 / PHI, F0, F0 * PHI, F0 * PHI ** 2, F0 * PHI ** 3, F0 * PHI ** 4]
    for pb in phi_bnds:
        if f_range[0] < pb < f_range[1]:
            ax.axvline(pb, color=C_PHI, linewidth=1.5, alpha=0.8, linestyle='--')

    # Find and mark troughs
    trough_idx, _ = find_peaks(-smoothed, prominence=0.001, distance=100)
    trough_hz = hz_grid[trough_idx]
    trough_hz = trough_hz[(trough_hz > 4) & (trough_hz < 50)]
    for th in trough_hz:
        idx = np.argmin(np.abs(hz_grid - th))
        ax.plot(th, smoothed[idx], 'v', color=C_NEG, markersize=8, zorder=5)

    ax.set_xscale('log')
    ax.set_xlim(f_range)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak density')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([4, 5, 7, 10, 13, 20, 30, 40, 50])
    ax.legend([
        Line2D([0], [0], color=C_PHI, linestyle='--', linewidth=1.5),
        Line2D([0], [0], marker='v', color=C_NEG, linestyle='None', markersize=8),
    ], ['$\\varphi$-lattice boundaries', 'Density troughs'], loc='upper right')
    n_peaks = len(all_freqs)
    n_ds = len(EC_DATASETS)
    ax.set_title(f'A. Peak frequency distribution ({n_peaks/1e6:.1f}M peaks, {n_ds} datasets)', fontweight='bold')

    # Panel B: model comparison bar chart
    ax = axes[1]
    models = ['$\\varphi$', '$\\sqrt{2}$', '$e-1$', '$\\sqrt{3}$', '$2^{1/3}$',
              'Octave', '$e$', 'Linear']
    bic_vals = [-24.20, -23.66, -23.06, -22.25, -20.83, -15.58, -12.88, -8.95]
    colors = [C_PHI if i == 0 else C_NULL for i in range(len(models))]
    colors[-1] = C_NEG  # linear in red

    bars = ax.barh(range(len(models)), bic_vals, color=colors, edgecolor='black',
                   linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('BIC (lower = better fit)')
    ax.set_title('B. Geometric series model comparison (density troughs)', fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    savefig(fig, 'fig1_peak_density')


# =========================================================================
# FIGURE 2: Aperiodic null
# =========================================================================

def fig2_aperiodic_null(all_freqs):
    """Aperiodic null test showing troughs are genuine."""
    n_hist = 500
    log_freqs = np.log(all_freqs)
    log_edges = np.linspace(np.log(3), np.log(55), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)

    counts, _ = np.histogram(log_freqs, bins=log_edges)
    real_smooth = gaussian_filter1d(counts.astype(float), sigma=5)
    envelope = gaussian_filter1d(counts.astype(float), sigma=40)
    envelope = np.maximum(envelope, 1)

    # Real troughs
    trough_idx, _ = find_peaks(-real_smooth, prominence=np.median(real_smooth) * 0.1,
                                distance=n_hist // 20)
    real_trough_hz = hz_centers[trough_idx]
    real_trough_hz = real_trough_hz[(real_trough_hz > 4) & (real_trough_hz < 50)]

    # Real trough depths
    real_depths = []
    for th in real_trough_hz:
        idx = np.argmin(np.abs(hz_centers - th))
        real_depths.append(real_smooth[idx] / envelope[idx])

    # Surrogates
    rng = np.random.default_rng(42)
    surr_max_depths = []
    for _ in range(200):
        surr = rng.poisson(envelope)
        surr_sm = gaussian_filter1d(surr.astype(float), sigma=5)
        t_idx, _ = find_peaks(-surr_sm, prominence=np.median(surr_sm) * 0.1,
                               distance=n_hist // 20)
        t_hz = hz_centers[t_idx]
        t_hz = t_hz[(t_hz > 4) & (t_hz < 50)]
        depths = []
        for th in t_hz:
            idx = np.argmin(np.abs(hz_centers - th))
            depths.append(surr_sm[idx] / (envelope[idx] + 1e-10))
        surr_max_depths.append(min(depths) if depths else 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Panel A: real density vs envelope
    ax = axes[0]
    ax.plot(hz_centers, real_smooth, color='steelblue', linewidth=1.5, label='Real')
    ax.plot(hz_centers, envelope, color=C_NEG, linewidth=2, label='Aperiodic envelope')
    for th in real_trough_hz:
        ax.axvline(th, color=C_POS, alpha=0.5, linewidth=1, linestyle='--')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([5, 7, 10, 13, 20, 30, 40])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak count (smoothed)')
    ax.legend(fontsize=8)
    ax.set_title('A. Real vs. aperiodic envelope', fontweight='bold')
    ax.set_xlim(3, 55)

    # Panel B: surrogate distribution
    ax = axes[1]
    ax.hist(surr_max_depths, bins=25, color=C_NULL, alpha=0.7, edgecolor='black', linewidth=0.5)
    real_deepest = min(real_depths)
    ax.axvline(real_deepest, color=C_NEG, linewidth=2,
               label=f'Real deepest = {real_deepest:.3f}')
    ax.set_xlabel('Deepest trough (count/envelope)')
    ax.set_ylabel('Surrogate count')
    ax.legend(fontsize=8)
    ax.set_title('B. Real vs. 200 surrogates', fontweight='bold')

    plt.tight_layout()
    savefig(fig, 'fig2_aperiodic_null')


# =========================================================================
# FIGURE 3: Boundary sweep heatmap
# =========================================================================

def fig3_boundary_sweep():
    """Boundary sweep simplicity heatmap with named systems."""
    csv_path = os.path.join(OUT_DIR, 'boundary_sweep', 'sweep_results.csv')
    if not os.path.exists(csv_path):
        print("  SKIP fig3: sweep_results.csv not found")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5, 4))

    pivot = df.pivot(index='ratio', columns='f0', values='simplicity_best')
    f0_vals = np.sort(df['f0'].unique())
    ratio_vals = np.sort(df['ratio'].unique())

    im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='YlGnBu',
                   extent=[f0_vals[0], f0_vals[-1], ratio_vals[0], ratio_vals[-1]])
    plt.colorbar(im, ax=ax, label='Profile simplicity ($R^2$)', shrink=0.9)

    # Mark named systems
    systems = {
        '$\\varphi$-lattice': (7.60, PHI, '*', C_PHI),
        'Octave': (8.0, 2.0, 's', 'white'),
        'Clinical': (8.0, 1.63, 'D', 'white'),
        '$\\sqrt{2}$': (8.0, 1.414, '^', 'white'),
    }
    for label, (f0, r, marker, color) in systems.items():
        if f0_vals[0] <= f0 <= f0_vals[-1] and ratio_vals[0] <= r <= ratio_vals[-1]:
            ax.plot(f0, r, marker, color=color, markersize=12,
                    markeredgecolor='black', markeredgewidth=1.2, zorder=10)

    ax.set_xlabel('$f_0$ (Hz)')
    ax.set_ylabel('Scaling ratio')
    ax.set_title('Profile simplicity across coordinate systems', fontweight='bold')

    # Legend
    handles = [Line2D([0], [0], marker=m, color=c, linestyle='None', markersize=8,
                       markeredgecolor='black', label=l)
               for l, (_, _, m, c) in systems.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    savefig(fig, 'fig3_boundary_sweep')


# =========================================================================
# FIGURE 4: Enrichment landscape heatmap
# =========================================================================

def fig4_enrichment_landscape():
    """5 bands × 13 positions enrichment heatmap."""
    fig, ax = plt.subplots(figsize=(7, 3))

    data = np.array([ENRICHMENT[b] for b in BAND_ORDER])
    vmax = 130
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')

    ax.set_xticks(range(len(POS_SHORT)))
    ax.set_xticklabels(POS_SHORT, fontsize=8)
    ax.set_yticks(range(len(BAND_ORDER)))
    ax.set_yticklabels([BAND_LABELS[b] for b in BAND_ORDER])
    ax.set_xlabel('$\\varphi$-lattice position')

    # Annotate cells
    for i in range(len(BAND_ORDER)):
        for j in range(len(POS_SHORT)):
            val = data[i, j]
            color = 'white' if abs(val) > 60 else 'black'
            ax.text(j, i, f'{val:+d}', ha='center', va='center',
                    fontsize=6.5, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Enrichment (%)', shrink=0.9, pad=0.02)
    n_ds = len(EC_DATASETS)
    ax.set_title(f'Enrichment landscape ({n_ds} datasets, cross-dataset mean)', fontweight='bold')

    plt.tight_layout()
    savefig(fig, 'fig4_enrichment_landscape')


# =========================================================================
# FIGURE 5: Cross-boundary architecture
# =========================================================================

def fig5_cross_boundary():
    """Four boundary types: cliff, void, bridge, weak."""
    # 6-dataset means (EEGMMIDB, LEMON, Dortmund, CHBMP, HBN merged, TDBRAIN)
    boundaries = [
        ('$\\theta / \\alpha$\n(Cliff)', 7.60, 'theta', 'alpha',
         [(7.09, +58), (7.28, +74), (7.40, +80), (7.60, +94)],
         [(7.60, -6), (7.81, -2), (7.94, +3), (8.15, +4)]),
        ('$\\alpha / \\beta_L$\n(Void)', 12.30, 'alpha', 'beta_low',
         [(11.46, -18), (11.77, -31), (11.97, -43), (12.30, -52)],
         [(12.30, -34), (12.63, -44), (12.85, -52), (13.19, -56)]),
        ('$\\beta_L / \\beta_H$\n(Bridge)', 19.90, 'beta_low', 'beta_high',
         [(18.55, +54), (19.06, +66), (19.38, +69), (19.90, +72)],
         [(19.90, +44), (20.44, +37), (20.78, +32), (21.35, +18)]),
        ('$\\beta_H / \\gamma$\n(Weak)', 32.19, 'beta_high', 'gamma',
         [(30.02, -5), (30.83, -6), (31.35, -9), (32.19, -12)],
         [(32.19, -12), (33.06, -16), (33.62, -16), (34.53, -20)]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7, 2.8), sharey=True)

    for ax, (title, bnd_hz, band_lo, band_hi, pts_lo, pts_hi) in zip(axes, boundaries):
        hz_lo = [p[0] for p in pts_lo]
        enr_lo = [p[1] for p in pts_lo]
        hz_hi = [p[0] for p in pts_hi]
        enr_hi = [p[1] for p in pts_hi]

        ax.plot(hz_lo, enr_lo, 'o-', color=BAND_COLORS[band_lo], markersize=4, linewidth=1.5)
        ax.plot(hz_hi, enr_hi, 's-', color=BAND_COLORS[band_hi], markersize=4, linewidth=1.5)
        ax.axvline(bnd_hz, color='black', linewidth=1, linestyle=':')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Hz', fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel('Enrichment (%)')
    plt.suptitle('Cross-boundary architecture', fontweight='bold', fontsize=11)
    plt.tight_layout()
    savefig(fig, 'fig5_cross_boundary')


# =========================================================================
# FIGURE 6: Cognitive correlations
# =========================================================================

_COGNITIVE_FALLBACK = [
    ('LPS', 'βL center_depl', -0.273),
    ('TAP', 'θ ushape', +0.267),
    ('TAP', 'θ boundary', +0.263),
    ('TAP', 'θ mountain', -0.260),
    ('RWT', 'γ inv_noble₄', +0.260),
    ('RWT', 'γ ramp_depth', +0.256),
    ('TMT', 'α inv_noble₆', -0.253),
    ('LPS', 'βL mountain', -0.250),
    ('LPS', 'βL ushape', +0.249),
    ('LPS', 'βL attractor', -0.245),
    ('LPS', 'γ inv_noble₄', +0.243),
    ('LPS', 'βL boundary', +0.240),
]

_BAND_ABBREV = {
    'theta': 'θ', 'alpha': 'α', 'beta_low': 'βL',
    'beta_high': 'βH', 'gamma': 'γ',
}

# Ordered longest-first so beta_low matches before beta
_BAND_PREFIXES = sorted(_BAND_ABBREV.keys(), key=len, reverse=True)


def _parse_feature(feature_name):
    """Split 'beta_low_ushape' -> ('beta_low', 'ushape'). Handles compound band names."""
    for prefix in _BAND_PREFIXES:
        if feature_name.startswith(prefix + '_'):
            return prefix, feature_name[len(prefix) + 1:]
        if feature_name == prefix:
            return prefix, prefix
    # Fallback: split on first underscore
    parts = feature_name.split('_', 1)
    return parts[0], (parts[1] if len(parts) > 1 else parts[0])


def _load_cognitive(n_top=12):
    """Load top FDR-significant cognitive correlations from analysis CSV."""
    csv_path = os.path.join(ANALYSIS_DIR, 'cognitive_correlations.csv')
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found, using fallback cognitive data")
        return _COGNITIVE_FALLBACK, 203
    df = pd.read_csv(csv_path)
    sig = df[df['significant'] == True].copy()
    sig = sig.sort_values('abs_rho', ascending=False).head(n_top)
    data = []
    for _, row in sig.iterrows():
        target = row['target']
        band_key, feat_name = _parse_feature(row['feature'])
        abbrev = _BAND_ABBREV.get(band_key, band_key)
        data.append((target, f'{abbrev} {feat_name}', round(row['rho'], 3)))
    n_subj = int(sig['n'].iloc[0]) if len(sig) > 0 else 203
    return data, n_subj


def fig6_cognitive():
    """Top FDR-significant cognitive correlations."""
    data, n_subj = _load_cognitive()

    fig, ax = plt.subplots(figsize=(5, 4))

    labels = [f'{d[0]}: {d[1]}' for d in data]
    rhos = [d[2] for d in data]
    colors = [C_POS if r > 0 else C_NEG for r in rhos]

    y = range(len(data))
    ax.barh(y, rhos, color=colors, edgecolor='black', linewidth=0.5, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Spearman $\\rho$')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title(f'Top {len(data)} FDR-significant cognition correlations\n'
                 f'(LEMON EC, $N={n_subj}$, $q<0.05$)',
                 fontweight='bold')
    rho_max = max(abs(r) for r in rhos) + 0.05 if rhos else 0.32
    ax.set_xlim(-rho_max, rho_max)

    plt.tight_layout()
    savefig(fig, 'fig6_cognitive')


# =========================================================================
# FIGURE 7: Inverted-U lifespan trajectory
# =========================================================================

_LIFESPAN_FALLBACK = [
    ('α asymmetry', +0.327, -0.241, C_ALPHA),
    ('α inv_noble₃', +0.316, -0.276, C_ALPHA),
    ('α inv_noble₄', +0.308, -0.199, C_ALPHA),
    ('α ramp_depth', +0.299, -0.205, C_ALPHA),
    ('βH center_depl', -0.263, +0.150, C_BETA_H),
    ('βL attractor', -0.135, +0.311, C_BETA_L),
]

_BAND_COLOR_MAP = {
    'alpha': C_ALPHA, 'theta': C_THETA, 'beta_low': C_BETA_L,
    'beta_high': C_BETA_H, 'gamma': C_GAMMA,
}


def _load_lifespan(n_top=6):
    """Load top inverted-U lifespan features from analysis CSV."""
    csv_path = os.path.join(ANALYSIS_DIR, 'lifespan_trajectory.csv')
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found, using fallback lifespan data")
        return _LIFESPAN_FALLBACK
    df = pd.read_csv(csv_path)
    inv_u = df[df['pattern'] == 'inverted-U'].copy()
    inv_u['abs_hbn'] = inv_u['hbn_rho'].abs()
    inv_u = inv_u.sort_values('abs_hbn', ascending=False).head(n_top)
    features = []
    for _, row in inv_u.iterrows():
        band_key, feat_name = _parse_feature(row['feature'])
        abbrev = _BAND_ABBREV.get(band_key, band_key)
        color = _BAND_COLOR_MAP.get(band_key, C_ALPHA)
        features.append((f'{abbrev} {feat_name}', round(row['hbn_rho'], 3),
                         round(row['dort_rho'], 3), color))
    return features


def fig7_lifespan():
    """Four-phase lifespan trajectory: rise, plateau, mid-life decline, late-life degradation."""
    features = _load_lifespan()

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Panel A: HBN vs Dortmund rho comparison
    ax = axes[0]
    labels = [f[0] for f in features]
    dev_rho = [f[1] for f in features]
    age_rho = [f[2] for f in features]
    colors = [f[3] for f in features]

    y = np.arange(len(features))
    ax.barh(y - 0.15, dev_rho, 0.3, color=[c for c in colors], alpha=0.7,
            edgecolor='black', linewidth=0.5, label='Development (HBN, 5-22)')
    ax.barh(y + 0.15, age_rho, 0.3, color=[c for c in colors], alpha=0.3,
            edgecolor='black', linewidth=0.5, hatch='//',
            label='Aging (Dortmund, 20-70)')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Spearman $\\rho$ with age')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title('A. Opposite age effects', fontweight='bold')
    ax.invert_yaxis()

    # Panel B: Four-phase trajectory schematic (HBN + Dortmund + TDBRAIN)
    ax = axes[1]

    # Phase 1: HBN developmental rise (5-22)
    x_hbn = np.linspace(5, 22, 80)
    y_hbn = 0.04 * (x_hbn - 5) + 0.1
    ax.plot(x_hbn, y_hbn, '-', linewidth=2.5, color=C_POS)
    ax.fill_between(x_hbn, y_hbn, alpha=0.15, color=C_POS, label='Differentiation ↑')

    # Gap at HBN-Dortmund splice (22-23)
    x_gap1 = np.linspace(22, 23, 10)
    y_gap_lo = y_hbn[-1]
    y_gap_hi = y_gap_lo + 0.05
    ax.plot(x_gap1, np.linspace(y_gap_lo, y_gap_hi, len(x_gap1)),
            '--', linewidth=1.5, color='gray', alpha=0.5)

    # Phase 2-3: Dortmund plateau + mid-life decline (23-70)
    x_dort = np.linspace(23, 70, 120)
    y_dort = y_gap_hi - 0.012 * (x_dort - 22)
    ax.plot(x_dort, y_dort, '-', linewidth=2.5, color=C_NEG)
    ax.fill_between(x_dort, y_dort, alpha=0.15, color=C_NEG, label='De-differentiation ↓')

    # Phase 4: TDBRAIN late-life degradation (70-88)
    # TDBRAIN α/β depletion: 47.8% (45-60) → 36.7% (60-90)
    # Normalized: continued decline steepening past 70
    x_tdb = np.linspace(70, 88, 40)
    y_dort_end = y_dort[-1]
    y_tdb = y_dort_end - 0.015 * (x_tdb - 70)  # steeper decline in late life
    ax.plot(x_tdb, y_tdb, '-', linewidth=2.5, color='#8e44ad')  # purple for TDBRAIN
    ax.fill_between(x_tdb, y_tdb, alpha=0.15, color='#8e44ad', label='Late-life degradation')

    # Dataset boundary annotations
    ax.axvline(21, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvline(70, color='black', linestyle='--', linewidth=1.0, alpha=0.5)

    # Dataset labels
    ax.text(13, 0.05, 'HBN\n(5–22)', ha='center', fontsize=6.5, color=C_THETA)
    ax.text(46, 0.05, 'Dortmund\n(20–70)', ha='center', fontsize=6.5, color=C_BETA_L)
    ax.text(79, 0.05, 'TDBRAIN\n(60–88)', ha='center', fontsize=6.5, color='#8e44ad')

    # Phase labels
    ax.annotate('rise', xy=(13, y_hbn[40]), fontsize=7, color=C_POS,
                fontweight='bold', ha='center', va='bottom')
    ax.annotate('plateau', xy=(30, y_gap_hi + 0.02), fontsize=7, color='gray',
                fontweight='bold', ha='center')
    ax.annotate('decline', xy=(55, y_dort[80] + 0.02), fontsize=7, color=C_NEG,
                fontweight='bold', ha='center')
    ax.annotate('degradation', xy=(79, y_tdb[20] + 0.02), fontsize=7,
                color='#8e44ad', fontweight='bold', ha='center')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Spectral differentiation')
    ax.set_title('B. Four-phase lifespan trajectory', fontweight='bold')
    ax.set_xlim(0, 93)
    ax.legend(fontsize=6.5, loc='upper right')

    plt.tight_layout()
    savefig(fig, 'fig7_lifespan')


# =========================================================================
# FIGURE 8: Reliability
# =========================================================================

_ICC_FALLBACK = {
    'band_medians': {'beta_low': 0.604, 'beta_high': 0.507, 'alpha': 0.454,
                     'theta': 0.382, 'gamma': 0.250},
    'best_feature': ('beta_low_ushape', 0.746),
    'overall_median': 0.421,
    'n_subj': 208,
}


def _load_icc():
    """Load test-retest ICC from analysis CSV."""
    csv_path = os.path.join(ANALYSIS_DIR, 'test_retest_reliability.csv')
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found, using fallback ICC data")
        return _ICC_FALLBACK
    df = pd.read_csv(csv_path)
    # Extract band from feature name using compound-aware parser
    df['band'] = df['feature'].apply(lambda f: _parse_feature(f)[0])
    band_medians = df.groupby('band')['icc'].median().to_dict()
    best_idx = df['icc'].idxmax()
    best_feature = df.loc[best_idx, 'feature']
    best_icc = df.loc[best_idx, 'icc']
    overall_median = df['icc'].median()
    n_subj = int(df['n'].iloc[0])
    return {
        'band_medians': band_medians,
        'best_feature': (best_feature, best_icc),
        'overall_median': overall_median,
        'n_subj': n_subj,
    }


def fig8_reliability():
    """Five-year test-retest ICC and group profile stability."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    icc_data = _load_icc()
    band_medians = icc_data['band_medians']
    best_feat, best_icc = icc_data['best_feature']
    overall_med = icc_data['overall_median']
    n_subj = icc_data['n_subj']

    # Panel A: ICC by band (sorted descending)
    ax = axes[0]
    band_keys = ['beta_low', 'beta_high', 'alpha', 'theta', 'gamma']
    band_keys = sorted([b for b in band_keys if b in band_medians],
                       key=lambda b: band_medians[b], reverse=True)
    band_labels = [BAND_LABELS.get(b, b) for b in band_keys]
    iccs = [round(band_medians[b], 3) for b in band_keys]
    colors = [BAND_COLORS[b] for b in band_keys]

    bars = ax.barh(range(len(band_labels)), iccs, color=colors, edgecolor='black',
                   linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(band_labels)))
    ax.set_yticklabels(band_labels)
    ax.set_xlabel('Median ICC (5-year)')
    ax.set_xlim(0, max(best_icc + 0.1, 0.8))

    # Mark best individual feature
    best_band = best_feat.replace('_ushape', '').replace('_asymmetry', '')
    best_short = best_feat.split('_', 2)[-1] if '_' in best_feat else best_feat
    best_band_idx = next((i for i, b in enumerate(band_keys) if best_feat.startswith(b)), 0)
    ax.plot(best_icc, best_band_idx, 'D', color='black', markersize=8, zorder=10)
    ax.annotate(f'{best_short}\nICC={best_icc:.3f}',
                xy=(best_icc, best_band_idx),
                xytext=(best_icc - 0.02, best_band_idx + 1.5),
                fontsize=7, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.axvline(overall_med, color='gray', linestyle=':', linewidth=1)
    ax.text(overall_med, len(band_keys) - 0.5, f'Overall\n{overall_med:.3f}',
            fontsize=7, ha='center', color='gray')
    ax.set_title(f'A. 5-year test-retest ICC\n(Dortmund, $N={n_subj}$)', fontweight='bold')
    ax.invert_yaxis()

    # Panel B: Group profile stability (13-position enrichment profile correlation
    # between sessions — a derived metric not stored in a separate output CSV,
    # so these values remain hard-coded from the Dortmund 5-year analysis)
    ax = axes[1]
    bands_b = ['Beta-low', 'Beta-high', 'Theta', 'Alpha', 'Gamma']
    r_vals = [0.988, 0.987, 0.983, 0.977, 0.964]
    colors_b = [BAND_COLORS[b] for b in ['beta_low', 'beta_high', 'theta', 'alpha', 'gamma']]

    bars = ax.barh(range(len(bands_b)), r_vals, color=colors_b, edgecolor='black',
                   linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(bands_b)))
    ax.set_yticklabels(bands_b)
    ax.set_xlabel('Profile $r$ (ses-1 vs ses-2)')
    ax.set_xlim(0.9, 1.0)
    ax.set_title('B. Group profile stability\n(5 years apart)', fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    savefig(fig, 'fig8_reliability')


# =========================================================================
# MAIN
# =========================================================================

def fig9_bootstrap():
    """Copy bootstrap figure to papers/images/ (already generated by bootstrap script)."""
    src = os.path.join(OUT_DIR, 'bootstrap_troughs', 'bootstrap_troughs.png')
    dst = os.path.join(IMG_DIR, 'fig9_bootstrap.png')
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        print(f"  Copied {dst}")
    else:
        print(f"  SKIP fig9: {src} not found (run bootstrap_trough_locations.py first)")


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Loading peak data...")
    all_freqs = load_all_freqs()
    print(f"Total: {len(all_freqs):,} peaks\n")

    print("Generating figures...")
    fig1_peak_density(all_freqs)
    fig2_aperiodic_null(all_freqs)
    fig3_boundary_sweep()
    fig4_enrichment_landscape()
    fig5_cross_boundary()
    fig6_cognitive()
    fig7_lifespan()
    fig8_reliability()
    fig9_bootstrap()

    print(f"\nAll figures saved to {IMG_DIR}/")


if __name__ == '__main__':
    main()
