#!/usr/bin/env python3
"""
Log-phi histogram of dominant peak frequencies per PHI-OCTAVE band,
with degree-7 noble position reference lines and frequency x-axis labels.

Each phi-octave (f₀·φⁿ to f₀·φⁿ⁺¹) gets one subplot showing the distribution
of u = frac(log_φ(f/f₀)) with all 14 degree-7 positions marked.
Top x-axis shows actual frequency in Hz for that octave.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────

F0 = 7.83
PHI = (1 + np.sqrt(5)) / 2

CONV_BANDS = ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

# Phi-octave definitions
PHI_OCTAVES = {
    -2: (F0 * PHI**-2, F0 * PHI**-1),   # 2.99 – 4.84 Hz
    -1: (F0 * PHI**-1, F0 * PHI**0),     # 4.84 – 7.83 Hz  (≈ theta)
     0: (F0 * PHI**0,  F0 * PHI**1),     # 7.83 – 12.67 Hz (≈ alpha)
     1: (F0 * PHI**1,  F0 * PHI**2),     # 12.67 – 20.50 Hz (≈ beta_low)
     2: (F0 * PHI**2,  F0 * PHI**3),     # 20.50 – 33.17 Hz (≈ beta_high)
     3: (F0 * PHI**3,  F0 * PHI**4),     # 33.17 – 53.66 Hz (≈ gamma)
}
PHI_OCT_LABELS = {
    -2: f'φ⁻² ({F0*PHI**-2:.2f} – {F0*PHI**-1:.2f} Hz)',
    -1: f'φ⁻¹ ({F0*PHI**-1:.2f} – {F0:.2f} Hz)',
     0: f'φ⁰ ({F0:.2f} – {F0*PHI:.2f} Hz)',
     1: f'φ¹ ({F0*PHI:.2f} – {F0*PHI**2:.2f} Hz)',
     2: f'φ² ({F0*PHI**2:.2f} – {F0*PHI**3:.2f} Hz)',
     3: f'φ³ ({F0*PHI**3:.2f} – {F0*PHI**4:.2f} Hz)',
}
OCTAVE_ORDER = [-1, 0, 1, 2, 3]  # skip -2 (too few peaks in delta)

# Degree-7 positions
POSITIONS_14 = {
    'boundary':    0.000,
    'noble_7':     (1/PHI)**7,
    'noble_6':     (1/PHI)**6,
    'noble_5':     (1/PHI)**5,
    'noble_4':     (1/PHI)**4,
    'noble_3':     (1/PHI)**3,
    'noble_2':     1 - 1/PHI,
    'attractor':   0.500,
    'noble_1':     1/PHI,
    'inv_noble_3': 1 - (1/PHI)**3,
    'inv_noble_4': 1 - (1/PHI)**4,
    'inv_noble_5': 1 - (1/PHI)**5,
    'inv_noble_6': 1 - (1/PHI)**6,
    'inv_noble_7': 1 - (1/PHI)**7,
}

POS_LABELS = {
    'boundary': 'bnd', 'noble_7': 'n₇', 'noble_6': 'n₆', 'noble_5': 'n₅',
    'noble_4': 'n₄', 'noble_3': 'n₃', 'noble_2': 'n₂', 'attractor': 'att',
    'noble_1': 'n₁', 'inv_noble_3': 'ĩ₃', 'inv_noble_4': 'ĩ₄',
    'inv_noble_5': 'ĩ₅', 'inv_noble_6': 'ĩ₆', 'inv_noble_7': 'ĩ₇',
}

# Visual styling for position lines
POS_STYLES = {
    'boundary':    {'color': '#E63946', 'lw': 2.5, 'ls': '-',  'alpha': 0.9},
    'attractor':   {'color': '#457B9D', 'lw': 2.5, 'ls': '-',  'alpha': 0.9},
    'noble_1':     {'color': '#0B6E4F', 'lw': 2.0, 'ls': '-',  'alpha': 0.85},
    'noble_2':     {'color': '#0B6E4F', 'lw': 1.8, 'ls': '--', 'alpha': 0.85},
    'noble_3':     {'color': '#0B6E4F', 'lw': 1.6, 'ls': '--', 'alpha': 0.80},
    'noble_4':     {'color': '#0B6E4F', 'lw': 1.4, 'ls': '--', 'alpha': 0.80},
    'noble_5':     {'color': '#0B6E4F', 'lw': 1.2, 'ls': '--', 'alpha': 0.75},
    'noble_6':     {'color': '#0B6E4F', 'lw': 1.0, 'ls': '--', 'alpha': 0.75},
    'noble_7':     {'color': '#0B6E4F', 'lw': 1.0, 'ls': '--', 'alpha': 0.75},
    'inv_noble_3': {'color': '#BF8B2E', 'lw': 1.6, 'ls': '--', 'alpha': 0.80},
    'inv_noble_4': {'color': '#BF8B2E', 'lw': 1.4, 'ls': '--', 'alpha': 0.80},
    'inv_noble_5': {'color': '#BF8B2E', 'lw': 1.2, 'ls': '--', 'alpha': 0.75},
    'inv_noble_6': {'color': '#BF8B2E', 'lw': 1.0, 'ls': '--', 'alpha': 0.75},
    'inv_noble_7': {'color': '#BF8B2E', 'lw': 1.0, 'ls': '--', 'alpha': 0.75},
}

OCTAVE_COLORS = {
    -2: '#9B59B6', -1: '#1982C4', 0: '#2ECC71',
     1: '#F39C12', 2: '#E74C3C', 3: '#8E44AD',
}

# ── Data ───────────────────────────────────────────────────────────────────

DATASETS = {
    'EEGMMIDB': 'exports_eegmmidb/replication/combined/per_subject_dominant_peaks.csv',
    'LEMON_EC': 'exports_lemon/replication/EC/per_subject_dominant_peaks.csv',
    'LEMON_EO': 'exports_lemon/replication/EO/per_subject_dominant_peaks.csv',
    'HBN_EC':   'exports_hbn/combined/per_subject_dominant_peaks.csv',
    'Dortmund_EC': '/Volumes/T9/dortmund_data/lattice_results_replication_v2/EyesClosed_pre/per_subject_dominant_peaks.csv',
}

BASE_DIR = Path('/Users/neurokinetikz/Code/schumann')


def lattice_coord(freq):
    """Compute u = frac(log_φ(freq / f₀))."""
    return np.log(freq / F0) / np.log(PHI) % 1.0


def phi_octave_index(freq):
    """Return the phi-octave index n such that f₀·φⁿ ≤ freq < f₀·φⁿ⁺¹."""
    return np.floor(np.log(freq / F0) / np.log(PHI)).astype(int)


def freq_ticks_for_octave(n_oct):
    """Generate frequency tick marks for a given phi-octave."""
    f_lo = F0 * PHI**n_oct
    f_hi = F0 * PHI**(n_oct + 1)
    span = f_hi - f_lo

    if span > 10:
        step = 2.0
    elif span > 5:
        step = 1.0
    elif span > 2:
        step = 0.5
    else:
        step = 0.25

    first = np.ceil(f_lo / step) * step
    nice_freqs = np.arange(first, f_hi + 0.01, step)

    # Include boundaries
    all_freqs = np.sort(np.unique(np.round(
        np.concatenate([[f_lo], nice_freqs, [f_hi]]), 2)))

    u_positions = []
    labels = []
    for f in all_freqs:
        u = np.log(f / F0) / np.log(PHI) % 1.0
        u_positions.append(u)
        labels.append(f'{f:.1f}')

    return u_positions, labels, f_lo, f_hi


def load_data():
    """Load all dominant peaks, pool into one array of frequencies per dataset."""
    all_freqs = []   # flat array of all peak frequencies
    all_freqs_by_ds = {}

    for name, path in DATASETS.items():
        fpath = Path(path) if path.startswith('/') else BASE_DIR / path
        if not fpath.exists():
            print(f"  SKIP {name}: {fpath}")
            continue
        df = pd.read_csv(fpath)
        print(f"  {name:<15} N={len(df)}")

        ds_freqs = []
        for band in CONV_BANDS:
            col = f'{band}_freq'
            if col in df.columns:
                vals = df[col].dropna().values
                ds_freqs.append(vals)
                all_freqs.append(vals)
        all_freqs_by_ds[name] = np.concatenate(ds_freqs)

    return np.concatenate(all_freqs), all_freqs_by_ds


def draw_position_lines(ax):
    """Draw all 14 degree-7 position reference lines on an axis."""
    for pos_name, pos_u in POSITIONS_14.items():
        style = POS_STYLES[pos_name]
        a = style['alpha']
        if pos_u == 0.0:
            ax.axvline(0.0, color=style['color'], lw=style['lw'],
                       ls=style['ls'], alpha=a, zorder=5)
            ax.axvline(1.0, color=style['color'], lw=style['lw'],
                       ls=style['ls'], alpha=a, zorder=5)
        else:
            ax.axvline(pos_u, color=style['color'], lw=style['lw'],
                       ls=style['ls'], alpha=a, zorder=5)


def draw_position_labels(ax):
    """Add text labels for all 14 positions at top of axis."""
    ylim = ax.get_ylim()
    for pos_name, pos_u in POSITIONS_14.items():
        short = POS_LABELS[pos_name]
        x = pos_u if pos_u > 0 else 0.008
        ax.text(x, ylim[1] * 0.97, short,
                ha='center', va='top', fontsize=6.5,
                color=POS_STYLES[pos_name]['color'], fontweight='bold',
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          alpha=0.6, edgecolor='none'))


def make_figure(all_freqs, n_bins=80):
    """Create phi-octave histogram figure with frequency x-axis labels."""
    n_oct = len(OCTAVE_ORDER)
    fig, axes = plt.subplots(n_oct, 1, figsize=(14, 4.5 * n_oct),
                             gridspec_kw={'hspace': 0.45})
    fig.suptitle('Dominant Peak Distributions by φ-Octave\n'
                 f'f₀ = {F0} Hz, base = φ, degree-7 positions '
                 f'(N={len(all_freqs):,} peaks across 5 datasets)',
                 fontsize=14, fontweight='bold', y=0.99)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    uniform = 1.0 / n_bins

    # Assign each frequency to its phi-octave
    oct_idx = phi_octave_index(all_freqs)

    for panel, n_o in enumerate(OCTAVE_ORDER):
        ax = axes[panel]
        f_lo, f_hi = PHI_OCTAVES[n_o]

        # Select frequencies in this octave
        mask = (oct_idx == n_o)
        freqs = all_freqs[mask]
        n_peaks = len(freqs)

        if n_peaks == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        u = lattice_coord(freqs)

        # Histogram
        counts, _ = np.histogram(u, bins=bins)
        density = counts / counts.sum()
        enrichment = (density - uniform) / uniform * 100

        # Bars with enrichment-based opacity
        color = OCTAVE_COLORS[n_o]
        for i in range(n_bins):
            a = 0.35 + 0.65 * min(1.0, abs(enrichment[i]) / 100)
            ax.bar(bin_centers[i], density[i], width=1.0/n_bins,
                   color=color, alpha=a, edgecolor='none')

        # Smooth overlay
        smooth = gaussian_filter1d(density, sigma=1.5)
        ax.plot(bin_centers, smooth, color=color, lw=2.5, alpha=0.9)

        # Uniform null
        ax.axhline(uniform, color='gray', lw=0.8, ls='--', alpha=0.5, zorder=1)

        # Position lines
        draw_position_lines(ax)

        # Band label
        ax.text(0.02, 0.92,
                f'{PHI_OCT_LABELS[n_o]}   N={n_peaks:,}',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.set_ylabel('Density', fontsize=10)

        # ── Frequency x-axis (top) ──
        u_ticks, freq_lbls, _, _ = freq_ticks_for_octave(n_o)
        ax2 = ax.twiny()
        ax2.set_xlim(0, 1)
        ax2.set_xticks(u_ticks)
        ax2.set_xticklabels(freq_lbls, fontsize=8.5, color='#444444')
        ax2.set_xlabel('Frequency (Hz)', fontsize=9, color='#444444', labelpad=4)
        ax2.tick_params(axis='x', length=4, pad=2, colors='#444444')

        # ── u-axis (bottom) with noble position tick marks ──
        u_tick_vals = sorted(POSITIONS_14.values())
        u_tick_labels = []
        for uv in u_tick_vals:
            for name, val in POSITIONS_14.items():
                if abs(val - uv) < 1e-6:
                    u_tick_labels.append(POS_LABELS[name])
                    break
        ax.set_xticks(u_tick_vals)
        ax.set_xticklabels(u_tick_labels, fontsize=7.5, rotation=0)
        ax.tick_params(axis='x', length=4, pad=2)

    # Bottom label
    axes[-1].set_xlabel('φ-octave coordinate  u = frac(log_φ(f / f₀))', fontsize=12)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#E63946', lw=2.5, ls='-', label='Boundary (u=0)'),
        Line2D([0], [0], color='#457B9D', lw=2.5, ls='-', label='Attractor (u=0.5)'),
        Line2D([0], [0], color='#0B6E4F', lw=1.8, ls='--', label='Nobles (n₁–n₇)'),
        Line2D([0], [0], color='#BF8B2E', lw=1.4, ls='--', label='Inv. nobles (ĩ₃–ĩ₇)'),
        Line2D([0], [0], color='gray', lw=0.8, ls='--', label='Uniform null'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.002))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def make_per_dataset_figure(all_freqs_by_ds, n_bins=60):
    """Per-dataset overlay version with phi-octave bands."""
    n_oct = len(OCTAVE_ORDER)
    fig, axes = plt.subplots(n_oct, 1, figsize=(14, 4.5 * n_oct),
                             gridspec_kw={'hspace': 0.45})
    fig.suptitle('Per-Dataset φ-Octave Distributions\n'
                 f'f₀ = {F0} Hz, degree-7 positions',
                 fontsize=14, fontweight='bold', y=0.99)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    uniform = 1.0 / n_bins

    dataset_colors = {
        'EEGMMIDB': '#1f77b4',
        'LEMON_EC': '#2ca02c',
        'LEMON_EO': '#98df8a',
        'HBN_EC': '#ff7f0e',
        'Dortmund_EC': '#d62728',
    }

    for panel, n_o in enumerate(OCTAVE_ORDER):
        ax = axes[panel]
        f_lo, f_hi = PHI_OCTAVES[n_o]

        for ds_name, ds_freqs in all_freqs_by_ds.items():
            oct_idx = phi_octave_index(ds_freqs)
            mask = (oct_idx == n_o)
            freqs = ds_freqs[mask]
            if len(freqs) == 0:
                continue
            u = lattice_coord(freqs)
            counts, _ = np.histogram(u, bins=bins)
            density = counts / counts.sum()
            smooth = gaussian_filter1d(density, sigma=1.5)
            ax.plot(bin_centers, smooth,
                    color=dataset_colors.get(ds_name, 'gray'),
                    lw=1.8, alpha=0.85,
                    label=f'{ds_name} (N={len(freqs)})')

        # Uniform reference
        ax.axhline(uniform, color='gray', lw=0.8, ls='--', alpha=0.5)

        # Position lines
        draw_position_lines(ax)

        ax.text(0.02, 0.92, PHI_OCT_LABELS[n_o],
                transform=ax.transAxes, fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.legend(fontsize=7.5, loc='upper right', ncol=2, framealpha=0.85)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)

        # Frequency x-axis (top)
        u_ticks, freq_lbls, _, _ = freq_ticks_for_octave(n_o)
        ax2 = ax.twiny()
        ax2.set_xlim(0, 1)
        ax2.set_xticks(u_ticks)
        ax2.set_xticklabels(freq_lbls, fontsize=8.5, color='#444444')
        ax2.set_xlabel('Frequency (Hz)', fontsize=9, color='#444444', labelpad=4)
        ax2.tick_params(axis='x', length=4, pad=2, colors='#444444')

        # u-axis with position labels
        u_tick_vals = sorted(POSITIONS_14.values())
        u_tick_labels = []
        for uv in u_tick_vals:
            for name, val in POSITIONS_14.items():
                if abs(val - uv) < 1e-6:
                    u_tick_labels.append(POS_LABELS[name])
                    break
        ax.set_xticks(u_tick_vals)
        ax.set_xticklabels(u_tick_labels, fontsize=7.5, rotation=0)

    axes[-1].set_xlabel('φ-octave coordinate  u = frac(log_φ(f / f₀))', fontsize=12)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    return fig


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading datasets...")
    all_freqs, all_freqs_by_ds = load_data()

    total = len(all_freqs)
    print(f"\n  TOTAL: {total:,} dominant peaks across {len(all_freqs_by_ds)} datasets")

    # Phi-octave breakdown
    oct_idx = phi_octave_index(all_freqs)
    for n_o in OCTAVE_ORDER:
        n = (oct_idx == n_o).sum()
        f_lo, f_hi = PHI_OCTAVES[n_o]
        print(f"  φ-oct {n_o:+d}: {n:>5,} peaks  ({f_lo:.2f}–{f_hi:.2f} Hz)")

    # Figure 1: Pooled
    print("\nCreating pooled φ-octave histogram...")
    fig1 = make_figure(all_freqs, n_bins=80)
    out1 = BASE_DIR / 'phi_octave_histograms_pooled.png'
    fig1.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out1}")
    plt.close(fig1)

    # Figure 2: Per-dataset
    print("Creating per-dataset overlay...")
    fig2 = make_per_dataset_figure(all_freqs_by_ds, n_bins=60)
    out2 = BASE_DIR / 'phi_octave_histograms_per_dataset.png'
    fig2.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out2}")
    plt.close(fig2)

    print("\nDone.")
