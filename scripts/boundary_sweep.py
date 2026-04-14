#!/usr/bin/env python3
"""
Boundary Sweep: Coordinate System Comparison for EEG Band Definitions
======================================================================

Sweeps the 2D parameter space (f0, ratio) to evaluate which coordinate
system best describes the empirical peak distribution across EEG frequency
bands.

For any coordinate system defined by a base frequency f0 and a scaling
ratio r, band boundaries are placed at f0 * r^n. Within each band,
peaks are mapped to a normalized coordinate u in [0, 1] via:

    u = log(freq / f_lower) / log(r)

This u-coordinate is INVARIANT to the choice of log base -- it depends
only on where the band boundaries fall in Hz.

Metrics evaluated at each (f0, r) point:
  1. Boundary sharpness   -- discontinuity in enrichment at band edges
  2. Profile simplicity   -- variance explained by a linear ramp per band
  3. Cross-band independence -- decorrelation of adjacent band profiles
  4. Cross-dataset consistency -- reproducibility across 9 datasets
  5. Peak coverage        -- fraction of peaks within the defined bands

Usage:
    python scripts/boundary_sweep.py
    python scripts/boundary_sweep.py --n-bins 16 --f0-range 6.0 9.0 --ratio-range 1.4 2.2
    python scripts/boundary_sweep.py --plot  # generate heatmap figures

Outputs to: outputs/boundary_sweep/
"""

import os
import sys
import argparse
import glob
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'boundary_sweep')

# Datasets (EC conditions only, matching v3 analysis)
EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp', 'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3', 'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
}

# Power filter: keep top 50% by power within each band (matching v3)
MIN_POWER_PCT = 50

# Default sweep ranges
DEFAULT_F0_RANGE = (6.0, 9.5)
DEFAULT_RATIO_RANGE = (1.3, 2.3)
DEFAULT_F0_STEPS = 36
DEFAULT_RATIO_STEPS = 36
DEFAULT_N_BINS = 12  # equal-spaced bins in u-space for fair comparison

# Named coordinate systems for annotation
NAMED_SYSTEMS = {
    'phi_lattice':     (7.60, 1.6180),
    'octave':          (8.00, 2.0000),
    'clinical_approx': (8.00, 1.6300),  # ~4, 8, 13, 21, 34 Hz
    'third_octave':    (7.60, 1.2599),  # 2^(1/3)
    'sqrt2':           (8.00, 1.4142),  # 2^(1/2)
}

# Approximate clinical EEG boundaries for comparison
CLINICAL_BOUNDARIES = [4.0, 8.0, 13.0, 30.0, 100.0]


# =========================================================================
# DATA LOADING
# =========================================================================

def load_all_peaks(min_power_pct=MIN_POWER_PCT):
    """Load peaks from all EC datasets. Returns dict of {name: DataFrame}.

    Each DataFrame has columns: freq, power, phi_octave (original assignment).
    We keep the raw freq values -- reassignment to bands happens per sweep point.
    """
    datasets = {}
    for name, subdir in EC_DATASETS.items():
        path = os.path.join(PEAK_BASE, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            print(f"  {name}: no data at {path}")
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        dfs = []
        for f in files:
            dfs.append(pd.read_csv(f, usecols=cols))
        peaks = pd.concat(dfs, ignore_index=True)

        # Apply power filter per original phi_octave band
        if has_power and min_power_pct > 0:
            filtered = []
            for octave in peaks['phi_octave'].unique():
                bp = peaks[peaks.phi_octave == octave]
                thresh = bp['power'].quantile(min_power_pct / 100)
                filtered.append(bp[bp['power'] >= thresh])
            peaks = pd.concat(filtered, ignore_index=True)

        datasets[name] = peaks
        print(f"  {name}: {len(files)} subjects, {len(peaks):,} peaks")
    return datasets


def pool_all_freqs(datasets):
    """Extract just the frequency arrays per dataset for fast sweep."""
    return {name: df['freq'].values for name, df in datasets.items()}


# =========================================================================
# COORDINATE SYSTEM MACHINERY
# =========================================================================

def compute_boundaries(f0, ratio, n_bands=5, lowest_octave=-1):
    """Compute band boundaries for a given (f0, ratio) system.

    Returns array of n_bands+1 boundary frequencies.
    With defaults: bands span f0*r^-1 to f0*r^4 (5 bands).
    """
    exponents = np.arange(lowest_octave, lowest_octave + n_bands + 1)
    return f0 * ratio ** exponents


def assign_bands(freqs, boundaries):
    """Assign each frequency to a band index (0..n_bands-1).

    Returns integer array, -1 for frequencies outside all bands.
    """
    assignments = np.full(len(freqs), -1, dtype=int)
    for i in range(len(boundaries) - 1):
        mask = (freqs >= boundaries[i]) & (freqs < boundaries[i + 1])
        assignments[mask] = i
    return assignments


def compute_u(freqs, f_lo, f_hi):
    """Map frequencies to normalized u-coordinate within a band.

    u = log(freq/f_lo) / log(f_hi/f_lo)  (invariant to log base)
    """
    log_ratio = np.log(f_hi / f_lo)
    return np.log(freqs / f_lo) / log_ratio


def hz_weighted_expected(n_bins, f_lo, f_hi):
    """Compute Hz-weighted expected fraction for each equal-width u-bin.

    Under a Hz-uniform null, peak density in u-space is proportional to
    d(Hz)/du = f_lo * ratio^u * ln(ratio). The expected fraction in
    bin [u_left, u_right] is:

        (ratio^u_right - ratio^u_left) / (ratio - 1)
    """
    ratio = f_hi / f_lo
    edges = np.linspace(0, 1, n_bins + 1)
    fracs = np.diff(ratio ** edges) / (ratio - 1)
    return fracs


# =========================================================================
# ENRICHMENT AT A SINGLE (f0, ratio) POINT
# =========================================================================

def compute_enrichment_profile(freqs, f0, ratio, n_bins=12, min_peaks=10):
    """Compute Hz-corrected enrichment profile for all bands.

    Returns dict of {band_idx: enrichment_array} where enrichment is in
    percentage points (0 = expected, +50 = 50% more than expected, etc.)
    """
    boundaries = compute_boundaries(f0, ratio)
    assignments = assign_bands(freqs, boundaries)
    n_bands = len(boundaries) - 1
    profiles = {}

    for b in range(n_bands):
        band_freqs = freqs[assignments == b]
        n = len(band_freqs)
        if n < min_peaks:
            profiles[b] = np.full(n_bins, np.nan)
            continue

        f_lo, f_hi = boundaries[b], boundaries[b + 1]
        u = compute_u(band_freqs, f_lo, f_hi)
        u = np.clip(u, 0, 1 - 1e-12)  # ensure u < 1

        # Bin counts (equal-width bins in u-space)
        counts, _ = np.histogram(u, bins=n_bins, range=(0, 1))

        # Hz-weighted expected counts
        expected_fracs = hz_weighted_expected(n_bins, f_lo, f_hi)
        expected = expected_fracs * n

        # Enrichment as percentage deviation from expected
        with np.errstate(divide='ignore', invalid='ignore'):
            enrichment = np.where(
                expected > 0,
                (counts / expected - 1) * 100,
                np.nan
            )
        profiles[b] = enrichment

    coverage = (assignments >= 0).sum() / len(freqs) if len(freqs) > 0 else 0
    return profiles, coverage


# =========================================================================
# METRIC FUNCTIONS
# =========================================================================

def metric_boundary_sharpness(profiles, n_bins):
    """Mean absolute enrichment discontinuity at band boundaries.

    At each boundary, compares the last bin of band_n with the first bin
    of band_n+1. Larger discontinuities mean cleaner band separation.
    """
    band_indices = sorted(profiles.keys())
    discontinuities = []
    for i in range(len(band_indices) - 1):
        b_lo = band_indices[i]
        b_hi = band_indices[i + 1]
        p_lo = profiles[b_lo]
        p_hi = profiles[b_hi]
        if np.any(np.isnan(p_lo)) or np.any(np.isnan(p_hi)):
            continue
        # Last bin of lower band vs first bin of upper band
        disc = abs(p_lo[-1] - p_hi[0])
        discontinuities.append(disc)
    return np.mean(discontinuities) if discontinuities else np.nan


def metric_profile_simplicity(profiles, n_bins):
    """Mean R² of linear fit to each band's enrichment profile.

    A monotonic ramp or flat profile yields high R². Complex shapes
    (multi-modal, non-monotonic) yield low R².
    """
    r2_values = []
    x = np.arange(n_bins)
    for b, profile in profiles.items():
        if np.any(np.isnan(profile)):
            continue
        # Fit linear model
        slope, intercept, r_value, _, _ = stats.linregress(x, profile)
        r2_values.append(r_value ** 2)
    return np.mean(r2_values) if r2_values else np.nan


def metric_profile_simplicity_best(profiles, n_bins):
    """Mean best-of-{linear, quadratic} R² per band.

    Allows both ramp (linear) and mountain/valley (quadratic) shapes.
    """
    r2_values = []
    x = np.arange(n_bins, dtype=float)
    for b, profile in profiles.items():
        if np.any(np.isnan(profile)):
            continue
        # Linear
        _, _, r_lin, _, _ = stats.linregress(x, profile)
        r2_lin = r_lin ** 2
        # Quadratic
        coeffs = np.polyfit(x, profile, 2)
        pred = np.polyval(coeffs, x)
        ss_res = np.sum((profile - pred) ** 2)
        ss_tot = np.sum((profile - np.mean(profile)) ** 2)
        r2_quad = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_values.append(max(r2_lin, r2_quad))
    return np.mean(r2_values) if r2_values else np.nan


def metric_band_independence(profiles, n_bins):
    """Mean absolute Pearson r between adjacent bands.

    Low values mean adjacent bands have distinct enrichment patterns.
    """
    band_indices = sorted(profiles.keys())
    correlations = []
    for i in range(len(band_indices) - 1):
        b_lo = band_indices[i]
        b_hi = band_indices[i + 1]
        p_lo = profiles[b_lo]
        p_hi = profiles[b_hi]
        if np.any(np.isnan(p_lo)) or np.any(np.isnan(p_hi)):
            continue
        r, _ = stats.pearsonr(p_lo, p_hi)
        correlations.append(abs(r))
    return np.mean(correlations) if correlations else np.nan


def metric_profile_contrast(profiles, n_bins):
    """Mean range (max - min) of enrichment across all bands.

    Higher contrast means the coordinate system resolves more structure.
    """
    contrasts = []
    for b, profile in profiles.items():
        if np.any(np.isnan(profile)):
            continue
        contrasts.append(np.max(profile) - np.min(profile))
    return np.mean(contrasts) if contrasts else np.nan


# =========================================================================
# CROSS-DATASET CONSISTENCY
# =========================================================================

def metric_cross_dataset_consistency(all_profiles, n_bins):
    """Mean pairwise correlation of enrichment profiles across datasets.

    For each band, correlate the enrichment profile from every pair of
    datasets. High values mean the pattern replicates.
    """
    # Collect profiles per band across datasets
    n_bands = 5
    band_profiles = {b: [] for b in range(n_bands)}
    for ds_name, profiles in all_profiles.items():
        for b, profile in profiles.items():
            if not np.any(np.isnan(profile)):
                band_profiles[b].append(profile)

    all_r = []
    for b in range(n_bands):
        profs = band_profiles[b]
        if len(profs) < 2:
            continue
        for i in range(len(profs)):
            for j in range(i + 1, len(profs)):
                r, _ = stats.pearsonr(profs[i], profs[j])
                all_r.append(r)
    return np.mean(all_r) if all_r else np.nan


# =========================================================================
# MAIN SWEEP
# =========================================================================

def run_sweep(freq_arrays, f0_range, ratio_range, f0_steps, ratio_steps,
              n_bins=12, verbose=True):
    """Sweep (f0, ratio) space and compute all metrics.

    freq_arrays: dict of {dataset_name: np.array of frequencies}
    Returns DataFrame with columns: f0, ratio, and all metrics.
    """
    f0_values = np.linspace(f0_range[0], f0_range[1], f0_steps)
    ratio_values = np.linspace(ratio_range[0], ratio_range[1], ratio_steps)

    # Pool all frequencies for the main metrics
    all_freqs = np.concatenate(list(freq_arrays.values()))
    n_total = len(all_freqs)

    if verbose:
        print(f"\nSweep: f0 [{f0_range[0]:.1f}, {f0_range[1]:.1f}] Hz "
              f"({f0_steps} steps) × ratio [{ratio_range[0]:.2f}, "
              f"{ratio_range[1]:.2f}] ({ratio_steps} steps)")
        print(f"  {n_total:,} pooled peaks, {n_bins} u-bins per band")
        print(f"  {f0_steps * ratio_steps} grid points\n")

    rows = []
    t0 = time.time()
    for i, f0 in enumerate(f0_values):
        for j, ratio in enumerate(ratio_values):
            # Pooled enrichment (all datasets combined)
            profiles_pooled, coverage = compute_enrichment_profile(
                all_freqs, f0, ratio, n_bins=n_bins
            )

            # Per-dataset enrichment for consistency metric
            per_ds_profiles = {}
            for ds_name, freqs in freq_arrays.items():
                profs, _ = compute_enrichment_profile(
                    freqs, f0, ratio, n_bins=n_bins
                )
                per_ds_profiles[ds_name] = profs

            # Compute metrics
            sharpness = metric_boundary_sharpness(profiles_pooled, n_bins)
            simplicity = metric_profile_simplicity(profiles_pooled, n_bins)
            simplicity_best = metric_profile_simplicity_best(profiles_pooled, n_bins)
            independence = metric_band_independence(profiles_pooled, n_bins)
            contrast = metric_profile_contrast(profiles_pooled, n_bins)
            consistency = metric_cross_dataset_consistency(per_ds_profiles, n_bins)

            # Composite score: higher is better
            # Normalize each metric to [0, 1] later; for now store raw
            rows.append({
                'f0': round(f0, 4),
                'ratio': round(ratio, 4),
                'sharpness': round(sharpness, 2) if not np.isnan(sharpness) else np.nan,
                'simplicity_linear': round(simplicity, 4) if not np.isnan(simplicity) else np.nan,
                'simplicity_best': round(simplicity_best, 4) if not np.isnan(simplicity_best) else np.nan,
                'independence': round(1 - independence, 4) if not np.isnan(independence) else np.nan,  # invert: higher = more independent
                'contrast': round(contrast, 2) if not np.isnan(contrast) else np.nan,
                'consistency': round(consistency, 4) if not np.isnan(consistency) else np.nan,
                'coverage': round(coverage, 4),
            })

        if verbose and (i + 1) % 6 == 0:
            elapsed = time.time() - t0
            pct = (i + 1) / f0_steps
            eta = elapsed / pct * (1 - pct)
            print(f"  f0={f0:.2f} Hz  [{pct:.0%}]  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    df = pd.DataFrame(rows)

    # Composite score: z-score each metric, weighted average
    # Simplicity and consistency are the most meaningful metrics.
    # Sharpness and contrast are informative but scale with band width,
    # so we downweight them. Coverage is informational only.
    metrics_weights = {
        'simplicity_best': 3.0,  # primary: clean patterns within bands
        'consistency': 3.0,      # primary: replicates across datasets
        'independence': 2.0,     # secondary: bands are distinct units
        'sharpness': 1.0,        # tertiary: scales with band width
        'contrast': 1.0,         # tertiary: scales with band width
        'coverage': 0.0,         # informational only, not in composite
    }
    for m in metrics_weights:
        col = df[m].values.astype(float)
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            mu, sigma = np.nanmean(col), np.nanstd(col)
            df[f'{m}_z'] = np.where(valid, (col - mu) / sigma if sigma > 0 else 0, np.nan)
        else:
            df[f'{m}_z'] = np.nan

    total_weight = sum(w for w in metrics_weights.values() if w > 0)
    df['composite'] = sum(
        df[f'{m}_z'] * w / total_weight
        for m, w in metrics_weights.items() if w > 0
    )

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Done in {elapsed:.1f}s")
    return df


# =========================================================================
# REPORTING
# =========================================================================

def report_results(df, n_bins):
    """Print summary and save results."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save full grid
    csv_path = os.path.join(OUT_DIR, 'sweep_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nFull grid saved to {csv_path}")

    # Find global optimum
    best_idx = df['composite'].idxmax()
    best = df.loc[best_idx]
    print(f"\n{'=' * 70}")
    print(f"  OPTIMAL COORDINATE SYSTEM (composite score)")
    print(f"{'=' * 70}")
    print(f"  f0 = {best['f0']:.2f} Hz,  ratio = {best['ratio']:.4f}")
    print(f"  Boundaries: ", end='')
    boundaries = compute_boundaries(best['f0'], best['ratio'])
    print(' | '.join(f"{b:.2f}" for b in boundaries) + ' Hz')
    print(f"\n  Metrics:")
    print(f"    Boundary sharpness:   {best['sharpness']:.1f} pp")
    print(f"    Profile simplicity:   {best['simplicity_best']:.3f} R²")
    print(f"    Band independence:    {best['independence']:.3f}")
    print(f"    Enrichment contrast:  {best['contrast']:.1f} pp")
    print(f"    Cross-dataset r:      {best['consistency']:.3f}")
    print(f"    Peak coverage:        {best['coverage']:.1%}")
    print(f"    Composite z-score:    {best['composite']:.3f}")

    # Evaluate named systems
    print(f"\n{'=' * 70}")
    print(f"  NAMED COORDINATE SYSTEM COMPARISON")
    print(f"{'=' * 70}")

    named_rows = []
    for sys_name, (f0, ratio) in NAMED_SYSTEMS.items():
        # Find nearest grid point
        dist = np.sqrt(((df['f0'] - f0) / 1.0) ** 2 +
                       ((df['ratio'] - ratio) / 0.1) ** 2)
        idx = dist.idxmin()
        row = df.loc[idx]
        named_rows.append({
            'system': sys_name,
            'f0': row['f0'],
            'ratio': row['ratio'],
            'sharpness': row['sharpness'],
            'simplicity': row['simplicity_best'],
            'independence': row['independence'],
            'contrast': row['contrast'],
            'consistency': row['consistency'],
            'coverage': row['coverage'],
            'composite': row['composite'],
        })
        boundaries = compute_boundaries(row['f0'], row['ratio'])
        bnd_str = ' | '.join(f"{b:.1f}" for b in boundaries)
        print(f"\n  {sys_name}:")
        print(f"    f0={row['f0']:.2f}, r={row['ratio']:.4f}  →  {bnd_str} Hz")
        print(f"    sharp={row['sharpness']:.0f}  simple={row['simplicity_best']:.3f}  "
              f"indep={row['independence']:.3f}  contrast={row['contrast']:.0f}  "
              f"consist={row['consistency']:.3f}  cover={row['coverage']:.1%}  "
              f"composite={row['composite']:.3f}")

    named_df = pd.DataFrame(named_rows)
    named_path = os.path.join(OUT_DIR, 'named_systems.csv')
    named_df.to_csv(named_path, index=False)
    print(f"\n  Named systems saved to {named_path}")

    # Top 10 grid points
    print(f"\n{'=' * 70}")
    print(f"  TOP 10 GRID POINTS (by composite)")
    print(f"{'=' * 70}")
    top10 = df.nlargest(10, 'composite')
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        boundaries = compute_boundaries(row['f0'], row['ratio'])
        bnd_str = ' | '.join(f"{b:.1f}" for b in boundaries)
        print(f"  {rank:2d}. f0={row['f0']:.2f} r={row['ratio']:.4f}  "
              f"composite={row['composite']:.3f}  [{bnd_str}]")

    return named_df


def generate_plots(df, n_bins):
    """Generate heatmap visualizations of the sweep results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    f0_vals = np.sort(df['f0'].unique())
    ratio_vals = np.sort(df['ratio'].unique())

    metrics_to_plot = [
        ('composite', 'Composite Score (z)', 'RdYlGn'),
        ('sharpness', 'Boundary Sharpness (pp)', 'YlOrRd'),
        ('simplicity_best', 'Profile Simplicity (R²)', 'YlGnBu'),
        ('independence', 'Band Independence', 'PuBuGn'),
        ('contrast', 'Enrichment Contrast (pp)', 'YlOrRd'),
        ('consistency', 'Cross-Dataset Consistency (r)', 'YlGnBu'),
        ('coverage', 'Peak Coverage', 'Greens'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for ax_idx, (metric, title, cmap) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        # Pivot to 2D grid
        pivot = df.pivot(index='ratio', columns='f0', values=metric)
        im = ax.imshow(
            pivot.values, aspect='auto', origin='lower', cmap=cmap,
            extent=[f0_vals[0], f0_vals[-1], ratio_vals[0], ratio_vals[-1]],
        )
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel('f₀ (Hz)')
        ax.set_ylabel('Ratio')
        ax.set_title(title)

        # Mark named systems
        for sys_name, (f0, ratio) in NAMED_SYSTEMS.items():
            if (f0_vals[0] <= f0 <= f0_vals[-1] and
                    ratio_vals[0] <= ratio <= ratio_vals[-1]):
                marker = {'phi_lattice': '*', 'octave': 's',
                           'clinical_approx': 'D', 'third_octave': '^',
                           'sqrt2': 'v'}.get(sys_name, 'o')
                ax.plot(f0, ratio, marker, color='white', markersize=10,
                        markeredgecolor='black', markeredgewidth=1.5)

    # Legend in last subplot
    axes[-1].axis('off')
    for sys_name, (f0, ratio) in NAMED_SYSTEMS.items():
        marker = {'phi_lattice': '*', 'octave': 's',
                   'clinical_approx': 'D', 'third_octave': '^',
                   'sqrt2': 'v'}.get(sys_name, 'o')
        axes[-1].plot([], [], marker, color='gray', markersize=10,
                      markeredgecolor='black', label=f'{sys_name} ({f0}, {ratio})')
    axes[-1].legend(loc='center', fontsize=11, frameon=False)
    axes[-1].set_title('Named Systems')

    plt.suptitle(f'Boundary Sweep: Coordinate System Evaluation\n'
                 f'({len(df)} grid points, {n_bins} u-bins per band)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, 'sweep_heatmaps.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Heatmaps saved to {fig_path}")
    plt.close()

    # --- Per-band enrichment profiles at named systems ---
    all_freqs = None  # Will be loaded fresh for profile plots
    return fig_path


def plot_named_profiles(freq_arrays, n_bins=12):
    """Plot enrichment profiles at each named coordinate system."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    all_freqs = np.concatenate(list(freq_arrays.values()))
    band_labels = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

    fig, axes = plt.subplots(len(NAMED_SYSTEMS), 5, figsize=(25, 4 * len(NAMED_SYSTEMS)))

    for row_idx, (sys_name, (f0, ratio)) in enumerate(NAMED_SYSTEMS.items()):
        profiles, coverage = compute_enrichment_profile(
            all_freqs, f0, ratio, n_bins=n_bins
        )
        boundaries = compute_boundaries(f0, ratio)

        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            profile = profiles.get(col_idx, np.full(n_bins, np.nan))
            x = np.linspace(0, 1, n_bins, endpoint=False) + 0.5 / n_bins
            hz_lo, hz_hi = boundaries[col_idx], boundaries[col_idx + 1]

            if not np.any(np.isnan(profile)):
                colors = ['#d32f2f' if v < -10 else '#388e3c' if v > 10 else '#757575'
                          for v in profile]
                ax.bar(x, profile, width=0.8 / n_bins, color=colors, edgecolor='black', linewidth=0.5)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_xlabel(f'u  [{hz_lo:.1f} – {hz_hi:.1f} Hz]')
            if col_idx == 0:
                ax.set_ylabel(f'{sys_name}\n(f₀={f0}, r={ratio:.3f})\nEnrichment %')
            if row_idx == 0:
                ax.set_title(band_labels[col_idx])

    plt.suptitle(f'Enrichment Profiles by Coordinate System ({n_bins} bins)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'named_system_profiles.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Profile comparison saved to {fig_path}")
    plt.close()


# =========================================================================
# BOUNDARY SLIDE ANALYSIS (fine-grained, per-boundary)
# =========================================================================

def boundary_slide(freq_arrays, n_bins=12, slide_pct=0.25, slide_steps=51):
    """For each inter-band boundary, slide it ±slide_pct and measure impact.

    Uses the phi-lattice as the baseline, then slides each boundary
    independently while holding others fixed.

    Returns DataFrame with per-boundary results.
    """
    from phi_frequency_model import PHI, F0

    all_freqs = np.concatenate(list(freq_arrays.values()))
    base_boundaries = compute_boundaries(F0, PHI)  # [4.70, 7.60, 12.30, 19.90, 32.19, 52.09]
    boundary_names = ['theta/alpha', 'alpha/beta_low', 'beta_low/beta_high', 'beta_high/gamma']
    band_labels = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

    rows = []
    for b_idx in range(1, len(base_boundaries) - 1):  # skip outermost boundaries
        base_hz = base_boundaries[b_idx]
        lo = base_hz * (1 - slide_pct)
        hi = base_hz * (1 + slide_pct)
        slide_values = np.linspace(lo, hi, slide_steps)

        for hz in slide_values:
            # Create modified boundary set
            modified = base_boundaries.copy()
            modified[b_idx] = hz

            # Ensure monotonicity
            if b_idx > 0 and hz <= modified[b_idx - 1]:
                continue
            if b_idx < len(modified) - 1 and hz >= modified[b_idx + 1]:
                continue

            # Assign peaks and compute enrichment per band
            assignments = assign_bands(all_freqs, modified)
            band_profiles = {}
            for band_i in range(len(modified) - 1):
                band_freqs = all_freqs[assignments == band_i]
                n = len(band_freqs)
                if n < 10:
                    band_profiles[band_i] = np.full(n_bins, np.nan)
                    continue
                f_lo_b, f_hi_b = modified[band_i], modified[band_i + 1]
                u = compute_u(band_freqs, f_lo_b, f_hi_b)
                u = np.clip(u, 0, 1 - 1e-12)
                counts, _ = np.histogram(u, bins=n_bins, range=(0, 1))
                expected_fracs = hz_weighted_expected(n_bins, f_lo_b, f_hi_b)
                expected = expected_fracs * n
                with np.errstate(divide='ignore', invalid='ignore'):
                    enrichment = np.where(expected > 0, (counts / expected - 1) * 100, np.nan)
                band_profiles[band_i] = enrichment

            # Metrics for the two bands flanking this boundary
            lower_band = b_idx - 1
            upper_band = b_idx

            p_lo = band_profiles.get(lower_band, np.full(n_bins, np.nan))
            p_hi = band_profiles.get(upper_band, np.full(n_bins, np.nan))

            # Discontinuity at this boundary
            disc = np.nan
            if not np.any(np.isnan(p_lo)) and not np.any(np.isnan(p_hi)):
                disc = abs(p_lo[-1] - p_hi[0])

            # Simplicity of both flanking bands
            simp_lo = simp_hi = np.nan
            x = np.arange(n_bins, dtype=float)
            for p, label in [(p_lo, 'lo'), (p_hi, 'hi')]:
                if np.any(np.isnan(p)):
                    continue
                # Best of linear/quadratic
                _, _, r_lin, _, _ = stats.linregress(x, p)
                r2_lin = r_lin ** 2
                coeffs = np.polyfit(x, p, 2)
                pred = np.polyval(coeffs, x)
                ss_res = np.sum((p - pred) ** 2)
                ss_tot = np.sum((p - np.mean(p)) ** 2)
                r2_quad = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                r2_best = max(r2_lin, r2_quad)
                if label == 'lo':
                    simp_lo = r2_best
                else:
                    simp_hi = r2_best

            # Contrast of both flanking bands
            contrast_lo = (np.nanmax(p_lo) - np.nanmin(p_lo)) if not np.all(np.isnan(p_lo)) else np.nan
            contrast_hi = (np.nanmax(p_hi) - np.nanmin(p_hi)) if not np.all(np.isnan(p_hi)) else np.nan

            rows.append({
                'boundary': boundary_names[b_idx - 1],
                'base_hz': round(base_hz, 2),
                'slide_hz': round(hz, 3),
                'slide_pct': round((hz - base_hz) / base_hz * 100, 2),
                'discontinuity': round(disc, 2) if not np.isnan(disc) else np.nan,
                'simplicity_lower': round(simp_lo, 4) if not np.isnan(simp_lo) else np.nan,
                'simplicity_upper': round(simp_hi, 4) if not np.isnan(simp_hi) else np.nan,
                'contrast_lower': round(contrast_lo, 2) if not np.isnan(contrast_lo) else np.nan,
                'contrast_upper': round(contrast_hi, 2) if not np.isnan(contrast_hi) else np.nan,
                'n_lower': int((assignments == lower_band).sum()),
                'n_upper': int((assignments == upper_band).sum()),
            })

    df = pd.DataFrame(rows)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'boundary_slide.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Boundary slide saved to {csv_path}")

    # Report per boundary
    print(f"\n{'=' * 70}")
    print(f"  BOUNDARY SLIDE ANALYSIS (±{slide_pct:.0%} around φ-lattice)")
    print(f"{'=' * 70}")
    for bnd_name in boundary_names:
        bnd_df = df[df['boundary'] == bnd_name]
        if bnd_df.empty:
            continue
        base = bnd_df.iloc[len(bnd_df) // 2]

        # Find optimal for each metric
        best_disc_idx = bnd_df['discontinuity'].idxmax()
        best_simp_idx = (bnd_df['simplicity_lower'].fillna(0) + bnd_df['simplicity_upper'].fillna(0)).idxmax()

        best_disc = bnd_df.loc[best_disc_idx]
        best_simp = bnd_df.loc[best_simp_idx]

        print(f"\n  {bnd_name} (base: {base['base_hz']:.2f} Hz)")
        print(f"    At base:        disc={base['discontinuity']:.0f} pp  "
              f"simp=({base['simplicity_lower']:.3f}, {base['simplicity_upper']:.3f})  "
              f"contrast=({base['contrast_lower']:.0f}, {base['contrast_upper']:.0f})")
        print(f"    Max disc at:    {best_disc['slide_hz']:.2f} Hz ({best_disc['slide_pct']:+.1f}%)  "
              f"disc={best_disc['discontinuity']:.0f} pp")
        print(f"    Max simplicity: {best_simp['slide_hz']:.2f} Hz ({best_simp['slide_pct']:+.1f}%)  "
              f"simp=({best_simp['simplicity_lower']:.3f}, {best_simp['simplicity_upper']:.3f})")

    return df


def plot_boundary_slide(slide_df):
    """Plot boundary slide results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    boundary_names = slide_df['boundary'].unique()
    fig, axes = plt.subplots(len(boundary_names), 3, figsize=(18, 4 * len(boundary_names)))
    if len(boundary_names) == 1:
        axes = axes[np.newaxis, :]

    for row_idx, bnd_name in enumerate(boundary_names):
        bnd_df = slide_df[slide_df['boundary'] == bnd_name]
        base_hz = bnd_df['base_hz'].iloc[0]
        x = bnd_df['slide_hz'].values

        # Discontinuity
        ax = axes[row_idx, 0]
        ax.plot(x, bnd_df['discontinuity'], 'b-', linewidth=2)
        ax.axvline(base_hz, color='red', linestyle='--', alpha=0.7, label=f'φ-lattice ({base_hz:.2f})')
        ax.set_ylabel('Boundary Discontinuity (pp)')
        ax.set_title(f'{bnd_name}')
        ax.legend(fontsize=9)

        # Simplicity
        ax = axes[row_idx, 1]
        ax.plot(x, bnd_df['simplicity_lower'], 'g-', linewidth=2, label='lower band')
        ax.plot(x, bnd_df['simplicity_upper'], 'b-', linewidth=2, label='upper band')
        ax.axvline(base_hz, color='red', linestyle='--', alpha=0.7)
        ax.set_ylabel('Profile Simplicity (R²)')
        ax.legend(fontsize=9)

        # Contrast
        ax = axes[row_idx, 2]
        ax.plot(x, bnd_df['contrast_lower'], 'g-', linewidth=2, label='lower band')
        ax.plot(x, bnd_df['contrast_upper'], 'b-', linewidth=2, label='upper band')
        ax.axvline(base_hz, color='red', linestyle='--', alpha=0.7)
        ax.set_ylabel('Enrichment Contrast (pp)')
        ax.legend(fontsize=9)

        for c in range(3):
            axes[row_idx, c].set_xlabel('Boundary frequency (Hz)')

    plt.suptitle('Boundary Slide: Per-Boundary Optimization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'boundary_slide.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Boundary slide plot saved to {fig_path}")
    plt.close()


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Boundary sweep: evaluate coordinate systems for EEG bands')
    parser.add_argument('--f0-range', type=float, nargs=2, default=list(DEFAULT_F0_RANGE),
                        help='f0 sweep range (Hz)')
    parser.add_argument('--ratio-range', type=float, nargs=2, default=list(DEFAULT_RATIO_RANGE),
                        help='ratio sweep range')
    parser.add_argument('--f0-steps', type=int, default=DEFAULT_F0_STEPS)
    parser.add_argument('--ratio-steps', type=int, default=DEFAULT_RATIO_STEPS)
    parser.add_argument('--n-bins', type=int, default=DEFAULT_N_BINS,
                        help='number of equal-width u-bins per band')
    parser.add_argument('--plot', action='store_true', help='generate plots')
    parser.add_argument('--slide', action='store_true',
                        help='run per-boundary slide analysis')
    parser.add_argument('--slide-only', action='store_true',
                        help='run only the boundary slide (skip full sweep)')
    parser.add_argument('--quick', action='store_true',
                        help='quick sweep (12×12 grid)')
    args = parser.parse_args()

    if args.quick:
        args.f0_steps = 12
        args.ratio_steps = 12

    print("Loading peak data...")
    datasets = load_all_peaks()
    freq_arrays = pool_all_freqs(datasets)
    total = sum(len(f) for f in freq_arrays.values())
    print(f"\nTotal: {len(freq_arrays)} datasets, {total:,} peaks")

    if not args.slide_only:
        # Full 2D sweep
        df = run_sweep(
            freq_arrays,
            f0_range=tuple(args.f0_range),
            ratio_range=tuple(args.ratio_range),
            f0_steps=args.f0_steps,
            ratio_steps=args.ratio_steps,
            n_bins=args.n_bins,
        )
        named_df = report_results(df, args.n_bins)

        if args.plot:
            generate_plots(df, args.n_bins)
            plot_named_profiles(freq_arrays, n_bins=args.n_bins)

    if args.slide or args.slide_only:
        slide_df = boundary_slide(freq_arrays, n_bins=args.n_bins)
        if args.plot:
            plot_boundary_slide(slide_df)


if __name__ == '__main__':
    main()
