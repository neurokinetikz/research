#!/usr/bin/env python3
"""
Within-Band Coordinate System Analysis
========================================

Given that phi-lattice boundaries are near-optimal (boundary_sweep.py),
this script asks: does the *internal* coordinate structure within each
band matter? Specifically:

1. SCALING: Is the enrichment pattern better described in linear Hz,
   log Hz, or some other warping within each band?

2. LANDMARK CAPTURE: Do the 12 phi-positions capture more of the
   enrichment signal than 12 equally-spaced or random positions?

3. FEATURE ALIGNMENT: Do enrichment extrema fall closer to phi-
   positions than expected by chance? (Permutation test)

4. PERIODICITY: Is there periodic structure in the high-resolution
   enrichment profile? At what spacing?

5. NOBLE vs RATIONAL: Are enrichment extrema closer to noble numbers
   (maximally irrational) or to simple rationals (1/2, 1/3, 2/3...)?

Usage:
    python scripts/within_band_coordinates.py
    python scripts/within_band_coordinates.py --plot
    python scripts/within_band_coordinates.py --n-fine 200 --n-perms 5000

Outputs to: outputs/within_band_coordinates/
"""

import os
import sys
import argparse
import glob
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'within_band_coordinates')

PHI_INV = 1.0 / PHI
MIN_POWER_PCT = 50

EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp', 'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3', 'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
}

BAND_HZ = {
    'theta': (F0 / PHI, F0),
    'alpha': (F0, F0 * PHI),
    'beta_low': (F0 * PHI, F0 * PHI ** 2),
    'beta_high': (F0 * PHI ** 2, F0 * PHI ** 3),
    'gamma': (F0 * PHI ** 3, F0 * PHI ** 4),
}
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

# Phi-lattice degree-6 positions (12 positions in [0, 1])
PHI_POSITIONS = np.array([
    0.0,                    # boundary
    PHI_INV ** 6,           # noble_6
    PHI_INV ** 5,           # noble_5
    PHI_INV ** 4,           # noble_4
    PHI_INV ** 3,           # noble_3
    PHI_INV ** 2,           # inv_noble_1
    0.5,                    # attractor
    PHI_INV,                # noble_1
    1 - PHI_INV ** 3,       # inv_noble_3
    1 - PHI_INV ** 4,       # inv_noble_4
    1 - PHI_INV ** 5,       # inv_noble_5
    1 - PHI_INV ** 6,       # inv_noble_6
])
PHI_POS_NAMES = [
    'boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3', 'inv_noble_1',
    'attractor', 'noble_1', 'inv_noble_3', 'inv_noble_4', 'inv_noble_5',
    'inv_noble_6',
]

# Simple rational positions (12 positions for fair comparison)
RATIONAL_POSITIONS = np.array([
    0.0, 1/12, 1/6, 1/4, 1/3, 5/12, 1/2, 7/12, 2/3, 3/4, 5/6, 11/12
])

# Noble numbers: the most irrational numbers (related to golden ratio)
# These are φ-based continued fraction convergents
NOBLE_NUMBERS = np.array(sorted([
    PHI_INV,            # 0.618... -- the "most irrational" number
    PHI_INV ** 2,       # 0.382...
    PHI_INV ** 3,       # 0.236...
    PHI_INV ** 4,       # 0.146...
    1 - PHI_INV,        # 0.382... (same as φ⁻²)
    1 - PHI_INV ** 2,   # 0.618... (same as φ⁻¹)
    1 - PHI_INV ** 3,   # 0.764...
    1 - PHI_INV ** 4,   # 0.854...
    0.5,                # attractor
]))

# Simple rationals for comparison
SIMPLE_RATIONALS = np.array(sorted(set([
    0.0, 1.0,
    1/2,
    1/3, 2/3,
    1/4, 3/4,
    1/5, 2/5, 3/5, 4/5,
    1/6, 5/6,
    1/8, 3/8, 5/8, 7/8,
])))


# =========================================================================
# DATA LOADING
# =========================================================================

def load_all_peaks():
    """Load and power-filter peaks from all EC datasets."""
    all_dfs = []
    for name, subdir in EC_DATASETS.items():
        path = os.path.join(PEAK_BASE, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        dfs = []
        for f in files:
            dfs.append(pd.read_csv(f, usecols=cols))
        peaks = pd.concat(dfs, ignore_index=True)
        if has_power and MIN_POWER_PCT > 0:
            filtered = []
            for octave in peaks['phi_octave'].unique():
                bp = peaks[peaks.phi_octave == octave]
                thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                filtered.append(bp[bp['power'] >= thresh])
            peaks = pd.concat(filtered, ignore_index=True)
        all_dfs.append(peaks)
        print(f"  {name}: {len(files)} subjects, {len(peaks):,} peaks")
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined['freq'].values


def get_band_freqs(all_freqs):
    """Split frequencies into bands using phi-lattice boundaries."""
    band_freqs = {}
    for band, (f_lo, f_hi) in BAND_HZ.items():
        mask = (all_freqs >= f_lo) & (all_freqs < f_hi)
        band_freqs[band] = all_freqs[mask]
    return band_freqs


# =========================================================================
# SCALING FUNCTIONS
# =========================================================================

def freq_to_u_log(freqs, f_lo, f_hi):
    """Log-frequency (standard phi-lattice coordinate)."""
    return np.log(freqs / f_lo) / np.log(f_hi / f_lo)


def freq_to_u_linear(freqs, f_lo, f_hi):
    """Linear-frequency coordinate."""
    return (freqs - f_lo) / (f_hi - f_lo)


def freq_to_u_erb(freqs, f_lo, f_hi):
    """ERB (Equivalent Rectangular Bandwidth) scale.
    ERB_N(f) = 21.4 * log10(0.00437 * f + 1)
    """
    def erb(f):
        return 21.4 * np.log10(0.00437 * f + 1)
    return (erb(freqs) - erb(f_lo)) / (erb(f_hi) - erb(f_lo))


def freq_to_u_mel(freqs, f_lo, f_hi):
    """Mel scale: m = 2595 * log10(1 + f/700)."""
    def mel(f):
        return 2595 * np.log10(1 + f / 700)
    return (mel(freqs) - mel(f_lo)) / (mel(f_hi) - mel(f_lo))


def freq_to_u_power(freqs, f_lo, f_hi, exponent=0.5):
    """Power-law scaling: u = (f^p - f_lo^p) / (f_hi^p - f_lo^p)."""
    return (freqs ** exponent - f_lo ** exponent) / (f_hi ** exponent - f_lo ** exponent)


SCALINGS = {
    'log': freq_to_u_log,
    'linear': freq_to_u_linear,
    'erb': freq_to_u_erb,
    'mel': freq_to_u_mel,
    'sqrt': lambda f, lo, hi: freq_to_u_power(f, lo, hi, 0.5),
}


# =========================================================================
# HIGH-RESOLUTION ENRICHMENT
# =========================================================================

def compute_hires_enrichment(freqs, f_lo, f_hi, scaling_fn, n_bins=200):
    """Compute high-resolution enrichment profile under a given scaling.

    The null hypothesis is always uniform in the *scaled* coordinate.
    Returns (bin_centers, enrichment_pct, counts, expected).
    """
    u = scaling_fn(freqs, f_lo, f_hi)
    u = np.clip(u, 0, 1 - 1e-12)
    n = len(u)
    counts, edges = np.histogram(u, bins=n_bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    expected = n / n_bins  # uniform null in scaled space
    with np.errstate(divide='ignore', invalid='ignore'):
        enrichment = (counts / expected - 1) * 100
    return centers, enrichment, counts, expected


def compute_hz_corrected_enrichment(freqs, f_lo, f_hi, scaling_fn, n_bins=200):
    """Enrichment with Hz-weighted null (correct for any scaling).

    The null assumes peaks are uniform in Hz. Expected count in each
    scaled bin depends on how much Hz range that bin covers.
    """
    u = scaling_fn(freqs, f_lo, f_hi)
    u = np.clip(u, 0, 1 - 1e-12)
    n = len(u)
    counts, edges = np.histogram(u, bins=n_bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2

    # Expected counts: how much Hz range does each bin cover?
    # Invert scaling at bin edges to get Hz values
    # For arbitrary scalings, use numerical inversion
    hz_edges = np.linspace(f_lo, f_hi, 10001)
    u_fine = scaling_fn(hz_edges, f_lo, f_hi)
    hz_fracs = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (u_fine >= edges[i]) & (u_fine < edges[i + 1])
        if mask.sum() >= 2:
            hz_range = hz_edges[mask]
            hz_fracs[i] = hz_range[-1] - hz_range[0]
        else:
            hz_fracs[i] = (f_hi - f_lo) / n_bins
    hz_fracs = hz_fracs / hz_fracs.sum()
    expected = hz_fracs * n

    with np.errstate(divide='ignore', invalid='ignore'):
        enrichment = np.where(expected > 0, (counts / expected - 1) * 100, 0)

    return centers, enrichment, counts, expected


# =========================================================================
# TEST 1: SCALING COMPARISON
# =========================================================================

def test_scaling(band_freqs, n_fine=200):
    """For each band and scaling, compute enrichment and measure simplicity.

    Simplicity = R² of best low-order polynomial fit (degree 1 or 2).
    Also: signal strength = max enrichment - min enrichment.
    Also: smoothness = 1 - (high-freq power / total power) in FFT.
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 1: SCALING COMPARISON")
    print(f"  (Hz-corrected enrichment, {n_fine} bins, best-of linear/quadratic R²)")
    print(f"{'=' * 70}")

    rows = []
    x = np.arange(n_fine, dtype=float)

    for band in BAND_ORDER:
        freqs = band_freqs[band]
        f_lo, f_hi = BAND_HZ[band]
        n = len(freqs)
        if n < 100:
            continue

        print(f"\n  {band} ({n:,} peaks, {f_lo:.2f}-{f_hi:.2f} Hz):")
        for scale_name, scale_fn in SCALINGS.items():
            centers, enrichment, counts, expected = compute_hz_corrected_enrichment(
                freqs, f_lo, f_hi, scale_fn, n_bins=n_fine
            )

            # Linear R²
            _, _, r_lin, _, _ = stats.linregress(x, enrichment)
            r2_lin = r_lin ** 2

            # Quadratic R²
            coeffs = np.polyfit(x, enrichment, 2)
            pred = np.polyval(coeffs, x)
            ss_res = np.sum((enrichment - pred) ** 2)
            ss_tot = np.sum((enrichment - np.mean(enrichment)) ** 2)
            r2_quad = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Cubic R²
            coeffs3 = np.polyfit(x, enrichment, 3)
            pred3 = np.polyval(coeffs3, x)
            ss_res3 = np.sum((enrichment - pred3) ** 2)
            r2_cubic = 1 - ss_res3 / ss_tot if ss_tot > 0 else 0

            r2_best = max(r2_lin, r2_quad)

            # Signal contrast
            contrast = np.max(enrichment) - np.min(enrichment)

            # Smoothness: fraction of variance in first 10% of FFT spectrum
            fft_power = np.abs(np.fft.rfft(enrichment - np.mean(enrichment))) ** 2
            total_power = fft_power.sum()
            n_low = max(1, len(fft_power) // 10)
            low_power = fft_power[:n_low].sum()
            smoothness = low_power / total_power if total_power > 0 else 0

            rows.append({
                'band': band, 'scaling': scale_name,
                'r2_linear': round(r2_lin, 4),
                'r2_quadratic': round(r2_quad, 4),
                'r2_cubic': round(r2_cubic, 4),
                'r2_best': round(r2_best, 4),
                'contrast': round(contrast, 1),
                'smoothness': round(smoothness, 4),
                'n_peaks': n,
            })
            print(f"    {scale_name:8s}  R²_lin={r2_lin:.3f}  R²_quad={r2_quad:.3f}  "
                  f"R²_cub={r2_cubic:.3f}  contrast={contrast:.0f}  smooth={smoothness:.3f}")

    df = pd.DataFrame(rows)
    return df


# =========================================================================
# TEST 2: LANDMARK CAPTURE
# =========================================================================

def voronoi_capture(enrichment_profile, positions, n_fine):
    """How well do K discrete positions capture the full enrichment profile?

    Uses Voronoi binning: each fine bin is assigned to its nearest position,
    then the position's value is the mean enrichment in its Voronoi cell.
    The captured profile is the Voronoi-reconstructed version.
    Returns R² of reconstruction vs original.
    """
    n_bins = len(enrichment_profile)
    bin_centers = np.linspace(0, 1, n_bins, endpoint=False) + 0.5 / n_bins

    # Assign each bin to nearest position (circular)
    pos = np.array(positions) % 1.0
    assignments = np.zeros(n_bins, dtype=int)
    for i, u in enumerate(bin_centers):
        dists = np.abs(u - pos)
        dists = np.minimum(dists, 1 - dists)
        assignments[i] = np.argmin(dists)

    # Reconstruct: each bin gets the mean enrichment of its Voronoi cell
    reconstructed = np.zeros(n_bins)
    for k in range(len(pos)):
        mask = assignments == k
        if mask.sum() > 0:
            val = enrichment_profile[mask].mean()
            reconstructed[mask] = val

    # R² of reconstruction
    ss_res = np.sum((enrichment_profile - reconstructed) ** 2)
    ss_tot = np.sum((enrichment_profile - enrichment_profile.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return r2


def test_landmark_capture(band_freqs, n_fine=200, n_random=1000):
    """Compare phi-positions vs equal-spaced vs random for capturing enrichment.

    For each band:
    1. Compute high-res Hz-corrected enrichment (200 bins)
    2. Test how well 12 phi-positions capture it (Voronoi R²)
    3. Test how well 12 equal-spaced positions capture it
    4. Test how well 12 rational positions capture it
    5. Bootstrap: 1000 random sets of 12 positions, get null distribution
    6. Report: percentile rank of phi-positions vs null
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 2: LANDMARK CAPTURE (12 positions, Voronoi R²)")
    print(f"  (phi vs equal vs rational vs {n_random} random sets)")
    print(f"{'=' * 70}")

    rows = []
    for band in BAND_ORDER:
        freqs = band_freqs[band]
        f_lo, f_hi = BAND_HZ[band]
        if len(freqs) < 100:
            continue

        # High-res enrichment in log space (the established null)
        centers, enrichment, _, _ = compute_hz_corrected_enrichment(
            freqs, f_lo, f_hi, freq_to_u_log, n_bins=n_fine
        )

        # Phi-positions capture
        r2_phi = voronoi_capture(enrichment, PHI_POSITIONS, n_fine)

        # Equal-spaced capture
        equal_pos = np.linspace(0, 1, 12, endpoint=False)
        r2_equal = voronoi_capture(enrichment, equal_pos, n_fine)

        # Rational positions capture
        r2_rational = voronoi_capture(enrichment, RATIONAL_POSITIONS, n_fine)

        # Random null distribution
        rng = np.random.default_rng(42)
        r2_random = []
        for _ in range(n_random):
            rand_pos = rng.uniform(0, 1, 12)
            r2_random.append(voronoi_capture(enrichment, rand_pos, n_fine))
        r2_random = np.array(r2_random)

        # Percentile rank of phi
        pct_phi = (r2_random < r2_phi).mean() * 100

        rows.append({
            'band': band,
            'r2_phi': round(r2_phi, 4),
            'r2_equal': round(r2_equal, 4),
            'r2_rational': round(r2_rational, 4),
            'r2_random_mean': round(r2_random.mean(), 4),
            'r2_random_p95': round(np.percentile(r2_random, 95), 4),
            'phi_percentile': round(pct_phi, 1),
            'phi_vs_equal': round(r2_phi - r2_equal, 4),
            'phi_vs_rational': round(r2_phi - r2_rational, 4),
        })
        print(f"\n  {band}:")
        print(f"    Phi positions:     R² = {r2_phi:.4f}")
        print(f"    Equal-spaced:      R² = {r2_equal:.4f}  (Δ = {r2_phi - r2_equal:+.4f})")
        print(f"    Rational (12ths):  R² = {r2_rational:.4f}  (Δ = {r2_phi - r2_rational:+.4f})")
        print(f"    Random (mean):     R² = {r2_random.mean():.4f}  "
              f"[95th pct: {np.percentile(r2_random, 95):.4f}]")
        print(f"    Phi percentile:    {pct_phi:.1f}%")

    return pd.DataFrame(rows)


# =========================================================================
# TEST 3: FEATURE ALIGNMENT
# =========================================================================

def test_feature_alignment(band_freqs, n_fine=200, n_perms=5000):
    """Do enrichment extrema fall at phi-positions more than by chance?

    For each band:
    1. Find peaks and troughs in the smoothed hi-res enrichment profile
    2. Measure mean distance from each extremum to nearest phi-position
    3. Permutation test: shuffle positions and recompute distances
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 3: FEATURE ALIGNMENT")
    print(f"  (Do enrichment extrema fall at phi-positions? {n_perms} permutations)")
    print(f"{'=' * 70}")

    rows = []
    for band in BAND_ORDER:
        freqs = band_freqs[band]
        f_lo, f_hi = BAND_HZ[band]
        if len(freqs) < 100:
            continue

        centers, enrichment, _, _ = compute_hz_corrected_enrichment(
            freqs, f_lo, f_hi, freq_to_u_log, n_bins=n_fine
        )

        # Smooth to suppress noise
        sigma = n_fine / 50  # ~4 bins
        smoothed = gaussian_filter1d(enrichment, sigma=sigma)

        # Find peaks and troughs
        peaks_idx, _ = find_peaks(smoothed, prominence=5)
        troughs_idx, _ = find_peaks(-smoothed, prominence=5)
        extrema_idx = np.concatenate([peaks_idx, troughs_idx])

        if len(extrema_idx) < 2:
            print(f"\n  {band}: too few extrema ({len(extrema_idx)}), skipping")
            continue

        extrema_u = centers[extrema_idx]

        # Distance from each extremum to nearest phi-position
        def min_dist_to_positions(u_vals, positions):
            dists = []
            for u in u_vals:
                d = np.abs(u - positions)
                d = np.minimum(d, 1 - d)  # circular
                dists.append(np.min(d))
            return np.mean(dists)

        obs_dist = min_dist_to_positions(extrema_u, PHI_POSITIONS)

        # Also check distance to rationals and equal-spaced
        dist_rational = min_dist_to_positions(extrema_u, SIMPLE_RATIONALS)
        dist_equal = min_dist_to_positions(extrema_u, np.linspace(0, 1, 12, endpoint=False))

        # Permutation null: random position sets
        rng = np.random.default_rng(42)
        null_dists = []
        for _ in range(n_perms):
            rand_pos = rng.uniform(0, 1, 12)
            null_dists.append(min_dist_to_positions(extrema_u, rand_pos))
        null_dists = np.array(null_dists)
        p_value = (null_dists <= obs_dist).mean()

        # Also: permutation test on extrema locations (more conservative)
        # Shuffle where extrema are and recompute distance to phi-positions
        null_dists_extrema = []
        for _ in range(n_perms):
            shuffled_u = rng.uniform(0, 1, len(extrema_u))
            null_dists_extrema.append(min_dist_to_positions(shuffled_u, PHI_POSITIONS))
        null_dists_extrema = np.array(null_dists_extrema)
        p_value_extrema = (null_dists_extrema <= obs_dist).mean()

        rows.append({
            'band': band,
            'n_extrema': len(extrema_idx),
            'n_peaks': len(peaks_idx),
            'n_troughs': len(troughs_idx),
            'dist_to_phi': round(obs_dist, 4),
            'dist_to_rational': round(dist_rational, 4),
            'dist_to_equal': round(dist_equal, 4),
            'null_mean': round(null_dists.mean(), 4),
            'p_rand_positions': round(p_value, 4),
            'p_rand_extrema': round(p_value_extrema, 4),
        })
        print(f"\n  {band} ({len(extrema_idx)} extrema: {len(peaks_idx)} peaks, "
              f"{len(troughs_idx)} troughs):")
        print(f"    Mean dist to phi-positions:  {obs_dist:.4f}")
        print(f"    Mean dist to rationals:      {dist_rational:.4f}")
        print(f"    Mean dist to equal-12:       {dist_equal:.4f}")
        print(f"    Null (random positions):     {null_dists.mean():.4f} ± {null_dists.std():.4f}")
        print(f"    p (random positions):        {p_value:.4f}")
        print(f"    p (random extrema):          {p_value_extrema:.4f}")

    return pd.DataFrame(rows)


# =========================================================================
# TEST 4: PERIODICITY ANALYSIS
# =========================================================================

def test_periodicity(band_freqs, n_fine=200):
    """Is there periodic structure in the enrichment profile?

    Computes the autocorrelation of the enrichment profile and checks
    for peaks at specific lags corresponding to:
    - φ⁻¹ ≈ 0.618 (noble_1 spacing)
    - φ⁻² ≈ 0.382 (inv_noble_1 spacing)
    - 1/2 = 0.500 (attractor spacing)
    - 1/3 ≈ 0.333
    - 1/4 = 0.250
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 4: PERIODICITY ANALYSIS")
    print(f"  (Autocorrelation of enrichment profile)")
    print(f"{'=' * 70}")

    test_lags = {
        'phi_inv': PHI_INV,
        'phi_inv2': PHI_INV ** 2,
        'half': 0.5,
        'third': 1/3,
        'quarter': 1/4,
        'fifth': 1/5,
        'sixth': 1/6,
    }

    rows = []
    for band in BAND_ORDER:
        freqs = band_freqs[band]
        f_lo, f_hi = BAND_HZ[band]
        if len(freqs) < 100:
            continue

        centers, enrichment, _, _ = compute_hz_corrected_enrichment(
            freqs, f_lo, f_hi, freq_to_u_log, n_bins=n_fine
        )

        # Normalize
        e = enrichment - enrichment.mean()
        norm = np.sum(e ** 2)
        if norm == 0:
            continue

        # Full autocorrelation
        acf = np.correlate(e, e, mode='full') / norm
        acf = acf[n_fine - 1:]  # positive lags only

        # FFT for power spectrum of enrichment
        fft_power = np.abs(np.fft.rfft(e)) ** 2
        freqs_fft = np.fft.rfftfreq(n_fine)  # in cycles per bin
        periods = np.zeros_like(freqs_fft)
        periods[1:] = 1.0 / freqs_fft[1:]  # in bins
        periods_u = periods / n_fine  # in u-space units

        # Find dominant FFT peak (exclude DC)
        peak_idx = np.argmax(fft_power[1:]) + 1
        dominant_period_u = periods_u[peak_idx] if peak_idx < len(periods_u) else np.nan
        dominant_power_frac = fft_power[peak_idx] / fft_power[1:].sum()

        print(f"\n  {band}:")
        print(f"    Dominant period: {dominant_period_u:.3f} u-units "
              f"({dominant_power_frac:.1%} of power)")

        row = {'band': band, 'dominant_period': round(dominant_period_u, 4),
               'dominant_power_frac': round(dominant_power_frac, 4)}

        # Check autocorrelation at specific lags
        for lag_name, lag_u in test_lags.items():
            lag_bins = int(round(lag_u * n_fine))
            if 0 < lag_bins < n_fine:
                acf_val = acf[lag_bins]
                row[f'acf_{lag_name}'] = round(acf_val, 4)
                print(f"    ACF at {lag_name} ({lag_u:.3f}): {acf_val:+.3f}")
            else:
                row[f'acf_{lag_name}'] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================================
# TEST 5: NOBLE vs RATIONAL
# =========================================================================

def test_noble_vs_rational(band_freqs, n_fine=200, n_perms=5000):
    """Are enrichment values at noble positions different from rationals?

    For each band:
    1. Compute high-res enrichment
    2. Sample enrichment at noble positions and at simple rationals
    3. Compare mean |enrichment| (unsigned) and mean enrichment (signed)
    4. Permutation test for significance
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 5: NOBLE vs RATIONAL ENRICHMENT")
    print(f"  (Are noble numbers enriched/depleted differently than rationals?)")
    print(f"{'=' * 70}")

    rows = []
    for band in BAND_ORDER:
        freqs = band_freqs[band]
        f_lo, f_hi = BAND_HZ[band]
        if len(freqs) < 100:
            continue

        centers, enrichment, _, _ = compute_hz_corrected_enrichment(
            freqs, f_lo, f_hi, freq_to_u_log, n_bins=n_fine
        )

        # Interpolate enrichment at specific positions
        def enr_at(positions):
            bins = np.clip((positions * n_fine).astype(int), 0, n_fine - 1)
            return enrichment[bins]

        noble_enr = enr_at(NOBLE_NUMBERS)
        rational_enr = enr_at(SIMPLE_RATIONALS[1:-1])  # exclude 0 and 1

        # Mean absolute enrichment (structural differentiation)
        noble_abs = np.mean(np.abs(noble_enr))
        rational_abs = np.mean(np.abs(rational_enr))

        # Mean signed enrichment
        noble_signed = np.mean(noble_enr)
        rational_signed = np.mean(rational_enr)

        # Variance of enrichment at positions (structure detection)
        noble_var = np.var(noble_enr)
        rational_var = np.var(rational_enr)

        # Permutation test: sample random positions and compare
        rng = np.random.default_rng(42)
        null_abs_diffs = []
        for _ in range(n_perms):
            n_total = len(NOBLE_NUMBERS) + len(SIMPLE_RATIONALS) - 2
            all_pos = rng.uniform(0, 1, n_total)
            set1 = enr_at(all_pos[:len(NOBLE_NUMBERS)])
            set2 = enr_at(all_pos[len(NOBLE_NUMBERS):])
            null_abs_diffs.append(np.mean(np.abs(set1)) - np.mean(np.abs(set2)))
        null_abs_diffs = np.array(null_abs_diffs)
        obs_diff = noble_abs - rational_abs
        p_abs = np.mean(np.abs(null_abs_diffs) >= np.abs(obs_diff))

        rows.append({
            'band': band,
            'noble_mean_abs': round(noble_abs, 1),
            'rational_mean_abs': round(rational_abs, 1),
            'noble_mean_signed': round(noble_signed, 1),
            'rational_mean_signed': round(rational_signed, 1),
            'noble_var': round(noble_var, 1),
            'rational_var': round(rational_var, 1),
            'diff_abs': round(obs_diff, 1),
            'p_abs': round(p_abs, 4),
        })
        print(f"\n  {band}:")
        print(f"    Noble numbers:    |enr|={noble_abs:.1f}  signed={noble_signed:+.1f}  "
              f"var={noble_var:.0f}")
        print(f"    Simple rationals: |enr|={rational_abs:.1f}  signed={rational_signed:+.1f}  "
              f"var={rational_var:.0f}")
        print(f"    Diff (noble-rat): {obs_diff:+.1f}  p={p_abs:.4f}")

    return pd.DataFrame(rows)


# =========================================================================
# PLOTTING
# =========================================================================

def generate_plots(band_freqs, n_fine=200):
    """Generate visualization of all within-band analyses."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Figure 1: Scaling comparison ---
    fig, axes = plt.subplots(len(BAND_ORDER), len(SCALINGS), figsize=(25, 20))

    for row, band in enumerate(BAND_ORDER):
        freqs = band_freqs[band]
        f_lo, f_hi = BAND_HZ[band]
        if len(freqs) < 100:
            continue
        for col, (scale_name, scale_fn) in enumerate(SCALINGS.items()):
            ax = axes[row, col]
            centers, enrichment, _, _ = compute_hz_corrected_enrichment(
                freqs, f_lo, f_hi, scale_fn, n_bins=n_fine
            )
            ax.plot(centers, enrichment, 'k-', linewidth=0.8, alpha=0.5)
            smoothed = gaussian_filter1d(enrichment, sigma=n_fine / 50)
            ax.plot(centers, smoothed, 'b-', linewidth=2)
            ax.axhline(0, color='gray', linewidth=0.5)

            # Mark phi positions
            if scale_name == 'log':
                for pos, name in zip(PHI_POSITIONS, PHI_POS_NAMES):
                    if name in ('attractor', 'noble_1'):
                        ax.axvline(pos, color='red', alpha=0.5, linewidth=1)
                    elif 'noble' in name or 'inv' in name:
                        ax.axvline(pos, color='orange', alpha=0.3, linewidth=0.8)

            ax.set_xlim(0, 1)
            if row == 0:
                ax.set_title(f'{scale_name}')
            if col == 0:
                ax.set_ylabel(f'{band}\nEnrichment %')
            if row == len(BAND_ORDER) - 1:
                ax.set_xlabel('u (scaled)')

    plt.suptitle('Enrichment Profiles Under Different Scalings (Hz-corrected null)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'scaling_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    # --- Figure 2: Hi-res enrichment with phi positions marked ---
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    for idx, band in enumerate(BAND_ORDER):
        ax = axes[idx]
        freqs = band_freqs[band]
        f_lo, f_hi = BAND_HZ[band]
        if len(freqs) < 100:
            continue
        centers, enrichment, _, _ = compute_hz_corrected_enrichment(
            freqs, f_lo, f_hi, freq_to_u_log, n_bins=n_fine
        )
        smoothed = gaussian_filter1d(enrichment, sigma=n_fine / 50)

        # Plot
        ax.fill_between(centers, 0, enrichment, alpha=0.15, color='steelblue')
        ax.plot(centers, smoothed, 'b-', linewidth=2.5, label='smoothed')
        ax.axhline(0, color='black', linewidth=0.8)

        # Mark phi positions
        colors = {'boundary': 'red', 'attractor': 'darkgreen', 'noble_1': 'purple'}
        for pos, name in zip(PHI_POSITIONS, PHI_POS_NAMES):
            color = colors.get(name, 'orange')
            lw = 2.0 if name in colors else 0.8
            alpha = 0.8 if name in colors else 0.4
            ax.axvline(pos, color=color, alpha=alpha, linewidth=lw,
                       linestyle='--' if name not in colors else '-')

        # Mark rational positions
        for r in [1/3, 2/3, 1/4, 3/4]:
            ax.axvline(r, color='gray', alpha=0.3, linewidth=0.8, linestyle=':')

        hz_ticks = [f_lo * (f_hi / f_lo) ** u for u in [0, 0.25, 0.5, 0.75, 1.0]]
        ax2 = ax.twiny()
        ax2.set_xlim(0, 1)
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_xticklabels([f'{h:.1f}' for h in hz_ticks], fontsize=9)
        ax2.set_xlabel('Hz', fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_ylabel(f'{band}\nEnrichment %')
        ax.set_title(f'{band}  ({f_lo:.2f}–{f_hi:.2f} Hz, {len(freqs):,} peaks)')

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='boundary'),
        Line2D([0], [0], color='darkgreen', linewidth=2, label='attractor'),
        Line2D([0], [0], color='purple', linewidth=2, label='noble_1 (φ⁻¹)'),
        Line2D([0], [0], color='orange', linewidth=1, alpha=0.5, label='other phi'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle=':', label='rationals'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)
    axes[-1].set_xlabel('u (log-frequency coordinate)')

    plt.suptitle('High-Resolution Enrichment with Phi-Lattice Positions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hires_enrichment.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    # --- Figure 3: Autocorrelation ---
    fig, axes = plt.subplots(5, 1, figsize=(14, 18))
    for idx, band in enumerate(BAND_ORDER):
        ax = axes[idx]
        freqs = band_freqs[band]
        f_lo, f_hi = BAND_HZ[band]
        if len(freqs) < 100:
            continue
        centers, enrichment, _, _ = compute_hz_corrected_enrichment(
            freqs, f_lo, f_hi, freq_to_u_log, n_bins=n_fine
        )
        e = enrichment - enrichment.mean()
        norm = np.sum(e ** 2)
        if norm == 0:
            continue
        acf = np.correlate(e, e, mode='full') / norm
        acf = acf[n_fine - 1:]
        lags_u = np.arange(n_fine) / n_fine

        ax.plot(lags_u[:n_fine // 2], acf[:n_fine // 2], 'b-', linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5)

        # Mark key lags
        for lag_name, lag_u, color in [
            ('φ⁻¹', PHI_INV, 'red'),
            ('φ⁻²', PHI_INV ** 2, 'orange'),
            ('1/2', 0.5, 'green'),
            ('1/3', 1/3, 'gray'),
        ]:
            ax.axvline(lag_u, color=color, alpha=0.6, linewidth=1.5,
                       linestyle='--', label=f'{lag_name} ({lag_u:.3f})')

        ax.set_xlim(0, 0.5)
        ax.set_ylabel(f'{band}\nACF')
        ax.legend(fontsize=8, ncol=4)

    axes[-1].set_xlabel('Lag (u-space)')
    plt.suptitle('Autocorrelation of Enrichment Profiles',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'autocorrelation.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    print(f"\n  Plots saved to {OUT_DIR}/")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Within-band coordinate system analysis')
    parser.add_argument('--n-fine', type=int, default=200,
                        help='number of fine bins for high-res enrichment')
    parser.add_argument('--n-perms', type=int, default=5000,
                        help='number of permutations for null tests')
    parser.add_argument('--n-random', type=int, default=1000,
                        help='number of random position sets for landmark test')
    parser.add_argument('--plot', action='store_true', help='generate plots')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading peak data...")
    all_freqs = load_all_peaks()
    print(f"\nTotal: {len(all_freqs):,} peaks")

    band_freqs = get_band_freqs(all_freqs)
    for band in BAND_ORDER:
        f_lo, f_hi = BAND_HZ[band]
        print(f"  {band}: {len(band_freqs[band]):,} peaks "
              f"({f_lo:.2f}-{f_hi:.2f} Hz)")

    # Run all tests
    t0 = time.time()

    df_scaling = test_scaling(band_freqs, n_fine=args.n_fine)
    df_scaling.to_csv(os.path.join(OUT_DIR, 'test1_scaling.csv'), index=False)

    df_landmark = test_landmark_capture(
        band_freqs, n_fine=args.n_fine, n_random=args.n_random)
    df_landmark.to_csv(os.path.join(OUT_DIR, 'test2_landmark.csv'), index=False)

    df_alignment = test_feature_alignment(
        band_freqs, n_fine=args.n_fine, n_perms=args.n_perms)
    df_alignment.to_csv(os.path.join(OUT_DIR, 'test3_alignment.csv'), index=False)

    df_period = test_periodicity(band_freqs, n_fine=args.n_fine)
    df_period.to_csv(os.path.join(OUT_DIR, 'test4_periodicity.csv'), index=False)

    df_noble = test_noble_vs_rational(
        band_freqs, n_fine=args.n_fine, n_perms=args.n_perms)
    df_noble.to_csv(os.path.join(OUT_DIR, 'test5_noble_rational.csv'), index=False)

    elapsed = time.time() - t0
    print(f"\n  All tests completed in {elapsed:.1f}s")

    if args.plot:
        generate_plots(band_freqs, n_fine=args.n_fine)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Test 1 (Scaling): Best scaling per band by R²_best:")
    for band in BAND_ORDER:
        row = df_scaling[df_scaling.band == band]
        if row.empty:
            continue
        best = row.loc[row['r2_best'].idxmax()]
        print(f"    {band:12s}  {best['scaling']:8s}  R²={best['r2_best']:.3f}")

    print(f"\n  Test 2 (Landmarks): Phi-position percentile rank:")
    for _, row in df_landmark.iterrows():
        print(f"    {row['band']:12s}  phi R²={row['r2_phi']:.3f}  "
              f"equal R²={row['r2_equal']:.3f}  "
              f"percentile={row['phi_percentile']:.0f}%")

    if not df_alignment.empty:
        print(f"\n  Test 3 (Feature alignment): p-values (random extrema test):")
        for _, row in df_alignment.iterrows():
            sig = '***' if row['p_rand_extrema'] < 0.001 else \
                  '**' if row['p_rand_extrema'] < 0.01 else \
                  '*' if row['p_rand_extrema'] < 0.05 else 'ns'
            print(f"    {row['band']:12s}  dist_phi={row['dist_to_phi']:.3f}  "
                  f"dist_rat={row['dist_to_rational']:.3f}  "
                  f"p={row['p_rand_extrema']:.4f} {sig}")

    print(f"\n  Test 5 (Noble vs Rational):")
    for _, row in df_noble.iterrows():
        print(f"    {row['band']:12s}  noble |enr|={row['noble_mean_abs']:.0f}  "
              f"rational |enr|={row['rational_mean_abs']:.0f}  "
              f"p={row['p_abs']:.4f}")

    print(f"\n  All results saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
