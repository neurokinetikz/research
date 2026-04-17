#!/usr/bin/env python3
"""
Definitive Test: Log vs Linear Frequency Scaling of EEG Peaks
==============================================================

Takes unbiased FOOOF-detected peaks (3.47M, 9 datasets, 2097 subjects),
and formally tests whether:

1. The peak frequency distribution shows clusters at log-spaced intervals
2. The inter-cluster ratio is better described by phi, e, 2, e-1, or none
3. Log-frequency scaling is statistically superior to linear-frequency
   scaling for describing peak density variation

Methods:
  - KDE-based peak density estimation in Hz-space
  - Trough detection to find empirical band boundaries
  - Model comparison: geometric series f0*r^n with different r values
  - BIC/AIC for formal model selection
  - Per-dataset replication

Usage:
    python scripts/log_scaling_test.py --plot

Outputs to: outputs/log_scaling_test/
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, argrelmin
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v4')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'log_scaling_test')

MIN_POWER_PCT = 50

# All datasets: 11 HBN releases + 4 non-HBN + TDBRAIN
EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp',
    'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2', 'hbn_R3': 'hbn_R3',
    'hbn_R4': 'hbn_R4', 'hbn_R5': 'hbn_R5', 'hbn_R6': 'hbn_R6',
    'hbn_R7': 'hbn_R7', 'hbn_R8': 'hbn_R8', 'hbn_R9': 'hbn_R9',
    'hbn_R10': 'hbn_R10', 'hbn_R11': 'hbn_R11',
    'tdbrain': 'tdbrain',
}

# Candidate models: name -> ratio
CANDIDATE_RATIOS = {
    'phi': PHI,                   # 1.6180
    'e_minus_1': np.e - 1,        # 1.7183
    'sqrt3': np.sqrt(3),          # 1.7321
    'octave': 2.0,                # 2.0000
    'e': np.e,                    # 2.7183
    'third_octave': 2 ** (1/3),   # 1.2599
    'sqrt2': np.sqrt(2),          # 1.4142
}


def load_all_peaks():
    """Load and power-filter peaks from all EC datasets."""
    all_data = {}
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
        all_data[name] = peaks['freq'].values
        print(f"  {name}: {len(files)} subjects, {len(peaks):,} peaks")
    return all_data


# =========================================================================
# TEST 1: Peak density and empirical boundary detection
# =========================================================================

def compute_peak_density(freqs, f_range=(3, 55), n_points=2000):
    """KDE of peak frequencies in linear Hz and log Hz.

    Returns both representations for comparison.
    """
    # Linear Hz density
    hz_grid = np.linspace(f_range[0], f_range[1], n_points)
    kde_hz = stats.gaussian_kde(freqs, bw_method=0.02)
    density_hz = kde_hz(hz_grid)

    # Log Hz density: transform to log space, compute KDE, transform back
    log_freqs = np.log(freqs)
    log_grid = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_points)
    kde_log = stats.gaussian_kde(log_freqs, bw_method=0.02)
    density_log = kde_log(log_grid)
    hz_grid_log = np.exp(log_grid)

    return hz_grid, density_hz, hz_grid_log, density_log


def find_density_troughs(hz_grid, density, min_prominence=0.001):
    """Find local minima (troughs) in the peak density curve.

    These are empirical band boundaries -- frequencies where peaks are scarce.
    """
    # Smooth to suppress noise
    smoothed = gaussian_filter1d(density, sigma=len(hz_grid) // 100)
    # Find troughs (minima = maxima of negative)
    trough_idx, props = find_peaks(-smoothed, prominence=min_prominence,
                                    distance=len(hz_grid) // 20)
    return hz_grid[trough_idx], smoothed[trough_idx], smoothed


def find_density_peaks(hz_grid, density, min_prominence=0.002):
    """Find local maxima (peaks) in the density curve.

    These are band centers -- frequencies where peaks concentrate.
    """
    smoothed = gaussian_filter1d(density, sigma=len(hz_grid) // 100)
    peak_idx, props = find_peaks(smoothed, prominence=min_prominence,
                                  distance=len(hz_grid) // 20)
    return hz_grid[peak_idx], smoothed[peak_idx], smoothed


# =========================================================================
# TEST 2: Geometric series model comparison
# =========================================================================

def fit_geometric_model(empirical_freqs, ratio, n_terms=6):
    """Find optimal f0 for a geometric series f0 * ratio^n.

    Minimizes sum of squared log-distances between empirical frequencies
    and nearest model frequency.

    Returns: f0, model_freqs, residuals, SSE
    """
    log_ratio = np.log(ratio)
    log_emp = np.log(empirical_freqs)

    best_f0 = None
    best_sse = np.inf

    # Sweep f0
    for f0_candidate in np.linspace(3.0, 12.0, 1000):
        log_f0 = np.log(f0_candidate)
        model_log = log_f0 + np.arange(-2, n_terms) * log_ratio
        model_log = model_log[(model_log > np.log(2)) & (model_log < np.log(60))]

        # Distance from each empirical freq to nearest model freq
        residuals = []
        for le in log_emp:
            min_dist = np.min(np.abs(le - model_log))
            residuals.append(min_dist)
        sse = np.sum(np.array(residuals) ** 2)

        if sse < best_sse:
            best_sse = sse
            best_f0 = f0_candidate
            best_residuals = residuals

    model_freqs = best_f0 * ratio ** np.arange(-2, n_terms)
    model_freqs = model_freqs[(model_freqs > 2) & (model_freqs < 60)]
    return best_f0, model_freqs, np.array(best_residuals), best_sse


def compute_bic(sse, n_obs, n_params):
    """BIC = n*ln(SSE/n) + k*ln(n)."""
    if sse <= 0 or n_obs <= 0:
        return np.inf
    return n_obs * np.log(sse / n_obs) + n_params * np.log(n_obs)


def test_geometric_models(empirical_centers, empirical_boundaries):
    """Compare geometric series models for both centers and boundaries."""
    print(f"\n{'=' * 70}")
    print(f"  TEST 2: GEOMETRIC SERIES MODEL COMPARISON")
    print(f"{'=' * 70}")

    results = []

    for target_name, target_freqs in [
        ('density_peaks', empirical_centers),
        ('density_troughs', empirical_boundaries),
    ]:
        if len(target_freqs) < 3:
            print(f"\n  {target_name}: too few points ({len(target_freqs)})")
            continue

        print(f"\n  Fitting to {target_name}: {np.round(target_freqs, 2)} Hz")

        # Compute consecutive ratios
        ratios = target_freqs[1:] / target_freqs[:-1]
        mean_ratio = np.exp(np.mean(np.log(ratios)))  # geometric mean
        std_ratio = np.std(np.log(ratios))
        print(f"  Consecutive ratios: {np.round(ratios, 3)}")
        print(f"  Geometric mean ratio: {mean_ratio:.4f}  (log-std: {std_ratio:.4f})")

        n_obs = len(target_freqs)

        # Free model: each frequency is a free parameter
        # SSE = 0, n_params = n_obs
        bic_free = compute_bic(1e-10, n_obs, n_obs)  # perfect fit

        for model_name, ratio in CANDIDATE_RATIOS.items():
            f0, model_freqs, residuals, sse = fit_geometric_model(
                target_freqs, ratio
            )
            # 1 free parameter (f0), ratio is fixed
            bic = compute_bic(sse, n_obs, 1)
            # Also compute with ratio as free parameter (2 params)
            results.append({
                'target': target_name,
                'model': model_name,
                'ratio': round(ratio, 4),
                'f0': round(f0, 3),
                'sse': round(sse, 6),
                'bic_1param': round(bic, 3),
                'mean_residual_cents': round(np.mean(np.abs(residuals)) * 1200 / np.log(2), 1),
                'max_residual_cents': round(np.max(np.abs(residuals)) * 1200 / np.log(2), 1),
                'model_freqs': np.round(model_freqs, 2).tolist(),
            })
            print(f"    {model_name:15s}  r={ratio:.4f}  f0={f0:.2f}  "
                  f"SSE={sse:.6f}  BIC={bic:.2f}  "
                  f"mean_err={np.mean(np.abs(residuals)) * 1200 / np.log(2):.0f} cents")

        # Free-ratio model: fit both f0 and ratio
        best_sse_free = np.inf
        best_f0_free = None
        best_ratio_free = None
        for r_try in np.linspace(1.2, 3.0, 500):
            f0, _, _, sse = fit_geometric_model(target_freqs, r_try)
            if sse < best_sse_free:
                best_sse_free = sse
                best_f0_free = f0
                best_ratio_free = r_try
        bic_free_ratio = compute_bic(best_sse_free, n_obs, 2)
        results.append({
            'target': target_name,
            'model': 'free_ratio',
            'ratio': round(best_ratio_free, 4),
            'f0': round(best_f0_free, 3),
            'sse': round(best_sse_free, 6),
            'bic_1param': round(bic_free_ratio, 3),
            'mean_residual_cents': 0,
            'max_residual_cents': 0,
            'model_freqs': [],
        })
        print(f"    {'free_ratio':15s}  r={best_ratio_free:.4f}  f0={best_f0_free:.2f}  "
              f"SSE={best_sse_free:.6f}  BIC={bic_free_ratio:.2f}  (2 free params)")

        # Linear model: equal spacing in Hz
        mean_spacing = np.mean(np.diff(target_freqs))
        linear_model = target_freqs[0] + np.arange(n_obs) * mean_spacing
        linear_residuals = np.log(target_freqs) - np.log(linear_model)
        linear_sse = np.sum(linear_residuals ** 2)
        linear_bic = compute_bic(linear_sse, n_obs, 2)  # f0 + spacing
        results.append({
            'target': target_name,
            'model': 'linear_Hz',
            'ratio': 0,
            'f0': round(target_freqs[0], 3),
            'sse': round(linear_sse, 6),
            'bic_1param': round(linear_bic, 3),
            'mean_residual_cents': round(np.mean(np.abs(linear_residuals)) * 1200 / np.log(2), 1),
            'max_residual_cents': round(np.max(np.abs(linear_residuals)) * 1200 / np.log(2), 1),
            'model_freqs': np.round(linear_model, 2).tolist(),
        })
        print(f"    {'linear_Hz':15s}  spacing={mean_spacing:.2f} Hz  "
              f"SSE={linear_sse:.6f}  BIC={linear_bic:.2f}")

    return pd.DataFrame(results)


# =========================================================================
# TEST 3: Log vs linear density description
# =========================================================================

def test_log_vs_linear_density(freqs, f_range=(4, 52), n_bins=100):
    """Is the peak density function simpler in log-Hz or linear-Hz space?

    Computes the histogram in both spaces and measures:
    - Entropy (lower = more structured)
    - Number of modes
    - Description length (how many parameters to describe the shape)
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 3: LOG vs LINEAR DENSITY DESCRIPTION")
    print(f"{'=' * 70}")

    # Linear Hz histogram
    counts_lin, edges_lin = np.histogram(freqs, bins=n_bins,
                                          range=f_range)
    density_lin = counts_lin / counts_lin.sum()

    # Log Hz histogram (equal-width bins in log space)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_bins + 1)
    counts_log, _ = np.histogram(np.log(freqs), bins=log_edges)
    density_log = counts_log / counts_log.sum()

    # 1. Entropy (bits)
    entropy_lin = stats.entropy(density_lin + 1e-12, base=2)
    entropy_log = stats.entropy(density_log + 1e-12, base=2)

    # 2. Flatness (ratio of geometric mean to arithmetic mean)
    # Flatter = closer to 1 = less structured
    def flatness(d):
        d = d[d > 0]
        return np.exp(np.mean(np.log(d))) / np.mean(d)

    flat_lin = flatness(density_lin)
    flat_log = flatness(density_log)

    # 3. Number of modes (peaks above 1.5x median)
    def count_modes(d, min_prom_frac=0.3):
        smoothed = gaussian_filter1d(d.astype(float), sigma=2)
        median_val = np.median(smoothed[smoothed > 0])
        peaks, _ = find_peaks(smoothed, prominence=median_val * min_prom_frac)
        return len(peaks)

    modes_lin = count_modes(density_lin)
    modes_log = count_modes(density_log)

    # 4. Polynomial fit comparison: which space needs fewer parameters?
    x_lin = np.arange(n_bins, dtype=float)
    x_log = np.arange(n_bins, dtype=float)

    poly_results = {}
    for name, x, d in [('linear', x_lin, density_lin), ('log', x_log, density_log)]:
        for degree in [2, 3, 4, 5, 6]:
            coeffs = np.polyfit(x, d, degree)
            pred = np.polyval(coeffs, x)
            ss_res = np.sum((d - pred) ** 2)
            ss_tot = np.sum((d - d.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            bic = n_bins * np.log(ss_res / n_bins + 1e-20) + (degree + 1) * np.log(n_bins)
            poly_results[(name, degree)] = {'r2': r2, 'bic': bic}

    print(f"\n  Entropy (lower = more structured):")
    print(f"    Linear Hz:  {entropy_lin:.3f} bits")
    print(f"    Log Hz:     {entropy_log:.3f} bits")
    print(f"    Winner:     {'log' if entropy_log < entropy_lin else 'linear'} "
          f"(Δ = {abs(entropy_lin - entropy_log):.3f} bits)")

    print(f"\n  Spectral flatness (lower = more peaked):")
    print(f"    Linear Hz:  {flat_lin:.4f}")
    print(f"    Log Hz:     {flat_log:.4f}")
    print(f"    Winner:     {'log' if flat_log < flat_lin else 'linear'}")

    print(f"\n  Number of modes:")
    print(f"    Linear Hz:  {modes_lin}")
    print(f"    Log Hz:     {modes_log}")

    print(f"\n  Polynomial fit R² (degree → R²):")
    for name in ['linear', 'log']:
        vals = [f"d{d}={poly_results[(name, d)]['r2']:.3f}" for d in [2, 3, 4, 5, 6]]
        print(f"    {name:8s}  {', '.join(vals)}")

    print(f"\n  Polynomial BIC (lower = better, penalizes complexity):")
    for degree in [3, 4, 5]:
        bic_lin = poly_results[('linear', degree)]['bic']
        bic_log = poly_results[('log', degree)]['bic']
        winner = 'log' if bic_log < bic_lin else 'linear'
        print(f"    degree {degree}:  linear={bic_lin:.1f}  log={bic_log:.1f}  "
              f"→ {winner} (ΔBIC={abs(bic_lin - bic_log):.1f})")

    return {
        'entropy_lin': entropy_lin, 'entropy_log': entropy_log,
        'flatness_lin': flat_lin, 'flatness_log': flat_log,
        'modes_lin': modes_lin, 'modes_log': modes_log,
        'poly_results': poly_results,
        'density_lin': density_lin, 'density_log': density_log,
        'edges_lin': edges_lin, 'log_edges': log_edges,
    }


# =========================================================================
# TEST 4: Per-dataset replication of boundary ratios
# =========================================================================

def test_per_dataset_ratios(all_data, f_range=(3, 55)):
    """For each dataset independently, find density troughs and compute ratios."""
    print(f"\n{'=' * 70}")
    print(f"  TEST 4: PER-DATASET BOUNDARY RATIO REPLICATION")
    print(f"{'=' * 70}")

    all_ratios = []
    rows = []
    for name, freqs in all_data.items():
        if len(freqs) < 1000:
            continue
        hz_grid, density_hz, _, _ = compute_peak_density(freqs, f_range)
        troughs, _, smoothed = find_density_troughs(hz_grid, density_hz)

        # Keep only troughs in plausible range
        troughs = troughs[(troughs > 5) & (troughs < 45)]
        if len(troughs) < 2:
            print(f"  {name}: {len(troughs)} troughs, skipping")
            continue

        ratios = troughs[1:] / troughs[:-1]
        geo_mean = np.exp(np.mean(np.log(ratios)))
        rows.append({
            'dataset': name,
            'n_peaks': len(freqs),
            'n_troughs': len(troughs),
            'troughs_hz': np.round(troughs, 2).tolist(),
            'ratios': np.round(ratios, 3).tolist(),
            'geo_mean_ratio': round(geo_mean, 4),
        })
        all_ratios.extend(ratios.tolist())
        print(f"  {name:12s}  troughs={np.round(troughs, 1)}  "
              f"ratios={np.round(ratios, 3)}  geo_mean={geo_mean:.3f}")

    if all_ratios:
        all_ratios = np.array(all_ratios)
        overall_mean = np.exp(np.mean(np.log(all_ratios)))
        overall_std = np.std(np.log(all_ratios))
        print(f"\n  Overall geometric mean ratio: {overall_mean:.4f} "
              f"(log-std: {overall_std:.4f})")
        print(f"  N ratios: {len(all_ratios)}")

        # One-sample t-tests against candidate constants
        log_ratios = np.log(all_ratios)
        print(f"\n  One-sample tests (H0: log(ratio) = log(constant)):")
        for cname, cval in CANDIDATE_RATIOS.items():
            t, p = stats.ttest_1samp(log_ratios, np.log(cval))
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"    vs {cname:15s} ({cval:.4f}):  t={t:+.3f}  p={p:.4f}  {sig}")

    return pd.DataFrame(rows)


# =========================================================================
# TEST 5: KDE BANDWIDTH SWEEP
# =========================================================================

def test_bandwidth_stability(all_freqs, f_range=(3, 55)):
    """Sweep smoothing bandwidth and show trough locations are stable.

    Uses histogram + Gaussian smoothing (fast) instead of KDE (slow).
    Sweeps sigma from narrow (2 bins) to wide (30 bins) on a 1000-bin
    log-frequency histogram.
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 5: SMOOTHING BANDWIDTH STABILITY")
    print(f"  (Do trough locations depend on smoothing?)")
    print(f"{'=' * 70}")

    n_hist = 1000
    log_freqs = np.log(all_freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)

    counts, _ = np.histogram(log_freqs, bins=log_edges)
    counts_f = counts.astype(float)

    sigmas = np.linspace(2, 30, 30)  # smoothing sigma in bins
    all_troughs = []
    rows = []

    for sigma in sigmas:
        smoothed = gaussian_filter1d(counts_f, sigma=sigma)
        median_val = np.median(smoothed[smoothed > 0])
        trough_idx, _ = find_peaks(-smoothed, prominence=median_val * 0.05,
                                    distance=n_hist // 30)
        trough_hz = hz_centers[trough_idx]
        trough_hz = trough_hz[(trough_hz > 4) & (trough_hz < 50)]

        all_troughs.append(trough_hz)
        rows.append({
            'sigma': round(sigma, 1),
            'n_troughs': len(trough_hz),
            'troughs_hz': np.round(trough_hz, 2).tolist(),
        })

        if len(trough_hz) >= 2:
            ratios = trough_hz[1:] / trough_hz[:-1]
            geo = np.exp(np.mean(np.log(ratios)))
            rows[-1]['geo_mean_ratio'] = round(geo, 4)

    # Identify stable troughs: present across >50% of smoothings
    # Bin all trough locations into 0.5 Hz bins
    all_t = np.concatenate(all_troughs)
    hist, edges = np.histogram(all_t, bins=np.arange(3, 55, 0.5))
    stable_bins = edges[:-1][hist > len(sigmas) * 0.5]
    stable_troughs = []
    for sb in stable_bins:
        # Find median trough location in this bin
        mask = (all_t >= sb) & (all_t < sb + 0.5)
        if mask.sum() > 0:
            stable_troughs.append(np.median(all_t[mask]))
    # Merge nearby stable troughs (within 1.5 Hz)
    if stable_troughs:
        merged = [stable_troughs[0]]
        for t in stable_troughs[1:]:
            if t - merged[-1] > 1.5:
                merged.append(t)
            else:
                merged[-1] = (merged[-1] + t) / 2
        stable_troughs = np.array(merged)
    else:
        stable_troughs = np.array([])

    print(f"\n  Swept {len(sigmas)} sigmas from {sigmas[0]:.1f} to {sigmas[-1]:.1f} bins")
    print(f"  Troughs found per smoothing: {min(r['n_troughs'] for r in rows)}-"
          f"{max(r['n_troughs'] for r in rows)}")

    print(f"\n  Stable troughs (present in >50% of bandwidths):")
    print(f"    {np.round(stable_troughs, 2)} Hz")
    if len(stable_troughs) > 1:
        ratios = stable_troughs[1:] / stable_troughs[:-1]
        print(f"    Consecutive ratios: {np.round(ratios, 3)}")
        print(f"    Geometric mean: {np.exp(np.mean(np.log(ratios))):.4f}")

    # Compute std of trough locations across bandwidths
    # For each stable trough, find its position at each bandwidth
    print(f"\n  Trough position stability (Hz std across bandwidths):")
    for st in stable_troughs:
        positions = []
        for trough_set in all_troughs:
            if len(trough_set) == 0:
                continue
            nearest_idx = np.argmin(np.abs(trough_set - st))
            if np.abs(trough_set[nearest_idx] - st) < 3:
                positions.append(trough_set[nearest_idx])
        if positions:
            arr = np.array(positions)
            print(f"    ~{st:.1f} Hz: mean={arr.mean():.2f} ± {arr.std():.2f} Hz  "
                  f"(found in {len(positions)}/{len(sigmas)} smoothings)")

    df = pd.DataFrame(rows)
    return df, stable_troughs, all_troughs, sigmas


# =========================================================================
# TEST 6: APERIODIC-ONLY NULL
# =========================================================================

def test_aperiodic_null(all_freqs, n_surrogates=200, f_range=(3, 55)):
    """Test whether troughs arise from oscillatory structure or 1/f density.

    Null model: Generate surrogate peak distributions that preserve the
    overall density envelope (1/f gradient) but destroy band structure.
    Uses histogram-based surrogates (Poisson resampling) for speed.
    """
    print(f"\n{'=' * 70}")
    print(f"  TEST 6: APERIODIC-ONLY NULL")
    print(f"  (Do troughs survive when band structure is destroyed?)")
    print(f"  ({n_surrogates} surrogates, histogram-based)")
    print(f"{'=' * 70}")

    n_hist = 500  # histogram bins for fast surrogate generation
    log_freqs = np.log(all_freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)

    # Real histogram
    real_counts, _ = np.histogram(log_freqs, bins=log_edges)
    real_smooth = gaussian_filter1d(real_counts.astype(float), sigma=5)

    # Smooth envelope: very wide Gaussian smoothing (destroys band structure)
    envelope = gaussian_filter1d(real_counts.astype(float), sigma=40)
    envelope = np.maximum(envelope, 1)  # prevent zeros

    # Real troughs in smoothed histogram
    trough_idx_real, _ = find_peaks(-real_smooth, prominence=np.median(real_smooth) * 0.1,
                                     distance=n_hist // 20)
    real_trough_hz = hz_centers[trough_idx_real]
    real_trough_hz = real_trough_hz[(real_trough_hz > 4) & (real_trough_hz < 50)]

    # Real trough depths (ratio of trough count to envelope)
    real_trough_depths = []
    for th in real_trough_hz:
        idx = np.argmin(np.abs(hz_centers - th))
        depth = real_smooth[idx] / envelope[idx] if envelope[idx] > 0 else 1
        real_trough_depths.append(depth)
    real_trough_depths = np.array(real_trough_depths)

    print(f"\n  Real troughs: {np.round(real_trough_hz, 2)} Hz")
    print(f"  Real trough depths (count/envelope): "
          f"{np.round(real_trough_depths, 3)}")

    # Generate surrogates: Poisson-sample from envelope
    rng = np.random.default_rng(42)

    surrogate_trough_counts = []
    surrogate_max_depths = []
    surrogate_mean_depths = []
    surrogate_geo_mean_ratios = []

    for s in range(n_surrogates):
        surr_counts = rng.poisson(envelope)
        surr_smooth = gaussian_filter1d(surr_counts.astype(float), sigma=5)

        trough_idx_surr, _ = find_peaks(
            -surr_smooth, prominence=np.median(surr_smooth) * 0.1,
            distance=n_hist // 20)
        surr_trough_hz = hz_centers[trough_idx_surr]
        surr_trough_hz = surr_trough_hz[
            (surr_trough_hz > 4) & (surr_trough_hz < 50)]

        surrogate_trough_counts.append(len(surr_trough_hz))

        surr_depths = []
        for th in surr_trough_hz:
            idx = np.argmin(np.abs(hz_centers - th))
            depth = surr_smooth[idx] / envelope[idx] if envelope[idx] > 0 else 1
            surr_depths.append(depth)

        if surr_depths:
            surrogate_max_depths.append(min(surr_depths))
            surrogate_mean_depths.append(np.mean(surr_depths))
        else:
            surrogate_max_depths.append(1.0)
            surrogate_mean_depths.append(1.0)

        if len(surr_trough_hz) >= 2:
            ratios = surr_trough_hz[1:] / surr_trough_hz[:-1]
            surrogate_geo_mean_ratios.append(
                np.exp(np.mean(np.log(ratios))))

    surrogate_trough_counts = np.array(surrogate_trough_counts)
    surrogate_max_depths = np.array(surrogate_max_depths)
    surrogate_mean_depths = np.array(surrogate_mean_depths)
    surrogate_geo_mean_ratios = np.array(surrogate_geo_mean_ratios)

    # Statistics
    print(f"\n  Surrogate analysis ({n_surrogates} surrogates):")
    print(f"    Surrogate trough count: {surrogate_trough_counts.mean():.1f} ± "
          f"{surrogate_trough_counts.std():.1f}  "
          f"(real: {len(real_trough_hz)})")

    real_deepest = min(real_trough_depths) if len(real_trough_depths) > 0 else 1
    p_depth = (surrogate_max_depths <= real_deepest).mean()
    print(f"    Deepest real trough depth: {real_deepest:.3f}")
    print(f"    Surrogate deepest: {surrogate_max_depths.mean():.3f} ± "
          f"{surrogate_max_depths.std():.3f}")
    print(f"    p(surrogate as deep as real): {p_depth:.4f}")

    if len(real_trough_depths) > 0:
        real_mean_depth = np.mean(real_trough_depths)
        p_mean = (surrogate_mean_depths <= real_mean_depth).mean()
        print(f"    Mean real trough depth: {real_mean_depth:.3f}")
        print(f"    Surrogate mean depth: {surrogate_mean_depths.mean():.3f} ± "
              f"{surrogate_mean_depths.std():.3f}")
        print(f"    p(surrogate mean as deep): {p_mean:.4f}")

    if len(real_trough_hz) >= 2:
        real_ratios = real_trough_hz[1:] / real_trough_hz[:-1]
        real_geo = np.exp(np.mean(np.log(real_ratios)))
        if len(surrogate_geo_mean_ratios) > 0:
            surr_geo_mean = np.mean(surrogate_geo_mean_ratios)
            surr_geo_std = np.std(surrogate_geo_mean_ratios)
            # How far is real geo mean from surrogate distribution?
            z_ratio = (real_geo - surr_geo_mean) / surr_geo_std if surr_geo_std > 0 else 0
            print(f"\n    Real trough ratio geo mean: {real_geo:.4f}")
            print(f"    Surrogate ratio geo mean: {surr_geo_mean:.4f} ± {surr_geo_std:.4f}")
            print(f"    z-score: {z_ratio:.2f}")

    # Per-trough analysis: for each real trough, check if surrogates
    # have a trough near the same location
    print(f"\n  Per-trough survival (how many surrogates have trough within 1 Hz):")
    trough_survival = {}
    # Reuse hz_centers, real_smooth, envelope from above
    for th in real_trough_hz:
        idx = np.argmin(np.abs(hz_centers - th))
        real_depth_here = real_smooth[idx] / (envelope[idx] + 1e-10)

        # Fast surrogate: Poisson sampling from envelope
        surr_depths = []
        for _ in range(n_surrogates):
            surr_counts = rng.poisson(envelope)
            surr_sm = gaussian_filter1d(surr_counts.astype(float), sigma=5)
            surr_depth = surr_sm[idx] / (envelope[idx] + 1e-10)
            surr_depths.append(surr_depth)
        surr_depths = np.array(surr_depths)
        p_here = (surr_depths <= real_depth_here).mean()
        trough_survival[round(th, 2)] = p_here
        sig = '***' if p_here < 0.001 else '**' if p_here < 0.01 else '*' if p_here < 0.05 else 'ns'
        print(f"    {th:.1f} Hz: depth_ratio={real_depth_here:.3f}  "
              f"surr={surr_depths.mean():.3f}±{surr_depths.std():.3f}  "
              f"p={p_here:.4f} {sig}")

    return {
        'real_trough_hz': real_trough_hz,
        'real_trough_depths': real_trough_depths,
        'surrogate_trough_counts': surrogate_trough_counts,
        'surrogate_max_depths': surrogate_max_depths,
        'surrogate_mean_depths': surrogate_mean_depths,
        'surrogate_geo_mean_ratios': surrogate_geo_mean_ratios,
        'trough_survival': trough_survival,
        'envelope': envelope,
        'hz_centers': hz_centers,
        'real_smooth': real_smooth,
    }


# =========================================================================
# PLOTTING
# =========================================================================

def generate_plots(all_freqs, all_data, empirical_centers, empirical_boundaries,
                   model_df, density_results):
    """Generate comprehensive visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Figure 1: Peak density in Hz and log-Hz with empirical boundaries ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    f_range = (3, 55)
    hz_grid, density_hz, hz_grid_log, density_log = compute_peak_density(
        all_freqs, f_range, n_points=3000
    )

    # Panel A: Linear Hz density
    ax = axes[0]
    ax.fill_between(hz_grid, 0, density_hz, alpha=0.3, color='steelblue')
    smoothed = gaussian_filter1d(density_hz, sigma=30)
    ax.plot(hz_grid, smoothed, 'b-', linewidth=2)
    for bnd in empirical_boundaries:
        ax.axvline(bnd, color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    for ctr in empirical_centers:
        ax.axvline(ctr, color='green', linewidth=1, alpha=0.5, linestyle=':')
    # Phi-lattice boundaries
    phi_bnds = [F0 / PHI, F0, F0 * PHI, F0 * PHI ** 2, F0 * PHI ** 3, F0 * PHI ** 4]
    for pb in phi_bnds:
        if f_range[0] < pb < f_range[1]:
            ax.axvline(pb, color='orange', linewidth=1, alpha=0.5, linestyle='-.')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak density')
    ax.set_title('A. Peak density in linear Hz')
    ax.set_xlim(f_range)

    # Panel B: Log Hz density
    ax = axes[1]
    ax.fill_between(hz_grid_log, 0, density_log, alpha=0.3, color='steelblue')
    smoothed_log = gaussian_filter1d(density_log, sigma=30)
    ax.plot(hz_grid_log, smoothed_log, 'b-', linewidth=2)
    for bnd in empirical_boundaries:
        ax.axvline(bnd, color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    for pb in phi_bnds:
        if f_range[0] < pb < f_range[1]:
            ax.axvline(pb, color='orange', linewidth=1, alpha=0.5, linestyle='-.')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz, log scale)')
    ax.set_ylabel('Peak density (in log-Hz)')
    ax.set_title('B. Peak density in log Hz')
    ax.set_xlim(f_range)

    # Panel C: Density in log-space (x-axis = log Hz)
    ax = axes[2]
    log_grid = np.linspace(np.log(f_range[0]), np.log(f_range[1]), 3000)
    log_kde = stats.gaussian_kde(np.log(all_freqs), bw_method=0.02)
    log_density = log_kde(log_grid)
    ax.fill_between(log_grid, 0, log_density, alpha=0.3, color='steelblue')
    smoothed_ld = gaussian_filter1d(log_density, sigma=30)
    ax.plot(log_grid, smoothed_ld, 'b-', linewidth=2)
    for bnd in empirical_boundaries:
        ax.axvline(np.log(bnd), color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    for pb in phi_bnds:
        if f_range[0] < pb < f_range[1]:
            ax.axvline(np.log(pb), color='orange', linewidth=1, alpha=0.5, linestyle='-.')
    ax.set_xlabel('log(Frequency)')
    ax.set_ylabel('Peak density')
    ax.set_title('C. Peak density in log-frequency space (x = ln(Hz))')

    legend_elements = [
        Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', label='empirical troughs'),
        Line2D([0], [0], color='green', linewidth=1, linestyle=':', label='empirical peaks'),
        Line2D([0], [0], color='orange', linewidth=1, linestyle='-.', label='φ-lattice boundaries'),
    ]
    axes[0].legend(handles=legend_elements, fontsize=9, loc='upper right')

    plt.suptitle(f'Peak Frequency Distribution (N={len(all_freqs):,} peaks, 9 datasets)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'peak_density.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Figure 2: Model comparison ---
    if model_df is not None and not model_df.empty:
        for target in model_df['target'].unique():
            tdf = model_df[model_df['target'] == target].copy()
            tdf = tdf.sort_values('sse')

            fig, ax = plt.subplots(figsize=(12, 6))
            colors = {'phi': 'gold', 'e_minus_1': 'coral', 'octave': 'skyblue',
                      'e': 'plum', 'sqrt2': 'lightgreen', 'sqrt3': 'lightyellow',
                      'third_octave': 'lightgray', 'free_ratio': 'white',
                      'linear_Hz': 'lightcoral'}
            bars = ax.barh(range(len(tdf)), tdf['sse'],
                          color=[colors.get(m, 'gray') for m in tdf['model']])
            ax.set_yticks(range(len(tdf)))
            ax.set_yticklabels([f"{r['model']} (r={r['ratio']:.3f})"
                                for _, r in tdf.iterrows()])
            ax.set_xlabel('SSE (lower = better fit)')
            ax.set_title(f'Geometric Series Model Comparison: {target}')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f'model_comparison_{target}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # --- Figure 3: Histogram comparison (log vs linear bins) ---
    if density_results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        edges_lin = density_results['edges_lin']
        centers_lin = (edges_lin[:-1] + edges_lin[1:]) / 2
        axes[0].bar(centers_lin, density_results['density_lin'],
                    width=np.diff(edges_lin), edgecolor='black', linewidth=0.3,
                    color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Peak distribution: equal-width Hz bins')

        log_edges = density_results['log_edges']
        centers_log = (log_edges[:-1] + log_edges[1:]) / 2
        axes[1].bar(centers_log, density_results['density_log'],
                    width=np.diff(log_edges), edgecolor='black', linewidth=0.3,
                    color='steelblue', alpha=0.7)
        axes[1].set_xlabel('log(Frequency)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Peak distribution: equal-width log-Hz bins')

        plt.suptitle('Log vs Linear Binning of Peak Frequencies',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'log_vs_linear_bins.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n  Plots saved to {OUT_DIR}/")


def plot_bandwidth_stability(bw_df, stable_troughs, all_troughs_list, sigmas):
    """Plot trough locations across smoothing bandwidths."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Scatter each trough at its smoothing sigma
    for i, (sigma, troughs) in enumerate(zip(sigmas, all_troughs_list)):
        for t in troughs:
            ax.plot(t, sigma, 'ko', markersize=3, alpha=0.4)

    # Mark stable troughs
    for st in stable_troughs:
        ax.axvline(st, color='red', linewidth=2, alpha=0.7, linestyle='--')

    # Mark phi-lattice boundaries
    phi_bnds = [F0 / PHI, F0, F0 * PHI, F0 * PHI ** 2, F0 * PHI ** 3]
    for pb in phi_bnds:
        ax.axvline(pb, color='orange', linewidth=1.5, alpha=0.5, linestyle='-.')

    ax.set_xlabel('Trough frequency (Hz)', fontsize=12)
    ax.set_ylabel('Smoothing sigma (bins)', fontsize=12)
    ax.set_title('Trough Location Stability Across Smoothing Bandwidths\n'
                 '(red dashed = stable troughs, orange = φ-lattice)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(3, 50)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'bandwidth_stability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Bandwidth stability plot saved")


def plot_aperiodic_null(null_results):
    """Plot aperiodic null test results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    hz_centers = null_results['hz_centers']
    real_smooth = null_results['real_smooth']
    envelope = null_results['envelope']

    # Panel A: Real density vs smooth envelope
    ax = axes[0, 0]
    ax.plot(hz_centers, real_smooth, 'b-', linewidth=1.5, label='real (smoothed)')
    ax.plot(hz_centers, envelope, 'r-', linewidth=2, label='smooth envelope (aperiodic)')
    for th in null_results['real_trough_hz']:
        ax.axvline(th, color='green', alpha=0.5, linewidth=1, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak count (smoothed)')
    ax.set_title('A. Real peak density vs aperiodic envelope')
    ax.legend(fontsize=9)
    ax.set_xlim(3, 55)

    # Panel B: Trough depth ratio (real / envelope)
    ax = axes[0, 1]
    ratio = real_smooth / (envelope + 1e-10)
    ax.plot(hz_centers, ratio, 'b-', linewidth=1)
    ax.axhline(1.0, color='black', linewidth=0.5)
    for th in null_results['real_trough_hz']:
        ax.axvline(th, color='green', alpha=0.5, linewidth=1, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Count / Envelope')
    ax.set_title('B. Deviation from smooth envelope')
    ax.set_xlim(3, 55)

    # Panel C: Surrogate trough count distribution
    ax = axes[1, 0]
    counts = null_results['surrogate_trough_counts']
    ax.hist(counts, bins=20, color='gray', alpha=0.7, edgecolor='black')
    ax.axvline(len(null_results['real_trough_hz']), color='red',
               linewidth=2, label=f"real = {len(null_results['real_trough_hz'])}")
    ax.set_xlabel('Number of troughs')
    ax.set_ylabel('Surrogate count')
    ax.set_title('C. Trough count: real vs surrogates')
    ax.legend()

    # Panel D: Surrogate deepest trough distribution
    ax = axes[1, 1]
    depths = null_results['surrogate_max_depths']
    ax.hist(depths, bins=30, color='gray', alpha=0.7, edgecolor='black')
    real_deepest = min(null_results['real_trough_depths'])
    ax.axvline(real_deepest, color='red', linewidth=2,
               label=f"real deepest = {real_deepest:.3f}")
    ax.set_xlabel('Deepest trough (density/envelope)')
    ax.set_ylabel('Surrogate count')
    ax.set_title('D. Deepest trough depth: real vs surrogates')
    ax.legend()

    plt.suptitle('Aperiodic-Only Null Test: Are Troughs Genuine?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'aperiodic_null.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Aperiodic null plot saved")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Definitive test of log vs linear frequency scaling')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("Loading peak data...")
    all_data = load_all_peaks()
    all_freqs = np.concatenate(list(all_data.values()))
    print(f"\nTotal: {len(all_freqs):,} peaks from {len(all_data)} datasets")

    # Test 1: Peak density and empirical boundaries
    print(f"\n{'=' * 70}")
    print(f"  TEST 1: PEAK DENSITY AND EMPIRICAL BOUNDARIES")
    print(f"{'=' * 70}")

    hz_grid, density_hz, hz_grid_log, density_log = compute_peak_density(all_freqs)

    # Find troughs (boundaries) and peaks (centers) in log-space density
    troughs, trough_vals, smoothed = find_density_troughs(hz_grid_log, density_log)
    centers, center_vals, _ = find_density_peaks(hz_grid_log, density_log)

    # Filter to reasonable range
    troughs = troughs[(troughs > 3) & (troughs < 50)]
    centers = centers[(centers > 3) & (centers < 50)]

    print(f"\n  Empirical density troughs (band boundaries):")
    print(f"    {np.round(troughs, 2)} Hz")
    if len(troughs) > 1:
        ratios = troughs[1:] / troughs[:-1]
        print(f"    Consecutive ratios: {np.round(ratios, 3)}")
        print(f"    Geometric mean: {np.exp(np.mean(np.log(ratios))):.4f}")

    print(f"\n  Empirical density peaks (band centers):")
    print(f"    {np.round(centers, 2)} Hz")
    if len(centers) > 1:
        ratios_c = centers[1:] / centers[:-1]
        print(f"    Consecutive ratios: {np.round(ratios_c, 3)}")
        print(f"    Geometric mean: {np.exp(np.mean(np.log(ratios_c))):.4f}")

    phi_bnds = np.array([F0 / PHI, F0, F0 * PHI, F0 * PHI ** 2, F0 * PHI ** 3])
    print(f"\n  φ-lattice boundaries for comparison:")
    print(f"    {np.round(phi_bnds, 2)} Hz")

    if len(troughs) > 0:
        # Distance from each empirical trough to nearest phi boundary
        for t in troughs:
            nearest = phi_bnds[np.argmin(np.abs(np.log(t) - np.log(phi_bnds)))]
            pct_off = (t - nearest) / nearest * 100
            print(f"    {t:.2f} Hz ↔ φ={nearest:.2f} Hz ({pct_off:+.1f}%)")

    # Test 2: Geometric series model comparison
    model_df = test_geometric_models(centers, troughs)
    model_df.to_csv(os.path.join(OUT_DIR, 'model_comparison.csv'), index=False)

    # Test 3: Log vs linear density
    density_results = test_log_vs_linear_density(all_freqs)

    # Test 4: Per-dataset replication
    ratio_df = test_per_dataset_ratios(all_data)
    ratio_df.to_csv(os.path.join(OUT_DIR, 'per_dataset_ratios.csv'), index=False)

    # Test 5: KDE bandwidth stability
    bw_df, stable_troughs, all_troughs_list, bw_values = test_bandwidth_stability(
        all_freqs)
    bw_df.to_csv(os.path.join(OUT_DIR, 'bandwidth_stability.csv'), index=False)

    # Test 6: Aperiodic-only null
    null_results = test_aperiodic_null(all_freqs, n_surrogates=200)

    elapsed = time.time() - t0
    print(f"\n  All tests completed in {elapsed:.1f}s")
    print(f"  Results saved to {OUT_DIR}/")

    if args.plot:
        generate_plots(all_freqs, all_data, centers, troughs,
                       model_df, density_results)
        plot_bandwidth_stability(bw_df, stable_troughs, all_troughs_list, bw_values)
        plot_aperiodic_null(null_results)


if __name__ == '__main__':
    main()
