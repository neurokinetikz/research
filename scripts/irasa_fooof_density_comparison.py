#!/usr/bin/env python3
"""
IRASA vs FOOOF Peak Density Comparison
=======================================

Plots and compares the log-frequency peak density histograms from
IRASA and FOOOF to understand WHY their trough structures differ.

Key questions:
  1. Is the θ/α trough present in IRASA density but below detection threshold?
  2. Do FOOOF fit-range boundaries create artifact troughs?
  3. How does peak bandwidth distribution differ (IRASA known to produce broader peaks)?
  4. Are the α/β and βH/γ troughs consistent across methods even if θ/α isn't?

Usage:
    python scripts/irasa_fooof_density_comparison.py

Outputs to: outputs/irasa_density_comparison/
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
IRASA_BASE = os.path.join(BASE_DIR, 'exports_irasa_v4')
FOOOF_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'irasa_density_comparison')

MIN_POWER_PCT = 50

EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp', 'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3', 'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
}

# FOOOF fit range boundaries (where aperiodic fits change)
FOOOF_FIT_BOUNDARIES = [4.70, 12.30, 19.90, 32.19, 52.09]


def load_all_freqs(base_dir, datasets, also_load_bandwidth=False):
    """Load all peak frequencies (and optionally bandwidths)."""
    all_freqs = []
    all_bw = []
    for name, subdir in datasets.items():
        path = os.path.join(base_dir, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        has_bw = 'bandwidth' in first.columns
        cols = ['freq']
        if has_power:
            cols.append('power')
        cols.append('phi_octave')
        if also_load_bandwidth and has_bw:
            cols.append('bandwidth')

        for f in files:
            try:
                df = pd.read_csv(f, usecols=cols)
            except Exception:
                continue
            if has_power and MIN_POWER_PCT > 0:
                filtered = []
                for octave in df['phi_octave'].unique():
                    bp = df[df.phi_octave == octave]
                    if len(bp) == 0:
                        continue
                    thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                    filtered.append(bp[bp['power'] >= thresh])
                if filtered:
                    df = pd.concat(filtered, ignore_index=True)
                else:
                    continue
            all_freqs.extend(df['freq'].values)
            if also_load_bandwidth and has_bw:
                all_bw.extend(df['bandwidth'].values)

        print(f"  {name}: loaded")

    if also_load_bandwidth:
        return np.array(all_freqs), np.array(all_bw)
    return np.array(all_freqs)


def compute_density(freqs, n_hist=2000, sigma=8, f_range=(3, 55)):
    """Compute smoothed density in log-frequency space."""
    log_freqs = np.log(freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)

    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)

    # Normalize to density
    smoothed = smoothed / smoothed.sum()

    return hz_centers, smoothed, counts


def find_troughs_detailed(hz_centers, smoothed):
    """Find troughs with detailed properties."""
    median_val = np.median(smoothed[smoothed > 0])
    trough_idx, props = find_peaks(-smoothed, prominence=median_val * 0.08,
                                    distance=len(hz_centers) // 25)
    return hz_centers[trough_idx], smoothed[trough_idx], trough_idx, props


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("IRASA vs FOOOF Peak Density Comparison")
    print("=" * 70)

    # Load both
    print("\n--- Loading FOOOF peaks ---")
    fooof_freqs, fooof_bw = load_all_freqs(FOOOF_BASE, EC_DATASETS, also_load_bandwidth=True)
    print(f"  Total: {len(fooof_freqs):,} peaks")

    print("\n--- Loading IRASA peaks ---")
    irasa_freqs, irasa_bw = load_all_freqs(IRASA_BASE, EC_DATASETS, also_load_bandwidth=True)
    print(f"  Total: {len(irasa_freqs):,} peaks")

    # --- 1. Density comparison ---
    print("\n--- Density Comparison ---")
    hz, fooof_density, fooof_counts = compute_density(fooof_freqs)
    _, irasa_density, irasa_counts = compute_density(irasa_freqs)

    # Find troughs in each
    fooof_trough_hz, fooof_trough_d, fooof_trough_idx, _ = find_troughs_detailed(hz, fooof_density)
    irasa_trough_hz, irasa_trough_d, irasa_trough_idx, _ = find_troughs_detailed(hz, irasa_density)

    fooof_in_range = fooof_trough_hz[(fooof_trough_hz > 4) & (fooof_trough_hz < 50)]
    irasa_in_range = irasa_trough_hz[(irasa_trough_hz > 4) & (irasa_trough_hz < 50)]

    print(f"\n  FOOOF troughs: {np.round(fooof_in_range, 2)} Hz")
    print(f"  IRASA troughs: {np.round(irasa_in_range, 2)} Hz")

    # --- 2. Zoom into θ/α region ---
    print("\n--- θ/α Region (6-10 Hz) Detail ---")
    mask_ta = (hz >= 6) & (hz <= 10)
    hz_ta = hz[mask_ta]

    print(f"\n  Hz       FOOOF density  IRASA density  Ratio")
    # Sample every ~0.25 Hz
    step = max(1, len(hz_ta) // 20)
    for i in range(0, len(hz_ta), step):
        f = hz_ta[i]
        fd = fooof_density[mask_ta][i]
        id_ = irasa_density[mask_ta][i]
        ratio = id_ / fd if fd > 0 else np.nan
        print(f"  {f:>6.2f}    {fd:.6f}       {id_:.6f}       {ratio:.3f}")

    # Is there a local minimum near 7.8 Hz in IRASA?
    mask_78 = (hz >= 7.0) & (hz <= 8.5)
    irasa_78 = irasa_density[mask_78]
    hz_78 = hz[mask_78]
    if len(irasa_78) > 0:
        min_idx = np.argmin(irasa_78)
        min_hz = hz_78[min_idx]
        min_val = irasa_78[min_idx]
        max_val = irasa_78.max()
        dip_pct = (1 - min_val / max_val) * 100
        print(f"\n  IRASA local minimum in [7.0-8.5]: {min_hz:.2f} Hz, "
              f"depth = {dip_pct:.1f}% of local max")

        # Same for FOOOF
        fooof_78 = fooof_density[mask_78]
        min_idx_f = np.argmin(fooof_78)
        min_hz_f = hz_78[min_idx_f]
        dip_pct_f = (1 - fooof_78[min_idx_f] / fooof_78.max()) * 100
        print(f"  FOOOF local minimum in [7.0-8.5]: {min_hz_f:.2f} Hz, "
              f"depth = {dip_pct_f:.1f}% of local max")

    # --- 3. Bandwidth comparison ---
    print("\n--- Bandwidth Comparison ---")
    if len(fooof_bw) > 0 and len(irasa_bw) > 0:
        # Overall
        print(f"  FOOOF bandwidth: median = {np.median(fooof_bw):.3f} Hz, "
              f"mean = {np.mean(fooof_bw):.3f} Hz")
        print(f"  IRASA bandwidth: median = {np.median(irasa_bw):.3f} Hz, "
              f"mean = {np.mean(irasa_bw):.3f} Hz")
        print(f"  Ratio (IRASA/FOOOF): {np.median(irasa_bw)/np.median(fooof_bw):.2f}×")

        # By frequency band
        bands = [(4, 7.6, 'theta'), (7.6, 12.3, 'alpha'), (12.3, 19.9, 'beta_low'),
                 (19.9, 32.2, 'beta_high'), (32.2, 52, 'gamma')]

        print(f"\n  {'Band':<12} {'FOOOF bw':>10} {'IRASA bw':>10} {'Ratio':>8}")
        print("  " + "-" * 45)
        for lo, hi, name in bands:
            f_mask = (fooof_freqs >= lo) & (fooof_freqs < hi)
            i_mask = (irasa_freqs >= lo) & (irasa_freqs < hi)
            if f_mask.sum() > 0 and i_mask.sum() > 0:
                f_bw = np.median(fooof_bw[f_mask])
                i_bw = np.median(irasa_bw[i_mask])
                print(f"  {name:<12} {f_bw:>10.3f} {i_bw:>10.3f} {i_bw/f_bw:>8.2f}×")

    # --- 4. Density at FOOOF fit boundaries ---
    print("\n--- Density at FOOOF Fit Range Boundaries ---")
    print("  If FOOOF fit boundaries create artifact troughs, density should")
    print("  dip at these frequencies in FOOOF but NOT in IRASA.\n")

    for boundary in FOOOF_FIT_BOUNDARIES:
        mask_b = (hz >= boundary * 0.95) & (hz <= boundary * 1.05)
        if mask_b.sum() > 0:
            f_min = fooof_density[mask_b].min()
            f_max = fooof_density[mask_b].max()
            i_min = irasa_density[mask_b].min()
            i_max = irasa_density[mask_b].max()
            f_dip = (1 - f_min / f_max) * 100 if f_max > 0 else 0
            i_dip = (1 - i_min / i_max) * 100 if i_max > 0 else 0

            print(f"  {boundary:.1f} Hz: FOOOF dip = {f_dip:.1f}%, IRASA dip = {i_dip:.1f}%")

    # --- 5. Peak count by narrow frequency bin ---
    print("\n--- Peak Count Ratio (IRASA/FOOOF) by Frequency ---")
    bin_edges = np.arange(4, 45, 1)
    print(f"  {'Hz range':<12} {'FOOOF':>10} {'IRASA':>10} {'Ratio':>8}")
    print("  " + "-" * 45)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        f_count = ((fooof_freqs >= lo) & (fooof_freqs < hi)).sum()
        i_count = ((irasa_freqs >= lo) & (irasa_freqs < hi)).sum()
        ratio = i_count / f_count if f_count > 0 else np.nan
        marker = ' ←' if (lo <= 7.82 < hi or lo <= 13.59 < hi or lo <= 24.75 < hi) else ''
        print(f"  [{lo:>4.0f}-{hi:>4.0f})  {f_count:>10,} {i_count:>10,} {ratio:>8.2f}{marker}")

    # Save density curves for plotting
    df_density = pd.DataFrame({
        'hz': hz,
        'fooof_density': fooof_density,
        'irasa_density': irasa_density,
    })
    df_density.to_csv(os.path.join(OUT_DIR, 'density_curves.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
