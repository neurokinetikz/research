#!/usr/bin/env python3
"""
EC vs EO Trough Position Comparison
====================================

Tests the prediction: when alpha power drops (eyes open), the θ/α trough
should shift UPWARD toward the lattice position (~8.12 Hz), because there
is less alpha mass pulling it down.

Uses the same trough-detection pipeline as bootstrap_trough_locations.py
but applied separately to EC and EO peak data from LEMON and Dortmund.

Also compares all 5 trough positions between conditions to see which
boundaries are condition-sensitive and which are fixed.

Usage:
    python scripts/ec_eo_trough_comparison.py

Outputs to: outputs/ec_eo_troughs/
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'ec_eo_troughs')
MIN_POWER_PCT = 50

PHI = (1 + np.sqrt(5)) / 2

# Datasets with both EC and EO
PAIRED_DATASETS = {
    'lemon': {'ec': 'lemon', 'eo': 'lemon_EO'},
    'dortmund': {'ec': 'dortmund', 'eo': 'dortmund_EO_pre'},
}

# Reference trough positions (from EC pooled analysis)
EC_TROUGHS_REF = np.array([5.0274, 7.8227, 13.5949, 24.7516, 34.3834])
TROUGH_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']


def load_subjects(subdir):
    """Load per-subject peak frequencies from a dataset directory."""
    path = os.path.join(PEAK_BASE, subdir)
    files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
    if not files:
        return []

    first = pd.read_csv(files[0], nrows=1)
    has_power = 'power' in first.columns
    cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])

    subjects = []
    for f in files:
        subj_id = os.path.basename(f).replace('_peaks.csv', '')
        df = pd.read_csv(f, usecols=cols)
        if has_power and MIN_POWER_PCT > 0:
            filtered = []
            for octave in df['phi_octave'].unique():
                bp = df[df.phi_octave == octave]
                thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                filtered.append(bp[bp['power'] >= thresh])
            df = pd.concat(filtered, ignore_index=True)
        subjects.append((subj_id, df['freq'].values))

    return subjects


def find_troughs(all_freqs, n_hist=1000, sigma=8, f_range=(3, 55)):
    """Find density troughs from pooled frequencies."""
    log_freqs = np.log(all_freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)

    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)

    median_val = np.median(smoothed[smoothed > 0])
    trough_idx, properties = find_peaks(-smoothed, prominence=median_val * 0.08,
                                        distance=n_hist // 25)
    trough_hz = hz_centers[trough_idx]
    trough_depths = smoothed[trough_idx]

    # Get local maxima on either side for depth calculation
    peak_idx, _ = find_peaks(smoothed, distance=n_hist // 25)
    peak_hz = hz_centers[peak_idx]
    peak_heights = smoothed[peak_idx]

    return trough_hz, trough_depths, hz_centers, smoothed


def match_troughs(detected, reference, max_log_dist=0.3):
    """Match detected troughs to reference positions."""
    matched = {}
    for i, ref in enumerate(reference):
        if len(detected) == 0:
            continue
        log_dists = np.abs(np.log(detected) - np.log(ref))
        nearest = np.argmin(log_dists)
        if log_dists[nearest] < max_log_dist:
            matched[i] = detected[nearest]
    return matched


def bootstrap_troughs(subjects, n_boot=500, seed=42):
    """Bootstrap over subjects to get trough position CIs."""
    rng = np.random.default_rng(seed)
    freq_arrays = [s[1] for s in subjects]
    n = len(subjects)

    boot_positions = {i: [] for i in range(5)}

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_freqs = np.concatenate([freq_arrays[i] for i in idx])
        troughs, _, _, _ = find_troughs(boot_freqs)

        matched = match_troughs(troughs, EC_TROUGHS_REF)
        for ref_idx, pos in matched.items():
            boot_positions[ref_idx].append(pos)

    results = {}
    for i in range(5):
        positions = np.array(boot_positions[i])
        if len(positions) >= 10:
            results[i] = {
                'median': np.median(positions),
                'ci_lo': np.percentile(positions, 2.5),
                'ci_hi': np.percentile(positions, 97.5),
                'mean': np.mean(positions),
                'sd': np.std(positions),
                'n_detected': len(positions),
                'detection_pct': len(positions) / n_boot * 100,
            }
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("EC vs EO Trough Position Comparison")
    print("Prediction: θ/α trough shifts UP with eyes open (less alpha mass)")
    print("=" * 70)

    all_results = []

    for ds_name, paths in PAIRED_DATASETS.items():
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 70}")

        # Load EC and EO
        ec_subjects = load_subjects(paths['ec'])
        eo_subjects = load_subjects(paths['eo'])

        print(f"  EC: {len(ec_subjects)} subjects, "
              f"{sum(len(s[1]) for s in ec_subjects):,} peaks")
        print(f"  EO: {len(eo_subjects)} subjects, "
              f"{sum(len(s[1]) for s in eo_subjects):,} peaks")

        if len(ec_subjects) < 10 or len(eo_subjects) < 10:
            print("  Insufficient subjects, skipping")
            continue

        # Find paired subjects
        ec_ids = {s[0] for s in ec_subjects}
        eo_ids = {s[0] for s in eo_subjects}
        paired_ids = ec_ids & eo_ids
        print(f"  Paired subjects: {len(paired_ids)}")

        # Pooled troughs
        ec_freqs = np.concatenate([s[1] for s in ec_subjects])
        eo_freqs = np.concatenate([s[1] for s in eo_subjects])

        ec_troughs, ec_depths, ec_hz, ec_smooth = find_troughs(ec_freqs)
        eo_troughs, eo_depths, eo_hz, eo_smooth = find_troughs(eo_freqs)

        print(f"\n  Pooled EC troughs: {np.round(ec_troughs[(ec_troughs>4)&(ec_troughs<50)], 2)}")
        print(f"  Pooled EO troughs: {np.round(eo_troughs[(eo_troughs>4)&(eo_troughs<50)], 2)}")

        # Match to reference positions
        ec_matched = match_troughs(ec_troughs, EC_TROUGHS_REF)
        eo_matched = match_troughs(eo_troughs, EC_TROUGHS_REF)

        print(f"\n  {'Trough':<8} {'EC (Hz)':>10} {'EO (Hz)':>10} {'Δ (Hz)':>10} {'Δ (%)':>8} {'Direction':>10}")
        print("  " + "-" * 60)

        for i, label in enumerate(TROUGH_LABELS):
            ec_pos = ec_matched.get(i, np.nan)
            eo_pos = eo_matched.get(i, np.nan)
            if not np.isnan(ec_pos) and not np.isnan(eo_pos):
                delta = eo_pos - ec_pos
                delta_pct = delta / ec_pos * 100
                direction = '↑ EO higher' if delta > 0 else '↓ EO lower'
                print(f"  {label:<8} {ec_pos:>10.4f} {eo_pos:>10.4f} {delta:>+10.4f} {delta_pct:>+8.2f} {direction:>10}")

                all_results.append({
                    'dataset': ds_name, 'trough': label,
                    'ec_hz': ec_pos, 'eo_hz': eo_pos,
                    'delta_hz': delta, 'delta_pct': delta_pct,
                })
            else:
                ec_str = f"{ec_pos:.4f}" if not np.isnan(ec_pos) else "not found"
                eo_str = f"{eo_pos:.4f}" if not np.isnan(eo_pos) else "not found"
                print(f"  {label:<8} {ec_str:>10} {eo_str:>10}          --")

        # Bootstrap CIs for EC and EO separately
        print(f"\n  --- Bootstrap CIs (500 iterations) ---")
        print(f"  Running EC bootstrap...")
        ec_boot = bootstrap_troughs(ec_subjects, n_boot=500)
        print(f"  Running EO bootstrap...")
        eo_boot = bootstrap_troughs(eo_subjects, n_boot=500)

        print(f"\n  {'Trough':<8} {'EC median':>10} {'EC CI':>20} {'EO median':>10} {'EO CI':>20} {'Overlap?':>10}")
        print("  " + "-" * 85)

        for i, label in enumerate(TROUGH_LABELS):
            ec_b = ec_boot.get(i)
            eo_b = eo_boot.get(i)
            if ec_b and eo_b:
                ec_ci = f"[{ec_b['ci_lo']:.3f}, {ec_b['ci_hi']:.3f}]"
                eo_ci = f"[{eo_b['ci_lo']:.3f}, {eo_b['ci_hi']:.3f}]"
                # Do CIs overlap?
                overlap = ec_b['ci_lo'] <= eo_b['ci_hi'] and eo_b['ci_lo'] <= ec_b['ci_hi']
                delta = eo_b['median'] - ec_b['median']
                print(f"  {label:<8} {ec_b['median']:>10.4f} {ec_ci:>20} {eo_b['median']:>10.4f} {eo_ci:>20} "
                      f"{'yes' if overlap else 'NO':>10}")

                all_results_idx = [r for r in all_results if r['dataset'] == ds_name and r['trough'] == label]
                if all_results_idx:
                    all_results_idx[0].update({
                        'ec_ci_lo': ec_b['ci_lo'], 'ec_ci_hi': ec_b['ci_hi'],
                        'eo_ci_lo': eo_b['ci_lo'], 'eo_ci_hi': eo_b['ci_hi'],
                        'ci_overlap': overlap,
                        'ec_detection_pct': ec_b['detection_pct'],
                        'eo_detection_pct': eo_b['detection_pct'],
                    })

        # Compute consecutive ratios
        print(f"\n  --- Consecutive Trough Ratios ---")
        for condition, matched in [('EC', ec_matched), ('EO', eo_matched)]:
            positions = [matched.get(i, np.nan) for i in range(5)]
            valid = [p for p in positions if not np.isnan(p)]
            if len(valid) >= 2:
                ratios = [valid[i+1]/valid[i] for i in range(len(valid)-1)]
                geo_mean = np.exp(np.mean(np.log(ratios)))
                print(f"  {condition}: ratios = {[f'{r:.4f}' for r in ratios]}, "
                      f"geo mean = {geo_mean:.4f} (φ = {PHI:.4f})")

    # --- Summary across datasets ---
    if all_results:
        print(f"\n{'=' * 70}")
        print("SUMMARY: EC → EO Trough Shifts")
        print(f"{'=' * 70}")

        df_results = pd.DataFrame(all_results)
        df_results.to_csv(os.path.join(OUT_DIR, 'ec_eo_trough_comparison.csv'), index=False)

        # Average shifts across datasets
        for label in TROUGH_LABELS:
            sub = df_results[df_results.trough == label]
            if len(sub) > 0:
                mean_delta = sub['delta_hz'].mean()
                print(f"  {label}: mean Δ = {mean_delta:+.4f} Hz across {len(sub)} dataset(s)")

        print(f"\n  Prediction check:")
        theta_alpha = df_results[df_results.trough == 'θ/α']
        if len(theta_alpha) > 0:
            mean_shift = theta_alpha['delta_hz'].mean()
            print(f"  θ/α shift EC→EO: {mean_shift:+.4f} Hz")
            if mean_shift > 0:
                print(f"  ✓ θ/α moves UPWARD with eyes open (less alpha mass pulling down)")
                print(f"    Consistent with two-forces model")
            else:
                print(f"  ✗ θ/α moves DOWNWARD with eyes open")
                print(f"    Against two-forces model prediction")

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
