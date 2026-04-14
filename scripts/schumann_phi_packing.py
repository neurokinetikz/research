#!/usr/bin/env python3
"""
φ as Emergent Packing Ratio: Constrained Optimization (Analysis 5.6)
====================================================================

Tests whether fixing avoidance zones at SR frequencies and optimizing
band placement between them recovers φ as the geometric mean ratio.

Approach:
  1. Fix 5 avoidance zones at SR1–SR5 frequencies
  2. Between each pair of adjacent avoidance zones, place 1 band center
  3. Optimize band center positions to minimize cross-frequency coupling
     (maximize pairwise irrationality of frequency ratios)
  4. Compute the geometric mean ratio of the optimal band centers
  5. Test whether it converges to φ ≈ 1.618

The "most irrational" objective uses the noble number distance: for a
ratio r, its irrationality is measured by how far it is from any simple
rational p/q with small q. φ is the most irrational number — it has the
slowest-converging continued fraction [1; 1, 1, 1, ...].

Usage:
    python scripts/schumann_phi_packing.py
    python scripts/schumann_phi_packing.py --n-samples 10000

Outputs to: outputs/schumann_alignment/
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'schumann_alignment')

PHI = (1 + np.sqrt(5)) / 2  # 1.6180339887...

# SR avoidance zones (nominal centers)
SR_NOMINAL = np.array([7.65, 13.55, 19.30, 25.30])
SR_RANGE_LO = np.array([7.2, 12.8, 18.2, 23.6])
SR_RANGE_HI = np.array([8.1, 14.3, 20.4, 27.0])


def irrationality_measure(r):
    """
    Measure how "irrational" a ratio r is using continued fraction convergence.

    For a ratio r > 1, compute its continued fraction coefficients and return
    a score where higher = more irrational. φ maximizes this score.

    Uses the metric: sum of 1/(a_i + 1) for first K continued fraction coefficients,
    where a_i are the CF coefficients. φ = [1;1,1,1,...] gives the maximum value.
    """
    if r <= 0:
        return -np.inf

    # Compute continued fraction coefficients
    K = 20
    x = r
    cf = []
    for _ in range(K):
        a = int(np.floor(x))
        cf.append(a)
        frac = x - a
        if frac < 1e-12:
            break
        x = 1.0 / frac

    # Score: inverse of CF coefficients (smaller coefficients = more irrational)
    # φ = [1;1,1,1,...] → all a_i = 1 → maximum score
    score = sum(1.0 / (a + 1) for a in cf)
    return score


def noble_number_distance(r):
    """Distance of ratio r from the nearest noble number."""
    # Noble numbers are of the form (p + φ) / (q + φ) for integers p, q
    # The most common ones near typical frequency ratios:
    nobles = [PHI, PHI**2, 1/PHI, 2+PHI, (1+PHI)/2, (2+PHI)/3]

    return min(abs(r - n) for n in nobles)


def desynchronization_objective(band_centers, avoidance_zones):
    """
    Objective to MINIMIZE: negative of pairwise irrationality of all frequency ratios.

    Given band centers (placed between avoidance zones), compute all pairwise
    ratios and sum their irrationality scores. Also penalize proximity to
    avoidance zones.
    """
    all_freqs = np.sort(np.concatenate([band_centers, avoidance_zones]))
    n = len(all_freqs)

    total_irrationality = 0
    n_pairs = 0

    # All pairwise ratios of band centers
    for i in range(len(band_centers)):
        for j in range(i + 1, len(band_centers)):
            ratio = band_centers[j] / band_centers[i]
            total_irrationality += irrationality_measure(ratio)
            n_pairs += 1

    # Penalty for being too close to avoidance zones
    penalty = 0
    for bc in band_centers:
        for az in avoidance_zones:
            dist = abs(np.log(bc) - np.log(az))
            if dist < 0.05:  # very close in log space
                penalty += 10 * (0.05 - dist)

    return -(total_irrationality / max(n_pairs, 1)) + penalty


def optimize_band_placement(avoidance_zones, freq_range=(3, 50)):
    """
    Find optimal band center positions between avoidance zones.

    Places one band center between each pair of adjacent avoidance zones,
    plus one below the lowest and one above the highest (if within range).
    """
    # Define intervals between avoidance zones
    boundaries = np.sort(avoidance_zones)
    intervals = []

    # Below lowest SR
    intervals.append((freq_range[0], boundaries[0] * 0.95))

    # Between each pair
    for i in range(len(boundaries) - 1):
        lo = boundaries[i] * 1.05
        hi = boundaries[i+1] * 0.95
        if hi > lo:
            intervals.append((lo, hi))

    # Above highest SR
    intervals.append((boundaries[-1] * 1.05, freq_range[1]))

    n_bands = len(intervals)
    print(f"  Placing {n_bands} band centers in {n_bands} intervals")
    for i, (lo, hi) in enumerate(intervals):
        print(f"    Interval {i}: [{lo:.2f}, {hi:.2f}] Hz")

    # Initial guess: geometric center of each interval
    x0 = np.array([np.sqrt(lo * hi) for lo, hi in intervals])

    # Bounds
    bounds = [(lo, hi) for lo, hi in intervals]

    # Optimize using differential evolution (global optimizer)
    result = differential_evolution(
        lambda x: desynchronization_objective(x, avoidance_zones),
        bounds=bounds,
        seed=42,
        maxiter=200,
        tol=1e-10,
    )

    return result.x, intervals, result


def compute_ratios(band_centers):
    """Compute consecutive and geometric mean ratios of sorted band centers."""
    centers = np.sort(band_centers)
    ratios = centers[1:] / centers[:-1]
    geo_mean = np.exp(np.mean(np.log(ratios)))
    return ratios, geo_mean


def monte_carlo_packing(avoidance_zones, n_samples=10000, freq_range=(3, 50)):
    """
    Monte Carlo: sample random band placements, compute their φ-distance.
    Shows whether optimization specifically converges to φ.
    """
    boundaries = np.sort(avoidance_zones)
    intervals = []
    intervals.append((freq_range[0], boundaries[0] * 0.95))
    for i in range(len(boundaries) - 1):
        lo = boundaries[i] * 1.05
        hi = boundaries[i+1] * 0.95
        if hi > lo:
            intervals.append((lo, hi))
    intervals.append((boundaries[-1] * 1.05, freq_range[1]))

    geo_means = []
    objectives = []

    for _ in range(n_samples):
        centers = np.array([np.random.uniform(lo, hi) for lo, hi in intervals])
        _, gm = compute_ratios(centers)
        obj = desynchronization_objective(centers, avoidance_zones)
        geo_means.append(gm)
        objectives.append(obj)

    return np.array(geo_means), np.array(objectives)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("φ as Emergent Packing Ratio: SR-Constrained Band Optimization")
    print("=" * 70)

    # --- 1. Optimize with SR nominal centers ---
    print("\n--- Optimization with SR Nominal Centers ---")
    opt_centers, intervals, result = optimize_band_placement(SR_NOMINAL)
    ratios, geo_mean = compute_ratios(opt_centers)

    print(f"\n  Optimal band centers: {np.sort(opt_centers)}")
    print(f"  Consecutive ratios: {ratios}")
    print(f"  Geometric mean ratio: {geo_mean:.6f}")
    print(f"  φ = {PHI:.6f}")
    print(f"  Distance from φ: {abs(geo_mean - PHI):.6f} ({abs(geo_mean - PHI)/PHI*100:.3f}%)")

    # --- 2. Sensitivity: SR range variation ---
    print("\n--- Sensitivity: Optimization with Sampled SR Positions ---")
    n_sensitivity = 20
    geo_means_sensitivity = []

    for i in range(n_sensitivity):
        # Sample SR positions from their observed ranges
        sr_sampled = np.array([
            np.random.uniform(lo, hi)
            for lo, hi in zip(SR_RANGE_LO, SR_RANGE_HI)
        ])
        try:
            opt_c, _, _ = optimize_band_placement(sr_sampled, freq_range=(3, 50))
            _, gm = compute_ratios(opt_c)
            geo_means_sensitivity.append(gm)
            print(f"    Sample {i+1}: geo_mean = {gm:.4f}")
        except Exception:
            continue

    if geo_means_sensitivity:
        gms = np.array(geo_means_sensitivity)
        print(f"  Geometric mean ratio: {np.mean(gms):.4f} ± {np.std(gms):.4f}")
        print(f"  Range: [{np.min(gms):.4f}, {np.max(gms):.4f}]")
        print(f"  φ falls within range: {np.min(gms) <= PHI <= np.max(gms)}")
        print(f"  Median: {np.median(gms):.4f}")

    # --- 3. Monte Carlo: baseline distribution ---
    print(f"\n--- Monte Carlo Baseline ({args.n_samples:,} random placements) ---")
    mc_geo_means, mc_objectives = monte_carlo_packing(SR_NOMINAL, args.n_samples)

    print(f"  Random placement geo-mean ratios:")
    print(f"    Mean: {np.mean(mc_geo_means):.4f} ± {np.std(mc_geo_means):.4f}")
    print(f"    Median: {np.median(mc_geo_means):.4f}")
    print(f"    Range: [{np.min(mc_geo_means):.4f}, {np.max(mc_geo_means):.4f}]")

    # Where does φ fall in the random distribution?
    phi_percentile = np.mean(mc_geo_means <= PHI) * 100
    print(f"    φ percentile: {phi_percentile:.1f}th")

    # Where does the optimized value fall?
    opt_percentile = np.mean(mc_geo_means <= geo_mean) * 100
    print(f"    Optimized value percentile: {opt_percentile:.1f}th")

    # --- 4. Comparison with non-SR constraints ---
    print("\n--- Control: Optimization with Equally-Spaced Avoidance Zones ---")
    # If we use 5 equally-log-spaced avoidance zones in [7, 35] Hz
    equal_log = np.exp(np.linspace(np.log(7), np.log(35), 5))
    print(f"  Equal log-spaced zones: {equal_log}")

    opt_equal, _, _ = optimize_band_placement(equal_log)
    _, gm_equal = compute_ratios(opt_equal)
    print(f"  Optimal geo-mean ratio (equal spacing): {gm_equal:.6f}")
    print(f"  Distance from φ: {abs(gm_equal - PHI):.6f}")

    # Harmonic series avoidance zones (like musical overtones)
    harmonic = np.array([8, 16, 24, 32, 40])  # 8 Hz harmonics
    print(f"\n  Harmonic zones (8n Hz): {harmonic}")
    opt_harm, _, _ = optimize_band_placement(harmonic, freq_range=(3, 45))
    _, gm_harm = compute_ratios(opt_harm)
    print(f"  Optimal geo-mean ratio (harmonic): {gm_harm:.6f}")
    print(f"  Distance from φ: {abs(gm_harm - PHI):.6f}")

    # --- Save results ---
    summary = pd.DataFrame([{
        'avoidance_type': 'SR_nominal',
        'avoidance_zones': str(SR_NOMINAL.tolist()),
        'optimal_band_centers': str(np.sort(opt_centers).tolist()),
        'consecutive_ratios': str(ratios.tolist()),
        'geometric_mean_ratio': geo_mean,
        'distance_from_phi': abs(geo_mean - PHI),
        'phi': PHI,
    }])

    if geo_means_sensitivity:
        summary = pd.concat([summary, pd.DataFrame([{
            'avoidance_type': 'SR_sampled_mean',
            'avoidance_zones': 'sampled from ranges',
            'optimal_band_centers': '',
            'consecutive_ratios': '',
            'geometric_mean_ratio': np.mean(gms),
            'distance_from_phi': abs(np.mean(gms) - PHI),
            'phi': PHI,
        }])], ignore_index=True)

    summary = pd.concat([summary, pd.DataFrame([
        {
            'avoidance_type': 'equal_log_spaced',
            'avoidance_zones': str(equal_log.tolist()),
            'optimal_band_centers': str(np.sort(opt_equal).tolist()),
            'consecutive_ratios': '',
            'geometric_mean_ratio': gm_equal,
            'distance_from_phi': abs(gm_equal - PHI),
            'phi': PHI,
        },
        {
            'avoidance_type': 'harmonic_8n',
            'avoidance_zones': str(harmonic.tolist()),
            'optimal_band_centers': str(np.sort(opt_harm).tolist()),
            'consecutive_ratios': '',
            'geometric_mean_ratio': gm_harm,
            'distance_from_phi': abs(gm_harm - PHI),
            'phi': PHI,
        },
    ])], ignore_index=True)

    summary.to_csv(os.path.join(OUT_DIR, 'phi_packing_optimization.csv'), index=False)

    mc_df = pd.DataFrame({
        'geo_mean_ratio': mc_geo_means,
        'objective': mc_objectives,
    })
    mc_df.to_csv(os.path.join(OUT_DIR, 'phi_packing_monte_carlo.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
