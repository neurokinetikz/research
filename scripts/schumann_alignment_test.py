#!/usr/bin/env python3
"""
Schumann Resonance–EEG Trough Alignment: Formal Permutation Test (Analysis 5.1)
================================================================================

Tests whether the observed 4/5 overlap between empirical EEG density troughs
and Schumann resonance mode ranges is statistically significant vs. chance.

Null model: place 5 troughs randomly in log-frequency space [3, 50] Hz with
the constraint that they are ordered (as empirical troughs must be).

Sensitivity analysis: varies SR range widths ±20%.

Usage:
    python scripts/schumann_alignment_test.py
    python scripts/schumann_alignment_test.py --n-perm 500000

Outputs to: outputs/schumann_alignment/
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'schumann_alignment')

# --- Empirical trough positions (bootstrap medians from spectral differentiation) ---
TROUGHS_HZ = np.array([5.03, 7.82, 13.59, 24.75, 34.38])
TROUGH_LABELS = ['T1 (δ/θ)', 'T2 (θ/α)', 'T3 (α/βL)', 'T4 (βL/βH)', 'T5 (βH/γ)']
TROUGH_CI_LO = np.array([4.98, 7.82, 13.40, 24.25, 34.18])
TROUGH_CI_HI = np.array([5.13, 7.85, 13.83, 26.01, 34.79])

# KDE-based trough positions (alternative estimate)
TROUGHS_KDE = np.array([5.15, 7.77, 13.43, 25.29, 35.24])

# --- Schumann resonance modes ---
SR_NOMINAL = np.array([7.65, 13.55, 19.30, 25.30])
SR_LABELS = ['SR1', 'SR2', 'SR3', 'SR4']

# SR ranges from empirical observations
SR_RANGE_LO = np.array([7.2, 12.8, 18.2, 23.6])
SR_RANGE_HI = np.array([8.1, 14.3, 20.4, 27.0])

# Relative amplitudes (SR1 = 1.0)
SR_AMPLITUDE = np.array([1.0, 0.5, 0.3, 0.15])


def count_overlaps(troughs, sr_lo, sr_hi):
    """Count how many troughs fall within any SR range."""
    n = 0
    for t in troughs:
        if np.any((t >= sr_lo) & (t <= sr_hi)):
            n += 1
    return n


def count_overlaps_ci(ci_lo, ci_hi, sr_lo, sr_hi):
    """Count overlaps where trough CI overlaps SR range."""
    n = 0
    for lo, hi in zip(ci_lo, ci_hi):
        if np.any((lo <= sr_hi) & (hi >= sr_lo)):
            n += 1
    return n


def permutation_test(n_perm, sr_lo, sr_hi, freq_range=(3, 50), n_troughs=5):
    """
    Place n_troughs randomly in log-frequency space, count SR overlaps.
    Troughs are sorted (ordered constraint).
    """
    log_lo, log_hi = np.log(freq_range[0]), np.log(freq_range[1])
    overlap_counts = np.zeros(n_perm, dtype=int)

    for i in range(n_perm):
        log_freqs = np.sort(np.random.uniform(log_lo, log_hi, n_troughs))
        freqs = np.exp(log_freqs)
        overlap_counts[i] = count_overlaps(freqs, sr_lo, sr_hi)

    return overlap_counts


def main():
    parser = argparse.ArgumentParser(description='Schumann-trough alignment test')
    parser.add_argument('--n-perm', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 70)
    print("Schumann Resonance–EEG Trough Alignment: Formal Permutation Test")
    print("=" * 70)

    # --- Observed overlaps ---
    obs_point = count_overlaps(TROUGHS_HZ, SR_RANGE_LO, SR_RANGE_HI)
    obs_ci = count_overlaps_ci(TROUGH_CI_LO, TROUGH_CI_HI, SR_RANGE_LO, SR_RANGE_HI)
    obs_kde = count_overlaps(TROUGHS_KDE, SR_RANGE_LO, SR_RANGE_HI)

    print(f"\nObserved overlaps (bootstrap medians):  {obs_point}/5")
    print(f"Observed overlaps (CI overlap):         {obs_ci}/5")
    print(f"Observed overlaps (KDE positions):      {obs_kde}/5")

    # Detail which troughs overlap which SR modes
    print("\nDetailed overlap table:")
    for i, (label, t, ci_lo, ci_hi) in enumerate(
            zip(TROUGH_LABELS, TROUGHS_HZ, TROUGH_CI_LO, TROUGH_CI_HI)):
        matches = []
        for j, sr_label in enumerate(SR_LABELS):
            if ci_lo <= SR_RANGE_HI[j] and ci_hi >= SR_RANGE_LO[j]:
                matches.append(f"{sr_label} [{SR_RANGE_LO[j]}-{SR_RANGE_HI[j]}]")
        match_str = ', '.join(matches) if matches else 'None'
        print(f"  {label}: {t:.2f} Hz [{ci_lo:.2f}, {ci_hi:.2f}] → {match_str}")

    # --- Main permutation test ---
    print(f"\nRunning permutation test (N={args.n_perm:,})...")
    null_counts = permutation_test(args.n_perm, SR_RANGE_LO, SR_RANGE_HI)

    p_ge_obs = np.mean(null_counts >= obs_point)
    mean_null = np.mean(null_counts)
    std_null = np.std(null_counts)

    print(f"\nNull distribution: mean={mean_null:.3f}, SD={std_null:.3f}")
    print(f"P(≥{obs_point} overlaps | random) = {p_ge_obs:.6f}")

    # Distribution of overlap counts
    print("\nNull distribution of overlap counts:")
    for k in range(6):
        frac = np.mean(null_counts == k)
        print(f"  {k}/5 overlaps: {frac:.4f} ({frac*100:.2f}%)")

    # --- Sensitivity analysis: vary SR range widths ---
    print("\n--- Sensitivity Analysis: SR Range Width Variation ---")
    scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    sensitivity_results = []

    for scale in scale_factors:
        sr_centers = (SR_RANGE_LO + SR_RANGE_HI) / 2
        sr_half_widths = (SR_RANGE_HI - SR_RANGE_LO) / 2 * scale
        scaled_lo = sr_centers - sr_half_widths
        scaled_hi = sr_centers + sr_half_widths

        obs_scaled = count_overlaps(TROUGHS_HZ, scaled_lo, scaled_hi)
        null_scaled = permutation_test(args.n_perm, scaled_lo, scaled_hi)
        p_scaled = np.mean(null_scaled >= obs_scaled)

        sensitivity_results.append({
            'sr_width_scale': scale,
            'observed_overlaps': obs_scaled,
            'null_mean': np.mean(null_scaled),
            'null_sd': np.std(null_scaled),
            'p_value': p_scaled,
        })
        print(f"  Scale {scale:.1f}×: obs={obs_scaled}/5, "
              f"null={np.mean(null_scaled):.2f}±{np.std(null_scaled):.2f}, "
              f"p={p_scaled:.6f}")

    # --- Additional test: excluding T1 (which has no SR counterpart) ---
    print("\n--- Restricted Test: T2-T5 only (excluding T1/δ-θ) ---")
    troughs_restricted = TROUGHS_HZ[1:]  # T2-T5
    obs_restricted = count_overlaps(troughs_restricted, SR_RANGE_LO, SR_RANGE_HI)

    # Null: 4 random troughs in [5, 50] (above delta range)
    null_restricted = permutation_test(
        args.n_perm, SR_RANGE_LO, SR_RANGE_HI,
        freq_range=(5, 50), n_troughs=4)
    p_restricted = np.mean(null_restricted >= obs_restricted)

    print(f"  Observed: {obs_restricted}/4 overlaps")
    print(f"  P(≥{obs_restricted} | random) = {p_restricted:.6f}")

    # --- Test with φ-constrained troughs ---
    # Null: troughs constrained to φ-ratio spacing (but random starting freq)
    print("\n--- φ-Constrained Null: Troughs with φ-ratio spacing, random f₀ ---")
    PHI = 1.6175  # observed geometric mean ratio
    n_phi_perm = args.n_perm
    phi_null_counts = np.zeros(n_phi_perm, dtype=int)

    for i in range(n_phi_perm):
        # Random starting frequency in log space
        log_f0 = np.random.uniform(np.log(3), np.log(50 / PHI**4))
        f0 = np.exp(log_f0)
        phi_troughs = f0 * PHI ** np.arange(5)
        if phi_troughs[-1] <= 50:
            phi_null_counts[i] = count_overlaps(phi_troughs, SR_RANGE_LO, SR_RANGE_HI)

    p_phi = np.mean(phi_null_counts >= obs_point)
    print(f"  P(≥{obs_point} overlaps | φ-spaced random start) = {p_phi:.6f}")
    print(f"  Null mean: {np.mean(phi_null_counts):.3f}")

    # --- Save results ---
    results = {
        'test': ['all_troughs_point', 'all_troughs_ci', 'all_troughs_kde',
                 'restricted_T2_T5', 'phi_constrained'],
        'n_troughs': [5, 5, 5, 4, 5],
        'observed_overlaps': [obs_point, obs_ci, obs_kde, obs_restricted, obs_point],
        'null_mean': [mean_null, mean_null, mean_null,
                      np.mean(null_restricted), np.mean(phi_null_counts)],
        'null_sd': [std_null, std_null, std_null,
                    np.std(null_restricted), np.std(phi_null_counts)],
        'p_value': [p_ge_obs, p_ge_obs, p_ge_obs, p_restricted, p_phi],
        'n_permutations': [args.n_perm] * 5,
    }
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUT_DIR, 'alignment_permutation_test.csv'), index=False)

    df_sensitivity = pd.DataFrame(sensitivity_results)
    df_sensitivity.to_csv(os.path.join(OUT_DIR, 'alignment_sensitivity.csv'), index=False)

    # Save null distribution
    df_null = pd.DataFrame({'null_overlap_count': null_counts})
    df_null.to_csv(os.path.join(OUT_DIR, 'null_distribution.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
