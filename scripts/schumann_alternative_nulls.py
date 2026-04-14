#!/usr/bin/env python3
"""
Alternative Null: Musical/Acoustic Scaling Comparison (Analysis 5.7)
====================================================================

Tests whether the trough–SR overlap is specific to SR, or whether troughs
align equally well with other log-spaced reference systems.

Comparison systems:
  - Schumann resonance modes (SR1–SR5)
  - Equal-tempered chromatic scale (ratio 2^(1/12) ≈ 1.059)
  - Bark scale critical bands
  - ERB (Equivalent Rectangular Bandwidth) scale
  - Harmonic series (integer multiples of a fundamental)
  - Equally log-spaced reference (matched to SR count)
  - Random log-uniform reference modes

If troughs align equally well with any log-spaced system, the SR alignment
is not specific. If troughs specifically align with SR but not other systems,
the connection is non-trivial.

Usage:
    python scripts/schumann_alternative_nulls.py

Outputs to: outputs/schumann_alignment/
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'schumann_alignment')

# Empirical troughs
TROUGHS_HZ = np.array([5.03, 7.82, 13.59, 24.75, 34.38])
TROUGH_LABELS = ['T1', 'T2', 'T3', 'T4', 'T5']

# Schumann resonance
SR_NOMINAL = np.array([7.65, 13.55, 19.30, 25.30])
SR_RANGE_LO = np.array([7.2, 12.8, 18.2, 23.6])
SR_RANGE_HI = np.array([8.1, 14.3, 20.4, 27.0])
SR_RANGE_WIDTH = SR_RANGE_HI - SR_RANGE_LO


def generate_reference_systems():
    """Generate various reference frequency systems in the 3-50 Hz range."""
    systems = {}

    # 1. Schumann resonance modes (benchmark)
    systems['Schumann'] = {
        'centers': SR_NOMINAL,
        'ranges_lo': SR_RANGE_LO,
        'ranges_hi': SR_RANGE_HI,
    }

    # 2. Equal-tempered chromatic scale (A0 = 27.5 Hz standard)
    # Notes in 3-50 Hz: generated from 12-TET
    A0 = 27.5
    semitone = 2**(1/12)
    chromatic_all = A0 * semitone ** np.arange(-50, 20)
    chromatic = chromatic_all[(chromatic_all >= 5) & (chromatic_all <= 40)]
    # Take every other note to get ~5 modes similar to SR count
    # Or use all notes and define narrow ranges
    chromatic_5 = chromatic[::max(1, len(chromatic)//5)][:5]
    systems['Chromatic_5'] = {
        'centers': chromatic_5,
        'ranges_lo': chromatic_5 * semitone**(-0.5),
        'ranges_hi': chromatic_5 * semitone**(0.5),
    }

    # 3. Bark scale critical band centers (bands that overlap 3-50 Hz)
    # Bark bands 1-6 approximately:
    bark_centers = np.array([50, 100, 150, 200, 300, 400])  # These are above EEG range
    # Low-frequency Bark: band 0.5-1 ≈ 50-100 Hz. Bark doesn't reach EEG frequencies.
    # Instead, use a downscaled version — or note that Bark is not applicable.
    # Let's use the Bark formula: z = 13 arctan(0.00076f) + 3.5 arctan((f/7500)^2)
    # In the 5-40 Hz range, Bark values are essentially linear (z < 0.5)
    # This means Bark is NOT log-spaced at EEG frequencies — it's nearly linear.
    bark_freqs = np.array([5, 10, 15, 20, 30])  # ~linear spacing
    bark_width = 2.0  # Hz, approximate at these low frequencies
    systems['Bark_approx'] = {
        'centers': bark_freqs,
        'ranges_lo': bark_freqs - bark_width,
        'ranges_hi': bark_freqs + bark_width,
    }

    # 4. ERB (Equivalent Rectangular Bandwidth) scale
    # ERB = 24.7 * (4.37 * f/1000 + 1) Hz
    # At 10 Hz: ERB = 24.7 * 1.0437 = 25.8 Hz (wider than the frequency itself!)
    # ERB scale is meaningless at EEG frequencies — note this.
    # For comparison, place 5 ERB-spaced centers starting from 5 Hz
    erb_centers = [5]
    for _ in range(4):
        f = erb_centers[-1]
        erb = 24.7 * (4.37 * f / 1000 + 1)
        erb_centers.append(f + erb * 0.5)  # half-ERB spacing
    erb_centers = np.array(erb_centers)
    erb_centers = erb_centers[erb_centers <= 50]
    erb_widths = 24.7 * (4.37 * erb_centers / 1000 + 1) * 0.2  # 20% of ERB
    systems['ERB'] = {
        'centers': erb_centers[:5],
        'ranges_lo': erb_centers[:5] - erb_widths[:5],
        'ranges_hi': erb_centers[:5] + erb_widths[:5],
    }

    # 5. Harmonic series (integer multiples)
    for fund in [5, 6, 7, 8]:
        harmonics = fund * np.arange(1, 8)
        harmonics = harmonics[(harmonics >= 5) & (harmonics <= 40)][:5]
        h_width = fund * 0.1  # 10% of fundamental
        systems[f'Harmonic_{fund}Hz'] = {
            'centers': harmonics,
            'ranges_lo': harmonics - h_width,
            'ranges_hi': harmonics + h_width,
        }

    # 6. Equally log-spaced (5 modes in 5-40 Hz)
    equal_log = np.exp(np.linspace(np.log(5), np.log(40), 5))
    # Match SR-like range widths (proportional to frequency)
    el_widths = equal_log * 0.05  # 5% of center frequency
    systems['Equal_log_5pct'] = {
        'centers': equal_log,
        'ranges_lo': equal_log - el_widths,
        'ranges_hi': equal_log + el_widths,
    }

    # Wider ranges (match mean SR relative width)
    mean_sr_rel_width = np.mean(SR_RANGE_WIDTH / SR_NOMINAL)
    el_widths_sr = equal_log * mean_sr_rel_width / 2
    systems['Equal_log_SR_width'] = {
        'centers': equal_log,
        'ranges_lo': equal_log - el_widths_sr,
        'ranges_hi': equal_log + el_widths_sr,
    }

    # 7. φ-spaced modes (the brain's own lattice)
    f0 = 7.60
    phi = 1.618
    phi_modes = f0 * phi ** np.array([-1, 0, 1, 2, 3])
    phi_widths = phi_modes * 0.05
    systems['Phi_lattice'] = {
        'centers': phi_modes,
        'ranges_lo': phi_modes - phi_widths,
        'ranges_hi': phi_modes + phi_widths,
    }

    return systems


def count_overlaps(troughs, sr_lo, sr_hi):
    """Count troughs falling within any reference range."""
    n = 0
    for t in troughs:
        if np.any((t >= sr_lo) & (t <= sr_hi)):
            n += 1
    return n


def mean_min_distance(troughs, centers):
    """Mean minimum log-frequency distance from each trough to nearest reference center."""
    dists = []
    for t in troughs:
        log_dists = np.abs(np.log(t) - np.log(centers))
        dists.append(np.min(log_dists))
    return np.mean(dists)


def permutation_p_value(observed_overlaps, ref_lo, ref_hi, n_perm=100000,
                        freq_range=(3, 50), n_troughs=5):
    """P-value for observed overlap count under random placement."""
    log_lo, log_hi = np.log(freq_range[0]), np.log(freq_range[1])
    null_counts = np.zeros(n_perm, dtype=int)
    for i in range(n_perm):
        freqs = np.exp(np.sort(np.random.uniform(log_lo, log_hi, n_troughs)))
        null_counts[i] = count_overlaps(freqs, ref_lo, ref_hi)
    return np.mean(null_counts >= observed_overlaps)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    print("=" * 70)
    print("Alternative Null: Trough Alignment with Multiple Reference Systems")
    print("=" * 70)

    systems = generate_reference_systems()

    results = []

    print(f"\n{'System':<25} {'N modes':>8} {'Overlaps':>10} {'Mean Δ(log f)':>15} {'p-value':>10}")
    print("-" * 75)

    for name, sys_data in systems.items():
        centers = sys_data['centers']
        lo = sys_data['ranges_lo']
        hi = sys_data['ranges_hi']

        n_modes = len(centers)
        overlaps = count_overlaps(TROUGHS_HZ, lo, hi)
        mean_dist = mean_min_distance(TROUGHS_HZ, centers)

        # Total range coverage (what fraction of [3,50] Hz in log-space is covered)
        log_total = np.log(50) - np.log(3)
        log_coverage = sum(np.log(h) - np.log(l) for l, h in zip(lo, hi) if h > l and l > 0)
        coverage_frac = log_coverage / log_total

        # Permutation p-value (smaller N for speed on non-key comparisons)
        n_perm = 100_000 if name == 'Schumann' else 50_000
        p_val = permutation_p_value(overlaps, lo, hi, n_perm=n_perm)

        print(f"{name:<25} {n_modes:>8} {overlaps:>10}/5 {mean_dist:>15.4f} {p_val:>10.4f}")

        results.append({
            'reference_system': name,
            'n_modes': n_modes,
            'mode_centers': str(np.round(centers, 2).tolist()),
            'overlaps_with_troughs': overlaps,
            'mean_log_distance': mean_dist,
            'coverage_fraction': coverage_frac,
            'p_value': p_val,
        })

    # --- Random reference baseline ---
    print("\n--- Random Reference Modes (1000 draws of 5 modes) ---")
    n_random = 1000
    random_overlaps = []
    random_dists = []

    for _ in range(n_random):
        # Random 5 modes in [5, 40] Hz with SR-like relative widths
        random_centers = np.exp(np.sort(np.random.uniform(np.log(5), np.log(40), 5)))
        mean_rel_width = np.mean(SR_RANGE_WIDTH / SR_NOMINAL)
        random_lo = random_centers * (1 - mean_rel_width / 2)
        random_hi = random_centers * (1 + mean_rel_width / 2)

        ov = count_overlaps(TROUGHS_HZ, random_lo, random_hi)
        md = mean_min_distance(TROUGHS_HZ, random_centers)
        random_overlaps.append(ov)
        random_dists.append(md)

    random_overlaps = np.array(random_overlaps)
    random_dists = np.array(random_dists)

    # Where does SR fall in the random distribution?
    sr_overlaps = count_overlaps(TROUGHS_HZ, SR_RANGE_LO, SR_RANGE_HI)
    sr_dist = mean_min_distance(TROUGHS_HZ, SR_NOMINAL)

    sr_overlap_percentile = np.mean(random_overlaps >= sr_overlaps) * 100
    sr_dist_percentile = np.mean(random_dists <= sr_dist) * 100

    print(f"  Random overlap distribution: {np.mean(random_overlaps):.2f} ± {np.std(random_overlaps):.2f}")
    print(f"  SR overlap ({sr_overlaps}) percentile among random: {sr_overlap_percentile:.1f}%")
    print(f"  Random distance distribution: {np.mean(random_dists):.4f} ± {np.std(random_dists):.4f}")
    print(f"  SR distance ({sr_dist:.4f}) percentile among random: {sr_dist_percentile:.1f}%")

    # --- Summary ---
    print("\n--- Specificity Assessment ---")
    df = pd.DataFrame(results)
    sr_row = df[df.reference_system == 'Schumann'].iloc[0]
    better_overlap = df[df.overlaps_with_troughs >= sr_row['overlaps_with_troughs']]
    better_dist = df[df.mean_log_distance <= sr_row['mean_log_distance']]

    print(f"  Systems with ≥{sr_row['overlaps_with_troughs']} overlaps: "
          f"{len(better_overlap)}/{len(df)} ({', '.join(better_overlap['reference_system'])})")
    print(f"  Systems with ≤{sr_row['mean_log_distance']:.4f} mean distance: "
          f"{len(better_dist)}/{len(df)} ({', '.join(better_dist['reference_system'])})")

    if len(better_overlap) <= 2:
        print("  → SR alignment appears SPECIFIC (few reference systems match)")
    else:
        print("  → SR alignment may NOT be specific (multiple systems match equally well)")

    # Save
    df.to_csv(os.path.join(OUT_DIR, 'alternative_nulls_comparison.csv'), index=False)

    pd.DataFrame({
        'random_overlaps': random_overlaps,
        'random_mean_log_distance': random_dists,
    }).to_csv(os.path.join(OUT_DIR, 'random_reference_baseline.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
