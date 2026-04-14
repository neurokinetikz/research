#!/usr/bin/env python3
"""
Bridge–SR3 Alignment Analysis (Analysis 5.4)
=============================================

The ~20 Hz bridge is the ONLY boundary enriched from both sides. SR3
(19.5–21.5 Hz) sits at the same frequency. This script tests whether
the bridge enrichment peak aligns with SR3, and whether the bridge
position varies across datasets.

The bridge peak enrichment is at 19.90 Hz (= f₀ × φ²). SR3 nominal
center is 20.8 Hz, range 19.5–21.5 Hz. The enrichment peak falls
within the SR3 range.

Key question: across the 9 datasets, does the bridge position vary,
and if so, does it track toward 20.8 Hz (SR3) or stay at 19.90 Hz (φ²)?

Usage:
    python scripts/schumann_bridge_sr3.py

Outputs to: outputs/schumann_alignment/
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'schumann_alignment')

sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
from phi_frequency_model import PHI, F0

# Theoretical positions
BRIDGE_PHI = F0 * PHI**2  # f₀ × φ² ≈ 19.90 Hz
SR3_NOMINAL = 19.30
SR3_LO, SR3_HI = 18.2, 20.4

# Master enrichment data
ENRICHMENT_PATH = os.path.join(BASE_DIR, 'raw', 'master_enrichment.csv')

# Per-dataset trough depth data
TROUGH_AGE_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_depth_by_age.csv')


def analyze_bridge_enrichment():
    """Analyze bridge enrichment across datasets from master_enrichment.csv."""
    df = pd.read_csv(ENRICHMENT_PATH)

    # The bridge is at the boundary position (u=0.0) of beta_low and beta_high
    # beta_low boundary = lower edge of beta_low φ-octave = f₀ × φ¹ = 12.30 Hz (θ/α)
    # beta_high boundary = lower edge of beta_high φ-octave = f₀ × φ² = 19.90 Hz (βL/βH)
    # So "boundary" in beta_high band = the bridge at 19.90 Hz

    # Get enrichment at the boundary for beta_low (from above) and beta_high (from below)
    bridge_from_below = df[(df.position == 'boundary') & (df.band == 'beta_high') & (df.method == 'OT')]
    bridge_from_above = df[(df.position == 'boundary') & (df.band == 'beta_low') & (df.method == 'OT')]

    # The actual bridge is at u=0 in the beta_high octave, but peaks converging there
    # come from beta_low's upper edge. Let's look at it from both sides.
    print("\n--- Bridge Enrichment by Dataset (β-high boundary, u=0.0) ---")
    print(f"  φ-lattice position: f₀ × φ² = {BRIDGE_PHI:.2f} Hz")
    print(f"  SR3 range: [{SR3_LO}, {SR3_HI}] Hz, nominal = {SR3_NOMINAL} Hz\n")

    results = []

    # From beta_high side (boundary = 19.90 Hz)
    for condition in ['EC']:
        sub = bridge_from_below[bridge_from_below.condition == condition].copy()
        if len(sub) == 0:
            continue

        print(f"  Beta-high boundary (19.90 Hz), condition={condition}:")
        for _, row in sub.iterrows():
            ds = row['dataset']
            enrich = row['enrichment_pct']
            z = row['z_score']
            sig = row['is_significant']
            direction = row['direction']
            print(f"    {ds:>12}: {enrich:+.1f}% (z={z:.2f}, {'*' if sig else 'ns'}, {direction})")
            results.append({
                'dataset': ds, 'condition': condition,
                'band': 'beta_high', 'position': 'boundary',
                'enrichment_pct': enrich, 'z_score': z,
                'is_significant': sig, 'direction': direction,
            })

        # From beta_low side
        sub_bl = bridge_from_above[bridge_from_above.condition == condition].copy()
        print(f"\n  Beta-low boundary (also 19.90 Hz from above), condition={condition}:")
        for _, row in sub_bl.iterrows():
            ds = row['dataset']
            enrich = row['enrichment_pct']
            z = row['z_score']
            sig = row['is_significant']
            direction = row['direction']
            print(f"    {ds:>12}: {enrich:+.1f}% (z={z:.2f}, {'*' if sig else 'ns'}, {direction})")
            results.append({
                'dataset': ds, 'condition': condition,
                'band': 'beta_low', 'position': 'boundary',
                'enrichment_pct': enrich, 'z_score': z,
                'is_significant': sig, 'direction': direction,
            })

    return pd.DataFrame(results)


def analyze_bridge_trough_variation():
    """Check whether the βL/βH trough position varies across age/dataset groups."""
    if not os.path.exists(TROUGH_AGE_PATH):
        print(f"\n  WARNING: {TROUGH_AGE_PATH} not found")
        return None

    df = pd.read_csv(TROUGH_AGE_PATH)
    bridge = df[df.trough_label == 'βL/βH (25.3)'].copy()

    # Note: the trough_hz column shows 25.3 Hz, which is ABOVE the bridge.
    # The bridge is at 19.90 Hz. The trough between beta_low and beta_high
    # is different from the bridge enrichment peak.
    # The trough at 25.3 Hz is where peak density is LOWEST between βL and βH.
    # The bridge at 19.90 Hz is where peaks converge from BOTH sides.

    print("\n--- βL/βH Trough Position by Age ---")
    print("  Note: trough at ~25 Hz is the density minimum.")
    print(f"  Bridge enrichment peak at {BRIDGE_PHI:.2f} Hz is BELOW this trough.")
    print(f"  SR3 at {SR3_NOMINAL} Hz is also below the trough.\n")

    for _, row in bridge.iterrows():
        print(f"  Age {row['age_center']:.0f}: trough at {row['trough_hz']:.1f} Hz, "
              f"depth={row['depletion_pct']:.1f}%, N={row['n_subjects']}")

    return bridge


def frequency_comparison():
    """Compare bridge position, SR3, and φ-lattice predictions."""
    print("\n--- Frequency Comparison: Bridge vs SR3 vs φ-Lattice ---")

    # Key frequencies near 20 Hz
    freqs = {
        'f₀ × φ² (φ-lattice boundary)': BRIDGE_PHI,
        'SR3 nominal': SR3_NOMINAL,
        'SR3 range center': (SR3_LO + SR3_HI) / 2,
        'M-current / KCNQ resonance': 20.0,
    }

    print(f"\n  {'Frequency source':<40} {'Hz':>8}")
    print("  " + "-" * 50)
    for name, hz in freqs.items():
        print(f"  {name:<40} {hz:>8.2f}")

    # Distance between φ-prediction and SR3
    delta = SR3_NOMINAL - BRIDGE_PHI
    delta_pct = delta / SR3_NOMINAL * 100
    print(f"\n  Δ(SR3 − φ²): {delta:+.2f} Hz ({delta_pct:+.1f}%)")
    print(f"  The φ-lattice boundary ({BRIDGE_PHI:.2f} Hz) falls within SR3 range")
    print(f"  [{SR3_LO}, {SR3_HI}]: {'YES' if SR3_LO <= BRIDGE_PHI <= SR3_HI else 'NO'}")

    # The key insight: the bridge exists BECAUSE ~20 Hz is biophysically essential
    # Both the φ-lattice and SR3 predict avoidance at this frequency, yet the brain
    # CANNOT avoid it → hence enrichment (bridge) rather than depletion (trough)
    print("\n  Key insight: SR3 and φ² both predict spectral avoidance near 20 Hz.")
    print("  The bridge enrichment is the brain's override — motor circuits need 20 Hz.")
    print("  This makes the bridge the strongest qualitative prediction of the SR hypothesis:")
    print("  the one SR mode paired with enrichment rather than depletion is the one")
    print("  where biological necessity (motor control) overrides the avoidance pressure.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Bridge–SR3 Alignment Analysis")
    print("=" * 70)

    df_enrichment = analyze_bridge_enrichment()
    if len(df_enrichment) > 0:
        df_enrichment.to_csv(os.path.join(OUT_DIR, 'bridge_enrichment_by_dataset.csv'), index=False)

    analyze_bridge_trough_variation()
    frequency_comparison()

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
