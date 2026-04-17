#!/usr/bin/env python3
"""
Final Audit Analyses
====================

1. Bridge EO slope analysis: wall signature vs depletion signature
2. Depth-width coupling: does IRASA zone width decrease as FOOOF depth increases?
3. Period concatenation under IRASA: does concat(α/β, θ/α) → δ/θ survive?
4. Effect size table for all key claims

Usage:
    python scripts/final_audit_analyses.py

Outputs to: outputs/final_audit/
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
FOOOF_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
IRASA_BASE = os.path.join(BASE_DIR, 'exports_irasa_v4')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'final_audit')

PHI = (1 + np.sqrt(5)) / 2
MIN_POWER_PCT = 50


def load_all_freqs(base_dir, datasets):
    """Load peak frequencies from multiple datasets."""
    all_freqs = []
    for name, subdir in datasets.items():
        path = os.path.join(base_dir, subdir)
        for f in sorted(glob.glob(os.path.join(path, '*_peaks.csv'))):
            try:
                df = pd.read_csv(f, usecols=['freq', 'power', 'phi_octave'])
            except Exception:
                continue
            filtered = []
            for octave in df['phi_octave'].unique():
                bp = df[df.phi_octave == octave]
                if len(bp) == 0:
                    continue
                thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                filtered.append(bp[bp['power'] >= thresh])
            if filtered:
                df = pd.concat(filtered, ignore_index=True)
                all_freqs.extend(df['freq'].values)
    return np.array(all_freqs)


def compute_density(freqs, n_hist=2000, sigma=8, f_range=(3, 55)):
    """Compute smoothed density in log-frequency space."""
    log_freqs = np.log(freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)
    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)
    return hz_centers, smoothed


def measure_trough_shape(hz, density, trough_hz, search_window=0.15):
    """Measure slope on each side of a trough."""
    log_hz = np.log(hz)
    log_trough = np.log(trough_hz)

    # Find the actual minimum near the target
    mask = np.abs(log_hz - log_trough) < search_window
    if mask.sum() == 0:
        return {}
    local_hz = hz[mask]
    local_d = density[mask]
    min_idx_local = np.argmin(local_d)
    min_hz = local_hz[min_idx_local]
    min_d = local_d[min_idx_local]

    # Find nearest peaks on each side
    global_min_idx = np.argmin(np.abs(hz - min_hz))

    # Left slope: from trough to nearest peak going left
    left_peak_d = min_d
    left_peak_hz = min_hz
    for i in range(global_min_idx - 1, max(0, global_min_idx - 200), -1):
        if density[i] > left_peak_d:
            left_peak_d = density[i]
            left_peak_hz = hz[i]
        if density[i] < left_peak_d * 0.95 and left_peak_d > min_d * 1.1:
            break

    # Right slope: from trough to nearest peak going right
    right_peak_d = min_d
    right_peak_hz = min_hz
    for i in range(global_min_idx + 1, min(len(density), global_min_idx + 200)):
        if density[i] > right_peak_d:
            right_peak_d = density[i]
            right_peak_hz = hz[i]
        if density[i] < right_peak_d * 0.95 and right_peak_d > min_d * 1.1:
            break

    left_slope = (left_peak_d - min_d) / abs(np.log(left_peak_hz) - np.log(min_hz)) if left_peak_hz != min_hz else 0
    right_slope = (right_peak_d - min_d) / abs(np.log(right_peak_hz) - np.log(min_hz)) if right_peak_hz != min_hz else 0

    return {
        'min_hz': min_hz, 'min_density': min_d,
        'left_peak_hz': left_peak_hz, 'left_peak_d': left_peak_d,
        'right_peak_hz': right_peak_hz, 'right_peak_d': right_peak_d,
        'left_slope': left_slope, 'right_slope': right_slope,
        'slope_ratio': right_slope / left_slope if left_slope > 0 else np.nan,
        'depth_left': (left_peak_d - min_d) / left_peak_d if left_peak_d > 0 else 0,
        'depth_right': (right_peak_d - min_d) / right_peak_d if right_peak_d > 0 else 0,
    }


# =============================================================
# 1. Bridge slope analysis: wall vs depletion
# =============================================================
def test_1_bridge_slope():
    print("\n" + "=" * 70)
    print("1. Bridge EO Slope Analysis: Wall Signature vs Depletion Signature")
    print("=" * 70)

    # A wall: steep slopes on BOTH sides (V-shaped, peaks accumulate on both flanks)
    # A depletion: steep on one side only (one-sided drop from motor suppression)

    ld = {'lemon': 'lemon', 'dortmund': 'dortmund'}
    eo = {'lemon_EO': 'lemon_EO', 'dortmund_EO': 'dortmund_EO_pre'}

    print("\n  Loading peaks...")
    ec_freqs = load_all_freqs(FOOOF_BASE, ld)
    eo_freqs = load_all_freqs(FOOOF_BASE, eo)

    hz_ec, d_ec = compute_density(ec_freqs)
    hz_eo, d_eo = compute_density(eo_freqs)

    # Find density minimum in 18-30 Hz for each condition
    for label, hz, density in [('EC', hz_ec, d_ec), ('EO', hz_eo, d_eo)]:
        mask = (hz >= 18) & (hz <= 30)
        local_hz = hz[mask]
        local_d = density[mask]
        min_idx = np.argmin(local_d)
        min_hz = local_hz[min_idx]

        shape = measure_trough_shape(hz, density, min_hz, search_window=0.25)

        print(f"\n  {label} density minimum at {min_hz:.2f} Hz:")
        print(f"    Left peak:  {shape.get('left_peak_hz', 0):.2f} Hz (depth = {shape.get('depth_left', 0):.1%})")
        print(f"    Right peak: {shape.get('right_peak_hz', 0):.2f} Hz (depth = {shape.get('depth_right', 0):.1%})")
        print(f"    Left slope:  {shape.get('left_slope', 0):.0f}")
        print(f"    Right slope: {shape.get('right_slope', 0):.0f}")
        print(f"    Slope ratio (R/L): {shape.get('slope_ratio', 0):.2f}")

        if shape.get('depth_left', 0) > 0.1 and shape.get('depth_right', 0) > 0.1:
            print(f"    → V-shaped: steep on BOTH sides → WALL signature")
        elif shape.get('depth_left', 0) > 0.1 or shape.get('depth_right', 0) > 0.1:
            print(f"    → One-sided: steep on one side only → DEPLETION signature")
        else:
            print(f"    → Shallow: neither side steep → PLATEAU")

    # Compare the α/β trough for reference (should be a clear wall)
    print(f"\n  For comparison, α/β trough (known wall):")
    for label, hz, density in [('EC', hz_ec, d_ec), ('EO', hz_eo, d_eo)]:
        shape = measure_trough_shape(hz, density, 13.6, search_window=0.15)
        print(f"    {label}: L slope={shape.get('left_slope',0):.0f}, R slope={shape.get('right_slope',0):.0f}, "
              f"ratio={shape.get('slope_ratio',0):.2f}, "
              f"L depth={shape.get('depth_left',0):.1%}, R depth={shape.get('depth_right',0):.1%}")


# =============================================================
# 2. Depth-width coupling
# =============================================================
def test_2_depth_width_coupling():
    print("\n" + "=" * 70)
    print("2. Depth-Width Coupling: Does IRASA Zone Width Decrease as Depth Increases?")
    print("=" * 70)

    width_df = pd.read_csv(os.path.join(BASE_DIR, 'outputs', 'sharpening_direction_tests', 'ab_zone_width_by_age.csv'))
    depth_df = pd.read_csv(os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_depth_by_age.csv'))

    # Get α/β depth per age bin
    ab_depth = depth_df[depth_df.trough_label == 'α/β (13.4)'][['age_center', 'depletion_pct']].copy()

    # Merge on age_center
    merged = width_df.merge(ab_depth, on='age_center', how='inner')

    if len(merged) < 3:
        print("  Insufficient matched age bins for correlation")
        return

    print(f"\n  {'Age':>6} {'IRASA width':>12} {'FOOOF depth':>12}")
    print("  " + "-" * 35)
    for _, row in merged.iterrows():
        print(f"  {row['age_center']:>6.0f} {row['zone_width']:>12.2f} Hz {row['depletion_pct']:>11.1f}%")

    rho, p = stats.spearmanr(merged['zone_width'], merged['depletion_pct'])
    print(f"\n  Width vs depth: ρ = {rho:+.3f} (p = {p:.3f}), N = {len(merged)}")

    if rho < -0.3:
        print(f"  → NEGATIVE: width decreases as depth increases → SHARPENING signature")
        print(f"    (Gaussian narrows as it deepens, conserving total suppression area)")
    elif rho > 0.3:
        print(f"  → POSITIVE: width increases with depth → BROADENING, not sharpening")
    else:
        print(f"  → WEAK/ABSENT: depth and width change independently")

    merged.to_csv(os.path.join(OUT_DIR, 'depth_width_coupling.csv'), index=False)


# =============================================================
# 3. Period concatenation under IRASA
# =============================================================
def test_3_irasa_period_concat():
    print("\n" + "=" * 70)
    print("3. Period Concatenation Under IRASA")
    print("=" * 70)

    # FOOOF troughs
    fooof_t = np.array([5.0274, 7.8227, 13.5949, 24.7516, 34.3834])

    # IRASA troughs (from pooled EC, 9 datasets)
    irasa_t = np.array([6.44, 11.86, 14.20, 30.61, 38.74])

    # IRASA double-trough: use midpoint of 11.86 and 14.20 as "α/β"
    irasa_ab_mid = (11.86 + 14.20) / 2  # 13.03

    # θ/α: IRASA has no clear feature here. Use 7.7 Hz from high-res raw PSD.
    raw_ta = 7.68  # from high-res PSD test

    print(f"\n  Period concatenation: concat(f1, f2) = f1*f2/(f1+f2)")
    print(f"\n  --- FOOOF ---")
    fc = fooof_t[2] * fooof_t[1] / (fooof_t[2] + fooof_t[1])
    print(f"  concat(α/β={fooof_t[2]:.2f}, θ/α={fooof_t[1]:.2f}) = {fc:.2f} Hz")
    print(f"  Observed δ/θ = {fooof_t[0]:.2f} Hz, error = {(fc-fooof_t[0])/fooof_t[0]*100:+.1f}%")

    print(f"\n  --- IRASA (using midpoint 13.03 for α/β zone) ---")
    ic_mid = irasa_ab_mid * raw_ta / (irasa_ab_mid + raw_ta)
    print(f"  concat(α/β_mid={irasa_ab_mid:.2f}, θ/α_raw={raw_ta:.2f}) = {ic_mid:.2f} Hz")
    print(f"  Observed IRASA δ/θ = {irasa_t[0]:.2f} Hz, error = {(ic_mid-irasa_t[0])/irasa_t[0]*100:+.1f}%")

    print(f"\n  --- IRASA (using lower trough 11.86 for α/β) ---")
    ic_lo = irasa_t[1] * raw_ta / (irasa_t[1] + raw_ta)
    print(f"  concat(α/β_lo={irasa_t[1]:.2f}, θ/α_raw={raw_ta:.2f}) = {ic_lo:.2f} Hz")
    print(f"  Observed IRASA δ/θ = {irasa_t[0]:.2f} Hz, error = {(ic_lo-irasa_t[0])/irasa_t[0]*100:+.1f}%")

    print(f"\n  --- IRASA (using upper trough 14.20 for α/β) ---")
    ic_hi = irasa_t[2] * raw_ta / (irasa_t[2] + raw_ta)
    print(f"  concat(α/β_hi={irasa_t[2]:.2f}, θ/α_raw={raw_ta:.2f}) = {ic_hi:.2f} Hz")
    print(f"  Observed IRASA δ/θ = {irasa_t[0]:.2f} Hz, error = {(ic_hi-irasa_t[0])/irasa_t[0]*100:+.1f}%")

    print(f"\n  Summary:")
    print(f"    FOOOF: concat → 4.97 Hz vs 5.03 (1.2% error) ✓")
    print(f"    IRASA midpoint: concat → {ic_mid:.2f} Hz vs 6.44 ({(ic_mid-6.44)/6.44*100:+.1f}%) ✗")
    print(f"    IRASA lower: concat → {ic_lo:.2f} Hz vs 6.44 ({(ic_lo-6.44)/6.44*100:+.1f}%) ✗")
    print(f"    IRASA upper: concat → {ic_hi:.2f} Hz vs 6.44 ({(ic_hi-6.44)/6.44*100:+.1f}%) ✗")
    print(f"\n  Period concatenation does NOT survive under IRASA.")
    print(f"  The IRASA δ/θ at 6.44 Hz is 1.4 Hz above FOOOF's 5.03 Hz,")
    print(f"  far from any concatenation prediction (~4.7-5.0 Hz).")


# =============================================================
# 4. Complete effect size table
# =============================================================
def test_4_effect_sizes():
    print("\n" + "=" * 70)
    print("4. Complete Effect Size Table for All Key Claims")
    print("=" * 70)

    claims = [
        ('α/β × externalizing (FOOOF)', 0.146, 902, 'FDR p=0.0002'),
        ('α/β × internalizing (FOOOF)', -0.116, 902, 'FDR p=0.005'),
        ('α/β × p-factor (FOOOF)', 0.099, 902, 'FDR p=0.016'),
        ('α/β × externalizing (IRASA)', 0.042, 860, 'NS after FDR'),
        ('α/β × internalizing (IRASA)', -0.050, 860, 'NS after FDR'),
        ('α/β × LPS fluid intel (FOOOF)', 0.162, 203, 'uncorrected p=0.021'),
        ('α/β × TMT executive (FOOOF)', -0.170, 203, 'uncorrected p<0.05'),
        ('θ/α × RWT verbal (FOOOF)', -0.171, 199, 'uncorrected p=0.016'),
        ('α/β depth × age (per-subj)', -0.419, 1732, 'p<0.0001'),
        ('βH/γ depth × age (per-subj)', -0.688, 1733, 'p<0.0001'),
        ('FOOOF-IRASA depth cross-method', 0.545, 880, 'p<1e-69'),
    ]

    print(f"\n  {'Claim':<40} {'ρ':>8} {'R²':>8} {'N':>6} {'Status':>20}")
    print("  " + "-" * 85)
    for claim, rho, n, status in claims:
        r2 = rho**2
        print(f"  {claim:<40} {rho:>+8.3f} {r2:>7.1%} {n:>6} {status:>20}")

    print(f"\n  Interpretation:")
    print(f"    Developmental effects are moderate (R² = 18-47%)")
    print(f"    Psychopathology effects are tiny (R² = 0.2-2.1%)")
    print(f"    Cognitive effects are tiny (R² = 2.6-2.9%) from single dataset")
    print(f"    Cross-method agreement is moderate (R² = 30%)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("FINAL AUDIT ANALYSES")
    print("=" * 70)

    test_1_bridge_slope()
    test_2_depth_width_coupling()
    test_3_irasa_period_concat()
    test_4_effect_sizes()

    print(f"\n\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
