#!/usr/bin/env python3
"""
Schumann Resonance–Trough Frequency Precision Analysis (Analysis 5.3)
=====================================================================

Quantifies how precisely trough positions match SR mode centers vs. ranges.
For each trough–SR pair, computes:
  (a) absolute distance from trough to SR nominal center
  (b) whether trough falls within ±1 SD of typical SR variation
  (c) z-score of trough position relative to SR range

The T2–SR1 alignment (bootstrap CI width 0.02 Hz) is the strongest candidate
for non-random correspondence.

Usage:
    python scripts/schumann_frequency_precision.py

Outputs to: outputs/schumann_alignment/
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'schumann_alignment')

# --- Empirical troughs ---
TROUGH_LABELS = ['T1 (δ/θ)', 'T2 (θ/α)', 'T3 (α/βL)', 'T4 (βL/βH)', 'T5 (βH/γ)']
TROUGHS_HZ = np.array([5.03, 7.82, 13.59, 24.75, 34.38])
TROUGH_CI_LO = np.array([4.98, 7.82, 13.40, 24.25, 34.18])
TROUGH_CI_HI = np.array([5.13, 7.85, 13.83, 26.01, 34.79])
TROUGH_CI_WIDTH = TROUGH_CI_HI - TROUGH_CI_LO

# KDE positions for comparison
TROUGHS_KDE = np.array([5.15, 7.77, 13.43, 25.29, 35.24])

# --- SR modes ---
SR_LABELS = ['SR1', 'SR2', 'SR3', 'SR4']
SR_NOMINAL = np.array([7.65, 13.55, 19.30, 25.30])
SR_RANGE_LO = np.array([7.2, 12.8, 18.2, 23.6])
SR_RANGE_HI = np.array([8.1, 14.3, 20.4, 27.0])
SR_RANGE_WIDTH = SR_RANGE_HI - SR_RANGE_LO
SR_RANGE_CENTER = (SR_RANGE_LO + SR_RANGE_HI) / 2
SR_RANGE_SD = SR_RANGE_WIDTH / 4  # approximate: range ≈ ±2 SD

# --- Trough-SR pairing ---
# T1 has no counterpart; T2↔SR1, T3↔SR2, bridge~SR3, T4↔SR4
# T5 has no counterpart (SR5 not reliably observed)
PAIRS = [
    ('T2 (θ/α)', 1, 'SR1', 0),
    ('T3 (α/βL)', 2, 'SR2', 1),
    ('T4 (βL/βH)', 3, 'SR4', 3),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Schumann Resonance–Trough Frequency Precision Analysis")
    print("=" * 70)

    results = []

    for trough_label, t_idx, sr_label, sr_idx in PAIRS:
        t_hz = TROUGHS_HZ[t_idx]
        t_ci_lo = TROUGH_CI_LO[t_idx]
        t_ci_hi = TROUGH_CI_HI[t_idx]
        t_ci_w = TROUGH_CI_WIDTH[t_idx]
        t_kde = TROUGHS_KDE[t_idx]

        sr_nom = SR_NOMINAL[sr_idx]
        sr_lo = SR_RANGE_LO[sr_idx]
        sr_hi = SR_RANGE_HI[sr_idx]
        sr_w = SR_RANGE_WIDTH[sr_idx]
        sr_ctr = SR_RANGE_CENTER[sr_idx]
        sr_sd = SR_RANGE_SD[sr_idx]

        # (a) Distance from trough to SR nominal center
        dist_to_nominal = t_hz - sr_nom
        dist_to_nominal_pct = abs(dist_to_nominal) / sr_nom * 100

        # (b) Z-score: how many SR SDs from the SR center?
        z_score = (t_hz - sr_ctr) / sr_sd if sr_sd > 0 else np.nan

        # (c) Whether trough falls within SR range
        in_range = sr_lo <= t_hz <= sr_hi

        # (d) Fractional position within SR range (0 = low edge, 1 = high edge)
        frac_pos = (t_hz - sr_lo) / sr_w if sr_w > 0 else np.nan

        # (e) Precision ratio: trough CI width / SR range width
        precision_ratio = t_ci_w / sr_w

        # (f) Same for KDE position
        dist_kde = t_kde - sr_nom
        kde_in_range = sr_lo <= t_kde <= sr_hi

        results.append({
            'trough': trough_label,
            'trough_hz': t_hz,
            'trough_ci_lo': t_ci_lo,
            'trough_ci_hi': t_ci_hi,
            'trough_ci_width': t_ci_w,
            'trough_kde_hz': t_kde,
            'sr_mode': sr_label,
            'sr_nominal_hz': sr_nom,
            'sr_range_lo': sr_lo,
            'sr_range_hi': sr_hi,
            'sr_range_width': sr_w,
            'dist_to_nominal_hz': dist_to_nominal,
            'dist_to_nominal_pct': dist_to_nominal_pct,
            'z_score_in_sr_range': z_score,
            'trough_in_sr_range': in_range,
            'fractional_position': frac_pos,
            'precision_ratio': precision_ratio,
            'kde_dist_to_nominal': dist_kde,
            'kde_in_sr_range': kde_in_range,
        })

    df = pd.DataFrame(results)

    print("\n--- Precision Summary ---")
    print(f"{'Pair':<20} {'Δf (Hz)':>10} {'Δf (%)':>10} {'z-score':>10} {'In range':>10} "
          f"{'CI width':>10} {'SR width':>10} {'Precision':>10}")
    print("-" * 100)
    for _, row in df.iterrows():
        pair = f"{row['trough']}/{row['sr_mode']}"
        print(f"{pair:<20} {row['dist_to_nominal_hz']:>+10.3f} {row['dist_to_nominal_pct']:>10.2f} "
              f"{row['z_score_in_sr_range']:>10.3f} {str(row['trough_in_sr_range']):>10} "
              f"{row['trough_ci_width']:>10.3f} {row['sr_range_width']:>10.1f} "
              f"{row['precision_ratio']:>10.4f}")

    # --- Highlight T2-SR1 ---
    t2 = df[df.sr_mode == 'SR1'].iloc[0]
    print(f"\n--- T2-SR1 Alignment Detail ---")
    print(f"  T2 bootstrap median: {t2['trough_hz']:.3f} Hz")
    print(f"  SR1 nominal center:  {t2['sr_nominal_hz']:.2f} Hz")
    print(f"  Distance:            {t2['dist_to_nominal_hz']:+.3f} Hz ({t2['dist_to_nominal_pct']:.2f}%)")
    print(f"  T2 CI width:         {t2['trough_ci_width']:.3f} Hz (most precise of all troughs)")
    print(f"  SR1 range:           [{t2['sr_range_lo']:.1f}, {t2['sr_range_hi']:.1f}] Hz")
    print(f"  Precision ratio:     {t2['precision_ratio']:.4f} (trough CI / SR range)")
    print(f"  KDE position:        {t2['trough_kde_hz']:.2f} Hz (Δ = {t2['kde_dist_to_nominal']:+.2f} Hz)")

    # --- Comparison: log-space distances ---
    print("\n--- Log-Frequency Space Distances ---")
    for _, row in df.iterrows():
        log_dist = abs(np.log(row['trough_hz']) - np.log(row['sr_nominal_hz']))
        log_sr_width = np.log(row['sr_range_hi']) - np.log(row['sr_range_lo'])
        log_frac = log_dist / log_sr_width if log_sr_width > 0 else np.nan
        print(f"  {row['trough']}/{row['sr_mode']}: "
              f"Δ(log f) = {log_dist:.4f}, SR log-width = {log_sr_width:.4f}, "
              f"fraction = {log_frac:.3f}")

    # --- Bootstrap CI overlap with SR ranges ---
    print("\n--- CI Overlap Assessment ---")
    for _, row in df.iterrows():
        ci_lo, ci_hi = row['trough_ci_lo'], row['trough_ci_hi']
        sr_lo, sr_hi = row['sr_range_lo'], row['sr_range_hi']

        overlap_lo = max(ci_lo, sr_lo)
        overlap_hi = min(ci_hi, sr_hi)

        if overlap_hi > overlap_lo:
            overlap_width = overlap_hi - overlap_lo
            ci_frac = overlap_width / (ci_hi - ci_lo) if ci_hi > ci_lo else 0
            sr_frac = overlap_width / (sr_hi - sr_lo)
            status = 'OVERLAP'
        else:
            overlap_width = 0
            ci_frac = 0
            sr_frac = 0
            gap = overlap_lo - overlap_hi
            status = f'GAP ({gap:.2f} Hz)'

        print(f"  {row['trough']}/{row['sr_mode']}: {status}")
        if overlap_width > 0:
            print(f"    Overlap: [{overlap_lo:.2f}, {overlap_hi:.2f}] ({overlap_width:.3f} Hz)")
            print(f"    CI fraction in SR: {ci_frac:.1%}, SR fraction in CI: {sr_frac:.3%}")

    # --- Save ---
    df.to_csv(os.path.join(OUT_DIR, 'frequency_precision.csv'), index=False)
    print(f"\nResults saved to {OUT_DIR}/frequency_precision.csv")


if __name__ == '__main__':
    main()
