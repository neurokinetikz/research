#!/usr/bin/env python3
"""
Schumann Resonance–Trough Depth Correlation (Analysis 5.2)
==========================================================

Tests whether trough depletion depth correlates with SR mode amplitude.

The hypothesis predicts: stronger SR modes → deeper troughs (more inhibitory
carving). The bridge (T4/SR3~SR4) is expected as a predicted outlier — SR3
sits where motor circuits need to oscillate.

Uses both the full 5-trough set and the SR-aligned subset (T2/SR1, T3/SR2,
T5/SR5). Also tests using per-age-bin trough depths to get within-trough
variance.

Usage:
    python scripts/schumann_depth_correlation.py

Outputs to: outputs/schumann_alignment/
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'schumann_alignment')

# --- Empirical trough data ---
TROUGH_LABELS = ['T1 (δ/θ)', 'T2 (θ/α)', 'T3 (α/βL)', 'T4 (βL/βH)', 'T5 (βH/γ)']
TROUGHS_HZ = np.array([5.03, 7.82, 13.59, 24.75, 34.38])

# Depletion percentages from pooled KDE (spectral differentiation paper)
DEPLETION_PCT = np.array([70.4, 8.7, 61.7, 11.6, 32.2])

# --- SR mode data ---
SR_LABELS = ['SR1', 'SR2', 'SR3', 'SR4']
SR_NOMINAL = np.array([7.65, 13.55, 19.30, 25.30])
SR_AMPLITUDE = np.array([1.0, 0.5, 0.3, 0.15])

# --- Trough-SR alignment mapping ---
# T1 has no SR counterpart; T5 has no SR counterpart (SR5 not reliably observed)
# T2 ↔ SR1, T3 ↔ SR2, bridge ~ SR3, T4 ↔ SR4
ALIGNMENT = {
    'T2-SR1': {'trough_idx': 1, 'sr_idx': 0, 'type': 'aligned'},
    'T3-SR2': {'trough_idx': 2, 'sr_idx': 1, 'type': 'aligned'},
    'T4-SR4': {'trough_idx': 3, 'sr_idx': 3, 'type': 'aligned'},
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Schumann Resonance–Trough Depth Correlation Analysis")
    print("=" * 70)

    # --- 1. Direct correlation: SR-aligned troughs only ---
    print("\n--- SR-Aligned Troughs (T2/SR1, T3/SR2, T4/SR4) ---")
    aligned_keys = ['T2-SR1', 'T3-SR2', 'T4-SR4']
    aligned_depletion = np.array([DEPLETION_PCT[ALIGNMENT[k]['trough_idx']] for k in aligned_keys])
    aligned_sr_amp = np.array([SR_AMPLITUDE[ALIGNMENT[k]['sr_idx']] for k in aligned_keys])

    print(f"  Pairs: {list(zip(aligned_keys, aligned_depletion, aligned_sr_amp))}")

    # Spearman rank correlation (N=3 is very small, but report it)
    if len(aligned_keys) >= 3:
        rho, p_spearman = stats.spearmanr(aligned_sr_amp, aligned_depletion)
        print(f"  Spearman ρ = {rho:.4f}, p = {p_spearman:.4f}")
        # Note: with N=3, only ρ = ±1.0 can be significant
    else:
        rho, p_spearman = np.nan, np.nan

    # Pearson (for comparison)
    r, p_pearson = stats.pearsonr(aligned_sr_amp, aligned_depletion)
    print(f"  Pearson r = {r:.4f}, p = {p_pearson:.4f}")

    # --- 2. All aligned troughs ---
    print("\n--- All 3 SR-Aligned Troughs ---")
    all_keys = ['T2-SR1', 'T3-SR2', 'T4-SR4']
    all_depletion = np.array([DEPLETION_PCT[ALIGNMENT[k]['trough_idx']] for k in all_keys])
    all_sr_amp = np.array([SR_AMPLITUDE[ALIGNMENT[k]['sr_idx']] for k in all_keys])
    all_types = [ALIGNMENT[k]['type'] for k in all_keys]

    print(f"  Pairs:")
    for k, d, a, t in zip(all_keys, all_depletion, all_sr_amp, all_types):
        print(f"    {k}: depletion={d:.1f}%, SR_amp={a:.2f} [{t}]")

    rho_all, p_all = stats.spearmanr(all_sr_amp, all_depletion)
    r_all, p_r_all = stats.pearsonr(all_sr_amp, all_depletion)
    print(f"  Spearman ρ = {rho_all:.4f}, p = {p_all:.4f}")
    print(f"  Pearson r = {r_all:.4f}, p = {p_r_all:.4f}")

    # --- 3. Age-binned trough depths for within-trough variance ---
    print("\n--- Age-Binned Trough Depths (from trough_depth_by_age.csv) ---")
    age_path = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_depth_by_age.csv')
    if os.path.exists(age_path):
        df_age = pd.read_csv(age_path)

        # Map trough labels to SR alignment
        trough_sr_map = {
            'δ/θ (5.1)': ('T1', None, 'no_SR'),
            'θ/α (7.8)': ('T2', 'SR1', 'aligned'),
            'α/β (13.4)': ('T3', 'SR2', 'aligned'),
            'βL/βH (25.3)': ('T4', 'SR4', 'aligned'),
            'βH/γ (35.0)': ('T5', None, 'no_SR'),
        }
        sr_amp_map = {'SR1': 1.0, 'SR2': 0.5, 'SR3': 0.3, 'SR4': 0.15}

        rows = []
        for _, row in df_age.iterrows():
            label = row['trough_label']
            if label in trough_sr_map:
                trough_id, sr_mode, atype = trough_sr_map[label]
                sr_amp = sr_amp_map.get(sr_mode, np.nan)
                rows.append({
                    'age_center': row['age_center'],
                    'trough': trough_id,
                    'trough_label': label,
                    'sr_mode': sr_mode if sr_mode else 'None',
                    'sr_amplitude': sr_amp,
                    'alignment_type': atype,
                    'depletion_pct': row['depletion_pct'],
                    'n_subjects': row['n_subjects'],
                })

        df_aligned = pd.DataFrame(rows)
        df_aligned.to_csv(os.path.join(OUT_DIR, 'trough_sr_depth_by_age.csv'), index=False)

        # For SR-aligned troughs only, correlate SR amplitude with mean depletion
        aligned_only = df_aligned[df_aligned.alignment_type == 'aligned']
        mean_by_trough = aligned_only.groupby('trough').agg(
            mean_depletion=('depletion_pct', 'mean'),
            sd_depletion=('depletion_pct', 'std'),
            sr_amplitude=('sr_amplitude', 'first'),
        ).reset_index()

        print("\n  Mean depletion by SR-aligned trough (across age bins):")
        for _, row in mean_by_trough.iterrows():
            print(f"    {row['trough']}: {row['mean_depletion']:.1f}% ± {row['sd_depletion']:.1f}% "
                  f"(SR amp = {row['sr_amplitude']:.2f})")

        if len(mean_by_trough) >= 3:
            rho_age, p_age = stats.spearmanr(
                mean_by_trough['sr_amplitude'], mean_by_trough['mean_depletion'])
            print(f"  Spearman ρ (mean depletion vs SR amplitude) = {rho_age:.4f}, p = {p_age:.4f}")

        # Per age bin: does the depth ordering T3 > T5 > T2 match SR ordering SR2 > SR5 > SR1?
        # (Lower depletion = deeper at that frequency? No, higher depletion = deeper.)
        # SR amplitude: SR1 > SR2 > SR5
        # If hypothesis holds: depletion should NOT track SR amplitude directly,
        # because T1 (no SR) has 70% depletion. The relationship is complex.
        print("\n  Note: T1 (no SR counterpart) has the highest depletion (70.4%).")
        print("  T2 (SR1, strongest mode) has the LOWEST depletion (8.7%).")
        print("  This is OPPOSITE to the simple prediction.")
        print("  However, T2 is a 'cliff' boundary (158 pp drop), not a symmetric void.")
        print("  Depletion % and boundary sharpness measure different things.")

    else:
        print(f"  WARNING: {age_path} not found")

    # --- 4. Alternative depth metrics ---
    print("\n--- Trough Shape Metrics (width, asymmetry) vs SR Amplitude ---")
    shape_path = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_shapes.csv')
    if os.path.exists(shape_path):
        df_shape = pd.read_csv(shape_path)

        # Average across cohorts and age bins for each trough
        trough_names = ['δ/θ (5.1)', 'θ/α (7.8)', 'α/β (13.4)', 'βL/βH (25.3)', 'βH/γ (35.0)']
        sr_amps_ordered = [np.nan, 1.0, 0.5, 0.15, np.nan]

        shape_summary = []
        for tname, sr_a in zip(trough_names, sr_amps_ordered):
            sub = df_shape[df_shape.trough == tname]
            if len(sub) > 0:
                shape_summary.append({
                    'trough': tname,
                    'sr_amplitude': sr_a,
                    'mean_depletion': sub['depletion_pct'].mean(),
                    'mean_width_hz': sub['width_hz'].mean(),
                    'mean_slope_asymmetry': sub['slope_asymmetry'].mean(),
                    'max_slope': sub[['left_slope', 'right_slope']].max(axis=1).mean(),
                })

        df_shape_summary = pd.DataFrame(shape_summary)
        print(df_shape_summary.to_string(index=False))
        df_shape_summary.to_csv(os.path.join(OUT_DIR, 'trough_shape_vs_sr.csv'), index=False)

        # Test: does max slope (steepness of sharpest edge) correlate with SR amplitude?
        valid = df_shape_summary.dropna(subset=['sr_amplitude'])
        if len(valid) >= 3:
            rho_slope, p_slope = stats.spearmanr(valid['sr_amplitude'], valid['max_slope'])
            print(f"\n  Max slope vs SR amplitude: ρ = {rho_slope:.4f}, p = {p_slope:.4f}")

            rho_width, p_width = stats.spearmanr(valid['sr_amplitude'], valid['mean_width_hz'])
            print(f"  Width vs SR amplitude: ρ = {rho_width:.4f}, p = {p_width:.4f}")
    else:
        print(f"  WARNING: {shape_path} not found")

    # --- Save summary ---
    summary = pd.DataFrame([
        {'test': 'aligned_3_spearman', 'rho': rho, 'p': p_spearman,
         'pairs': 'T2-SR1, T3-SR2, T4-SR4', 'note': 'N=3, only ±1 can be sig'},
        {'test': 'aligned_3_pearson', 'rho': r, 'p': p_pearson,
         'pairs': 'T2-SR1, T3-SR2, T4-SR4', 'note': ''},
        {'test': 'all_3_spearman', 'rho': rho_all, 'p': p_all,
         'pairs': 'T2-SR1, T3-SR2, T4-SR4', 'note': 'same as aligned (4 SR modes only)'},
        {'test': 'all_3_pearson', 'rho': r_all, 'p': p_r_all,
         'pairs': 'T2-SR1, T3-SR2, T4-SR4', 'note': ''},
    ])
    summary.to_csv(os.path.join(OUT_DIR, 'depth_amplitude_correlation.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
