#!/usr/bin/env python3
"""
Developmental Trajectory Comparison: SR-Aligned vs Non-Aligned Troughs (Analysis 5.5)
=====================================================================================

Tests whether SR-aligned troughs (T2, T3, T5) show different developmental
trajectories than the non-SR-aligned trough (T1), with T4 as the bridge.

Prediction from SR-scaffold hypothesis:
  - SR-aligned troughs (T2, T3, T5) should share an "immature → deepens" pattern
    (inhibitory circuits learning to carve avoidance zones)
  - T1 (δ/θ, no SR) should show the opposite: "over-deep in children → regresses"
    (generator-dominated, not inhibitory-carved)
  - T4 (bridge) should show weak or no developmental trend

Uses:
  - Per-subject trough depths from outputs/trough_depth_by_age/
  - Age-binned trough depths from the same directory

Usage:
    python scripts/schumann_developmental_trajectories.py

Outputs to: outputs/schumann_alignment/
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'schumann_alignment')

PER_SUBJECT_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'per_subject_trough_depths.csv')
AGE_BIN_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_depth_by_age.csv')

# Trough classification
TROUGH_CLASSIFICATION = {
    'depth_δ/θ': {'label': 'T1 (δ/θ)', 'sr_class': 'no_SR', 'mechanism': 'generator-dominated'},
    'depth_θ/α': {'label': 'T2 (θ/α)', 'sr_class': 'SR-aligned', 'mechanism': 'inhibitory-boundary'},
    'depth_α/β': {'label': 'T3 (α/β)', 'sr_class': 'SR-aligned', 'mechanism': 'inhibitory-boundary'},
    'depth_βL/βH': {'label': 'T4 (βL/βH)', 'sr_class': 'SR-aligned', 'mechanism': 'inhibitory-boundary'},
    'depth_βH/γ': {'label': 'T5 (βH/γ)', 'sr_class': 'no_SR', 'mechanism': 'unknown'},
}

AGE_LABEL_MAP = {
    'δ/θ (5.1)': 'T1 (δ/θ)',
    'θ/α (7.8)': 'T2 (θ/α)',
    'α/β (13.4)': 'T3 (α/β)',
    'βL/βH (25.3)': 'T4 (βL/βH)',
    'βH/γ (35.0)': 'T5 (βH/γ)',
}

SR_CLASS_MAP = {
    'T1 (δ/θ)': 'no_SR',
    'T2 (θ/α)': 'SR-aligned',
    'T3 (α/β)': 'SR-aligned',
    'T4 (βL/βH)': 'SR-aligned',
    'T5 (βH/γ)': 'no_SR',
}


def per_subject_analysis():
    """Per-subject trough depth vs age correlations, grouped by SR classification."""
    if not os.path.exists(PER_SUBJECT_PATH):
        print(f"  WARNING: {PER_SUBJECT_PATH} not found")
        return None

    df = pd.read_csv(PER_SUBJECT_PATH)
    print(f"  Loaded {len(df)} subjects")

    # Only use subjects with age data
    df = df.dropna(subset=['age'])
    print(f"  {len(df)} subjects with age data")

    results = []
    depth_cols = [c for c in df.columns if c.startswith('depth_')]

    for col in depth_cols:
        info = TROUGH_CLASSIFICATION.get(col)
        if info is None:
            continue

        valid = df[['age', col]].dropna()
        if len(valid) < 10:
            continue

        # Spearman correlation with age
        rho, p = stats.spearmanr(valid['age'], valid[col])

        # Also fit linear regression for slope direction
        slope, intercept, r_val, p_lin, se = stats.linregress(valid['age'], valid[col])

        # Depth ratio interpretation:
        # depth_ratio < 1 means trough is deeper (depleted)
        # depth_ratio > 1 means trough is actually a bridge (enriched)
        # Positive slope = trough FILLING IN with age (becoming shallower)
        # Negative slope = trough DEEPENING with age

        results.append({
            'trough': info['label'],
            'sr_class': info['sr_class'],
            'mechanism': info['mechanism'],
            'n_subjects': len(valid),
            'spearman_rho': rho,
            'spearman_p': p,
            'linear_slope': slope,
            'linear_r': r_val,
            'linear_p': p_lin,
            'direction': 'filling' if slope > 0 else 'deepening',
            'mean_depth': valid[col].mean(),
            'sd_depth': valid[col].std(),
        })

    df_results = pd.DataFrame(results)
    return df_results


def age_bin_analysis():
    """Analyze age-binned trough depth trajectories."""
    if not os.path.exists(AGE_BIN_PATH):
        print(f"  WARNING: {AGE_BIN_PATH} not found")
        return None

    df = pd.read_csv(AGE_BIN_PATH)

    # Add SR classification
    df['trough_id'] = df['trough_label'].map(AGE_LABEL_MAP)
    df['sr_class'] = df['trough_id'].map(SR_CLASS_MAP)

    # Compute trajectory statistics per trough
    results = []
    for trough_id in ['T1 (δ/θ)', 'T2 (θ/α)', 'T3 (α/β)', 'T4 (βL/βH)', 'T5 (βH/γ)']:
        sub = df[df.trough_id == trough_id].sort_values('age_center')
        if len(sub) < 3:
            continue

        sr_class = SR_CLASS_MAP[trough_id]
        ages = sub['age_center'].values
        depths = sub['depletion_pct'].values

        # Spearman correlation
        rho, p = stats.spearmanr(ages, depths)

        # Child vs adult comparison
        child = sub[sub.age_center <= 15]['depletion_pct']
        adult = sub[sub.age_center >= 25]['depletion_pct']

        child_mean = child.mean() if len(child) > 0 else np.nan
        adult_mean = adult.mean() if len(adult) > 0 else np.nan
        change = adult_mean - child_mean  # positive = deepening, negative = filling

        # Developmental ratio (adult/child)
        dev_ratio = adult_mean / child_mean if child_mean > 0 else np.nan

        results.append({
            'trough': trough_id,
            'sr_class': sr_class,
            'n_age_bins': len(sub),
            'child_depletion_mean': child_mean,
            'adult_depletion_mean': adult_mean,
            'change_child_to_adult': change,
            'developmental_ratio': dev_ratio,
            'direction': 'deepening' if change > 0 else 'filling',
            'spearman_rho_with_age': rho,
            'spearman_p': p,
        })

    return pd.DataFrame(results)


def test_class_differences(df_per_subj):
    """Test whether SR-aligned vs non-aligned troughs differ in developmental slope."""
    if df_per_subj is None:
        return

    print("\n--- Class Comparison ---")

    sr_aligned = df_per_subj[df_per_subj.sr_class == 'SR-aligned']
    no_sr = df_per_subj[df_per_subj.sr_class == 'no_SR']
    bridge = df_per_subj[df_per_subj.sr_class == 'bridge']

    print(f"\n  SR-aligned troughs (T2, T3, T5):")
    for _, row in sr_aligned.iterrows():
        direction_symbol = '↑' if row['direction'] == 'deepening' else '↓'
        print(f"    {row['trough']}: ρ={row['spearman_rho']:+.3f} (p={row['spearman_p']:.4f}) "
              f"{direction_symbol} {row['direction']} with age")

    print(f"\n  No-SR trough (T1):")
    for _, row in no_sr.iterrows():
        direction_symbol = '↑' if row['direction'] == 'deepening' else '↓'
        print(f"    {row['trough']}: ρ={row['spearman_rho']:+.3f} (p={row['spearman_p']:.4f}) "
              f"{direction_symbol} {row['direction']} with age")

    print(f"\n  Bridge (T4):")
    for _, row in bridge.iterrows():
        direction_symbol = '↑' if row['direction'] == 'deepening' else '↓'
        print(f"    {row['trough']}: ρ={row['spearman_rho']:+.3f} (p={row['spearman_p']:.4f}) "
              f"{direction_symbol} {row['direction']} with age")

    # Test: do all SR-aligned troughs share the same sign of age correlation?
    sr_slopes = sr_aligned['spearman_rho'].values
    all_same_sign = np.all(sr_slopes > 0) or np.all(sr_slopes < 0)
    print(f"\n  All SR-aligned troughs same direction? {'YES' if all_same_sign else 'NO'}")

    if len(no_sr) > 0 and len(sr_aligned) > 0:
        no_sr_rho = no_sr['spearman_rho'].values[0]
        sr_mean_rho = sr_aligned['spearman_rho'].mean()
        print(f"  T1 (no SR) age-ρ: {no_sr_rho:+.3f}")
        print(f"  SR-aligned mean age-ρ: {sr_mean_rho:+.3f}")
        print(f"  Opposite directions? {'YES' if np.sign(no_sr_rho) != np.sign(sr_mean_rho) else 'NO'}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Developmental Trajectory: SR-Aligned vs Non-Aligned Troughs")
    print("=" * 70)

    # --- Per-subject analysis ---
    print("\n--- Per-Subject Trough Depth vs Age ---")
    df_per_subj = per_subject_analysis()

    if df_per_subj is not None:
        print(f"\n  {'Trough':<15} {'SR Class':<12} {'ρ':>8} {'p':>10} {'Direction':<12} {'N':>6}")
        print("  " + "-" * 70)
        for _, row in df_per_subj.iterrows():
            print(f"  {row['trough']:<15} {row['sr_class']:<12} {row['spearman_rho']:>+8.3f} "
                  f"{row['spearman_p']:>10.4f} {row['direction']:<12} {row['n_subjects']:>6}")

        df_per_subj.to_csv(os.path.join(OUT_DIR, 'developmental_per_subject.csv'), index=False)
        test_class_differences(df_per_subj)

    # --- Age-bin analysis ---
    print("\n\n--- Age-Binned Trough Depth Trajectories ---")
    df_age_bin = age_bin_analysis()

    if df_age_bin is not None:
        print(f"\n  {'Trough':<15} {'SR Class':<12} {'Child':>8} {'Adult':>8} {'Change':>8} {'Direction':<10}")
        print("  " + "-" * 70)
        for _, row in df_age_bin.iterrows():
            print(f"  {row['trough']:<15} {row['sr_class']:<12} "
                  f"{row['child_depletion_mean']:>7.1f}% {row['adult_depletion_mean']:>7.1f}% "
                  f"{row['change_child_to_adult']:>+7.1f}% {row['direction']:<10}")

        df_age_bin.to_csv(os.path.join(OUT_DIR, 'developmental_age_bins.csv'), index=False)

        # Prediction check
        print("\n--- Prediction Check ---")
        print("  SR-scaffold predicts:")
        print("    SR-aligned (T2,T3,T4): immature → deepens with age")
        print("    Non-aligned (T1,T5):   over-deep → fills with age or no clear trend")

        for _, row in df_age_bin.iterrows():
            predicted = {
                'SR-aligned': 'deepening',
                'no_SR': 'filling',
            }.get(row['sr_class'], '?')
            observed = row['direction']
            match = (predicted == observed)
            print(f"    {row['trough']}: predicted={predicted}, observed={observed} "
                  f"{'✓' if match else '✗'}")

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
