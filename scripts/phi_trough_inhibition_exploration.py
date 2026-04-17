#!/usr/bin/env python3
"""
φ-Trough Inhibition Exploration
================================

The spectral differentiation paper found troughs at φ-spaced frequencies.
This script explores the hypothesis that inhibitory circuits carve these
troughs specifically at φ-spacing because φ is the most irrational number,
maximizing desynchronization between adjacent bands.

Analyses:
  1. Per-subject trough ratios: does individual variation in ratio proximity
     to φ predict trough depth, cognition, or psychopathology?
  2. Trough depth covariance: do troughs deepen together (shared inhibitory
     mechanism) or independently?
  3. Ratio-depth relationship: do troughs at ratios closer to φ have
     different depth properties?
  4. The bridge as inhibitory failure: does bridge enrichment at 20 Hz
     scale with depth of flanking troughs?
  5. GABA signature: the α/β trough dissociates externalizing (+) from
     internalizing (-). Is this specific to the deepest-carving trough?

Usage:
    python scripts/phi_trough_inhibition_exploration.py

Outputs to: outputs/phi_trough_inhibition/
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'phi_trough_inhibition')

PHI = (1 + np.sqrt(5)) / 2

PER_SUBJECT_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'per_subject_trough_depths.csv')
AGE_BIN_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_depth_by_age.csv')
PSYCHOPATH_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_depth_psychopathology.csv')
COGNITION_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_depth_cognition.csv')
SHAPE_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_shapes.csv')

# Trough positions (bootstrap medians)
TROUGH_HZ = np.array([5.0274, 7.8227, 13.5949, 24.7516, 34.3834])
TROUGH_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']
TROUGH_DEPTH_COLS = ['depth_δ/θ', 'depth_θ/α', 'depth_α/β', 'depth_βL/βH', 'depth_βH/γ']

# Consecutive ratios
RATIOS = TROUGH_HZ[1:] / TROUGH_HZ[:-1]
RATIO_LABELS = ['T2/T1', 'T3/T2', 'T4/T3', 'T5/T4']


def analysis_1_per_subject_phi_proximity():
    """Do individuals with more φ-like trough ratios have deeper troughs?"""
    print("\n" + "=" * 70)
    print("Analysis 1: Per-Subject Trough Ratios and φ-Proximity")
    print("=" * 70)

    df = pd.read_csv(PER_SUBJECT_PATH)
    df = df.dropna(subset=['age'])
    print(f"  N = {len(df)} subjects")

    # depth_ratio < 1 means trough is present (depleted)
    # depth_ratio > 1 means no trough (enriched)
    # We need trough POSITIONS per subject to compute ratios.
    # But we only have trough DEPTHS at fixed positions.
    # The trough positions are fixed from pooled KDE -- individual subjects
    # don't have individual trough positions.
    #
    # What we CAN do: compute a composite "trough depth score" -- how much
    # overall inhibitory carving does this subject show?

    depth_cols = TROUGH_DEPTH_COLS

    # Mean trough depth across all 5 boundaries (lower = deeper troughs)
    valid = df.dropna(subset=depth_cols)
    valid = valid.copy()
    valid['mean_depth'] = valid[depth_cols].mean(axis=1)
    valid['mean_depletion'] = (1 - valid['mean_depth']) * 100  # % depletion

    # Coefficient of variation across troughs -- do some people have uniform
    # trough depth vs. highly variable?
    valid['depth_sd'] = valid[depth_cols].std(axis=1)
    valid['depth_cv'] = valid['depth_sd'] / valid['mean_depth'].clip(lower=0.01)

    print(f"\n  Mean trough depth ratio: {valid['mean_depth'].mean():.3f} ± {valid['mean_depth'].std():.3f}")
    print(f"  Mean depletion: {valid['mean_depletion'].mean():.1f}% ± {valid['mean_depletion'].std():.1f}%")

    # Does overall trough depth correlate with age?
    rho_age, p_age = stats.spearmanr(valid['age'], valid['mean_depletion'])
    print(f"\n  Mean depletion vs age: ρ = {rho_age:+.3f}, p = {p_age:.4f}")

    # Does trough depth variability correlate with age?
    rho_cv, p_cv = stats.spearmanr(valid['age'], valid['depth_cv'])
    print(f"  Depth CV vs age: ρ = {rho_cv:+.3f}, p = {p_cv:.4f}")

    # Per-trough depth correlation with mean (which troughs drive the composite?)
    print("\n  Per-trough correlation with mean depletion:")
    for col, label in zip(depth_cols, TROUGH_LABELS):
        depletion = (1 - valid[col]) * 100
        rho, p = stats.spearmanr(depletion, valid['mean_depletion'])
        print(f"    {label}: ρ = {rho:.3f} (p = {p:.1e})")

    valid.to_csv(os.path.join(OUT_DIR, 'per_subject_composite_depth.csv'), index=False)
    return valid


def analysis_2_trough_covariance(df_subjects):
    """Do troughs deepen together or independently?"""
    print("\n" + "=" * 70)
    print("Analysis 2: Trough Depth Covariance Structure")
    print("=" * 70)

    depth_cols = TROUGH_DEPTH_COLS

    # Convert to depletion (1 - depth_ratio) so higher = deeper trough
    depletion_df = pd.DataFrame()
    for col, label in zip(depth_cols, TROUGH_LABELS):
        depletion_df[label] = 1 - df_subjects[col]

    # Correlation matrix
    corr = depletion_df.corr(method='spearman')
    print("\n  Spearman correlation matrix of trough depths:")
    print(corr.round(3).to_string())

    # Age-partialed correlations
    print("\n  Age-partialed correlations (partial Spearman via residuals):")
    age = df_subjects['age'].values
    residuals = {}
    for label in TROUGH_LABELS:
        rho_age, _ = stats.spearmanr(age, depletion_df[label])
        # Rank-based residuals
        age_rank = stats.rankdata(age)
        depth_rank = stats.rankdata(depletion_df[label])
        slope, intercept, _, _, _ = stats.linregress(age_rank, depth_rank)
        residuals[label] = depth_rank - (slope * age_rank + intercept)

    resid_df = pd.DataFrame(residuals)
    corr_partial = resid_df.corr()
    print(corr_partial.round(3).to_string())

    # Key question: are adjacent troughs more correlated than distant ones?
    adjacent = []
    non_adjacent = []
    for i in range(len(TROUGH_LABELS)):
        for j in range(i + 1, len(TROUGH_LABELS)):
            r = corr_partial.iloc[i, j]
            if abs(i - j) == 1:
                adjacent.append(r)
                print(f"  Adjacent: {TROUGH_LABELS[i]}–{TROUGH_LABELS[j]}: r = {r:.3f}")
            else:
                non_adjacent.append(r)

    print(f"\n  Mean adjacent correlation: {np.mean(adjacent):.3f}")
    print(f"  Mean non-adjacent correlation: {np.mean(non_adjacent):.3f}")
    t, p = stats.mannwhitneyu(adjacent, non_adjacent, alternative='greater')
    print(f"  Adjacent > non-adjacent? U = {t:.1f}, p = {p:.4f}")

    corr_partial.to_csv(os.path.join(OUT_DIR, 'trough_covariance_age_partialed.csv'))
    return corr_partial


def analysis_3_ratio_depth_relationship():
    """Do boundaries at ratios closer to φ show different properties?"""
    print("\n" + "=" * 70)
    print("Analysis 3: Trough Ratio Distance from φ vs Trough Properties")
    print("=" * 70)

    ratios = RATIOS
    dist_from_phi = np.abs(ratios - PHI)

    # Pooled trough depletion (from paper)
    pooled_depletion = np.array([70.4, 8.7, 61.7, 11.6, 32.2])

    # Each ratio spans two troughs. The ratio T(i+1)/T(i) relates to the
    # SPACE between trough i and trough i+1. The depth of that space is
    # not directly a trough -- it's the PEAK between them.
    # But we can relate each ratio to the average depth of its flanking troughs.
    flanking_mean = [(pooled_depletion[i] + pooled_depletion[i+1]) / 2 for i in range(4)]

    print(f"\n  {'Ratio':<10} {'Value':>8} {'|r-φ|':>8} {'Left depth':>12} {'Right depth':>12} {'Flanking mean':>14}")
    print("  " + "-" * 70)
    for i, (label, r, d) in enumerate(zip(RATIO_LABELS, ratios, dist_from_phi)):
        print(f"  {label:<10} {r:>8.4f} {d:>8.4f} {pooled_depletion[i]:>11.1f}% {pooled_depletion[i+1]:>11.1f}% {flanking_mean[i]:>13.1f}%")

    # Test: does distance from φ predict anything about trough depth?
    rho_depth, p_depth = stats.spearmanr(dist_from_phi, flanking_mean)
    print(f"\n  |ratio - φ| vs flanking mean depth: ρ = {rho_depth:.3f}, p = {p_depth:.3f}")

    # Age-binned: which boundaries deepen fastest?
    if os.path.exists(AGE_BIN_PATH):
        df_age = pd.read_csv(AGE_BIN_PATH)

        print("\n  Developmental deepening rate by trough:")
        rates = []
        for label in ['δ/θ (5.1)', 'θ/α (7.8)', 'α/β (13.4)', 'βL/βH (25.3)', 'βH/γ (35.0)']:
            sub = df_age[df_age.trough_label == label]
            if len(sub) >= 3:
                rho, p = stats.spearmanr(sub['age_center'], sub['depletion_pct'])
                rates.append(rho)
                direction = '↑ deepens' if rho > 0 else '↓ fills'
                print(f"    {label}: ρ = {rho:+.3f} (p = {p:.4f}) {direction}")
            else:
                rates.append(np.nan)

        # The deepening rate for each RATIO's flanking troughs
        print("\n  Ratio flanking developmental rate:")
        for i, (label, d) in enumerate(zip(RATIO_LABELS, dist_from_phi)):
            mean_rate = (rates[i] + rates[i+1]) / 2 if not (np.isnan(rates[i]) or np.isnan(rates[i+1])) else np.nan
            print(f"    {label} (|r-φ|={d:.3f}): mean flanking ρ_age = {mean_rate:+.3f}")


def analysis_4_bridge_as_inhibitory_failure(df_subjects):
    """Does the bridge at ~20 Hz reflect inhibitory failure?"""
    print("\n" + "=" * 70)
    print("Analysis 4: Bridge at ~20 Hz as Inhibitory Failure")
    print("=" * 70)

    # The bridge is at f₀ × φ² ≈ 19.90 Hz, between α/β and βL/βH troughs.
    # If inhibition carves troughs, the bridge is where carving fails.
    # Prediction: subjects with deeper flanking troughs (α/β and βL/βH)
    # might show a LESS enriched bridge -- inhibition is stronger.
    # Or: subjects with shallower flanking troughs show MORE bridge
    # enrichment -- inhibition is weaker, peaks spill into the bridge.

    # depth_α/β = trough on one side, depth_βL/βH = trough on other side
    # Lower depth_ratio = deeper trough = stronger inhibition

    valid = df_subjects.dropna(subset=['depth_α/β', 'depth_βL/βH']).copy()

    # Mean flanking trough depth (lower = more inhibition)
    valid['flanking_inhibition'] = (valid['depth_α/β'] + valid['depth_βL/βH']) / 2

    # The "bridge" would be measured by enrichment at u=0 in beta_high
    # We don't have per-subject bridge enrichment in this CSV, but we can
    # look at it differently: the ratio of the two flanking depths tells us
    # about the asymmetry of inhibition around the bridge.

    valid['trough_asymmetry'] = valid['depth_α/β'] / valid['depth_βL/βH'].clip(lower=0.01)

    print(f"  N = {len(valid)} subjects")
    print(f"  Mean α/β depth ratio: {valid['depth_α/β'].mean():.3f}")
    print(f"  Mean βL/βH depth ratio: {valid['depth_βL/βH'].mean():.3f}")
    print(f"  Mean flanking inhibition: {valid['flanking_inhibition'].mean():.3f}")

    # Does flanking inhibition change with age?
    rho, p = stats.spearmanr(valid['age'], valid['flanking_inhibition'])
    print(f"\n  Flanking inhibition vs age: ρ = {rho:+.3f}, p = {p:.4f}")
    print(f"  (Negative = stronger inhibition with age = deeper flanking troughs)")

    # Does asymmetry change with age?
    rho_asym, p_asym = stats.spearmanr(valid['age'], valid['trough_asymmetry'])
    print(f"  Trough asymmetry vs age: ρ = {rho_asym:+.3f}, p = {p_asym:.4f}")

    # The α/β trough deepens with age while βL/βH fills.
    # This means the bridge is being carved from ONE side (alpha side)
    # but not the other (beta-high side).
    # This is consistent with PV+ maturation being band-specific.

    # Dataset split: children vs adults
    children = valid[valid.age < 18]
    adults = valid[valid.age >= 25]

    print(f"\n  Children (N={len(children)}):")
    print(f"    α/β depth: {children['depth_α/β'].mean():.3f}, βL/βH depth: {children['depth_βL/βH'].mean():.3f}")
    print(f"    Asymmetry: {children['trough_asymmetry'].mean():.3f}")

    print(f"  Adults (N={len(adults)}):")
    print(f"    α/β depth: {adults['depth_α/β'].mean():.3f}, βL/βH depth: {adults['depth_βL/βH'].mean():.3f}")
    print(f"    Asymmetry: {adults['trough_asymmetry'].mean():.3f}")

    # Key question: does the α/β trough depth predict cognitive performance?
    # This would link inhibitory carving directly to function.


def analysis_5_gaba_signature():
    """Is the externalizing/internalizing dissociation specific to α/β?"""
    print("\n" + "=" * 70)
    print("Analysis 5: GABA Signature -- Psychopathology at Each Trough")
    print("=" * 70)

    if not os.path.exists(PSYCHOPATH_PATH):
        print("  Psychopathology data not found")
        return

    df = pd.read_csv(PSYCHOPATH_PATH)
    raw = df[df.type == 'raw']

    print("\n  Raw correlations (trough depth ratio vs psychopathology):")
    print(f"  Note: depth_ratio < 1 = deeper trough. Positive ρ = shallower trough → more pathology")
    print(f"        Negative ρ = deeper trough → more pathology\n")

    print(f"  {'Variable':<16} {'δ/θ':>12} {'θ/α':>12} {'α/β':>12} {'βL/βH':>12} {'βH/γ':>12}")
    print("  " + "-" * 76)

    for var in ['externalizing', 'internalizing', 'p_factor', 'attention']:
        row_data = []
        for trough in ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']:
            sub = raw[(raw.variable == var) & (raw.trough == trough)]
            if len(sub) > 0:
                rho = sub.iloc[0]['rho']
                p = sub.iloc[0]['p']
                sig = '*' if p < 0.05 else ' '
                row_data.append(f"{rho:+.3f}{sig}")
            else:
                row_data.append("   --   ")
        print(f"  {var:<16} {'  '.join(row_data)}")

    # Age-partialed
    partialed = df[df.type == 'age_partialed']
    print(f"\n  Age-partialed correlations:")
    print(f"  {'Variable':<16} {'δ/θ':>12} {'θ/α':>12} {'α/β':>12} {'βL/βH':>12} {'βH/γ':>12}")
    print("  " + "-" * 76)

    for var in ['externalizing', 'internalizing', 'p_factor', 'attention']:
        row_data = []
        for trough in ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']:
            sub = partialed[(partialed.variable == var) & (partialed.trough == trough)]
            if len(sub) > 0:
                rho = sub.iloc[0]['rho']
                p = sub.iloc[0]['p']
                sig = '*' if p < 0.05 else ' '
                row_data.append(f"{rho:+.3f}{sig}")
            else:
                row_data.append("   --   ")
        print(f"  {var:<16} {'  '.join(row_data)}")

    # The key finding: α/β trough is the ONLY one that dissociates
    # externalizing (+0.146, p<0.001) from internalizing (-0.116, p<0.001)
    # Externalizing: shallower α/β trough → more externalizing (GABA deficit)
    # Internalizing: deeper α/β trough → more internalizing (GABA excess)
    print("\n  Key finding: α/β trough is the only boundary showing double dissociation")
    print("    Externalizing ↔ shallower α/β (GABA deficit → insufficient carving)")
    print("    Internalizing ↔ deeper α/β (GABA excess → excessive carving)")
    print("    This is the most deeply carved boundary in adults (~60% depletion)")
    print("    It's also the boundary that deepens most during PV+ maturation (ages 8-19)")


def analysis_6_cognition_at_troughs():
    """Does trough depth predict cognitive performance?"""
    print("\n" + "=" * 70)
    print("Analysis 6: Trough Depth vs Cognitive Performance (LEMON)")
    print("=" * 70)

    if not os.path.exists(COGNITION_PATH):
        print("  Cognition data not found")
        return

    df = pd.read_csv(COGNITION_PATH)
    raw = df[df.type == 'raw']

    print(f"\n  {'Test':<16} {'δ/θ':>12} {'θ/α':>12} {'α/β':>12} {'βL/βH':>12} {'βH/γ':>12}")
    print("  " + "-" * 76)

    tests = raw['test'].unique()
    for test in sorted(tests):
        row_data = []
        for trough in ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']:
            sub = raw[(raw.test == test) & (raw.trough == trough)]
            if len(sub) > 0:
                rho = sub.iloc[0]['rho']
                p = sub.iloc[0]['p']
                sig = '*' if p < 0.05 else ' '
                row_data.append(f"{rho:+.3f}{sig}")
            else:
                row_data.append("   --   ")
        print(f"  {test:<16} {'  '.join(row_data)}")

    print("\n  Notable: LPS (fluid intelligence) shows α/β +0.162* and βL/βH -0.171*")
    print("  Deeper α/β trough → higher LPS (inhibitory precision → better cognition)")
    print("  Deeper βL/βH trough → lower LPS (this trough may not be purely inhibitory)")


def analysis_7_phi_as_desynchronization():
    """Why φ? Theoretical analysis of mode-locking resistance."""
    print("\n" + "=" * 70)
    print("Analysis 7: φ as Optimal Desynchronization Ratio")
    print("=" * 70)

    # For two coupled oscillators at frequencies f1 and f2, mode-locking
    # susceptibility depends on the rationality of f2/f1.
    # Simple rationals (1/1, 2/1, 3/2, 4/3, ...) are Arnold tongue centers.
    # φ is the "most irrational" number -- farthest from all p/q.

    # Compute distance to nearest simple rational for each trough ratio
    # and for reference values
    test_ratios = {
        'T2/T1 (δ/θ → θ/α)': RATIOS[0],
        'T3/T2 (θ/α → α/β)': RATIOS[1],
        'T4/T3 (α/β → βL/βH)': RATIOS[2],
        'T5/T4 (βL/βH → βH/γ)': RATIOS[3],
        'φ': PHI,
        '3/2': 1.5,
        '2/1': 2.0,
        '5/3': 5/3,
        '√2': np.sqrt(2),
        'e/2': np.e / 2,
    }

    # Simple rationals up to denominator 8
    rationals = []
    for q in range(1, 9):
        for p in range(q, 3 * q):
            rationals.append(p / q)
    rationals = sorted(set(rationals))

    print(f"\n  {'Ratio':<30} {'Value':>8} {'Nearest p/q':>12} {'Distance':>10} {'p/q':>6}")
    print("  " + "-" * 70)

    results = []
    for name, r in test_ratios.items():
        # Find nearest simple rational
        dists = [(abs(r - pq), pq) for pq in rationals]
        min_dist, nearest = min(dists)

        # Find the fraction
        best_p, best_q = 0, 1
        for q in range(1, 9):
            p = round(r * q)
            if abs(p/q - r) < abs(best_p/best_q - r):
                best_p, best_q = p, q

        print(f"  {name:<30} {r:>8.4f} {nearest:>12.4f} {min_dist:>10.4f} {best_p}/{best_q}")
        results.append({'ratio': name, 'value': r, 'nearest_rational': nearest,
                        'distance_to_rational': min_dist, 'fraction': f'{best_p}/{best_q}'})

    # φ's continued fraction has all 1s: [1; 1, 1, 1, ...]
    # This means it's the SLOWEST to converge, maximally avoiding all rationals.
    # Any oscillator pair with ratio near φ is maximally resistant to mode-locking.

    print("\n  φ = [1; 1, 1, 1, ...] -- slowest-converging continued fraction")
    print("  For inhibitory boundary placement, φ-spacing means:")
    print("    • Adjacent bands cannot mode-lock (ratio too irrational)")
    print("    • Cross-band interference is minimized")
    print("    • Each band maintains independent phase dynamics")
    print("    • The geometric mean of trough ratios (1.617) ≈ φ (1.618)")
    print("    • Individual ratios VARY (1.39 to 1.82) but average to φ")

    # The variation around φ is itself interesting
    print("\n  Individual ratio variation around φ:")
    for label, r in zip(RATIO_LABELS, RATIOS):
        dev = r - PHI
        dev_pct = dev / PHI * 100
        print(f"    {label}: {r:.4f} ({dev:+.4f}, {dev_pct:+.1f}%)")

    geo_mean = np.exp(np.mean(np.log(RATIOS)))
    print(f"\n  Geometric mean: {geo_mean:.4f} (bootstrap CI: [1.609, 1.623])")
    print(f"  The constraint is on the PRODUCT, not individual ratios.")
    print(f"  Analogous to KAM theory: the orbit avoids ALL resonances,")
    print(f"  not just the nearest one. Global φ-scaling is the signature.")

    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, 'mode_locking_analysis.csv'), index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("φ-Trough Inhibition Exploration")
    print("Hypothesis: Inhibitory circuits carve spectral boundaries at")
    print("φ-spaced frequencies to maximize inter-band desynchronization")
    print("=" * 70)

    df_subjects = analysis_1_per_subject_phi_proximity()
    analysis_2_trough_covariance(df_subjects)
    analysis_3_ratio_depth_relationship()
    analysis_4_bridge_as_inhibitory_failure(df_subjects)
    analysis_5_gaba_signature()
    analysis_6_cognition_at_troughs()
    analysis_7_phi_as_desynchronization()

    print(f"\n\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
