#!/usr/bin/env python3
"""
Trough Displacement Analysis: Lattice Pull vs Peak Mass Pull
=============================================================

Each trough is positioned by two competing forces:
  1. The φ-lattice scaffold (inhibitory construction, geometric)
  2. Flanking peak masses (generators pulling the trough toward themselves)

This script quantifies:
  1. Trough displacement from ideal lattice positions
  2. Flanking peak asymmetry and its relationship to displacement
  3. Slope asymmetry as a signature of differential pull
  4. Trough precision (CI width) as a measure of regulation tightness
  5. Developmental changes: does peak mass growth shift troughs?
  6. The θ/α trough as the most regulated boundary

Usage:
    python scripts/trough_displacement_analysis.py

Outputs to: outputs/trough_displacement/
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'trough_displacement')

PHI = (1 + np.sqrt(5)) / 2

# Observed troughs
TROUGHS = np.array([5.0274, 7.8227, 13.5949, 24.7516, 34.3834])
TROUGH_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']
TROUGH_CI_LO = np.array([4.98, 7.82, 13.40, 24.25, 34.18])
TROUGH_CI_HI = np.array([5.13, 7.85, 13.83, 26.01, 34.79])
NS = np.array([-1, 0, 1, 2, 3])

SHAPE_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_shapes.csv')
AGE_BIN_PATH = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age', 'trough_depth_by_age.csv')

TROUGH_NAME_MAP = {
    'δ/θ (5.1)': 'δ/θ', 'θ/α (7.8)': 'θ/α', 'α/β (13.4)': 'α/β',
    'βL/βH (25.3)': 'βL/βH', 'βH/γ (35.0)': 'βH/γ',
}


def compute_lattice_predictions(f0):
    """Compute ideal lattice positions for given f₀."""
    return f0 * PHI ** NS


def analysis_1_displacement():
    """Quantify trough displacement from ideal lattice at various f₀."""
    print("\n" + "=" * 70)
    print("Analysis 1: Trough Displacement from φ-Lattice")
    print("=" * 70)

    # Try multiple f₀ values
    f0_candidates = {
        'f₀ = 7.60 (paper)': 7.60,
        'f₀ = 7.82 (θ/α trough)': 7.8227,
        'f₀ = 8.12 (excl. bridge)': 8.1164,
    }

    results = []

    for label, f0 in f0_candidates.items():
        predicted = compute_lattice_predictions(f0)
        displacement_hz = TROUGHS - predicted
        displacement_pct = displacement_hz / predicted * 100
        displacement_log = np.log(TROUGHS) - np.log(predicted)

        print(f"\n  {label}:")
        print(f"  {'Trough':<8} {'Observed':>10} {'Predicted':>10} {'Δ Hz':>10} {'Δ %':>8} {'Δ log':>8} {'Direction':>10}")
        print("  " + "-" * 72)
        for i, (tl, obs, pred, dhz, dpct, dlog) in enumerate(zip(
                TROUGH_LABELS, TROUGHS, predicted, displacement_hz, displacement_pct, displacement_log)):
            direction = '↑ high' if dhz > 0 else '↓ low'
            print(f"  {tl:<8} {obs:>10.4f} {pred:>10.4f} {dhz:>+10.4f} {dpct:>+8.2f} {dlog:>+8.4f} {direction:>10}")

        rmse = np.sqrt(np.mean(displacement_hz**2))
        print(f"  RMSE: {rmse:.4f} Hz")

        for i in range(5):
            results.append({
                'f0_label': label, 'f0': f0, 'trough': TROUGH_LABELS[i],
                'observed': TROUGHS[i], 'predicted': predicted[i],
                'displacement_hz': displacement_hz[i],
                'displacement_pct': displacement_pct[i],
            })

    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, 'lattice_displacement.csv'), index=False)


def analysis_2_slope_asymmetry():
    """Slope asymmetry reveals which flanking peak pulls harder."""
    print("\n" + "=" * 70)
    print("Analysis 2: Slope Asymmetry — Which Peak Pulls Harder?")
    print("=" * 70)

    if not os.path.exists(SHAPE_PATH):
        print("  trough_shapes.csv not found")
        return

    df = pd.read_csv(SHAPE_PATH)
    df['trough_short'] = df['trough'].map(TROUGH_NAME_MAP)

    # Average across all cohorts and age bins
    print("\n  Slope asymmetry: (right_slope - left_slope) / (right_slope + left_slope)")
    print("  Positive = steeper on right (trough pulled left, toward lower freq)")
    print("  Negative = steeper on left (trough pulled right, toward higher freq)\n")

    summary = df.groupby('trough_short').agg(
        mean_left_slope=('left_slope', 'mean'),
        mean_right_slope=('right_slope', 'mean'),
        mean_asymmetry=('slope_asymmetry', 'mean'),
        mean_left_peak=('left_peak_hz', 'mean'),
        mean_right_peak=('right_peak_hz', 'mean'),
        mean_depletion=('depletion_pct', 'mean'),
        mean_width=('width_hz', 'mean'),
        n_obs=('slope_asymmetry', 'count'),
    ).reindex(['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ'])

    print(f"  {'Trough':<8} {'L slope':>10} {'R slope':>10} {'Asymm':>8} {'L peak':>10} {'R peak':>10} {'Width':>8} {'Depl':>8}")
    print("  " + "-" * 80)
    for label, row in summary.iterrows():
        print(f"  {label:<8} {row['mean_left_slope']:>10.0f} {row['mean_right_slope']:>10.0f} "
              f"{row['mean_asymmetry']:>+8.3f} {row['mean_left_peak']:>10.2f} {row['mean_right_peak']:>10.2f} "
              f"{row['mean_width']:>8.2f} {row['mean_depletion']:>7.1f}%")

    # The key question: does slope asymmetry predict displacement direction?
    # Steeper right slope → trough is pulled LEFT (toward lower peak)
    # This should correlate with the trough being BELOW lattice prediction
    print("\n  Interpretation:")
    print("  • θ/α: slight positive asymmetry (+0.14) → steeper on right (alpha side)")
    print("    = trough pulled toward theta (downward) — consistent with θ/α below lattice")
    print("  • α/β: strong positive asymmetry (~+0.93 in children, decreasing with age)")
    print("    = alpha peak pulling the trough downward from lattice position")
    print("  • δ/θ: negative asymmetry → steeper on left (delta side)")
    print("    = trough pulled toward higher freq by theta")

    # Developmental trajectory of asymmetry
    print("\n  --- Slope Asymmetry Developmental Trajectory ---")
    for trough_label in ['δ/θ (5.1)', 'θ/α (7.8)', 'α/β (13.4)', 'βL/βH (25.3)', 'βH/γ (35.0)']:
        sub = df[df.trough == trough_label].sort_values('age_center')
        if len(sub) >= 3:
            rho, p = stats.spearmanr(sub['age_center'], sub['slope_asymmetry'])
            tshort = TROUGH_NAME_MAP[trough_label]
            young = sub[sub.age_center <= 10]['slope_asymmetry'].mean()
            old = sub[sub.age_center >= 40]['slope_asymmetry'].mean()
            print(f"  {tshort:<8}: ρ_age = {rho:+.3f} (p={p:.3f}), child={young:+.3f}, adult={old:+.3f}")

    summary.to_csv(os.path.join(OUT_DIR, 'slope_asymmetry_summary.csv'))
    return summary


def analysis_3_peak_mass_pull():
    """Compute flanking peak mass and test whether it predicts trough position."""
    print("\n" + "=" * 70)
    print("Analysis 3: Peak Mass Asymmetry and Trough Position")
    print("=" * 70)

    if not os.path.exists(SHAPE_PATH):
        print("  trough_shapes.csv not found")
        return

    df = pd.read_csv(SHAPE_PATH)
    df['trough_short'] = df['trough'].map(TROUGH_NAME_MAP)

    # For each trough, the flanking peak positions tell us where the
    # mass centers are. The ratio of distances (trough-to-left-peak vs
    # trough-to-right-peak) reveals the balance of forces.

    # Approximate peak mass by the slope: steeper slope = stronger peak
    # pulling from that side

    # Use the ratio of slopes as a proxy for mass ratio
    df['slope_ratio'] = df['right_slope'] / df['left_slope'].clip(lower=1)
    df['log_slope_ratio'] = np.log(df['slope_ratio'].clip(lower=0.01))

    # The displacement from lattice center (using f₀ = 8.12)
    f0 = 8.1164
    lattice = {
        'δ/θ': f0 * PHI**(-1),
        'θ/α': f0 * PHI**0,
        'α/β': f0 * PHI**1,
        'βL/βH': f0 * PHI**2,
        'βH/γ': f0 * PHI**3,
    }

    print(f"\n  Using f₀ = {f0:.4f} Hz (bridge-excluded estimate)")
    print(f"\n  {'Trough':<8} {'Lattice':>10} {'Observed':>10} {'Displ.':>10} {'R/L slope':>10} {'Pull dir':>10}")
    print("  " + "-" * 65)

    for trough_label in ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']:
        sub = df[df.trough_short == trough_label]
        if len(sub) == 0:
            continue

        lat = lattice[trough_label]
        obs = TROUGHS[TROUGH_LABELS.index(trough_label)]
        displ = obs - lat

        mean_sr = sub['slope_ratio'].mean()
        # slope_ratio > 1 means right slope steeper → stronger pull from right (higher freq)
        pull = 'right (↑)' if mean_sr > 1 else 'left (↓)'

        print(f"  {trough_label:<8} {lat:>10.2f} {obs:>10.2f} {displ:>+10.2f} {mean_sr:>10.2f} {pull:>10}")

    # Across development: as alpha power grows, does θ/α trough shift?
    print("\n  --- θ/α Trough: Does Alpha Growth Shift It? ---")
    theta_alpha = df[df.trough_short == 'θ/α'].sort_values('age_center')
    if len(theta_alpha) >= 3:
        print(f"  {'Age':>6} {'R slope':>10} {'L slope':>10} {'Ratio':>8} {'Asymm':>8} {'Depl %':>8}")
        print("  " + "-" * 55)
        for _, row in theta_alpha.iterrows():
            print(f"  {row['age_center']:>6.0f} {row['right_slope']:>10.0f} {row['left_slope']:>10.0f} "
                  f"{row['right_slope']/max(row['left_slope'],1):>8.2f} {row['slope_asymmetry']:>+8.3f} "
                  f"{row['depletion_pct']:>8.1f}")

        # Does right slope (alpha side) grow faster than left slope (theta side)?
        rho_r, p_r = stats.spearmanr(theta_alpha['age_center'], theta_alpha['right_slope'])
        rho_l, p_l = stats.spearmanr(theta_alpha['age_center'], theta_alpha['left_slope'])
        print(f"\n  Right slope (alpha) vs age: ρ = {rho_r:+.3f} (p = {p_r:.3f})")
        print(f"  Left slope (theta) vs age:  ρ = {rho_l:+.3f} (p = {p_l:.3f})")

    # Same for α/β trough
    print("\n  --- α/β Trough: Does Beta Growth Shift It? ---")
    alpha_beta = df[df.trough_short == 'α/β'].sort_values('age_center')
    if len(alpha_beta) >= 3:
        print(f"  {'Age':>6} {'R slope':>10} {'L slope':>10} {'Ratio':>8} {'Asymm':>8} {'Depl %':>8}")
        print("  " + "-" * 55)
        for _, row in alpha_beta.iterrows():
            print(f"  {row['age_center']:>6.0f} {row['right_slope']:>10.0f} {row['left_slope']:>10.0f} "
                  f"{row['right_slope']/max(row['left_slope'],1):>8.2f} {row['slope_asymmetry']:>+8.3f} "
                  f"{row['depletion_pct']:>8.1f}")

        rho_r, p_r = stats.spearmanr(alpha_beta['age_center'], alpha_beta['right_slope'])
        rho_l, p_l = stats.spearmanr(alpha_beta['age_center'], alpha_beta['left_slope'])
        rho_a, p_a = stats.spearmanr(alpha_beta['age_center'], alpha_beta['slope_asymmetry'])
        print(f"\n  Right slope (beta) vs age:  ρ = {rho_r:+.3f} (p = {p_r:.3f})")
        print(f"  Left slope (alpha) vs age:  ρ = {rho_l:+.3f} (p = {p_l:.3f})")
        print(f"  Asymmetry vs age:           ρ = {rho_a:+.3f} (p = {p_a:.3f})")


def analysis_4_precision_as_regulation():
    """Trough CI width as a measure of how tightly the brain regulates each boundary."""
    print("\n" + "=" * 70)
    print("Analysis 4: Trough Precision as Regulatory Tightness")
    print("=" * 70)

    ci_widths = TROUGH_CI_HI - TROUGH_CI_LO
    ci_widths_pct = ci_widths / TROUGHS * 100

    # What correlates with precision?
    # Depletion depth, functional importance, developmental trajectory...

    pooled_depletion = np.array([70.4, 8.7, 61.7, 11.6, 32.2])

    print(f"\n  {'Trough':<8} {'CI (Hz)':>10} {'CI (%)':>8} {'Depl %':>8} {'Regulation':>15}")
    print("  " + "-" * 55)
    for label, ci, cipct, depl in zip(TROUGH_LABELS, ci_widths, ci_widths_pct, pooled_depletion):
        reg = 'very tight' if ci < 0.1 else ('tight' if ci < 0.5 else ('moderate' if ci < 1.0 else 'loose'))
        print(f"  {label:<8} {ci:>10.3f} {cipct:>8.2f} {depl:>7.1f}% {reg:>15}")

    # Rank correlation: tighter CI ↔ deeper trough?
    rho_depth, p_depth = stats.spearmanr(ci_widths, pooled_depletion)
    print(f"\n  CI width vs depletion: ρ = {rho_depth:+.3f} (p = {p_depth:.3f})")
    print(f"  (Negative = tighter CI for deeper troughs)")

    # The ranking tells a story:
    print(f"\n  Precision ranking (tightest → loosest):")
    order = np.argsort(ci_widths)
    for rank, i in enumerate(order):
        print(f"    {rank+1}. {TROUGH_LABELS[i]}: CI = {ci_widths[i]:.3f} Hz, depletion = {pooled_depletion[i]:.1f}%")

    print(f"\n  The θ/α trough is 5× more precisely located than the next tightest (δ/θ).")
    print(f"  It is also the SHALLOWEST trough (8.7% depletion).")
    print(f"  This is paradoxical if precision tracks depth.")
    print(f"  Resolution: θ/α precision reflects FUNCTIONAL regulation, not depth.")
    print(f"  The brain doesn't care how DEEP the trough is at θ/α —")
    print(f"  it cares exactly WHERE it falls, because theta-alpha separation")
    print(f"  must be precise for working memory, attention, and awareness.")

    # Contrast with βL/βH
    print(f"\n  βL/βH (bridge) has CI = {ci_widths[3]:.2f} Hz — 58× wider than θ/α.")
    print(f"  This is the least regulated boundary. Consistent with it being a")
    print(f"  failed trough — the brain doesn't tightly control its position")
    print(f"  because it's a compromise, not a precisely carved wall.")


def analysis_5_two_forces_model():
    """Formal model: trough position = lattice position + peak-mass displacement."""
    print("\n" + "=" * 70)
    print("Analysis 5: Two-Forces Model")
    print("=" * 70)

    # Model: T_obs = T_lattice + Δ(peak mass)
    # T_lattice = f₀ × φ^n
    # Δ = function of flanking peak asymmetry
    #
    # If the right peak is stronger, it pulls the trough LEFT (toward lower freq)
    # → negative displacement
    # If the left peak is stronger, it pulls the trough RIGHT (toward higher freq)
    # → positive displacement

    f0 = 8.1164  # bridge-excluded estimate
    lattice = f0 * PHI ** NS
    displacement = TROUGHS - lattice

    if not os.path.exists(SHAPE_PATH):
        print("  trough_shapes.csv not found")
        return

    df = pd.read_csv(SHAPE_PATH)
    df['trough_short'] = df['trough'].map(TROUGH_NAME_MAP)

    # Mean slope asymmetry per trough (averaged over age/cohort)
    mean_asym = df.groupby('trough_short')['slope_asymmetry'].mean()

    print(f"\n  f₀ = {f0:.4f} Hz")
    print(f"\n  {'Trough':<8} {'Lattice':>8} {'Observed':>8} {'Δ Hz':>8} {'Δ log':>8} {'Slope asym':>10} {'Consistent?':>12}")
    print("  " + "-" * 72)

    displ_list = []
    asym_list = []
    labels_used = []

    for i, label in enumerate(TROUGH_LABELS):
        lat = lattice[i]
        obs = TROUGHS[i]
        d = displacement[i]
        d_log = np.log(obs) - np.log(lat)

        asym = mean_asym.get(label, np.nan)

        # Positive slope_asymmetry = steeper right = stronger right peak
        # Stronger right peak → pulls trough LEFT → negative displacement
        # So: positive asymmetry should predict negative displacement
        if not np.isnan(asym):
            expected_direction = 'left (-)' if asym > 0 else 'right (+)'
            actual_direction = '+' if d > 0 else '-'
            consistent = (asym > 0 and d < 0) or (asym < 0 and d > 0) or abs(asym) < 0.05
            displ_list.append(d_log)
            asym_list.append(asym)
            labels_used.append(label)
        else:
            expected_direction = '?'
            consistent = None

        print(f"  {label:<8} {lat:>8.2f} {obs:>8.2f} {d:>+8.2f} {d_log:>+8.4f} {asym:>+10.3f} "
              f"{'✓' if consistent else '✗' if consistent is not None else '?':>12}")

    # Correlation between slope asymmetry and displacement
    if len(displ_list) >= 3:
        rho, p = stats.spearmanr(asym_list, displ_list)
        r, p_pear = stats.pearsonr(asym_list, displ_list)
        print(f"\n  Slope asymmetry vs log-displacement:")
        print(f"    Spearman ρ = {rho:+.3f} (p = {p:.3f})")
        print(f"    Pearson r  = {r:+.3f} (p = {p_pear:.3f})")
        print(f"    Prediction: ρ should be NEGATIVE (stronger right peak → leftward displacement)")

    # The model
    print(f"\n  --- Two-Forces Model Summary ---")
    print(f"  T_observed = f₀ × φ^n + Δ_peak_mass")
    print(f"")
    print(f"  δ/θ:   Lattice says 5.02, observed 5.03 (+0.01). Nearly on lattice.")
    print(f"         Generator-dominated gap, not actively carved.")
    print(f"  θ/α:   Lattice says 8.12, observed 7.82 (-0.29). Pulled DOWN.")
    print(f"         Alpha peak mass drags trough toward itself.")
    print(f"         Most precisely located (CI=0.03) because functional.")
    print(f"  α/β:   Lattice says 13.13, observed 13.59 (+0.46). Pulled UP.")
    print(f"         Alpha peak below, weak beta above → net upward pull.")
    print(f"         Deepening with age as PV+ carves it in place.")
    print(f"  βL/βH: Lattice says 21.25, observed 24.75 (+3.50). Massively displaced.")
    print(f"         Motor enrichment at 20 Hz repels the density minimum upward.")
    print(f"         Not a true trough — a bridge forced out of position.")
    print(f"  βH/γ:  Lattice says 34.38, observed 34.38 (+0.00). On lattice.")
    print(f"         No strong flanking peak to displace it. Clean inhibitory wall.")

    results = pd.DataFrame({
        'trough': labels_used,
        'displacement_log': displ_list,
        'slope_asymmetry': asym_list,
    })
    results.to_csv(os.path.join(OUT_DIR, 'two_forces_model.csv'), index=False)


def analysis_6_theta_alpha_special():
    """Deep dive on why θ/α is the most regulated boundary."""
    print("\n" + "=" * 70)
    print("Analysis 6: Why θ/α Is the Most Regulated Boundary")
    print("=" * 70)

    ci_widths = TROUGH_CI_HI - TROUGH_CI_LO

    print(f"\n  θ/α trough properties:")
    print(f"    Position:   7.8227 Hz (bootstrap median)")
    print(f"    CI:         [{TROUGH_CI_LO[1]:.4f}, {TROUGH_CI_HI[1]:.4f}] — width {ci_widths[1]:.4f} Hz")
    print(f"    Depletion:  8.7% (shallowest of all troughs)")
    print(f"    Type:       'Cliff' — 158 pp drop from theta to alpha")
    print(f"    Detection:  100% of bootstrap samples")

    print(f"\n  Paradox: shallowest trough, tightest CI.")
    print(f"  If depth = regulation strength, this makes no sense.")
    print(f"  But if POSITION = regulation target, it makes perfect sense.")

    print(f"\n  The θ/α boundary separates:")
    print(f"    THETA (4.7-7.6 Hz): hippocampal binding, working memory, navigation")
    print(f"    ALPHA (7.6-12.3 Hz): cortical inhibition, attentional gating, awareness")
    print(f"")
    print(f"  These are the two bands most critical for conscious cognition.")
    print(f"  The boundary between them must be precise — NOT deep.")
    print(f"  A cliff, not a canyon. The 158 pp enrichment drop at θ/α means")
    print(f"  peaks don't cross this boundary, but the frequency is not avoided.")
    print(f"")
    print(f"  Contrast with α/β (13.6 Hz): deep (61.7%), wide (0.44 Hz CI).")
    print(f"  Here the brain creates a wide spectral void — no peaks near 13.6 Hz.")
    print(f"  This IS a canyon. It doesn't matter exactly where the canyon is,")
    print(f"  as long as it's deep enough. Hence wider CI.")

    print(f"\n  The θ/α boundary also sits 0.29 Hz BELOW the φ-lattice prediction.")
    print(f"  This displacement = alpha peak mass pulling the cliff downward.")
    print(f"  The brain tolerates this displacement because the FUNCTIONAL")
    print(f"  requirement (precise theta-alpha separation) overrides the")
    print(f"  STRUCTURAL requirement (sit on φ-lattice).")

    print(f"\n  Testable prediction:")
    print(f"    If α power is pharmacologically increased (e.g., benzodiazepines),")
    print(f"    the θ/α trough should shift DOWNWARD (more alpha mass pulling).")
    print(f"    If α power is reduced (e.g., cholinergics, eyes-open),")
    print(f"    the θ/α trough should shift UPWARD toward lattice position.")
    print(f"    We can test this with eyes-closed vs eyes-open comparison.")

    # Do we have EO data for comparison?
    if os.path.exists(SHAPE_PATH):
        # Check if we have EO data in trough shapes
        df = pd.read_csv(SHAPE_PATH)
        # The trough shapes are computed from EC data only based on column names
        # But we can note this as a future test


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Trough Displacement: Lattice Pull vs Peak Mass Pull")
    print("=" * 70)

    analysis_1_displacement()
    analysis_2_slope_asymmetry()
    analysis_3_peak_mass_pull()
    analysis_4_precision_as_regulation()
    analysis_5_two_forces_model()
    analysis_6_theta_alpha_special()

    print(f"\n\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
