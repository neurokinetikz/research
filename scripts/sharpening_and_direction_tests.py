#!/usr/bin/env python3
"""
Three Tests from the Fibonacci Inhibition Framework
=====================================================

Test 1: Does the α/β suppression zone narrow with age? (Sharpening prediction)
  IRASA finds two troughs (~11.8 and ~14.2 Hz) bracketing the α/β zone.
  If PV+ maturation sharpens this boundary, the gap should shrink with age.

Test 2: Do GABA/cognition correlations replicate with IRASA trough depths?
  Per-subject trough depths from IRASA peaks at fixed trough positions.
  Rerun psychopathology (HBN) and cognitive (LEMON) correlations.

Test 3: Period addition vs frequency addition direction.
  Both give φ, but predict opposite construction directions.
  Which fits the observed trough positions better?

Usage:
    python scripts/sharpening_and_direction_tests.py

Outputs to: outputs/sharpening_direction_tests/
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
IRASA_BASE = os.path.join(BASE_DIR, 'exports_irasa_v4')
FOOOF_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'sharpening_direction_tests')

PHI = (1 + np.sqrt(5)) / 2
MIN_POWER_PCT = 50

FOOOF_TROUGHS = np.array([5.0274, 7.8227, 13.5949, 24.7516, 34.3834])
TROUGH_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']

# Fixed trough positions for per-subject depth computation
TROUGH_HZ = {'δ/θ': 5.08, 'θ/α': 7.81, 'α/β': 13.42, 'βL/βH': 25.30, 'βH/γ': 35.04}

# Dataset groups
HBN_DATASETS = ['hbn_R1', 'hbn_R2', 'hbn_R3', 'hbn_R4', 'hbn_R6']
ADULT_DATASETS = ['lemon', 'dortmund', 'eegmmidb', 'chbmp']

# HBN demographics
HBN_DEMO_PATHS = {
    'hbn_R1': '/Volumes/T9/hbn_data/cmi_bids_R1/participants.tsv',
    'hbn_R2': '/Volumes/T9/hbn_data/cmi_bids_R2/participants.tsv',
    'hbn_R3': '/Volumes/T9/hbn_data/cmi_bids_R3/participants.tsv',
    'hbn_R4': '/Volumes/T9/hbn_data/cmi_bids_R4/participants.tsv',
    'hbn_R6': '/Volumes/T9/hbn_data/cmi_bids_R6/participants.tsv',
}


def load_subject_peaks(base_dir, dataset):
    """Load per-subject peak frequencies from one dataset."""
    path = os.path.join(base_dir, dataset)
    files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
    subjects = {}
    for f in files:
        subj_id = os.path.basename(f).replace('_peaks.csv', '')
        try:
            df = pd.read_csv(f, usecols=['freq', 'power', 'phi_octave'])
        except Exception:
            continue
        # Power filter: top 50% per band
        filtered = []
        for octave in df['phi_octave'].unique():
            bp = df[df.phi_octave == octave]
            if len(bp) == 0:
                continue
            thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
            filtered.append(bp[bp['power'] >= thresh])
        if filtered:
            df = pd.concat(filtered, ignore_index=True)
            subjects[subj_id] = df['freq'].values
    return subjects


def find_troughs_in_freqs(freqs, n_hist=1000, sigma=8, f_range=(3, 55)):
    """Find density troughs from pooled frequencies."""
    log_freqs = np.log(freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)
    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)
    trough_idx, _ = find_peaks(-smoothed, prominence=np.median(smoothed[smoothed > 0]) * 0.05,
                                distance=n_hist // 30)
    return hz_centers[trough_idx]


def compute_trough_depth_at_position(freqs, trough_hz, window_hz=1.0):
    """Compute trough depth ratio at a fixed position using local peak density."""
    if len(freqs) == 0:
        return np.nan
    log_freqs = np.log(freqs)
    log_trough = np.log(trough_hz)
    log_window = np.log(1 + window_hz / trough_hz)

    # Count in trough window
    n_trough = np.sum(np.abs(log_freqs - log_trough) < log_window / 2)
    # Count in flanking windows
    n_left = np.sum((log_freqs > log_trough - 1.5 * log_window) &
                    (log_freqs < log_trough - 0.5 * log_window))
    n_right = np.sum((log_freqs > log_trough + 0.5 * log_window) &
                     (log_freqs < log_trough + 1.5 * log_window))
    n_flank = (n_left + n_right) / 2

    if n_flank == 0:
        return np.nan
    return n_trough / n_flank  # < 1 means trough is present


# =====================================================================
# TEST 1: α/β suppression zone width vs age
# =====================================================================
def test_1_sharpening():
    print("\n" + "=" * 70)
    print("TEST 1: Does the α/β Suppression Zone Narrow with Age?")
    print("=" * 70)

    # Strategy: pool IRASA peaks by age bin, find troughs in 10-16 Hz,
    # measure the width between the two α/β-zone troughs

    # Load HBN demographics
    demo = {}
    for release, path in HBN_DEMO_PATHS.items():
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            for _, row in df.iterrows():
                pid = row.get('participant_id', row.get('Participant', ''))
                age = row.get('Age', row.get('age', np.nan))
                if pd.notna(age) and pid:
                    demo[pid] = {'age': float(age), 'release': release}

    # Load IRASA peaks per subject for HBN + adults
    all_datasets = {ds: ds for ds in HBN_DATASETS + ADULT_DATASETS}
    print("  Loading IRASA peaks by dataset...")

    age_bins = [(5, 8), (8, 11), (11, 14), (14, 18), (20, 35), (35, 50), (50, 70)]
    bin_results = []

    for lo, hi in age_bins:
        pooled_freqs = []

        for ds_name in HBN_DATASETS:
            subjects = load_subject_peaks(IRASA_BASE, ds_name)
            for subj_id, freqs in subjects.items():
                info = demo.get(subj_id)
                if info and lo <= info['age'] < hi:
                    pooled_freqs.extend(freqs)

        if hi > 18:
            # Use adult datasets - we don't have per-subject ages easily,
            # so pool all adults for the adult bins
            # Dortmund has ages in participants.tsv
            dort_demo_path = '/Volumes/T9/dortmund_data_dl/participants.tsv'
            dort_demo = {}
            if os.path.exists(dort_demo_path):
                df = pd.read_csv(dort_demo_path, sep='\t')
                for _, row in df.iterrows():
                    pid = str(row.get('participant_id', ''))
                    age = row.get('age', np.nan)
                    if pd.notna(age):
                        dort_demo[pid] = float(age)

            for ds_name in ['dortmund']:
                subjects = load_subject_peaks(IRASA_BASE, ds_name)
                for subj_id, freqs in subjects.items():
                    age = dort_demo.get(subj_id, np.nan)
                    if pd.notna(age) and lo <= age < hi:
                        pooled_freqs.extend(freqs)

        pooled_freqs = np.array(pooled_freqs)
        if len(pooled_freqs) < 5000:
            print(f"  Age {lo}-{hi}: insufficient peaks ({len(pooled_freqs)}), skipping")
            continue

        # Find troughs in 10-16 Hz zone
        troughs = find_troughs_in_freqs(pooled_freqs)
        ab_troughs = troughs[(troughs > 10) & (troughs < 16)]

        if len(ab_troughs) >= 2:
            # Take the two most prominent (lowest and highest in the zone)
            lower = ab_troughs[0]
            upper = ab_troughs[-1]
            width = upper - lower
            center = (lower + upper) / 2
        elif len(ab_troughs) == 1:
            lower = ab_troughs[0]
            upper = ab_troughs[0]
            width = 0
            center = ab_troughs[0]
        else:
            lower, upper, width, center = np.nan, np.nan, np.nan, np.nan

        n_peaks = len(pooled_freqs)
        print(f"  Age {lo:>2}-{hi:>2}: {n_peaks:>8,} peaks, "
              f"troughs in 10-16: {np.round(ab_troughs, 2)}, width = {width:.2f} Hz")

        bin_results.append({
            'age_lo': lo, 'age_hi': hi, 'age_center': (lo + hi) / 2,
            'n_peaks': n_peaks, 'lower_trough': lower, 'upper_trough': upper,
            'zone_width': width, 'zone_center': center,
            'n_troughs_detected': len(ab_troughs),
        })

    df_bins = pd.DataFrame(bin_results)
    if len(df_bins) >= 3:
        valid = df_bins.dropna(subset=['zone_width'])
        if len(valid) >= 3:
            rho, p = stats.spearmanr(valid['age_center'], valid['zone_width'])
            print(f"\n  Zone width vs age: ρ = {rho:+.3f} (p = {p:.3f})")
            if rho < 0:
                print(f"  ✓ Zone NARROWS with age — consistent with sharpening")
            else:
                print(f"  ✗ Zone does NOT narrow with age")

    df_bins.to_csv(os.path.join(OUT_DIR, 'ab_zone_width_by_age.csv'), index=False)
    return df_bins


# =====================================================================
# TEST 2: IRASA per-subject trough depths → functional correlations
# =====================================================================
def test_2_irasa_functional():
    print("\n" + "=" * 70)
    print("TEST 2: Do GABA/Cognition Correlations Replicate with IRASA?")
    print("=" * 70)

    # Compute per-subject trough depths from IRASA peaks
    print("  Computing per-subject IRASA trough depths...")

    # HBN: psychopathology
    demo = {}
    for release, path in HBN_DEMO_PATHS.items():
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            for _, row in df.iterrows():
                pid = row.get('participant_id', row.get('Participant', ''))
                age = row.get('Age', row.get('age', np.nan))
                if pd.notna(age) and pid:
                    # Also get psychopathology if available
                    entry = {'age': float(age)}
                    for col in ['p_factor', 'attention', 'internalizing', 'externalizing']:
                        if col in df.columns:
                            entry[col] = row.get(col, np.nan)
                    demo[pid] = entry

    rows_irasa = []
    rows_fooof = []

    for ds_name in HBN_DATASETS:
        # IRASA
        irasa_subjects = load_subject_peaks(IRASA_BASE, ds_name)
        for subj_id, freqs in irasa_subjects.items():
            info = demo.get(subj_id, {})
            if not info:
                continue
            row = {'subject': subj_id, 'dataset': ds_name, 'method': 'irasa'}
            row.update(info)
            for trough_label, trough_hz in TROUGH_HZ.items():
                row[f'depth_{trough_label}'] = compute_trough_depth_at_position(freqs, trough_hz)
            rows_irasa.append(row)

        # FOOOF for comparison
        fooof_subjects = load_subject_peaks(FOOOF_BASE, ds_name)
        for subj_id, freqs in fooof_subjects.items():
            info = demo.get(subj_id, {})
            if not info:
                continue
            row = {'subject': subj_id, 'dataset': ds_name, 'method': 'fooof'}
            row.update(info)
            for trough_label, trough_hz in TROUGH_HZ.items():
                row[f'depth_{trough_label}'] = compute_trough_depth_at_position(freqs, trough_hz)
            rows_fooof.append(row)

    df_irasa = pd.DataFrame(rows_irasa)
    df_fooof = pd.DataFrame(rows_fooof)

    print(f"  IRASA: {len(df_irasa)} subjects with trough depths")
    print(f"  FOOOF: {len(df_fooof)} subjects with trough depths")

    # Psychopathology correlations
    psych_vars = ['externalizing', 'internalizing', 'p_factor', 'attention']
    trough_cols = [f'depth_{t}' for t in TROUGH_LABELS]

    print(f"\n  --- Psychopathology Correlations (α/β trough) ---")
    print(f"  {'Variable':<16} {'FOOOF ρ':>10} {'FOOOF p':>10} {'IRASA ρ':>10} {'IRASA p':>10} {'Same sign?':>12}")
    print("  " + "-" * 70)

    results = []
    for var in psych_vars:
        for method_label, df_m in [('fooof', df_fooof), ('irasa', df_irasa)]:
            valid = df_m.dropna(subset=[var, 'depth_α/β'])
            if len(valid) < 30:
                continue
            rho, p = stats.spearmanr(valid[var], valid['depth_α/β'])
            results.append({
                'variable': var, 'method': method_label, 'trough': 'α/β',
                'rho': rho, 'p': p, 'n': len(valid),
            })

    df_results = pd.DataFrame(results)
    for var in psych_vars:
        fooof_row = df_results[(df_results.variable == var) & (df_results.method == 'fooof')]
        irasa_row = df_results[(df_results.variable == var) & (df_results.method == 'irasa')]
        if len(fooof_row) > 0 and len(irasa_row) > 0:
            f_rho = fooof_row.iloc[0]['rho']
            f_p = fooof_row.iloc[0]['p']
            i_rho = irasa_row.iloc[0]['rho']
            i_p = irasa_row.iloc[0]['p']
            same_sign = '✓' if np.sign(f_rho) == np.sign(i_rho) else '✗'
            print(f"  {var:<16} {f_rho:>+10.3f} {f_p:>10.4f} {i_rho:>+10.3f} {i_p:>10.4f} {same_sign:>12}")

    # Also do all troughs for externalizing/internalizing
    print(f"\n  --- All Troughs × Externalizing (IRASA) ---")
    for trough_label in TROUGH_LABELS:
        col = f'depth_{trough_label}'
        valid = df_irasa.dropna(subset=['externalizing', col])
        if len(valid) >= 30:
            rho, p = stats.spearmanr(valid['externalizing'], valid[col])
            sig = '*' if p < 0.05 else ' '
            print(f"    {trough_label}: ρ = {rho:+.3f} (p = {p:.4f}) {sig}")

    print(f"\n  --- All Troughs × Internalizing (IRASA) ---")
    for trough_label in TROUGH_LABELS:
        col = f'depth_{trough_label}'
        valid = df_irasa.dropna(subset=['internalizing', col])
        if len(valid) >= 30:
            rho, p = stats.spearmanr(valid['internalizing'], valid[col])
            sig = '*' if p < 0.05 else ' '
            print(f"    {trough_label}: ρ = {rho:+.3f} (p = {p:.4f}) {sig}")

    df_results.to_csv(os.path.join(OUT_DIR, 'irasa_functional_correlations.csv'), index=False)


# =====================================================================
# TEST 3: Period addition vs frequency addition
# =====================================================================
def test_3_direction():
    print("\n" + "=" * 70)
    print("TEST 3: Period Addition vs Frequency Addition Direction")
    print("=" * 70)

    T = FOOOF_TROUGHS
    labels = TROUGH_LABELS

    print("\n  Frequency addition: T(n) + T(n+1) → T(n+2)")
    print("  (Interaction ABOVE both inputs)")
    freq_add_errors = []
    for i in range(len(T) - 2):
        predicted = T[i] + T[i+1]
        observed = T[i+2]
        error = (predicted - observed) / observed * 100
        freq_add_errors.append(abs(error))
        print(f"    {labels[i]} + {labels[i+1]} = {predicted:.2f} Hz, "
              f"observed {labels[i+2]} = {observed:.2f} Hz, error = {error:+.1f}%")

    print(f"\n  Period addition: 1/(1/T(n) + 1/T(n+1)) → T(n-1)")
    print("  (Interaction BELOW both inputs)")
    period_add_errors = []
    for i in range(2, len(T)):
        predicted = T[i] * T[i-1] / (T[i] + T[i-1])  # harmonic mean-like
        observed = T[i-2]
        error = (predicted - observed) / observed * 100
        period_add_errors.append(abs(error))
        print(f"    concat({labels[i]}, {labels[i-1]}) = {predicted:.2f} Hz, "
              f"observed {labels[i-2]} = {observed:.2f} Hz, error = {error:+.1f}%")

    mean_freq_err = np.mean(freq_add_errors)
    mean_period_err = np.mean(period_add_errors)

    print(f"\n  Mean |error|:")
    print(f"    Frequency addition: {mean_freq_err:.1f}%")
    print(f"    Period addition:    {mean_period_err:.1f}%")

    if mean_freq_err < mean_period_err:
        print(f"  → Frequency addition fits better ({mean_freq_err:.1f}% vs {mean_period_err:.1f}%)")
    elif mean_period_err < mean_freq_err:
        print(f"  → Period addition fits better ({mean_period_err:.1f}% vs {mean_freq_err:.1f}%)")
    else:
        print(f"  → Both fit equally")

    # Also test with lattice positions (f₀ × φⁿ) rather than observed troughs
    print(f"\n  --- Using ideal lattice (f₀ = 8.12 Hz) ---")
    f0 = 8.1164
    L = f0 * PHI ** np.array([-1, 0, 1, 2, 3])

    print(f"  Frequency addition: L(n) + L(n+1) should equal L(n+2)")
    for i in range(3):
        predicted = L[i] + L[i+1]
        actual = L[i+2]
        error = (predicted - actual) / actual * 100
        print(f"    {L[i]:.2f} + {L[i+1]:.2f} = {predicted:.2f}, lattice = {actual:.2f}, error = {error:+.4f}%")

    print(f"\n  Period addition: 1/(1/L(n) + 1/L(n+1)) should equal L(n-1)")
    for i in range(2, 5):
        predicted = L[i] * L[i-1] / (L[i] + L[i-1])
        actual = L[i-2]
        error = (predicted - actual) / actual * 100
        print(f"    concat({L[i]:.2f}, {L[i-1]:.2f}) = {predicted:.2f}, lattice = {actual:.2f}, error = {error:+.4f}%")

    print(f"\n  Note: On an ideal φ-lattice, BOTH operations are exact (error ≈ 0)")
    print(f"  because φ² = φ + 1 and 1/φ = φ - 1 are equivalent.")
    print(f"  The direction question can only be resolved with displaced troughs")
    print(f"  (where errors differ) or with a causal experiment (tACS).")

    results = pd.DataFrame({
        'test': ['freq_add_mean_error', 'period_add_mean_error'],
        'value': [mean_freq_err, mean_period_err],
    })
    results.to_csv(os.path.join(OUT_DIR, 'direction_test.csv'), index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Sharpening, Functional Replication, and Direction Tests")
    print("=" * 70)

    test_1_sharpening()
    test_2_irasa_functional()
    test_3_direction()

    print(f"\n\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
