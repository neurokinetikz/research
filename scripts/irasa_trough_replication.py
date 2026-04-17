#!/usr/bin/env python3
"""
IRASA Trough Replication: Method-Independent φ-Lattice Validation
=================================================================

The spectral differentiation paper found 5 φ-spaced troughs using FOOOF
peak extraction. IRASA (Irregular Resampling Auto-Spectral Analysis)
uses a fundamentally different aperiodic removal method. If the troughs
and φ-scaling are real spectral features (not FOOOF artifacts), they
must replicate under IRASA.

Tests:
  1. Trough positions: do the same 5 troughs appear at the same Hz?
  2. Geometric mean ratio: does it lock to φ?
  3. Bootstrap CIs: how precise are IRASA troughs?
  4. EC vs EO: does the bridge collapse replicate?
  5. f₀ estimation: does the lattice seed converge to ~8.12 Hz?
  6. Per-dataset consistency: do troughs replicate in each dataset?

Usage:
    python scripts/irasa_trough_replication.py
    python scripts/irasa_trough_replication.py --n-boot 1000

Outputs to: outputs/irasa_trough_replication/
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
IRASA_BASE = os.path.join(BASE_DIR, 'exports_irasa_v4')
FOOOF_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'irasa_trough_replication')

PHI = (1 + np.sqrt(5)) / 2
MIN_POWER_PCT = 50

# FOOOF reference troughs
FOOOF_TROUGHS = np.array([5.0274, 7.8227, 13.5949, 24.7516, 34.3834])
TROUGH_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']

# 9 core EC datasets
EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp', 'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3', 'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
}

EO_DATASETS = {
    'lemon_EO': 'lemon_EO',
    'dortmund_EO': 'dortmund_EO_pre',
}


def load_subjects(base_dir, datasets):
    """Load per-subject peak frequencies."""
    subjects = []
    for name, subdir in datasets.items():
        path = os.path.join(base_dir, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            print(f"  {name}: no files found at {path}")
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        count = 0
        for f in files:
            subj_id = os.path.basename(f).replace('_peaks.csv', '')
            try:
                df = pd.read_csv(f, usecols=cols)
            except Exception:
                continue
            if has_power and MIN_POWER_PCT > 0:
                filtered = []
                for octave in df['phi_octave'].unique():
                    bp = df[df.phi_octave == octave]
                    if len(bp) == 0:
                        continue
                    thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                    filtered.append(bp[bp['power'] >= thresh])
                if filtered:
                    df = pd.concat(filtered, ignore_index=True)
                else:
                    continue
            if len(df) > 0:
                subjects.append((name, subj_id, df['freq'].values))
                count += 1
        print(f"  {name}: {count} subjects, {sum(len(s[2]) for s in subjects if s[0]==name):,} peaks")
    return subjects


def find_troughs(all_freqs, n_hist=1000, sigma=8, f_range=(3, 55)):
    """Find density troughs using log-space histogram + smoothing."""
    log_freqs = np.log(all_freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)

    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)

    median_val = np.median(smoothed[smoothed > 0])
    trough_idx, _ = find_peaks(-smoothed, prominence=median_val * 0.08,
                                distance=n_hist // 25)
    trough_hz = hz_centers[trough_idx]
    trough_hz = trough_hz[(trough_hz > 4) & (trough_hz < 50)]
    return trough_hz


def match_troughs(detected, reference, max_log_dist=0.3):
    """Match detected troughs to reference positions."""
    matched = {}
    for i, ref in enumerate(reference):
        if len(detected) == 0:
            continue
        log_dists = np.abs(np.log(detected) - np.log(ref))
        nearest = np.argmin(log_dists)
        if log_dists[nearest] < max_log_dist:
            matched[i] = detected[nearest]
    return matched


def bootstrap_troughs(subjects, n_boot=500, seed=42):
    """Bootstrap over subjects to get trough position CIs."""
    rng = np.random.default_rng(seed)

    # Group by dataset for stratified resampling
    dataset_groups = {}
    for i, (ds, _, _) in enumerate(subjects):
        if ds not in dataset_groups:
            dataset_groups[ds] = []
        dataset_groups[ds].append(i)

    freq_arrays = [s[2] for s in subjects]

    boot_positions = {i: [] for i in range(5)}
    boot_geo_means = []

    for b in range(n_boot):
        idx = []
        for ds, ds_indices in dataset_groups.items():
            sampled = rng.choice(ds_indices, size=len(ds_indices), replace=True)
            idx.extend(sampled)
        boot_freqs = np.concatenate([freq_arrays[i] for i in idx])
        troughs = find_troughs(boot_freqs)

        matched = match_troughs(troughs, FOOOF_TROUGHS)
        for ref_idx, pos in matched.items():
            boot_positions[ref_idx].append(pos)

        if len(troughs) >= 2:
            ratios = troughs[1:] / troughs[:-1]
            boot_geo_means.append(np.exp(np.mean(np.log(ratios))))

        if (b + 1) % 100 == 0:
            print(f"    Bootstrap {b+1}/{n_boot}")

    results = {}
    for i in range(5):
        positions = np.array(boot_positions[i])
        if len(positions) >= 10:
            results[i] = {
                'median': np.median(positions),
                'ci_lo': np.percentile(positions, 2.5),
                'ci_hi': np.percentile(positions, 97.5),
                'n_detected': len(positions),
                'detection_pct': len(positions) / n_boot * 100,
            }

    return results, np.array(boot_geo_means)


def per_dataset_troughs(subjects):
    """Find troughs in each dataset independently."""
    dataset_groups = {}
    for ds, subj_id, freqs in subjects:
        if ds not in dataset_groups:
            dataset_groups[ds] = []
        dataset_groups[ds].append(freqs)

    results = []
    for ds, freq_list in dataset_groups.items():
        all_freqs = np.concatenate(freq_list)
        troughs = find_troughs(all_freqs)
        matched = match_troughs(troughs, FOOOF_TROUGHS)

        for i, label in enumerate(TROUGH_LABELS):
            pos = matched.get(i, np.nan)
            results.append({
                'dataset': ds, 'trough': label,
                'position_hz': pos, 'detected': not np.isnan(pos),
                'n_subjects': len(freq_list),
                'n_peaks': len(all_freqs),
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-boot', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("IRASA Trough Replication")
    print("=" * 70)

    # --- 1. Load IRASA peaks ---
    print("\n--- Loading IRASA peaks (EC, 9 datasets) ---")
    irasa_subjects = load_subjects(IRASA_BASE, EC_DATASETS)
    total_peaks = sum(len(s[2]) for s in irasa_subjects)
    print(f"  Total: {len(irasa_subjects)} subjects, {total_peaks:,} peaks")

    # --- 2. Pooled trough detection ---
    print("\n--- Pooled Trough Detection ---")
    all_freqs = np.concatenate([s[2] for s in irasa_subjects])
    irasa_troughs = find_troughs(all_freqs)
    irasa_troughs_filt = irasa_troughs[(irasa_troughs > 4) & (irasa_troughs < 50)]

    print(f"  IRASA troughs: {np.round(irasa_troughs_filt, 2)} Hz")
    print(f"  FOOOF troughs: {np.round(FOOOF_TROUGHS, 2)} Hz")

    # Match and compare
    matched = match_troughs(irasa_troughs_filt, FOOOF_TROUGHS)
    print(f"\n  {'Trough':<8} {'FOOOF':>10} {'IRASA':>10} {'Δ (Hz)':>10} {'Δ (%)':>8} {'Match?':>8}")
    print("  " + "-" * 55)

    irasa_positions = []
    for i, label in enumerate(TROUGH_LABELS):
        fooof_pos = FOOOF_TROUGHS[i]
        irasa_pos = matched.get(i, np.nan)
        if not np.isnan(irasa_pos):
            delta = irasa_pos - fooof_pos
            delta_pct = delta / fooof_pos * 100
            print(f"  {label:<8} {fooof_pos:>10.4f} {irasa_pos:>10.4f} {delta:>+10.4f} {delta_pct:>+8.2f} {'✓':>8}")
            irasa_positions.append(irasa_pos)
        else:
            print(f"  {label:<8} {fooof_pos:>10.4f} {'not found':>10} {'--':>10} {'--':>8} {'✗':>8}")

    # --- 3. Geometric mean ratio ---
    if len(irasa_positions) >= 2:
        irasa_sorted = np.sort(irasa_positions)
        irasa_ratios = irasa_sorted[1:] / irasa_sorted[:-1]
        irasa_geo_mean = np.exp(np.mean(np.log(irasa_ratios)))

        fooof_ratios = FOOOF_TROUGHS[1:] / FOOOF_TROUGHS[:-1]
        fooof_geo_mean = np.exp(np.mean(np.log(fooof_ratios)))

        print(f"\n  Geometric mean ratio:")
        print(f"    FOOOF: {fooof_geo_mean:.4f}")
        print(f"    IRASA: {irasa_geo_mean:.4f}")
        print(f"    φ:     {PHI:.4f}")
        print(f"    IRASA - φ: {irasa_geo_mean - PHI:+.4f}")

    # --- 4. Bootstrap CIs ---
    print(f"\n--- Bootstrap CIs ({args.n_boot} iterations, stratified) ---")
    boot_results, boot_geo_means = bootstrap_troughs(irasa_subjects, n_boot=args.n_boot)

    print(f"\n  {'Trough':<8} {'Median':>10} {'CI':>24} {'Det %':>8}")
    print("  " + "-" * 55)
    for i, label in enumerate(TROUGH_LABELS):
        b = boot_results.get(i)
        if b:
            ci = f"[{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]"
            print(f"  {label:<8} {b['median']:>10.4f} {ci:>24} {b['detection_pct']:>7.1f}%")

    if len(boot_geo_means) > 0:
        print(f"\n  Bootstrap geometric mean ratio:")
        print(f"    Mean:   {np.mean(boot_geo_means):.4f} ± {np.std(boot_geo_means):.4f}")
        print(f"    Median: {np.median(boot_geo_means):.4f}")
        print(f"    CI:     [{np.percentile(boot_geo_means, 2.5):.4f}, {np.percentile(boot_geo_means, 97.5):.4f}]")
        print(f"    φ = {PHI:.4f}: {'WITHIN CI' if np.percentile(boot_geo_means, 2.5) <= PHI <= np.percentile(boot_geo_means, 97.5) else 'OUTSIDE CI'}")

    # --- 5. f₀ estimation ---
    print(f"\n--- f₀ Estimation from IRASA Troughs ---")
    ns = np.array([-1, 0, 1, 2, 3])
    for i, label in enumerate(TROUGH_LABELS):
        b = boot_results.get(i)
        if b:
            f0_est = b['median'] / PHI**ns[i]
            print(f"  {label} (n={ns[i]:+d}): f₀ = {b['median']:.4f} / φ^{ns[i]} = {f0_est:.4f} Hz")

    # Excluding bridge
    excl_bridge = [boot_results[i]['median'] for i in [0, 1, 2, 4] if i in boot_results]
    ns_excl = np.array([-1, 0, 1, 3])
    if len(excl_bridge) == 4:
        log_resid = np.log(excl_bridge) - ns_excl * np.log(PHI)
        f0_excl = np.exp(np.mean(log_resid))
        print(f"\n  f₀ (excl. bridge): {f0_excl:.4f} Hz")
        print(f"  FOOOF f₀ (excl. bridge): 8.1164 Hz")

    # --- 6. Per-dataset troughs ---
    print(f"\n--- Per-Dataset Trough Detection ---")
    df_per_ds = per_dataset_troughs(irasa_subjects)

    for label in TROUGH_LABELS:
        sub = df_per_ds[df_per_ds.trough == label]
        detected = sub[sub.detected]
        n_det = len(detected)
        n_total = len(sub)
        if n_det > 0:
            positions = detected['position_hz']
            print(f"  {label}: detected in {n_det}/{n_total} datasets, "
                  f"mean = {positions.mean():.2f} ± {positions.std():.2f} Hz")
        else:
            print(f"  {label}: detected in {n_det}/{n_total} datasets")

    df_per_ds.to_csv(os.path.join(OUT_DIR, 'per_dataset_troughs.csv'), index=False)

    # --- 7. EC vs EO (IRASA) ---
    print(f"\n--- EC vs EO Trough Comparison (IRASA) ---")
    print("  Loading EO data...")
    eo_subjects = load_subjects(IRASA_BASE, EO_DATASETS)

    if len(eo_subjects) > 0:
        eo_freqs = np.concatenate([s[2] for s in eo_subjects])
        eo_troughs = find_troughs(eo_freqs)
        eo_troughs_filt = eo_troughs[(eo_troughs > 4) & (eo_troughs < 50)]
        print(f"  EO troughs: {np.round(eo_troughs_filt, 2)} Hz")

        eo_matched = match_troughs(eo_troughs_filt, FOOOF_TROUGHS)
        ec_matched = matched  # from pooled EC above (but filtered to LEMON+Dortmund)

        # Recompute EC with just LEMON and Dortmund for fair comparison
        ec_ld_subjects = [s for s in irasa_subjects if s[0] in ('lemon', 'dortmund')]
        ec_ld_freqs = np.concatenate([s[2] for s in ec_ld_subjects])
        ec_ld_troughs = find_troughs(ec_ld_freqs)
        ec_ld_matched = match_troughs(ec_ld_troughs, FOOOF_TROUGHS)

        print(f"\n  {'Trough':<8} {'EC (Hz)':>10} {'EO (Hz)':>10} {'Δ (Hz)':>10}")
        print("  " + "-" * 42)
        for i, label in enumerate(TROUGH_LABELS):
            ec_pos = ec_ld_matched.get(i, np.nan)
            eo_pos = eo_matched.get(i, np.nan)
            if not np.isnan(ec_pos) and not np.isnan(eo_pos):
                delta = eo_pos - ec_pos
                print(f"  {label:<8} {ec_pos:>10.4f} {eo_pos:>10.4f} {delta:>+10.4f}")

    # --- Save summary ---
    summary_rows = []
    for i, label in enumerate(TROUGH_LABELS):
        b = boot_results.get(i)
        row = {
            'trough': label,
            'fooof_hz': FOOOF_TROUGHS[i],
            'irasa_pooled_hz': matched.get(i, np.nan),
        }
        if b:
            row.update({
                'irasa_bootstrap_median': b['median'],
                'irasa_ci_lo': b['ci_lo'],
                'irasa_ci_hi': b['ci_hi'],
                'detection_pct': b['detection_pct'],
            })
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(OUT_DIR, 'irasa_vs_fooof_troughs.csv'), index=False)

    if len(boot_geo_means) > 0:
        pd.DataFrame({'boot_geo_mean': boot_geo_means}).to_csv(
            os.path.join(OUT_DIR, 'irasa_bootstrap_geo_means.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
