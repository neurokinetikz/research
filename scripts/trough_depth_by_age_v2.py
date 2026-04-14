#!/usr/bin/env python3
"""
Analysis 1b: Within-Dataset Trough Depth Trajectories + Bootstrap CIs
=====================================================================
Controls for cross-dataset confounds by running HBN-only and Dortmund-only
trajectories. Adds subject-level bootstrap CIs on each age-bin depth.

Usage:
    python scripts/trough_depth_by_age_v2.py [--n-boot 500] [--plot]
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age')
MIN_POWER_PCT = 50

KNOWN_TROUGHS_HZ = np.array([5.08, 7.81, 13.42, 25.30, 35.04])
TROUGH_LABELS = ['δ/θ (5.1)', 'θ/α (7.8)', 'α/β (13.4)', 'βL/βH (25.3)', 'βH/γ (35.0)']

HBN_RELEASES = ['R1', 'R2', 'R3', 'R4', 'R6']
HBN_DEMO_TEMPLATE = '/Volumes/T9/hbn_data/cmi_bids_{release}/participants.tsv'
DORTMUND_DEMO = '/Volumes/T9/dortmund_data/participants.tsv'
LEMON_DEMO = ('/Volumes/T9/lemon_data/behavioral/'
              'Behavioural_Data_MPILMBB_LEMON/'
              'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')

LEMON_AGE_MAP = {
    '20-25': 22.5, '25-30': 27.5, '30-35': 32.5, '35-40': 37.5,
    '55-60': 57.5, '60-65': 62.5, '65-70': 67.5, '70-75': 72.5, '75-80': 77.5,
}


def load_demographics():
    """Load age for all subjects with peak data. Returns {subject_id: age}."""
    age_map = {}
    for release in HBN_RELEASES:
        tsv = HBN_DEMO_TEMPLATE.format(release=release)
        if not os.path.exists(tsv):
            continue
        df = pd.read_csv(tsv, sep='\t')
        for _, row in df.iterrows():
            if pd.notna(row.get('age')):
                age_map[row['participant_id']] = float(row['age'])

    if os.path.exists(DORTMUND_DEMO):
        df = pd.read_csv(DORTMUND_DEMO, sep='\t')
        for _, row in df.iterrows():
            if pd.notna(row.get('age')):
                age_map[row['participant_id']] = float(row['age'])

    if os.path.exists(LEMON_DEMO):
        df = pd.read_csv(LEMON_DEMO)
        for _, row in df.iterrows():
            mid = LEMON_AGE_MAP.get(str(row.get('Age', '')), np.nan)
            if pd.notna(mid):
                age_map[row['ID']] = mid

    return age_map


def load_peaks_with_age(age_map, dataset_filter=None):
    """Load per-subject peaks for subjects with age data.
    Returns list of (subject_id, dataset_base, age, freq_array).
    dataset_filter: None=all, 'hbn', 'dortmund', 'lemon'
    """
    datasets = {
        'hbn_R1': 'hbn', 'hbn_R2': 'hbn', 'hbn_R3': 'hbn',
        'hbn_R4': 'hbn', 'hbn_R6': 'hbn',
        'dortmund': 'dortmund', 'lemon': 'lemon',
    }

    subjects = []
    for subdir, base in datasets.items():
        if dataset_filter and base != dataset_filter:
            continue
        path = os.path.join(PEAK_BASE, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        for f in files:
            subj_id = os.path.basename(f).replace('_peaks.csv', '')
            if subj_id not in age_map:
                continue
            df = pd.read_csv(f, usecols=cols)
            if has_power and MIN_POWER_PCT > 0:
                filtered = []
                for octave in df['phi_octave'].unique():
                    bp = df[df.phi_octave == octave]
                    thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                    filtered.append(bp[bp['power'] >= thresh])
                df = pd.concat(filtered, ignore_index=True)
            subjects.append((subj_id, base, age_map[subj_id], df['freq'].values))

    return subjects


def measure_trough_depths(freqs, n_hist=1000, sigma_detail=8, sigma_envelope=40,
                          f_range=(3, 55)):
    """Measure depth ratio at each known trough position."""
    log_freqs = np.log(freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)

    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma_detail)
    envelope = gaussian_filter1d(counts.astype(float), sigma=sigma_envelope)

    depths = {}
    for trough_hz, label in zip(KNOWN_TROUGHS_HZ, TROUGH_LABELS):
        idx = np.argmin(np.abs(hz_centers - trough_hz))
        env_val = envelope[idx]
        if env_val > 0:
            depth_ratio = smoothed[idx] / env_val
        else:
            depth_ratio = np.nan
        depths[label] = depth_ratio

    return depths


def bootstrap_age_bin(subjects_in_bin, n_boot=500, seed=42):
    """Bootstrap subjects within an age bin to get CIs on trough depths.
    Returns dict of {trough_label: (mean, ci_lo, ci_hi)}.
    """
    rng = np.random.default_rng(seed)
    n = len(subjects_in_bin)
    freq_arrays = [s[3] for s in subjects_in_bin]

    boot_depths = {label: [] for label in TROUGH_LABELS}

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_freqs = np.concatenate([freq_arrays[i] for i in idx])
        depths = measure_trough_depths(boot_freqs)
        for label, ratio in depths.items():
            boot_depths[label].append((1 - ratio) * 100)  # depletion %

    results = {}
    for label in TROUGH_LABELS:
        vals = np.array(boot_depths[label])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 10:
            results[label] = (np.mean(vals), np.percentile(vals, 2.5),
                              np.percentile(vals, 97.5))
        else:
            results[label] = (np.nan, np.nan, np.nan)

    return results


def run_analysis(subjects, age_bins, label_prefix, n_boot=500):
    """Run age-binned trough depth analysis with bootstrap CIs."""
    rows = []
    for lo, hi in age_bins:
        bin_subjects = [s for s in subjects if lo <= s[2] < hi]
        if len(bin_subjects) < 15:
            continue

        bin_center = (lo + hi) / 2
        bin_freqs = np.concatenate([s[3] for s in bin_subjects])

        # Point estimates
        depths = measure_trough_depths(bin_freqs)

        # Bootstrap CIs
        boot_cis = bootstrap_age_bin(bin_subjects, n_boot=n_boot,
                                      seed=int(lo * 100 + hi))

        ds_counts = {}
        for s in bin_subjects:
            ds_counts[s[1]] = ds_counts.get(s[1], 0) + 1

        print(f"  Age {lo:2.0f}-{hi:2.0f} | N={len(bin_subjects):4d} | "
              f"{len(bin_freqs):>8,} peaks | "
              f"{', '.join(f'{k}:{v}' for k, v in sorted(ds_counts.items()))}")

        for label in TROUGH_LABELS:
            depletion = (1 - depths[label]) * 100
            mean_boot, ci_lo, ci_hi = boot_cis[label]
            rows.append({
                'cohort': label_prefix,
                'age_lo': lo, 'age_hi': hi, 'age_center': bin_center,
                'n_subjects': len(bin_subjects), 'n_peaks': len(bin_freqs),
                'trough_label': label,
                'trough_hz': KNOWN_TROUGHS_HZ[TROUGH_LABELS.index(label)],
                'depletion_pct': depletion,
                'boot_mean': mean_boot, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'ci_width': ci_hi - ci_lo if not np.isnan(ci_hi) else np.nan,
            })

    return pd.DataFrame(rows)


def print_trajectory_summary(df, cohort_label):
    """Print trajectory statistics for a cohort."""
    print(f"\n{'=' * 70}")
    print(f"  {cohort_label} TRAJECTORY SUMMARY")
    print(f"{'=' * 70}")

    for label in TROUGH_LABELS:
        sub = df[df.trough_label == label].sort_values('age_center')
        if len(sub) < 3:
            continue

        x = sub['age_center'].values
        y = sub['depletion_pct'].values

        max_idx = np.argmax(y)
        min_idx = np.argmin(y)
        rho, p = spearmanr(x, y)

        print(f"\n  {label}:")
        print(f"    Range: {y.min():.1f}% - {y.max():.1f}% depletion")
        print(f"    Deepest at age {x[max_idx]:.0f}, shallowest at age {x[min_idx]:.0f}")
        print(f"    Linear trend: rho={rho:.3f}, p={p:.4f}")

        # Typical CI width
        ci_widths = sub['ci_width'].dropna()
        if len(ci_widths) > 0:
            print(f"    Median CI width: {ci_widths.median():.1f} pp")

        if len(x) >= 5:
            coeffs = np.polyfit(x, y, 2)
            vertex_age = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else np.nan
            is_inverted_u = coeffs[0] < 0 and x.min() < vertex_age < x.max()
            print(f"    Quadratic: a={coeffs[0]:.4f}, vertex={vertex_age:.1f}")
            print(f"    Inverted-U within range: {'YES' if is_inverted_u else 'no'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-boot', type=int, default=500)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading demographics...")
    age_map = load_demographics()
    print(f"  {len(age_map)} subjects with age data")

    # --- HBN-only (ages 5-21, 2-year bins for finer resolution) ---
    hbn_bins = [(5, 7), (7, 9), (9, 11), (11, 13), (13, 15), (15, 17), (17, 21)]
    print("\n" + "=" * 70)
    print("  HBN-ONLY (ages 5-21)")
    print("=" * 70)
    hbn_subjects = load_peaks_with_age(age_map, dataset_filter='hbn')
    print(f"  {len(hbn_subjects)} HBN subjects loaded")
    print()
    hbn_results = run_analysis(hbn_subjects, hbn_bins, 'HBN', n_boot=args.n_boot)
    print_trajectory_summary(hbn_results, 'HBN-ONLY')

    # --- Dortmund-only (ages 20-70, 5-year bins) ---
    dort_bins = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45),
                 (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)]
    print("\n" + "=" * 70)
    print("  DORTMUND-ONLY (ages 20-70)")
    print("=" * 70)
    dort_subjects = load_peaks_with_age(age_map, dataset_filter='dortmund')
    print(f"  {len(dort_subjects)} Dortmund subjects loaded")
    print()
    dort_results = run_analysis(dort_subjects, dort_bins, 'Dortmund', n_boot=args.n_boot)
    print_trajectory_summary(dort_results, 'DORTMUND-ONLY')

    # --- Combined (all datasets, 5-year bins) ---
    all_bins = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35),
                (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)]
    print("\n" + "=" * 70)
    print("  ALL DATASETS COMBINED (ages 5-70)")
    print("=" * 70)
    all_subjects = load_peaks_with_age(age_map)
    print(f"  {len(all_subjects)} subjects loaded")
    print()
    all_results = run_analysis(all_subjects, all_bins, 'All', n_boot=args.n_boot)
    print_trajectory_summary(all_results, 'ALL COMBINED')

    # Save all results
    combined = pd.concat([hbn_results, dort_results, all_results], ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, 'trough_depth_by_age_v2.csv'), index=False)
    print(f"\nAll results saved to {OUT_DIR}/trough_depth_by_age_v2.csv")

    if args.plot:
        generate_plot(hbn_results, dort_results, all_results)


def generate_plot(hbn_results, dort_results, all_results):
    """3-column figure: HBN | Dortmund | Combined, one row per trough."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
    cohorts = [
        ('HBN (5-21)', hbn_results),
        ('Dortmund (20-70)', dort_results),
        ('Combined (5-70)', all_results),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(15, 18), sharex='col')

    for col, (cohort_name, df) in enumerate(cohorts):
        for row, (label, color) in enumerate(zip(TROUGH_LABELS, colors)):
            ax = axes[row, col]
            sub = df[df.trough_label == label].sort_values('age_center')
            if len(sub) < 2:
                ax.text(0.5, 0.5, 'insufficient data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')
                if row == 0:
                    ax.set_title(cohort_name, fontweight='bold', fontsize=11)
                if col == 0:
                    ax.set_ylabel(label, fontweight='bold', fontsize=10)
                continue

            x = sub['age_center'].values
            y = sub['depletion_pct'].values
            ci_lo = sub['ci_lo'].values
            ci_hi = sub['ci_hi'].values

            # CI band
            valid = ~np.isnan(ci_lo) & ~np.isnan(ci_hi)
            if valid.any():
                ax.fill_between(x[valid], ci_lo[valid], ci_hi[valid],
                                color=color, alpha=0.2)

            # Data points + line
            ax.plot(x, y, 'o-', color=color, markersize=5, linewidth=1.5)

            # N annotations
            for _, r in sub.iterrows():
                ax.annotate(f"n={int(r['n_subjects'])}", (r['age_center'], r['ci_hi']),
                            textcoords="offset points", xytext=(0, 6), fontsize=5.5,
                            ha='center', color='gray')

            # Quadratic fit if enough points
            if len(x) >= 5:
                coeffs = np.polyfit(x, y, 2)
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=color, alpha=0.4,
                        linewidth=1)
                vertex = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else np.nan
                if x.min() < vertex < x.max():
                    ax.axvline(vertex, color=color, alpha=0.3, linestyle=':', linewidth=0.8)

            # Zero line
            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)

            ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(cohort_name, fontweight='bold', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{label}\nDepletion (%)', fontweight='bold', fontsize=9)
            if row == 4:
                ax.set_xlabel('Age (years)')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'trough_depth_by_age_within_dataset.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {path}")
    plt.close()


if __name__ == '__main__':
    main()
