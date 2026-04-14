#!/usr/bin/env python3
"""
Analysis 1: Trough Depth Developmental Trajectory
==================================================
Tests whether spectral trough depths follow an inverted-U trajectory
across the lifespan, as predicted by the GABAergic inhibition framework.

Bins subjects by age (5-year bins), pools peaks per bin, computes
log-frequency KDEs, and measures trough depth at the 5 known positions.

Datasets with age data: HBN (5-21), LEMON (20-80), Dortmund (20-70)

Usage:
    python scripts/trough_depth_by_age.py [--plot]
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age')
MIN_POWER_PCT = 50

# Known trough positions from pooled analysis
KNOWN_TROUGHS_HZ = np.array([5.08, 7.81, 13.42, 25.30, 35.04])
TROUGH_LABELS = ['δ/θ (5.1)', 'θ/α (7.8)', 'α/β (13.4)', 'βL/βH (25.3)', 'βH/γ (35.0)']

# Demographics paths
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

# 5-year age bins
AGE_BINS = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35),
            (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)]


def load_demographics():
    """Load age for all subjects with peak data. Returns {subject_id: age}."""
    age_map = {}

    # HBN: continuous age
    for release in HBN_RELEASES:
        tsv = HBN_DEMO_TEMPLATE.format(release=release)
        if not os.path.exists(tsv):
            continue
        df = pd.read_csv(tsv, sep='\t')
        for _, row in df.iterrows():
            pid = row['participant_id']
            age = row.get('age', np.nan)
            if pd.notna(age):
                age_map[pid] = float(age)

    # Dortmund: continuous age
    if os.path.exists(DORTMUND_DEMO):
        df = pd.read_csv(DORTMUND_DEMO, sep='\t')
        for _, row in df.iterrows():
            pid = row['participant_id']
            age = row.get('age', np.nan)
            if pd.notna(age):
                age_map[pid] = float(age)

    # LEMON: binned age -> midpoints
    if os.path.exists(LEMON_DEMO):
        df = pd.read_csv(LEMON_DEMO)
        for _, row in df.iterrows():
            pid = row['ID']
            age_bin = row.get('Age', '')
            mid = LEMON_AGE_MAP.get(str(age_bin), np.nan)
            if pd.notna(mid):
                age_map[pid] = mid

    return age_map


def load_peaks_with_age(age_map):
    """Load per-subject peaks for subjects that have age data.
    Returns list of (subject_id, dataset, age, freq_array).
    """
    # Only load datasets that have age data
    datasets = {
        'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2', 'hbn_R3': 'hbn_R3',
        'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
        'dortmund': 'dortmund', 'lemon': 'lemon',
    }

    subjects = []
    for name, subdir in datasets.items():
        path = os.path.join(PEAK_BASE, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        matched = 0
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
            subjects.append((subj_id, name, age_map[subj_id], df['freq'].values))
            matched += 1
        print(f"  {name}: {matched}/{len(files)} subjects with age data")

    return subjects


def measure_trough_depths(freqs, n_hist=1000, sigma_detail=8, sigma_envelope=40,
                          f_range=(3, 55)):
    """Measure depth ratio at each known trough position.

    Depth = smoothed_density / envelope_density at each trough.
    Values < 1.0 indicate depletion; lower = deeper trough.
    """
    log_freqs = np.log(freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)

    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma_detail)
    envelope = gaussian_filter1d(counts.astype(float), sigma=sigma_envelope)

    depths = {}
    for trough_hz, label in zip(KNOWN_TROUGHS_HZ, TROUGH_LABELS):
        # Find nearest bin
        idx = np.argmin(np.abs(hz_centers - trough_hz))
        env_val = envelope[idx]
        if env_val > 0:
            depth_ratio = smoothed[idx] / env_val
        else:
            depth_ratio = np.nan
        depletion = (1 - depth_ratio) * 100 if not np.isnan(depth_ratio) else np.nan
        depths[label] = {
            'depth_ratio': depth_ratio,
            'depletion_pct': depletion,
            'smoothed_count': smoothed[idx],
            'envelope_count': env_val,
        }

    return depths, hz_centers, smoothed, envelope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading demographics...")
    age_map = load_demographics()
    print(f"  Total subjects with age: {len(age_map)}")

    print("\nLoading peaks with age metadata...")
    subjects = load_peaks_with_age(age_map)
    print(f"\nTotal subjects with age + peaks: {len(subjects)}")

    ages = np.array([s[2] for s in subjects])
    print(f"Age range: {ages.min():.1f} - {ages.max():.1f}")

    # --- Pooled baseline (all ages) ---
    print("\n--- Pooled baseline (all ages) ---")
    all_freqs = np.concatenate([s[3] for s in subjects])
    pooled_depths, _, _, _ = measure_trough_depths(all_freqs)
    print(f"  N = {len(subjects)} subjects, {len(all_freqs):,} peaks")
    for label, d in pooled_depths.items():
        print(f"  {label}: depth={d['depth_ratio']:.3f}  depletion={d['depletion_pct']:.1f}%")

    # --- Age-binned analysis ---
    print("\n--- Age-binned trough depths ---")
    rows = []
    bin_kdes = {}

    for lo, hi in AGE_BINS:
        bin_subjects = [s for s in subjects if lo <= s[2] < hi]
        if len(bin_subjects) < 20:
            print(f"  Age {lo}-{hi}: {len(bin_subjects)} subjects (skipping, < 20)")
            continue

        bin_freqs = np.concatenate([s[3] for s in bin_subjects])
        n_peaks = len(bin_freqs)
        n_subj = len(bin_subjects)
        bin_center = (lo + hi) / 2

        # Dataset composition
        ds_counts = {}
        for s in bin_subjects:
            ds = s[1].replace('hbn_', 'hbn-')
            base = ds.split('-')[0] if '-' in ds else ds
            ds_counts[base] = ds_counts.get(base, 0) + 1

        depths, hz_centers, smoothed, envelope = measure_trough_depths(bin_freqs)
        bin_kdes[bin_center] = (hz_centers, smoothed, envelope)

        print(f"\n  Age {lo}-{hi} (center={bin_center}): N={n_subj} subjects, "
              f"{n_peaks:,} peaks  [{', '.join(f'{k}:{v}' for k, v in sorted(ds_counts.items()))}]")

        for label, d in depths.items():
            print(f"    {label}: depth={d['depth_ratio']:.3f}  "
                  f"depletion={d['depletion_pct']:.1f}%")
            rows.append({
                'age_lo': lo,
                'age_hi': hi,
                'age_center': bin_center,
                'n_subjects': n_subj,
                'n_peaks': n_peaks,
                'trough_label': label,
                'trough_hz': KNOWN_TROUGHS_HZ[TROUGH_LABELS.index(label)],
                'depth_ratio': d['depth_ratio'],
                'depletion_pct': d['depletion_pct'],
                'datasets': str(ds_counts),
            })

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUT_DIR, 'trough_depth_by_age.csv'), index=False)
    print(f"\nResults saved to {OUT_DIR}/trough_depth_by_age.csv")

    # --- Summary: developmental trajectory per trough ---
    print("\n" + "=" * 70)
    print("DEVELOPMENTAL TRAJECTORY SUMMARY")
    print("=" * 70)

    for label in TROUGH_LABELS:
        sub = results[results.trough_label == label].sort_values('age_center')
        if len(sub) < 3:
            continue
        depletions = sub['depletion_pct'].values
        ages_c = sub['age_center'].values

        # Find max depletion (deepest trough = most inhibition)
        max_idx = np.argmax(depletions)
        min_idx = np.argmin(depletions)

        # Spearman correlation with age
        from scipy.stats import spearmanr
        rho, p = spearmanr(ages_c, depletions)

        # Test for inverted-U: fit quadratic, check negative leading coefficient
        if len(ages_c) >= 5:
            coeffs = np.polyfit(ages_c, depletions, 2)
            vertex_age = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else np.nan
            is_inverted_u = coeffs[0] < 0 and 5 < vertex_age < 70
        else:
            coeffs = [np.nan, np.nan, np.nan]
            vertex_age = np.nan
            is_inverted_u = False

        print(f"\n  {label}:")
        print(f"    Range: {depletions.min():.1f}% - {depletions.max():.1f}% depletion")
        print(f"    Deepest at age {ages_c[max_idx]:.0f}, shallowest at age {ages_c[min_idx]:.0f}")
        print(f"    Linear trend: rho={rho:.3f}, p={p:.4f}")
        if not np.isnan(vertex_age):
            print(f"    Quadratic fit: a={coeffs[0]:.4f}, vertex={vertex_age:.1f} years")
            print(f"    Inverted-U: {'YES' if is_inverted_u else 'no'}")

    if args.plot:
        generate_plot(results, bin_kdes)


def generate_plot(results, bin_kdes):
    """Publication-quality developmental trajectory figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

    # Panel A: All troughs developmental trajectory
    ax = axes[0, 0]
    for i, (label, color) in enumerate(zip(TROUGH_LABELS, colors)):
        sub = results[results.trough_label == label].sort_values('age_center')
        ax.plot(sub['age_center'], sub['depletion_pct'], 'o-', color=color,
                label=label, markersize=5, linewidth=1.5)
        # Quadratic fit
        if len(sub) >= 5:
            x = sub['age_center'].values
            y = sub['depletion_pct'].values
            coeffs = np.polyfit(x, y, 2)
            x_fit = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=color, alpha=0.4)

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Trough depletion (%)')
    ax.set_title('A. Trough depth across lifespan', fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Panels B-F: individual troughs with more detail
    for i, (label, color) in enumerate(zip(TROUGH_LABELS, colors)):
        ax = axes[(i + 1) // 3, (i + 1) % 3]
        sub = results[results.trough_label == label].sort_values('age_center')
        if len(sub) < 3:
            ax.set_visible(False)
            continue

        x = sub['age_center'].values
        y = sub['depletion_pct'].values
        ax.plot(x, y, 'o-', color=color, markersize=6, linewidth=2)

        # Add N labels
        for _, row in sub.iterrows():
            ax.annotate(f"n={int(row['n_subjects'])}", (row['age_center'], row['depletion_pct']),
                       textcoords="offset points", xytext=(0, 8), fontsize=6, ha='center',
                       color='gray')

        # Quadratic fit
        if len(x) >= 5:
            coeffs = np.polyfit(x, y, 2)
            x_fit = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=color, alpha=0.5)
            vertex = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else np.nan
            if 5 < vertex < 70:
                ax.axvline(vertex, color=color, alpha=0.3, linestyle=':')
                ax.text(vertex, ax.get_ylim()[1], f'vertex={vertex:.0f}',
                       fontsize=7, ha='center', va='top', color=color)

        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Depletion (%)')
        ax.set_title(f'{"BCDEF"[i]}. {label}', fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'trough_depth_by_age.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {path}")
    plt.close()


if __name__ == '__main__':
    main()
