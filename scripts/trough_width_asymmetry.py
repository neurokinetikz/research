#!/usr/bin/env python3
"""
Analysis 4: Trough Width and Asymmetry
=======================================
Characterizes each trough's shape beyond depth: width at half-depth
and left/right flank asymmetry. Under the inhibitory model:
  - Inhibitory ceiling (δ/θ): steeper high-frequency flank
  - Inhibitory floor (α/β): steeper low-frequency flank
  - Excitatory attractor competition (θ/α): more symmetric

Also examines developmental changes in width/asymmetry within HBN
and Dortmund.

Usage:
    python scripts/trough_width_asymmetry.py [--plot]
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

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
    age_map = {}
    for release in HBN_RELEASES:
        tsv = HBN_DEMO_TEMPLATE.format(release=release)
        if os.path.exists(tsv):
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


def load_peaks(age_map, dataset_filter=None):
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
            df = pd.read_csv(f, usecols=cols)
            if has_power and MIN_POWER_PCT > 0:
                filtered = []
                for octave in df['phi_octave'].unique():
                    bp = df[df.phi_octave == octave]
                    thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                    filtered.append(bp[bp['power'] >= thresh])
                df = pd.concat(filtered, ignore_index=True)
            subjects.append((subj_id, base, age_map.get(subj_id, np.nan), df['freq'].values))
    return subjects


def measure_trough_shapes(freqs, n_hist=1000, sigma=8, f_range=(3, 55)):
    """Measure width, asymmetry, and flank slopes for each trough.

    Returns dict of {label: {width_hz, width_log, asymmetry, left_slope, right_slope,
                             depth_ratio, trough_actual_hz}}.

    Asymmetry = (right_slope - left_slope) / (right_slope + left_slope)
    Positive asymmetry = steeper right (high-freq) flank.
    """
    log_freqs = np.log(freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)
    bin_width_log = log_centers[1] - log_centers[0]

    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)
    envelope = gaussian_filter1d(counts.astype(float), sigma=40)

    results = {}
    for trough_hz, label in zip(KNOWN_TROUGHS_HZ, TROUGH_LABELS):
        trough_idx = np.argmin(np.abs(hz_centers - trough_hz))
        trough_val = smoothed[trough_idx]
        env_val = envelope[trough_idx]
        depth_ratio = trough_val / env_val if env_val > 0 else np.nan

        # Find the peaks (local maxima) flanking this trough
        # Search left for the nearest peak
        left_peak_idx = trough_idx
        for k in range(trough_idx - 1, max(0, trough_idx - 200), -1):
            if smoothed[k] >= smoothed[k - 1] and smoothed[k] >= smoothed[k + 1]:
                left_peak_idx = k
                break

        # Search right for the nearest peak
        right_peak_idx = trough_idx
        for k in range(trough_idx + 1, min(n_hist - 1, trough_idx + 200)):
            if smoothed[k] >= smoothed[k - 1] and smoothed[k] >= smoothed[k + 1]:
                right_peak_idx = k
                break

        left_peak_val = smoothed[left_peak_idx]
        right_peak_val = smoothed[right_peak_idx]

        # Half-depth level (midpoint between trough and mean of flanking peaks)
        half_level = trough_val + 0.5 * ((left_peak_val + right_peak_val) / 2 - trough_val)

        # Width at half-depth: find where smoothed crosses half_level on each side
        left_half_idx = trough_idx
        for k in range(trough_idx, left_peak_idx, -1):
            if smoothed[k] >= half_level:
                left_half_idx = k
                break

        right_half_idx = trough_idx
        for k in range(trough_idx, right_peak_idx):
            if smoothed[k] >= half_level:
                right_half_idx = k
                break

        width_log = (right_half_idx - left_half_idx) * bin_width_log
        width_hz = hz_centers[right_half_idx] - hz_centers[left_half_idx]

        # Flank slopes (in density units per log-Hz)
        left_dist = (trough_idx - left_peak_idx) * bin_width_log
        right_dist = (right_peak_idx - trough_idx) * bin_width_log

        left_slope = abs(left_peak_val - trough_val) / left_dist if left_dist > 0 else np.nan
        right_slope = abs(right_peak_val - trough_val) / right_dist if right_dist > 0 else np.nan

        # Asymmetry index: positive = steeper right (high-freq) flank
        denom = left_slope + right_slope
        if denom > 0 and not np.isnan(left_slope) and not np.isnan(right_slope):
            asymmetry = (right_slope - left_slope) / denom
        else:
            asymmetry = np.nan

        # Left vs right half-widths (in log-space)
        left_half_width = (trough_idx - left_half_idx) * bin_width_log
        right_half_width = (right_half_idx - trough_idx) * bin_width_log
        if left_half_width + right_half_width > 0:
            width_asymmetry = (right_half_width - left_half_width) / (right_half_width + left_half_width)
        else:
            width_asymmetry = np.nan

        results[label] = {
            'depth_ratio': depth_ratio,
            'depletion_pct': (1 - depth_ratio) * 100 if not np.isnan(depth_ratio) else np.nan,
            'width_log': width_log,
            'width_hz': width_hz,
            'left_slope': left_slope,
            'right_slope': right_slope,
            'slope_asymmetry': asymmetry,
            'width_asymmetry': width_asymmetry,
            'left_peak_hz': hz_centers[left_peak_idx],
            'right_peak_hz': hz_centers[right_peak_idx],
            'left_half_width': left_half_width,
            'right_half_width': right_half_width,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    age_map = load_demographics()
    all_subjects = load_peaks(age_map)
    hbn_subjects = [s for s in all_subjects if s[1] == 'hbn']
    dort_subjects = [s for s in all_subjects if s[1] == 'dortmund']
    print(f"  Total: {len(all_subjects)} | HBN: {len(hbn_subjects)} | Dort: {len(dort_subjects)}")

    # ===================================================================
    # Part 1: Pooled trough shapes
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Pooled trough shapes (all subjects)")
    print("=" * 70)

    all_freqs = np.concatenate([s[3] for s in all_subjects])
    pooled = measure_trough_shapes(all_freqs)

    print(f"\n  {'Trough':>15s} | {'Depth':>7s} | {'Width':>6s} | "
          f"{'L slope':>8s} | {'R slope':>8s} | {'Slope asym':>10s} | {'Width asym':>10s}")
    print("  " + "-" * 85)
    for label in TROUGH_LABELS:
        s = pooled[label]
        print(f"  {label:>15s} | {s['depletion_pct']:5.1f}% | "
              f"{s['width_log']:.3f} | {s['left_slope']:8.1f} | {s['right_slope']:8.1f} | "
              f"{s['slope_asymmetry']:+.3f}      | {s['width_asymmetry']:+.3f}")

    print(f"\n  Flanking peaks:")
    for label in TROUGH_LABELS:
        s = pooled[label]
        print(f"    {label}: left peak at {s['left_peak_hz']:.1f} Hz, "
              f"right peak at {s['right_peak_hz']:.1f} Hz")

    print(f"\n  Asymmetry interpretation:")
    print(f"    Positive slope_asymmetry = steeper right (high-freq) flank")
    print(f"    Negative slope_asymmetry = steeper left (low-freq) flank")
    for label in TROUGH_LABELS:
        s = pooled[label]
        asym = s['slope_asymmetry']
        if asym > 0.1:
            desc = "steeper HIGH-freq flank"
        elif asym < -0.1:
            desc = "steeper LOW-freq flank"
        else:
            desc = "approximately SYMMETRIC"
        print(f"    {label}: asymmetry = {asym:+.3f} → {desc}")

    # ===================================================================
    # Part 2: HBN developmental trajectory of shape
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 2: HBN developmental changes in trough shape")
    print("=" * 70)

    hbn_bins = [(5, 8), (8, 11), (11, 14), (14, 17), (17, 21)]
    rows = []
    for lo, hi in hbn_bins:
        bin_subj = [s for s in hbn_subjects if lo <= s[2] < hi]
        if len(bin_subj) < 30:
            continue
        bin_freqs = np.concatenate([s[3] for s in bin_subj])
        shapes = measure_trough_shapes(bin_freqs)
        center = (lo + hi) / 2

        for label in TROUGH_LABELS:
            s = shapes[label]
            rows.append({
                'cohort': 'HBN', 'age_lo': lo, 'age_hi': hi, 'age_center': center,
                'n_subjects': len(bin_subj), 'trough': label,
                **{k: v for k, v in s.items()},
            })

    hbn_df = pd.DataFrame(rows)

    # Print trajectory for key troughs
    for label in ['δ/θ (5.1)', 'α/β (13.4)', 'θ/α (7.8)']:
        sub = hbn_df[hbn_df.trough == label].sort_values('age_center')
        if len(sub) < 3:
            continue
        print(f"\n  {label}:")
        print(f"    {'Age':>8s} | {'Depth':>7s} | {'Width':>6s} | {'S.asym':>7s} | {'W.asym':>7s}")
        for _, r in sub.iterrows():
            print(f"    {r['age_center']:6.1f}y | {r['depletion_pct']:5.1f}% | "
                  f"{r['width_log']:.3f} | {r['slope_asymmetry']:+.3f} | {r['width_asymmetry']:+.3f}")

        # Correlation of asymmetry with age
        x = sub['age_center'].values
        for metric in ['slope_asymmetry', 'width_log']:
            y = sub[metric].values
            valid = ~np.isnan(y)
            if valid.sum() >= 3:
                rho, p = spearmanr(x[valid], y[valid])
                print(f"    {metric} vs age: ρ = {rho:.3f}, p = {p:.3f}")

    # ===================================================================
    # Part 3: Dortmund aging trajectory of shape
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 3: Dortmund aging changes in trough shape")
    print("=" * 70)

    dort_bins = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70)]
    dort_rows = []
    for lo, hi in dort_bins:
        bin_subj = [s for s in dort_subjects if lo <= s[2] < hi]
        if len(bin_subj) < 30:
            continue
        bin_freqs = np.concatenate([s[3] for s in bin_subj])
        shapes = measure_trough_shapes(bin_freqs)
        center = (lo + hi) / 2

        for label in TROUGH_LABELS:
            s = shapes[label]
            dort_rows.append({
                'cohort': 'Dortmund', 'age_lo': lo, 'age_hi': hi, 'age_center': center,
                'n_subjects': len(bin_subj), 'trough': label,
                **{k: v for k, v in s.items()},
            })

    dort_df = pd.DataFrame(dort_rows)

    for label in ['α/β (13.4)', 'θ/α (7.8)', 'βL/βH (25.3)']:
        sub = dort_df[dort_df.trough == label].sort_values('age_center')
        if len(sub) < 3:
            continue
        print(f"\n  {label}:")
        print(f"    {'Age':>8s} | {'Depth':>7s} | {'Width':>6s} | {'S.asym':>7s} | {'W.asym':>7s}")
        for _, r in sub.iterrows():
            print(f"    {r['age_center']:6.1f}y | {r['depletion_pct']:5.1f}% | "
                  f"{r['width_log']:.3f} | {r['slope_asymmetry']:+.3f} | {r['width_asymmetry']:+.3f}")

        x = sub['age_center'].values
        for metric in ['slope_asymmetry', 'width_log']:
            y = sub[metric].values
            valid = ~np.isnan(y)
            if valid.sum() >= 3:
                rho, p = spearmanr(x[valid], y[valid])
                print(f"    {metric} vs age: ρ = {rho:.3f}, p = {p:.3f}")

    # Save all results
    all_rows = rows + dort_rows
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(OUT_DIR, 'trough_shapes.csv'), index=False)

    # Save pooled summary
    pooled_rows = []
    for label in TROUGH_LABELS:
        pooled_rows.append({'trough': label, **pooled[label]})
    pd.DataFrame(pooled_rows).to_csv(os.path.join(OUT_DIR, 'trough_shapes_pooled.csv'), index=False)
    print(f"\nResults saved to {OUT_DIR}/trough_shapes*.csv")

    if args.plot:
        generate_plot(pooled, hbn_df, dort_df)


def generate_plot(pooled, hbn_df, dort_df):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Pooled shape summary (asymmetry vs width)
    ax = axes[0]
    for i, label in enumerate(TROUGH_LABELS):
        s = pooled[label]
        ax.scatter(s['width_log'], s['slope_asymmetry'], c=colors[i], s=150,
                   zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(label, (s['width_log'], s['slope_asymmetry']),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Trough width (log-Hz)')
    ax.set_ylabel('Slope asymmetry\n(+right steeper, -left steeper)')
    ax.set_title('A. Trough shape (pooled)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel B: HBN developmental changes in asymmetry
    ax = axes[1]
    for i, label in enumerate(['δ/θ (5.1)', 'θ/α (7.8)', 'α/β (13.4)']):
        sub = hbn_df[hbn_df.trough == label].sort_values('age_center')
        if len(sub) >= 3:
            ax.plot(sub['age_center'], sub['slope_asymmetry'], 'o-',
                    color=colors[TROUGH_LABELS.index(label)], label=label,
                    markersize=5, linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Slope asymmetry')
    ax.set_title('B. Asymmetry development (HBN)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Dortmund aging changes in asymmetry
    ax = axes[2]
    for i, label in enumerate(['α/β (13.4)', 'θ/α (7.8)', 'βL/βH (25.3)']):
        sub = dort_df[dort_df.trough == label].sort_values('age_center')
        if len(sub) >= 3:
            ax.plot(sub['age_center'], sub['slope_asymmetry'], 'o-',
                    color=colors[TROUGH_LABELS.index(label)], label=label,
                    markersize=5, linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Slope asymmetry')
    ax.set_title('C. Asymmetry aging (Dortmund)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'trough_shapes.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {path}")
    plt.close()


if __name__ == '__main__':
    main()
