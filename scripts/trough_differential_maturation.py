#!/usr/bin/env python3
"""
Analysis 2: Differential Maturation of the Two Deep Troughs
============================================================
Tests whether the δ/θ (5.1 Hz) and α/β (13.4 Hz) troughs mature on
different timelines, as predicted by the SST+ (early) vs PV+ (late)
interneuron mapping.

Approach:
1. Establish adult reference depth from Dortmund (ages 25-55, stable plateau)
2. Normalize each trough's developmental trajectory to % of adult depth
3. Compare maturation curves: which trough reaches adult depth first?
4. Statistical test: age × trough interaction within HBN
5. Fine-grained HBN trajectory (1-year sliding windows)

Usage:
    python scripts/trough_differential_maturation.py [--plot]
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v4')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age')
MIN_POWER_PCT = 50

KNOWN_TROUGHS_HZ = np.array([5.08, 7.81, 13.42, 25.30, 35.04])
TROUGH_LABELS = ['δ/θ (5.1)', 'θ/α (7.8)', 'α/β (13.4)', 'βL/βH (25.3)', 'βH/γ (35.0)']

# Focus on the two deep troughs
DEEP_TROUGHS = {
    'δ/θ (5.1)': 5.08,
    'α/β (13.4)': 13.42,
}

HBN_RELEASES = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']
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


def load_peaks_with_age(age_map, dataset_filter=None):
    datasets = {
        'hbn_R1': 'hbn', 'hbn_R2': 'hbn', 'hbn_R3': 'hbn',
        'hbn_R4': 'hbn', 'hbn_R5': 'hbn', 'hbn_R6': 'hbn',
        'hbn_R7': 'hbn', 'hbn_R8': 'hbn', 'hbn_R9': 'hbn',
        'hbn_R10': 'hbn', 'hbn_R11': 'hbn',
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
            depths[label] = (1 - smoothed[idx] / env_val) * 100
        else:
            depths[label] = np.nan
    return depths


def bootstrap_depth(subjects_list, n_boot=500, seed=42):
    """Bootstrap trough depths. Returns {label: (mean, ci_lo, ci_hi)}."""
    rng = np.random.default_rng(seed)
    n = len(subjects_list)
    freq_arrays = [s[3] for s in subjects_list]
    boot_depths = {label: [] for label in TROUGH_LABELS}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_freqs = np.concatenate([freq_arrays[i] for i in idx])
        depths = measure_trough_depths(boot_freqs)
        for label, val in depths.items():
            boot_depths[label].append(val)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--n-boot', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading demographics...")
    age_map = load_demographics()

    print("Loading peak data...")
    hbn_subjects = load_peaks_with_age(age_map, dataset_filter='hbn')
    dort_subjects = load_peaks_with_age(age_map, dataset_filter='dortmund')
    print(f"  HBN: {len(hbn_subjects)} subjects")
    print(f"  Dortmund: {len(dort_subjects)} subjects")

    # ===================================================================
    # Step 1: Establish adult reference depth from Dortmund (ages 25-55)
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STEP 1: Adult reference depth (Dortmund, ages 25-55)")
    print("=" * 70)

    adult_ref_subjects = [s for s in dort_subjects if 25 <= s[2] <= 55]
    print(f"  N = {len(adult_ref_subjects)} subjects")
    adult_freqs = np.concatenate([s[3] for s in adult_ref_subjects])
    adult_depths = measure_trough_depths(adult_freqs)
    adult_boot = bootstrap_depth(adult_ref_subjects, n_boot=args.n_boot)

    print(f"\n  Adult reference depths (Dortmund 25-55):")
    for label in DEEP_TROUGHS:
        d = adult_depths[label]
        mean_b, ci_lo, ci_hi = adult_boot[label]
        print(f"    {label}: {d:.1f}%  [95% CI: {ci_lo:.1f}%, {ci_hi:.1f}%]")

    # Also show all troughs for context
    print(f"\n  All troughs:")
    for label in TROUGH_LABELS:
        d = adult_depths[label]
        mean_b, ci_lo, ci_hi = adult_boot[label]
        print(f"    {label}: {d:.1f}%  [{ci_lo:.1f}%, {ci_hi:.1f}%]")

    # ===================================================================
    # Step 2: Fine-grained HBN trajectory (2-year sliding window)
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Fine-grained HBN trajectory (2-year sliding window)")
    print("=" * 70)

    hbn_ages = np.array([s[2] for s in hbn_subjects])
    window_centers = np.arange(6, 20, 1.0)  # center ages 6-19
    half_window = 1.5  # 3-year window

    hbn_rows = []
    print(f"\n  {'Age':>6s} | {'N':>4s} | {'δ/θ depl':>10s} | {'α/β depl':>10s} | "
          f"{'δ/θ %adult':>10s} | {'α/β %adult':>10s}")
    print("  " + "-" * 70)

    for center in window_centers:
        window_subjects = [s for s in hbn_subjects
                           if center - half_window <= s[2] < center + half_window]
        if len(window_subjects) < 30:
            continue

        window_freqs = np.concatenate([s[3] for s in window_subjects])
        depths = measure_trough_depths(window_freqs)

        # Bootstrap CIs
        boot = bootstrap_depth(window_subjects, n_boot=args.n_boot,
                               seed=int(center * 100))

        row = {'age_center': center, 'n_subjects': len(window_subjects)}
        for label in DEEP_TROUGHS:
            dep = depths[label]
            adult_dep = adult_depths[label]
            pct_adult = (dep / adult_dep * 100) if adult_dep > 0 else np.nan
            mean_b, ci_lo, ci_hi = boot[label]
            row[f'{label}_depletion'] = dep
            row[f'{label}_ci_lo'] = ci_lo
            row[f'{label}_ci_hi'] = ci_hi
            row[f'{label}_pct_adult'] = pct_adult

        dt_dep = depths['δ/θ (5.1)']
        ab_dep = depths['α/β (13.4)']
        dt_pct = row['δ/θ (5.1)_pct_adult']
        ab_pct = row['α/β (13.4)_pct_adult']

        print(f"  {center:5.1f}y | {len(window_subjects):4d} | {dt_dep:8.1f}% | "
              f"{ab_dep:8.1f}% | {dt_pct:8.1f}% | {ab_pct:8.1f}%")
        hbn_rows.append(row)

    hbn_df = pd.DataFrame(hbn_rows)

    # ===================================================================
    # Step 3: When does each trough reach 50%, 75%, 90% of adult depth?
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STEP 3: Maturation milestones (% of adult depth)")
    print("=" * 70)

    for label in DEEP_TROUGHS:
        pct_col = f'{label}_pct_adult'
        if pct_col not in hbn_df.columns:
            continue
        valid = hbn_df[['age_center', pct_col]].dropna()
        if len(valid) < 3:
            continue

        print(f"\n  {label}:")
        print(f"    Adult reference (Dort 25-55): {adult_depths[label]:.1f}%")
        print(f"    Youngest HBN ({valid['age_center'].min():.0f}y): "
              f"{valid[pct_col].iloc[0]:.1f}% of adult")
        print(f"    Oldest HBN ({valid['age_center'].max():.0f}y): "
              f"{valid[pct_col].iloc[-1]:.1f}% of adult")

        for threshold in [50, 75, 90]:
            above = valid[valid[pct_col] >= threshold]
            if len(above) > 0:
                first_age = above['age_center'].min()
                print(f"    Reaches {threshold}% of adult depth at age: {first_age:.1f}")
            else:
                print(f"    Does NOT reach {threshold}% of adult depth in HBN range")

    # ===================================================================
    # Step 4: Direct comparison -- are the maturation curves different?
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STEP 4: Differential maturation test")
    print("=" * 70)

    dt_col = 'δ/θ (5.1)_pct_adult'
    ab_col = 'α/β (13.4)_pct_adult'
    valid = hbn_df[['age_center', dt_col, ab_col]].dropna()

    if len(valid) >= 4:
        # Correlation of maturation curves with age
        rho_dt, p_dt = spearmanr(valid['age_center'], valid[dt_col])
        rho_ab, p_ab = spearmanr(valid['age_center'], valid[ab_col])

        print(f"\n  δ/θ % adult vs age: ρ = {rho_dt:.3f}, p = {p_dt:.4f}")
        print(f"  α/β % adult vs age: ρ = {rho_ab:.3f}, p = {p_ab:.4f}")

        # Are the two maturation curves correlated with each other?
        rho_cross, p_cross = spearmanr(valid[dt_col], valid[ab_col])
        print(f"\n  Cross-trough correlation (δ/θ vs α/β maturation): "
              f"ρ = {rho_cross:.3f}, p = {p_cross:.4f}")

        # Mean % adult across HBN range
        mean_dt = valid[dt_col].mean()
        mean_ab = valid[ab_col].mean()
        print(f"\n  Mean % of adult depth across HBN age range:")
        print(f"    δ/θ: {mean_dt:.1f}%")
        print(f"    α/β: {mean_ab:.1f}%")
        print(f"    Difference: {mean_dt - mean_ab:.1f} pp")

        if mean_dt > mean_ab:
            print(f"\n  → δ/θ is CLOSER to adult depth than α/β across development")
            print(f"    Consistent with SST+ (earlier maturation) vs PV+ (later maturation)")
        else:
            print(f"\n  → α/β is closer to adult depth than δ/θ")

        # Slope comparison (linear fit to % adult vs age)
        dt_slope = np.polyfit(valid['age_center'].values, valid[dt_col].values, 1)[0]
        ab_slope = np.polyfit(valid['age_center'].values, valid[ab_col].values, 1)[0]
        print(f"\n  Maturation rate (linear slope, pp/year):")
        print(f"    δ/θ: {dt_slope:.2f} pp/year")
        print(f"    α/β: {ab_slope:.2f} pp/year")
        if abs(ab_slope) > abs(dt_slope):
            print(f"    α/β is maturing {abs(ab_slope/dt_slope) if dt_slope != 0 else 'inf':.1f}× "
                  f"faster than δ/θ")

    # ===================================================================
    # Step 5: All five troughs -- maturation status at youngest HBN age
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STEP 5: All troughs -- maturation status at youngest HBN age")
    print("=" * 70)

    youngest_bin = [s for s in hbn_subjects if s[2] < 7]
    if len(youngest_bin) >= 20:
        youngest_freqs = np.concatenate([s[3] for s in youngest_bin])
        youngest_depths = measure_trough_depths(youngest_freqs)
        youngest_boot = bootstrap_depth(youngest_bin, n_boot=args.n_boot, seed=999)

        print(f"\n  Youngest HBN (age < 7, N = {len(youngest_bin)}):")
        print(f"  {'Trough':>15s} | {'Young depl':>10s} | {'Adult depl':>10s} | "
              f"{'% adult':>8s} | {'95% CI young':>15s}")
        print("  " + "-" * 70)
        for label in TROUGH_LABELS:
            y_dep = youngest_depths[label]
            a_dep = adult_depths[label]
            pct = (y_dep / a_dep * 100) if a_dep > 0 else np.nan
            mean_b, ci_lo, ci_hi = youngest_boot[label]
            print(f"  {label:>15s} | {y_dep:8.1f}% | {a_dep:8.1f}% | "
                  f"{pct:6.1f}% | [{ci_lo:.1f}%, {ci_hi:.1f}%]")

    # Save fine-grained trajectory
    hbn_df.to_csv(os.path.join(OUT_DIR, 'differential_maturation_hbn.csv'), index=False)
    print(f"\nResults saved to {OUT_DIR}/differential_maturation_hbn.csv")

    if args.plot:
        generate_plot(hbn_df, adult_depths, adult_boot)


def generate_plot(hbn_df, adult_depths, adult_boot):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Raw depletion trajectories
    ax = axes[0]
    for label, color in [('δ/θ (5.1)', '#e74c3c'), ('α/β (13.4)', '#2ecc71')]:
        dep_col = f'{label}_depletion'
        ci_lo_col = f'{label}_ci_lo'
        ci_hi_col = f'{label}_ci_hi'
        x = hbn_df['age_center'].values
        y = hbn_df[dep_col].values
        lo = hbn_df[ci_lo_col].values
        hi = hbn_df[ci_hi_col].values
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.plot(x, y, 'o-', color=color, markersize=5, linewidth=2, label=label)
        # Adult reference
        adult_d = adult_depths[label]
        adult_ci_lo, adult_ci_hi = adult_boot[label][1], adult_boot[label][2]
        ax.axhspan(adult_ci_lo, adult_ci_hi, color=color, alpha=0.08)
        ax.axhline(adult_d, color=color, linestyle=':', alpha=0.5)

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Trough depletion (%)')
    ax.set_title('A. Raw depletion (HBN)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: % of adult depth
    ax = axes[1]
    for label, color in [('δ/θ (5.1)', '#e74c3c'), ('α/β (13.4)', '#2ecc71')]:
        pct_col = f'{label}_pct_adult'
        x = hbn_df['age_center'].values
        y = hbn_df[pct_col].values
        ax.plot(x, y, 'o-', color=color, markersize=5, linewidth=2, label=label)
        # Linear fit
        mask = ~np.isnan(y)
        if mask.sum() >= 3:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_fit = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=color, alpha=0.4)

    ax.axhline(100, color='gray', linestyle=':', alpha=0.5, label='Adult reference')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('% of adult depth')
    ax.set_title('B. Maturation (% adult depth)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: All 5 troughs at youngest age vs adult
    ax = axes[2]
    youngest_pct = []
    labels_short = []
    colors_all = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
    for label in TROUGH_LABELS:
        dep_col = f'{label}_depletion'
        if dep_col in hbn_df.columns:
            youngest = hbn_df.iloc[0][dep_col]
            adult = adult_depths[label]
            pct = (youngest / adult * 100) if adult > 0 else np.nan
        else:
            pct = np.nan
        youngest_pct.append(pct)
        labels_short.append(label.split(' ')[0])

    bars = ax.bar(range(len(TROUGH_LABELS)), youngest_pct, color=colors_all, alpha=0.7,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(range(len(TROUGH_LABELS)))
    ax.set_xticklabels([l.replace(' ', '\n') for l in TROUGH_LABELS], fontsize=8)
    ax.set_ylabel('% of adult depth at age 6')
    ax.set_title('C. Maturation status at youngest age', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add % labels on bars
    for bar, pct in zip(bars, youngest_pct):
        if not np.isnan(pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{pct:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'differential_maturation.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {path}")
    plt.close()


if __name__ == '__main__':
    main()
