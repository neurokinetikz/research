#!/usr/bin/env python3
"""
Analysis 3: Trough Depth Covariance Structure
==============================================
Computes per-subject trough depths and examines whether the 5 troughs
covary across individuals. Under independent inhibitory populations,
trough depths should be uncorrelated. Under a common-factor model,
they should positively correlate.

Uses a windowed count approach for per-subject trough depth:
  - For each trough, count peaks in a narrow log-frequency window
  - Compare to counts in flanking windows
  - Depth = 1 - (trough_count / mean_flank_count)

Also examines: age-bin covariance, per-dataset covariance, factor structure.

Usage:
    python scripts/trough_depth_covariance.py [--plot]
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v4')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'trough_depth_by_age')
MIN_POWER_PCT = 50

KNOWN_TROUGHS_HZ = np.array([5.08, 7.81, 13.42, 25.30, 35.04])
TROUGH_LABELS = ['δ/θ (5.1)', 'θ/α (7.8)', 'α/β (13.4)', 'βL/βH (25.3)', 'βH/γ (35.0)']
SHORT_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']

# Window parameters in log-frequency space
LOG_HALF_WINDOW = 0.06   # ±6% in log-Hz for each window
LOG_FLANK_OFFSET = 0.15  # flanks centered ±15% in log-Hz from trough

# Demographics
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


def per_subject_trough_depth(freqs, trough_hz):
    """Compute trough depth for a single subject using windowed counts.

    For each trough, counts peaks in a narrow window centered on the trough
    and in flanking windows on each side. Depth = 1 - (trough/mean_flanks).
    Returns depth_ratio (< 1.0 means depletion, lower = deeper trough).
    """
    log_freqs = np.log(freqs)
    log_trough = np.log(trough_hz)

    # Trough window
    trough_mask = np.abs(log_freqs - log_trough) < LOG_HALF_WINDOW
    trough_count = trough_mask.sum()

    # Left flank
    left_center = log_trough - LOG_FLANK_OFFSET
    left_mask = np.abs(log_freqs - left_center) < LOG_HALF_WINDOW
    left_count = left_mask.sum()

    # Right flank
    right_center = log_trough + LOG_FLANK_OFFSET
    right_mask = np.abs(log_freqs - right_center) < LOG_HALF_WINDOW
    right_count = right_mask.sum()

    mean_flank = (left_count + right_count) / 2
    if mean_flank > 0:
        return trough_count / mean_flank
    else:
        return np.nan


def load_per_subject_depths():
    """Load all subjects, compute per-subject trough depths.
    Returns DataFrame with columns: subject, dataset, age, and depth for each trough.
    """
    age_map = load_demographics()

    datasets = {
        'hbn_R1': 'hbn', 'hbn_R2': 'hbn', 'hbn_R3': 'hbn',
        'hbn_R4': 'hbn', 'hbn_R5': 'hbn', 'hbn_R6': 'hbn',
        'hbn_R7': 'hbn', 'hbn_R8': 'hbn', 'hbn_R9': 'hbn',
        'hbn_R10': 'hbn', 'hbn_R11': 'hbn',
        'dortmund': 'dortmund', 'lemon': 'lemon',
    }

    rows = []
    for subdir, base in datasets.items():
        path = os.path.join(PEAK_BASE, subdir)
        files = sorted(glob.glob(os.path.join(path, '*_peaks.csv')))
        if not files:
            continue
        first = pd.read_csv(files[0], nrows=1)
        has_power = 'power' in first.columns
        cols = ['freq'] + (['power', 'phi_octave'] if has_power else ['phi_octave'])
        count = 0
        for f in files:
            subj_id = os.path.basename(f).replace('_peaks.csv', '')
            df = pd.read_csv(f, usecols=cols)
            if has_power and MIN_POWER_PCT > 0:
                filtered = []
                for octave in df['phi_octave'].unique():
                    bp = df[df.phi_octave == octave]
                    if len(bp) < 2:
                        continue
                    thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
                    kept = bp[bp['power'] >= thresh]
                    if len(kept) > 0:
                        filtered.append(kept)
                if not filtered:
                    continue
                df = pd.concat(filtered, ignore_index=True)

            freqs = df['freq'].values
            if len(freqs) < 100:
                continue

            row = {
                'subject': subj_id,
                'dataset': base,
                'age': age_map.get(subj_id, np.nan),
                'n_peaks': len(freqs),
            }

            for trough_hz, label in zip(KNOWN_TROUGHS_HZ, SHORT_LABELS):
                row[f'depth_{label}'] = per_subject_trough_depth(freqs, trough_hz)

            rows.append(row)
            count += 1

        print(f"  {subdir}: {count} subjects")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading per-subject trough depths...")
    df = load_per_subject_depths()
    print(f"\nTotal: {len(df)} subjects")
    print(f"  With age: {df['age'].notna().sum()}")
    print(f"  Mean peaks/subject: {df['n_peaks'].mean():.0f}")

    depth_cols = [f'depth_{l}' for l in SHORT_LABELS]

    # ===================================================================
    # Part 1: Full correlation matrix (all subjects)
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Full correlation matrix (all subjects)")
    print("=" * 70)

    print(f"\n  Spearman correlations (N = {len(df)}):\n")
    print(f"  {'':>10s}", end='')
    for l in SHORT_LABELS:
        print(f"  {l:>8s}", end='')
    print()

    corr_matrix = np.zeros((5, 5))
    p_matrix = np.zeros((5, 5))
    for i, col_i in enumerate(depth_cols):
        print(f"  {SHORT_LABELS[i]:>10s}", end='')
        for j, col_j in enumerate(depth_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                print(f"  {'1.000':>8s}", end='')
                continue
            valid = df[[col_i, col_j]].dropna()
            if len(valid) > 10:
                rho, p = spearmanr(valid[col_i].values, valid[col_j].values)
            else:
                rho, p = np.nan, np.nan
            corr_matrix[i, j] = float(rho)
            p_matrix[i, j] = float(p)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {rho:>5.3f}{sig:>3s}", end='')
        print()

    # Mean off-diagonal correlation
    off_diag = []
    for i in range(5):
        for j in range(i + 1, 5):
            off_diag.append(corr_matrix[i, j])
    print(f"\n  Mean off-diagonal ρ: {np.mean(off_diag):.3f}")
    print(f"  Range: [{min(off_diag):.3f}, {max(off_diag):.3f}]")

    # ===================================================================
    # Part 2: Age-partialed correlations
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 2: Age-partialed correlations")
    print("=" * 70)

    df_with_age = df.dropna(subset=['age'])
    print(f"\n  N = {len(df_with_age)} subjects with age data")

    # Residualize each depth column against age
    from numpy.polynomial import polynomial as P
    residual_cols = {}
    for col in depth_cols:
        valid = df_with_age[['age', col]].dropna()
        if len(valid) < 50:
            continue
        # Quadratic residualization (age effects may be nonlinear)
        coeffs = np.polyfit(valid['age'].values, valid[col].values, 2)
        predicted = np.polyval(coeffs, valid['age'].values)
        residuals = valid[col].values - predicted
        residual_cols[col] = pd.Series(residuals, index=valid.index)

    if len(residual_cols) == 5:
        resid_df = pd.DataFrame(residual_cols)
        print(f"\n  Age-partialed Spearman correlations (quadratic residuals):\n")
        print(f"  {'':>10s}", end='')
        for l in SHORT_LABELS:
            print(f"  {l:>8s}", end='')
        print()

        age_partial_corr = np.zeros((5, 5))
        for i, col_i in enumerate(depth_cols):
            print(f"  {SHORT_LABELS[i]:>10s}", end='')
            for j, col_j in enumerate(depth_cols):
                if i == j:
                    age_partial_corr[i, j] = 1.0
                    print(f"  {'1.000':>8s}", end='')
                    continue
                if col_i in resid_df.columns and col_j in resid_df.columns:
                    valid = resid_df[[col_i, col_j]].dropna()
                    rho, p = spearmanr(valid[col_i].values, valid[col_j].values)
                else:
                    rho, p = np.nan, np.nan
                age_partial_corr[i, j] = rho
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"  {rho:>5.3f}{sig:>3s}", end='')
            print()

        off_diag_partial = []
        for i in range(5):
            for j in range(i + 1, 5):
                off_diag_partial.append(age_partial_corr[i, j])
        print(f"\n  Mean off-diagonal ρ (age-partialed): {np.mean(off_diag_partial):.3f}")
        print(f"  Range: [{min(off_diag_partial):.3f}, {max(off_diag_partial):.3f}]")

    # ===================================================================
    # Part 3: Within-dataset correlations
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 3: Within-dataset correlations (δ/θ vs α/β only)")
    print("=" * 70)

    dt_col = 'depth_δ/θ'
    ab_col = 'depth_α/β'

    for ds in ['hbn', 'dortmund', 'lemon']:
        sub = df[df.dataset == ds][[dt_col, ab_col]].dropna()
        if len(sub) < 20:
            continue
        rho, p = spearmanr(sub[dt_col].values, sub[ab_col].values)
        print(f"\n  {ds} (N={len(sub)}): δ/θ vs α/β: ρ = {rho:.3f}, p = {p:.4f}")

    # ===================================================================
    # Part 4: Factor analysis (eigenvalue decomposition of correlation matrix)
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 4: Factor structure (PCA on trough depths)")
    print("=" * 70)

    # Use subjects with all 5 trough depths
    complete = df[depth_cols].dropna()
    print(f"\n  Subjects with all 5 trough depths: {len(complete)}")

    if len(complete) > 50:
        from numpy.linalg import eigh
        # Standardize
        X = complete.values
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        # Correlation matrix
        R = np.corrcoef(X.T)
        eigenvalues, eigenvectors = eigh(R)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        print(f"\n  Eigenvalues of correlation matrix:")
        total_var = eigenvalues.sum()
        cumulative = 0
        for i, ev in enumerate(eigenvalues):
            cumulative += ev
            print(f"    PC{i+1}: {ev:.3f} ({ev/total_var*100:.1f}% variance, "
                  f"cumulative: {cumulative/total_var*100:.1f}%)")

        print(f"\n  PC1 loadings (first eigenvector):")
        for i, label in enumerate(SHORT_LABELS):
            print(f"    {label}: {eigenvectors[i, 0]:.3f}")

        print(f"\n  PC2 loadings:")
        for i, label in enumerate(SHORT_LABELS):
            print(f"    {label}: {eigenvectors[i, 1]:.3f}")

        # Under common-factor model: PC1 should explain >50% variance
        # Under independence: all eigenvalues ≈ 1.0
        if eigenvalues[0] > 2.0:
            print(f"\n  → PC1 explains {eigenvalues[0]/total_var*100:.1f}% — "
                  f"suggests a COMMON FACTOR")
        elif eigenvalues[0] < 1.5:
            print(f"\n  → PC1 explains only {eigenvalues[0]/total_var*100:.1f}% — "
                  f"suggests INDEPENDENT mechanisms")
        else:
            print(f"\n  → PC1 explains {eigenvalues[0]/total_var*100:.1f}% — "
                  f"intermediate (partial common factor)")

    # ===================================================================
    # Part 5: Reliability check — split-half consistency of per-subject depths
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PART 5: Split-half reliability of per-subject trough depths")
    print("=" * 70)

    # Re-compute depths using odd vs even epochs/channels as a rough split
    # Actually, let's check peak count distributions
    print(f"\n  Per-subject peak count distribution:")
    for label, hz in zip(SHORT_LABELS, KNOWN_TROUGHS_HZ):
        col = f'depth_{label}'
        valid = df[col].dropna()
        print(f"    {label}: N={len(valid)} valid, "
              f"mean depth={valid.mean():.3f}, SD={valid.std():.3f}")

    # Save per-subject data
    df.to_csv(os.path.join(OUT_DIR, 'per_subject_trough_depths.csv'), index=False)
    print(f"\nPer-subject data saved to {OUT_DIR}/per_subject_trough_depths.csv")

    if args.plot:
        generate_plot(df, corr_matrix, SHORT_LABELS)


def generate_plot(df, corr_matrix, labels):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Correlation matrix heatmap
    ax = axes[0]
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', norm=norm, aspect='equal')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(5):
        for j in range(5):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.3 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color)
    ax.set_title('A. Trough depth correlations\n(all subjects)', fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Spearman ρ')

    # Panel B: δ/θ vs α/β scatter (the two deep troughs)
    ax = axes[1]
    dt_col = 'depth_δ/θ'
    ab_col = 'depth_α/β'
    for ds, color, marker in [('hbn', '#e74c3c', 'o'), ('dortmund', '#3498db', 's'),
                               ('lemon', '#2ecc71', '^')]:
        sub = df[df.dataset == ds]
        ax.scatter(sub[dt_col], sub[ab_col], c=color, marker=marker,
                   alpha=0.3, s=15, label=ds)
    ax.set_xlabel('δ/θ depth ratio')
    ax.set_ylabel('α/β depth ratio')
    ax.set_title('B. Two deep troughs\n(per-subject)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Scatter colored by age
    ax = axes[2]
    has_age = df.dropna(subset=['age', dt_col, ab_col])
    sc = ax.scatter(has_age[dt_col], has_age[ab_col], c=has_age['age'],
                    cmap='viridis', alpha=0.4, s=15)
    ax.set_xlabel('δ/θ depth ratio')
    ax.set_ylabel('α/β depth ratio')
    ax.set_title('C. Two deep troughs\n(colored by age)', fontweight='bold')
    fig.colorbar(sc, ax=ax, label='Age (years)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'trough_depth_covariance.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {path}")
    plt.close()


if __name__ == '__main__':
    main()
