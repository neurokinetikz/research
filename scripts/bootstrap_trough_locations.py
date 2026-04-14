#!/usr/bin/env python3
"""
Bootstrap Trough Locations: Subject-Level Resampling
=====================================================

Addresses the non-independence concern: the pooled KDE treats 3.47M peaks
as independent, but they're nested within ~2K subjects. This script
resamples subjects (not peaks), recomputes the pooled KDE for each
bootstrap iteration, finds troughs, and reports confidence intervals
on trough positions and their ratios.

Usage:
    python scripts/bootstrap_trough_locations.py
    python scripts/bootstrap_trough_locations.py --n-boot 2000 --plot

Outputs to: outputs/bootstrap_troughs/
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from phi_frequency_model import PHI, F0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PEAK_BASE = os.path.join(BASE_DIR, 'exports_adaptive_v3')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'bootstrap_troughs')
MIN_POWER_PCT = 50

EC_DATASETS = {
    'eegmmidb': 'eegmmidb', 'lemon': 'lemon', 'dortmund': 'dortmund',
    'chbmp': 'chbmp', 'hbn_R1': 'hbn_R1', 'hbn_R2': 'hbn_R2',
    'hbn_R3': 'hbn_R3', 'hbn_R4': 'hbn_R4', 'hbn_R6': 'hbn_R6',
}


def load_per_subject_freqs():
    """Load peaks per subject. Returns list of (dataset, subject_id, freq_array)."""
    subjects = []
    for name, subdir in EC_DATASETS.items():
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
            subjects.append((name, subj_id, df['freq'].values))
        print(f"  {name}: {len(files)} subjects")
    return subjects


def find_troughs_from_freqs(all_freqs, n_hist=1000, sigma=8, f_range=(3, 55)):
    """Find density troughs from a frequency array using histogram + smoothing."""
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


def run_bootstrap(subjects, n_boot=1000, seed=42, stratified=True):
    """Bootstrap over subjects, find troughs at each iteration.

    If stratified=True, resamples within each dataset independently,
    preserving the dataset composition across bootstrap iterations.
    This prevents large datasets from dominating the trough geometry.
    """
    rng = np.random.default_rng(seed)

    # Group subjects by dataset
    dataset_groups = {}
    for i, (ds, subj_id, freqs) in enumerate(subjects):
        if ds not in dataset_groups:
            dataset_groups[ds] = []
        dataset_groups[ds].append(i)

    n_subjects = len(subjects)
    freq_arrays = [s[2] for s in subjects]

    # Real troughs (all subjects)
    all_freqs = np.concatenate(freq_arrays)
    real_troughs = find_troughs_from_freqs(all_freqs)
    print(f"\n  Real troughs: {np.round(real_troughs, 2)} Hz")
    if len(real_troughs) >= 2:
        real_ratios = real_troughs[1:] / real_troughs[:-1]
        real_geo = np.exp(np.mean(np.log(real_ratios)))
        print(f"  Real ratios: {np.round(real_ratios, 3)}")
        print(f"  Real geo mean: {real_geo:.4f}")

    if stratified:
        print(f"  Stratified by {len(dataset_groups)} datasets: "
              + ", ".join(f"{k}({len(v)})" for k, v in dataset_groups.items()))

    # Bootstrap
    boot_troughs = []
    boot_geo_means = []
    boot_n_troughs = []
    t0 = time.time()

    for b in range(n_boot):
        if stratified:
            # Resample within each dataset, preserving composition
            idx = []
            for ds, ds_indices in dataset_groups.items():
                sampled = rng.choice(ds_indices, size=len(ds_indices), replace=True)
                idx.extend(sampled)
            idx = np.array(idx)
        else:
            idx = rng.choice(n_subjects, size=n_subjects, replace=True)
        boot_freqs = np.concatenate([freq_arrays[i] for i in idx])

        troughs = find_troughs_from_freqs(boot_freqs)
        boot_troughs.append(troughs)
        boot_n_troughs.append(len(troughs))

        if len(troughs) >= 2:
            ratios = troughs[1:] / troughs[:-1]
            boot_geo_means.append(np.exp(np.mean(np.log(ratios))))

        if (b + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed
            eta = (n_boot - b - 1) / rate
            print(f"    Bootstrap {b+1}/{n_boot}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\n  Bootstrap complete: {n_boot} iterations in {elapsed:.1f}s")

    return real_troughs, boot_troughs, np.array(boot_geo_means), np.array(boot_n_troughs)


def analyze_results(real_troughs, boot_troughs, boot_geo_means, boot_n_troughs):
    """Compute CIs and test candidate constants."""
    os.makedirs(OUT_DIR, exist_ok=True)

    n_boot = len(boot_geo_means)

    print(f"\n{'=' * 70}")
    print(f"  BOOTSTRAP RESULTS ({n_boot} subject-level resamples)")
    print(f"{'=' * 70}")

    # Number of troughs
    print(f"\n  Number of troughs detected:")
    print(f"    Real: {len(real_troughs)}")
    print(f"    Bootstrap median: {np.median(boot_n_troughs):.0f}")
    print(f"    Bootstrap range: {boot_n_troughs.min()}-{boot_n_troughs.max()}")
    for n in range(2, 8):
        pct = (boot_n_troughs == n).mean() * 100
        if pct > 0:
            print(f"    {n} troughs: {pct:.1f}%")

    # Per-trough position CIs
    print(f"\n  Trough position confidence intervals:")
    # Match each bootstrap trough to nearest real trough
    trough_positions = {i: [] for i in range(len(real_troughs))}
    for bt in boot_troughs:
        for t in bt:
            # Assign to nearest real trough
            if len(real_troughs) > 0:
                dists = np.abs(np.log(t) - np.log(real_troughs))
                nearest = np.argmin(dists)
                if dists[nearest] < 0.3:  # within ~35% in Hz
                    trough_positions[nearest].append(t)

    trough_cis = []
    for i, rt in enumerate(real_troughs):
        positions = np.array(trough_positions[i])
        if len(positions) < 10:
            print(f"    Trough {i+1} ({rt:.2f} Hz): insufficient bootstrap samples "
                  f"({len(positions)})")
            trough_cis.append((rt, np.nan, np.nan, np.nan, 0))
            continue
        ci_lo = np.percentile(positions, 2.5)
        ci_hi = np.percentile(positions, 97.5)
        detection_rate = len(positions) / n_boot * 100
        print(f"    Trough {i+1}: {rt:.2f} Hz  "
              f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]  "
              f"width={ci_hi-ci_lo:.2f} Hz  "
              f"detected in {detection_rate:.0f}% of bootstraps")
        trough_cis.append((rt, ci_lo, ci_hi, ci_hi - ci_lo, detection_rate))

    # Geometric mean ratio CI
    if len(boot_geo_means) > 0:
        geo_ci_lo = np.percentile(boot_geo_means, 2.5)
        geo_ci_hi = np.percentile(boot_geo_means, 97.5)
        geo_median = np.median(boot_geo_means)

        print(f"\n  Geometric mean ratio of consecutive troughs:")
        if len(real_troughs) >= 2:
            real_ratios = real_troughs[1:] / real_troughs[:-1]
            real_geo = np.exp(np.mean(np.log(real_ratios)))
            print(f"    Real: {real_geo:.4f}")
        print(f"    Bootstrap median: {geo_median:.4f}")
        print(f"    Bootstrap 95% CI: [{geo_ci_lo:.4f}, {geo_ci_hi:.4f}]")
        print(f"    Bootstrap SD: {boot_geo_means.std():.4f}")

        # Test candidate constants
        print(f"\n  Candidate constants within 95% CI?")
        candidates = {
            'phi': PHI,
            'e-1': np.e - 1,
            'sqrt2': np.sqrt(2),
            'sqrt3': np.sqrt(3),
            'octave': 2.0,
            'e': np.e,
        }
        for name, val in candidates.items():
            within = geo_ci_lo <= val <= geo_ci_hi
            # Bootstrap p-value (two-sided)
            p = 2 * min(
                (boot_geo_means <= val).mean(),
                (boot_geo_means >= val).mean()
            )
            status = "YES" if within else "no"
            print(f"    {name:8s} ({val:.4f}): {status}  "
                  f"p={p:.4f}")

    # Save results
    ci_df = pd.DataFrame(trough_cis,
                          columns=['real_hz', 'ci_lo', 'ci_hi', 'ci_width',
                                   'detection_pct'])
    ci_df.to_csv(os.path.join(OUT_DIR, 'trough_position_cis.csv'), index=False)

    geo_df = pd.DataFrame({
        'boot_geo_mean': boot_geo_means
    })
    geo_df.to_csv(os.path.join(OUT_DIR, 'bootstrap_geo_means.csv'), index=False)

    return trough_cis, boot_geo_means


def generate_plots(real_troughs, trough_cis, boot_geo_means):
    """Publication-quality bootstrap results figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Panel A: Trough position CIs
    ax = axes[0]
    valid = [(rt, lo, hi, w, d) for rt, lo, hi, w, d in trough_cis
             if not np.isnan(lo)]
    for i, (rt, lo, hi, w, d) in enumerate(valid):
        color = '#2ecc71' if d > 80 else '#e67e22' if d > 50 else '#e74c3c'
        ax.errorbar(rt, i, xerr=[[rt - lo], [hi - rt]], fmt='o',
                    color=color, markersize=6, capsize=4, linewidth=1.5)
        ax.text(hi + 0.5, i, f'{d:.0f}%', fontsize=8, va='center', color='gray')

    # Mark phi boundaries
    phi_bnds = [F0 / PHI, F0, F0 * PHI, F0 * PHI ** 2, F0 * PHI ** 3]
    for pb in phi_bnds:
        ax.axvline(pb, color='#D4A017', linewidth=1, alpha=0.5, linestyle='--')

    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels([f'{v[0]:.1f} Hz' for v in valid])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('A. Trough positions (95% CI)', fontweight='bold', fontsize=10)
    ax.invert_yaxis()

    # Panel B: Geometric mean ratio distribution
    ax = axes[1]
    ax.hist(boot_geo_means, bins=50, color='steelblue', alpha=0.7,
            edgecolor='black', linewidth=0.3)

    ci_lo = np.percentile(boot_geo_means, 2.5)
    ci_hi = np.percentile(boot_geo_means, 97.5)
    ax.axvspan(ci_lo, ci_hi, alpha=0.15, color='steelblue',
               label=f'95% CI [{ci_lo:.3f}, {ci_hi:.3f}]')

    # Mark candidates
    markers = {'$\\varphi$': (PHI, '#D4A017'), '$e-1$': (np.e - 1, '#e74c3c'),
               '$\\sqrt{2}$': (np.sqrt(2), '#3498db'), '2': (2.0, '#95a5a6')}
    for label, (val, color) in markers.items():
        ax.axvline(val, color=color, linewidth=2, linestyle='--', label=f'{label} = {val:.3f}')

    ax.set_xlabel('Geometric mean ratio')
    ax.set_ylabel('Bootstrap count')
    ax.set_title('B. Inter-trough ratio (subject bootstrap)', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'bootstrap_troughs.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\n  Plot saved to {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap trough locations over subjects')
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading per-subject peak data...")
    subjects = load_per_subject_freqs()
    print(f"\nTotal: {len(subjects)} subjects")

    real_troughs, boot_troughs, boot_geo_means, boot_n_troughs = run_bootstrap(
        subjects, n_boot=args.n_boot)

    trough_cis, boot_geo_means = analyze_results(
        real_troughs, boot_troughs, boot_geo_means, boot_n_troughs)

    if args.plot:
        generate_plots(real_troughs, trough_cis, boot_geo_means)


if __name__ == '__main__':
    main()
