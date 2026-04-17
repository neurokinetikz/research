#!/usr/bin/env python3
"""
TDBRAIN Regional Enrichment and Trough Analyses
=================================================

1. Regional enrichment: frontal/central/temporal/parietal/occipital
2. Trough detection pooled and by age bin
3. Trough depth developmental trajectory (pooled KDE method, not per-subject)

Usage:
    python scripts/tdbrain_regional_trough.py
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
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'tdbrain_analysis')
PEAK_DIR = os.path.join(BASE_DIR, 'exports_adaptive_v4', 'tdbrain')
PARTICIPANTS_PATH = os.path.expanduser(
    '~/Desktop/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.tsv')

sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
from phi_frequency_model import PHI, F0

MIN_POWER_PCT = 50

# TDBRAIN 26-channel regional grouping (10-10 system)
REGIONS = {
    'frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4'],
    'central': ['C3', 'Cz', 'C4'],
    'temporal': ['T7', 'T8'],
    'parietal': ['CP3', 'CPz', 'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8'],
    'occipital': ['O1', 'Oz', 'O2'],
}

KNOWN_TROUGHS_HZ = np.array([5.08, 7.81, 13.42, 25.30, 35.04])
TROUGH_LABELS = ['δ/θ (5.1)', 'θ/α (7.8)', 'α/β (13.4)', 'βL/βH (25.3)', 'βH/γ (35.0)']

AGE_BINS = [(5, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 60), (60, 90)]


def load_peaks_by_channel(peak_dir, subject_ids=None):
    """Load peaks with channel information."""
    files = sorted(glob.glob(os.path.join(peak_dir, '*_peaks.csv')))
    dfs = []
    for f in files:
        sub_id = os.path.basename(f).replace('_peaks.csv', '')
        if subject_ids is not None and sub_id not in subject_ids:
            continue
        try:
            df = pd.read_csv(f, usecols=['channel', 'freq', 'power', 'phi_octave'])
        except Exception:
            continue
        # Power filter per band
        filtered = []
        for octave in df['phi_octave'].unique():
            bp = df[df.phi_octave == octave]
            if len(bp) == 0:
                continue
            thresh = bp['power'].quantile(MIN_POWER_PCT / 100)
            filtered.append(bp[bp['power'] >= thresh])
        if filtered:
            df = pd.concat(filtered, ignore_index=True)
            df['subject'] = sub_id
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def find_troughs(freqs, n_hist=1000, sigma=8, f_range=(3, 55)):
    """Find density troughs."""
    log_freqs = np.log(freqs)
    log_edges = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_hist + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    hz_centers = np.exp(log_centers)
    counts, _ = np.histogram(log_freqs, bins=log_edges)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)
    median_val = np.median(smoothed[smoothed > 0])
    trough_idx, _ = find_peaks(-smoothed, prominence=median_val * 0.05,
                                distance=n_hist // 30)
    return hz_centers[trough_idx], smoothed


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
            depletion = (1 - depth_ratio) * 100
        else:
            depth_ratio, depletion = np.nan, np.nan
        depths[label] = {'depth_ratio': depth_ratio, 'depletion_pct': depletion}
    return depths


def load_participants():
    """Load TDBRAIN participants."""
    df = pd.read_csv(PARTICIPANTS_PATH, sep='\t')
    df['age_float'] = df['age'].str.replace(',', '.').astype(float)
    pid_str = df['participants_ID'].astype(str)
    df['subject_key'] = pid_str.where(pid_str.str.startswith('sub-'), 'sub-' + pid_str)
    df = df[df['DISC/REP'] == 'DISCOVERY']
    return df


# =============================================================
# 1. Regional Enrichment
# =============================================================
def analysis_regional():
    print("\n" + "=" * 70)
    print("1. REGIONAL ENRICHMENT")
    print("=" * 70)

    all_peaks = load_peaks_by_channel(PEAK_DIR)
    print(f"  Total peaks: {len(all_peaks):,} from {all_peaks['subject'].nunique()} subjects")

    # Assign channels to regions
    channel_to_region = {}
    for region, channels in REGIONS.items():
        for ch in channels:
            channel_to_region[ch] = region

    all_peaks['region'] = all_peaks['channel'].map(channel_to_region)

    print(f"\n  Peaks by region:")
    for region in ['frontal', 'central', 'temporal', 'parietal', 'occipital']:
        n = (all_peaks['region'] == region).sum()
        pct = n / len(all_peaks) * 100
        print(f"    {region}: {n:,} ({pct:.1f}%)")

    # Trough detection per region
    print(f"\n  Troughs by region:")
    region_results = []
    for region in ['frontal', 'central', 'temporal', 'parietal', 'occipital']:
        region_freqs = all_peaks[all_peaks.region == region]['freq'].values
        if len(region_freqs) < 1000:
            print(f"    {region}: insufficient peaks")
            continue
        troughs, _ = find_troughs(region_freqs)
        troughs = troughs[(troughs > 4) & (troughs < 50)]
        print(f"    {region}: {np.round(troughs, 2)} Hz")

        # Trough depths per region
        depths = measure_trough_depths(region_freqs)
        for label, d in depths.items():
            region_results.append({
                'region': region, 'trough': label,
                'depletion_pct': d['depletion_pct'],
                'depth_ratio': d['depth_ratio'],
            })

    df_r = pd.DataFrame(region_results)
    if len(df_r) > 0:
        print(f"\n  Trough depths by region:")
        pivot = df_r.pivot(index='trough', columns='region', values='depletion_pct')
        pivot = pivot[['frontal', 'central', 'temporal', 'parietal', 'occipital']]
        print(pivot.round(1).to_string())
        df_r.to_csv(os.path.join(OUT_DIR, 'tdbrain_regional_trough_depths.csv'), index=False)

    # Alpha mountain by region (is it occipital-dominant?)
    print(f"\n  Alpha peak density by region (u ∈ [0.3, 0.7]):")
    for region in ['frontal', 'central', 'temporal', 'parietal', 'occipital']:
        rp = all_peaks[(all_peaks.region == region) & (all_peaks.phi_octave == 'n+0')]
        if len(rp) > 10:
            u = (np.log(rp['freq'].values / F0) / np.log(PHI)) % 1.0
            mid_frac = ((u >= 0.3) & (u <= 0.7)).mean()
            print(f"    {region}: {mid_frac:.1%} in mid-octave (mountain)")


# =============================================================
# 2. Trough Depth Developmental Trajectory
# =============================================================
def analysis_trough_trajectory():
    print("\n" + "=" * 70)
    print("2. TROUGH DEPTH DEVELOPMENTAL TRAJECTORY")
    print("=" * 70)

    demo = load_participants()
    age_map = dict(zip(demo['subject_key'], demo['age_float']))

    all_peaks = load_peaks_by_channel(PEAK_DIR)
    all_peaks['age'] = all_peaks['subject'].map(age_map)

    results = []
    print(f"\n  {'Age bin':<12} {'N':>6} {'Peaks':>10} {'δ/θ':>8} {'θ/α':>8} {'α/β':>8} {'βL/βH':>8} {'βH/γ':>8}")
    print("  " + "-" * 75)

    for lo, hi in AGE_BINS:
        bin_peaks = all_peaks[(all_peaks.age >= lo) & (all_peaks.age < hi)]
        n_subj = bin_peaks['subject'].nunique()
        if n_subj < 20:
            continue

        freqs = bin_peaks['freq'].values
        depths = measure_trough_depths(freqs)

        depls = [depths[label]['depletion_pct'] for label in TROUGH_LABELS]
        print(f"  {lo:>2}-{hi:<8} {n_subj:>6} {len(freqs):>10,} " +
              "  ".join(f"{d:>6.1f}%" if not np.isnan(d) else "   nan" for d in depls))

        for label, d in depths.items():
            results.append({
                'age_lo': lo, 'age_hi': hi,
                'age_center': (lo + hi) / 2,
                'n_subjects': n_subj,
                'n_peaks': len(freqs),
                'trough': label,
                'depletion_pct': d['depletion_pct'],
                'depth_ratio': d['depth_ratio'],
            })

    df_r = pd.DataFrame(results)
    if len(df_r) > 0:
        # Developmental trajectory per trough
        print(f"\n  Age correlation per trough:")
        for label in TROUGH_LABELS:
            sub = df_r[df_r.trough == label].dropna(subset=['depletion_pct'])
            if len(sub) >= 3:
                rho, p = stats.spearmanr(sub['age_center'], sub['depletion_pct'])
                direction = '↑ deepens' if rho > 0 else '↓ fills'
                print(f"    {label}: ρ = {rho:+.3f} (p = {p:.3f}) {direction}")

        df_r.to_csv(os.path.join(OUT_DIR, 'tdbrain_trough_depth_by_age.csv'), index=False)

    # Also compute geometric mean ratio per age bin
    print(f"\n  Geometric mean ratio per age bin:")
    for lo, hi in AGE_BINS:
        bin_peaks = all_peaks[(all_peaks.age >= lo) & (all_peaks.age < hi)]
        if bin_peaks['subject'].nunique() < 20:
            continue
        freqs = bin_peaks['freq'].values
        troughs, _ = find_troughs(freqs)
        troughs = troughs[(troughs > 4) & (troughs < 50)]
        if len(troughs) >= 2:
            ratios = troughs[1:] / troughs[:-1]
            geo = np.exp(np.mean(np.log(ratios)))
            print(f"    {lo:>2}-{hi:<4}: {len(troughs)} troughs, geo mean = {geo:.4f} "
                  f"(φ = {PHI:.4f})")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TDBRAIN REGIONAL AND TROUGH ANALYSES")
    print("=" * 70)

    analysis_regional()
    analysis_trough_trajectory()

    print(f"\n\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
