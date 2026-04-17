#!/usr/bin/env python3
"""
Raw PSD Trough Test: Do Troughs Exist Before Decomposition?
============================================================

The critical test: compute grand-average power spectra from raw EEG
(no FOOOF, no IRASA), remove the 1/f trend with a simple non-parametric
method, and look for density minima. If the troughs are real spectral
features, they must be visible in the raw data.

Uses LEMON (N=203, clean data, 59 channels) and Dortmund (N=608, 64 channels)
as primary datasets. Computes Welch PSD per subject per channel, averages
across channels, then examines population-level spectral structure.

Usage:
    python scripts/raw_psd_trough_test.py
    python scripts/raw_psd_trough_test.py --max-subjects 50  # quick test

Outputs to: outputs/raw_psd_troughs/
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'raw_psd_troughs')

# Data locations
LEMON_PATH = '/Volumes/T9/lemon_data/eeg_preprocessed/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed'
DORTMUND_PATH = '/Volumes/T9/dortmund_data_dl'

FOOOF_TROUGHS = np.array([5.03, 7.82, 13.59, 24.75, 34.38])
TROUGH_LABELS = ['δ/θ', 'θ/α', 'α/β', 'βL/βH', 'βH/γ']

PHI = (1 + np.sqrt(5)) / 2


def load_lemon_subjects(max_subjects=None):
    """Load LEMON EC .set files using MNE."""
    try:
        import mne
        mne.set_log_level('ERROR')
    except ImportError:
        print("  MNE not available, skipping LEMON")
        return []

    set_files = sorted(glob.glob(os.path.join(LEMON_PATH, '*_EC.set')))
    if max_subjects:
        set_files = set_files[:max_subjects]

    subjects = []
    for f in set_files:
        subj_id = os.path.basename(f).replace('_EC.set', '')
        try:
            raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
            # Pick only EEG channels, exclude EOG/misc
            raw.pick_types(eeg=True, exclude='bads')
            data = raw.get_data()  # (n_channels, n_samples)
            fs = raw.info['sfreq']
            subjects.append((subj_id, data, fs))
        except Exception as e:
            continue

    print(f"  LEMON: loaded {len(subjects)} subjects")
    return subjects


def load_dortmund_subjects(max_subjects=None):
    """Load Dortmund EC .set files using MNE."""
    try:
        import mne
        mne.set_log_level('ERROR')
    except ImportError:
        print("  MNE not available, skipping Dortmund")
        return []

    # Dortmund structure: sub-*/eeg/sub-*_task-EyesClosed_eeg.set
    set_files = sorted(glob.glob(os.path.join(
        DORTMUND_PATH, 'sub-*', 'eeg', 'sub-*_task-EyesClosed_eeg.set')))
    if not set_files:
        # Try alternative structure
        set_files = sorted(glob.glob(os.path.join(
            DORTMUND_PATH, 'derivatives', 'preprocessed', 'sub-*', 'eeg', '*EyesClosed*.set')))
    if not set_files:
        set_files = sorted(glob.glob(os.path.join(DORTMUND_PATH, '**', '*EyesClosed*.set'), recursive=True))

    if max_subjects:
        set_files = set_files[:max_subjects]

    subjects = []
    for f in set_files:
        subj_id = os.path.basename(f).split('_')[0]
        try:
            raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
            raw.pick_types(eeg=True, exclude='bads')
            data = raw.get_data()
            fs = raw.info['sfreq']
            subjects.append((subj_id, data, fs))
        except Exception as e:
            continue

    print(f"  Dortmund: loaded {len(subjects)} subjects")
    return subjects


def compute_subject_psd(data, fs, nperseg=None, fmin=1, fmax=55):
    """Compute Welch PSD averaged across channels for one subject."""
    n_channels, n_samples = data.shape

    if nperseg is None:
        nperseg = min(int(fs * 4), n_samples)  # 4-second windows
    noverlap = nperseg // 2

    freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap,
                       axis=1)  # psd shape: (n_channels, n_freqs)

    # Average across channels (geometric mean in log space)
    log_psd_mean = np.mean(np.log10(np.maximum(psd, 1e-30)), axis=0)

    # Frequency mask
    mask = (freqs >= fmin) & (freqs <= fmax)

    return freqs[mask], log_psd_mean[mask]


def remove_aperiodic(freqs, log_psd, method='median_filter', kernel_frac=0.4):
    """Remove 1/f trend from log-PSD using non-parametric method."""
    if method == 'median_filter':
        # Median filter in log-frequency space
        kernel_size = max(3, int(len(log_psd) * kernel_frac))
        if kernel_size % 2 == 0:
            kernel_size += 1
        aperiodic = median_filter(log_psd, size=kernel_size)
        oscillatory = log_psd - aperiodic
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        window = max(5, int(len(log_psd) * kernel_frac))
        if window % 2 == 0:
            window += 1
        aperiodic = savgol_filter(log_psd, window, 2)
        oscillatory = log_psd - aperiodic
    elif method == 'polynomial':
        # Fit polynomial in log-log space
        log_freqs = np.log10(freqs)
        coeffs = np.polyfit(log_freqs, log_psd, 2)
        aperiodic = np.polyval(coeffs, log_freqs)
        oscillatory = log_psd - aperiodic
    else:
        raise ValueError(f"Unknown method: {method}")

    return oscillatory, aperiodic


def find_spectral_troughs(freqs, oscillatory, sigma=3):
    """Find troughs in the oscillatory (1/f-removed) spectrum."""
    smoothed = gaussian_filter1d(oscillatory, sigma=sigma)
    trough_idx, props = find_peaks(-smoothed, prominence=0.01, distance=10)
    return freqs[trough_idx], smoothed[trough_idx], smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-subjects', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Raw PSD Trough Test: Do Troughs Exist Before Decomposition?")
    print("=" * 70)

    # Load subjects
    print("\n--- Loading EEG Data ---")
    all_subjects = []

    lemon = load_lemon_subjects(args.max_subjects)
    all_subjects.extend([('lemon', *s) for s in lemon])

    if not args.max_subjects or args.max_subjects > 50:
        dortmund = load_dortmund_subjects(args.max_subjects)
        all_subjects.extend([('dortmund', *s) for s in dortmund])

    if len(all_subjects) == 0:
        print("  No subjects loaded! Check data paths.")
        return

    print(f"\n  Total: {len(all_subjects)} subjects")

    # Compute per-subject PSD
    print("\n--- Computing Per-Subject PSD ---")
    psd_collection = []
    common_freqs = None

    for i, (dataset, subj_id, data, fs) in enumerate(all_subjects):
        try:
            freqs, log_psd = compute_subject_psd(data, fs)
            psd_collection.append({
                'dataset': dataset, 'subject': subj_id,
                'freqs': freqs, 'log_psd': log_psd, 'fs': fs,
            })

            if common_freqs is None:
                common_freqs = freqs
        except Exception as e:
            continue

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(all_subjects)} subjects processed")

    print(f"  Successfully computed PSD for {len(psd_collection)} subjects")

    if len(psd_collection) < 10:
        print("  Too few subjects, aborting")
        return

    # Interpolate all PSDs to common frequency grid
    from scipy.interpolate import interp1d

    freq_grid = np.linspace(1, 50, 500)
    psd_matrix = []

    for entry in psd_collection:
        f_interp = interp1d(entry['freqs'], entry['log_psd'],
                           kind='linear', fill_value='extrapolate')
        psd_matrix.append(f_interp(freq_grid))

    psd_matrix = np.array(psd_matrix)
    grand_mean = np.mean(psd_matrix, axis=0)
    grand_median = np.median(psd_matrix, axis=0)

    print(f"\n  Grand average PSD: {psd_matrix.shape[0]} subjects × {len(freq_grid)} freqs")

    # --- Remove aperiodic and find troughs ---
    print("\n--- Aperiodic Removal and Trough Detection ---")

    methods = ['median_filter', 'polynomial']
    all_trough_results = []

    for method in methods:
        for kernel in [0.3, 0.4, 0.5] if method == 'median_filter' else [None]:
            label = f"{method}" + (f"_k{kernel}" if kernel else "")
            kw = {'kernel_frac': kernel} if kernel else {}

            osc_mean, ap_mean = remove_aperiodic(freq_grid, grand_mean, method=method, **kw)
            osc_median, ap_median = remove_aperiodic(freq_grid, grand_median, method=method, **kw)

            # Find troughs
            for source, osc, desc in [('mean', osc_mean, 'grand mean'),
                                       ('median', osc_median, 'grand median')]:
                trough_hz, trough_val, smoothed = find_spectral_troughs(freq_grid, osc)
                trough_hz_filt = trough_hz[(trough_hz > 4) & (trough_hz < 45)]

                print(f"\n  {label} ({desc}):")
                print(f"    Troughs: {np.round(trough_hz_filt, 2)} Hz")

                for th in trough_hz_filt:
                    # Match to known troughs
                    dists = np.abs(np.log(th) - np.log(FOOOF_TROUGHS))
                    nearest_idx = np.argmin(dists)
                    nearest_label = TROUGH_LABELS[nearest_idx]
                    nearest_dist = th - FOOOF_TROUGHS[nearest_idx]
                    all_trough_results.append({
                        'method': label, 'source': source,
                        'trough_hz': th, 'nearest_fooof': FOOOF_TROUGHS[nearest_idx],
                        'nearest_label': nearest_label, 'delta_hz': nearest_dist,
                    })

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Raw PSD Troughs vs FOOOF Troughs")
    print("=" * 70)

    df_results = pd.DataFrame(all_trough_results)

    for label in TROUGH_LABELS:
        matches = df_results[df_results.nearest_label == label]
        if len(matches) > 0:
            mean_pos = matches['trough_hz'].mean()
            std_pos = matches['trough_hz'].std()
            n_detections = len(matches)
            total_methods = len(df_results.groupby(['method', 'source']))
            print(f"  {label}: detected {n_detections}/{total_methods} times, "
                  f"mean = {mean_pos:.2f} ± {std_pos:.2f} Hz "
                  f"(FOOOF: {FOOOF_TROUGHS[TROUGH_LABELS.index(label)]:.2f})")
        else:
            print(f"  {label}: NOT detected in any method")

    # Geometric mean ratio of detected troughs
    # Use median_filter_k0.4 mean as the "best" estimate
    best = df_results[(df_results.method == 'median_filter_k0.4') & (df_results.source == 'mean')]
    if len(best) >= 2:
        best_troughs = np.sort(best['trough_hz'].values)
        ratios = best_troughs[1:] / best_troughs[:-1]
        geo_mean = np.exp(np.mean(np.log(ratios)))
        print(f"\n  Geometric mean ratio (raw PSD, best method): {geo_mean:.4f}")
        print(f"  FOOOF geometric mean: 1.6172")
        print(f"  φ = {PHI:.4f}")

    # Save
    df_results.to_csv(os.path.join(OUT_DIR, 'raw_psd_trough_detections.csv'), index=False)

    # Save grand average PSD for plotting
    df_psd = pd.DataFrame({'freq_hz': freq_grid, 'grand_mean_log_psd': grand_mean,
                           'grand_median_log_psd': grand_median})
    df_psd.to_csv(os.path.join(OUT_DIR, 'grand_average_psd.csv'), index=False)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
