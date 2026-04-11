#!/usr/bin/env python3
"""
P20 Overlap-Trim Extraction for HBN — RestingState, Per-Release
================================================================

Extracts FOOOF peaks through full n+3 gamma octave (FREQ_CEIL=55 Hz).
No notch filter needed — 60 Hz is above our range.
Each HBN release is treated as a separate dataset.

HBN: 5 releases (R1-R4, R6), 129 channels, 500 Hz → 250 Hz, SET format.

Usage:
    python scripts/run_hbn_p20_extraction.py --release R1
    python scripts/run_hbn_p20_extraction.py --release R4 --resume-from 100
    python scripts/run_hbn_p20_extraction.py --release all
"""

import os
import sys
import time
import argparse
import logging
import warnings
import gc
from glob import glob

import numpy as np
import pandas as pd
from scipy.signal import welch
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from lemon_utils import _get_peak_params, _get_aperiodic_params, _get_r_squared

try:
    from specparam import SpectralModel
except ImportError:
    from fooof import FOOOF as SpectralModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# =========================================================================
PHI = (1 + np.sqrt(5)) / 2
F0_DEFAULT = 7.83

TARGET_FS = 250.0
NPERSEG = 1000           # 4s at 250 Hz → 0.25 Hz resolution
FREQ_CEIL = 55.0         # Full n+3 octave + margin
FREQ_FLOOR = 1.0
R2_MIN = 0.70
PAD_OCTAVES = 0.5
FILTER_LO = 1.0

DATA_DIR = '/Volumes/T9/hbn_data'
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_hbn')

RELEASES = ['R1', 'R2', 'R3', 'R4', 'R6']

FOOOF_BASE_PARAMS = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
)


# =========================================================================
# BAND BUILDING
# =========================================================================

def build_target_bands(f0, freq_ceil=FREQ_CEIL, freq_floor=FREQ_FLOOR,
                       offset=0.0, base=None):
    if base is None:
        base = PHI
    bands = []
    for n in range(-4, 10):
        target_lo = f0 * base ** (n + offset)
        target_hi = f0 * base ** (n + 1 + offset)
        if target_lo >= freq_ceil:
            break
        if target_hi <= freq_floor:
            continue
        target_lo = max(target_lo, freq_floor)
        target_hi = min(target_hi, freq_ceil)
        if target_hi - target_lo < 0.5:
            continue
        fit_lo = f0 * base ** (n + offset - PAD_OCTAVES)
        fit_hi = f0 * base ** (n + 1 + offset + PAD_OCTAVES)
        fit_lo = max(fit_lo, freq_floor)
        fit_hi = min(fit_hi, freq_ceil)
        bands.append({'name': f'n{n:+d}', 'n': n,
                      'target_lo': target_lo, 'target_hi': target_hi,
                      'fit_lo': fit_lo, 'fit_hi': fit_hi})
    return bands


def merge_narrow_targets(bands, min_width_hz=1.5, min_bins=12, freq_res=0.25):
    merged = []
    i = 0
    while i < len(bands):
        b = bands[i].copy()
        width = b['target_hi'] - b['target_lo']
        n_bins = int(width / freq_res)
        if width < min_width_hz or n_bins < min_bins:
            while (i + 1 < len(bands) and
                   (b['target_hi'] - b['target_lo'] < min_width_hz or
                    int((b['target_hi'] - b['target_lo']) / freq_res) < min_bins)):
                i += 1
                b['target_hi'] = bands[i]['target_hi']
                b['fit_hi'] = bands[i]['fit_hi']
            b['name'] = f'{b["name"]}_merged'
        if b['fit_lo'] > b['target_lo']:
            b['fit_lo'] = b['target_lo'] * 0.7
        if b['fit_hi'] < b['target_hi']:
            b['fit_hi'] = b['target_hi'] * 1.3
        merged.append(b)
        i += 1
    return merged


# =========================================================================
# DATA LOADING
# =========================================================================

def discover_resting_subjects(release_dir):
    """Find all subjects with RestingState .set files in a release."""
    pattern = os.path.join(release_dir, 'sub-*', 'eeg',
                           'sub-*_task-RestingState_eeg.set')
    files = sorted(glob(pattern))
    # Deduplicate by subject ID
    subjects = []
    seen = set()
    for f in files:
        basename = os.path.basename(f)
        sub_id = basename.split('_task-')[0]
        if sub_id not in seen:
            subjects.append((sub_id, f))
            seen.add(sub_id)
    return subjects


def load_subject_raw(set_path, freq_ceil=FREQ_CEIL):
    """Load HBN .set file, pick EEG, downsample, filter."""
    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception as e:
        return None, {'status': f'load_error: {e}'}

    # Pick EEG channels only
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(eeg_picks) == 0:
        return None, {'status': 'no_eeg_channels'}
    raw.pick(eeg_picks)

    # Downsample from 500 Hz to 250 Hz
    if raw.info['sfreq'] > TARGET_FS:
        raw.resample(TARGET_FS, verbose=False)

    # Bandpass filter
    filter_hi = min(freq_ceil + 5, raw.info['sfreq'] / 2 - 1)
    raw.filter(FILTER_LO, filter_hi, verbose=False)

    # NO notch filter — 60 Hz is above FREQ_CEIL of 55 Hz

    duration = raw.n_times / raw.info['sfreq']
    if duration < 30:
        return None, {'status': f'too_short_{duration:.0f}s'}

    info = {
        'status': 'ok',
        'n_channels': len(raw.ch_names),
        'duration_sec': duration,
        'sfreq': raw.info['sfreq'],
    }
    return raw, info


# =========================================================================
# CORE EXTRACTION
# =========================================================================

def extract_overlap_trim_subject(raw_clean, f0, fs=TARGET_FS,
                                 nperseg=NPERSEG, overlap=0.5,
                                 freq_ceil=FREQ_CEIL, offset=0.0,
                                 r2_min=R2_MIN, base=None):
    freq_res = fs / nperseg
    raw_bands = build_target_bands(f0, freq_ceil, FREQ_FLOOR, offset, base=base)
    bands = merge_narrow_targets(raw_bands, min_width_hz=1.5,
                                 min_bins=12, freq_res=freq_res)
    ch_names = raw_clean.ch_names
    noverlap = int(nperseg * overlap)
    all_peaks = []
    band_stats = []

    for band in bands:
        bname = band['name']
        target_lo, target_hi = band['target_lo'], band['target_hi']
        fit_lo, fit_hi = band['fit_lo'], band['fit_hi']
        fit_width = fit_hi - fit_lo

        max_n_peaks = max(3, min(15, int(fit_width / 1.5)))
        max_peak_width = min(fit_width * 0.6, 12.0)
        peak_width_limits = [max(0.5, 2 * freq_res), max_peak_width]

        fooof_params = {**FOOOF_BASE_PARAMS,
                        'max_n_peaks': max_n_peaks,
                        'peak_width_limits': peak_width_limits}

        band_r2s, band_ap = [], []
        n_fitted, n_passed, n_kept, n_trimmed = 0, 0, 0, 0

        for ch in ch_names:
            try:
                data = raw_clean.get_data(picks=[ch])[0]
            except Exception:
                continue
            if len(data) < nperseg:
                continue

            freqs, psd = welch(data, fs, nperseg=nperseg, noverlap=noverlap)
            if (freqs >= fit_lo).sum() < 10:
                continue

            n_fitted += 1
            sm = SpectralModel(**fooof_params)
            try:
                sm.fit(freqs, psd, [fit_lo, fit_hi])
            except Exception:
                continue

            r2 = _get_r_squared(sm)
            if np.isnan(r2) or r2 < r2_min:
                continue

            n_passed += 1
            band_r2s.append(r2)
            _, exp = _get_aperiodic_params(sm)
            band_ap.append(exp)

            for row in _get_peak_params(sm):
                if target_lo <= row[0] < target_hi:
                    all_peaks.append({
                        'channel': ch, 'freq': row[0], 'power': row[1],
                        'bandwidth': row[2], 'phi_octave': bname,
                        'phi_octave_n': band['n'],
                        'target_lo': target_lo, 'target_hi': target_hi,
                        'fit_lo': fit_lo, 'fit_hi': fit_hi})
                    n_kept += 1
                else:
                    n_trimmed += 1

        band_stats.append({
            'band_name': bname, 'band_n': band['n'],
            'target_lo': round(target_lo, 3), 'target_hi': round(target_hi, 3),
            'fit_lo': round(fit_lo, 3), 'fit_hi': round(fit_hi, 3),
            'n_channels_fitted': n_fitted, 'n_channels_passed': n_passed,
            'mean_r_squared': np.mean(band_r2s) if band_r2s else np.nan,
            'mean_aperiodic_exponent': np.mean(band_ap) if band_ap else np.nan,
            'peaks_kept': n_kept, 'peaks_trimmed': n_trimmed})

    peaks_df = pd.DataFrame(all_peaks) if all_peaks else pd.DataFrame(
        columns=['channel', 'freq', 'power', 'bandwidth',
                 'phi_octave', 'phi_octave_n',
                 'target_lo', 'target_hi', 'fit_lo', 'fit_hi'])
    return peaks_df, pd.DataFrame(band_stats)


# =========================================================================
# MAIN
# =========================================================================

def process_release(release, f0, freq_ceil, resume_from=None, output_dir=None):
    """Process all RestingState subjects in one HBN release."""
    release_dir = os.path.join(DATA_DIR, f'cmi_bids_{release}')
    if not os.path.isdir(release_dir):
        log.error(f"Release directory not found: {release_dir}")
        return

    out_dir = output_dir or os.path.join(
        OUTPUT_BASE, f'p20_{release}_f0{f0:.2f}')
    os.makedirs(out_dir, exist_ok=True)

    subjects = discover_resting_subjects(release_dir)
    log.info(f"\nHBN {release}: {len(subjects)} subjects with RestingState")
    log.info(f"  Output: {out_dir}")

    if resume_from:
        subjects = subjects[resume_from:]
        log.info(f"  Resuming from index {resume_from} ({len(subjects)} remaining)")

    summary_rows = []
    t_start = time.time()

    for i, (sub_id, set_path) in enumerate(subjects):
        out_path = os.path.join(out_dir, f'{sub_id}_peaks.csv')
        band_path = os.path.join(out_dir, f'{sub_id}_band_info.csv')

        if os.path.exists(out_path):
            log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: skipping (exists)")
            continue

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw, info = load_subject_raw(set_path, freq_ceil=freq_ceil)

        if raw is None:
            log.warning(f"  [{i+1}/{len(subjects)}] {sub_id} — {info['status']}")
            summary_rows.append({
                'subject_id': sub_id, 'status': info['status'], 'n_peaks': 0})
            continue

        try:
            peaks_df, band_info_df = extract_overlap_trim_subject(
                raw, f0=f0, fs=raw.info['sfreq'], nperseg=NPERSEG,
                overlap=0.5, freq_ceil=freq_ceil)
        except Exception as e:
            log.error(f"  {sub_id}: {e}")
            summary_rows.append({
                'subject_id': sub_id, 'status': f'error: {e}', 'n_peaks': 0})
            del raw; gc.collect()
            continue

        peaks_df.to_csv(out_path, index=False)
        band_info_df.to_csv(band_path, index=False)

        elapsed = time.time() - t0
        n_peaks = len(peaks_df)
        mean_r2 = band_info_df['mean_r_squared'].mean()

        summary_rows.append({
            'subject_id': sub_id, 'status': 'ok', 'n_peaks': n_peaks,
            'n_channels': info['n_channels'],
            'duration_sec': round(info['duration_sec'], 1),
            'mean_r_squared': round(mean_r2, 4) if not np.isnan(mean_r2) else np.nan,
            'time_sec': round(elapsed, 1)})

        log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: {n_peaks} peaks "
                 f"R²={mean_r2:.3f} ch={info['n_channels']} "
                 f"dur={info['duration_sec']:.0f}s {elapsed:.1f}s")

        del raw; gc.collect()

        if (i + 1) % 20 == 0:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    total_time = time.time() - t_start
    n_ok = (summary_df['status'] == 'ok').sum() if len(summary_df) > 0 else 0
    log.info(f"\n  {release} DONE: {n_ok}/{len(subjects)} subjects in {total_time/60:.1f} min")
    if n_ok > 0:
        ok = summary_df[summary_df['status'] == 'ok']
        log.info(f"  Total peaks: {ok['n_peaks'].sum():,}")
        log.info(f"  Mean peaks/subject: {ok['n_peaks'].mean():.0f}")
        log.info(f"  Mean R²: {ok['mean_r_squared'].mean():.3f}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='P20 OT extraction for HBN RestingState (per-release)')
    parser.add_argument('--release', type=str, default='R1',
                        help='Release to process: R1, R2, R3, R4, R6, or "all"')
    parser.add_argument('--f0', type=float, default=F0_DEFAULT)
    parser.add_argument('--freq-ceil', type=float, default=FREQ_CEIL)
    parser.add_argument('--resume-from', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    f0 = args.f0
    freq_ceil = args.freq_ceil

    log.info(f"P20 OT Extraction — HBN RestingState")
    log.info(f"  f0={f0}, freq_ceil={freq_ceil}, target_fs={TARGET_FS}")
    log.info(f"  No notch filter (60 Hz above range)")

    # Show bands
    raw_bands = build_target_bands(f0, freq_ceil)
    bands = merge_narrow_targets(raw_bands)
    log.info(f"\nBands ({len(bands)}):")
    for b in bands:
        log.info(f"  {b['name']:>12}  [{b['target_lo']:5.2f}, {b['target_hi']:5.2f}]  "
                 f"fit [{b['fit_lo']:5.2f}, {b['fit_hi']:5.2f}]")

    releases = RELEASES if args.release.lower() == 'all' else [args.release]

    for release in releases:
        log.info(f"\n{'='*60}")
        log.info(f"  Processing HBN {release}")
        log.info(f"{'='*60}")
        process_release(release, f0, freq_ceil,
                        resume_from=args.resume_from,
                        output_dir=args.output_dir)


if __name__ == '__main__':
    main()
