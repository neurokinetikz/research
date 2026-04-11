#!/usr/bin/env python3
"""
Theta + Alpha Only Extraction for Dortmund, CHBMP, HBN
========================================================

Extracts FOOOF peaks for theta (n-1) and alpha (n+0) bands only,
using min_bins=10 to prevent merging. Supplements the existing
P20 extractions which merged these bands.

Usage:
    python scripts/run_theta_alpha_extraction.py --dataset dortmund
    python scripts/run_theta_alpha_extraction.py --dataset chbmp
    python scripts/run_theta_alpha_extraction.py --dataset hbn --release R1
    python scripts/run_theta_alpha_extraction.py --dataset hbn --release all
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

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83
R2_MIN = 0.70
PAD_OCTAVES = 0.5
FILTER_LO = 1.0

FOOOF_BASE_PARAMS = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
)

# Only extract theta and alpha
TARGET_BANDS = [
    {'name': 'n-1', 'n': -1,
     'target_lo': F0 * PHI**-1, 'target_hi': F0 * PHI**0,
     'fit_lo': F0 * PHI**(-1 - PAD_OCTAVES), 'fit_hi': F0 * PHI**(0 + PAD_OCTAVES)},
    {'name': 'n+0', 'n': 0,
     'target_lo': F0 * PHI**0, 'target_hi': F0 * PHI**1,
     'fit_lo': F0 * PHI**(0 - PAD_OCTAVES), 'fit_hi': F0 * PHI**(1 + PAD_OCTAVES)},
]


def extract_theta_alpha(raw_clean, fs, nperseg):
    """Extract peaks for theta and alpha bands only."""
    noverlap = nperseg // 2
    freq_res = fs / nperseg
    ch_names = raw_clean.ch_names
    all_peaks = []
    band_stats = []

    for band in TARGET_BANDS:
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

        band_r2s = []
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
            if np.isnan(r2) or r2 < R2_MIN:
                continue

            n_passed += 1
            band_r2s.append(r2)

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
            'n_channels_passed': n_passed,
            'mean_r_squared': np.mean(band_r2s) if band_r2s else np.nan,
            'peaks_kept': n_kept})

    peaks_df = pd.DataFrame(all_peaks) if all_peaks else pd.DataFrame(
        columns=['channel', 'freq', 'power', 'bandwidth',
                 'phi_octave', 'phi_octave_n',
                 'target_lo', 'target_hi', 'fit_lo', 'fit_hi'])
    return peaks_df, pd.DataFrame(band_stats)


# =========================================================================
# DATASET LOADERS
# =========================================================================

def load_dortmund(sub_id, data_dir='/Volumes/T9/dortmund_data_dl'):
    edf_path = os.path.join(data_dir, sub_id, 'ses-1', 'eeg',
                            f'{sub_id}_ses-1_task-EyesClosed_acq-pre_eeg.edf')
    if not os.path.isfile(edf_path):
        return None, 250, 1000
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')
    if raw.info['sfreq'] > 250:
        raw.resample(250, verbose=False)
    raw.filter(FILTER_LO, 30, verbose=False)  # only need up to alpha
    return raw, 250, 1000


def load_chbmp(sub_id, data_dir='/Volumes/T9/CHBMP/BIDS_dataset'):
    edf_path = os.path.join(data_dir, sub_id, 'ses-V01', 'eeg',
                            f'{sub_id}_ses-V01_task-protmap_eeg.edf')
    if not os.path.isfile(edf_path):
        return None, 200, 800

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False,
                               misc=False, exclude='bads')
    raw.pick(eeg_picks)
    raw.filter(FILTER_LO, 30, verbose=False)

    # Crop to EC segments (before hyperventilation)
    try:
        events_path = os.path.join(data_dir, sub_id, 'ses-V01', 'eeg',
                                   f'{sub_id}_ses-V01_task-protmap_events.tsv')
        if os.path.isfile(events_path):
            events = pd.read_csv(events_path, sep='\t')
            if 'value' in events.columns:
                hv = events[events['value'] == 67]
                hv_onset = hv['onset'].values[0] if len(hv) > 0 else float('inf')
                ec_onsets = events[(events['value'] == 65) & (events['onset'] < hv_onset)]['onset'].values
                all_onsets = sorted(events[events['onset'] < hv_onset]['onset'].values)
                raws = []
                for onset in ec_onsets:
                    later = [t for t in all_onsets if t > onset]
                    offset = later[0] if later else hv_onset
                    if offset - onset >= 2.0:
                        try:
                            seg = raw.copy().crop(tmin=onset, tmax=min(offset, raw.times[-1]))
                            if len(seg.times) > 0:
                                raws.append(seg)
                        except Exception:
                            continue
                if raws:
                    raw = mne.concatenate_raws(raws)
    except Exception:
        raw.crop(tmin=0, tmax=min(1000.0, raw.times[-1]))

    fs = raw.info['sfreq']
    nperseg = int(4.0 * fs)
    return raw, fs, nperseg


def load_hbn(set_path):
    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception:
        return None, 250, 1000
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(eeg_picks) == 0:
        return None, 250, 1000
    raw.pick(eeg_picks)
    if raw.info['sfreq'] > 250:
        raw.resample(250, verbose=False)
    raw.filter(FILTER_LO, 30, verbose=False)
    return raw, 250, 1000


# =========================================================================
# MAIN
# =========================================================================

def process_subjects(subjects, loader_fn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    t_start = time.time()

    for i, (sub_id, load_arg) in enumerate(subjects):
        out_path = os.path.join(out_dir, f'{sub_id}_theta_alpha.csv')
        if os.path.exists(out_path):
            log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: skipping")
            continue

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw, fs, nperseg = loader_fn(sub_id, load_arg) if load_arg else loader_fn(sub_id)

        if raw is None:
            summary_rows.append({'subject_id': sub_id, 'status': 'no_data', 'n_peaks': 0})
            continue

        try:
            peaks_df, band_info = extract_theta_alpha(raw, fs, nperseg)
        except Exception as e:
            log.error(f"  {sub_id}: {e}")
            summary_rows.append({'subject_id': sub_id, 'status': f'error', 'n_peaks': 0})
            del raw; gc.collect()
            continue

        peaks_df.to_csv(out_path, index=False)
        elapsed = time.time() - t0
        n_theta = len(peaks_df[peaks_df.phi_octave == 'n-1'])
        n_alpha = len(peaks_df[peaks_df.phi_octave == 'n+0'])

        summary_rows.append({
            'subject_id': sub_id, 'status': 'ok',
            'n_peaks': len(peaks_df), 'n_theta': n_theta, 'n_alpha': n_alpha,
            'time_sec': round(elapsed, 1)})

        log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: θ={n_theta} α={n_alpha} {elapsed:.1f}s")
        del raw; gc.collect()

        if (i + 1) % 20 == 0:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, 'extraction_summary.csv'), index=False)
    total = time.time() - t_start
    n_ok = (summary_df['status'] == 'ok').sum() if len(summary_df) > 0 else 0
    log.info(f"  Done: {n_ok}/{len(subjects)} in {total/60:.1f} min")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Theta+Alpha extraction')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['dortmund', 'chbmp', 'hbn'])
    parser.add_argument('--release', type=str, default='R1',
                        help='HBN release (R1-R6 or all)')
    args = parser.parse_args()

    log.info(f"Theta+Alpha extraction — {args.dataset}")
    log.info(f"  Theta (n-1): [{TARGET_BANDS[0]['target_lo']:.2f}, {TARGET_BANDS[0]['target_hi']:.2f}] Hz")
    log.info(f"  Alpha (n+0): [{TARGET_BANDS[1]['target_lo']:.2f}, {TARGET_BANDS[1]['target_hi']:.2f}] Hz")

    if args.dataset == 'dortmund':
        data_dir = '/Volumes/T9/dortmund_data_dl'
        subs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and
                       os.path.isdir(os.path.join(data_dir, d, 'ses-1', 'eeg'))])
        subjects = [(s, None) for s in subs]
        out_dir = 'exports_dortmund/theta_alpha_EC_pre'

        def loader(sub_id, _=None):
            return load_dortmund(sub_id)

        process_subjects(subjects, loader, out_dir)

    elif args.dataset == 'chbmp':
        data_dir = '/Volumes/T9/CHBMP/BIDS_dataset'
        pattern = os.path.join(data_dir, 'sub-CBM*', 'ses-V01', 'eeg', '*_eeg.edf')
        edf_files = sorted(glob(pattern))
        seen = set()
        subjects = []
        for f in edf_files:
            for p in f.split(os.sep):
                if p.startswith('sub-CBM') and p not in seen:
                    subjects.append((p, None))
                    seen.add(p)
                    break
        out_dir = 'exports_chbmp/theta_alpha_EC'

        def loader(sub_id, _=None):
            return load_chbmp(sub_id)

        process_subjects(subjects, loader, out_dir)

    elif args.dataset == 'hbn':
        releases = ['R1', 'R2', 'R3', 'R4', 'R6'] if args.release.lower() == 'all' else [args.release]

        for release in releases:
            release_dir = os.path.join('/Volumes/T9/hbn_data', f'cmi_bids_{release}')
            pattern = os.path.join(release_dir, 'sub-*', 'eeg', '*RestingState_eeg.set')
            files = sorted(glob(pattern))
            seen = set()
            subjects = []
            for f in files:
                sub_id = os.path.basename(f).split('_task-')[0]
                if sub_id not in seen:
                    subjects.append((sub_id, f))
                    seen.add(sub_id)

            out_dir = f'exports_hbn/theta_alpha_{release}'
            log.info(f"\nHBN {release}: {len(subjects)} subjects")

            def loader(sub_id, set_path):
                return load_hbn(set_path)

            process_subjects(subjects, loader, out_dir)


if __name__ == '__main__':
    main()
