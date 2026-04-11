#!/usr/bin/env python3
"""
Adaptive-Resolution Overlap-Trim Extraction
=============================================

Uses band-specific nperseg so that all 14 degree-7 lattice positions
are resolvable in every band. Lower bands get longer Welch windows
to achieve finer frequency resolution proportional to band width.

Target: freq_res < min_position_separation / 2 in each band,
ensuring at least 2 spectral bins per enrichment bin.

Band       nperseg   Window   Freq res
Theta      7,853     31.4s    0.032 Hz
Alpha      4,854     19.4s    0.052 Hz
Beta-low   2,999     12.0s    0.083 Hz
Beta-high  1,854      7.4s    0.135 Hz
Gamma      1,145      4.6s    0.218 Hz

Usage:
    python scripts/run_adaptive_resolution_extraction.py --dataset eegmmidb
    python scripts/run_adaptive_resolution_extraction.py --dataset lemon
    python scripts/run_adaptive_resolution_extraction.py --dataset dortmund
    python scripts/run_adaptive_resolution_extraction.py --dataset chbmp
    python scripts/run_adaptive_resolution_extraction.py --dataset hbn --release all
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
# CONSTANTS
# =========================================================================
PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83
TARGET_FS = 250.0
R2_MIN = 0.70
PAD_OCTAVES = 0.5
FILTER_LO = 1.0
FREQ_CEIL = 55.0

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_adaptive')

FOOOF_BASE_PARAMS = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
)

# 14 degree-7 positions
INV = 1.0 / PHI
POS_14 = sorted(set([0.0, 0.5] +
                     [round(INV**k, 6) for k in range(1, 8)] +
                     [round(1 - INV**k, 6) for k in range(1, 8)]))
POS_14 = [p for p in POS_14 if 0 <= p < 1]
MIN_SEP = min(POS_14[i+1] - POS_14[i] for i in range(len(POS_14)-1))


def compute_adaptive_nperseg(f0, n, fs):
    """Compute nperseg for band n that resolves all 14 degree-7 positions.

    Target: freq_res <= min_position_separation_hz / 2
    (at least 2 spectral bins per enrichment half-width)
    """
    lo = f0 * PHI ** n
    hi = f0 * PHI ** (n + 1)
    width = hi - lo
    min_sep_hz = MIN_SEP * width
    needed_res = min_sep_hz / 2
    nperseg = int(np.ceil(fs / needed_res))
    # Round up to even number
    if nperseg % 2 != 0:
        nperseg += 1
    return nperseg


# =========================================================================
# BAND BUILDING WITH ADAPTIVE RESOLUTION
# =========================================================================

def build_adaptive_bands(f0, fs, freq_ceil=FREQ_CEIL, freq_floor=1.0):
    """Build phi-octave bands with per-band nperseg for equal resolution."""
    bands = []
    for n in range(-4, 10):
        target_lo = f0 * PHI ** n
        target_hi = f0 * PHI ** (n + 1)

        if target_lo >= freq_ceil:
            break
        if target_hi <= freq_floor:
            continue

        target_lo = max(target_lo, freq_floor)
        target_hi = min(target_hi, freq_ceil)
        target_width = target_hi - target_lo

        if target_width < 0.5:
            continue

        fit_lo = f0 * PHI ** (n - PAD_OCTAVES)
        fit_hi = f0 * PHI ** (n + 1 + PAD_OCTAVES)
        fit_lo = max(fit_lo, freq_floor)
        fit_hi = min(fit_hi, freq_ceil)

        # Adaptive nperseg
        nperseg = compute_adaptive_nperseg(f0, n, fs)
        # Cap at reasonable maximum (60s window)
        max_nperseg = int(60 * fs)
        if nperseg > max_nperseg:
            nperseg = max_nperseg

        freq_res = fs / nperseg

        bands.append({
            'name': f'n{n:+d}',
            'n': n,
            'target_lo': target_lo,
            'target_hi': target_hi,
            'fit_lo': fit_lo,
            'fit_hi': fit_hi,
            'nperseg': nperseg,
            'freq_res': freq_res,
        })

    return bands


def merge_narrow_bands(bands, min_width_hz=1.5):
    """Merge bands that are too narrow, keeping the finest nperseg."""
    merged = []
    i = 0
    while i < len(bands):
        b = bands[i].copy()
        width = b['target_hi'] - b['target_lo']

        if width < min_width_hz:
            while (i + 1 < len(bands) and
                   (b['target_hi'] - b['target_lo'] < min_width_hz)):
                i += 1
                b['target_hi'] = bands[i]['target_hi']
                b['fit_hi'] = bands[i]['fit_hi']
                # Use the finest resolution (largest nperseg) of merged bands
                b['nperseg'] = max(b['nperseg'], bands[i]['nperseg'])
                b['freq_res'] = min(b['freq_res'], bands[i]['freq_res'])
            b['name'] = f'{b["name"]}_merged'

        if b['fit_lo'] > b['target_lo']:
            b['fit_lo'] = b['target_lo'] * 0.7
        if b['fit_hi'] < b['target_hi']:
            b['fit_hi'] = b['target_hi'] * 1.3

        merged.append(b)
        i += 1
    return merged


# =========================================================================
# CORE EXTRACTION
# =========================================================================

def extract_adaptive_subject(raw_clean, f0, fs, freq_ceil=FREQ_CEIL,
                             r2_min=R2_MIN):
    """Overlap-trim extraction with adaptive nperseg per band."""
    raw_bands = build_adaptive_bands(f0, fs, freq_ceil)
    bands = merge_narrow_bands(raw_bands)
    ch_names = raw_clean.ch_names
    all_peaks = []
    band_stats = []

    for band in bands:
        bname = band['name']
        target_lo, target_hi = band['target_lo'], band['target_hi']
        fit_lo, fit_hi = band['fit_lo'], band['fit_hi']
        fit_width = fit_hi - fit_lo
        nperseg = band['nperseg']
        freq_res = band['freq_res']
        noverlap = nperseg // 2

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
            fit_mask = (freqs >= fit_lo) & (freqs <= fit_hi)
            if fit_mask.sum() < 10:
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

            for row in _get_peak_params(sm):
                if target_lo <= row[0] < target_hi:
                    all_peaks.append({
                        'channel': ch, 'freq': row[0], 'power': row[1],
                        'bandwidth': row[2], 'phi_octave': bname,
                        'phi_octave_n': band['n'],
                        'target_lo': target_lo, 'target_hi': target_hi,
                        'fit_lo': fit_lo, 'fit_hi': fit_hi,
                        'nperseg': nperseg, 'freq_res': freq_res})
                    n_kept += 1
                else:
                    n_trimmed += 1

        band_stats.append({
            'band_name': bname, 'band_n': band['n'],
            'target_lo': round(target_lo, 3), 'target_hi': round(target_hi, 3),
            'fit_lo': round(fit_lo, 3), 'fit_hi': round(fit_hi, 3),
            'nperseg': nperseg, 'freq_res': round(freq_res, 4),
            'window_sec': round(nperseg / fs, 1),
            'n_channels_fitted': n_fitted, 'n_channels_passed': n_passed,
            'mean_r_squared': np.mean(band_r2s) if band_r2s else np.nan,
            'peaks_kept': n_kept, 'peaks_trimmed': n_trimmed})

    peaks_df = pd.DataFrame(all_peaks) if all_peaks else pd.DataFrame(
        columns=['channel', 'freq', 'power', 'bandwidth',
                 'phi_octave', 'phi_octave_n',
                 'target_lo', 'target_hi', 'fit_lo', 'fit_hi',
                 'nperseg', 'freq_res'])
    return peaks_df, pd.DataFrame(band_stats)


# =========================================================================
# DATASET LOADERS
# =========================================================================

def load_eegmmidb(sub_id, data_dir='/Volumes/T9/eegmmidb'):
    """Load all runs for one EEGMMIDB subject."""
    raws = []
    for run in range(1, 15):
        edf_path = os.path.join(data_dir, f'S{sub_id:03d}', f'S{sub_id:03d}R{run:02d}.edf')
        if os.path.exists(edf_path):
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                raws.append(raw)
            except Exception:
                continue
    if not raws:
        return None
    for i, raw in enumerate(raws):
        if abs(raw.info['sfreq'] - 160) > 0.1:
            raws[i] = raw.resample(160, verbose=False)
    raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
    raw.filter(FILTER_LO, 59, verbose=False)
    return raw


def load_lemon(sub_id, data_dir='/Volumes/T9/lemon_data/eeg_preprocessed/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed',
               condition='EC'):
    """Load LEMON preprocessed EC or EO file."""
    path = os.path.join(data_dir, f'{sub_id}_{condition}.set')
    if not os.path.isfile(path):
        return None
    try:
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    except Exception:
        return None
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(eeg_picks) == 0:
        return None
    raw.pick(eeg_picks)
    nyq = raw.info['sfreq'] / 2.0
    h_freq = min(59, nyq - 1)
    raw.filter(FILTER_LO, h_freq, verbose=False)
    return raw


def load_dortmund(sub_id, data_dir='/Volumes/T9/dortmund_data_dl',
                  task='EyesClosed', acq='pre', ses='1'):
    """Load Dortmund EDF by task, acquisition, and session."""
    edf_path = os.path.join(data_dir, sub_id, f'ses-{ses}', 'eeg',
                            f'{sub_id}_ses-{ses}_task-{task}_acq-{acq}_eeg.edf')
    if not os.path.isfile(edf_path):
        return None
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None
    raw.pick_types(eeg=True, exclude='bads')
    if raw.info['sfreq'] > TARGET_FS:
        raw.resample(TARGET_FS, verbose=False)
    raw.filter(FILTER_LO, 59, verbose=False)
    return raw


def load_chbmp(sub_id, data_dir='/Volumes/T9/CHBMP/BIDS_dataset'):
    """Load CHBMP EC segments."""
    edf_path = os.path.join(data_dir, sub_id, 'ses-V01', 'eeg',
                            f'{sub_id}_ses-V01_task-protmap_eeg.edf')
    if not os.path.isfile(edf_path):
        return None
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False,
                               misc=False, exclude='bads')
    raw.pick(eeg_picks)
    raw.filter(FILTER_LO, 59, verbose=False)
    # Crop to EC (before hyperventilation)
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
    return raw


def load_hbn(set_path):
    """Load HBN RestingState .set file."""
    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception:
        return None
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(eeg_picks) == 0:
        return None
    raw.pick(eeg_picks)
    if raw.info['sfreq'] > TARGET_FS:
        raw.resample(TARGET_FS, verbose=False)
    raw.filter(FILTER_LO, 59, verbose=False)
    return raw


# =========================================================================
# GENERIC PROCESSOR
# =========================================================================

def process_subjects(subjects, loader_fn, out_dir, dataset_label):
    """Process all subjects for a dataset."""
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    t_start = time.time()

    # Show band structure for first subject's fs
    log.info(f"\n{dataset_label}: {len(subjects)} subjects")
    log.info(f"  Output: {out_dir}")

    # Preview bands at 250 Hz
    bands = build_adaptive_bands(F0, TARGET_FS)
    bands = merge_narrow_bands(bands)
    log.info(f"\n  Adaptive bands ({len(bands)}):")
    log.info(f"  {'name':>12s}  {'target':>18s}  {'nperseg':>8s}  {'window':>8s}  {'freq_res':>9s}")
    for b in bands:
        tgt = f"[{b['target_lo']:5.2f}, {b['target_hi']:5.2f}]"
        log.info(f"  {b['name']:>12s}  {tgt:>18s}  {b['nperseg']:>8d}  "
                 f"{b['nperseg']/TARGET_FS:>7.1f}s  {b['freq_res']:>8.4f}Hz")

    for i, (sub_id, load_arg) in enumerate(subjects):
        out_path = os.path.join(out_dir, f'{sub_id}_peaks.csv')
        band_path = os.path.join(out_dir, f'{sub_id}_band_info.csv')

        if os.path.exists(out_path):
            log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: skipping")
            continue

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = loader_fn(sub_id, load_arg)

        if raw is None:
            summary_rows.append({'subject_id': sub_id, 'status': 'no_data', 'n_peaks': 0})
            continue

        fs = raw.info['sfreq']
        duration = raw.n_times / fs

        try:
            peaks_df, band_info = extract_adaptive_subject(raw, F0, fs)
        except Exception as e:
            log.error(f"  {sub_id}: {e}")
            summary_rows.append({'subject_id': sub_id, 'status': f'error', 'n_peaks': 0})
            del raw; gc.collect()
            continue

        peaks_df.to_csv(out_path, index=False)
        band_info.to_csv(band_path, index=False)

        elapsed = time.time() - t0
        n_peaks = len(peaks_df)
        mean_r2 = band_info['mean_r_squared'].mean()
        n_bands = (band_info['n_channels_passed'] > 0).sum()

        summary_rows.append({
            'subject_id': sub_id, 'status': 'ok', 'n_peaks': n_peaks,
            'n_channels': len(raw.ch_names), 'duration_sec': round(duration, 1),
            'n_bands': int(n_bands),
            'mean_r_squared': round(mean_r2, 4) if not np.isnan(mean_r2) else np.nan,
            'time_sec': round(elapsed, 1)})

        log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: {n_peaks} peaks "
                 f"({n_bands} bands) R²={mean_r2:.3f} {elapsed:.1f}s")

        del raw; gc.collect()

        if (i + 1) % 20 == 0:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    total = time.time() - t_start
    n_ok = (summary_df['status'] == 'ok').sum() if len(summary_df) > 0 else 0
    log.info(f"\n  {dataset_label} DONE: {n_ok}/{len(subjects)} in {total/60:.1f} min")
    if n_ok > 0:
        ok = summary_df[summary_df['status'] == 'ok']
        log.info(f"  Total peaks: {ok['n_peaks'].sum():,}")
        log.info(f"  Mean peaks/subject: {ok['n_peaks'].mean():.0f}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive-resolution overlap-trim extraction')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eegmmidb', 'lemon', 'dortmund', 'chbmp', 'hbn'])
    parser.add_argument('--release', type=str, default='R1',
                        help='HBN release (R1-R6 or all)')
    parser.add_argument('--condition', type=str, default=None,
                        help='Condition override (e.g., EO for LEMON, EC-pre/EO-post for Dortmund)')
    parser.add_argument('--session', type=str, default='1',
                        help='Session number for Dortmund (1 or 2)')
    args = parser.parse_args()

    if args.dataset == 'eegmmidb':
        data_dir = '/Volumes/T9/eegmmidb'
        subs = sorted(set(
            d for d in os.listdir(data_dir)
            if d.startswith('S') and os.path.isdir(os.path.join(data_dir, d))))
        subjects = [(f'S{int(s[1:]):03d}', None) for s in subs]
        out_dir = os.path.join(OUTPUT_BASE, 'eegmmidb')

        def loader(sub_id, _=None):
            return load_eegmmidb(int(sub_id[1:]))

        process_subjects(subjects, loader, out_dir, 'EEGMMIDB')

    elif args.dataset == 'lemon':
        data_dir = '/Volumes/T9/lemon_data/eeg_preprocessed/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed'
        cond = args.condition or 'EC'
        files = sorted(glob(os.path.join(data_dir, f'sub-*_{cond}.set')))
        subjects = [(os.path.basename(f).replace(f'_{cond}.set', ''), None) for f in files]
        suffix = f'_EO' if cond == 'EO' else ''
        out_dir = os.path.join(OUTPUT_BASE, f'lemon{suffix}')
        process_subjects(subjects, lambda sid, _=None: load_lemon(sid, condition=cond),
                         out_dir, f'LEMON {cond}')

    elif args.dataset == 'dortmund':
        data_dir = '/Volumes/T9/dortmund_data_dl'
        ses = args.session
        ses_dir = f'ses-{ses}'
        subs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and
                       os.path.isdir(os.path.join(data_dir, d, ses_dir, 'eeg'))])
        subjects = [(s, None) for s in subs]
        # Parse condition: EO-pre, EC-pre (default), EO-post, EC-post
        cond = args.condition or 'EC-pre'
        task = 'EyesOpen' if cond.startswith('EO') else 'EyesClosed'
        acq = 'post' if cond.endswith('post') else 'pre'
        suffix = '' if (cond == 'EC-pre' and ses == '1') else f'_{cond.replace("-", "_")}'
        ses_suffix = f'_ses2' if ses == '2' else ''
        out_dir = os.path.join(OUTPUT_BASE, f'dortmund{suffix}{ses_suffix}')
        process_subjects(subjects,
                         lambda sid, _=None: load_dortmund(sid, task=task, acq=acq, ses=ses),
                         out_dir, f'Dortmund {cond}')

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
        out_dir = os.path.join(OUTPUT_BASE, 'chbmp')
        process_subjects(subjects, lambda sid, _=None: load_chbmp(sid), out_dir, 'CHBMP')

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
            out_dir = os.path.join(OUTPUT_BASE, f'hbn_{release}')
            process_subjects(subjects, lambda sid, path: load_hbn(path),
                             out_dir, f'HBN {release}')


if __name__ == '__main__':
    main()
