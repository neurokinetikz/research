#!/usr/bin/env python3
"""
Overlap-Trim Per-Phi-Octave FOOOF Extraction for CHBMP
=======================================================

Adapted from EEGMMIDB overlap-trim extraction for CHBMP dataset.
CHBMP: 282 subjects, 62-120 channels, 200 Hz, EDF, Cuban power grid (60 Hz).

For P20 analysis: FREQ_CEIL=55 Hz to capture the full n+3 gamma octave
(33.17-53.67 Hz). No notch filter needed since 60 Hz is above our range.

Usage:
    python scripts/run_chbmp_phioctave_overlap_trim.py
    python scripts/run_chbmp_phioctave_overlap_trim.py --freq-ceil 75
    python scripts/run_chbmp_phioctave_overlap_trim.py --condition EC
"""

import os
import sys
import time
import argparse
import logging
import warnings
import gc

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# =========================================================================
# CONSTANTS
# =========================================================================
PHI = (1 + np.sqrt(5)) / 2
F0_DEFAULT = 7.83

SFREQ = 200.0
NPERSEG = 800              # 4s at 200 Hz → 0.25 Hz resolution
FREQ_CEIL = 55.0           # Full n+3 octave (53.67 Hz) + margin
FREQ_FLOOR = 1.0
R2_MIN = 0.70
PAD_OCTAVES = 0.5
FILTER_LO = 1.0
MIN_CONDITION_DURATION = 10.0

DATA_DIR = '/Volumes/T9/CHBMP/BIDS_dataset'
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_chbmp')

FOOOF_BASE_PARAMS = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
)


# =========================================================================
# BAND BUILDING (same as EEGMMIDB/LEMON overlap-trim)
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

        target_width = target_hi - target_lo
        if target_width < 0.5:
            continue

        fit_lo = f0 * base ** (n + offset - PAD_OCTAVES)
        fit_hi = f0 * base ** (n + 1 + offset + PAD_OCTAVES)
        fit_lo = max(fit_lo, freq_floor)
        fit_hi = min(fit_hi, freq_ceil)

        bands.append({
            'name': f'n{n:+d}',
            'n': n,
            'target_lo': target_lo,
            'target_hi': target_hi,
            'fit_lo': fit_lo,
            'fit_hi': fit_hi,
        })

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
# DATA LOADING (adapted from run_chbmp_phi_replication.py)
# =========================================================================

def parse_events(data_dir, sub_id):
    """Parse events.tsv to get EC/EO segment boundaries."""
    events_path = os.path.join(data_dir, sub_id, 'ses-V01', 'eeg',
                               f'{sub_id}_ses-V01_task-protmap_events.tsv')
    if not os.path.isfile(events_path):
        return None

    try:
        events = pd.read_csv(events_path, sep='\t')
    except Exception:
        return None

    if 'value' not in events.columns:
        return None

    hv_rows = events[events['value'] == 67]
    hv_onset = hv_rows['onset'].values[0] if len(hv_rows) > 0 else float('inf')

    rest_events = events[events['onset'] < hv_onset].copy()

    ec_onsets = rest_events[rest_events['value'] == 65]['onset'].values
    eo_onsets = rest_events[rest_events['value'] == 66]['onset'].values

    all_onsets = sorted(list(ec_onsets) + list(eo_onsets))

    segments = {'EO': [], 'EC': []}

    for onset in ec_onsets:
        later = [t for t in all_onsets if t > onset]
        offset = later[0] if later else hv_onset
        if offset - onset >= 2.0:
            segments['EC'].append((onset, offset))

    for onset in eo_onsets:
        later = [t for t in all_onsets if t > onset]
        offset = later[0] if later else hv_onset
        if offset - onset >= 2.0:
            segments['EO'].append((onset, offset))

    return segments


def load_subject_raw(sub_id, data_dir, condition='combined', freq_ceil=FREQ_CEIL):
    """Load and preprocess one CHBMP subject."""
    edf_path = os.path.join(data_dir, sub_id, 'ses-V01', 'eeg',
                            f'{sub_id}_ses-V01_task-protmap_eeg.edf')
    if not os.path.isfile(edf_path):
        return None, {'status': 'no_file'}

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        return None, {'status': f'load_error: {e}'}

    # Pick EEG only
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False,
                               ecg=False, misc=False, exclude='bads')
    raw.pick(eeg_picks)

    # Highpass at 1 Hz, lowpass below Nyquist
    filter_hi = min(freq_ceil + 5, raw.info['sfreq'] / 2 - 1)
    raw.filter(FILTER_LO, filter_hi, verbose=False)

    # NO notch filter — 60 Hz is above our FREQ_CEIL of 55 Hz
    # (and we want clean spectrum through the full n+3 octave)

    # Crop to resting-state condition
    segments = parse_events(data_dir, sub_id)
    if condition == 'combined' and segments:
        all_segs = sorted(segments['EC'] + segments['EO'], key=lambda x: x[0])
        if all_segs:
            raws = []
            for onset, offset in all_segs:
                try:
                    seg = raw.copy().crop(tmin=onset,
                                          tmax=min(offset, raw.times[-1]))
                    if len(seg.times) > 0:
                        raws.append(seg)
                except Exception:
                    continue
            if raws:
                raw = mne.concatenate_raws(raws)
        else:
            raw.crop(tmin=0, tmax=min(1000.0, raw.times[-1]))
    elif condition in ('EC', 'EO') and segments and segments.get(condition):
        raws = []
        for onset, offset in segments[condition]:
            try:
                seg = raw.copy().crop(tmin=onset,
                                      tmax=min(offset, raw.times[-1]))
                if len(seg.times) > 0:
                    raws.append(seg)
            except Exception:
                continue
        if raws:
            raw = mne.concatenate_raws(raws)
        else:
            return None, {'status': f'no_{condition}_segments'}
    elif not segments:
        raw.crop(tmin=0, tmax=min(1000.0, raw.times[-1]))

    duration = raw.n_times / raw.info['sfreq']
    if duration < MIN_CONDITION_DURATION:
        return None, {'status': f'too_short_{duration:.0f}s'}

    info = {
        'status': 'ok',
        'n_channels': len(raw.ch_names),
        'duration_sec': duration,
        'sfreq': raw.info['sfreq'],
    }

    return raw, info


# =========================================================================
# CORE EXTRACTION (identical to EEGMMIDB)
# =========================================================================

def extract_overlap_trim_subject(raw_clean, f0, fs=SFREQ,
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
        target_lo = band['target_lo']
        target_hi = band['target_hi']
        fit_lo = band['fit_lo']
        fit_hi = band['fit_hi']

        fit_width = fit_hi - fit_lo
        target_width = target_hi - target_lo

        max_n_peaks = max(3, min(15, int(fit_width / 1.5)))
        max_peak_width = min(fit_width * 0.6, 12.0)
        peak_width_limits = [0.2, max_peak_width]

        fooof_params = {
            **FOOOF_BASE_PARAMS,
            'max_n_peaks': max_n_peaks,
            'peak_width_limits': peak_width_limits,
        }

        band_r2s = []
        band_aperiodic_exps = []
        n_fitted = 0
        n_passed = 0
        n_peaks_kept = 0
        n_peaks_trimmed = 0

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

            offset_val, exponent = _get_aperiodic_params(sm)
            band_aperiodic_exps.append(exponent)

            peak_params = _get_peak_params(sm)

            for row in peak_params:
                peak_freq = row[0]

                if target_lo <= peak_freq < target_hi:
                    all_peaks.append({
                        'channel': ch,
                        'freq': peak_freq,
                        'power': row[1],
                        'bandwidth': row[2],
                        'phi_octave': bname,
                        'phi_octave_n': band['n'],
                        'target_lo': target_lo,
                        'target_hi': target_hi,
                        'fit_lo': fit_lo,
                        'fit_hi': fit_hi,
                    })
                    n_peaks_kept += 1
                else:
                    n_peaks_trimmed += 1

        band_stats.append({
            'band_name': bname,
            'band_n': band['n'],
            'target_lo': round(target_lo, 3),
            'target_hi': round(target_hi, 3),
            'target_width': round(target_width, 3),
            'fit_lo': round(fit_lo, 3),
            'fit_hi': round(fit_hi, 3),
            'fit_width': round(fit_width, 3),
            'max_n_peaks': max_n_peaks,
            'n_channels_fitted': n_fitted,
            'n_channels_passed': n_passed,
            'mean_r_squared': np.mean(band_r2s) if band_r2s else np.nan,
            'mean_aperiodic_exponent': (np.mean(band_aperiodic_exps)
                                        if band_aperiodic_exps else np.nan),
            'peaks_kept': n_peaks_kept,
            'peaks_trimmed': n_peaks_trimmed,
        })

    peaks_df = pd.DataFrame(all_peaks) if all_peaks else pd.DataFrame(
        columns=['channel', 'freq', 'power', 'bandwidth',
                 'phi_octave', 'phi_octave_n',
                 'target_lo', 'target_hi', 'fit_lo', 'fit_hi'])
    band_info_df = pd.DataFrame(band_stats)

    return peaks_df, band_info_df


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Overlap-trim per-phi-octave FOOOF extraction for CHBMP')
    parser.add_argument('--f0', type=float, default=F0_DEFAULT)
    parser.add_argument('--base', type=float, default=None)
    parser.add_argument('--offset', type=float, default=0.0)
    parser.add_argument('--condition', type=str, default='EC',
                        choices=['EC', 'EO', 'combined'])
    parser.add_argument('--resume-from', type=int, default=None)
    parser.add_argument('--freq-ceil', type=float, default=FREQ_CEIL)
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    f0 = args.f0
    offset = args.offset
    freq_ceil = args.freq_ceil
    data_dir = os.path.abspath(args.data_dir)
    condition = args.condition
    base = args.base
    base_val = base if base is not None else PHI
    base_label = f'base{base:.2f}' if base is not None else 'phi'

    # Output directory
    offset_tag = f'_offset{offset:.1f}' if offset != 0.0 else ''
    base_tag = f'_{base_label}' if base is not None else ''
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(
            OUTPUT_BASE,
            f'p20_overlap_trim_{condition}_f0{f0:.2f}{base_tag}{offset_tag}')
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"Overlap-Trim Per-Octave FOOOF Extraction — CHBMP (base={base_val:.4f})")
    log.info(f"  f₀ = {f0} Hz, base = {base_val:.4f}, offset = {offset}")
    log.info(f"  fs = {SFREQ} Hz, nperseg = {NPERSEG} "
             f"(freq_res = {SFREQ/NPERSEG:.3f} Hz)")
    log.info(f"  freq_ceil = {freq_ceil} Hz")
    log.info(f"  condition: {condition}")
    log.info(f"  R² threshold: {R2_MIN}")
    log.info(f"  padding: {PAD_OCTAVES} octaves per side")
    log.info(f"  data: {data_dir}")
    log.info(f"  output: {out_dir}")

    # Show band structure
    freq_res = SFREQ / NPERSEG
    raw_bands = build_target_bands(f0, freq_ceil, FREQ_FLOOR, offset, base=base)
    bands = merge_narrow_targets(raw_bands, min_width_hz=1.5,
                                 min_bins=12, freq_res=freq_res)

    log.info(f"\nBands ({len(bands)}):")
    log.info(f"  {'name':>12}  {'target':>18}  {'fit window':>18}  {'trim zone':>12}")
    for b in bands:
        tgt = f"[{b['target_lo']:5.2f}, {b['target_hi']:5.2f}]"
        fit = f"[{b['fit_lo']:5.2f}, {b['fit_hi']:5.2f}]"
        pad_lo = b['target_lo'] - b['fit_lo']
        pad_hi = b['fit_hi'] - b['target_hi']
        trim = f"-{pad_lo:.1f}/+{pad_hi:.1f} Hz"
        log.info(f"  {b['name']:>12}  {tgt:>18}  {fit:>18}  {trim:>12}")

    # Discover subjects
    import glob as globmod
    pattern = os.path.join(data_dir, 'sub-CBM*', 'ses-V01', 'eeg',
                           'sub-CBM*_ses-V01_task-protmap_eeg.edf')
    edf_files = sorted(globmod.glob(pattern))
    subjects = []
    seen = set()
    for f in edf_files:
        for p in f.split(os.sep):
            if p.startswith('sub-CBM') and p not in seen:
                subjects.append(p)
                seen.add(p)
                break

    if args.resume_from:
        subjects = subjects[args.resume_from:]
        log.info(f"Resuming from index {args.resume_from} "
                 f"({len(subjects)} subjects remaining)")
    else:
        log.info(f"\n{len(subjects)} subjects to process")

    summary_rows = []
    t_start = time.time()

    for i, sub_id in enumerate(subjects):
        out_path = os.path.join(out_dir, f'{sub_id}_peaks.csv')
        band_path = os.path.join(out_dir, f'{sub_id}_band_info.csv')

        if os.path.exists(out_path):
            log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: already done, skipping")
            continue

        t0 = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw, info = load_subject_raw(sub_id, data_dir,
                                         condition=condition,
                                         freq_ceil=freq_ceil)

        if raw is None:
            log.warning(f"  [{i+1}/{len(subjects)}] {sub_id} — {info.get('status')}")
            summary_rows.append({
                'subject_id': sub_id, 'status': info.get('status', 'no_data'),
                'n_peaks': 0,
            })
            continue

        load_time = time.time() - t0
        fs_actual = raw.info['sfreq']

        try:
            peaks_df, band_info_df = extract_overlap_trim_subject(
                raw, f0=f0, fs=fs_actual, nperseg=NPERSEG,
                overlap=0.5, freq_ceil=freq_ceil, offset=offset,
                r2_min=R2_MIN, base=base)
        except Exception as e:
            log.error(f"  {sub_id}: extraction failed: {e}")
            summary_rows.append({
                'subject_id': sub_id, 'status': f'error: {e}', 'n_peaks': 0,
            })
            del raw
            gc.collect()
            continue

        peaks_df.to_csv(out_path, index=False)
        band_info_df.to_csv(band_path, index=False)

        elapsed = time.time() - t0
        n_peaks = len(peaks_df)
        n_bands_ok = (band_info_df['n_channels_passed'] > 0).sum()
        mean_r2 = band_info_df['mean_r_squared'].mean()
        total_kept = band_info_df['peaks_kept'].sum()
        total_trimmed = band_info_df['peaks_trimmed'].sum()
        trim_pct = (total_trimmed / (total_kept + total_trimmed) * 100
                    if (total_kept + total_trimmed) > 0 else 0)

        summary_rows.append({
            'subject_id': sub_id,
            'status': 'ok',
            'n_peaks': n_peaks,
            'n_channels': info.get('n_channels', 0),
            'duration_sec': round(info.get('duration_sec', 0), 1),
            'n_bands_total': len(band_info_df),
            'n_bands_with_peaks': int(n_bands_ok),
            'mean_r_squared': round(mean_r2, 4) if not np.isnan(mean_r2) else np.nan,
            'peaks_trimmed_pct': round(trim_pct, 1),
            'load_time_sec': round(load_time, 1),
            'extraction_time_sec': round(elapsed, 1),
        })

        log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: {n_peaks} peaks "
                 f"({n_bands_ok}/{len(band_info_df)} bands) "
                 f"R²={mean_r2:.3f} trim={trim_pct:.0f}% "
                 f"ch={info['n_channels']} dur={info['duration_sec']:.0f}s "
                 f"{elapsed:.1f}s")

        del raw
        gc.collect()

        if (i + 1) % 10 == 0:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, 'extraction_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    total_time = time.time() - t_start
    n_ok = (summary_df['status'] == 'ok').sum() if len(summary_df) > 0 else 0

    log.info(f"\n{'='*60}")
    log.info(f"DONE: {n_ok}/{len(subjects)} subjects in {total_time/60:.1f} min")
    if n_ok > 0:
        ok_df = summary_df[summary_df['status'] == 'ok']
        log.info(f"  Mean peaks/subject: {ok_df['n_peaks'].mean():.0f}")
        log.info(f"  Mean R²: {ok_df['mean_r_squared'].mean():.3f}")
        log.info(f"  Mean trim rate: {ok_df['peaks_trimmed_pct'].mean():.0f}%")
        log.info(f"  Total peaks: {ok_df['n_peaks'].sum():,}")
    log.info(f"  Summary: {summary_path}")


if __name__ == '__main__':
    main()
