#!/usr/bin/env python3
"""
Test merged theta+alpha FOOOF extraction vs per-band extraction.

Fits one FOOOF model across the combined theta+alpha range (4.70-12.30 Hz)
with a single aperiodic fit, then assigns detected peaks to theta or alpha
by frequency. Compares peak density around f0=7.60 Hz against the standard
per-band extraction where theta and alpha get separate FOOOF fits.

Runs on first 20 Dortmund subjects for speed.

Usage:
    python scripts/test_merged_theta_alpha.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from lemon_utils import _get_peak_params, _get_r_squared

try:
    from specparam import SpectralModel
except ImportError:
    from fooof import FOOOF as SpectralModel

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.60
TARGET_FS = 250.0
R2_MIN = 0.70
FILTER_LO = 1.0

FOOOF_BASE_PARAMS = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
)

# Degree-7 min separation for nperseg calculation
INV = 1.0 / PHI
POS_14 = sorted(set([0.0, 0.5] +
                     [round(INV**k, 6) for k in range(1, 8)] +
                     [round(1 - INV**k, 6) for k in range(1, 8)]))
POS_14 = [p for p in POS_14 if 0 <= p < 1]
MIN_SEP = min(POS_14[i+1] - POS_14[i] for i in range(len(POS_14)-1))


def compute_adaptive_nperseg(f0, n, fs):
    lo = f0 * PHI ** n
    hi = f0 * PHI ** (n + 1)
    width = hi - lo
    min_sep_hz = MIN_SEP * width
    needed_res = min_sep_hz / 2
    nperseg = int(np.ceil(fs / needed_res))
    if nperseg % 2 != 0:
        nperseg += 1
    return nperseg


def load_dortmund(sub_id, data_dir='/Volumes/T9/dortmund_data_dl'):
    edf_path = os.path.join(data_dir, sub_id, 'ses-1', 'eeg',
                            f'{sub_id}_ses-1_task-EyesClosed_acq-pre_eeg.edf')
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


def extract_perband(raw, fs):
    """Standard per-band extraction: separate FOOOF for theta and alpha."""
    bands = [
        {'name': 'n-1', 'n': -1,
         'target_lo': F0 * PHI**-1, 'target_hi': F0,
         'fit_lo': F0 * PHI**-1.5, 'fit_hi': F0 * PHI**0.5,
         'nperseg': compute_adaptive_nperseg(F0, -1, fs)},
        {'name': 'n+0', 'n': 0,
         'target_lo': F0, 'target_hi': F0 * PHI,
         'fit_lo': F0 * PHI**-0.5, 'fit_hi': F0 * PHI**1.5,
         'nperseg': compute_adaptive_nperseg(F0, 0, fs)},
    ]

    all_peaks = []
    for band in bands:
        nperseg = min(band['nperseg'], int(60 * fs))
        freq_res = fs / nperseg
        fit_lo, fit_hi = band['fit_lo'], band['fit_hi']
        fit_width = fit_hi - fit_lo
        target_lo, target_hi = band['target_lo'], band['target_hi']

        max_n_peaks = max(3, min(15, int(fit_width / 1.5)))
        peak_width_limits = [max(0.5, 2 * freq_res), min(fit_width * 0.6, 12.0)]
        fooof_params = {**FOOOF_BASE_PARAMS,
                        'max_n_peaks': max_n_peaks,
                        'peak_width_limits': peak_width_limits}

        for ch in raw.ch_names:
            try:
                data = raw.get_data(picks=[ch])[0]
            except Exception:
                continue
            if len(data) < nperseg:
                continue
            freqs, psd = welch(data, fs, nperseg=nperseg, noverlap=nperseg//2)
            sm = SpectralModel(**fooof_params)
            try:
                sm.fit(freqs, psd, [fit_lo, fit_hi])
            except Exception:
                continue
            r2 = _get_r_squared(sm)
            if np.isnan(r2) or r2 < R2_MIN:
                continue
            for row in _get_peak_params(sm):
                if target_lo <= row[0] < target_hi:
                    all_peaks.append({
                        'freq': row[0], 'power': row[1],
                        'phi_octave': band['name'], 'method': 'perband'})

    return pd.DataFrame(all_peaks)


def extract_merged(raw, fs):
    """Merged extraction: single FOOOF across theta+alpha (4.70-12.30 Hz)."""
    target_lo = F0 * PHI**-1   # 4.70
    target_hi = F0 * PHI       # 12.30

    # Pad for FOOOF fit
    fit_lo = max(1.0, F0 * PHI**-1.5)  # ~3.66
    fit_hi = F0 * PHI**1.5              # ~15.88

    # Use the finer nperseg (theta's, since it needs more resolution)
    nperseg = compute_adaptive_nperseg(F0, -1, fs)
    nperseg = min(nperseg, int(60 * fs))
    freq_res = fs / nperseg
    fit_width = fit_hi - fit_lo

    max_n_peaks = max(3, min(20, int(fit_width / 1.0)))
    peak_width_limits = [max(0.5, 2 * freq_res), min(fit_width * 0.4, 12.0)]
    fooof_params = {**FOOOF_BASE_PARAMS,
                    'max_n_peaks': max_n_peaks,
                    'peak_width_limits': peak_width_limits}

    all_peaks = []
    for ch in raw.ch_names:
        try:
            data = raw.get_data(picks=[ch])[0]
        except Exception:
            continue
        if len(data) < nperseg:
            continue
        freqs, psd = welch(data, fs, nperseg=nperseg, noverlap=nperseg//2)
        sm = SpectralModel(**fooof_params)
        try:
            sm.fit(freqs, psd, [fit_lo, fit_hi])
        except Exception:
            continue
        r2 = _get_r_squared(sm)
        if np.isnan(r2) or r2 < R2_MIN:
            continue
        for row in _get_peak_params(sm):
            if target_lo <= row[0] < target_hi:
                # Assign to theta or alpha by frequency
                octave = 'n-1' if row[0] < F0 else 'n+0'
                all_peaks.append({
                    'freq': row[0], 'power': row[1],
                    'phi_octave': octave, 'method': 'merged'})

    return pd.DataFrame(all_peaks)


def main():
    data_dir = '/Volumes/T9/dortmund_data_dl'
    subs = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and
                   os.path.isdir(os.path.join(data_dir, d, 'ses-1', 'eeg'))])[:20]

    print(f"Testing merged theta+alpha extraction on {len(subs)} Dortmund subjects")
    print(f"f0 = {F0} Hz, theta = [{F0/PHI:.2f}, {F0:.2f}], alpha = [{F0:.2f}, {F0*PHI:.2f}]")

    all_perband = []
    all_merged = []

    for i, sub_id in enumerate(subs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = load_dortmund(sub_id)
        if raw is None:
            continue
        fs = raw.info['sfreq']

        t0 = time.time()
        pb = extract_perband(raw, fs)
        t1 = time.time()
        mg = extract_merged(raw, fs)
        t2 = time.time()

        all_perband.append(pb)
        all_merged.append(mg)

        print(f"  [{i+1}/{len(subs)}] {sub_id}: perband={len(pb)} peaks ({t1-t0:.1f}s), "
              f"merged={len(mg)} peaks ({t2-t1:.1f}s)")

        del raw

    pb_all = pd.concat(all_perband, ignore_index=True)
    mg_all = pd.concat(all_merged, ignore_index=True)

    print(f"\nTotal: perband={len(pb_all):,} peaks, merged={len(mg_all):,} peaks")
    print(f"  Theta: perband={len(pb_all[pb_all.phi_octave=='n-1']):,}, "
          f"merged={len(mg_all[mg_all.phi_octave=='n-1']):,}")
    print(f"  Alpha: perband={len(pb_all[pb_all.phi_octave=='n+0']):,}, "
          f"merged={len(mg_all[mg_all.phi_octave=='n+0']):,}")

    # Peak density comparison around f0
    print(f"\n{'='*70}")
    print(f"  PEAK DENSITY AROUND f0={F0} Hz (0.1 Hz bins)")
    print(f"  Per-band = separate FOOOF for theta and alpha")
    print(f"  Merged = one FOOOF across 4.70-12.30 Hz")
    print(f"{'='*70}")

    bins = np.arange(5.0, 12.0, 0.1)
    pb_counts, _ = np.histogram(pb_all['freq'].values, bins=bins)
    mg_counts, _ = np.histogram(mg_all['freq'].values, bins=bins)

    print(f"\n  {'Hz':<10s} {'Perband':>8s} {'Merged':>8s} {'Ratio':>7s}  {'Perband':40s} {'Merged'}")
    print(f"  {'-'*90}")

    max_c = max(max(pb_counts), max(mg_counts))
    for j in range(len(pb_counts)):
        hz = bins[j]
        pc = pb_counts[j]
        mc = mg_counts[j]
        ratio = mc / pc if pc > 0 else float('inf')
        pb_bar = '#' * max(0, int(pc / max_c * 35))
        mg_bar = '#' * max(0, int(mc / max_c * 35))
        marker = ''
        if abs(hz - F0) < 0.05:
            marker = ' <-- f0'
        print(f"  {hz:>5.1f}     {pc:>7,}  {mc:>7,}  {ratio:>6.2f}  {pb_bar:<40s} {mg_bar}{marker}")

    # Enrichment comparison for theta and alpha
    print(f"\n{'='*70}")
    print(f"  ENRICHMENT COMPARISON (Hz-corrected)")
    print(f"{'='*70}")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from run_all_f0_760_analyses import compute_enrichment, BAND_ORDER

    pb_enr = compute_enrichment(pb_all)
    mg_enr = compute_enrichment(mg_all)

    for band in ['theta', 'alpha']:
        pb_b = pb_enr[pb_enr['band'] == band]
        mg_b = mg_enr[mg_enr['band'] == band]

        print(f"\n  {band.upper()}")
        print(f"  {'Position':<16s} {'Perband':>8s} {'Merged':>8s} {'Δ':>6s}")
        print(f"  {'-'*40}")

        all_pos = list(pb_b['position'].values)
        for pos in all_pos:
            pb_row = pb_b[pb_b['position'] == pos]
            mg_row = mg_b[mg_b['position'] == pos]
            pv = int(pb_row.iloc[0]['enrichment_pct']) if (not pb_row.empty and not np.isnan(pb_row.iloc[0]['enrichment_pct'])) else None
            mv = int(mg_row.iloc[0]['enrichment_pct']) if (not mg_row.empty and not np.isnan(mg_row.iloc[0]['enrichment_pct'])) else None
            if pv is None and mv is None:
                continue
            delta = (mv - pv) if pv is not None and mv is not None else None
            sp = f'{pv:+d}%' if pv is not None else '—'
            sm = f'{mv:+d}%' if mv is not None else '—'
            sd = f'{delta:+d}' if delta is not None else '—'
            pos_d = 'bnd_hi' if pos == 'boundary_hi' else pos
            print(f"  {pos_d:<16s} {sp:>8s} {sm:>8s} {sd:>6s}")


if __name__ == '__main__':
    main()
