#!/usr/bin/env python3
"""Off-frequency detector control for the SR1 spectral peak.

Test whether the detector's envelope-band center frequency f0 drives the
location of the event-locked spectral peak. Re-runs the four-stream
composite-v2 detector (envelope × Kuramoto R × PLV × MSC) with f0 shifted
to an alternative center, on LEMON EC subjects. Computes the event-locked
log10(event/baseline) ratio spectrum on the events the off-frequency
detector finds, and reports the peak location.

Predictions:
  - If the SR1 alignment is genuine (a property of the brain signal at
    ~7.83 Hz, not an artifact of the detector): off-frequency detectors
    should still produce event-locked spectra with peaks near 7.83 Hz,
    OR fail to find an event population at all.
  - If the spectral peak is detector-induced: off-frequency detectors
    should produce event-locked peaks at the shifted f0 (e.g., ~8.6 Hz
    for f0=8.6 detection).

Usage:
    SIE_OFF_F0=8.6 SIE_OFF_N=50 SIE_OFF_WORKERS=4 \\
      python scripts/sie_off_frequency_detector_control.py

Outputs:
  outputs/schumann/images/psd_timelapse/lemon_composite/
    off_frequency_event_locked_psd_f0_{F0}.csv

Default subset: 50 LEMON EC subjects (matched to random-window null pilot).
"""
from __future__ import annotations
import os
import sys
import glob as globfn
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.detect_ignition import _composite_streams, _composite_S, _composite_refine_onset
from scripts.run_sie_extraction import load_lemon
from scipy import signal
import mne

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse', 'lemon_composite')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie',
                           'lemon_composite')
os.makedirs(OUT_DIR, exist_ok=True)

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
FREQ_LO, FREQ_HI = 2.0, 25.0
COMP_THRESH = 1.5
MIN_ISI = 2.0
EDGE_S = 5.0


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def detect_offset(Y, fs, f0_offset):
    """Run composite-v2 detector with f0 and R_band BOTH shifted."""
    half_bw = 0.6
    R_band = (f0_offset - 0.6, f0_offset + 0.6)  # symmetric around f0
    t, env, R, P, M = _composite_streams(Y, fs, f0=f0_offset, half_bw=half_bw,
                                         R_band=R_band)
    S = _composite_S(env, R, P, M)
    mask = (t >= t[0] + EDGE_S) & (t <= t[-1] - EDGE_S)
    S_m = S.copy()
    S_m[~mask] = -np.inf
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    peak_idx, _ = signal.find_peaks(
        S_m, distance=max(1, int(round(MIN_ISI / dt))), height=COMP_THRESH,
    )
    onsets = []
    for pi in peak_idx:
        t_on = _composite_refine_onset(t, env, R, P, M, float(t[pi]))
        onsets.append(t_on)
    return np.array(sorted(onsets))


def process_subject(args):
    sub_id, f0_offset = args
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6  # to µV
    rec_dur = X.shape[1] / fs
    if rec_dur < 60:
        return None

    onsets = detect_offset(X, fs, f0_offset)
    if len(onsets) < 1:
        return None

    y = X.mean(axis=0)
    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    fmask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    f_band = freqs[fmask]

    # Baseline = median over all sliding windows
    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)[fmask]
        base_rows.append(psd)
    if len(base_rows) < 10:
        return None
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    # Event-locked PSDs at +1 s lag (B27 convention)
    ev_rows = []
    for tc_base in onsets:
        tc = tc_base + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[fmask]
        ev_rows.append(psd)
    if not ev_rows:
        return None
    event_avg = np.nanmean(np.array(ev_rows), axis=0)
    log_ratio = np.log10(event_avg / baseline)
    return (sub_id, f_band, log_ratio, len(ev_rows), float(rec_dur))


def main():
    f0_offset = float(os.environ.get('SIE_OFF_F0', '8.6'))
    n_max = int(os.environ.get('SIE_OFF_N', '50'))
    n_workers = int(os.environ.get('SIE_OFF_WORKERS', '4'))

    csvs = sorted(globfn.glob(os.path.join(EVENTS_BASE, 'sub-*_sie_events.csv')))
    sub_ids = [os.path.basename(p).replace('_sie_events.csv', '') for p in csvs]
    sub_ids = sub_ids[:n_max]
    print(f'Off-frequency detector control')
    print(f'  f0_offset = {f0_offset:.2f} Hz (default detector: 7.83 Hz)')
    print(f'  R_band    = ({f0_offset-0.6:.2f}, {f0_offset+0.6:.2f}) Hz')
    print(f'  Subjects  = {len(sub_ids)} (LEMON EC)')
    print(f'  Workers   = {n_workers}')

    args_list = [(s, f0_offset) for s in sub_ids]
    results = []
    with Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_subject, args_list)):
            if res is not None:
                results.append(res)
            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(args_list)} processed; {len(results)} valid')

    if not results:
        print('No valid results.')
        return

    sub_ids_done, f_bands, ratios, n_evs, durs = zip(*results)
    f_band = f_bands[0]
    R = np.array(ratios)
    print(f'Pooled {len(results)} subjects')
    print(f'  Mean events per subject: {np.mean(n_evs):.1f}  '
          f'(median {np.median(n_evs):.1f}, range {min(n_evs)}-{max(n_evs)})')

    # Grand mean + bootstrap CI
    grand_mean = np.nanmean(R, axis=0)
    n_boot = 1000
    rng = np.random.default_rng(2026)
    bm = np.zeros((n_boot, R.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, R.shape[0], R.shape[0])
        bm[b] = np.nanmean(R[idx], axis=0)
    ci_lo = np.nanpercentile(bm, 2.5, axis=0)
    ci_hi = np.nanpercentile(bm, 97.5, axis=0)

    out_path = os.path.join(OUT_DIR, f'off_frequency_event_locked_psd_f0_{f0_offset:.1f}.csv')
    df = pd.DataFrame({
        'freq_hz': f_band,
        'log_ratio_grand': grand_mean,
        'lo': ci_lo,
        'hi': ci_hi,
    })
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')

    # Report peak in three relevant ranges
    print()
    print('Peak frequency in three regions:')
    for lo, hi, label in [(7.0, 8.5, 'SR1 band'),
                           (f0_offset - 0.5, f0_offset + 0.5, f'off-f0 band ({f0_offset} Hz)'),
                           (4.0, 13.0, 'broad alpha-region zoom')]:
        m = (f_band >= lo) & (f_band <= hi)
        if m.any():
            sub = grand_mean[m]
            sub_f = f_band[m]
            pi = int(np.argmax(sub))
            print(f'  {label:>32s} [{lo:.1f}-{hi:.1f}]: peak at '
                  f'{sub_f[pi]:.2f} Hz, value {sub[pi]:+.4f}')

    # Compute peak in SR1 band vs off-f0 band as a discriminator
    sr1_m = (f_band >= 7.0) & (f_band <= 8.5)
    off_m = (f_band >= f0_offset - 0.5) & (f_band <= f0_offset + 0.5)
    if sr1_m.any() and off_m.any():
        sr1_peak = float(np.max(grand_mean[sr1_m]))
        off_peak = float(np.max(grand_mean[off_m]))
        print()
        print(f'SR1-band peak ({sr1_peak:+.4f}) vs off-f0-band peak ({off_peak:+.4f})')
        if sr1_peak > off_peak:
            print(f'  -> Event-locked spectrum still favors SR1 over the detector center f0={f0_offset}')
            print(f'     (consistent with SR1 alignment being a brain-signal property,')
            print(f'      not driven by detector f0)')
        else:
            print(f'  -> Event-locked spectrum tracks the detector f0={f0_offset}')
            print(f'     (consistent with detector-induced peak)')


if __name__ == '__main__':
    main()
