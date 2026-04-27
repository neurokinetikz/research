#!/usr/bin/env python3
"""Test (i) definitive: specparam (FOOOF) on cohort-aggregate event-locked
PSDs for canonical and off-f0 detectors.

For each f0 in {7.6, 8.6, 12.0}:
  1. Run composite-v2 detector at f0 on 50 LEMON EC subjects
  2. For each subject, compute mean event-locked PSD (raw, in V^2/Hz)
  3. Compute cohort-aggregate (geometric mean across subjects)
  4. specparam-fit the aggregate over [2, 30] Hz
  5. Read off aperiodic-corrected residuals at:
     - Schumann modes: SR1=7.83, SR2=14.0, SR3=19.95, SR4=26.0
     - f0 stack: f0, 2*f0, 3*f0

This directly tests whether SR3 is genuinely elevated above the 1/f
aperiodic floor under canonical detection (and under off-f0 detection
to control for narrowband-detector-induced peak signatures).

Output:
  outputs/schumann/images/psd_timelapse/lemon_composite/
    off_frequency_specparam_residuals.csv
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
from specparam import SpectralModel

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
FREQ_LO, FREQ_HI = 2.0, 30.0
COMP_THRESH = 1.5
MIN_ISI = 2.0
EDGE_S = 5.0

F0_LIST = [7.6, 8.6, 12.0]


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def detect_offset(Y, fs, f0):
    half_bw = 0.6
    R_band = (f0 - 0.6, f0 + 0.6)
    t, env, R, P, M = _composite_streams(Y, fs, f0=f0, half_bw=half_bw, R_band=R_band)
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
    sub_id, f0 = args
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    rec_dur = X.shape[1] / fs
    if rec_dur < 60:
        return None

    onsets = detect_offset(X, fs, f0)
    if len(onsets) < 1:
        return None

    y = X.mean(axis=0)
    nperseg = int(round(EV_WIN_SEC * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    fmask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    f_band = freqs[fmask]

    # Raw event-locked PSDs
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
    # Subject-mean PSD (linear)
    sub_psd = np.nanmean(np.array(ev_rows), axis=0)
    return (sub_id, f_band, sub_psd, len(ev_rows))


def fit_specparam(freqs, psd_linear, label=''):
    """Fit specparam (knee=False) over [2, 30] Hz; return fit object + peaks."""
    sm = SpectralModel(
        peak_width_limits=(0.5, 4.0),
        max_n_peaks=8,
        min_peak_height=0.05,
        peak_threshold=2.0,
        aperiodic_mode='fixed',
        verbose=False,
    )
    try:
        sm.fit(freqs, psd_linear, freq_range=(2.0, 30.0))
    except Exception as e:
        print(f'  {label}: specparam fit failed ({e})')
        return None
    return sm


def residual_at(sm, target_hz, half_bw=0.4):
    """log10(observed power) minus aperiodic-only fit at target ±half_bw, max.
    Uses the actual data, not the model peak component, so we capture genuine
    elevation above the 1/f fit even when specparam declined to fit a peak."""
    if sm is None:
        return np.nan
    f = sm.data.freqs
    obs = sm.data.power_spectrum  # log10 power (specparam stores in log10)
    aper = sm.results.model._ap_fit  # log10 aperiodic-only fit
    resid = obs - aper
    m = (f >= target_hz - half_bw) & (f <= target_hz + half_bw)
    if not m.any():
        return np.nan
    return float(np.max(resid[m]))


def main():
    n_max = int(os.environ.get('SIE_OFF_N', '50'))
    n_workers = int(os.environ.get('SIE_OFF_WORKERS', '4'))

    csvs = sorted(globfn.glob(os.path.join(EVENTS_BASE, 'sub-*_sie_events.csv')))
    sub_ids = [os.path.basename(p).replace('_sie_events.csv', '') for p in csvs]
    sub_ids = sub_ids[:n_max]
    print(f'Test (i) definitive: specparam fits on event-locked aggregate PSDs')
    print(f'  Subjects = {len(sub_ids)} (LEMON EC)')
    print()

    # Schumann + each f0 stack targets
    targets = {
        'SR1 (7.83)':   7.83,
        'SR2 (14.0)':   14.0,
        'SR3 (19.95)':  19.95,
        'SR4 (26.0)':   26.0,
        'f0=8.6':       8.6,
        '2*8.6=17.2':   17.2,
        '3*8.6=25.8':   25.8,
        'f0=12.0':      12.0,
        '2*12.0=24.0':  24.0,
    }

    results_rows = []
    aggregates = {}
    for f0 in F0_LIST:
        print(f'=== f0 = {f0:.1f} Hz ===')
        args_list = [(s, f0) for s in sub_ids]
        per_sub = []
        with Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_subject, args_list)):
                if res is not None:
                    per_sub.append(res)
                if (i + 1) % 25 == 0:
                    print(f'  {i+1}/{len(args_list)} processed; {len(per_sub)} valid')
        if not per_sub:
            continue
        # Geometric-mean aggregate across subjects (= mean of log-PSD)
        sub_ids_done, f_bands, sub_psds, n_evs = zip(*per_sub)
        f_band = f_bands[0]
        psds = np.array(sub_psds)  # n_sub × n_freq, linear
        log_psds = np.log10(psds + 1e-30)
        agg_log = np.nanmean(log_psds, axis=0)
        agg_lin = 10 ** agg_log  # geometric-mean linear PSD
        aggregates[f0] = (f_band, agg_lin, len(per_sub), int(np.sum(n_evs)))
        print(f'  Aggregated {len(per_sub)} subjects, {sum(n_evs)} events total')

        sm = fit_specparam(f_band, agg_lin, label=f'f0={f0}')
        if sm is None:
            continue
        ap = sm.get_params('aperiodic')
        peaks = sm.get_params('peak')
        if ap is not None and len(ap) >= 2:
            print(f'  Aperiodic offset={ap[0]:.3f}, exponent={ap[1]:.3f}')
        if peaks is not None and np.atleast_2d(peaks).size > 0:
            peaks_arr = np.atleast_2d(peaks)
            if peaks_arr.shape[1] >= 3:
                print(f'  Detected peaks (CF, PW, BW):')
                for pp in peaks_arr:
                    print(f'    {pp[0]:>5.2f} Hz  PW={pp[1]:.3f}  BW={pp[2]:.3f}')
        else:
            print('  No peaks detected.')

        for label, fc in targets.items():
            r = residual_at(sm, fc)
            results_rows.append(dict(
                f0_detector=f0, target_label=label, target_hz=fc,
                residual_log10=r,
            ))
        print()

    df = pd.DataFrame(results_rows)
    out_path = os.path.join(OUT_DIR, 'off_frequency_specparam_residuals.csv')
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')
    print()

    # Pivot for display
    pv = df.pivot(index='target_label', columns='f0_detector', values='residual_log10')
    print('=' * 78)
    print('Specparam residuals (log10 power above 1/f aperiodic fit) at each mode')
    print('=' * 78)
    print()
    # Order rows in a sensible way
    order = ['SR1 (7.83)', 'SR2 (14.0)', 'SR3 (19.95)', 'SR4 (26.0)',
             'f0=8.6', '2*8.6=17.2', '3*8.6=25.8',
             'f0=12.0', '2*12.0=24.0']
    pv = pv.reindex(order)
    print(pv.to_string(float_format=lambda x: f'{x:+.3f}' if np.isfinite(x) else '   --'))
    print()

    # Headline comparison
    print('=' * 78)
    print('HEADLINE COMPARISON: SR3 elevation under canonical vs off-f0 detection')
    print('=' * 78)
    canon = pv.get(7.6, pd.Series(dtype=float))
    off86 = pv.get(8.6, pd.Series(dtype=float))
    off12 = pv.get(12.0, pd.Series(dtype=float))
    sr3_c = canon.get('SR3 (19.95)', np.nan)
    sr3_8 = off86.get('SR3 (19.95)', np.nan)
    sr3_2 = off12.get('SR3 (19.95)', np.nan)
    sr1_c = canon.get('SR1 (7.83)', np.nan)
    print(f'  SR1 residual canonical: {sr1_c:+.3f}')
    print(f'  SR3 residual canonical: {sr3_c:+.3f}')
    print(f'  SR3 residual f0=8.6:    {sr3_8:+.3f}')
    print(f'  SR3 residual f0=12.0:   {sr3_2:+.3f}')
    print()
    if np.isfinite(sr3_c):
        if sr3_c > 0.05:
            print('  -> SR3 IS elevated above 1/f floor under canonical detection.')
            print('     The "SR1+SR3 elevated" odd-mode claim survives specparam fit.')
        else:
            print('  -> SR3 is NOT meaningfully elevated above 1/f floor under canonical.')
            print('     The "SR1+SR3 elevated" odd-mode claim does NOT survive specparam.')
            print('     Recommend: tighten paper to "SR1 elevated; other Schumann modes')
            print('     at floor".')


if __name__ == '__main__':
    main()
