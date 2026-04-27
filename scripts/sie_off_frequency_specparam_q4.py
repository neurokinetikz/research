#!/usr/bin/env python3
"""Test (i) Q4-scoped: specparam fits on Q4 canonical event-locked aggregates.

For each f0 in {7.6, 8.6, 12.0}:
  1. Run composite-v2 detector at f0 on 50 LEMON EC subjects
  2. For each event, compute envelope-z trajectory (f0-band) on [-5, +5] s
  3. Build PER-DETECTOR cohort template (mean envelope-z trajectory)
  4. Score each event's template_rho = Pearson(event_traj, template)
  5. Q4 filter: keep events with template_rho >= per-subject 75th percentile
     (subjects with <4 events: skipped)
  6. Compute event-locked PSDs on Q4 events only
  7. Cohort-aggregate PSDs, specparam-fit, read off residuals

Output:
  outputs/schumann/images/psd_timelapse/lemon_composite/
    off_frequency_specparam_q4_residuals.csv
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
from lib.detect_ignition import (_composite_streams, _composite_S,
                                  _composite_refine_onset, bandpass_safe)
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

# Template / Q4 scoring
TPL_PRE = 5.0   # template runs from -5 to +5 s relative to onset (paper convention)
TPL_POST = 5.0
TPL_STEP = 0.1
TGRID = np.arange(-TPL_PRE, TPL_POST + TPL_STEP / 2, TPL_STEP)
Q4_QUANTILE = 0.75

F0_LIST = [7.6, 8.6, 12.0]


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-9:
        return x - med
    return (x - med) / mad


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


def compute_envelope_z(Y, fs, f0):
    """Whole-recording envelope-z at TPL_STEP resolution, in the f0 ± 0.6 Hz band."""
    half_bw = 0.6
    y = Y.mean(axis=0)
    yb = bandpass_safe(y, fs, f0 - half_bw, f0 + half_bw)
    env = np.abs(signal.hilbert(yb))
    nwin = int(round(1.0 * fs))
    nstep = int(round(TPL_STEP * fs))
    env_vals = []
    centers = []
    for i in range(0, len(env) - nwin + 1, nstep):
        env_vals.append(float(np.mean(env[i:i+nwin])))
        centers.append((i + nwin / 2) / fs)
    t = np.array(centers)
    zE = robust_z(np.array(env_vals))
    return t, zE


def process_subject_pass1(args):
    """Pass 1: detect events, compute per-event envelope-z trajectories
    (for template building). Returns (sub_id, fs, onsets, traj_per_event)."""
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
    if len(onsets) < 4:
        return None
    t_stream, zE_stream = compute_envelope_z(X, fs, f0)
    trajs = []
    valid_onsets = []
    for t0 in onsets:
        rel = t_stream - t0
        m = (rel >= -TPL_PRE - 0.3) & (rel <= TPL_POST + 0.3)
        if m.sum() < int((TPL_PRE + TPL_POST) * 5):
            continue
        traj = np.interp(TGRID, rel[m], zE_stream[m], left=np.nan, right=np.nan)
        if np.any(~np.isfinite(traj)):
            continue
        trajs.append(traj)
        valid_onsets.append(t0)
    if len(trajs) < 4:
        return None
    return (sub_id, fs, np.array(valid_onsets), np.array(trajs))


def process_subject_pass2(args):
    """Pass 2: with template+rhos in hand, compute Q4 event-locked PSDs."""
    sub_id, f0, q4_t0s = args
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None or len(q4_t0s) == 0:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    nperseg = int(round(EV_WIN_SEC * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    fmask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    f_band = freqs[fmask]
    ev_rows = []
    for tc_base in q4_t0s:
        tc = tc_base + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[fmask]
        ev_rows.append(psd)
    if not ev_rows:
        return None
    sub_psd = np.nanmean(np.array(ev_rows), axis=0)
    return (sub_id, f_band, sub_psd, len(ev_rows))


def fit_specparam(freqs, psd_linear):
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
    except Exception:
        return None
    return sm


def residual_at(sm, target_hz, half_bw=0.4):
    if sm is None:
        return np.nan
    f = sm.data.freqs
    obs = sm.data.power_spectrum
    aper = sm.results.model._ap_fit
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
    print(f'Test (i) Q4-scoped: per-detector cohort template, top-quartile per subject')
    print(f'  Subjects = {len(sub_ids)} (LEMON EC), workers = {n_workers}')
    print()

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
    for f0 in F0_LIST:
        print(f'=== f0 = {f0:.1f} Hz ===')
        # PASS 1: detect + collect envelope trajectories
        args1 = [(s, f0) for s in sub_ids]
        pass1 = []
        with Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_subject_pass1, args1)):
                if res is not None:
                    pass1.append(res)
                if (i + 1) % 25 == 0:
                    print(f'  pass1: {i+1}/{len(args1)} processed; {len(pass1)} valid')
        if not pass1:
            continue
        # Build per-detector cohort template (grand mean across all events)
        all_trajs = np.concatenate([p[3] for p in pass1], axis=0)
        template = np.nanmean(all_trajs, axis=0)
        m_core = (TGRID >= -5) & (TGRID <= 5)
        tmpl_core = template[m_core] - np.nanmean(template[m_core])
        print(f'  pass1: {len(pass1)} subjects; '
              f'{all_trajs.shape[0]} events; template built on [-5, +5] s')

        # Per-event template_rho, then Q4 filter
        pass2_args = []
        n_total_q4 = 0
        for sub_id, fs, onsets, trajs in pass1:
            ev_core = trajs[:, m_core]
            rhos = []
            for k in range(ev_core.shape[0]):
                ev_centered = ev_core[k] - np.nanmean(ev_core[k])
                num = np.nansum(ev_centered * tmpl_core)
                den = np.sqrt(np.nansum(ev_centered**2) * np.nansum(tmpl_core**2))
                rhos.append(num / den if den > 0 else np.nan)
            rhos = np.array(rhos)
            if np.sum(np.isfinite(rhos)) < 4:
                continue
            thr = np.nanquantile(rhos, Q4_QUANTILE)
            q4_mask = rhos >= thr
            q4_t0s = onsets[q4_mask]
            n_total_q4 += len(q4_t0s)
            if len(q4_t0s) >= 1:
                pass2_args.append((sub_id, f0, q4_t0s))
        print(f'  Q4 filter: {n_total_q4} Q4 events across {len(pass2_args)} subjects '
              f'(top quartile per subject by template_rho)')

        # PASS 2: compute event-locked PSDs on Q4 events
        pass2 = []
        with Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_subject_pass2, pass2_args)):
                if res is not None:
                    pass2.append(res)
                if (i + 1) % 25 == 0:
                    print(f'  pass2: {i+1}/{len(pass2_args)} processed; {len(pass2)} valid')
        if not pass2:
            continue
        sub_ids_done, f_bands, sub_psds, n_evs = zip(*pass2)
        f_band = f_bands[0]
        psds = np.array(sub_psds)
        log_psds = np.log10(psds + 1e-30)
        agg_log = np.nanmean(log_psds, axis=0)
        agg_lin = 10 ** agg_log
        print(f'  Aggregated {len(pass2)} subjects, {sum(n_evs)} Q4 events')
        sm = fit_specparam(f_band, agg_lin)
        if sm is None:
            continue
        ap = sm.get_params('aperiodic')
        peaks = sm.get_params('peak')
        if ap is not None and len(ap) >= 2:
            print(f'  Aperiodic offset={ap[0]:.3f}, exponent={ap[1]:.3f}')
        if peaks is not None:
            peaks_arr = np.atleast_2d(peaks)
            if peaks_arr.shape[1] >= 3 and peaks_arr.size > 0:
                print('  Detected peaks (CF, PW, BW):')
                for pp in peaks_arr:
                    print(f'    {pp[0]:>5.2f} Hz  PW={pp[1]:.3f}  BW={pp[2]:.3f}')
        for label, fc in targets.items():
            r = residual_at(sm, fc)
            results_rows.append(dict(
                f0_detector=f0, target_label=label, target_hz=fc,
                residual_log10=r,
            ))
        print()

    df = pd.DataFrame(results_rows)
    out_path = os.path.join(OUT_DIR, 'off_frequency_specparam_q4_residuals.csv')
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}\n')

    pv = df.pivot(index='target_label', columns='f0_detector', values='residual_log10')
    order = ['SR1 (7.83)', 'SR2 (14.0)', 'SR3 (19.95)', 'SR4 (26.0)',
             'f0=8.6', '2*8.6=17.2', '3*8.6=25.8',
             'f0=12.0', '2*12.0=24.0']
    pv = pv.reindex(order)
    print('=' * 78)
    print('Specparam residuals (log10 above 1/f) at each mode -- Q4 EVENTS ONLY')
    print('=' * 78)
    print()
    print(pv.to_string(float_format=lambda x: f'{x:+.3f}' if np.isfinite(x) else '   --'))
    print()
    print('Compare to all-event analysis: off_frequency_specparam_residuals.csv')

    # Odd/even ratio at Schumann stack per detector
    print()
    print('Odd/even ratio at Schumann stack (Q4 events):')
    for f0 in F0_LIST:
        if f0 not in pv.columns:
            continue
        col = pv[f0]
        sr1 = col.get('SR1 (7.83)', np.nan)
        sr2 = col.get('SR2 (14.0)', np.nan)
        sr3 = col.get('SR3 (19.95)', np.nan)
        sr4 = col.get('SR4 (26.0)', np.nan)
        odd = sr1 + sr3
        even = sr2 + sr4
        if even > 0:
            ratio = f'{odd/even:.2f}x'
        else:
            ratio = f'odd={odd:+.3f}, even={even:+.3f} (denom <= 0)'
        label = f'f0={f0:.1f}' + (' (canon)' if f0 == 7.6 else '')
        print(f'  {label}: {ratio}')


if __name__ == '__main__':
    main()
