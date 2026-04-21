#!/usr/bin/env python3
"""Quick S₃ vs S₄ composite test (MSC vs no-MSC).

For a few LEMON subjects: compute streams with and without MSC, detect
events, measure timing, and compare event overlap against the already-
extracted S₄ events in exports_sie/lemon_composite/.
"""
from __future__ import annotations
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import bandpass

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie',
                           'lemon_composite')

F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)
STEP_SEC = 0.1
WIN_SEC = 1.0
THRESHOLD = 1.5


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-9: return x - med
    return (x - med) / mad


def compute_streams(Y, fs, include_msc=True):
    """Return (t, env, R, PLV, MSC_or_None)."""
    y = Y.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))

    Xb = bandpass(Y, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    ref_b = np.median(Xb, axis=0)
    ph_ref = np.angle(signal.hilbert(ref_b))
    dphi = ph - ph_ref[None, :]
    ref_raw = np.median(Y, axis=0)

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    nperseg_msc = max(int(round(0.5 * fs)), 32)

    centers, env_v, R_v, P_v, M_v = [], [], [], [], []
    for i in range(0, Y.shape[1] - nwin + 1, nstep):
        seg_ph = ph[:, i:i+nwin]
        R_v.append(float(np.mean(np.abs(np.mean(np.exp(1j * seg_ph), axis=0)))))
        pseg = dphi[:, i:i+nwin]
        P_v.append(float(np.mean(np.abs(np.mean(np.exp(1j * pseg), axis=1)))))
        env_v.append(float(np.mean(env[i:i+nwin])))
        if include_msc:
            ref_seg = ref_raw[i:i+nwin]
            msc = []
            for ci in range(Y.shape[0]):
                try:
                    f_c, Cxy = signal.coherence(Y[ci, i:i+nwin], ref_seg,
                                                  fs=fs,
                                                  nperseg=min(nperseg_msc, nwin))
                    msc.append(float(Cxy[int(np.argmin(np.abs(f_c - F0)))]))
                except Exception:
                    pass
            M_v.append(float(np.mean(msc)) if msc else np.nan)
        centers.append((i + nwin/2) / fs)
    t = np.array(centers)
    env_arr = np.array(env_v); R_arr = np.array(R_v); P_arr = np.array(P_v)
    M_arr = np.array(M_v) if include_msc else None
    return t, env_arr, R_arr, P_arr, M_arr


def compose_and_detect(env, R, P, M, threshold=THRESHOLD, n_streams=4):
    zE = robust_z(env); zR = robust_z(R); zP = robust_z(P)
    if n_streams == 4:
        zM = robust_z(M)
        prod = (np.clip(zE, 0, None) * np.clip(zR, 0, None) *
                np.clip(zP, 0, None) * np.clip(zM, 0, None))
    else:
        prod = (np.clip(zE, 0, None) * np.clip(zR, 0, None) *
                np.clip(zP, 0, None))
    S = np.cbrt(prod) if n_streams == 4 else np.power(prod, 1/3)
    # Masking edges + peak detection
    return S


def detect_peaks(t, S, threshold, min_isi=2.0, edge=5.0):
    mask = (t >= t[0] + edge) & (t <= t[-1] - edge)
    S_m = S.copy(); S_m[~mask] = -np.inf
    peaks, _ = signal.find_peaks(S_m,
                                  distance=int(round(min_isi / STEP_SEC)),
                                  height=threshold)
    return t[peaks]


def align_sets(a, b, tol=2.0):
    """Fraction of a within tol of some b."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(np.mean([np.min(np.abs(b - x)) <= tol for x in a]))


def main():
    rng = np.random.default_rng(42)
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=4)]
    subs = rng.choice(ok['subject_id'].values,
                       size=min(5, len(ok)), replace=False).tolist()
    print(f"Testing on {len(subs)} LEMON subjects")
    print(f"Threshold: S ≥ {THRESHOLD}")
    print()
    print(f"{'sub':<12}{'S4_time':>10}{'S3_time':>10}{'speedup':>10}"
          f"{'n_S4':>8}{'n_S3':>8}{'%S4→S3':>10}{'%S3→S4':>10}"
          f"{'n_legacy':>10}")
    print('-' * 90)

    for sub in subs:
        raw = load_lemon(sub, condition='EC')
        if raw is None: continue
        Y = raw.get_data() * 1e6
        fs = raw.info['sfreq']

        t0 = time.time()
        t, env, R, P, M = compute_streams(Y, fs, include_msc=True)
        t_s4 = time.time() - t0
        S4 = compose_and_detect(env, R, P, M, n_streams=4)
        peaks_s4 = detect_peaks(t, S4, THRESHOLD)

        t0 = time.time()
        t3, env3, R3, P3, _ = compute_streams(Y, fs, include_msc=False)
        t_s3 = time.time() - t0
        S3 = compose_and_detect(env3, R3, P3, None, n_streams=3)
        peaks_s3 = detect_peaks(t3, S3, THRESHOLD)

        # Legacy S₄-extracted event file for reference
        ev_path = os.path.join(EVENTS_DIR, f'{sub}_sie_events.csv')
        legacy = pd.read_csv(ev_path) if os.path.isfile(ev_path) else None
        leg_times = np.array(legacy['t0_net'].values) if legacy is not None else np.array([])

        s4_to_s3 = align_sets(peaks_s4, peaks_s3) * 100
        s3_to_s4 = align_sets(peaks_s3, peaks_s4) * 100
        print(f"{sub:<12}{t_s4:>10.1f}{t_s3:>10.1f}{t_s4/max(t_s3,0.001):>9.1f}x"
              f"{len(peaks_s4):>8}{len(peaks_s3):>8}"
              f"{s4_to_s3:>9.0f}%{s3_to_s4:>9.0f}%"
              f"{len(leg_times):>10}")


if __name__ == '__main__':
    main()
