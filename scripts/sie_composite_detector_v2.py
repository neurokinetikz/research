#!/usr/bin/env python3
"""
Composite-detector v2 (reference implementation for lib/ port).

Canonical spec for the planned Stage-1 replacement:

  Streams (all at 7.83 ± 0.6 Hz, 100-ms resolution, 1-s sliding window):
    zE(t)  : z-scored envelope of mean-across-channel signal at F0
    zR(t)  : Kuramoto order parameter R across all channels in 7.2–8.4 Hz
    zP(t)  : mean PLV to median reference (bandpassed) in 7.2–8.4 Hz
    zM(t)  : mean MSC to median reference at F0

  Detection trigger (fires event candidates):
    S(t) = cbrt( max(zE,0) · max(zR,0) · max(zP,0) · max(zM,0) )
    - robust-z each stream against whole recording (median / 1.4826·MAD)
    - local maxima of S ≥ threshold, min ISI = 2 s, edge mask = 5 s
    - threshold: calibrated via phase-shuffle surrogate to target FAR
      (default 0.01/s) OR fixed at S ≥ 1.5 for comparability

  Per-event onset refinement (t_onset = NADIR):
    Within [t_detect - 3.0, t_detect + 0.4] s:
      onset_score(t) = zE_local(t) + zR_local(t) + zP_local(t) + zM_local(t)
      where each _local is z-scored against pre-event baseline [-5, -3] s
    t_onset = argmin(onset_score) in the above window

  Per-event outputs:
    t_detect     : S-peak time (detection trigger)
    t_onset      : joint-dip nadir (= physiological onset, = t₀ going forward)
    t_peak       : argmax S in [t_onset, t_onset + 3]
    duration     : return-to-baseline time − t_onset
    stream values at t_onset, t_peak
    harmonic frequencies at 20-s window centered on t_onset
      (the paper's downstream analyses continue to use 20-s windows;
       narrow-window variants stay as an option)

This file is a self-contained reference for porting into
lib/detect_ignition.py (replacing stages 1 and 3 together) once the
surrounding analyses in ANALYSES.md are complete.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.sie_perionset_triple_average import bandpass

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)
STEP_SEC = 0.1
WIN_SEC = 1.0
EDGE_SEC = 5.0
MIN_ISI_SEC = 2.0

# Onset-refinement search window (seconds, relative to t_detect)
ONSET_SEARCH = (-3.0, 0.4)

# Local baseline for onset z-scoring (seconds, relative to t_detect)
LOCAL_BASELINE = (-5.0, -3.0)


# -------------------------------------------------------------------------
# Stream computation
# -------------------------------------------------------------------------
def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-9:
        return x - med
    return (x - med) / mad


def compute_streams(X_uV: np.ndarray, fs: float):
    """
    Compute the four detector streams for a multichannel segment.

    Returns
    -------
    t       : (T,) time grid in seconds, windowed at STEP_SEC
    env     : (T,) envelope at F0, averaged over STEP_SEC windows
    R       : (T,) Kuramoto order parameter across channels in R_BAND
    PLV     : (T,) mean PLV to median reference in R_BAND
    MSC     : (T,) mean MSC to median reference at F0
    """
    y = X_uV.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))  # raw envelope; z-scoring happens downstream

    Xb = bandpass(X_uV, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    ref_b = np.median(Xb, axis=0)
    ph_ref = np.angle(signal.hilbert(ref_b))
    dphi = ph - ph_ref[None, :]

    # Raw signal (unfiltered) and its median ref for MSC — avoids
    # saturation that occurs when all channels are pre-filtered to the
    # same band before coherence.
    ref_raw = np.median(X_uV, axis=0)

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    # nperseg for coherence: use 0.5 s to give multiple segments per 1 s window
    nperseg_msc = max(int(round(0.5 * fs)), 32)

    centers, env_v, R_v, P_v, M_v = [], [], [], [], []
    for i in range(0, X_uV.shape[1] - nwin + 1, nstep):
        seg_ph = ph[:, i:i+nwin]
        R_t = np.abs(np.mean(np.exp(1j * seg_ph), axis=0))
        R_v.append(float(np.mean(R_t)))
        pseg = dphi[:, i:i+nwin]
        plv = np.abs(np.mean(np.exp(1j * pseg), axis=1))
        P_v.append(float(np.mean(plv)))
        env_v.append(float(np.mean(env[i:i+nwin])))
        # MSC on RAW signals at F0
        ref_seg = ref_raw[i:i+nwin]
        msc_per_ch = []
        for ci in range(X_uV.shape[0]):
            try:
                f_c, Cxy = signal.coherence(X_uV[ci, i:i+nwin], ref_seg, fs=fs,
                                              nperseg=min(nperseg_msc, nwin))
                msc_per_ch.append(float(Cxy[int(np.argmin(np.abs(f_c - F0)))]))
            except Exception:
                pass
        M_v.append(float(np.mean(msc_per_ch)) if msc_per_ch else np.nan)
        centers.append((i + nwin/2) / fs)
    return (np.array(centers), np.array(env_v), np.array(R_v),
            np.array(P_v), np.array(M_v))


# -------------------------------------------------------------------------
# Composite, detection, refinement
# -------------------------------------------------------------------------
def composite_S(env: np.ndarray, R: np.ndarray, P: np.ndarray,
                 M: np.ndarray) -> np.ndarray:
    """Four-stream composite (robust-z on whole recording)."""
    zE, zR, zP, zM = robust_z(env), robust_z(R), robust_z(P), robust_z(M)
    return np.cbrt(
        np.clip(zE, 0, None) * np.clip(zR, 0, None) *
        np.clip(zP, 0, None) * np.clip(zM, 0, None)
    )


def detect_events(t: np.ndarray, S: np.ndarray,
                  threshold: float = 1.5,
                  min_isi: float = MIN_ISI_SEC,
                  edge_sec: float = EDGE_SEC):
    """Local maxima of S above threshold, with min-ISI and edge masking."""
    mask = (t >= t[0] + edge_sec) & (t <= t[-1] - edge_sec)
    S_m = S.copy()
    S_m[~mask] = -np.inf
    peak_idx, _ = signal.find_peaks(S_m, distance=int(round(min_isi / STEP_SEC)),
                                      height=threshold)
    return t[peak_idx], S_m[peak_idx]


def refine_onset_nadir(t: np.ndarray, env: np.ndarray, R: np.ndarray,
                         P: np.ndarray, M: np.ndarray,
                         t_detect: float,
                         search: tuple = ONSET_SEARCH,
                         baseline: tuple = LOCAL_BASELINE) -> float:
    """
    Refine per-event onset to the joint-dip nadir.

    Score = zE_local + zR_local + zP_local + zM_local
    (each locally z-scored against [baseline] relative to t_detect)

    Returns
    -------
    t_onset : float, the nadir time in seconds (absolute).
    """
    rel = t - t_detect
    base_mask = (rel >= baseline[0]) & (rel < baseline[1])
    srch_mask = (rel >= search[0]) & (rel <= search[1])
    if not base_mask.any() or not srch_mask.any():
        return t_detect

    def lz(x):
        if base_mask.sum() < 3:
            return x - np.nanmean(x)
        mu = np.nanmean(x[base_mask])
        sd = np.nanstd(x[base_mask])
        if not np.isfinite(sd) or sd < 1e-9: sd = 1.0
        return (x - mu) / sd

    score = lz(env) + lz(R) + lz(P) + lz(M)
    score_m = np.where(srch_mask, score, np.inf)
    return float(t[int(np.nanargmin(score_m))])


def find_peak(t: np.ndarray, S: np.ndarray, t_onset: float,
               post_sec: float = 3.0) -> float:
    """Peak time of S within [t_onset, t_onset + post_sec]."""
    mask = (t >= t_onset) & (t <= t_onset + post_sec)
    if not mask.any():
        return t_onset
    S_m = np.where(mask, S, -np.inf)
    return float(t[int(np.nanargmax(S_m))])


# -------------------------------------------------------------------------
# Top-level per-recording entry point
# -------------------------------------------------------------------------
@dataclass
class CompositeEvent:
    t_onset: float      # nadir — physiological onset (= new t₀)
    t_detect: float     # S-peak — detection trigger
    t_peak: float       # rebound peak within [t_onset, t_onset + 3]
    S_at_detect: float
    env_at_onset: float
    R_at_onset: float
    PLV_at_onset: float
    MSC_at_onset: float
    env_at_peak: float
    R_at_peak: float
    PLV_at_peak: float
    MSC_at_peak: float


def detect_sie_composite(X_uV: np.ndarray, fs: float,
                           threshold: float = 1.5) -> list[CompositeEvent]:
    """
    Full composite SIE detection + nadir-based onset refinement.

    Parameters
    ----------
    X_uV : (n_channels, n_samples) EEG in µV
    fs : sampling rate (Hz)
    threshold : S-peak threshold for detection (default 1.5)

    Returns
    -------
    list of CompositeEvent
    """
    t, env, R, P, M = compute_streams(X_uV, fs)
    S = composite_S(env, R, P, M)
    t_detects, S_vals = detect_events(t, S, threshold=threshold)

    out: list[CompositeEvent] = []
    for td, sv in zip(t_detects, S_vals):
        t_onset = refine_onset_nadir(t, env, R, P, M, td)
        t_peak = find_peak(t, S, t_onset)

        def val_at(arr, tq):
            idx = int(np.argmin(np.abs(t - tq)))
            return float(arr[idx])

        out.append(CompositeEvent(
            t_onset=t_onset, t_detect=float(td), t_peak=t_peak,
            S_at_detect=float(sv),
            env_at_onset=val_at(env, t_onset), R_at_onset=val_at(R, t_onset),
            PLV_at_onset=val_at(P, t_onset), MSC_at_onset=val_at(M, t_onset),
            env_at_peak=val_at(env, t_peak), R_at_peak=val_at(R, t_peak),
            PLV_at_peak=val_at(P, t_peak), MSC_at_peak=val_at(M, t_peak),
        ))
    return out


# -------------------------------------------------------------------------
# CLI for quick testing on a single LEMON subject
# -------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    from scripts.run_sie_extraction import load_lemon

    p = argparse.ArgumentParser()
    p.add_argument('--subject', default='sub-010249')
    p.add_argument('--threshold', type=float, default=1.5)
    args = p.parse_args()

    raw = load_lemon(args.subject, condition='EC')
    if raw is None:
        print(f'{args.subject}: no data')
        sys.exit(1)
    X = raw.get_data() * 1e6
    fs = raw.info['sfreq']
    events = detect_sie_composite(X, fs, threshold=args.threshold)
    print(f'Subject {args.subject}: {len(events)} events (threshold S≥{args.threshold})')
    df = pd.DataFrame([asdict(e) for e in events])
    if not df.empty:
        df['onset_to_peak'] = df['t_peak'] - df['t_onset']
        df['detect_to_onset'] = df['t_onset'] - df['t_detect']
        print(df[['t_onset', 't_detect', 't_peak',
                  'onset_to_peak', 'detect_to_onset',
                  'R_at_onset', 'R_at_peak',
                  'PLV_at_onset', 'PLV_at_peak']].to_string(index=False))
