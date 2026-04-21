#!/usr/bin/env python3
"""
B24 — Subject-level spectral differentiation vs ignition-peak proximity.

Bridges the SIE work with the spectral-differentiation paper. For each LEMON
subject:

  1. Aggregate PSD (full-recording 8-s Welch, 0.125 Hz native) → full spectrum.
  2. Detect oscillatory peaks in [2, 20] Hz after 1/f removal (log-linear fit
     on 2-5 ∪ 9-20 Hz, subtract fit).
  3. Per-subject features:
     - IAF (argmax in [7, 13] on 1/f-corrected spectrum)
     - Alpha peak sharpness (1 / FWHM around alpha peak)
     - Presence of a dedicated peak in the SR sub-range [7.5, 8.2] Hz, with
       its height relative to local background
     - Theta peak presence in [4.7, 7.6]
  4. Ignition-side per-subject features (reuse B22-style per-event peaks):
     - Mean event-peak frequency, SD, and mean |peak − 7.83|
     - Mean event-boost ratio

Correlate across subjects:
  - Alpha-peak sharpness × ignition-proximity (mean |peak − 7.83|)
  - Standing SR peak presence × ignition-proximity
  - Alpha peak freq (IAF) × mean ignition peak (sanity — already done in B20)

Tests whether subjects with better-organized (more sharply differentiated)
aggregate spectra also produce cleaner ignition attractor hits.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, mannwhitneyu
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

# Aggregate PSD params
AGG_WIN_SEC = 8.0
AGG_HOP_SEC = 2.0
AGG_NFFT_MULT = 4
AGG_LO, AGG_HI = 2.0, 20.0
APERIODIC_RANGES = [(2.0, 5.0), (9.0, 20.0)]

# Event PSD params
EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
EV_LO, EV_HI = 6.5, 9.0

# Band landmarks
SR_RANGE = (7.5, 8.2)       # narrow SR-peak check
ALPHA_RANGE = (7.6, 12.3)   # φ-lattice alpha band
THETA_RANGE = (4.7, 7.6)    # φ-lattice theta band
SCHUMANN_F = 7.83


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def parabolic_peak(y, x):
    k = int(np.argmax(y))
    if 1 <= k < len(y) - 1 and y[k-1] > 0 and y[k+1] > 0:
        y0, y1, y2 = y[k-1], y[k], y[k+1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = max(-1.0, min(1.0, delta))
        return float(x[k] + delta * (x[1] - x[0]))
    return float(x[k])


def fwhm_around(y, x, pk_idx):
    half = y[pk_idx] / 2
    L = pk_idx
    while L > 0 and y[L] > half:
        L -= 1
    R = pk_idx
    while R < len(y) - 1 and y[R] > half:
        R += 1
    return float(x[R] - x[L])


def aperiodic_subtract(freqs, psd, ranges=APERIODIC_RANGES):
    """Return psd − 1/f aperiodic estimate (in log space)."""
    mask = np.zeros_like(freqs, dtype=bool)
    for lo, hi in ranges:
        mask |= (freqs >= lo) & (freqs <= hi)
    logf = np.log10(freqs)
    logp = np.log10(psd + 1e-20)
    good = mask & np.isfinite(logp) & (logp > -10)
    if good.sum() < 8:
        return psd.copy()
    A = np.column_stack([logf[good], np.ones(good.sum())])
    coefs, *_ = np.linalg.lstsq(A, logp[good], rcond=None)
    a, b = float(coefs[0]), float(coefs[1])
    aperiodic_log = a * logf + b
    # Return linear-scale residual (peaks above 1/f)
    return psd - 10 ** aperiodic_log


def subject_aggregate_psd(y, fs):
    nperseg = int(round(AGG_WIN_SEC * fs))
    nhop = int(round(AGG_HOP_SEC * fs))
    nfft = nperseg * AGG_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= AGG_LO) & (freqs <= AGG_HI)
    psds = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)[mask]
        psds.append(psd)
    if len(psds) < 5:
        return None, None
    agg = np.nanmedian(np.array(psds), axis=0)
    return freqs[mask], agg


def event_peak(y, fs, t0_net):
    nperseg = int(round(EV_WIN_SEC * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= EV_LO) & (freqs <= EV_HI)
    f_band = freqs[mask]
    nhop = int(round(1.0 * fs))

    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        base_rows.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if len(base_rows) < 10:
        return np.nan
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    tc = t0_net + EV_LAG_S
    i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
    i1 = i0 + nperseg
    if i0 < 0 or i1 > len(y):
        return np.nan
    psd = welch_one(y[i0:i1], fs, nfft)[mask]
    ratio = (psd + 1e-20) / (baseline + 1e-20)
    return parabolic_peak(ratio, f_band)


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 3:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    freqs, agg = subject_aggregate_psd(y, fs)
    if freqs is None:
        return None
    residual = aperiodic_subtract(freqs, agg)

    # Alpha peak on 1/f-corrected
    alpha_mask = (freqs >= ALPHA_RANGE[0]) & (freqs <= ALPHA_RANGE[1])
    alpha_idx = np.where(alpha_mask)[0]
    if not np.any(residual[alpha_mask] > 0):
        return None
    k_a = int(alpha_idx[np.argmax(residual[alpha_idx])])
    alpha_peak_f = parabolic_peak(residual[alpha_idx], freqs[alpha_idx])
    alpha_peak_pow = float(residual[k_a])
    alpha_fwhm = fwhm_around(residual, freqs, k_a)
    alpha_sharpness = 1.0 / max(alpha_fwhm, 1e-3)

    # SR peak presence in [7.5, 8.2]
    sr_mask = (freqs >= SR_RANGE[0]) & (freqs <= SR_RANGE[1])
    sr_residual = residual[sr_mask]
    sr_present = bool(np.any(sr_residual > 0) and
                       np.max(sr_residual) / max(alpha_peak_pow, 1e-6) > 0.1)
    sr_peak_f = (parabolic_peak(sr_residual, freqs[sr_mask])
                  if np.any(sr_residual > 0) else np.nan)
    sr_peak_ratio_to_alpha = float(np.max(sr_residual) / max(alpha_peak_pow, 1e-6))

    # Theta peak in [4.7, 7.6]
    theta_mask = (freqs >= THETA_RANGE[0]) & (freqs <= THETA_RANGE[1])
    theta_residual = residual[theta_mask]
    theta_peak_f = (parabolic_peak(theta_residual, freqs[theta_mask])
                     if np.any(theta_residual > 0) else np.nan)
    theta_peak_ratio = float(np.max(theta_residual) / max(alpha_peak_pow, 1e-6)) \
        if np.any(theta_residual > 0) else 0.0

    # Event peaks
    peaks = []
    for _, ev in events.iterrows():
        p = event_peak(y, fs, float(ev['t0_net']))
        if np.isfinite(p):
            peaks.append(p)
    if len(peaks) < 3:
        return None
    peaks = np.array(peaks)
    return {
        'subject_id': sub_id,
        'n_events': int(len(peaks)),
        # Spectral diff features
        'alpha_peak_hz': alpha_peak_f,
        'alpha_peak_pow': alpha_peak_pow,
        'alpha_fwhm_hz': alpha_fwhm,
        'alpha_sharpness': alpha_sharpness,
        'sr_peak_hz': sr_peak_f,
        'sr_peak_pow_rel_alpha': sr_peak_ratio_to_alpha,
        'sr_peak_present': sr_present,
        'theta_peak_hz': theta_peak_f,
        'theta_peak_pow_rel_alpha': theta_peak_ratio,
        # Ignition features
        'event_peak_mean': float(np.mean(peaks)),
        'event_peak_median': float(np.median(peaks)),
        'event_peak_sd': float(np.std(peaks, ddof=1)),
        'event_peak_dist_mean': float(np.mean(np.abs(peaks - SCHUMANN_F))),
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects ≥3 events: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        out = pool.map(process_subject, tasks)
    df = pd.DataFrame([r for r in out if r is not None])
    df.to_csv(os.path.join(OUT_DIR, 'subject_spectral_diff_vs_ignition.csv'),
               index=False)
    print(f"Subjects with complete features: {len(df)}")

    print(f"\n=== Aggregate features summary ===")
    for c in ['alpha_peak_hz', 'alpha_sharpness', 'sr_peak_pow_rel_alpha',
              'theta_peak_pow_rel_alpha']:
        print(f"  {c:30s} median {df[c].median():.3f}  IQR [{df[c].quantile(.25):.3f}, {df[c].quantile(.75):.3f}]")
    pct_sr_present = df['sr_peak_present'].mean() * 100
    print(f"  subjects with standing SR peak (pow > 10% alpha): {pct_sr_present:.0f}%")

    print(f"\n=== Ignition features summary ===")
    for c in ['event_peak_mean', 'event_peak_sd', 'event_peak_dist_mean']:
        print(f"  {c:30s} median {df[c].median():.3f}  IQR [{df[c].quantile(.25):.3f}, {df[c].quantile(.75):.3f}]")

    # Correlations
    print(f"\n=== Subject-level Spearman correlations ===")
    pairs = [
        ('alpha_peak_hz', 'event_peak_mean'),
        ('alpha_peak_hz', 'event_peak_dist_mean'),
        ('alpha_sharpness', 'event_peak_mean'),
        ('alpha_sharpness', 'event_peak_dist_mean'),
        ('alpha_sharpness', 'event_peak_sd'),
        ('sr_peak_pow_rel_alpha', 'event_peak_dist_mean'),
        ('sr_peak_pow_rel_alpha', 'event_peak_mean'),
        ('theta_peak_pow_rel_alpha', 'event_peak_dist_mean'),
    ]
    for a, b in pairs:
        sub = df.dropna(subset=[a, b])
        if len(sub) < 10:
            continue
        rho, p = spearmanr(sub[a], sub[b])
        print(f"  {a:30s} × {b:25s}  ρ={rho:+.3f}  p={p:.3g}  n={len(sub)}")

    # SR-peak-present vs SR-peak-absent
    present = df[df['sr_peak_present']]
    absent = df[~df['sr_peak_present']]
    if len(present) >= 5 and len(absent) >= 5:
        print(f"\n=== SR-peak-present vs absent (n_present={len(present)}, "
              f"n_absent={len(absent)}) ===")
        for c in ['event_peak_mean', 'event_peak_dist_mean', 'event_peak_sd']:
            u, p = mannwhitneyu(present[c], absent[c])
            print(f"  {c:30s}: present {present[c].median():.3f}  absent {absent[c].median():.3f}  "
                  f"MWU p={p:.3g}")

    # Plot: scatter of alpha_sharpness vs event_peak_dist_mean
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, xcol, xlab in [
        (axes[0], 'alpha_sharpness', '1/FWHM alpha peak'),
        (axes[1], 'sr_peak_pow_rel_alpha', 'SR peak / alpha peak (aperiodic-corrected)'),
        (axes[2], 'alpha_peak_hz', 'alpha peak freq (IAF)'),
    ]:
        sub = df.dropna(subset=[xcol, 'event_peak_dist_mean'])
        rho, p = spearmanr(sub[xcol], sub['event_peak_dist_mean'])
        ax.scatter(sub[xcol], sub['event_peak_dist_mean'], s=20, alpha=0.5,
                    color='steelblue', edgecolor='k', lw=0.3)
        ax.set_xlabel(xlab)
        ax.set_ylabel('mean |event_peak − 7.83| (Hz)')
        ax.set_title(f'{xcol} vs ignition-proximity · ρ={rho:+.2f} p={p:.2g}')
        ax.grid(alpha=0.3)

    plt.suptitle('B24 — Subject-level spectral differentiation × ignition-peak proximity',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'subject_spectral_diff_vs_ignition.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/subject_spectral_diff_vs_ignition.png")


if __name__ == '__main__':
    main()
