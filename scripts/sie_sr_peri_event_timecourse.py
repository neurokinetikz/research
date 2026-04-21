#!/usr/bin/env python3
"""
B14 — Peri-event time course of the SR-band boost.

Does the 7.0-8.2 Hz narrowband enhancement ramp up pre-event, peak at t0_net,
and decay? Or switch on abruptly? Does the time course differ Q1 vs Q4?

For each subject:
  1. Full-recording sliding Welch (4-s window, 1-s hop) on mean-channel signal
  2. Per-window SR-band peak power (parabolic-refined argmax in [7.0, 8.2] Hz)
  3. Per-window 1/f-normalized excess (log10 peak power − log10 aperiodic fit
     at peak freq; aperiodic fit on [2, 5] ∪ [9, 20] Hz)
  4. All-window median → baseline; log-boost at time t = log10(peak_p(t)) − baseline
  5. For each event: interpolate raw log-boost and 1/f-normalized excess onto
     [-20, +20] s grid at 1-s step, rel to t0_net

Then: subject-level average within each template_rho quartile, cohort grand
mean with cluster bootstrap 95% CI, plotted Q1 vs Q4.

Tests: does the SR-band enhancement ramp pre-event or switch on abruptly,
and does the time course vary with event quality?
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality', 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

WIN_SEC = 4.0
HOP_SEC = 1.0
FREQ_LO, FREQ_HI = 2.0, 20.0
APERIODIC_RANGES = [(2.0, 5.0), (9.0, 20.0)]
SR_LO, SR_HI = 7.0, 8.2
NFFT_MULT = 4

TGRID = np.arange(-20.0, 20.0 + 0.5, 1.0)


def sliding_welch(x, fs):
    nperseg = int(round(WIN_SEC * fs))
    nhop = int(round(HOP_SEC * fs))
    nfft = nperseg * NFFT_MULT
    freqs_full = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_mask = (freqs_full >= FREQ_LO) & (freqs_full <= FREQ_HI)
    freqs = freqs_full[f_mask]
    win = signal.windows.hann(nperseg)
    win_pow = np.sum(win ** 2)
    t_cent, cols = [], []
    for i in range(0, len(x) - nperseg + 1, nhop):
        seg = x[i:i + nperseg] - np.mean(x[i:i + nperseg])
        X = np.fft.rfft(seg * win, nfft)
        psd = (np.abs(X) ** 2) / (fs * win_pow)
        psd[1:-1] *= 2.0
        cols.append(psd[f_mask])
        t_cent.append((i + nperseg / 2) / fs)
    return np.array(t_cent), freqs, np.array(cols).T


def aperiodic_mask(freqs, ranges):
    m = np.zeros_like(freqs, dtype=bool)
    for lo, hi in ranges:
        m |= (freqs >= lo) & (freqs <= hi)
    return m


def per_window_sr(freqs, P):
    """Return peak_f, peak_p, log_excess (log10 peak / aperiodic)."""
    ap_m = aperiodic_mask(freqs, APERIODIC_RANGES)
    sr_m = (freqs >= SR_LO) & (freqs <= SR_HI)
    idx_sr = np.where(sr_m)[0]
    f_sr = freqs[idx_sr]
    logf = np.log10(freqs)
    logf_ap = logf[ap_m]
    n_t = P.shape[1]
    peak_f = np.full(n_t, np.nan)
    peak_p = np.full(n_t, np.nan)
    excess = np.full(n_t, np.nan)
    for j in range(n_t):
        col = P[:, j]
        col_sr = col[idx_sr]
        if not np.isfinite(col_sr).any() or np.all(col_sr <= 0):
            continue
        k = int(np.argmax(col_sr))
        if 1 <= k < len(col_sr) - 1 and col_sr[k-1] > 0 and col_sr[k+1] > 0:
            y0, y1, y2 = col_sr[k-1], col_sr[k], col_sr[k+1]
            denom = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            delta = max(-1.0, min(1.0, delta))
            df = f_sr[1] - f_sr[0]
            f_k = f_sr[k] + delta * df
        else:
            f_k = f_sr[k]
        peak_f[j] = f_k
        peak_p[j] = col_sr[k]
        logp_ap = np.log10(col[ap_m] + 1e-20)
        good = np.isfinite(logp_ap) & (logp_ap > -10)
        if good.sum() < 8:
            continue
        A = np.column_stack([logf_ap[good], np.ones(good.sum())])
        try:
            coefs, *_ = np.linalg.lstsq(A, logp_ap[good], rcond=None)
            a, b = float(coefs[0]), float(coefs[1])
            excess[j] = np.log10(col_sr[k] + 1e-20) - (a * np.log10(f_k) + b)
        except Exception:
            pass
    return peak_f, peak_p, excess


def process_subject(args):
    sub_id, df_sub = args
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    t, freqs, P = sliding_welch(y, fs)
    peak_f, peak_p, excess = per_window_sr(freqs, P)

    # Baselines
    med_logp = float(np.nanmedian(np.log10(peak_p + 1e-20)))
    med_excess = float(np.nanmedian(excess))

    # Per-window log-boost (raw) and normalized excess relative to baseline
    logp_series = np.log10(peak_p + 1e-20) - med_logp
    excess_series = excess - med_excess

    # Interpolate per-event
    buckets = {q: {'logp': [], 'excess': []} for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    for _, ev in df_sub.iterrows():
        te = float(ev['t0_net'])
        q = ev['rho_q']
        rel = t - te
        mask = (rel >= TGRID[0] - 2) & (rel <= TGRID[-1] + 2)
        if mask.sum() < len(TGRID) * 0.5:
            continue
        lp_i = np.interp(TGRID, rel[mask], logp_series[mask], left=np.nan, right=np.nan)
        ex_i = np.interp(TGRID, rel[mask], excess_series[mask], left=np.nan, right=np.nan)
        buckets[q]['logp'].append(lp_i)
        buckets[q]['excess'].append(ex_i)

    out = {'subject_id': sub_id}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        for key in ['logp', 'excess']:
            arr = buckets[q][key]
            out[f'{q}_{key}'] = np.nanmean(np.array(arr), axis=0) if arr else None
            out[f'{q}_{key}_n'] = len(arr)
    return out


def bootstrap_ci(mat, n_boot=1000, seed=0):
    mat = np.array(mat)
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    if len(mat) < 2:
        return (np.nanmean(mat, axis=0),
                np.full(mat.shape[1], np.nan),
                np.full(mat.shape[1], np.nan))
    rng = np.random.default_rng(seed)
    grand = np.nanmean(mat, axis=0)
    boots = np.zeros((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, len(mat), size=len(mat))
        boots[b] = np.nanmean(mat[idx], axis=0)
    return grand, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def main():
    qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    tasks = [(sid, g) for sid, g in qual.groupby('subject_id')]
    print(f"Subjects: {len(tasks)}  events: {len(qual)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    def stack(q, key):
        arr = [r[f'{q}_{key}'] for r in results if r[f'{q}_{key}'] is not None]
        return np.array(arr) if arr else np.empty((0, len(TGRID)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    # raw log-boost (convert to ratio for display: 10 ** logp)
    ax = axes[0]
    for q, color in [('Q1', '#4575b4'), ('Q4', '#d73027')]:
        mat = stack(q, 'logp')
        grand, lo, hi = bootstrap_ci(mat)
        # convert to ratio
        ratio = 10 ** grand
        lo_r = 10 ** lo; hi_r = 10 ** hi
        n_sub = len(mat); n_ev = sum(r[f'{q}_logp_n'] for r in results)
        ax.plot(TGRID, ratio, color=color, lw=2,
                label=f'{q} n_sub={n_sub} n_ev={n_ev}')
        ax.fill_between(TGRID, lo_r, hi_r, color=color, alpha=0.22)
    ax.axhline(1.0, color='k', lw=0.7)
    ax.axvline(0, color='k', ls='--', lw=0.6)
    ax.set_xlabel('time rel. t0_net (s)')
    ax.set_ylabel('SR-band peak power / subject baseline (×)')
    ax.set_title('Raw SR-band peak power (×) vs time')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    # 1/f-normalized excess (convert to ratio)
    ax = axes[1]
    for q, color in [('Q1', '#4575b4'), ('Q4', '#d73027')]:
        mat = stack(q, 'excess')
        grand, lo, hi = bootstrap_ci(mat)
        ratio = 10 ** grand
        lo_r = 10 ** lo; hi_r = 10 ** hi
        n_sub = len(mat)
        ax.plot(TGRID, ratio, color=color, lw=2,
                label=f'{q} n_sub={n_sub}')
        ax.fill_between(TGRID, lo_r, hi_r, color=color, alpha=0.22)
    ax.axhline(1.0, color='k', lw=0.7)
    ax.axvline(0, color='k', ls='--', lw=0.6)
    ax.set_xlabel('time rel. t0_net (s)')
    ax.set_ylabel('1/f-normalized SR excess (×)')
    ax.set_title('1/f-normalized SR-band excess (×) vs time')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    plt.suptitle('B14 — Peri-event time course of SR-band [7.0, 8.2 Hz] boost',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sr_band_peri_event_timecourse.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # Print key peri-event numbers
    print("\n=== Peri-event time course, Q1 vs Q4 (peak ratio & timing) ===")
    for q in ['Q1', 'Q4']:
        mat = stack(q, 'logp')
        grand, _, _ = bootstrap_ci(mat)
        ratio = 10 ** grand
        if len(ratio) == 0 or np.all(np.isnan(ratio)):
            continue
        # peak latency
        pk = int(np.nanargmax(ratio))
        print(f"  {q} raw peak {ratio[pk]:.2f}× at t = {TGRID[pk]:+.1f} s  "
              f"(t=-10 baseline {ratio[np.argmin(np.abs(TGRID+10))]:.2f}×)")
    print()
    for q in ['Q1', 'Q4']:
        mat = stack(q, 'excess')
        grand, _, _ = bootstrap_ci(mat)
        ratio = 10 ** grand
        if len(ratio) == 0 or np.all(np.isnan(ratio)):
            continue
        pk = int(np.nanargmax(ratio))
        print(f"  {q} 1/f-norm peak {ratio[pk]:.2f}× at t = {TGRID[pk]:+.1f} s  "
              f"(t=-10 baseline {ratio[np.argmin(np.abs(TGRID+10))]:.2f}×)")

    print(f"\nSaved: {OUT_DIR}/sr_band_peri_event_timecourse.png")


if __name__ == '__main__':
    main()
