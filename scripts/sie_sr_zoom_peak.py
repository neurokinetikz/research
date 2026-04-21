#!/usr/bin/env python3
"""
B19 — Fine-resolution peak frequency of the ignition-time enhancement.

Zooms into [6.5, 9.0] Hz at ~0.05 Hz resolution to locate the exact peak
frequency of the ignition-time narrowband enhancement. Tests:

  (a) Does the Q4 event-time boost peak exactly at 7.83 Hz (Schumann
      fundamental), at 7.60 Hz (φ-lattice θ-α boundary), or elsewhere?
  (b) Is the peak sharp (single narrow bump) or smeared (broad shoulder)?
  (c) Does Q1 show a different peak (e.g., at individual alpha frequency)?

Method:
  - Per subject, compute baseline PSD = median across all 4-s windows of the
    recording, in the 6.5-9.0 Hz range at 0.05 Hz resolution.
  - For each Q4 event: a single 4-s Welch PSD centered at t0_net + 1.0 s (the
    Q4 peak time from B14). Average across events for subject. Similarly for Q1.
  - Per subject: ratio = event_PSD / baseline_PSD.
  - Cohort: log-average ratio across subjects.
  - Find peak frequency and FWHM of the cohort ratio curve.
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
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality', 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

WIN_SEC = 4.0
HOP_SEC = 1.0
ZOOM_LO, ZOOM_HI = 6.5, 9.0
NFFT_MULT = 16   # → ~0.015 Hz bin at fs=250 (4-s window)
EVENT_LAG_S = 1.0   # Q4 peak lag from B14


def welch_one(seg, fs, nfft):
    """Single-window PSD on an already-extracted segment."""
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def sliding_psd_all(x, fs, nfft, f_mask, hop_samples, nperseg):
    """Full-recording sliding Welch; return stacked PSDs within f_mask."""
    out = []
    for i in range(0, len(x) - nperseg + 1, hop_samples):
        seg = x[i:i + nperseg]
        psd = welch_one(seg, fs, nfft)
        out.append(psd[f_mask])
    return np.array(out)   # (n_windows, n_f)


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
    nperseg = int(round(WIN_SEC * fs))
    nhop = int(round(HOP_SEC * fs))
    nfft = nperseg * NFFT_MULT
    freqs_full = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_mask = (freqs_full >= ZOOM_LO) & (freqs_full <= ZOOM_HI)
    freqs = freqs_full[f_mask]

    # Baseline PSD: median across all sliding windows
    psd_all = sliding_psd_all(y, fs, nfft, f_mask, nhop, nperseg)
    if len(psd_all) < 10:
        return None
    psd_baseline = np.nanmedian(psd_all, axis=0)

    # Per-quartile event-time PSDs: 4-s window centered at t0_net + 1.0 s
    event_psds = {'Q1': [], 'Q4': []}
    for _, ev in df_sub.iterrows():
        t0 = float(ev['t0_net']) + EVENT_LAG_S
        q = ev['rho_q']
        if q not in event_psds:
            continue
        i0 = int(round((t0 - WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        seg = y[i0:i1]
        psd = welch_one(seg, fs, nfft)[f_mask]
        event_psds[q].append(psd)

    out = {'subject_id': sub_id, 'freqs': freqs, 'baseline': psd_baseline}
    for q in ['Q1', 'Q4']:
        arr = event_psds[q]
        if arr:
            out[f'{q}_event'] = np.nanmean(np.array(arr), axis=0)
            out[f'{q}_n'] = len(arr)
        else:
            out[f'{q}_event'] = None
            out[f'{q}_n'] = 0
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


def fwhm(y, x):
    pk = int(np.nanargmax(y))
    half = y[pk] / 2
    L = pk
    while L > 0 and y[L] > half:
        L -= 1
    R = pk
    while R < len(y) - 1 and y[R] > half:
        R += 1
    return x[R] - x[L], x[pk]


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

    freqs = results[0]['freqs']
    print(f"Zoom grid: {freqs[0]:.3f}-{freqs[-1]:.3f} Hz, {len(freqs)} bins "
          f"(step {freqs[1]-freqs[0]:.4f} Hz)")

    # Per-subject log ratios
    log_ratios = {'Q1': [], 'Q4': []}
    for r in results:
        base = r['baseline']
        for q in ['Q1', 'Q4']:
            ev = r[f'{q}_event']
            if ev is None:
                continue
            ratio = np.log10(ev + 1e-20) - np.log10(base + 1e-20)
            log_ratios[q].append(ratio)

    for q in ['Q1', 'Q4']:
        mat = np.array(log_ratios[q])
        print(f"\n=== {q}: n_subjects={len(mat)} ===")
        grand, lo, hi = bootstrap_ci(mat)
        ratio = 10 ** grand
        pk = int(np.nanargmax(ratio))
        w, peak_f = fwhm(ratio, freqs)
        print(f"  Peak ratio: {ratio[pk]:.2f}× at {peak_f:.3f} Hz")
        print(f"  FWHM: {w:.2f} Hz")
        # specific landmarks
        for landmark in [7.60, 7.83, 8.00]:
            j = int(np.argmin(np.abs(freqs - landmark)))
            print(f"  @ {landmark:.2f} Hz: {ratio[j]:.2f}×")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for q, color in [('Q1', '#4575b4'), ('Q4', '#d73027')]:
        mat = np.array(log_ratios[q])
        if len(mat) == 0:
            continue
        grand, lo, hi = bootstrap_ci(mat)
        ratio = 10 ** grand
        lo_r = 10 ** lo; hi_r = 10 ** hi
        pk = int(np.nanargmax(ratio))
        ax.plot(freqs, ratio, color=color, lw=2,
                label=f'{q} peak {ratio[pk]:.2f}× @ {freqs[pk]:.2f} Hz (n={len(mat)})')
        ax.fill_between(freqs, lo_r, hi_r, color=color, alpha=0.22)
    ax.axvline(7.60, color='k', ls='--', lw=0.7, label='φ-lattice θ-α boundary 7.60 Hz')
    ax.axvline(7.83, color='green', ls=':', lw=0.7, label='Schumann f₀ = 7.83 Hz')
    ax.axhline(1.0, color='k', lw=0.6)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('event-time PSD / baseline PSD (×)')
    ax.set_title('B19 — Fine-resolution ignition-time spectral enhancement (Q1 vs Q4)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(ZOOM_LO, ZOOM_HI)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sr_zoom_peak.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/sr_zoom_peak.png")


if __name__ == '__main__':
    main()
