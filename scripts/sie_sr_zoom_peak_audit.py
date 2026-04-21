#!/usr/bin/env python3
"""
B19-audit — Audit of B19's 7.828 Hz peak-location claim.

Checks:
  1. Native-resolution reality check: peak location with nfft_mult=1 (no zero-
     padding). Should match nfft_mult=16 within the native 0.25 Hz bin.
  2. Per-subject peak-location distribution. Is the cohort peak a robust common
     peak, or a smear with a coincidental average near 7.83 Hz?
  3. Peak stability across event lag (t0+0.5, +1.0, +1.5, +2.0 s).
  4. Ratio vs log-ratio vs event-PSD alone: does the baseline normalization
     shift the peak location?
  5. Single-event peak distribution (instead of subject-averaged peaks).
  6. Also report SE of cohort peak as std / sqrt(n).

Uses Q4 events only; Q1 is control.
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
EVENT_LAGS = [0.5, 1.0, 1.5, 2.0]


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def process_subject_audit(args):
    sub_id, df_sub, nfft_mult = args
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
    nfft = nperseg * nfft_mult
    freqs_full = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_mask = (freqs_full >= ZOOM_LO) & (freqs_full <= ZOOM_HI)
    freqs = freqs_full[f_mask]

    # Baseline: median across all 4-s sliding windows
    psd_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)[f_mask]
        psd_rows.append(psd)
    psd_baseline = np.nanmedian(np.array(psd_rows), axis=0)

    # Event-time PSDs per lag, Q4 only
    event_data = {lag: {'events': [], 'psds': []} for lag in EVENT_LAGS}
    for _, ev in df_sub.iterrows():
        t0 = float(ev['t0_net'])
        if ev['rho_q'] != 'Q4':
            continue
        for lag in EVENT_LAGS:
            tc = t0 + lag
            i0 = int(round((tc - WIN_SEC / 2) * fs))
            i1 = i0 + nperseg
            if i0 < 0 or i1 > len(y):
                continue
            seg = y[i0:i1]
            psd = welch_one(seg, fs, nfft)[f_mask]
            event_data[lag]['events'].append(ev['t0_net'])
            event_data[lag]['psds'].append(psd)

    out = {'subject_id': sub_id, 'freqs': freqs, 'baseline': psd_baseline}
    for lag in EVENT_LAGS:
        arr = event_data[lag]['psds']
        out[f'lag{lag}_event_psd'] = np.nanmean(np.array(arr), axis=0) if arr else None
        out[f'lag{lag}_event_psds_all'] = np.array(arr) if arr else None
        out[f'lag{lag}_n'] = len(arr)
    return out


def find_peak(ratio, freqs):
    if np.all(np.isnan(ratio)):
        return np.nan
    return float(freqs[int(np.nanargmax(ratio))])


def main():
    qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    tasks_mult_list = [1, 4, 16]

    print("=" * 70)
    print("AUDIT 1: Peak location vs zero-padding (native vs interpolated)")
    print("=" * 70)

    for nfft_mult in tasks_mult_list:
        tasks = [(sid, g, nfft_mult) for sid, g in qual.groupby('subject_id')]
        n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
        with Pool(n_workers) as pool:
            results = pool.map(process_subject_audit, tasks)
        results = [r for r in results if r is not None]
        if not results:
            continue
        freqs = results[0]['freqs']
        bin_width = freqs[1] - freqs[0]

        # Cohort ratio at lag 1.0
        log_ratios = []
        for r in results:
            base = r['baseline']; ev = r['lag1.0_event_psd']
            if ev is None: continue
            log_ratios.append(np.log10(ev + 1e-20) - np.log10(base + 1e-20))
        grand = np.nanmean(np.array(log_ratios), axis=0)
        ratio = 10 ** grand
        pk_f = find_peak(ratio, freqs)
        pk_r = ratio[int(np.nanargmax(ratio))]
        print(f"  nfft_mult={nfft_mult}  bin_width={bin_width:.4f} Hz  "
              f"cohort peak at {pk_f:.4f} Hz  ratio={pk_r:.2f}×")

    print()
    print("=" * 70)
    print("AUDIT 2: Per-subject peak distribution (nfft_mult=16)")
    print("=" * 70)

    # Reuse the nfft_mult=16 results from above (last iteration)
    # Actually we need to re-fetch them; simpler to redo:
    nfft_mult = 16
    tasks = [(sid, g, nfft_mult) for sid, g in qual.groupby('subject_id')]
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject_audit, tasks)
    results = [r for r in results if r is not None]
    freqs = results[0]['freqs']

    # Per-subject peak location
    per_sub_peaks = []
    for r in results:
        base = r['baseline']; ev = r['lag1.0_event_psd']
        if ev is None: continue
        ratio = (ev + 1e-20) / (base + 1e-20)
        pk_f = find_peak(ratio, freqs)
        if np.isfinite(pk_f):
            per_sub_peaks.append(pk_f)
    pks = np.array(per_sub_peaks)
    print(f"  n_subjects: {len(pks)}")
    print(f"  Mean: {np.mean(pks):.3f} Hz   Median: {np.median(pks):.3f} Hz")
    print(f"  Std: {np.std(pks):.3f} Hz   IQR: [{np.percentile(pks, 25):.3f}, {np.percentile(pks, 75):.3f}]")
    print(f"  SE of mean: {np.std(pks) / np.sqrt(len(pks)):.4f} Hz")
    print(f"  95% range: [{np.percentile(pks, 2.5):.3f}, {np.percentile(pks, 97.5):.3f}] Hz")
    print(f"  Fraction within ±0.1 Hz of 7.83: {np.mean(np.abs(pks - 7.83) <= 0.1)*100:.0f}%")
    print(f"  Fraction within ±0.25 Hz of 7.83: {np.mean(np.abs(pks - 7.83) <= 0.25)*100:.0f}%")

    print()
    print("=" * 70)
    print("AUDIT 3: Peak stability across event-lag (nfft_mult=16)")
    print("=" * 70)

    for lag in EVENT_LAGS:
        log_ratios = []
        for r in results:
            base = r['baseline']; ev = r[f'lag{lag}_event_psd']
            if ev is None: continue
            log_ratios.append(np.log10(ev + 1e-20) - np.log10(base + 1e-20))
        grand = np.nanmean(np.array(log_ratios), axis=0)
        ratio = 10 ** grand
        pk_f = find_peak(ratio, freqs)
        pk_r = ratio[int(np.nanargmax(ratio))]
        print(f"  lag={lag:.1f}s  peak {pk_r:.2f}× @ {pk_f:.4f} Hz")

    print()
    print("=" * 70)
    print("AUDIT 4: Baseline-free event PSD (no normalization)")
    print("=" * 70)

    event_rows = []
    for r in results:
        ev = r['lag1.0_event_psd']
        if ev is None: continue
        event_rows.append(ev)
    ev_grand = np.nanmean(np.array(event_rows), axis=0)
    pk_f = find_peak(ev_grand, freqs)
    pk_p = ev_grand[int(np.nanargmax(ev_grand))]
    print(f"  Event PSD (no baseline norm): peak {pk_p:.3g} at {pk_f:.4f} Hz")
    base_rows = []
    for r in results:
        base_rows.append(r['baseline'])
    base_grand = np.nanmean(np.array(base_rows), axis=0)
    pk_f_b = find_peak(base_grand, freqs)
    print(f"  Baseline PSD: peak {base_grand[int(np.nanargmax(base_grand))]:.3g} at {pk_f_b:.4f} Hz")
    print(f"  → Baseline peak shift vs event peak: {pk_f - pk_f_b:+.4f} Hz")

    print()
    print("=" * 70)
    print("AUDIT 5: Single-event peak locations (not subject-averaged)")
    print("=" * 70)

    single_peaks = []
    for r in results:
        arr = r['lag1.0_event_psds_all']
        base = r['baseline']
        if arr is None or arr.shape[0] == 0: continue
        for i in range(arr.shape[0]):
            ratio = (arr[i] + 1e-20) / (base + 1e-20)
            pf = find_peak(ratio, freqs)
            if np.isfinite(pf):
                single_peaks.append(pf)
    sp = np.array(single_peaks)
    print(f"  n_events: {len(sp)}")
    print(f"  Mean: {np.mean(sp):.3f} Hz   Median: {np.median(sp):.3f} Hz")
    print(f"  Std: {np.std(sp):.3f} Hz   IQR: [{np.percentile(sp, 25):.3f}, {np.percentile(sp, 75):.3f}]")
    print(f"  SE of mean: {np.std(sp) / np.sqrt(len(sp)):.4f} Hz")

    # Plot histograms of per-subject and single-event peak locations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(pks, bins=np.linspace(6.5, 9.0, 40), color='firebrick',
             edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(7.83, color='k', ls='--', lw=1.2, label='Schumann 7.83 Hz')
    ax.axvline(7.60, color='gray', ls=':', lw=1, label='φ-boundary 7.60 Hz')
    ax.axvline(np.mean(pks), color='blue', ls='-', lw=1.5,
                label=f'Cohort mean {np.mean(pks):.3f} Hz')
    ax.set_xlabel('per-subject Q4 peak freq (Hz)')
    ax.set_ylabel('subjects')
    ax.set_title(f'Per-subject peaks (n={len(pks)})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(sp, bins=np.linspace(6.5, 9.0, 40), color='seagreen',
             edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(7.83, color='k', ls='--', lw=1.2, label='Schumann 7.83 Hz')
    ax.axvline(7.60, color='gray', ls=':', lw=1, label='φ-boundary 7.60 Hz')
    ax.axvline(np.mean(sp), color='blue', ls='-', lw=1.5,
                label=f'Single-event mean {np.mean(sp):.3f} Hz')
    ax.set_xlabel('single-event Q4 peak freq (Hz)')
    ax.set_ylabel('events')
    ax.set_title(f'Per-event peaks (n={len(sp)})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle('B19-audit — peak-location distributions', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sr_zoom_peak_audit.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/sr_zoom_peak_audit.png")


if __name__ == '__main__':
    main()
