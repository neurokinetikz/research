#!/usr/bin/env python3
"""
B19 re-run on composite v2 detector.

Identical analysis to scripts/sie_sr_zoom_peak.py; cohort-parameterized; reads
composite template_rho. Also reports the B19-audit per-subject peak-location
distribution (std, fraction within ±0.1 Hz of 7.83) so we see both the cohort
grand-mean precision AND the per-subject spread without a second script.

Per subject: full-recording sliding Welch at nfft_mult=16 (~0.015 Hz bin) on
6.5-9.0 Hz. Baseline = median PSD across windows. Per-event 4-s window centered
at t0_net + 1.0 s (B14 Q4 peak time). Subject-level Q4 log ratio = log10(event)
− log10(baseline). Cohort log-average; peak frequency + FWHM.

Outputs to outputs/schumann/images/psd_timelapse/<cohort>_composite/.

Usage:
    python scripts/sie_sr_zoom_peak_composite.py --cohort lemon
    python scripts/sie_sr_zoom_peak_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

WIN_SEC = 4.0
HOP_SEC = 1.0
ZOOM_LO, ZOOM_HI = 6.5, 9.0
NFFT_MULT = 16
EVENT_LAG_S = 1.0

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    qual = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, qual
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, qual
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, qual
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, qual
    if cohort == 'srm':
        return load_srm, {}, qual
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, qual
    if cohort == 'dortmund':
        return load_dortmund, {}, qual
    if cohort == 'chbmp':
        return load_chbmp, {}, qual
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, qual
    raise ValueError(f"unsupported cohort {cohort!r}")


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def sliding_psd_all(x, fs, nfft, f_mask, hop_samples, nperseg):
    out = []
    for i in range(0, len(x) - nperseg + 1, hop_samples):
        seg = x[i:i + nperseg]
        psd = welch_one(seg, fs, nfft)
        out.append(psd[f_mask])
    return np.array(out)


_LOADER = None
_LOADER_KW = None


def _init_worker(loader_name, loader_kw):
    global _LOADER, _LOADER_KW
    _LOADER_KW = loader_kw
    _LOADER = {
        'load_lemon': load_lemon,
        'load_tdbrain': load_tdbrain,
        'load_srm': load_srm,
        'load_dortmund': load_dortmund,
        'load_chbmp': load_chbmp,
        'load_hbn_by_subject': load_hbn_by_subject,
    }[loader_name]


def process_subject(args):
    sub_id, df_sub = args
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
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

    psd_all = sliding_psd_all(y, fs, nfft, f_mask, nhop, nperseg)
    if len(psd_all) < 10:
        return None
    psd_baseline = np.nanmedian(psd_all, axis=0)

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
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'psd_timelapse', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    tasks = [(sid, g) for sid, g in qual.groupby('subject_id')]
    print(f"Cohort: {args.cohort} (composite v2)")
    print(f"Subjects: {len(tasks)}  events: {len(qual)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    freqs = results[0]['freqs']
    print(f"Zoom grid: {freqs[0]:.3f}-{freqs[-1]:.3f} Hz, {len(freqs)} bins "
          f"(step {freqs[1]-freqs[0]:.4f} Hz)")

    log_ratios = {'Q1': [], 'Q4': []}
    subject_peaks = {'Q1': [], 'Q4': []}
    for r in results:
        base = r['baseline']
        for q in ['Q1', 'Q4']:
            ev = r[f'{q}_event']
            if ev is None:
                continue
            ratio = np.log10(ev + 1e-20) - np.log10(base + 1e-20)
            log_ratios[q].append(ratio)
            # per-subject peak of the ratio curve
            pk = int(np.nanargmax(ratio))
            subject_peaks[q].append(freqs[pk])

    for q in ['Q1', 'Q4']:
        mat = np.array(log_ratios[q])
        peaks = np.array(subject_peaks[q])
        print(f"\n=== {args.cohort} composite · {q}: n_subjects={len(mat)} ===")
        grand, lo, hi = bootstrap_ci(mat)
        ratio = 10 ** grand
        pk = int(np.nanargmax(ratio))
        w, peak_f = fwhm(ratio, freqs)
        n = len(mat)
        se = float(np.nanstd(peaks, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        within01 = float(np.mean(np.abs(peaks - 7.83) <= 0.10) * 100)
        print(f"  Cohort peak: {ratio[pk]:.2f}× at {peak_f:.3f} Hz (FWHM {w:.2f} Hz)")
        print(f"  Per-subject peak distribution: "
              f"mean {np.mean(peaks):.3f}, std {np.std(peaks, ddof=1):.3f}, "
              f"median {np.median(peaks):.3f} Hz")
        print(f"  Cohort-mean SE (std/sqrt(n)): {se:.3f} Hz")
        print(f"  % subjects within ±0.10 Hz of 7.83: {within01:.1f}%")
        for landmark in [7.60, 7.83, 8.00]:
            j = int(np.argmin(np.abs(freqs - landmark)))
            print(f"  @ {landmark:.2f} Hz: {ratio[j]:.2f}×")

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
    ax.set_title(f'B19 · fine-resolution ignition-time enhancement · {args.cohort} composite v2')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(ZOOM_LO, ZOOM_HI)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sr_zoom_peak.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    np.savez(os.path.join(out_dir, 'sr_zoom_peak.npz'),
             freqs=freqs,
             Q1=np.array(log_ratios['Q1']),
             Q4=np.array(log_ratios['Q4']),
             Q1_subject_peaks=np.array(subject_peaks['Q1']),
             Q4_subject_peaks=np.array(subject_peaks['Q4']))
    print(f"\nSaved: {out_dir}/sr_zoom_peak.png")
    print(f"Saved: {out_dir}/sr_zoom_peak.npz")


if __name__ == '__main__':
    main()
