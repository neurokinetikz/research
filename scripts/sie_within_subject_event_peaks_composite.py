#!/usr/bin/env python3
"""
B22 re-run on composite v2 detector.

Identical analysis to scripts/sie_within_subject_event_peaks.py; cohort-
parameterized. Reads composite v2 events from exports_sie/<cohort>_composite/.

Per subject (≥3 composite events):
  1. Baseline PSD from full-recording sliding Welch in [6.5, 9.0] Hz
  2. Per-event peak freq = argmax (4-s event PSD at t0+1.0s) / baseline
  3. Within-subject SD, IQR of event peaks
Cohort-level:
  within_var = mean(SD²), between_var = var(means)
  trait ICC = between_var / (between_var + within_var)

Outputs to outputs/schumann/images/psd_timelapse/<cohort>_composite/.

Usage:
    python scripts/sie_within_subject_event_peaks_composite.py --cohort lemon
    python scripts/sie_within_subject_event_peaks_composite.py --cohort lemon_EO
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

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
ZOOM_LO, ZOOM_HI = 6.5, 9.0

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events
    if cohort == 'srm':
        return load_srm, {}, events
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events
    if cohort == 'dortmund':
        return load_dortmund, {}, events
    if cohort == 'chbmp':
        return load_chbmp, {}, events
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events
    raise ValueError(f"unsupported cohort {cohort!r}")


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
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 3:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= ZOOM_LO) & (freqs <= ZOOM_HI)
    f_band = freqs[mask]

    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)[mask]
        base_rows.append(psd)
    if len(base_rows) < 10:
        return None
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    peaks = []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((t0 - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[mask]
        ratio = (psd + 1e-20) / (baseline + 1e-20)
        peaks.append(parabolic_peak(ratio, f_band))
    peaks = np.array(peaks)
    if len(peaks) < 3:
        return None
    return {
        'subject_id': sub_id,
        'n_events': int(len(peaks)),
        'peak_mean': float(np.mean(peaks)),
        'peak_median': float(np.median(peaks)),
        'peak_sd': float(np.std(peaks, ddof=1)),
        'peak_iqr': float(np.percentile(peaks, 75) - np.percentile(peaks, 25)),
        'peaks': peaks.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'psd_timelapse', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} composite  ·  subjects with >=3 events: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    sds = np.array([r['peak_sd'] for r in results])
    means = np.array([r['peak_mean'] for r in results])
    iqrs = np.array([r['peak_iqr'] for r in results])
    ns = np.array([r['n_events'] for r in results])

    print(f"\n=== {args.cohort} composite · within-subject event-peak variability ===")
    print(f"  n_subjects with ≥3 events: {len(results)}")
    print(f"  events per subject: median {np.median(ns):.0f}  range {ns.min()}-{ns.max()}")
    print(f"  Within-subject SD of event peaks: median {np.median(sds):.3f} Hz  "
          f"IQR [{np.percentile(sds,25):.3f}, {np.percentile(sds,75):.3f}]")
    print(f"  Within-subject IQR of event peaks: median {np.median(iqrs):.3f} Hz")
    print(f"  Between-subject SD of per-subject means: {np.std(means, ddof=1):.3f} Hz")
    print(f"  Per-subject mean distribution: mean {np.mean(means):.3f}   "
          f"std {np.std(means):.3f}")

    all_peaks = np.concatenate([r['peaks'] for r in results])
    print(f"\n  Pooled event peak freq SD (all events): {np.std(all_peaks, ddof=1):.3f} Hz")
    print(f"  Pooled event peak freq mean: {np.mean(all_peaks):.3f} Hz")

    within_var = float(np.mean(sds ** 2))
    between_var = float(np.var(means, ddof=1))
    icc_est = between_var / (between_var + within_var) if (between_var + within_var) > 0 else np.nan
    print(f"\n  Trait-ICC = between_var / (between + within) = {icc_est:.3f}")
    print(f"    (between_var {between_var:.4f}; within_var {within_var:.4f})")
    print(f"  B22 envelope LEMON: within-subject SD 0.62 Hz, between-subject SD 0.32 Hz, ICC ≈ 0.19")

    rows = [{k: v for k, v in r.items() if k != 'peaks'} for r in results]
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, 'within_subject_event_peaks.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(sds, bins=25, color='firebrick', edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(np.median(sds), color='blue', ls='--', lw=1.5,
                label=f'median {np.median(sds):.2f} Hz')
    ax.axvline(0.62, color='gray', ls=':', lw=1.5,
                label='envelope B22 (0.62)')
    ax.axvline(1.12, color='black', ls=':', lw=1.5,
                label='Dortmund retest Δ std (1.12)')
    ax.set_xlabel('within-subject SD of event peak freq (Hz)')
    ax.set_ylabel('subjects')
    ax.set_title(f'{args.cohort} composite · within-subj SD (n={len(sds)})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    order = np.argsort(means)
    for j, i in enumerate(order):
        peaks = results[i]['peaks']
        ax.scatter([j] * len(peaks), peaks, s=8, alpha=0.4, color='steelblue')
        ax.errorbar(j, means[i], yerr=sds[i], fmt='o', color='firebrick',
                     capsize=0, markersize=3, elinewidth=0.5)
    ax.axhline(7.83, color='green', ls='--', lw=1, label='Schumann 7.83 Hz')
    ax.set_xlabel('subject (sorted by mean peak)')
    ax.set_ylabel('event peak freq (Hz)')
    ax.set_title(f'{args.cohort} composite · per-subject peaks, sorted')
    ax.legend(fontsize=9)
    ax.set_ylim(ZOOM_LO, ZOOM_HI)
    ax.grid(alpha=0.3)

    plt.suptitle(f'B22 · within-subject event-to-event peak variability · {args.cohort} composite v2',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'within_subject_event_peaks.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/within_subject_event_peaks.png")


if __name__ == '__main__':
    main()
