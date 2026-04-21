#!/usr/bin/env python3
"""
B27 — Event-locked aggregate PSD across all three cohorts.

For each subject in LEMON, HBN R4, TDBRAIN (462 total):
  1. Compute per-subject average event-window PSD (4-s Welch centered at
     t0_net + 1 s, the Q4/ignition peak lag from B14).
  2. Compute per-subject baseline PSD = median over all 4-s sliding windows
     (1-s hop) of the recording.
  3. Per-subject event/baseline ratio spectrum (wide band [2, 25] Hz).

Grand-average the log10(event/baseline) ratio across all 462 subjects, with
subject-level bootstrap 95% CI. Looking for the species-level event-locked
7.83 Hz bump that was absent in the standing aggregate (B26).

Output: single canonical figure showing the event-locked population
spectral signature.
"""
from __future__ import annotations
import glob as globfn
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon, load_hbn, load_tdbrain

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
FREQ_LO, FREQ_HI = 2.0, 25.0


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def locate_hbn_set(sub_id, release):
    pat = f'/Volumes/T9/hbn_data/cmi_bids_{release}/{sub_id}/eeg/*RestingState_eeg.set'
    files = sorted(globfn.glob(pat))
    return files[0] if files else None


def load_recording(sub_id, dataset, release=None):
    if dataset == 'lemon':
        return load_lemon(sub_id, condition='EC')
    if dataset == 'hbn':
        set_path = locate_hbn_set(sub_id, release)
        if not set_path:
            return None
        return load_hbn(set_path)
    if dataset == 'tdbrain':
        return load_tdbrain(sub_id, condition='EC')
    raise ValueError(f"Unknown dataset: {dataset}")


def process_subject(args):
    sub_id, events_path, dataset, release = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 1:
        return None
    try:
        raw = load_recording(sub_id, dataset, release)
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
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    f_band = freqs[mask]

    # Baseline (all sliding windows)
    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)[mask]
        base_rows.append(psd)
    if len(base_rows) < 10:
        return None
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    # Event windows
    ev_rows = []
    for _, ev in events.iterrows():
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[mask]
        ev_rows.append(psd)
    if not ev_rows:
        return None
    ev_avg = np.nanmean(np.array(ev_rows), axis=0)

    log_ratio = np.log10(ev_avg + 1e-20) - np.log10(baseline + 1e-20)
    return {
        'subject_id': sub_id,
        'dataset': dataset,
        'n_events': int(len(ev_rows)),
        'freqs': f_band,
        'log_ratio': log_ratio,
    }


def bootstrap_ci(mat, n_boot=1000, seed=0):
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


def gather_subjects(dataset, release=None):
    if dataset == 'hbn':
        events_dir = os.path.join(EVENTS_BASE, f'hbn_{release}')
    elif dataset == 'tdbrain':
        events_dir = os.path.join(EVENTS_BASE, 'tdbrain')
    else:
        events_dir = os.path.join(EVENTS_BASE, 'lemon')
    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep, dataset, release))
    return tasks


def main():
    datasets = [('lemon', None), ('hbn', 'R4'), ('tdbrain', None)]
    all_tasks = []
    for ds, rel in datasets:
        all_tasks.extend(gather_subjects(ds, rel))
    print(f"Total subjects: {len(all_tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, all_tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    # Common grid
    common = np.arange(FREQ_LO, FREQ_HI + 0.005, 0.05)
    per_dataset = {}
    for r in results:
        ds = r['dataset']
        per_dataset.setdefault(ds, []).append(
            np.interp(common, r['freqs'], r['log_ratio']))

    # Pooled
    all_mat = np.vstack([np.array(v) for v in per_dataset.values()])
    grand_all, lo_all, hi_all = bootstrap_ci(all_mat)

    print(f"\n=== Pooled event-locked aggregate (n={len(all_mat)}) ===")
    # Peak detection: local maxima with height > 0.02
    peaks, _ = find_peaks(grand_all, height=0.02, distance=5)
    print(f"  Local peaks (log-ratio > 0.02):")
    for i in peaks:
        print(f"    {common[i]:.3f} Hz  log_ratio {grand_all[i]:+.4f}  "
              f"({10 ** grand_all[i]:.3f}×)  CI [{lo_all[i]:+.3f}, {hi_all[i]:+.3f}]")

    # Per-cohort
    print(f"\n=== Per-cohort event-locked aggregate ===")
    grand_per_ds = {}
    for ds, arr in per_dataset.items():
        mat = np.array(arr)
        grand, lo, hi = bootstrap_ci(mat)
        grand_per_ds[ds] = (grand, lo, hi, mat)
        peaks_ds, _ = find_peaks(grand, height=0.02, distance=5)
        print(f"  {ds}  n={len(mat)}:")
        for i in peaks_ds:
            print(f"    {common[i]:.3f} Hz  log_ratio {grand[i]:+.4f}  ({10 ** grand[i]:.2f}×)")

    # Landmarks
    print(f"\n=== Pooled landmark values ===")
    for t in [7.0, 7.60, 7.83, 8.10, 8.50, 9.45, 14.3, 20.8]:
        i = int(np.argmin(np.abs(common - t)))
        print(f"  {common[i]:.3f} Hz: log_ratio {grand_all[i]:+.4f}  ({10 ** grand_all[i]:.2f}×)")

    # Save CSVs
    pd.DataFrame({'freq_hz': common, 'log_ratio_grand_mean': grand_all,
                   'ci_lo': lo_all, 'ci_hi': hi_all,
                   'n_subjects': len(all_mat)}).to_csv(
        os.path.join(OUT_DIR, 'event_locked_aggregate_pooled.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # Panel 1: per cohort
    ax = axes[0]
    colors = {'lemon': '#2166ac', 'hbn': '#d7301f', 'tdbrain': '#2d8659'}
    for ds, (grand, lo, hi, mat) in grand_per_ds.items():
        ratio = 10 ** grand
        ax.plot(common, ratio, color=colors.get(ds, 'k'), lw=1.6,
                label=f'{ds.upper()} (n={len(mat)})')
        ax.fill_between(common, 10 ** lo, 10 ** hi,
                         color=colors.get(ds, 'k'), alpha=0.2)
    ax.axhline(1.0, color='k', lw=0.6)
    ax.axvline(7.83, color='green', ls='--', lw=0.7, alpha=0.7,
                label='Schumann 7.83 Hz')
    ax.axvline(7.60, color='gray', ls=':', lw=0.7, alpha=0.7,
                label='φ-boundary 7.60 Hz')
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_ylabel('event / baseline PSD (×)')
    ax.set_title('Per-cohort event-locked aggregate')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # Panel 2: pooled
    ax = axes[1]
    ratio = 10 ** grand_all
    ax.plot(common, ratio, color='k', lw=2,
             label=f'All pooled (n={len(all_mat)})')
    ax.fill_between(common, 10 ** lo_all, 10 ** hi_all,
                     color='gray', alpha=0.3)
    ax.axhline(1.0, color='k', lw=0.6)
    ax.axvline(7.83, color='green', ls='--', lw=1, alpha=0.7,
                label='Schumann 7.83 Hz')
    ax.axvline(7.60, color='gray', ls=':', lw=0.7, alpha=0.7,
                label='φ-boundary 7.60 Hz')
    for h in [14.3, 20.8]:
        ax.axvline(h, color='green', ls=':', lw=0.5, alpha=0.5)
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('event / baseline PSD (×)')
    ax.set_title(f'Pooled event-locked aggregate — {len(all_mat)} subjects')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    plt.suptitle('B27 — Event-locked population-aggregate PSD',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'event_locked_aggregate.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/event_locked_aggregate.png")


if __name__ == '__main__':
    main()
