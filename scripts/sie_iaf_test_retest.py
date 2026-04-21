#!/usr/bin/env python3
"""
B21 — Within-subject test-retest stability of the ignition-peak frequency.

For each Dortmund subject with events in both EC-post sessions (ses-1 and ses-2):
  1. Compute IAF from full recording (each session separately).
  2. Compute ignition peak from event-time PSDs averaged across that session's
     events (same method as B20).
Correlate sessions. Reports ICC(2,1), Pearson r, and Spearman ρ for:
  - IAF (positive control — should replicate; IAF is a known stable trait)
  - Ignition peak (test — is it as stable as IAF?)

Tests whether the ignition-peak frequency is a stable subject trait (ICC > 0.5)
or session-dependent (ICC → 0). A stable trait would imply the population mean
at ~7.7-7.8 Hz reflects consistent individual subject values, not session-level
averaging of random peaks.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_dortmund

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse', 'dortmund_retest')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

CONDITION = 'EC_post'
DORT_TASK = 'EyesClosed'
DORT_ACQ = 'post'

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
ZOOM_LO, ZOOM_HI = 6.5, 9.0

IAF_WIN_SEC = 8.0
IAF_HOP_SEC = 2.0
IAF_NFFT_MULT = 4
IAF_LO, IAF_HI = 7.0, 13.0


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


def compute_iaf(y, fs):
    nperseg = int(round(IAF_WIN_SEC * fs))
    nhop = int(round(IAF_HOP_SEC * fs))
    nfft = nperseg * IAF_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= IAF_LO) & (freqs <= IAF_HI)
    psds = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)[mask]
        psds.append(psd)
    if not psds:
        return np.nan
    grand = np.nanmedian(np.array(psds), axis=0)
    return parabolic_peak(grand, freqs[mask])


def compute_ignition_peak(y, fs, t_events):
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
        return np.nan
    baseline = np.nanmedian(np.array(base_rows), axis=0)
    ev_rows = []
    for t0 in t_events:
        tc = t0 + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[mask]
        ev_rows.append(psd)
    if not ev_rows:
        return np.nan
    ev_avg = np.nanmean(np.array(ev_rows), axis=0)
    ratio = (ev_avg + 1e-20) / (baseline + 1e-20)
    return parabolic_peak(ratio, f_band)


def process_session(args):
    sub_id, events_path, ses = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 1:
        return None
    try:
        raw = load_dortmund(sub_id, task=DORT_TASK, acq=DORT_ACQ, ses=str(ses))
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    iaf = compute_iaf(y, fs)
    ign = compute_ignition_peak(y, fs, events['t0_net'].astype(float).values)
    return {
        'subject_id': sub_id,
        'ses': ses,
        'n_events': int(len(events)),
        'iaf_hz': iaf,
        'ignition_peak_hz': ign,
    }


def icc_2_1(x1, x2):
    """ICC(2,1) — two-way random effects, single measurement, absolute agreement."""
    n = len(x1)
    if n < 3:
        return np.nan
    mean_1 = np.mean(x1); mean_2 = np.mean(x2)
    grand_mean = (mean_1 + mean_2) / 2
    subj_means = (x1 + x2) / 2
    # Mean squares
    ms_between_subj = 2 * np.var(subj_means, ddof=1)
    ms_between_ses = n * ((mean_1 - grand_mean) ** 2 + (mean_2 - grand_mean) ** 2)
    resid = 0.0
    for i in range(n):
        for j, x_ij in enumerate([x1[i], x2[i]]):
            row_mean_j = mean_1 if j == 0 else mean_2
            resid += (x_ij - subj_means[i] - row_mean_j + grand_mean) ** 2
    ms_residual = resid / (n - 1)
    denom = ms_between_subj + (ms_between_ses - ms_residual) / n + ms_residual
    if denom <= 0:
        return np.nan
    return (ms_between_subj - ms_residual) / denom


def main():
    ses1_summary = pd.read_csv(os.path.join(EVENTS_BASE, f'dortmund_{CONDITION}',
                                              'extraction_summary.csv'))
    ses2_summary = pd.read_csv(os.path.join(EVENTS_BASE, f'dortmund_{CONDITION}_ses2',
                                              'extraction_summary.csv'))
    ok1 = set(ses1_summary[(ses1_summary['status'] == 'ok') & (ses1_summary['n_events'] >= 1)]['subject_id'])
    ok2 = set(ses2_summary[(ses2_summary['status'] == 'ok') & (ses2_summary['n_events'] >= 1)]['subject_id'])
    both = sorted(ok1 & ok2)
    print(f"Subjects with ≥1 events in both EC_post ses1 AND ses2: {len(both)}")

    tasks = []
    for sub_id in both:
        ep1 = os.path.join(EVENTS_BASE, f'dortmund_{CONDITION}',
                            f'{sub_id}_sie_events.csv')
        ep2 = os.path.join(EVENTS_BASE, f'dortmund_{CONDITION}_ses2',
                            f'{sub_id}_sie_events.csv')
        tasks.append((sub_id, ep1, 1))
        tasks.append((sub_id, ep2, 2))
    print(f"Tasks: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_session, tasks)
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'per_session_peaks.csv'), index=False)

    # Pivot so each subject has ses-1 and ses-2 values in columns
    wide = df.pivot(index='subject_id', columns='ses',
                     values=['iaf_hz', 'ignition_peak_hz', 'n_events'])
    wide.columns = [f'{a}_ses{b}' for a, b in wide.columns]
    both_good = wide.dropna(subset=['iaf_hz_ses1', 'iaf_hz_ses2',
                                      'ignition_peak_hz_ses1', 'ignition_peak_hz_ses2'])
    both_good.to_csv(os.path.join(OUT_DIR, 'per_subject_retest.csv'))
    print(f"\nSubjects with complete IAF+ignition in both sessions: {len(both_good)}")

    for metric in ['iaf_hz', 'ignition_peak_hz']:
        x1 = both_good[f'{metric}_ses1'].values
        x2 = both_good[f'{metric}_ses2'].values
        icc = icc_2_1(x1, x2)
        r, p_r = pearsonr(x1, x2)
        rho, p_rho = spearmanr(x1, x2)
        diff = x2 - x1
        label = 'IAF' if 'iaf' in metric else 'Ignition peak'
        print(f"\n=== {label} test-retest ===")
        print(f"  ICC(2,1) = {icc:.3f}")
        print(f"  Pearson r = {r:.3f}  p = {p_r:.2g}")
        print(f"  Spearman ρ = {rho:.3f}  p = {p_rho:.2g}")
        print(f"  ses1 mean {np.mean(x1):.3f} ± {np.std(x1):.3f} Hz")
        print(f"  ses2 mean {np.mean(x2):.3f} ± {np.std(x2):.3f} Hz")
        print(f"  ses2 − ses1: mean {np.mean(diff):+.3f} Hz, std {np.std(diff):.3f} Hz")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, label in [(axes[0], 'iaf_hz', 'IAF'),
                                (axes[1], 'ignition_peak_hz', 'Ignition peak')]:
        x1 = both_good[f'{metric}_ses1'].values
        x2 = both_good[f'{metric}_ses2'].values
        icc = icc_2_1(x1, x2)
        r, _ = pearsonr(x1, x2)
        ax.scatter(x1, x2, s=30, alpha=0.6, color='steelblue',
                    edgecolor='k', lw=0.3)
        lo, hi = min(x1.min(), x2.min()) - 0.3, max(x1.max(), x2.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='identity')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f'{label} session 1 (Hz)')
        ax.set_ylabel(f'{label} session 2 (Hz)')
        ax.set_title(f'{label} retest · ICC={icc:.2f}  r={r:.2f}  n={len(x1)}')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('B21 — Dortmund EC-post test-retest (IAF vs ignition peak)',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'test_retest.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/test_retest.png")


if __name__ == '__main__':
    main()
