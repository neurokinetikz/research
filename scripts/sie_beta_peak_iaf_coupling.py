#!/usr/bin/env python3
"""
B28 — Is the ~20 Hz event-locked peak 2×IAF-coupled or fixed?

B27 showed a secondary event-locked peak at 20.1 Hz (3.76×, prominence 0.10
above broadband floor). Candidate interpretations:

  A. 2×IAF coupling    → slope 2.0 when fitting beta_peak ~ α·IAF + β, b ≈ 0
  B. Fixed at ~20 Hz   → slope ≈ 0 (like 7.83 Hz peak is fixed)
  C. φ²·SR1 (20.5 Hz)  → fixed but different intercept

For each subject, compute:
  - IAF (alpha peak in [7, 13] from aperiodic-corrected aggregate PSD)
  - β event-locked peak: argmax in [16, 24] Hz of event-window/baseline ratio
    (4-s windows at t0_net + 1 s)
  - Also compute 7.83 Hz event peak as positive-control (should be IAF-indep.)

Correlate β_peak vs IAF across subjects. Report Spearman, OLS slope.

If slope ~2 and intercept ~0 → 2×IAF harmonic.
If slope ~0 → fixed frequency (φ²·SR1 or other fixed mechanism).
Pearson r between 7.83-peak and 20-peak within subject: high → shared origin,
low → independent.
"""
from __future__ import annotations
import argparse
import glob as globfn
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, pearsonr
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

IAF_WIN_SEC = 8.0
IAF_HOP_SEC = 2.0
IAF_NFFT_MULT = 4
IAF_LO, IAF_HI = 7.0, 13.0

BETA_LO, BETA_HI = 16.0, 24.0   # where 2×IAF would sit for IAF 8-12
SR_LO, SR_HI = 7.0, 8.3

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


def compute_iaf(y, fs):
    nperseg = int(round(IAF_WIN_SEC * fs))
    nhop = int(round(IAF_HOP_SEC * fs))
    nfft = nperseg * IAF_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= IAF_LO) & (freqs <= IAF_HI)
    psds = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psds.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if not psds:
        return np.nan
    return parabolic_peak(np.nanmedian(np.array(psds), axis=0), freqs[mask])


def event_peak_in_band(y, fs, t_events, f_lo, f_hi):
    nperseg = int(round(EV_WIN_SEC * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    f_band = freqs[mask]
    nhop = int(round(1.0 * fs))
    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        base_rows.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if len(base_rows) < 10:
        return np.nan, np.nan
    baseline = np.nanmedian(np.array(base_rows), axis=0)
    ev_rows = []
    for t0 in t_events:
        tc = t0 + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        ev_rows.append(welch_one(y[i0:i1], fs, nfft)[mask])
    if not ev_rows:
        return np.nan, np.nan
    ev_avg = np.nanmean(np.array(ev_rows), axis=0)
    ratio = (ev_avg + 1e-20) / (baseline + 1e-20)
    pk_f = parabolic_peak(ratio, f_band)
    pk_r = float(ratio[int(np.argmax(ratio))])
    return pk_f, pk_r


def process_subject(args):
    sub_id, events_path, dataset, release = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 3:
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
    iaf = compute_iaf(y, fs)
    t_events = events['t0_net'].astype(float).values
    sr_f, sr_r = event_peak_in_band(y, fs, t_events, SR_LO, SR_HI)
    beta_f, beta_r = event_peak_in_band(y, fs, t_events, BETA_LO, BETA_HI)
    return {
        'subject_id': sub_id,
        'dataset': dataset,
        'iaf_hz': iaf,
        'sr_peak_hz': sr_f,
        'sr_peak_ratio': sr_r,
        'beta_peak_hz': beta_f,
        'beta_peak_ratio': beta_r,
        'beta_over_iaf': beta_f / iaf if iaf > 0 else np.nan,
        'n_events': int(len(t_events)),
    }


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
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='lemon,hbn,tdbrain')
    ap.add_argument('--hbn-release', default='R4')
    args = ap.parse_args()

    datasets = args.datasets.split(',')
    tasks = []
    for ds in datasets:
        rel = args.hbn_release if ds == 'hbn' else None
        tasks.extend(gather_subjects(ds, rel))
    print(f"Total subjects: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(os.path.join(OUT_DIR, 'beta_peak_iaf_coupling.csv'), index=False)
    print(f"Successful: {len(df)}")

    # Drop rows with missing values
    good = df.dropna(subset=['iaf_hz', 'beta_peak_hz', 'sr_peak_hz']).copy()
    print(f"Complete rows: {len(good)}")

    print(f"\n=== Distributions ===")
    for c in ['iaf_hz', 'sr_peak_hz', 'beta_peak_hz', 'beta_over_iaf']:
        v = good[c]
        print(f"  {c:18s}  mean {v.mean():.3f}  median {v.median():.3f}  SD {v.std():.3f}")

    # Three-way cohort stratification
    print(f"\n=== Per cohort (beta peak freq and distribution) ===")
    for ds in datasets:
        sub = good[good['dataset'] == ds]
        if len(sub) < 5:
            continue
        slope, intercept = np.polyfit(sub['iaf_hz'].values,
                                        sub['beta_peak_hz'].values, 1)
        rho, p = spearmanr(sub['iaf_hz'], sub['beta_peak_hz'])
        print(f"  {ds} n={len(sub)}  beta-peak mean {sub['beta_peak_hz'].mean():.2f} "
              f"  OLS slope {slope:.3f}  intercept {intercept:.3f}  "
              f"  ρ_IAF×β={rho:+.3f} p={p:.2g}")

    # Pooled
    rho, p = spearmanr(good['iaf_hz'], good['beta_peak_hz'])
    r, _ = pearsonr(good['iaf_hz'], good['beta_peak_hz'])
    slope, intercept = np.polyfit(good['iaf_hz'].values,
                                   good['beta_peak_hz'].values, 1)
    print(f"\n=== Pooled (n={len(good)}) IAF × beta_peak ===")
    print(f"  Spearman ρ = {rho:+.3f}  p = {p:.3g}")
    print(f"  Pearson r  = {r:+.3f}")
    print(f"  OLS slope = {slope:.3f}  intercept = {intercept:.3f}")
    print(f"  Hypothesis predictions:")
    print(f"    A. 2×IAF coupling:     slope ≈ 2.0, intercept ≈ 0")
    print(f"    B. Fixed at 20.0 Hz:   slope ≈ 0.0, intercept ≈ 20")
    print(f"    C. Fixed at 20.5 (φ²·SR1): slope ≈ 0.0, intercept ≈ 20.5")
    # Compare residual sum of squares for three models
    rss_a = np.sum((good['beta_peak_hz'].values - 2.0 * good['iaf_hz'].values) ** 2)
    rss_b = np.sum((good['beta_peak_hz'].values - 20.0) ** 2)
    rss_c = np.sum((good['beta_peak_hz'].values - 20.5) ** 2)
    rss_ols = np.sum((good['beta_peak_hz'].values - (slope * good['iaf_hz'].values + intercept)) ** 2)
    print(f"\n  Residual SS (lower = better fit):")
    print(f"    A (β = 2·IAF):            {rss_a:.2f}")
    print(f"    B (β = 20.0):             {rss_b:.2f}")
    print(f"    C (β = 20.5):             {rss_c:.2f}")
    print(f"    OLS (best-fit line):      {rss_ols:.2f}")

    # Within-subject SR-peak vs beta-peak correlation (do these co-vary?)
    rho_shared, p_shared = spearmanr(good['sr_peak_ratio'], good['beta_peak_ratio'])
    print(f"\n=== Do SR (7.83) and beta (20) peaks co-vary across subjects? ===")
    print(f"  Spearman ρ(SR amplitude, β amplitude) = {rho_shared:+.3f}  p={p_shared:.2g}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: β_peak vs IAF
    ax = axes[0]
    colors = {'lemon': '#2166ac', 'hbn': '#d7301f', 'tdbrain': '#2d8659'}
    for ds in datasets:
        sub = good[good['dataset'] == ds]
        ax.scatter(sub['iaf_hz'], sub['beta_peak_hz'], s=20, alpha=0.55,
                    color=colors.get(ds, 'k'),
                    edgecolor='k', lw=0.3, label=f'{ds.upper()} (n={len(sub)})')
    rng = np.array([good['iaf_hz'].min() - 0.3, good['iaf_hz'].max() + 0.3])
    ax.plot(rng, 2.0 * rng, 'k--', lw=1.0, label='A: 2×IAF')
    ax.plot(rng, [20.0, 20.0], color='#1a9641', ls=':', lw=1.0, label='B: fixed 20.0')
    ax.plot(rng, [20.5, 20.5], color='#8c1a1a', ls=':', lw=1.0,
             label='C: φ²·SR1 = 20.5')
    ax.plot(rng, slope * rng + intercept, color='purple', lw=1.5,
             label=f'OLS: {slope:.2f}·IAF + {intercept:.2f}')
    ax.set_xlabel('IAF (Hz)')
    ax.set_ylabel('event-locked β peak freq (Hz)')
    ax.set_title(f'β peak vs IAF  ·  ρ={rho:+.2f} p={p:.2g}  n={len(good)}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: SR-amplitude vs β-amplitude
    ax = axes[1]
    for ds in datasets:
        sub = good[good['dataset'] == ds]
        ax.scatter(sub['sr_peak_ratio'], sub['beta_peak_ratio'], s=20, alpha=0.55,
                    color=colors.get(ds, 'k'),
                    edgecolor='k', lw=0.3, label=f'{ds.upper()}')
    ax.set_xlabel('SR (7.83 Hz) event peak amplitude (×)')
    ax.set_ylabel('β (~20 Hz) event peak amplitude (×)')
    ax.set_title(f'Do SR and β amplitudes co-vary?  ρ={rho_shared:+.2f} p={p_shared:.2g}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle('B28 — 20 Hz event-locked peak: 2×IAF vs fixed-frequency',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'beta_peak_iaf_coupling.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/beta_peak_iaf_coupling.png")


if __name__ == '__main__':
    main()
