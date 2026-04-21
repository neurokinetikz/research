#!/usr/bin/env python3
"""
B20b — IAF-coupling test replication on HBN and TDBRAIN.

Generalized version of B20 that works on any of lemon / hbn / tdbrain, without
requiring a template_rho quality filter (uses all events). Tests whether the
fixed-frequency finding from LEMON Q4 replicates on full-event sets from other
cohorts.

Usage:
  python scripts/sie_iaf_coupling_multi.py --dataset lemon
  python scripts/sie_iaf_coupling_multi.py --dataset hbn --release R4
  python scripts/sie_iaf_coupling_multi.py --dataset tdbrain

Reports per-cohort:
  - IAF distribution
  - Ignition-peak distribution
  - Spearman/Pearson correlation (IAF × ignition_peak)
  - OLS slope (H1: IAF lock → 1; H2: fixed-freq → 0)
  - Gap distribution (ignition_peak − IAF)
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

OUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                         'images', 'psd_timelapse')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
ZOOM_LO, ZOOM_HI = 6.5, 9.0

IAF_WIN_SEC = 8.0
IAF_HOP_SEC = 2.0
IAF_NFFT_MULT = 4
IAF_LO, IAF_HI = 7.0, 13.0

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
        return np.nan, np.nan
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
        return np.nan, np.nan
    ev_avg = np.nanmean(np.array(ev_rows), axis=0)
    ratio = (ev_avg + 1e-20) / (baseline + 1e-20)
    return parabolic_peak(ratio, f_band), float(ratio[int(np.argmax(ratio))])


def process_subject(args):
    sub_id, events_path, dataset, release = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 2:
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
    ign_f, ign_r = compute_ignition_peak(y, fs, t_events)
    return {
        'subject_id': sub_id,
        'iaf_hz': iaf,
        'ignition_peak_hz': ign_f,
        'ignition_peak_ratio': ign_r,
        'n_events': int(len(t_events)),
        'gap_hz': (ign_f - iaf) if (np.isfinite(iaf) and np.isfinite(ign_f)) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['lemon', 'hbn', 'tdbrain'])
    ap.add_argument('--release', default=None)
    args = ap.parse_args()

    if args.dataset == 'hbn':
        assert args.release, 'HBN requires --release'
        events_dir = os.path.join(EVENTS_BASE, f'hbn_{args.release}')
        label = f'HBN-{args.release}'
        out_dir = os.path.join(OUT_BASE, f'hbn_{args.release}')
    elif args.dataset == 'tdbrain':
        events_dir = os.path.join(EVENTS_BASE, 'tdbrain')
        label = 'TDBRAIN'
        out_dir = os.path.join(OUT_BASE, 'tdbrain')
    else:
        events_dir = os.path.join(EVENTS_BASE, 'lemon')
        label = 'LEMON'
        out_dir = os.path.join(OUT_BASE, 'lemon_iaf')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep, args.dataset, args.release))
    print(f"{label}  subjects: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'iaf_vs_ignition_peak.csv'), index=False)

    good = df.dropna(subset=['iaf_hz', 'ignition_peak_hz']).copy()
    print(f"Successful: {len(df)}   Both IAF+ignition: {len(good)}")

    if len(good) < 5:
        print("Not enough subjects.")
        return

    print(f"\n=== {label} IAF ===")
    print(f"  mean {good['iaf_hz'].mean():.2f}   median {good['iaf_hz'].median():.2f}   std {good['iaf_hz'].std():.2f}   n={len(good)}")
    print(f"  5-95%: [{good['iaf_hz'].quantile(0.05):.2f}, {good['iaf_hz'].quantile(0.95):.2f}]")

    print(f"\n=== {label} Ignition peak ===")
    print(f"  mean {good['ignition_peak_hz'].mean():.2f}   median {good['ignition_peak_hz'].median():.2f}   std {good['ignition_peak_hz'].std():.2f}")

    print(f"\n=== Gap (ignition_peak − IAF) ===")
    print(f"  mean {good['gap_hz'].mean():+.2f}   median {good['gap_hz'].median():+.2f}   std {good['gap_hz'].std():.2f}")
    print(f"  frac ignition < IAF: {(good['gap_hz'] < 0).mean()*100:.0f}%")

    rho, p_sp = spearmanr(good['iaf_hz'], good['ignition_peak_hz'])
    r, p_pr = pearsonr(good['iaf_hz'], good['ignition_peak_hz'])
    x = good['iaf_hz'].values
    y_vals = good['ignition_peak_hz'].values
    slope, intercept = np.polyfit(x, y_vals, 1)
    print(f"\n=== {label} IAF × ignition_peak ===")
    print(f"  Spearman ρ = {rho:.3f}   p = {p_sp:.3g}")
    print(f"  Pearson  r = {r:.3f}   p = {p_pr:.3g}")
    print(f"  OLS: ignition = {slope:.3f} × IAF + {intercept:.3f}")
    print(f"  H1 (IAF lock) predicts slope=1; H2 (fixed {SCHUMANN_F} Hz) predicts slope=0, intercept={SCHUMANN_F}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(good['iaf_hz'], good['ignition_peak_hz'], s=30, alpha=0.6,
                color='steelblue', edgecolor='k', lw=0.3)
    rng = np.array([good['iaf_hz'].min() - 0.5, good['iaf_hz'].max() + 0.5])
    ax.plot(rng, rng, 'k--', lw=1, label='H1: ignition = IAF')
    ax.axhline(SCHUMANN_F, color='green', lw=1.2, ls=':',
                label=f'H2: fixed {SCHUMANN_F} Hz')
    ax.plot(rng, slope * rng + intercept, color='red', lw=1.5,
             label=f'OLS: {slope:.2f}×IAF + {intercept:.2f}')
    ax.set_xlabel('IAF (Hz)')
    ax.set_ylabel('ignition peak (Hz)')
    ax.set_title(f'{label}: IAF vs ignition · ρ={rho:.2f} p={p_sp:.2g} n={len(good)}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(good['gap_hz'], bins=25, color='firebrick', edgecolor='k',
             lw=0.3, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(good['gap_hz'].mean(), color='blue', ls='--', lw=1.5,
                label=f"mean {good['gap_hz'].mean():+.2f}")
    ax.set_xlabel('ignition peak − IAF (Hz)')
    ax.set_ylabel('subjects')
    ax.set_title(f'{label} gap distribution')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(f'B20b — IAF-coupling · {label}', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'iaf_vs_ignition_peak.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/iaf_vs_ignition_peak.png")


if __name__ == '__main__':
    main()
