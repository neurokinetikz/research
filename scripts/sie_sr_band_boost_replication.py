#!/usr/bin/env python3
"""
B17 — HBN / TDBRAIN replication of the SR-band event-boost (B12).

Cross-dataset test of the paper's central finding. Replicates the cohort-scale
SR-band [7.0, 8.2 Hz] event-boost analysis on HBN (R4, largest release) and
TDBRAIN, comparing to the LEMON numbers:

  - Per-subject median event-boost (raw and vs baseline)
  - Wilcoxon per-subject boost vs 1.0×
  - Fraction of time dominant 6-9 Hz peak sits in the SR band

Usage:
  python scripts/sie_sr_band_boost_replication.py --dataset hbn --release R4
  python scripts/sie_sr_band_boost_replication.py --dataset tdbrain

Event files expected at:
  exports_sie/hbn_R{N}/sub-*_sie_events.csv
  exports_sie/tdbrain/sub-*_sie_events.csv
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
from scipy.stats import wilcoxon
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_hbn, load_tdbrain

OUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                         'images', 'psd_timelapse')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

WIN_SEC = 4.0
HOP_SEC = 1.0
FREQ_LO, FREQ_HI = 4.0, 9.0
BROAD_LO, BROAD_HI = 6.0, 9.0
SR_LO, SR_HI = 7.0, 8.2
NFFT_MULT = 4


def locate_hbn_set(sub_id, release):
    pat = f'/Volumes/T9/hbn_data/cmi_bids_{release}/{sub_id}/eeg/*RestingState_eeg.set'
    files = sorted(globfn.glob(pat))
    return files[0] if files else None


def load_recording(sub_id, dataset, release=None):
    if dataset == 'hbn':
        set_path = locate_hbn_set(sub_id, release)
        if not set_path:
            return None
        return load_hbn(set_path)
    if dataset == 'tdbrain':
        # condition arg matters for tdbrain; default to EC (which is what Stage 1 used)
        return load_tdbrain(sub_id, condition='EC')
    raise ValueError(f"Unknown dataset: {dataset}")


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


def band_peak(freqs, P, f_lo, f_hi):
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    idx_band = np.where(mask)[0]
    peaks = np.full(P.shape[1], np.nan)
    peak_pow = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        col = P[idx_band, j]
        if not np.isfinite(col).any() or np.all(col == 0):
            continue
        k = int(np.argmax(col))
        peaks[j] = freqs[idx_band][k]
        peak_pow[j] = col[k]
    return peaks, peak_pow


def process_subject(args):
    sub_id, events_path, dataset, release = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) == 0:
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

    t, freqs, P = sliding_welch(y, fs)
    peak_f_broad, _ = band_peak(freqs, P, BROAD_LO, BROAD_HI)
    in_sr = (peak_f_broad >= SR_LO) & (peak_f_broad <= SR_HI)
    pct_peak_in_sr = float(np.nanmean(in_sr) * 100)

    _, peak_p_sr = band_peak(freqs, P, SR_LO, SR_HI)
    baseline_median = float(np.nanmedian(peak_p_sr))

    rows = []
    for _, ev in events.iterrows():
        te = float(ev['t0_net'])
        j = int(np.argmin(np.abs(t - te)))
        if 0 <= j < len(t):
            boost = (peak_p_sr[j] / baseline_median) if baseline_median > 0 else np.nan
            rows.append({
                'subject_id': sub_id,
                't0_net': te,
                'sr_peak_p': float(peak_p_sr[j]),
                'broad_peak_f': float(peak_f_broad[j]),
                'broad_in_sr': bool(in_sr[j]),
                'sr_boost': float(boost),
            })

    return {
        'subject_id': sub_id,
        'rows': rows,
        'pct_peak_in_sr': pct_peak_in_sr,
        'baseline_sr': baseline_median,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['hbn', 'tdbrain'])
    ap.add_argument('--release', default=None, help='HBN release (R1-R6)')
    args = ap.parse_args()

    if args.dataset == 'hbn':
        assert args.release, "HBN requires --release"
        events_dir = os.path.join(EVENTS_BASE, f'hbn_{args.release}')
        label = f'HBN-{args.release}'
        out_dir = os.path.join(OUT_BASE, f'hbn_{args.release}')
    else:
        events_dir = os.path.join(EVENTS_BASE, 'tdbrain')
        label = 'TDBRAIN'
        out_dir = os.path.join(OUT_BASE, 'tdbrain')
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
    print(f"Successful subjects: {len(results)}")

    all_rows, sub_rows = [], []
    for r in results:
        all_rows.extend(r['rows'])
        boosts = np.array([row['sr_boost'] for row in r['rows']
                            if np.isfinite(row['sr_boost'])])
        sub_rows.append({
            'subject_id': r['subject_id'],
            'n_events': len(r['rows']),
            'pct_peak_in_sr': r['pct_peak_in_sr'],
            'event_boost_median': float(np.median(boosts)) if len(boosts) else np.nan,
        })
    ev = pd.DataFrame(all_rows)
    sub = pd.DataFrame(sub_rows)
    ev.to_csv(os.path.join(out_dir, 'per_event_sr_boost.csv'), index=False)
    sub.to_csv(os.path.join(out_dir, 'per_subject_sr_summary.csv'), index=False)

    vals = sub['pct_peak_in_sr'].dropna()
    b = sub['event_boost_median'].dropna()

    print(f"\n=== {label}: % time broadband peak in SR [7.0, 8.2] Hz ===")
    print(f"  median {vals.median():.1f}%  IQR [{vals.quantile(.25):.1f}, {vals.quantile(.75):.1f}]")
    print(f"\n=== {label}: per-subject SR-band event boost ===")
    print(f"  median {b.median():.2f}×   IQR [{b.quantile(.25):.2f}, {b.quantile(.75):.2f}]")
    print(f"  pct subjects boost >= 1.5×: {(b>=1.5).mean()*100:.1f}%")
    print(f"  pct subjects boost <= 1.0×: {(b<=1.0).mean()*100:.1f}%")
    if len(b) > 3:
        stat, pval = wilcoxon(b - 1.0)
        print(f"  Wilcoxon vs 1.0×: stat={stat:.1f}, p={pval:.3g}")

    # LEMON reference
    print(f"\n=== LEMON reference (from B12) ===")
    print(f"  % peak in SR: 27.7% (IQR 18.9-34.7)")
    print(f"  event boost: 1.53× (IQR 1.19-2.19)")
    print(f"  pct >=1.5×: 54.2%   pct <=1.0×: 13.5%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(vals, bins=30, color='steelblue', edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(vals.median(), color='firebrick', ls='--', lw=1.5,
               label=f'{label} median {vals.median():.1f}%')
    ax.axvline(27.7, color='k', ls=':', lw=1, label='LEMON 27.7%')
    ax.set_xlabel('% time broadband peak in [7.0, 8.2] Hz')
    ax.set_ylabel('subjects')
    ax.set_title(f'{label}: peak-in-SR (n={len(sub)})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(b, bins=np.linspace(0, 6, 40), color='seagreen',
             edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(1.0, color='k', lw=0.8, label='no boost')
    ax.axvline(b.median(), color='firebrick', ls='--', lw=1.5,
                label=f'{label} median {b.median():.2f}×')
    ax.axvline(1.53, color='k', ls=':', lw=1, label='LEMON 1.53×')
    ax.set_xlabel('per-subject event boost (×)')
    ax.set_ylabel('subjects')
    ax.set_title(f'{label}: event boost')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(f'B17 — {label} replication of SR-band event-boost',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sr_band_replication.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/sr_band_replication.png")


if __name__ == '__main__':
    main()
