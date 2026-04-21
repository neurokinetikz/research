#!/usr/bin/env python3
"""
B31 — Peri-event time course of SR vs β band amplitudes.

B14 showed the SR-band boost peaks at +1 s post-t0_net. What about the β
companion (~20 Hz from B27/B28)? Do the two bands emerge concurrently
(shared generator) or with different latencies (sequential engagement)?

For each subject in LEMON, HBN R4, TDBRAIN: full-recording sliding Welch,
extract per-window power in two bands:

  SR band: [7.0, 8.2] Hz     (peak at 7.83)
  β  band: [19, 21] Hz        (peak at 20)

Log-ratio over subject baseline = log10(band power at t) − log10(all-time
median). Per event, interpolate both time courses onto [-15, +15] s grid
rel to t0_net. Grand-average per cohort and pooled, subject-level cluster
bootstrap.

Metrics per band: peak time, peak amplitude, rise slope (10% → 90% of peak).
"""
from __future__ import annotations
import glob as globfn
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
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

WIN_SEC = 4.0
HOP_SEC = 1.0
NFFT_MULT = 4
SR_RANGE = (7.0, 8.2)
BETA_RANGE = (19.0, 21.0)
PSD_LO, PSD_HI = 2.0, 25.0

TGRID = np.arange(-15.0, 15.0 + 0.5, 1.0)


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

    nperseg = int(round(WIN_SEC * fs))
    nhop = int(round(HOP_SEC * fs))
    nfft = nperseg * NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= PSD_LO) & (freqs <= PSD_HI)
    sr_m = (freqs >= SR_RANGE[0]) & (freqs <= SR_RANGE[1])
    bt_m = (freqs >= BETA_RANGE[0]) & (freqs <= BETA_RANGE[1])

    t_cent, sr_pow, bt_pow = [], [], []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)
        sr_pow.append(float(np.mean(psd[sr_m])))
        bt_pow.append(float(np.mean(psd[bt_m])))
        t_cent.append((i + nperseg / 2) / fs)
    t_cent = np.array(t_cent)
    sr_pow = np.array(sr_pow)
    bt_pow = np.array(bt_pow)
    # Log-ratio over subject baseline
    sr_log = np.log10(sr_pow + 1e-20) - np.log10(np.nanmedian(sr_pow) + 1e-20)
    bt_log = np.log10(bt_pow + 1e-20) - np.log10(np.nanmedian(bt_pow) + 1e-20)

    sr_traces, bt_traces = [], []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        rel = t_cent - t0
        m = (rel >= TGRID[0] - 1) & (rel <= TGRID[-1] + 1)
        if m.sum() == 0:
            continue
        sr_traces.append(np.interp(TGRID, rel[m], sr_log[m],
                                    left=np.nan, right=np.nan))
        bt_traces.append(np.interp(TGRID, rel[m], bt_log[m],
                                    left=np.nan, right=np.nan))
    if not sr_traces:
        return None
    return {
        'subject_id': sub_id,
        'dataset': dataset,
        'sr_trace': np.nanmean(np.array(sr_traces), axis=0),
        'bt_trace': np.nanmean(np.array(bt_traces), axis=0),
    }


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


def rise_10_90(trace, t=TGRID):
    """Rise time from 10% to 90% of peak (in s)."""
    if np.all(np.isnan(trace)):
        return np.nan, np.nan, np.nan
    pk = int(np.nanargmax(trace))
    pk_val = trace[pk]
    if pk_val <= 0:
        return np.nan, np.nan, np.nan
    # Walk backward from peak to find 10% and 90% crossings
    ten = 0.1 * pk_val
    ninety = 0.9 * pk_val
    i10 = pk
    while i10 > 0 and trace[i10] > ten:
        i10 -= 1
    i90 = pk
    while i90 > 0 and trace[i90] > ninety:
        i90 -= 1
    return float(t[i10]), float(t[i90]), float(t[pk])


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

    sr_mat = np.array([r['sr_trace'] for r in results])
    bt_mat = np.array([r['bt_trace'] for r in results])
    sr_grand, sr_lo, sr_hi = bootstrap_ci(sr_mat)
    bt_grand, bt_lo, bt_hi = bootstrap_ci(bt_mat)

    sr_ten, sr_ninety, sr_peak = rise_10_90(sr_grand)
    bt_ten, bt_ninety, bt_peak = rise_10_90(bt_grand)
    print(f"\n=== Timing of peri-event amplitude rises (pooled, n={len(results)}) ===")
    print(f"  SR band [7.0-8.2]:  10% @ {sr_ten:+.1f}s  90% @ {sr_ninety:+.1f}s  "
          f"peak @ {sr_peak:+.1f}s  (peak log = {sr_grand[int(sr_peak/1+15)]:+.3f})")
    print(f"  β band [19-21]:     10% @ {bt_ten:+.1f}s  90% @ {bt_ninety:+.1f}s  "
          f"peak @ {bt_peak:+.1f}s  (peak log = {bt_grand[int(bt_peak/1+15)]:+.3f})")
    lag = bt_peak - sr_peak
    print(f"  Δ peak (β − SR) = {lag:+.1f} s  → ", end='')
    if abs(lag) <= 0.5:
        print("CONCURRENT engagement")
    elif lag > 0:
        print(f"β LAGS SR by {lag:.1f} s")
    else:
        print(f"β LEADS SR by {-lag:.1f} s")

    # Correlation of full time-courses: if shared generator, traces should be
    # highly correlated across time
    r_corr = np.corrcoef(sr_grand, bt_grand)[0, 1]
    print(f"\n  Pearson r(SR_grand, β_grand) across TGRID: {r_corr:+.3f}")
    # Amplitude ratio at peak
    peak_ratio_sr = 10 ** sr_grand[int(sr_peak + 15)]
    peak_ratio_bt = 10 ** bt_grand[int(bt_peak + 15)]
    print(f"  SR peak amplitude: {peak_ratio_sr:.2f}×   β peak amplitude: {peak_ratio_bt:.2f}×")

    # Per-cohort peak times
    print(f"\n=== Per-cohort peak times ===")
    for ds in set(r['dataset'] for r in results):
        sub_sr = np.array([r['sr_trace'] for r in results if r['dataset'] == ds])
        sub_bt = np.array([r['bt_trace'] for r in results if r['dataset'] == ds])
        sr_g, _, _ = bootstrap_ci(sub_sr)
        bt_g, _, _ = bootstrap_ci(sub_bt)
        _, _, sr_pk = rise_10_90(sr_g)
        _, _, bt_pk = rise_10_90(bt_g)
        print(f"  {ds} (n={len(sub_sr)}): SR peak {sr_pk:+.1f}s  β peak {bt_pk:+.1f}s  Δ={bt_pk-sr_pk:+.1f}s")

    # Save CSV
    pd.DataFrame({'t': TGRID,
                   'sr_grand': sr_grand, 'sr_lo': sr_lo, 'sr_hi': sr_hi,
                   'bt_grand': bt_grand, 'bt_lo': bt_lo, 'bt_hi': bt_hi}).to_csv(
        os.path.join(OUT_DIR, 'peri_event_sr_vs_beta.csv'), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(TGRID, 10 ** sr_grand, color='#8c1a1a', lw=2.2,
            label=f'SR band [7.0–8.2 Hz]  peak +{sr_peak:.1f}s')
    ax.fill_between(TGRID, 10 ** sr_lo, 10 ** sr_hi, color='#8c1a1a', alpha=0.22)
    ax.plot(TGRID, 10 ** bt_grand, color='#2166ac', lw=2.2,
            label=f'β band [19–21 Hz]  peak +{bt_peak:.1f}s')
    ax.fill_between(TGRID, 10 ** bt_lo, 10 ** bt_hi, color='#2166ac', alpha=0.22)
    ax.axhline(1.0, color='k', lw=0.7)
    ax.axvline(0, color='k', ls='--', lw=0.7)
    ax.set_xlabel('time rel. t0_net (s)', fontsize=11)
    ax.set_ylabel('band power / baseline (×)', fontsize=11)
    ax.set_title(f'B31 — Peri-event SR vs β time course (n = {len(results)} subjects, 3 cohorts)')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'peri_event_sr_vs_beta.png'),
                 dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/peri_event_sr_vs_beta.png")


if __name__ == '__main__':
    main()
