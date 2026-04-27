#!/usr/bin/env python3
"""
B31 re-run on composite v2 detector.

Peri-event time course of SR [7.0, 8.2] and β [19, 21] band amplitudes,
interpolated onto [−15, +15] s grid rel t0_net. Log-ratio over subject
baseline. Grand-average with subject-level cluster bootstrap.

Tests whether SR and β peaks are:
  - concurrent (shared generator): Δ peak time ≈ 0
  - sequential (SR leads β): envelope B31 found Δ = +2 s (β lags SR)
  - other

Envelope B31: SR peaks at +1 s, β at +3 s, shape correlation r = 0.956.
β is a delayed amplitude-coupled echo, not a parallel resonance.

Cohort-parameterized.

Usage:
    python scripts/sie_peri_event_sr_vs_beta_composite.py --cohort lemon
    python scripts/sie_peri_event_sr_vs_beta_composite.py --cohort lemon_EO
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
NFFT_MULT = 4
SR_RANGE = (7.0, 8.2)
BETA_RANGE = (19.0, 21.0)
PSD_LO, PSD_HI = 2.0, 25.0

TGRID = np.arange(-15.0, 15.0 + 0.5, 1.0)

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
    if len(events) < 1:
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

    nperseg = int(round(WIN_SEC * fs))
    nhop = int(round(HOP_SEC * fs))
    nfft = nperseg * NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
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
    sr_log = np.log10(sr_pow + 1e-20) - np.log10(np.nanmedian(sr_pow) + 1e-20)
    bt_log = np.log10(bt_pow + 1e-20) - np.log10(np.nanmedian(bt_pow) + 1e-20)

    sr_traces, bt_traces = [], []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        rel = t_cent - t0
        m = (rel >= TGRID[0] - 1) & (rel <= TGRID[-1] + 1)
        if m.sum() == 0:
            continue
        sr_traces.append(np.interp(TGRID, rel[m], sr_log[m], left=np.nan, right=np.nan))
        bt_traces.append(np.interp(TGRID, rel[m], bt_log[m], left=np.nan, right=np.nan))
    if not sr_traces:
        return None
    return {
        'subject_id': sub_id,
        'n_events': len(sr_traces),
        'sr_trace': np.nanmean(np.array(sr_traces), axis=0),
        'bt_trace': np.nanmean(np.array(bt_traces), axis=0),
    }


def bootstrap_ci(mat, n_boot=1000, seed=0):
    mat = np.array(mat)
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    if len(mat) < 2:
        return np.nanmean(mat, axis=0), np.full(mat.shape[1], np.nan), np.full(mat.shape[1], np.nan)
    rng = np.random.default_rng(seed)
    grand = np.nanmean(mat, axis=0)
    boots = np.zeros((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, len(mat), size=len(mat))
        boots[b] = np.nanmean(mat[idx], axis=0)
    return grand, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def rise_10_90(trace, t=TGRID):
    if np.all(np.isnan(trace)):
        return np.nan, np.nan, np.nan
    pk = int(np.nanargmax(trace))
    pk_val = trace[pk]
    if pk_val <= 0:
        return np.nan, np.nan, np.nan
    ten = 0.1 * pk_val; ninety = 0.9 * pk_val
    i10 = pk
    while i10 > 0 and trace[i10] > ten:
        i10 -= 1
    i90 = pk
    while i90 > 0 and trace[i90] > ninety:
        i90 -= 1
    return float(t[i10]), float(t[i90]), float(t[pk])


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
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")
    n_events = sum(r['n_events'] for r in results)
    print(f"Total events: {n_events}")

    sr_mat = np.array([r['sr_trace'] for r in results])
    bt_mat = np.array([r['bt_trace'] for r in results])
    sr_grand, sr_lo, sr_hi = bootstrap_ci(sr_mat)
    bt_grand, bt_lo, bt_hi = bootstrap_ci(bt_mat)

    sr_ten, sr_ninety, sr_peak = rise_10_90(sr_grand)
    bt_ten, bt_ninety, bt_peak = rise_10_90(bt_grand)

    def val_at(trace, t_val):
        idx = int(np.argmin(np.abs(TGRID - t_val)))
        return float(trace[idx])

    print(f"\n=== {args.cohort} composite · peri-event SR vs β timing ===")
    print(f"(envelope B31: SR peak +1 s, β peak +3 s; shape r = 0.956)")
    print(f"  SR band [7.0-8.2]:  peak @ {sr_peak:+.1f}s  (log {val_at(sr_grand, sr_peak):+.3f})  10%@{sr_ten:+.1f}s  90%@{sr_ninety:+.1f}s")
    print(f"  β band [19-21]:     peak @ {bt_peak:+.1f}s  (log {val_at(bt_grand, bt_peak):+.3f})  10%@{bt_ten:+.1f}s  90%@{bt_ninety:+.1f}s")
    lag = bt_peak - sr_peak
    print(f"  Δ peak (β − SR) = {lag:+.1f} s  →", end=' ')
    if abs(lag) <= 0.5:
        print("CONCURRENT engagement")
    elif lag > 0:
        print(f"β LAGS SR by {lag:.1f} s")
    else:
        print(f"β LEADS SR by {-lag:.1f} s")

    r_corr = np.corrcoef(sr_grand, bt_grand)[0, 1]
    print(f"\n  Pearson r(SR_grand, β_grand) across TGRID: {r_corr:+.3f}")
    peak_ratio_sr = 10 ** val_at(sr_grand, sr_peak)
    peak_ratio_bt = 10 ** val_at(bt_grand, bt_peak)
    print(f"  SR peak amplitude: {peak_ratio_sr:.2f}×   β peak amplitude: {peak_ratio_bt:.2f}×")

    # Save CSV + plot
    pd.DataFrame({'t': TGRID,
                   'sr_grand': sr_grand, 'sr_lo': sr_lo, 'sr_hi': sr_hi,
                   'bt_grand': bt_grand, 'bt_lo': bt_lo, 'bt_hi': bt_hi}).to_csv(
        os.path.join(out_dir, 'peri_event_sr_vs_beta.csv'), index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(TGRID, 10 ** sr_grand, color='#8c1a1a', lw=2.2,
            label=f'SR band [7.0–8.2 Hz]  peak {sr_peak:+.1f}s')
    ax.fill_between(TGRID, 10 ** sr_lo, 10 ** sr_hi, color='#8c1a1a', alpha=0.22)
    ax.plot(TGRID, 10 ** bt_grand, color='#2166ac', lw=2.2,
            label=f'β band [19–21 Hz]  peak {bt_peak:+.1f}s')
    ax.fill_between(TGRID, 10 ** bt_lo, 10 ** bt_hi, color='#2166ac', alpha=0.22)
    ax.axhline(1.0, color='k', lw=0.7)
    ax.axvline(0, color='k', ls='--', lw=0.7)
    ax.set_xlabel('time rel. t0_net (s)')
    ax.set_ylabel('band power / baseline (×)')
    ax.set_title(f'B31 · SR vs β peri-event · {args.cohort} composite v2\n'
                 f'Δ peak = {lag:+.1f} s  ·  shape r = {r_corr:+.3f}  ·  n = {len(results)} subjects, {n_events} events')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'peri_event_sr_vs_beta.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/peri_event_sr_vs_beta.png")


if __name__ == '__main__':
    main()
