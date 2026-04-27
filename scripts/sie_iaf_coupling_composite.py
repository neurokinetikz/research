#!/usr/bin/env python3
"""
B20 / B20b re-run on composite v2 detector.

Per-subject IAF vs ignition-peak correlation test:
  IAF       = argmax PSD in [7.0, 13.0] Hz (full-recording 8-s Welch)
  ignition  = argmax [event-PSD / baseline-PSD] in [6.5, 9.0] Hz
              with event PSD = 4-s Welch centered at t0_net + 1.0 s

Uses composite v2 events from exports_sie/<cohort>_composite/ and (optionally)
filters to template_rho Q4 from the composite quality CSV. Reports both
all-events and Q4-only statistics so the reader can compare cleanly.

Tests (for each cohort):
  H1 IAF-lock: slope(ignition | IAF) = 1
  H2 Fixed-f: slope = 0, intercept ≈ 7.83 Hz

Usage:
    python scripts/sie_iaf_coupling_composite.py --cohort lemon
    python scripts/sie_iaf_coupling_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
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
IAF_WIN_SEC = 8.0
IAF_HOP_SEC = 2.0
IAF_NFFT_MULT = 4
IAF_LO, IAF_HI = 7.0, 13.0
SCHUMANN_F = 7.83

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    qual = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events, qual
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events, qual
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events, qual
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events, qual
    if cohort == 'srm':
        return load_srm, {}, events, qual
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events, qual
    if cohort == 'dortmund':
        return load_dortmund, {}, events, qual
    if cohort == 'chbmp':
        return load_chbmp, {}, events, qual
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events, qual
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
    sub_id, events_path, q4_events_only = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 2:
        return None
    # If Q4 filter is specified, events has been pre-filtered in main
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    iaf = compute_iaf(y, fs)
    t_events = events['t0_net'].astype(float).values

    # All-events ignition peak
    ign_f_all, ign_r_all = compute_ignition_peak(y, fs, t_events)

    return {
        'subject_id': sub_id,
        'iaf_hz': iaf,
        'ignition_peak_hz': ign_f_all,
        'ignition_peak_ratio': ign_r_all,
        'n_events': int(len(t_events)),
        'gap_hz': (ign_f_all - iaf) if (np.isfinite(iaf) and np.isfinite(ign_f_all)) else np.nan,
    }


def run_tasks(tasks, workers, loader_name, loader_kw):
    with Pool(workers, initializer=_init_worker,
              initargs=(loader_name, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    return [r for r in results if r is not None]


def summarize(df, label):
    good = df.dropna(subset=['iaf_hz', 'ignition_peak_hz']).copy()
    print(f"\n=== {label} · n with IAF+ignition = {len(good)} ===")
    if len(good) < 5:
        print("  Not enough subjects.")
        return None
    print(f"  IAF      mean {good['iaf_hz'].mean():.2f}   median {good['iaf_hz'].median():.2f}   std {good['iaf_hz'].std():.2f}   5-95% [{good['iaf_hz'].quantile(0.05):.2f}, {good['iaf_hz'].quantile(0.95):.2f}]")
    print(f"  ignit.   mean {good['ignition_peak_hz'].mean():.2f}   median {good['ignition_peak_hz'].median():.2f}   std {good['ignition_peak_hz'].std():.2f}")
    print(f"  gap      mean {good['gap_hz'].mean():+.2f}   median {good['gap_hz'].median():+.2f}   std {good['gap_hz'].std():.2f}   frac<0 {(good['gap_hz']<0).mean()*100:.0f}%")
    rho, p_sp = spearmanr(good['iaf_hz'], good['ignition_peak_hz'])
    r, p_pr = pearsonr(good['iaf_hz'], good['ignition_peak_hz'])
    slope, intercept = np.polyfit(good['iaf_hz'], good['ignition_peak_hz'], 1)
    print(f"  IAF × ignition   Spearman ρ = {rho:+.3f} p={p_sp:.3g}   Pearson r = {r:+.3f} p={p_pr:.3g}")
    print(f"  OLS slope = {slope:+.3f}   intercept = {intercept:+.3f}")
    print(f"  (H1 IAF-lock predicts slope=1; H2 fixed-f {SCHUMANN_F} Hz predicts slope=0 intercept={SCHUMANN_F})")
    return dict(label=label, n=len(good), rho=rho, rho_p=p_sp, r=r, r_p=p_pr,
                slope=slope, intercept=intercept,
                iaf_mean=good['iaf_hz'].mean(), iaf_std=good['iaf_hz'].std(),
                ign_mean=good['ignition_peak_hz'].mean(),
                ign_std=good['ignition_peak_hz'].std(),
                gap_mean=good['gap_hz'].mean(),
                gap_std=good['gap_hz'].std())


def scatter_plot(df, stats, label, fname):
    good = df.dropna(subset=['iaf_hz', 'ignition_peak_hz']).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(good['iaf_hz'], good['ignition_peak_hz'], s=30, alpha=0.6,
                color='steelblue', edgecolor='k', lw=0.3)
    rng = np.array([good['iaf_hz'].min() - 0.5, good['iaf_hz'].max() + 0.5])
    ax.plot(rng, rng, 'k--', lw=1, label='H1: ignition = IAF')
    ax.axhline(SCHUMANN_F, color='green', lw=1.2, ls=':', label=f'H2: fixed {SCHUMANN_F} Hz')
    ax.plot(rng, stats['slope'] * rng + stats['intercept'], color='red', lw=1.5,
             label=f"OLS: {stats['slope']:+.2f}×IAF + {stats['intercept']:+.2f}")
    ax.set_xlabel('IAF (Hz)')
    ax.set_ylabel('ignition peak (Hz)')
    ax.set_title(f"{label}: IAF vs ignition · ρ={stats['rho']:+.2f} p={stats['rho_p']:.2g} n={stats['n']}")
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

    plt.suptitle(f'B20/B20b · composite v2 · {label}', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'psd_timelapse', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]

    # All-events run
    tasks_all = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks_all.append((r['subject_id'], ep, False))
    print(f"Cohort: {args.cohort} composite  ·  subjects with >=3 events: {len(tasks_all)}")
    print(f"Workers: {args.workers}")

    results_all = run_tasks(tasks_all, args.workers, loader.__name__, loader_kw)
    df_all = pd.DataFrame(results_all)
    df_all.to_csv(os.path.join(out_dir, 'iaf_vs_ignition_peak_all.csv'), index=False)
    stats_all = summarize(df_all, f'{args.cohort} composite · all events')
    if stats_all:
        scatter_plot(df_all, stats_all, f'{args.cohort} composite · all events',
                     os.path.join(out_dir, 'iaf_vs_ignition_peak_all.png'))

    # Q4-only run (if quality CSV available)
    if os.path.isfile(quality_csv):
        qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        q4 = qual[qual['rho_q'] == 'Q4']
        # Build a temporary Q4-only events file per subject in the output dir
        q4_dir = os.path.join(out_dir, '_q4_events_tmp')
        os.makedirs(q4_dir, exist_ok=True)
        tasks_q4 = []
        for sid, g in q4.groupby('subject_id'):
            if len(g) < 2:
                continue
            tmp = os.path.join(q4_dir, f'{sid}_sie_events.csv')
            g[['subject_id', 't0_net']].to_csv(tmp, index=False)
            tasks_q4.append((sid, tmp, True))
        print(f"\nQ4-only subjects (with >=2 Q4 events): {len(tasks_q4)}")

        results_q4 = run_tasks(tasks_q4, args.workers, loader.__name__, loader_kw)
        df_q4 = pd.DataFrame(results_q4)
        df_q4.to_csv(os.path.join(out_dir, 'iaf_vs_ignition_peak_q4.csv'), index=False)
        stats_q4 = summarize(df_q4, f'{args.cohort} composite · Q4 only')
        if stats_q4:
            scatter_plot(df_q4, stats_q4, f'{args.cohort} composite · Q4 only',
                         os.path.join(out_dir, 'iaf_vs_ignition_peak_q4.png'))

        # Cleanup temp
        for f in os.listdir(q4_dir):
            os.remove(os.path.join(q4_dir, f))
        os.rmdir(q4_dir)


if __name__ == '__main__':
    main()
