#!/usr/bin/env python3
"""
B23 re-run on composite v2 detector.

Identical analysis to scripts/sie_event_peak_covariates.py; cohort-
parameterized. Reads composite v2 events + composite template_rho.

Per composite event: compute single-event peak freq in [6.5, 9.0] Hz
(4-s window at t0_net + 1.0 s, ratioed to baseline). Correlate against:
  template_rho, sr1_z_max, duration_s, HSI, sr_score, event_peak_ratio
Both pooled (cross-subject) and within-subject (z-scored per subject).

Outputs to outputs/schumann/images/psd_timelapse/<cohort>_composite/.

Usage:
    python scripts/sie_event_peak_covariates_composite.py --cohort lemon
    python scripts/sie_event_peak_covariates_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr
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

    rows = []
    for _, ev in events.iterrows():
        t0_net = float(ev['t0_net'])
        tc = t0_net + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[mask]
        ratio = (psd + 1e-20) / (baseline + 1e-20)
        peak_f = parabolic_peak(ratio, f_band)
        peak_r = float(ratio[int(np.argmax(ratio))])
        rows.append({
            'subject_id': sub_id,
            't0_net': t0_net,
            'event_peak_f': peak_f,
            'event_peak_ratio': peak_r,
            'event_peak_dist_schumann': float(abs(peak_f - SCHUMANN_F)),
            'sr1_z_max': float(ev.get('sr1_z_max', np.nan)),
            'duration_s': float(ev.get('duration_s', np.nan)),
            'HSI': float(ev.get('HSI', np.nan)),
            'sr_score': float(ev.get('sr_score', np.nan)),
        })
    return rows


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
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 1)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} composite  ·  subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        out = pool.map(process_subject, tasks)
    rows = [r for sub_rows in out if sub_rows for r in sub_rows]
    ev = pd.DataFrame(rows)
    print(f"Events scored: {len(ev)}")

    try:
        qual = pd.read_csv(quality_csv)
        ev['key'] = ev.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        qual['key'] = qual.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        # composite quality CSV only has template_rho; merge that
        ev = ev.merge(qual[['key', 'template_rho']], on='key', how='left')
        ev = ev.drop(columns=['key'])
        print(f"Events merged with template_rho: "
              f"{ev['template_rho'].notna().sum()}/{len(ev)}")
    except Exception as e:
        print(f"Quality merge failed: {e}")

    ev.to_csv(os.path.join(out_dir, 'per_event_peak_covariates.csv'), index=False)

    cov_cols = ['template_rho', 'sr1_z_max', 'duration_s', 'HSI', 'sr_score',
                 'event_peak_ratio']
    cov_cols = [c for c in cov_cols if c in ev.columns]
    print(f"\n=== {args.cohort} composite · pooled Spearman correlations (n = {len(ev)}) ===")
    print(f"{'covariate':<20} {'vs event_peak_f':>22}  {'vs |peak_f − 7.83|':>22}")
    pooled_results = {}
    for c in cov_cols:
        s = ev.dropna(subset=[c, 'event_peak_f'])
        if len(s) < 10:
            continue
        rho_f, p_f = spearmanr(s[c], s['event_peak_f'])
        rho_d, p_d = spearmanr(s[c], s['event_peak_dist_schumann'])
        pooled_results[c] = (rho_f, p_f, rho_d, p_d, len(s))
        print(f"{c:<20} ρ={rho_f:+.3f} p={p_f:.2g}   ρ={rho_d:+.3f} p={p_d:.2g}")

    print(f"\n=== {args.cohort} composite · within-subject Spearman (z-scored per subject) ===")
    print(f"{'covariate':<20} {'peak_f_wz':>22}  {'dist_wz':>22}")
    within_results = {}
    ev_wz = ev.copy()
    ev_wz['event_peak_f_wz'] = ev.groupby('subject_id')['event_peak_f'].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12) if len(x) >= 2 else 0)
    ev_wz['event_peak_dist_wz'] = ev.groupby('subject_id')['event_peak_dist_schumann'].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12) if len(x) >= 2 else 0)
    for c in cov_cols:
        s = ev_wz.dropna(subset=[c, 'event_peak_f_wz'])
        if len(s) < 10:
            continue
        c_wz = s.groupby('subject_id')[c].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12) if len(x) >= 2 else 0)
        rho_f, p_f = spearmanr(c_wz, s['event_peak_f_wz'])
        rho_d, p_d = spearmanr(c_wz, s['event_peak_dist_wz'])
        within_results[c] = (rho_f, p_f, rho_d, p_d, len(s))
        print(f"{c:<20} ρ={rho_f:+.3f} p={p_f:.2g}   ρ={rho_d:+.3f} p={p_d:.2g}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plot_cols = ['template_rho', 'sr1_z_max', 'HSI', 'sr_score',
                  'duration_s', 'event_peak_ratio']
    plot_cols = [c for c in plot_cols if c in ev.columns]
    for i, c in enumerate(plot_cols[:6]):
        ax = axes[i // 3, i % 3]
        s = ev.dropna(subset=[c, 'event_peak_f'])
        if len(s) < 10:
            continue
        ax.scatter(s[c], s['event_peak_f'], s=6, alpha=0.35,
                    color='steelblue', edgecolor='none')
        rho, p = spearmanr(s[c], s['event_peak_f'])
        ax.axhline(SCHUMANN_F, color='green', ls='--', lw=0.7)
        ax.set_xlabel(c); ax.set_ylabel('event peak freq (Hz)')
        ax.set_title(f'{c} vs peak_f · ρ={rho:+.2f} p={p:.2g} n={len(s)}')
        ax.set_ylim(ZOOM_LO, ZOOM_HI)
        ax.grid(alpha=0.3)

    plt.suptitle(f'B23 · event-peak vs properties · {args.cohort} composite v2',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'event_peak_covariates.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/event_peak_covariates.png")


if __name__ == '__main__':
    main()
