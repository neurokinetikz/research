#!/usr/bin/env python3
"""
B29 re-run on composite v2 detector.

Per composite event: compute β peak in [16, 24] Hz (event/baseline ratio at
t0_net + 1 s), SR peak in [6.5, 9.0] Hz, and distances from 20.0 and 7.83 Hz
attractors. Correlate with event-level template_ρ (composite quality CSV).

Envelope B29 finding: template_ρ × |β − 20| ρ = +0.03 (p = 0.34) — NOT
morphology-structured, unlike SR1 (template_ρ × |α − 7.83| ρ = −0.38 per B23).

Cohort-parameterized.

Usage:
    python scripts/sie_beta_peak_covariates_composite.py --cohort lemon
    python scripts/sie_beta_peak_covariates_composite.py --cohort lemon_EO
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
SR_LO, SR_HI = 6.5, 9.0
BETA_LO, BETA_HI = 16.0, 24.0
SCHUMANN_F = 7.83
BETA_F = 20.0

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
    sr_mask = (freqs >= SR_LO) & (freqs <= SR_HI)
    bt_mask = (freqs >= BETA_LO) & (freqs <= BETA_HI)
    f_sr = freqs[sr_mask]; f_bt = freqs[bt_mask]

    base_sr = []; base_bt = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)
        base_sr.append(psd[sr_mask]); base_bt.append(psd[bt_mask])
    if len(base_sr) < 10:
        return None
    base_sr = np.nanmedian(np.array(base_sr), axis=0)
    base_bt = np.nanmedian(np.array(base_bt), axis=0)

    rows = []
    for _, ev in events.iterrows():
        t0_net = float(ev['t0_net'])
        tc = t0_net + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)
        sr_ratio = (psd[sr_mask] + 1e-20) / (base_sr + 1e-20)
        bt_ratio = (psd[bt_mask] + 1e-20) / (base_bt + 1e-20)
        sr_pf = parabolic_peak(sr_ratio, f_sr)
        sr_pr = float(sr_ratio[int(np.argmax(sr_ratio))])
        bt_pf = parabolic_peak(bt_ratio, f_bt)
        bt_pr = float(bt_ratio[int(np.argmax(bt_ratio))])
        rows.append({
            'subject_id': sub_id, 't0_net': t0_net,
            'sr_peak_f': sr_pf, 'sr_peak_r': sr_pr,
            'sr_peak_dist_783': float(abs(sr_pf - SCHUMANN_F)),
            'bt_peak_f': bt_pf, 'bt_peak_r': bt_pr,
            'bt_peak_dist_20': float(abs(bt_pf - BETA_F)),
            'bt_peak_dist_20_5': float(abs(bt_pf - 20.5)),
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
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        out = pool.map(process_subject, tasks)
    rows = [r for sub_rows in out if sub_rows for r in sub_rows]
    ev = pd.DataFrame(rows)
    print(f"Events scored: {len(ev)}")

    # Merge composite template_ρ
    try:
        qual = pd.read_csv(quality_csv)
        ev['key'] = ev.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        qual['key'] = qual.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        ev = ev.merge(qual[['key', 'template_rho']], on='key', how='left').drop(columns=['key'])
        print(f"Events with template_rho: {ev['template_rho'].notna().sum()}/{len(ev)}")
    except Exception as e:
        print(f"Quality merge failed: {e}")

    ev.to_csv(os.path.join(out_dir, 'per_event_beta_peak.csv'), index=False)

    print(f"\n=== {args.cohort} composite · β peak distribution (event level) ===")
    print(f"  mean {ev['bt_peak_f'].mean():.3f}  median {ev['bt_peak_f'].median():.3f}  "
          f"SD {ev['bt_peak_f'].std():.3f}")
    print(f"  |β − 20.0| median {ev['bt_peak_dist_20'].median():.3f}  mean {ev['bt_peak_dist_20'].mean():.3f}")
    print(f"  |β − 20.5| median {ev['bt_peak_dist_20_5'].median():.3f}")

    # Pooled Spearman correlations
    targets = [('bt_peak_f', 'β peak freq'),
                ('bt_peak_dist_20', '|β − 20.0|'),
                ('bt_peak_dist_20_5', '|β − 20.5|'),
                ('bt_peak_r', 'β peak amplitude')]
    print(f"\n=== {args.cohort} composite · pooled template_ρ correlations ===")
    print(f"(envelope B29: template_ρ × |β − 20| ρ = +0.03 p = 0.34 — NOT morphology-structured)")
    print(f"(envelope B23: template_ρ × |SR − 7.83| ρ = −0.38 p = 10⁻³²)")
    for tc, label in targets:
        sub = ev.dropna(subset=['template_rho', tc])
        if len(sub) < 10:
            continue
        rho, p = spearmanr(sub['template_rho'], sub[tc])
        print(f"  template_rho × {label:22s}  ρ={rho:+.3f}  p={p:.3g}  n={len(sub)}")

    # Cross-check: template_ρ × |SR − 7.83| should reproduce §20 B23 result
    sub = ev.dropna(subset=['template_rho', 'sr_peak_dist_783'])
    if len(sub) >= 10:
        rho, p = spearmanr(sub['template_rho'], sub['sr_peak_dist_783'])
        print(f"\n  [cross-check] template_rho × |SR − 7.83|  ρ={rho:+.3f}  p={p:.3g}  "
              f"(§20 B23: EC ρ=−0.381)")

    # SR-peak × β-peak position within events
    sub = ev.dropna(subset=['sr_peak_f', 'bt_peak_f'])
    if len(sub) >= 10:
        rho, p = spearmanr(sub['sr_peak_f'], sub['bt_peak_f'])
        print(f"\n=== Within-events: SR-peak × β-peak positions ===")
        print(f"  Spearman ρ = {rho:+.3f}  p = {p:.3g}")

    # Plot: β-peak freq distribution + template_ρ scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(ev['bt_peak_f'].dropna(), bins=40, color='slategray', edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(BETA_F, color='red', ls='--', label='β = 20 Hz')
    ax.axvline(ev['bt_peak_f'].median(), color='firebrick', ls=':', label=f"median {ev['bt_peak_f'].median():.2f} Hz")
    ax.set_xlabel('β-peak frequency (Hz)')
    ax.set_ylabel('events')
    ax.set_title(f'β-peak distribution · mean {ev["bt_peak_f"].mean():.2f}  SD {ev["bt_peak_f"].std():.2f}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    sub = ev.dropna(subset=['template_rho', 'bt_peak_dist_20'])
    if len(sub):
        ax.scatter(sub['template_rho'], sub['bt_peak_dist_20'], s=6, alpha=0.3, color='steelblue')
        rho_b, p_b = spearmanr(sub['template_rho'], sub['bt_peak_dist_20'])
        ax.set_title(f'template_ρ × |β − 20|  ρ={rho_b:+.2f} p={p_b:.2g}')
    ax.set_xlabel('template_ρ'); ax.set_ylabel('|β peak − 20.0| (Hz)')
    ax.grid(alpha=0.3)

    ax = axes[2]
    sub = ev.dropna(subset=['template_rho', 'sr_peak_dist_783'])
    if len(sub):
        ax.scatter(sub['template_rho'], sub['sr_peak_dist_783'], s=6, alpha=0.3, color='firebrick')
        rho_s, p_s = spearmanr(sub['template_rho'], sub['sr_peak_dist_783'])
        ax.set_title(f'template_ρ × |SR − 7.83|  ρ={rho_s:+.2f} p={p_s:.2g}  (cross-check)')
    ax.set_xlabel('template_ρ'); ax.set_ylabel('|SR peak − 7.83| (Hz)')
    ax.grid(alpha=0.3)

    plt.suptitle(f'B29 · β-peak event-level covariates · {args.cohort} composite v2',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'beta_peak_covariates.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/beta_peak_covariates.png")


if __name__ == '__main__':
    main()
