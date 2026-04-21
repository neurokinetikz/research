#!/usr/bin/env python3
"""
B29 — β-peak (~20 Hz) event-level covariates.

Analogue of B23 but for the 20 Hz peak. For every LEMON event compute the
β peak (argmax of event/baseline ratio in [16, 24] Hz) and its amplitude.
Correlate with event-level properties (template_rho, peak_S, etc.):

  (a) Does template_rho predict β-peak attractor proximity |β − 20.0|
      the way it predicts SR attractor proximity |α − 7.83| (B23: ρ = −0.38)?
  (b) Is the β-peak amplitude driven by the same quality axes as the SR peak?
  (c) Within each event, do the SR-peak and β-peak positions co-vary?
"""
from __future__ import annotations
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
from scripts.run_sie_extraction import load_lemon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality', 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
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


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 1:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
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
    f_sr = freqs[sr_mask]
    f_bt = freqs[bt_mask]

    # Baselines
    base_sr = []
    base_bt = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)
        base_sr.append(psd[sr_mask])
        base_bt.append(psd[bt_mask])
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
            'sr_peak_dist': float(abs(sr_pf - SCHUMANN_F)),
            'bt_peak_f': bt_pf, 'bt_peak_r': bt_pr,
            'bt_peak_dist_20': float(abs(bt_pf - BETA_F)),
            'bt_peak_dist_20_5': float(abs(bt_pf - 20.5)),
        })
    return rows


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 1)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        out = pool.map(process_subject, tasks)
    rows = [r for sub_rows in out if sub_rows for r in sub_rows]
    ev = pd.DataFrame(rows)
    print(f"Events scored: {len(ev)}")

    # Merge template_rho
    try:
        qual = pd.read_csv(QUALITY_CSV)
        ev['key'] = ev.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        qual['key'] = qual.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        ev = ev.merge(qual[['key', 'template_rho', 'peak_S', 'spatial_coh',
                              'baseline_calm', 'S_fwhm_s']],
                       on='key', how='left').drop(columns=['key'])
        print(f"Events with template_rho: {ev['template_rho'].notna().sum()}/{len(ev)}")
    except Exception as e:
        print(f"Quality merge failed: {e}")

    ev.to_csv(os.path.join(OUT_DIR, 'per_event_beta_peak.csv'), index=False)

    print(f"\n=== β peak distribution (event level) ===")
    print(f"  mean {ev['bt_peak_f'].mean():.3f}  median {ev['bt_peak_f'].median():.3f}  "
          f"SD {ev['bt_peak_f'].std():.3f}")
    print(f"  |β−20.0| median {ev['bt_peak_dist_20'].median():.3f}  mean {ev['bt_peak_dist_20'].mean():.3f}")
    print(f"  |β−20.5| median {ev['bt_peak_dist_20_5'].median():.3f}")

    # Pooled + within-subject Spearman correlations
    print(f"\n=== Pooled Spearman corrs (event-level, n = {len(ev)}) ===")
    targets = [('bt_peak_f', 'β peak freq'),
                ('bt_peak_dist_20', '|β − 20.0|'),
                ('bt_peak_dist_20_5', '|β − 20.5|'),
                ('bt_peak_r', 'β peak amplitude')]
    cov_cols = ['template_rho', 'peak_S', 'S_fwhm_s', 'baseline_calm',
                 'spatial_coh']
    print(f"{'covariate':<16}" + ''.join(f"{t[1]:>22}" for t in targets))
    for c in cov_cols:
        line = f"{c:<16}"
        for tc, _ in targets:
            sub = ev.dropna(subset=[c, tc])
            if len(sub) < 10:
                line += f"{'—':>22}"; continue
            rho, p = spearmanr(sub[c], sub[tc])
            line += f" ρ={rho:+.3f} p={p:.2g}"
        print(line)

    # Within-subject version (removes between-subject confounds)
    print(f"\n=== Within-subject Spearman corrs ===")
    ev_s = ev.copy()
    def wz(x):
        return (x - x.mean()) / (x.std(ddof=1) + 1e-12) if len(x) >= 2 else 0
    for tc, _ in targets:
        ev_s[tc + '_wz'] = ev.groupby('subject_id')[tc].transform(wz)
    print(f"{'covariate':<16}" + ''.join(f"{t[1]:>22}" for t in targets))
    for c in cov_cols:
        c_wz = ev.groupby('subject_id')[c].transform(wz)
        line = f"{c:<16}"
        for tc, _ in targets:
            m = ev_s[tc + '_wz'].notna() & c_wz.notna()
            if m.sum() < 10:
                line += f"{'—':>22}"; continue
            rho, p = spearmanr(c_wz[m], ev_s.loc[m, tc + '_wz'])
            line += f" ρ={rho:+.3f} p={p:.2g}"
        print(line)

    # Cross-peak within-event: does β-peak position correlate with SR-peak position?
    rho_pos, p_pos = spearmanr(ev['sr_peak_f'], ev['bt_peak_f'])
    rho_amp, p_amp = spearmanr(ev['sr_peak_r'], ev['bt_peak_r'])
    rho_dist_sr, p_dist_sr = spearmanr(ev['sr_peak_dist'], ev['bt_peak_dist_20'])
    print(f"\n=== Within-event SR × β cross-correlation ===")
    print(f"  peak freq:  ρ(SR_f, β_f) = {rho_pos:+.3f}  p={p_pos:.2g}")
    print(f"  amplitude: ρ(SR_amp, β_amp) = {rho_amp:+.3f}  p={p_amp:.2g}")
    print(f"  distance-to-attractor:  ρ(|SR−7.83|, |β−20.0|) = {rho_dist_sr:+.3f}  p={p_dist_sr:.2g}")

    # Stratify by template_rho quartile — plot SR and β peak distributions
    if 'template_rho' in ev.columns and ev['template_rho'].notna().sum() >= 20:
        ev_q = ev.dropna(subset=['template_rho']).copy()
        ev_q['rho_q'] = pd.qcut(ev_q['template_rho'], 4,
                                  labels=['Q1', 'Q2', 'Q3', 'Q4'])
        print(f"\n=== β and SR peak by template_rho quartile ===")
        print(f"{'q':<4}{'n':>5}{'SR mean':>10}{'SR dist':>10}{'β mean':>10}{'β dist20':>10}")
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            sub = ev_q[ev_q['rho_q'] == q]
            print(f"{q:<4}{len(sub):>5}"
                  f"{sub['sr_peak_f'].median():>10.3f}"
                  f"{sub['sr_peak_dist'].median():>10.3f}"
                  f"{sub['bt_peak_f'].median():>10.3f}"
                  f"{sub['bt_peak_dist_20'].median():>10.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    # β distance vs template_rho
    ax = axes[0]
    sub = ev.dropna(subset=['template_rho', 'bt_peak_dist_20'])
    ax.scatter(sub['template_rho'], sub['bt_peak_dist_20'], s=8, alpha=0.35,
                color='steelblue')
    rho, p = spearmanr(sub['template_rho'], sub['bt_peak_dist_20'])
    ax.set_xlabel('template_rho')
    ax.set_ylabel('|β peak − 20.0|  (Hz)')
    ax.set_title(f'β-attractor proximity vs morphology  ρ={rho:+.2f} p={p:.2g}')
    ax.grid(alpha=0.3)

    # SR distance vs template_rho (B23 reference)
    ax = axes[1]
    sub = ev.dropna(subset=['template_rho', 'sr_peak_dist'])
    ax.scatter(sub['template_rho'], sub['sr_peak_dist'], s=8, alpha=0.35,
                color='firebrick')
    rho, p = spearmanr(sub['template_rho'], sub['sr_peak_dist'])
    ax.set_xlabel('template_rho')
    ax.set_ylabel('|SR peak − 7.83|  (Hz)')
    ax.set_title(f'SR-attractor proximity vs morphology  ρ={rho:+.2f} p={p:.2g}  (B23)')
    ax.grid(alpha=0.3)

    # SR distance vs β distance (shared mechanism?)
    ax = axes[2]
    sub = ev.dropna(subset=['sr_peak_dist', 'bt_peak_dist_20'])
    ax.scatter(sub['sr_peak_dist'], sub['bt_peak_dist_20'], s=8, alpha=0.35,
                color='purple')
    rho, p = spearmanr(sub['sr_peak_dist'], sub['bt_peak_dist_20'])
    ax.set_xlabel('|SR peak − 7.83|  (Hz)')
    ax.set_ylabel('|β peak − 20.0|  (Hz)')
    ax.set_title(f'SR × β attractor proximity  ρ={rho:+.2f} p={p:.2g}')
    ax.grid(alpha=0.3)

    plt.suptitle('B29 — β (~20 Hz) event peak stratified by event morphology',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'beta_peak_covariates.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/beta_peak_covariates.png")


if __name__ == '__main__':
    main()
