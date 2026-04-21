#!/usr/bin/env python3
"""
B23 — Event-level peak frequency vs event-property covariates.

For every LEMON event: compute the single-event peak frequency (from a 4-s
window at t0_net + 1 s, ratioed to subject baseline). Then correlate with
event-level properties:

  - template_rho             (shape fidelity)
  - peak_S                   (composite-detector amplitude)
  - spatial_coh              (channel nadir simultaneity)
  - S_fwhm_s                 (composite duration)
  - baseline_calm            (pre-event noise floor)
  - sr1_z_max                (envelope z-peak amplitude, from events CSV)
  - duration_s               (Stage 1 window duration)

Also: |peak_f − 7.83| as "distance from Schumann" — which event properties
predict events that land closer to 7.83 Hz?

Two questions:
  (a) Is the peak-frequency jitter (SD 0.62 Hz, B22) structured by event
      properties, or is it approximately event-independent noise?
  (b) Do high-quality events (high template_rho, high peak_S) land closer to
      7.83 Hz, or is peak frequency orthogonal to quality?
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
ZOOM_LO, ZOOM_HI = 6.5, 9.0
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
    mask = (freqs >= ZOOM_LO) & (freqs <= ZOOM_HI)
    f_band = freqs[mask]

    # Baseline PSD
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


def partial_out_subject(df, cols):
    """Z-score each column within subject to remove subject-level mean.
    This isolates within-subject correlations from between-subject ones."""
    out = df.copy()
    for c in cols:
        out[c + '_wz'] = df.groupby('subject_id')[c].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12))
    return out


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

    # Merge quality
    try:
        qual = pd.read_csv(QUALITY_CSV)
        ev['key'] = ev.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        qual['key'] = qual.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        ev = ev.merge(qual[['key', 'template_rho', 'peak_S',
                              'spatial_coh', 'baseline_calm', 'S_fwhm_s']],
                       on='key', how='left')
        ev = ev.drop(columns=['key'])
        print(f"Events merged with quality: {ev['template_rho'].notna().sum()}/{len(ev)}")
    except Exception as e:
        print(f"Quality merge failed: {e}")

    ev.to_csv(os.path.join(OUT_DIR, 'per_event_peak_covariates.csv'), index=False)

    # Correlations with event_peak_f and with dist_to_schumann
    cov_cols = ['template_rho', 'peak_S', 'spatial_coh', 'baseline_calm',
                 'S_fwhm_s', 'sr1_z_max', 'duration_s', 'HSI', 'sr_score',
                 'event_peak_ratio']
    cov_cols = [c for c in cov_cols if c in ev.columns]
    print(f"\n=== Pooled Spearman correlations (n = {len(ev)}) ===")
    print(f"{'covariate':<20} {'vs event_peak_f':>18}  {'vs |peak_f − 7.83|':>20}")
    for c in cov_cols:
        s = ev.dropna(subset=[c, 'event_peak_f'])
        if len(s) < 10:
            continue
        rho_f, p_f = spearmanr(s[c], s['event_peak_f'])
        rho_d, p_d = spearmanr(s[c], s['event_peak_dist_schumann'])
        print(f"{c:<20} ρ={rho_f:+.3f} p={p_f:.2g}   ρ={rho_d:+.3f} p={p_d:.2g}")

    # Within-subject correlations (z-scored per subject)
    print(f"\n=== Within-subject Spearman correlations (removes between-subject effects) ===")
    print(f"{'covariate':<20} {'ρ(within vs peak_f_wz)':>25}")
    ev_sub = ev.copy()
    ev_sub['event_peak_f_wz'] = ev.groupby('subject_id')['event_peak_f'].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12) if len(x) >= 2 else 0)
    ev_sub['event_peak_dist_wz'] = ev.groupby('subject_id')['event_peak_dist_schumann'].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12) if len(x) >= 2 else 0)
    for c in cov_cols:
        s = ev_sub.dropna(subset=[c, 'event_peak_f_wz'])
        if len(s) < 10:
            continue
        c_wz = s.groupby('subject_id')[c].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12) if len(x) >= 2 else 0)
        rho_f, p_f = spearmanr(c_wz, s['event_peak_f_wz'])
        rho_d, p_d = spearmanr(c_wz, s['event_peak_dist_wz'])
        print(f"{c:<20} ρ_peakf={rho_f:+.3f} p={p_f:.2g}   ρ_dist={rho_d:+.3f} p={p_d:.2g}")

    # Plot scatter matrix for key covariates vs peak_f
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plot_cols = ['template_rho', 'peak_S', 'sr1_z_max',
                  'baseline_calm', 'S_fwhm_s', 'event_peak_ratio']
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

    plt.suptitle('B23 — Event-level peak frequency vs event properties (pooled)',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'event_peak_covariates.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/event_peak_covariates.png")


if __name__ == '__main__':
    main()
