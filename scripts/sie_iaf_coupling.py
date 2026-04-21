#!/usr/bin/env python3
"""
B20 — IAF-coupling test for the ignition peak.

Per subject, compute:
  IAF: individual alpha frequency = argmax of PSD in [7.0, 13.0] Hz from full-
       recording Welch (8-s windows, 75% overlap — 0.125 Hz native resolution).
  ignition_peak: argmax of the per-subject event-time-PSD / baseline ratio in
                 [6.5, 9.0] Hz, using 4-s windows centered at t0_net + 1 s for
                 Q4 events only (same as B19).

Correlate IAF vs ignition_peak across subjects. Three possible outcomes:

  (a) Strong IAF coupling (ρ → 1): ignition peak = subject's lower-alpha edge,
      NOT a fixed frequency. Geophysical framing collapses.
  (b) Weak/no IAF coupling (ρ → 0) with mean near 7.83: ignition peak is fixed
      at ~7.83 Hz across subjects regardless of IAF. Supports fixed-frequency
      claim.
  (c) Intermediate coupling (ρ ~ 0.3-0.6): partial IAF dependence; both a fixed
      component and an IAF-scaled component.

Also reports ignition_peak − IAF distribution and tests against a null of
"ignition peak = 7.83 Hz regardless of IAF".
"""
from __future__ import annotations
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
from scripts.run_sie_extraction import load_lemon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality', 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

# Event-time PSD params (same as B19)
EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
ZOOM_LO, ZOOM_HI = 6.5, 9.0

# IAF PSD params (longer window for better resolution)
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
    """Argmax with parabolic refinement in bin units."""
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
        return np.nan, np.nan
    grand = np.nanmedian(np.array(psds), axis=0)
    f_band = freqs[mask]
    iaf = parabolic_peak(grand, f_band)
    return iaf, float(np.max(grand))


def compute_ignition_peak(y, fs, t_events_q4):
    """Average 4-s event-time PSDs over Q4 events; baseline = median across all
    4-s sliding windows; return peak of event/baseline ratio in [6.5, 9.0]."""
    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))  # 1-s sliding hop
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= ZOOM_LO) & (freqs <= ZOOM_HI)
    f_band = freqs[mask]

    # Baseline
    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)[mask]
        base_rows.append(psd)
    if len(base_rows) < 10:
        return np.nan, np.nan
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    # Event windows
    ev_rows = []
    for t0 in t_events_q4:
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
    peak_f = parabolic_peak(ratio, f_band)
    peak_r = float(ratio[int(np.argmax(ratio))])
    return peak_f, peak_r


def process_subject(args):
    sub_id, df_sub = args
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    iaf, iaf_pow = compute_iaf(y, fs)
    q4_times = df_sub.loc[df_sub['rho_q'] == 'Q4', 't0_net'].astype(float).values
    if len(q4_times) == 0:
        return None
    ign_f, ign_r = compute_ignition_peak(y, fs, q4_times)

    return {
        'subject_id': sub_id,
        'iaf_hz': iaf,
        'iaf_peak_power': iaf_pow,
        'ignition_peak_hz': ign_f,
        'ignition_peak_ratio': ign_r,
        'n_q4': int(len(q4_times)),
        'gap_hz': (ign_f - iaf) if (np.isfinite(iaf) and np.isfinite(ign_f)) else np.nan,
    }


def main():
    qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    qual_q4 = qual[qual['rho_q'] == 'Q4']
    tasks = [(sid, qual[qual['subject_id'] == sid])
              for sid in qual_q4['subject_id'].unique()]
    print(f"Subjects with Q4 events: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'iaf_vs_ignition_peak.csv'), index=False)
    print(f"Successful: {len(df)}")

    good = df.dropna(subset=['iaf_hz', 'ignition_peak_hz']).copy()
    print(f"With both IAF and ignition peak: {len(good)}")

    print(f"\n=== IAF distribution ===")
    print(f"  mean {good['iaf_hz'].mean():.2f} Hz   median {good['iaf_hz'].median():.2f}   std {good['iaf_hz'].std():.2f}")
    print(f"  5-95 %: [{good['iaf_hz'].quantile(0.05):.2f}, {good['iaf_hz'].quantile(0.95):.2f}] Hz")

    print(f"\n=== Ignition peak distribution ===")
    print(f"  mean {good['ignition_peak_hz'].mean():.2f}   median {good['ignition_peak_hz'].median():.2f}   std {good['ignition_peak_hz'].std():.2f}")

    print(f"\n=== Gap (ignition_peak − IAF) distribution ===")
    print(f"  mean {good['gap_hz'].mean():+.2f} Hz   median {good['gap_hz'].median():+.2f}   std {good['gap_hz'].std():.2f}")
    print(f"  fraction of subjects with gap < 0 (ignition below IAF): {(good['gap_hz'] < 0).mean()*100:.0f}%")

    rho, p_sp = spearmanr(good['iaf_hz'], good['ignition_peak_hz'])
    r, p_pr = pearsonr(good['iaf_hz'], good['ignition_peak_hz'])
    print(f"\n=== IAF × ignition_peak correlation ===")
    print(f"  Spearman ρ = {rho:.3f}   p = {p_sp:.3g}")
    print(f"  Pearson  r = {r:.3f}   p = {p_pr:.3g}")

    # What would each hypothesis predict?
    # H1 (IAF lock): ignition_peak = IAF → slope 1, intercept 0
    # H2 (Schumann fixed): ignition_peak = 7.83 regardless of IAF → slope 0
    # Fit linear regression ignition = slope * IAF + intercept
    x = good['iaf_hz'].values
    y = good['ignition_peak_hz'].values
    slope, intercept = np.polyfit(x, y, 1)
    print(f"\n=== OLS fit: ignition_peak = {slope:.3f} × IAF + {intercept:.3f} ===")
    print(f"  H1 (IAF lock) predicts slope=1, intercept=0")
    print(f"  H2 (Schumann fixed at 7.83) predicts slope=0, intercept=7.83")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(good['iaf_hz'], good['ignition_peak_hz'], s=30, alpha=0.6,
               color='steelblue', edgecolor='k', lw=0.3)
    iaf_rng = np.array([good['iaf_hz'].min() - 0.5, good['iaf_hz'].max() + 0.5])
    ax.plot(iaf_rng, iaf_rng, 'k--', lw=1, label='H1: ignition = IAF')
    ax.axhline(SCHUMANN_F, color='green', lw=1.2, ls=':',
                label=f'H2: fixed {SCHUMANN_F} Hz')
    ax.plot(iaf_rng, slope * iaf_rng + intercept, color='red', lw=1.5,
             label=f'OLS: {slope:.2f}×IAF + {intercept:.2f}')
    ax.set_xlabel('IAF (Hz)')
    ax.set_ylabel('ignition peak (Hz)')
    ax.set_title(f'IAF vs ignition peak · ρ={rho:.2f} p={p_sp:.2g} n={len(good)}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(good['gap_hz'], bins=30, color='firebrick', edgecolor='k',
             lw=0.3, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8, label='0 (IAF lock)')
    ax.axvline(good['gap_hz'].mean(), color='blue', ls='--', lw=1.5,
                label=f"mean {good['gap_hz'].mean():+.2f} Hz")
    ax.set_xlabel('ignition peak − IAF (Hz)')
    ax.set_ylabel('subjects')
    ax.set_title('Gap distribution')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle('B20 — IAF-coupling test for the ignition peak',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'iaf_vs_ignition_peak.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/iaf_vs_ignition_peak.png")


if __name__ == '__main__':
    main()
