#!/usr/bin/env python3
"""
B18 — Peri-event trajectory in φ-lattice band coordinates.

Uses the spectral-differentiation φ-lattice boundaries (4.70, 7.60, 12.30,
19.90, 32.19 Hz) to represent each 4-s Welch window as a point in 6-D band-
power space:
  B1: [0, 4.70] Hz       (delta/low-theta)
  B2: [4.70, 7.60] Hz    (theta)
  B3: [7.60, 12.30] Hz   (alpha)
  B4: [12.30, 19.90] Hz  (low-beta)
  B5: [19.90, 32.19] Hz  (high-beta)
  B6: [32.19, 45.0] Hz   (gamma)

Also two sub-band probes around the SR range to localize which side of the
theta/alpha φ-lattice boundary the event sits on:
  SR_theta: [7.0, 7.60]  (upper theta, below φ-boundary)
  SR_alpha: [7.60, 8.2]  (lower alpha, above φ-boundary)

Per subject, per event, interpolate log-band-power time courses onto a
[-15, +15] s grid rel to t0_net. Stratify by template_rho quartile (Q1/Q4).
Cohort grand mean with subject-level cluster bootstrap 95% CI.

Tests:
  (a) Is the ignition excursion *confined* to a single φ-lattice band, or does
      it spread to neighbors?
  (b) Does the SR-band boost sit on the theta side, the alpha side, or span the
      φ-lattice theta-alpha boundary?
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
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

WIN_SEC = 4.0
HOP_SEC = 1.0
FREQ_LO, FREQ_HI = 1.0, 45.0
NFFT_MULT = 4

# φ-lattice band edges from scripts/regenerate_combined_with_continuous.py
PHI_EDGES = [1.0, 4.70, 7.60, 12.30, 19.90, 32.19, 45.0]
PHI_LABELS = ['δ/lθ', 'θ', 'α', 'β-lo', 'β-hi', 'γ']

# SR sub-bands straddling the theta/alpha φ-lattice boundary (7.60 Hz)
SR_BANDS = {'SR_θ': (7.0, 7.60), 'SR_α': (7.60, 8.2)}

TGRID = np.arange(-15.0, 15.0 + 0.5, 1.0)


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
    return np.array(t_cent), freqs, np.array(cols).T  # (n_f, n_t)


def band_power_series(freqs, P, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() == 0:
        return np.full(P.shape[1], np.nan)
    return np.nanmean(P[mask, :], axis=0)


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

    t, freqs, P = sliding_welch(y, fs)
    # 6 φ-lattice band power series + 2 SR sub-bands
    bands = {}
    for i in range(len(PHI_LABELS)):
        bands[PHI_LABELS[i]] = band_power_series(freqs, P,
                                                  PHI_EDGES[i], PHI_EDGES[i+1])
    for name, (lo, hi) in SR_BANDS.items():
        bands[name] = band_power_series(freqs, P, lo, hi)

    # Log10-baseline-subtracted (per-subject baseline = median over all time)
    baselined = {}
    for name, bp in bands.items():
        med = np.nanmedian(bp)
        baselined[name] = np.log10(bp + 1e-20) - np.log10(med + 1e-20)

    buckets = {q: {name: [] for name in bands.keys()} for q in ['Q1', 'Q4']}
    for _, ev in df_sub.iterrows():
        t0 = float(ev['t0_net'])
        q = ev['rho_q']
        if q not in ['Q1', 'Q4']:
            continue
        rel = t - t0
        m = (rel >= TGRID[0] - 1) & (rel <= TGRID[-1] + 1)
        if m.sum() == 0:
            continue
        for name, series in baselined.items():
            buckets[q][name].append(
                np.interp(TGRID, rel[m], series[m], left=np.nan, right=np.nan))

    out = {'subject_id': sub_id}
    for q in ['Q1', 'Q4']:
        for name in bands.keys():
            arr = buckets[q][name]
            out[f'{q}_{name}'] = np.nanmean(np.array(arr), axis=0) if arr else None
            out[f'{q}_{name}_n'] = len(arr)
    return out


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


def main():
    qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    tasks = [(sid, g) for sid, g in qual.groupby('subject_id')]
    print(f"Subjects: {len(tasks)}  events: {len(qual)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    def stack(q, name):
        arr = [r[f'{q}_{name}'] for r in results if r[f'{q}_{name}'] is not None]
        return np.array(arr) if arr else np.empty((0, len(TGRID)))

    # Peak log-boost @ t=+1s (approximately Q4's SR peak time)
    peak_idx = int(np.argmin(np.abs(TGRID - 1.0)))
    print(f"\n=== φ-lattice band log-boost at t=+1s (peak of Q4 SR response) ===")
    print(f"{'band':<8} {'Q1 log':<10} {'Q1 ratio':<10} {'Q4 log':<10} {'Q4 ratio':<10}")
    for name in PHI_LABELS + list(SR_BANDS.keys()):
        for q in ['Q1', 'Q4']:
            mat = stack(q, name)
            grand, _, _ = bootstrap_ci(mat)
            v = grand[peak_idx] if np.isfinite(grand[peak_idx]) else np.nan
            ratio = 10 ** v if np.isfinite(v) else np.nan
            if q == 'Q1':
                q1_v, q1_r = v, ratio
        print(f"{name:<8} {q1_v:+.3f}     {q1_r:.2f}×      {v:+.3f}     {ratio:.2f}×")

    # Plot: 2 rows (Q1, Q4), 6 panels (φ-lattice bands), with SR sub-bands overlaid on α panel
    colors_phi = plt.cm.viridis(np.linspace(0.0, 0.9, len(PHI_LABELS)))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for qi, q in enumerate(['Q1', 'Q4']):
        ax = axes[qi]
        for i, name in enumerate(PHI_LABELS):
            mat = stack(q, name)
            grand, lo, hi = bootstrap_ci(mat)
            ratio = 10 ** grand
            ax.plot(TGRID, ratio, color=colors_phi[i], lw=2, label=f'{name}')
            ax.fill_between(TGRID, 10 ** lo, 10 ** hi, color=colors_phi[i], alpha=0.15)
        # SR sub-bands as dashed overlays
        for name, color, ls in [('SR_θ', '#8c2d04', '--'),
                                  ('SR_α', '#d94801', ':')]:
            mat = stack(q, name)
            grand, lo, hi = bootstrap_ci(mat)
            ratio = 10 ** grand
            ax.plot(TGRID, ratio, color=color, lw=2, ls=ls, label=name)
        ax.axhline(1.0, color='k', lw=0.7)
        ax.axvline(0, color='k', ls='--', lw=0.5)
        ax.set_xlabel('time rel. t0_net (s)')
        ax.set_ylabel('band power / baseline (×)')
        ax.set_title(f'template_rho {q} — φ-lattice band trajectories')
        ax.legend(fontsize=8, loc='upper left', ncol=2)
        ax.grid(alpha=0.3)

    plt.suptitle('B18 — Peri-event trajectory in φ-lattice band coordinates',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'phi_lattice_trajectory.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # Compact table figure: bar chart of Q4 peak-boost per band
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    names = PHI_LABELS + list(SR_BANDS.keys())
    ratios = []
    for name in names:
        mat = stack('Q4', name)
        grand, _, _ = bootstrap_ci(mat)
        ratios.append(10 ** grand[peak_idx] if np.isfinite(grand[peak_idx]) else np.nan)
    bar_colors = list(colors_phi) + ['#8c2d04', '#d94801']
    ax.bar(range(len(names)), ratios, color=bar_colors, edgecolor='k', lw=0.5)
    ax.axhline(1.0, color='k', lw=0.7)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel('Q4 peak-boost @ t=+1s (×)')
    ax.set_title('B18 — Q4 ignition excursion per band (peak at t=+1s)')
    for i, r in enumerate(ratios):
        if np.isfinite(r):
            ax.text(i, r + 0.02, f'{r:.2f}×', ha='center', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'phi_lattice_peak_bars.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {OUT_DIR}/phi_lattice_trajectory.png")
    print(f"Saved: {OUT_DIR}/phi_lattice_peak_bars.png")


if __name__ == '__main__':
    main()
