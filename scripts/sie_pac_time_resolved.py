#!/usr/bin/env python3
"""
B36 — Time-resolved PAC after ignition events.

B35 measured PAC in a single 6-s window at t0+1s and found null. But the
discovery paper reported PAC INCREASING in time after ignitions, so the
onset window may miss a post-event buildup. This script computes the
Tort MI comodulogram in six consecutive non-overlapping 6-s windows:

  centers: -6, 0 (= t0+1), +6, +12, +18, +24 s
  width:   6 s each

For each window, computes the full 13 × 13 PAC grid. Per-subject mean
across Q4 events, then grand-average across subjects.

Outputs:
  - Comodulogram series (6 maps, one per time window)
  - Δ MI from baseline (the −6 s window) at each post-event window
  - Time courses for a few landmark cells: θ-γ, α-γ, SR1-γ, SR1-SR3
"""
from __future__ import annotations
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
from scripts.run_sie_extraction import load_lemon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality', 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

PHASE_FREQS = np.arange(2.0, 14.5, 1.0)
PHASE_BW = 1.0
AMP_FREQS = np.arange(17.5, 80, 5.0)
AMP_BW = 2.5

WIN_SEC = 6.0
# Window centers relative to t0_net (event onset in detector)
WIN_CENTERS = np.array([-6.0, 0.0, 6.0, 12.0, 18.0, 24.0])
N_PHASE_BINS = 18


def bandpass(x, fs, lo, hi, order=4):
    ny = 0.5 * fs
    if hi >= ny:
        hi = ny - 1e-3
    if lo <= 0:
        lo = 0.1
    if lo >= hi:
        return np.zeros_like(x)
    b, a = signal.butter(order, [lo / ny, hi / ny], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


def tort_mi(phase, amp, n_bins=N_PHASE_BINS):
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.digitize(phase, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_amp = np.array([np.nanmean(amp[idx == k]) if np.sum(idx == k) > 0 else np.nan
                          for k in range(n_bins)])
    if np.any(np.isnan(mean_amp)) or np.sum(mean_amp) <= 0:
        return np.nan
    p = mean_amp / np.sum(mean_amp)
    kl = np.nansum(p * np.log(p * n_bins + 1e-12))
    return float(kl / np.log(n_bins))


def comodulogram(y_win, fs):
    ny = 0.5 * fs
    out = np.full((len(PHASE_FREQS), len(AMP_FREQS)), np.nan)
    amp_envs = []
    for af in AMP_FREQS:
        if af + AMP_BW >= ny:
            amp_envs.append(None)
            continue
        f = bandpass(y_win, fs, af - AMP_BW, af + AMP_BW)
        amp_envs.append(np.abs(signal.hilbert(f)))
    for i, pf in enumerate(PHASE_FREQS):
        f = bandpass(y_win, fs, pf - PHASE_BW / 2, pf + PHASE_BW / 2)
        phase = np.angle(signal.hilbert(f))
        for j, af in enumerate(AMP_FREQS):
            if amp_envs[j] is None:
                continue
            if af - AMP_BW <= pf + PHASE_BW / 2 + 1:
                continue
            out[i, j] = tort_mi(phase, amp_envs[j])
    return out


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    try:
        qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1','Q2','Q3','Q4'])
        q4 = qual[(qual['subject_id'] == sub_id) & (qual['rho_q'] == 'Q4')]
        q4_times = set(q4['t0_net'].round(3).values)
        events['t0_round'] = events['t0_net'].round(3)
        events = events[events['t0_round'].isin(q4_times)]
    except Exception:
        pass
    if len(events) < 1:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 160:
        return None
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    nperseg = int(round(WIN_SEC * fs))
    t_end = raw.times[-1]

    # For each event, compute comodulogram at each WIN_CENTER offset
    maps_per_win = [[] for _ in WIN_CENTERS]  # length W lists of (n_phi, n_amp)
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        for wi, wc in enumerate(WIN_CENTERS):
            tc = t0 + wc
            i0 = int(round((tc - WIN_SEC / 2) * fs))
            i1 = i0 + nperseg
            if i0 < 0 or i1 > len(y):
                continue
            m = comodulogram(y[i0:i1], fs)
            maps_per_win[wi].append(m)

    # Each window must have at least 1 event
    if any(len(m) == 0 for m in maps_per_win):
        return None
    per_win = [np.nanmean(np.array(m), axis=0) for m in maps_per_win]
    return {
        'subject_id': sub_id,
        'n_events': int(len(events)),
        'per_win': per_win,   # list of (n_phi, n_amp)
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects: {len(tasks)}  windows: {WIN_CENTERS.tolist()}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    # Stack per-window grand averages: shape (n_win, n_phi, n_amp)
    stack = np.array([np.array(r['per_win']) for r in results])  # (n_sub, n_win, phi, amp)
    grand = np.nanmean(stack, axis=0)  # (n_win, phi, amp)

    # Δ from baseline (first window, -6s before t0)
    base = grand[0]
    dmap = grand - base[None, :, :]

    # Landmark cells to track over time
    def idx_ph(f):
        return int(np.argmin(np.abs(PHASE_FREQS - f)))
    def idx_amp(f):
        return int(np.argmin(np.abs(AMP_FREQS - f)))

    cells = [
        ('θ4-γ30',  4.0, 32.5),
        ('θ6-γ30',  6.0, 32.5),
        ('θ7-γ40',  7.0, 42.5),
        ('SR1-γ30', 8.0, 32.5),
        ('SR1-γ40', 8.0, 42.5),
        ('α10-γ40', 10.0, 42.5),
        ('SR1-SR3', 8.0, 20.0),
    ]
    print(f"\n=== Time-resolved MI at landmark cells (×10⁻⁴) ===")
    print(f"{'cell':<10} " + '  '.join(f"t={wc:+.0f}s" for wc in WIN_CENTERS))
    for name, pf, af in cells:
        i = idx_ph(pf); j = idx_amp(af)
        if i is None or j is None:
            continue
        row = [grand[w, i, j] for w in range(len(WIN_CENTERS))]
        print(f"{name:<10} " + '  '.join(f"{1e4*v:>+6.2f}" for v in row))

    print(f"\n=== Δ MI from baseline (t=-6s) at landmark cells (×10⁻⁴) ===")
    print(f"{'cell':<10} " + '  '.join(f"t={wc:+.0f}s" for wc in WIN_CENTERS))
    for name, pf, af in cells:
        i = idx_ph(pf); j = idx_amp(af)
        row = [dmap[w, i, j] for w in range(len(WIN_CENTERS))]
        print(f"{name:<10} " + '  '.join(f"{1e4*v:>+6.2f}" for v in row))

    # Per-cell Wilcoxon: is later window > baseline across subjects?
    print(f"\n=== Cells with largest positive Δ MI at t=+24s ===")
    final_dmap = stack[:, -1, :, :] - stack[:, 0, :, :]
    mean_dmap = np.nanmean(final_dmap, axis=0)
    flat = np.argsort(-mean_dmap.flatten())[:10]
    for idx in flat:
        i, j = np.unravel_index(idx, mean_dmap.shape)
        d = final_dmap[:, i, j]
        d = d[np.isfinite(d)]
        if len(d) > 5 and np.any(d != 0):
            _, p = wilcoxon(d)
        else:
            p = np.nan
        print(f"  phase {PHASE_FREQS[i]:>5.1f} × amp {AMP_FREQS[j]:>5.1f}  "
              f"Δ MI at t=+24s: {1e4*mean_dmap[i,j]:+.3f}×10⁻⁴  p={p:.3g}")

    np.savez(os.path.join(OUT_DIR, 'pac_time_resolved.npz'),
             phase_freqs=PHASE_FREQS, amp_freqs=AMP_FREQS,
             win_centers=WIN_CENTERS, grand=grand, dmap=dmap)

    # Plot: 6-panel comodulogram one per window (diff from baseline)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    vmax = np.nanmax(np.abs(dmap[np.isfinite(dmap)]))
    for wi, ax in enumerate(axes.flatten()):
        im = ax.imshow(dmap[wi], aspect='auto', origin='lower',
                        extent=[AMP_FREQS[0] - 2.5, AMP_FREQS[-1] + 2.5,
                                 PHASE_FREQS[0] - 0.5, PHASE_FREQS[-1] + 0.5],
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('amp freq (Hz)')
        ax.set_ylabel('phase freq (Hz)')
        ax.set_title(f't = {WIN_CENTERS[wi]:+.0f} s  (Δ from t=-6s)')
        plt.colorbar(im, ax=ax, label='Δ MI', shrink=0.75)
        ax.axhline(7.82, color='green', ls='--', lw=0.6, alpha=0.6)
        ax.axhline(4.0,  color='gray',  ls=':',  lw=0.5, alpha=0.5)

    plt.suptitle(f'B36 — Time-resolved PAC Δ from baseline (t=-6s) · LEMON Q4 (n={len(results)})',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pac_time_resolved.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # Plot: time courses of landmark cells
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cells)))
    for c, (name, pf, af) in enumerate(cells):
        i = idx_ph(pf); j = idx_amp(af)
        trace = grand[:, i, j]
        ax.plot(WIN_CENTERS, 1e4 * trace, 'o-', color=colors[c],
                 label=f'{name} ({pf}Hz→{af}Hz)')
    ax.axvline(0, color='k', ls='--', lw=0.6)
    ax.set_xlabel('window center time rel. t0_net (s)')
    ax.set_ylabel('Tort MI (×10⁻⁴)')
    ax.set_title('Landmark cells: PAC over time relative to ignition onset')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pac_time_resolved_landmarks.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/pac_time_resolved.png")
    print(f"Saved: {OUT_DIR}/pac_time_resolved_landmarks.png")


if __name__ == '__main__':
    main()
