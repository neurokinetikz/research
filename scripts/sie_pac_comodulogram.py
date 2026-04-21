#!/usr/bin/env python3
"""
B35 — Phase-amplitude coupling comodulogram during ignition events.

Standard theta/alpha/SR1 phase × γ amplitude PAC using Tort modulation index.
Per LEMON event (template_rho Q4 subset for cleanest signal): compute Tort MI
across a phase frequency grid (2-14 Hz, 1-Hz bands) × amplitude frequency
grid (15-80 Hz, 5-Hz bands). Compare event windows to matched non-event
controls.

Outputs:
  - Event-window mean comodulogram
  - Control-window mean comodulogram
  - Event − Control difference map (event-specific PAC)
  - Per-subject MI at key cells (SR1 phase × γ amp, θ phase × γ amp)
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

# Phase frequencies (center ± 0.5 Hz half-bandwidth)
PHASE_FREQS = np.arange(2.0, 14.5, 1.0)    # 2, 3, 4, ..., 14
PHASE_BW = 1.0
# Amplitude frequencies (center ± 2.5 Hz half-bandwidth)
AMP_FREQS = np.arange(17.5, 80, 5.0)        # 17.5, 22.5, ..., 77.5
AMP_BW = 2.5

# Event window
EVENT_WIN = 6.0     # 6-s window
EVENT_LAG = 1.0     # centered at t0_net + 1 s
MIN_GAP_FROM_EVENT = 30.0
N_PHASE_BINS = 18


def bandpass(x, fs, lo, hi, order=4):
    ny = 0.5 * fs
    if hi >= ny:
        hi = ny - 1e-3
    if lo <= 0:
        lo = 0.1
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
    """Tort MI at every (phase_f, amp_f) combination for a segment."""
    out = np.full((len(PHASE_FREQS), len(AMP_FREQS)), np.nan)
    # Pre-filter amplitude envelopes
    amp_envs = []
    for af in AMP_FREQS:
        filt = bandpass(y_win, fs, af - AMP_BW, af + AMP_BW)
        amp_envs.append(np.abs(signal.hilbert(filt)))
    for i, pf in enumerate(PHASE_FREQS):
        filt = bandpass(y_win, fs, pf - PHASE_BW / 2, pf + PHASE_BW / 2)
        phase = np.angle(signal.hilbert(filt))
        for j, af in enumerate(AMP_FREQS):
            if af - AMP_BW <= pf + PHASE_BW / 2 + 1:
                continue  # Skip overlapping phase/amp bands
            out[i, j] = tort_mi(phase, amp_envs[j])
    return out


def sample_control_times(t_events, n, t_end, seed=0):
    rng = np.random.default_rng(seed)
    lo = EVENT_WIN / 2 + 1
    hi = t_end - EVENT_WIN / 2 - 1
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        t = rng.uniform(lo, hi)
        if len(t_events) == 0 or np.min(np.abs(t - t_events)) >= MIN_GAP_FROM_EVENT:
            out.append(t)
        tries += 1
    return np.array(out)


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    # Restrict to Q4 events via quality CSV
    try:
        qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
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
    t_end = raw.times[-1]
    nperseg = int(round(EVENT_WIN * fs))

    t_events = events['t0_net'].astype(float).values
    t_controls = sample_control_times(t_events, len(t_events), t_end,
                                        seed=abs(hash(sub_id)) % (2**31))

    def score(t_center):
        i0 = int(round((t_center - EVENT_WIN / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            return None
        return comodulogram(y[i0:i1], fs)

    ev_maps = []
    for t0 in t_events:
        m = score(t0 + EVENT_LAG)
        if m is not None:
            ev_maps.append(m)
    ct_maps = []
    for tc in t_controls:
        m = score(tc)
        if m is not None:
            ct_maps.append(m)
    if len(ev_maps) < 1 or len(ct_maps) < 1:
        return None
    return {
        'subject_id': sub_id,
        'n_ev': len(ev_maps),
        'n_ct': len(ct_maps),
        'ev_mean': np.nanmean(np.array(ev_maps), axis=0),
        'ct_mean': np.nanmean(np.array(ct_maps), axis=0),
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects: {len(tasks)}  phase grid: {len(PHASE_FREQS)}  "
          f"amp grid: {len(AMP_FREQS)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    ev_stack = np.array([r['ev_mean'] for r in results])
    ct_stack = np.array([r['ct_mean'] for r in results])
    ev_grand = np.nanmean(ev_stack, axis=0)
    ct_grand = np.nanmean(ct_stack, axis=0)
    diff = ev_grand - ct_grand

    # Cell-wise Wilcoxon event vs control
    n_phi, n_amp = ev_grand.shape
    pvals = np.ones_like(ev_grand)
    for i in range(n_phi):
        for j in range(n_amp):
            e = ev_stack[:, i, j]
            c = ct_stack[:, i, j]
            d = e - c
            d = d[np.isfinite(d)]
            if len(d) >= 10 and np.any(d != 0):
                try:
                    _, p = wilcoxon(d)
                    pvals[i, j] = p
                except Exception:
                    pass

    # Print top cells
    print(f"\n=== Top 10 cells by event−control difference ===")
    flat_idx = np.argsort(-diff.flatten())
    for k in range(min(10, n_phi * n_amp)):
        fi, fj = np.unravel_index(flat_idx[k], diff.shape)
        print(f"  phase {PHASE_FREQS[fi]:>5.1f} × amp {AMP_FREQS[fj]:>5.1f}  "
              f"event MI {ev_grand[fi,fj]*1e4:>7.3f}  ctl {ct_grand[fi,fj]*1e4:>7.3f}  "
              f"Δ {diff[fi,fj]*1e4:>+7.3f} (×10⁻⁴)  p={pvals[fi,fj]:.2g}")

    print(f"\n=== Cells passing Bonferroni (p < 0.05 / {n_phi*n_amp}) ===")
    thr = 0.05 / (n_phi * n_amp)
    sig = pvals < thr
    for fi, fj in zip(*np.where(sig)):
        print(f"  phase {PHASE_FREQS[fi]:.1f} × amp {AMP_FREQS[fj]:.1f}  "
              f"Δ MI {diff[fi,fj]*1e4:+.3f}×10⁻⁴  p={pvals[fi,fj]:.2g}")

    # Save arrays
    np.savez(os.path.join(OUT_DIR, 'pac_comodulogram.npz'),
             phase_freqs=PHASE_FREQS, amp_freqs=AMP_FREQS,
             event_grand=ev_grand, control_grand=ct_grand, diff=diff,
             pvals=pvals)

    # Plot 3-panel: event, control, diff
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = np.nanmax(np.abs(np.concatenate([ev_grand.flatten(), ct_grand.flatten()])))
    vdiff = np.nanmax(np.abs(diff[np.isfinite(diff)]))
    for ax, mat, title in [
        (axes[0], ev_grand, f'Event (mean of subject means, n={len(ev_stack)})'),
        (axes[1], ct_grand, f'Control (non-event, matched N)'),
    ]:
        im = ax.imshow(mat, aspect='auto', origin='lower',
                        extent=[AMP_FREQS[0] - 2.5, AMP_FREQS[-1] + 2.5,
                                 PHASE_FREQS[0] - 0.5, PHASE_FREQS[-1] + 0.5],
                        cmap='viridis', vmin=0, vmax=vmax)
        ax.set_xlabel('amplitude freq (Hz)')
        ax.set_ylabel('phase freq (Hz)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Tort MI', shrink=0.8)
    ax = axes[2]
    im = ax.imshow(diff, aspect='auto', origin='lower',
                    extent=[AMP_FREQS[0] - 2.5, AMP_FREQS[-1] + 2.5,
                             PHASE_FREQS[0] - 0.5, PHASE_FREQS[-1] + 0.5],
                    cmap='RdBu_r', vmin=-vdiff, vmax=vdiff)
    # Mark Bonferroni-significant cells
    for fi, fj in zip(*np.where(sig)):
        ax.plot(AMP_FREQS[fj], PHASE_FREQS[fi], 'k*', markersize=9)
    ax.axhline(7.82, color='#1a9641', ls='--', lw=0.7, alpha=0.8)
    ax.axhline(14.30, color='#666', ls=':', lw=0.6, alpha=0.6)
    ax.axvline(19.95, color='#1a9641', ls='--', lw=0.7, alpha=0.8)
    ax.set_xlabel('amplitude freq (Hz)')
    ax.set_ylabel('phase freq (Hz)')
    ax.set_title(f'Event − Control (★ = Bonferroni p<0.05)')
    plt.colorbar(im, ax=ax, label='Δ Tort MI', shrink=0.8)

    plt.suptitle(f'B35 — PAC comodulogram · LEMON Q4 events (n={len(results)} subjects)',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pac_comodulogram.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/pac_comodulogram.png")


if __name__ == '__main__':
    main()
