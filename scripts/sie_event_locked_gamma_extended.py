#!/usr/bin/env python3
"""
B30 — Event-locked aggregate extended to γ-band [2, 50] Hz.

B27 capped at 25 Hz and revealed the secondary β peak at ~20 Hz. Extends the
frequency range to 50 Hz to check for companion peaks at higher-order φ-lattice
points or Schumann harmonics:

  - 2×SR1 = 15.66 Hz (Schumann 2nd-harmonic integer-multiple)
  - Schumann 2nd harmonic empirical: 14.3 Hz
  - φ·SR1 = 12.67 Hz (also close to φ-lattice α-β boundary 12.30)
  - 3×SR1 = 23.49 Hz
  - Schumann 3rd harmonic empirical: 20.8 Hz
  - β-γ lattice boundary: 32.19 Hz
  - φ³·SR1 = 33.17 Hz (very close to the β-γ boundary)
  - γ lattice peak range 32-50 Hz
"""
from __future__ import annotations
import glob as globfn
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, peak_prominences
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon, load_hbn, load_tdbrain

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
FREQ_LO, FREQ_HI = 2.0, 50.0


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def locate_hbn_set(sub_id, release):
    pat = f'/Volumes/T9/hbn_data/cmi_bids_{release}/{sub_id}/eeg/*RestingState_eeg.set'
    files = sorted(globfn.glob(pat))
    return files[0] if files else None


def load_recording(sub_id, dataset, release=None):
    if dataset == 'lemon':
        return load_lemon(sub_id, condition='EC')
    if dataset == 'hbn':
        set_path = locate_hbn_set(sub_id, release)
        if not set_path:
            return None
        return load_hbn(set_path)
    if dataset == 'tdbrain':
        return load_tdbrain(sub_id, condition='EC')
    raise ValueError(f"Unknown dataset: {dataset}")


def process_subject(args):
    sub_id, events_path, dataset, release = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 1:
        return None
    try:
        raw = load_recording(sub_id, dataset, release)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 2 * FREQ_HI:
        return None
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    f_band = freqs[mask]

    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        base_rows.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if len(base_rows) < 10:
        return None
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    ev_rows = []
    for _, ev in events.iterrows():
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        ev_rows.append(welch_one(y[i0:i1], fs, nfft)[mask])
    if not ev_rows:
        return None
    ev_avg = np.nanmean(np.array(ev_rows), axis=0)
    return {
        'subject_id': sub_id,
        'dataset': dataset,
        'freqs': f_band,
        'log_ratio': np.log10(ev_avg + 1e-20) - np.log10(baseline + 1e-20),
    }


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


def gather_subjects(dataset, release=None):
    if dataset == 'hbn':
        events_dir = os.path.join(EVENTS_BASE, f'hbn_{release}')
    elif dataset == 'tdbrain':
        events_dir = os.path.join(EVENTS_BASE, 'tdbrain')
    else:
        events_dir = os.path.join(EVENTS_BASE, 'lemon')
    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep, dataset, release))
    return tasks


def main():
    datasets = [('lemon', None), ('hbn', 'R4'), ('tdbrain', None)]
    all_tasks = []
    for ds, rel in datasets:
        all_tasks.extend(gather_subjects(ds, rel))
    print(f"Total subjects: {len(all_tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, all_tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    # Common grid
    common = np.arange(FREQ_LO, FREQ_HI + 0.025, 0.05)
    all_mat = []
    per_ds = {}
    for r in results:
        interp = np.interp(common, r['freqs'], r['log_ratio'])
        per_ds.setdefault(r['dataset'], []).append(interp)
        all_mat.append(interp)
    all_mat = np.array(all_mat)
    grand, lo, hi = bootstrap_ci(all_mat)

    # Find peaks in the pooled residual
    peaks, _ = find_peaks(grand, distance=5, prominence=0.02)
    proms = peak_prominences(grand, peaks)[0]
    print(f"\n=== Pooled event-locked aggregate peaks [2, 50] Hz (n = {len(all_mat)}) ===")
    print(f"{'freq':>8} {'log_ratio':>10} {'prominence':>12} {'ratio_×':>10}")
    for i, p in zip(peaks, proms):
        print(f"{common[i]:>8.2f} {grand[i]:>+10.4f} {p:>+12.4f} {10**grand[i]:>10.2f}")

    # Landmark check
    schumann_harm = [7.83, 14.3, 20.8, 27.3, 33.8]
    phi_lattice = [4.70, 7.60, 12.30, 19.90, 32.19]
    phi_harm_sr = [7.83 * (1.618 ** n) for n in range(-1, 4)]  # 4.84, 7.83, 12.67, 20.50, 33.17
    integer_sr = [7.83 * n for n in range(1, 5)]  # 7.83, 15.66, 23.49, 31.32
    print(f"\n=== Pooled values at landmarks ===")
    for label, vals in [('Schumann harmonics', schumann_harm),
                        ('φ-lattice boundaries', phi_lattice),
                        ('φ^n · SR1',             phi_harm_sr),
                        ('integer · SR1',         integer_sr)]:
        print(f"\n  {label}:")
        for v in vals:
            if v < FREQ_LO or v > FREQ_HI:
                continue
            i = int(np.argmin(np.abs(common - v)))
            r = grand[i]
            print(f"    {v:.3f} Hz → {common[i]:.3f} Hz  log_ratio {r:+.3f}  ({10**r:.2f}×)")

    # Per-cohort peaks in γ range [28, 50]
    print(f"\n=== Per-cohort γ-band (>28 Hz) peaks ===")
    for ds, arr in per_ds.items():
        mat = np.array(arr)
        gr, _, _ = bootstrap_ci(mat)
        peaks_ds, _ = find_peaks(gr, distance=5, prominence=0.015)
        mask_hi = common[peaks_ds] > 28
        peaks_hi = peaks_ds[mask_hi]
        print(f"  {ds} (n={len(mat)}): {len(peaks_hi)} peaks > 28 Hz")
        for i in peaks_hi[:5]:
            print(f"    {common[i]:.2f} Hz  {10**gr[i]:.2f}×")

    pd.DataFrame({'freq_hz': common, 'log_ratio_grand_mean': grand,
                   'ci_lo': lo, 'ci_hi': hi,
                   'n_subjects': len(all_mat)}).to_csv(
        os.path.join(OUT_DIR, 'event_locked_aggregate_gamma.csv'), index=False)

    # Plot with landmarks
    fig, ax = plt.subplots(figsize=(14, 6))
    ratio = 10 ** grand
    ax.plot(common, ratio, color='#8c1a1a', lw=1.6, label=f'n = {len(all_mat)}')
    ax.fill_between(common, 10 ** lo, 10 ** hi, color='#8c1a1a', alpha=0.22)
    ax.axhline(1.0, color='k', lw=0.6)
    # Schumann harmonics (solid green dashed)
    for k, h in enumerate(schumann_harm):
        if FREQ_LO <= h <= FREQ_HI:
            ax.axvline(h, color='#1a9641',
                        ls='--' if k == 0 else ':',
                        lw=1 if k == 0 else 0.5, alpha=0.7)
    # φ-lattice boundaries (gray dotted)
    for b in phi_lattice:
        if FREQ_LO <= b <= FREQ_HI:
            ax.axvline(b, color='#777', ls=':', lw=0.6, alpha=0.6)
    # Mark detected peaks
    for i, p in zip(peaks, proms):
        if p > 0.05 and common[i] > 6:
            ax.annotate(f'{common[i]:.1f}',
                         xy=(common[i], ratio[i]),
                         xytext=(common[i], ratio[i] + 0.4),
                         fontsize=8, ha='center',
                         color='#444')
    ax.set_xlabel('frequency (Hz)', fontsize=11)
    ax.set_ylabel('event / baseline PSD (×)', fontsize=11)
    ax.set_title(f'B30 — Event-locked aggregate extended to γ-band (pooled 3 cohorts, n={len(all_mat)})')
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'event_locked_aggregate_gamma.png'),
                 dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/event_locked_aggregate_gamma.png")


if __name__ == '__main__':
    main()
