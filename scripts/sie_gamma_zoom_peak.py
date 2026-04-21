#!/usr/bin/env python3
"""
B32 — Fine-resolution γ-band zoom on template_rho Q4 events.

B30 found a weak local peak at 33.00 Hz (prominence 0.043) in the pooled
event-locked aggregate, aligned with both φ⁴·SR1 = 33.17 Hz and the φ-lattice
β-γ boundary 32.19 Hz. To test whether this is a real attractor or noise:

  - Q4 filter (high-template_rho events only) — if 33 Hz is a true attractor,
    Q4 should sharpen it the way Q4 sharpened 7.83 Hz (5.9× → 5.7× in B19
    with clearer peak shape).
  - Zoom resolution: nfft_mult=16 for 0.016 Hz bins over [28, 40] Hz.
  - Per-cohort replication: does peak land at same position in LEMON/HBN/TDBRAIN?

Tests:
  (a) Is there a clear peak in Q4 events in [28, 40] Hz?
  (b) Peak position — 33.17 (φ⁴·SR1), 32.19 (β-γ boundary), or elsewhere?
  (c) Cross-cohort replicability of peak location
"""
from __future__ import annotations
import glob as globfn
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon, load_hbn, load_tdbrain

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality', 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
ZOOM_LO, ZOOM_HI = 28.0, 40.0

SCHUMANN_F = 7.83
PHI = 1.6180339887
PHI4_SR1 = SCHUMANN_F * PHI ** 3   # 33.17 Hz (φ⁴·SR1 in our n=1..4 counting)
BETA_GAMMA_BOUND = 32.19


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
    sub_id, events_path, dataset, release, q4_only = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if q4_only and dataset == 'lemon':
        # Only LEMON has template_rho Q4 labels
        qual = pd.read_csv(QUALITY_CSV)
        qual = qual.dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4,
                                 labels=['Q1', 'Q2', 'Q3', 'Q4'])
        qual_sub = qual[(qual['subject_id'] == sub_id) & (qual['rho_q'] == 'Q4')]
        keys = set(qual_sub['t0_net'].round(3).values)
        events['t0_round'] = events['t0_net'].round(3)
        events = events[events['t0_round'].isin(keys)]
    if len(events) < 1:
        return None
    try:
        raw = load_recording(sub_id, dataset, release)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 2 * ZOOM_HI:
        return None
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
        'n_events': int(len(ev_rows)),
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--q4', action='store_true',
                     help='Filter LEMON events to template_rho Q4')
    args = ap.parse_args()

    datasets = [('lemon', None), ('hbn', 'R4'), ('tdbrain', None)]
    all_tasks = []
    for ds, rel in datasets:
        base = gather_subjects(ds, rel)
        all_tasks.extend([(sid, ep, ds, rel, args.q4) for sid, ep, ds, rel in base])
    print(f"Total subjects: {len(all_tasks)}  Q4 filter: {args.q4}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, all_tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    # Interpolate to common grid
    common = np.arange(ZOOM_LO, ZOOM_HI + 0.005, 0.02)
    per_ds = {}
    all_mat = []
    for r in results:
        interp = np.interp(common, r['freqs'], r['log_ratio'])
        per_ds.setdefault(r['dataset'], []).append(interp)
        all_mat.append(interp)
    all_mat = np.array(all_mat)

    grand, lo, hi = bootstrap_ci(all_mat)

    # Peak detection (strict prominence)
    peaks, props = find_peaks(grand, distance=5, prominence=0.01)
    suffix = "_q4" if args.q4 else ""
    print(f"\n=== Pooled γ-zoom [{ZOOM_LO}, {ZOOM_HI}] Hz "
          f"n={len(all_mat)} subjects ===")
    print(f"{'freq':>8} {'log_r':>10} {'prom':>8} {'x_ratio':>10}")
    for i in peaks:
        prom = props['prominences'][list(peaks).index(i)] if 'prominences' in props else np.nan
        print(f"{common[i]:>8.3f} {grand[i]:>+10.4f} {prom:>8.4f} {10**grand[i]:>10.3f}")

    # Landmark check
    print(f"\nKey landmarks:")
    for label, freq in [('β-γ boundary', BETA_GAMMA_BOUND),
                         ('φ⁴·SR1 (precise)', PHI4_SR1),
                         ('2×φ²·SR1 reference', 2 * SCHUMANN_F * PHI ** 2),
                         ('SR5 empirical', 33.8),
                         ('34 Hz (reference)', 34.0),
                         ('36 Hz (generic γ)', 36.0)]:
        if freq < ZOOM_LO or freq > ZOOM_HI:
            continue
        i = int(np.argmin(np.abs(common - freq)))
        print(f"  {label:22s} {freq:.3f} → {common[i]:.3f}  log {grand[i]:+.4f} ({10**grand[i]:.3f}×)")

    # Per-cohort peaks
    print(f"\n=== Per-cohort γ-zoom peaks ===")
    for ds, arr in per_ds.items():
        mat = np.array(arr)
        gr, _, _ = bootstrap_ci(mat)
        pks, props_ds = find_peaks(gr, distance=5, prominence=0.005)
        print(f"  {ds} (n={len(mat)}):")
        if len(pks) == 0:
            print(f"    no peaks found above prominence 0.005")
            continue
        for i in pks[:5]:
            prom = props_ds['prominences'][list(pks).index(i)]
            print(f"    {common[i]:.3f} Hz  log {gr[i]:+.4f} ({10**gr[i]:.3f}×)  prom {prom:.4f}")

    pd.DataFrame({'freq_hz': common, 'log_ratio_grand_mean': grand,
                   'ci_lo': lo, 'ci_hi': hi,
                   'n_subjects': len(all_mat)}).to_csv(
        os.path.join(OUT_DIR, f'gamma_zoom_peak{suffix}.csv'), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ratio = 10 ** grand
    ax.plot(common, ratio, color='#8c1a1a', lw=1.6,
            label=f'pooled n = {len(all_mat)}')
    ax.fill_between(common, 10 ** lo, 10 ** hi, color='#8c1a1a', alpha=0.22)
    ax.axhline(1.0, color='k', lw=0.6)
    ax.axvline(BETA_GAMMA_BOUND, color='#666', ls=':', lw=0.9, alpha=0.7,
                label=f'β-γ boundary {BETA_GAMMA_BOUND}')
    ax.axvline(PHI4_SR1, color='#1a9641', ls='--', lw=0.9, alpha=0.7,
                label=f'φ⁴·SR1 {PHI4_SR1:.2f}')
    # Annotate detected peaks
    for i in peaks:
        if props['prominences'][list(peaks).index(i)] > 0.01:
            ax.annotate(f'{common[i]:.2f}',
                         xy=(common[i], ratio[i]),
                         xytext=(common[i], ratio[i] + 0.1),
                         fontsize=8.5, ha='center')
    ax.set_xlabel('frequency (Hz)', fontsize=11)
    ax.set_ylabel('event / baseline PSD (×)', fontsize=11)
    title = f'B32 — γ-band zoom [{ZOOM_LO}-{ZOOM_HI} Hz]'
    if args.q4:
        title += ' (LEMON Q4 template_rho only)'
    ax.set_title(title)
    ax.set_xlim(ZOOM_LO, ZOOM_HI)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'gamma_zoom_peak{suffix}.png'),
                 dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/gamma_zoom_peak{suffix}.png")


if __name__ == '__main__':
    main()
