#!/usr/bin/env python3
"""
B18 re-run on composite v2 detector.

Identical analysis to scripts/sie_phi_lattice_trajectory.py; cohort-
parameterized; reads composite template_rho.

Per subject: sliding Welch + log10 band power relative to baseline for:
  6 φ-lattice bands: [1, 4.70, 7.60, 12.30, 19.90, 32.19, 45] Hz
  2 SR sub-bands: SR_θ [7.0, 7.60], SR_α [7.60, 8.2]
Interpolated onto [-15, +15] s grid rel t0_net, stratified by Q1 vs Q4.

Outputs to outputs/schumann/images/psd_timelapse/<cohort>_composite/.

Usage:
    python scripts/sie_phi_lattice_trajectory_composite.py --cohort lemon
    python scripts/sie_phi_lattice_trajectory_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
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

WIN_SEC = 4.0
HOP_SEC = 1.0
FREQ_LO, FREQ_HI = 1.0, 45.0
NFFT_MULT = 4

PHI_EDGES = [1.0, 4.70, 7.60, 12.30, 19.90, 32.19, 45.0]
PHI_LABELS = ['δ/lθ', 'θ', 'α', 'β-lo', 'β-hi', 'γ']
SR_BANDS = {'SR_θ': (7.0, 7.60), 'SR_α': (7.60, 8.2)}
TGRID = np.arange(-15.0, 15.0 + 0.5, 1.0)

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    qual = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, qual
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, qual
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, qual
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, qual
    if cohort == 'srm':
        return load_srm, {}, qual
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, qual
    if cohort == 'dortmund':
        return load_dortmund, {}, qual
    if cohort == 'chbmp':
        return load_chbmp, {}, qual
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, qual
    raise ValueError(f"unsupported cohort {cohort!r}")


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
    return np.array(t_cent), freqs, np.array(cols).T


def band_power_series(freqs, P, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() == 0:
        return np.full(P.shape[1], np.nan)
    return np.nanmean(P[mask, :], axis=0)


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
    sub_id, df_sub = args
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    t, freqs, P = sliding_welch(y, fs)
    bands = {}
    for i in range(len(PHI_LABELS)):
        bands[PHI_LABELS[i]] = band_power_series(freqs, P,
                                                  PHI_EDGES[i], PHI_EDGES[i+1])
    for name, (lo, hi) in SR_BANDS.items():
        bands[name] = band_power_series(freqs, P, lo, hi)

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
                np.full(mat.shape[1] if mat.ndim == 2 else len(TGRID), np.nan),
                np.full(mat.shape[1] if mat.ndim == 2 else len(TGRID), np.nan))
    rng = np.random.default_rng(seed)
    grand = np.nanmean(mat, axis=0)
    boots = np.zeros((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, len(mat), size=len(mat))
        boots[b] = np.nanmean(mat[idx], axis=0)
    return grand, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'psd_timelapse', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    tasks = [(sid, g) for sid, g in qual.groupby('subject_id')]
    print(f"Cohort: {args.cohort} (composite v2)")
    print(f"Subjects: {len(tasks)}  events: {len(qual)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    def stack(q, name):
        arr = [r[f'{q}_{name}'] for r in results if r[f'{q}_{name}'] is not None]
        return np.array(arr) if arr else np.empty((0, len(TGRID)))

    peak_idx = int(np.argmin(np.abs(TGRID - 1.0)))
    print(f"\n=== {args.cohort} composite — φ-lattice band log-boost at t=+1s ===")
    print(f"{'band':<8} {'Q1 log':<10} {'Q1 ratio':<10} {'Q4 log':<10} {'Q4 ratio':<10}")
    for name in PHI_LABELS + list(SR_BANDS.keys()):
        q1_v, q1_r = np.nan, np.nan
        v, ratio = np.nan, np.nan
        for q in ['Q1', 'Q4']:
            mat = stack(q, name)
            grand, _, _ = bootstrap_ci(mat)
            v = grand[peak_idx] if np.isfinite(grand[peak_idx]) else np.nan
            ratio = 10 ** v if np.isfinite(v) else np.nan
            if q == 'Q1':
                q1_v, q1_r = v, ratio
        print(f"{name:<8} {q1_v:+.3f}     {q1_r:.2f}×      {v:+.3f}     {ratio:.2f}×")

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
        for name, color, ls in [('SR_θ', '#8c2d04', '--'), ('SR_α', '#d94801', ':')]:
            mat = stack(q, name)
            grand, lo, hi = bootstrap_ci(mat)
            ratio = 10 ** grand
            ax.plot(TGRID, ratio, color=color, lw=2, ls=ls, label=name)
        ax.axhline(1.0, color='k', lw=0.7)
        ax.axvline(0, color='k', ls='--', lw=0.5)
        ax.set_xlabel('time rel. t0_net (s)')
        ax.set_ylabel('band power / baseline (×)')
        ax.set_title(f'{args.cohort} composite · template_rho {q}')
        ax.legend(fontsize=8, loc='upper left', ncol=2)
        ax.grid(alpha=0.3)

    plt.suptitle(f'B18 · peri-event φ-lattice trajectory · {args.cohort} composite v2',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phi_lattice_trajectory.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

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
    ax.set_title(f'B18 · {args.cohort} composite · Q4 ignition excursion per band')
    for i, r in enumerate(ratios):
        if np.isfinite(r):
            ax.text(i, r + 0.02, f'{r:.2f}×', ha='center', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phi_lattice_peak_bars.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    np.savez(os.path.join(out_dir, 'phi_lattice_trajectory.npz'),
             TGRID=TGRID, PHI_LABELS=np.array(PHI_LABELS, dtype=object),
             **{f'{q}_{name}': stack(q, name)
                for q in ['Q1', 'Q4'] for name in PHI_LABELS + list(SR_BANDS.keys())})

    print(f"\nSaved: {out_dir}/phi_lattice_trajectory.png")
    print(f"Saved: {out_dir}/phi_lattice_peak_bars.png")
    print(f"Saved: {out_dir}/phi_lattice_trajectory.npz")


if __name__ == '__main__':
    main()
