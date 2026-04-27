#!/usr/bin/env python3
"""
B36 re-run on composite v2 detector.

Time-resolved Tort PAC comodulogram: 13×13 grid at 6 non-overlapping 6-s
windows centered at [−6, 0, +6, +12, +18, +24] s rel t0_net. Per composite
Q4 event, compute all 6 maps; per-subject mean; cohort grand-average.

Envelope B36: no progressive post-event PAC buildup; best p for Δ at +24 s
= 0.057. Does not replicate discovery paper at this timescale.

Cohort-parameterized. LEMON EC only (most informative per envelope).

Usage:
    python scripts/sie_pac_time_resolved_composite.py --cohort lemon
"""
from __future__ import annotations
import argparse
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
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

PHASE_FREQS = np.arange(2.0, 14.5, 1.0)
PHASE_BW = 1.0
AMP_FREQS = np.arange(17.5, 80, 5.0)
AMP_BW = 2.5

WIN_SEC = 6.0
WIN_CENTERS = np.array([-6.0, 0.0, 6.0, 12.0, 18.0, 24.0])
N_PHASE_BINS = 18

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


def bandpass(x, fs, lo, hi, order=4):
    ny = 0.5 * fs
    if hi >= ny: hi = ny - 1e-3
    if lo <= 0: lo = 0.1
    if lo >= hi: return np.zeros_like(x)
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
            amp_envs.append(None); continue
        f = bandpass(y_win, fs, af - AMP_BW, af + AMP_BW)
        amp_envs.append(np.abs(signal.hilbert(f)))
    for i, pf in enumerate(PHASE_FREQS):
        f = bandpass(y_win, fs, pf - PHASE_BW / 2, pf + PHASE_BW / 2)
        phase = np.angle(signal.hilbert(f))
        for j, af in enumerate(AMP_FREQS):
            if amp_envs[j] is None: continue
            if af - AMP_BW <= pf + PHASE_BW / 2 + 1: continue
            out[i, j] = tort_mi(phase, amp_envs[j])
    return out


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
    sub_id, events_path, quality_csv = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    try:
        qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
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
        raw = _LOADER(sub_id, **_LOADER_KW)
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

    maps_per_win = [[] for _ in WIN_CENTERS]
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
    if any(len(m) == 0 for m in maps_per_win):
        return None
    per_win = [np.nanmean(np.array(m), axis=0) for m in maps_per_win]
    return {
        'subject_id': sub_id, 'n_events': int(len(events)),
        'per_win': per_win,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'coupling', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep, quality_csv))
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)} (Q4)")
    print(f"  windows: {WIN_CENTERS.tolist()}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    stack = np.array([np.array(r['per_win']) for r in results])
    grand = np.nanmean(stack, axis=0)
    base = grand[0]
    dmap = grand - base[None, :, :]

    def idx_ph(f): return int(np.argmin(np.abs(PHASE_FREQS - f)))
    def idx_amp(f): return int(np.argmin(np.abs(AMP_FREQS - f)))

    cells = [
        ('θ4-γ30',  4.0, 32.5),
        ('θ6-γ30',  6.0, 32.5),
        ('θ7-γ40',  7.0, 42.5),
        ('SR1-γ30', 8.0, 32.5),
        ('SR1-γ40', 8.0, 42.5),
        ('α10-γ40', 10.0, 42.5),
        ('SR1-SR3', 8.0, 20.0),
    ]
    print(f"\n=== {args.cohort} composite · time-resolved MI at landmark cells (×10⁻⁴) ===")
    print(f"{'cell':<10} " + '  '.join(f"t={wc:+.0f}s" for wc in WIN_CENTERS))
    for name, pf, af in cells:
        i = idx_ph(pf); j = idx_amp(af)
        row = [grand[w, i, j] for w in range(len(WIN_CENTERS))]
        print(f"{name:<10} " + '  '.join(f"{1e4*v:>+6.2f}" for v in row))

    print(f"\n=== {args.cohort} composite · Δ MI from baseline (t=-6s) at landmark cells (×10⁻⁴) ===")
    print(f"{'cell':<10} " + '  '.join(f"t={wc:+.0f}s" for wc in WIN_CENTERS))
    for name, pf, af in cells:
        i = idx_ph(pf); j = idx_amp(af)
        row = [dmap[w, i, j] for w in range(len(WIN_CENTERS))]
        print(f"{name:<10} " + '  '.join(f"{1e4*v:>+6.2f}" for v in row))

    print(f"\n=== Top 10 cells by Δ MI at t=+24s (post-event buildup test) ===")
    print(f"(envelope B36: best p at t=+24s = 0.057, no progressive buildup)")
    final_dmap = stack[:, -1, :, :] - stack[:, 0, :, :]
    mean_dmap = np.nanmean(final_dmap, axis=0)
    flat = np.argsort(-mean_dmap.flatten())[:10]
    best_p = 1.0
    for idx in flat:
        i, j = np.unravel_index(idx, mean_dmap.shape)
        d = final_dmap[:, i, j]
        d = d[np.isfinite(d)]
        if len(d) > 5 and np.any(d != 0):
            _, p = wilcoxon(d)
        else:
            p = np.nan
        best_p = min(best_p, p) if np.isfinite(p) else best_p
        print(f"  phase {PHASE_FREQS[i]:>5.1f} × amp {AMP_FREQS[j]:>5.1f}  "
              f"Δ MI at t=+24s: {1e4*mean_dmap[i,j]:+.3f}×10⁻⁴  p={p:.3g}")

    # Any Bonferroni-pass at t=+24s?
    n_tests = mean_dmap.size
    thr = 0.05 / n_tests
    any_sig = False
    for i in range(mean_dmap.shape[0]):
        for j in range(mean_dmap.shape[1]):
            d = final_dmap[:, i, j]
            d = d[np.isfinite(d)]
            if len(d) > 5 and np.any(d != 0):
                _, p = wilcoxon(d)
                if p < thr:
                    any_sig = True
    print(f"\n  Any cell passes Bonferroni at t=+24s (p<{thr:.1e}): {any_sig}")
    print(f"  Best uncorrected p: {best_p:.3g}")

    np.savez(os.path.join(out_dir, 'pac_time_resolved.npz'),
             phase_freqs=PHASE_FREQS, amp_freqs=AMP_FREQS,
             win_centers=WIN_CENTERS, grand=grand, dmap=dmap)

    # Plot: 6-panel comodulogram (Δ from baseline)
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

    plt.suptitle(f'B36 · time-resolved PAC Δ from baseline · {args.cohort} composite v2 · Q4 (n={len(results)})',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pac_time_resolved.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # Landmark time courses
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
    ax.set_title(f'Landmark cells: PAC over time · {args.cohort} composite')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pac_time_resolved_landmarks.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/pac_time_resolved.png")


if __name__ == '__main__':
    main()
