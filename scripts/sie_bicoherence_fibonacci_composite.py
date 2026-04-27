#!/usr/bin/env python3
"""
B37 re-run on composite v2 detector.

Targeted bicoherence + Fibonacci amplitude-product coupling at five explicit
Fibonacci-additive bifrequency pairs:

  1. (7.82, 7.82) → 15.64        self-doubling of SR1
  2. (4.84, 7.82) → 12.66        φ⁻²·SR1 + SR1 → φ·SR1
  3. (7.82, 12.67) → 20.49       SR1 + φ·SR1 → φ²·SR1 (near SR3)
  4. (7.82, 19.95) → 27.77       SR1 + SR3
  5. (12.67, 19.95) → 32.62      φ·SR1 + SR3 (near β-γ boundary)

Per bifreq per window:
  - phase-triad PLV |<exp(i·(φ_a + φ_b − φ_c))>|  (bicoherence proxy)
  - amplitude-product correlation corr(|A_a|·|A_b|, |A_c|)

Event windows: 6 s centered at t0_net + 1 s. Q4 only via template_ρ.

Envelope B37 finding: null at 4/5 Fibonacci-additive pairs; positive only at
SR1+SR1→15.64 (waveform asymmetry / non-sinusoidal SR1, not cross-generator).

Cohort-parameterized.

Usage:
    python scripts/sie_bicoherence_fibonacci_composite.py --cohort lemon
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

BAND_HW = 0.6
WIN_SEC = 6.0
EVENT_LAG = 1.0
MIN_GAP_FROM_EVENT = 30.0

BIFREQS = [
    ('SR1+SR1→15.64',    7.82,  7.82, 15.64),
    ('4.84+SR1→12.66',   4.84,  7.82, 12.66),
    ('SR1+12.67→20.49',  7.82, 12.67, 20.49),
    ('SR1+SR3→27.77',    7.82, 19.95, 27.77),
    ('12.67+SR3→32.62', 12.67, 19.95, 32.62),
]

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


def bandpass(x, fs, center, hw, order=4):
    ny = 0.5 * fs
    lo = max(0.1, center - hw)
    hi = min(ny - 1e-3, center + hw)
    if lo >= hi:
        return np.zeros_like(x)
    b, a = signal.butter(order, [lo / ny, hi / ny], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


def triad_plv(phase_a, phase_b, phase_c):
    d = phase_a + phase_b - phase_c
    return float(np.abs(np.mean(np.exp(1j * d))))


def score_window(y_win, fs):
    out = np.full((len(BIFREQS), 2), np.nan)
    for k, (_, fa, fb, fc) in enumerate(BIFREQS):
        ba = bandpass(y_win, fs, fa, BAND_HW)
        bb = bandpass(y_win, fs, fb, BAND_HW) if fa != fb else ba
        bc = bandpass(y_win, fs, fc, BAND_HW)
        if np.all(ba == 0) or np.all(bc == 0):
            continue
        za = signal.hilbert(ba)
        zb = signal.hilbert(bb)
        zc = signal.hilbert(bc)
        plv = triad_plv(np.angle(za), np.angle(zb), np.angle(zc))
        amp_prod = np.abs(za) * np.abs(zb)
        amp_c = np.abs(zc)
        if amp_prod.std() > 0 and amp_c.std() > 0:
            r = np.corrcoef(amp_prod, amp_c)[0, 1]
        else:
            r = np.nan
        out[k, 0] = plv
        out[k, 1] = r
    return out


def sample_control_times(t_events, n, t_end, seed=0):
    rng = np.random.default_rng(seed)
    lo = WIN_SEC / 2 + 1
    hi = t_end - WIN_SEC / 2 - 1
    out, tries = [], 0
    while len(out) < n and tries < n * 200:
        t = rng.uniform(lo, hi)
        if len(t_events) == 0 or np.min(np.abs(t - t_events)) >= MIN_GAP_FROM_EVENT:
            out.append(t)
        tries += 1
    return np.array(out)


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
    if fs < 80:
        return None
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    t_end = raw.times[-1]

    t_events = events['t0_net'].astype(float).values
    t_controls = sample_control_times(t_events, len(t_events), t_end,
                                        seed=abs(hash(sub_id)) % (2**31))

    nperseg = int(round(WIN_SEC * fs))

    def run_window(t_center):
        i0 = int(round((t_center - WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            return None
        return score_window(y[i0:i1], fs)

    ev_mats = []
    for t0 in t_events:
        s = run_window(t0 + EVENT_LAG)
        if s is not None:
            ev_mats.append(s)
    ct_mats = []
    for tc in t_controls:
        s = run_window(tc)
        if s is not None:
            ct_mats.append(s)
    if not ev_mats or not ct_mats:
        return None
    ev = np.nanmean(np.array(ev_mats), axis=0)
    ct = np.nanmean(np.array(ct_mats), axis=0)
    return {
        'subject_id': sub_id,
        'n_ev': len(ev_mats),
        'n_ct': len(ct_mats),
        'ev_plv': ev[:, 0], 'ct_plv': ct[:, 0],
        'ev_rcorr': ev[:, 1], 'ct_rcorr': ct[:, 1],
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
    print(f"  bifreqs: {len(BIFREQS)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    ev_plv = np.array([r['ev_plv'] for r in results])
    ct_plv = np.array([r['ct_plv'] for r in results])
    ev_r = np.array([r['ev_rcorr'] for r in results])
    ct_r = np.array([r['ct_rcorr'] for r in results])

    print(f"\n=== {args.cohort} composite · Phase-triad PLV: event vs control ===")
    print(f"{'bifreq':<26} {'event':>10} {'control':>10} {'Δ':>10} {'p':>10}")
    for k, (name, fa, fb, fc) in enumerate(BIFREQS):
        e = ev_plv[:, k]; c = ct_plv[:, k]; d = e - c
        d = d[np.isfinite(d)]
        _, p = wilcoxon(d) if len(d) > 5 and np.any(d != 0) else (np.nan, np.nan)
        print(f"{name:<26} {np.nanmean(e):>10.4f} {np.nanmean(c):>10.4f} "
              f"{np.nanmean(e-c):>+10.4f} {p:>10.3g}")

    print(f"\n=== {args.cohort} composite · Amplitude-product corr |A_a·A_b| vs |A_c|: event vs control ===")
    print(f"{'bifreq':<26} {'event':>10} {'control':>10} {'Δ':>10} {'p':>10}")
    for k, (name, fa, fb, fc) in enumerate(BIFREQS):
        e = ev_r[:, k]; c = ct_r[:, k]; d = e - c
        d = d[np.isfinite(d)]
        _, p = wilcoxon(d) if len(d) > 5 and np.any(d != 0) else (np.nan, np.nan)
        print(f"{name:<26} {np.nanmean(e):>10.4f} {np.nanmean(c):>10.4f} "
              f"{np.nanmean(e-c):>+10.4f} {p:>10.3g}")

    rows = []
    for r in results:
        for k, (name, *_) in enumerate(BIFREQS):
            rows.append({
                'subject_id': r['subject_id'], 'bifreq': name,
                'event_plv': r['ev_plv'][k], 'control_plv': r['ct_plv'][k],
                'event_rcorr': r['ev_rcorr'][k], 'control_rcorr': r['ct_rcorr'][k],
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'bicoherence_fibonacci.csv'),
                                index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    x = np.arange(len(BIFREQS))
    w = 0.35

    ax = axes[0]
    ev_mean = np.nanmean(ev_plv, axis=0)
    ev_sem = np.nanstd(ev_plv, axis=0) / np.sqrt(np.sum(np.isfinite(ev_plv), axis=0))
    ct_mean = np.nanmean(ct_plv, axis=0)
    ct_sem = np.nanstd(ct_plv, axis=0) / np.sqrt(np.sum(np.isfinite(ct_plv), axis=0))
    ax.bar(x - w/2, ev_mean, w, yerr=ev_sem, color='firebrick', alpha=0.8, label='event')
    ax.bar(x + w/2, ct_mean, w, yerr=ct_sem, color='gray', alpha=0.8, label='control')
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BIFREQS], rotation=20, ha='right')
    ax.set_ylabel('phase-triad PLV')
    ax.set_title('Bicoherence-like phase coupling at Fibonacci-additive bifreqs')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ev_r_mean = np.nanmean(ev_r, axis=0)
    ev_r_sem = np.nanstd(ev_r, axis=0) / np.sqrt(np.sum(np.isfinite(ev_r), axis=0))
    ct_r_mean = np.nanmean(ct_r, axis=0)
    ct_r_sem = np.nanstd(ct_r, axis=0) / np.sqrt(np.sum(np.isfinite(ct_r), axis=0))
    ax.bar(x - w/2, ev_r_mean, w, yerr=ev_r_sem, color='firebrick', alpha=0.8, label='event')
    ax.bar(x + w/2, ct_r_mean, w, yerr=ct_r_sem, color='gray', alpha=0.8, label='control')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BIFREQS], rotation=20, ha='right')
    ax.set_ylabel('corr(|A_a·A_b|, |A_c|)')
    ax.set_title('Fibonacci amplitude-product coupling at bifreqs')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(f'B37 composite — Bicoherence + Fibonacci amplitude coupling '
                 f'({args.cohort} Q4, n={len(results)})', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'bicoherence_fibonacci.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/bicoherence_fibonacci.png")


if __name__ == '__main__':
    main()
