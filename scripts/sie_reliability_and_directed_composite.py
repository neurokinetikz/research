#!/usr/bin/env python3
"""
B44 re-run on composite v2 detector.

Envelope cross-correlation lag between SR1 (7-8.3), β16 (14.5-17.5), SR3
(19.5-20.4) envelopes in Q4 event windows (8 s at t0_net + 1 s) vs matched
controls (≥30 s from events). For each pair, report (lag_peak, r_at_peak,
r_at_zero_lag) event vs control + Wilcoxon.

Envelope B44: event vs control lag differences indistinguishable (|Δlag|
≤ 0.24 s, all p > 0.2). No directed coupling at envelope level; B31 peak-
time offsets reflect response functions not causal flow.

Cohort-parameterized.

Usage:
    python scripts/sie_reliability_and_directed_composite.py --cohort lemon
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

SR1_BAND = (7.0, 8.3)
BETA16_BAND = (14.5, 17.5)
SR3_BAND = (19.5, 20.4)

EVENT_WIN = 8.0
EVENT_LAG = 1.0
MIN_GAP_FROM_EVENT = 30.0
LAGS_S = np.arange(-3.0, 3.0 + 0.05, 0.05)

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
    lo = max(0.1, lo); hi = min(ny - 1e-3, hi)
    if lo >= hi:
        return np.zeros_like(x)
    b, a = signal.butter(order, [lo / ny, hi / ny], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


def envelope_xcorr_lag(env_a, env_b, fs, lags_s=LAGS_S):
    a = (env_a - env_a.mean()) / (env_a.std() + 1e-12)
    b = (env_b - env_b.mean()) / (env_b.std() + 1e-12)
    n = len(a)
    out = np.zeros(len(lags_s))
    for i, l in enumerate(lags_s):
        ls = int(round(l * fs))
        if ls >= 0:
            x = a[:n - ls]; y = b[ls:]
        else:
            x = a[-ls:]; y = b[:n + ls]
        if len(x) < 10:
            out[i] = np.nan
        else:
            out[i] = np.corrcoef(x, y)[0, 1]
    return out


def sample_control_times(t_events, n, t_end, seed=0):
    rng = np.random.default_rng(seed)
    lo = EVENT_WIN / 2 + 1
    hi = t_end - EVENT_WIN / 2 - 1
    out = []; tries = 0
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
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4,
                                 labels=['Q1','Q2','Q3','Q4'])
        q4 = qual[(qual['subject_id'] == sub_id) & (qual['rho_q'] == 'Q4')]
        q4_times = set(q4['t0_net'].round(3).values)
        events['t0_round'] = events['t0_net'].round(3)
        events = events[events['t0_round'].isin(q4_times)]
    except Exception:
        pass
    if len(events) < 2:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 60:
        return None
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    t_end = raw.times[-1]

    env_sr1 = np.abs(signal.hilbert(bandpass(y, fs, *SR1_BAND)))
    env_b16 = np.abs(signal.hilbert(bandpass(y, fs, *BETA16_BAND)))
    env_sr3 = np.abs(signal.hilbert(bandpass(y, fs, *SR3_BAND)))

    t_events = events['t0_net'].astype(float).values
    t_controls = sample_control_times(t_events, len(t_events), t_end,
                                        seed=abs(hash(sub_id)) % (2**31))

    nperseg = int(round(EVENT_WIN * fs))

    def run_window(t_center):
        i0 = int(round((t_center - EVENT_WIN / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            return None
        s1 = env_sr1[i0:i1]; b1 = env_b16[i0:i1]; s3 = env_sr3[i0:i1]

        def xc(a, b):
            xc_v = envelope_xcorr_lag(a, b, fs)
            pk = int(np.nanargmax(xc_v))
            return LAGS_S[pk], xc_v[pk], xc_v[len(LAGS_S) // 2]
        out = {}
        for lab, pair in [('sr1_b16', (s1, b1)),
                           ('sr1_sr3', (s1, s3)),
                           ('b16_sr3', (b1, s3))]:
            lag_pk, r_pk, r_0 = xc(*pair)
            out[f'{lab}_lag'] = lag_pk
            out[f'{lab}_rpk'] = r_pk
            out[f'{lab}_r0'] = r_0
        return out

    ev_rows, ct_rows = [], []
    for t in t_events:
        r = run_window(t + EVENT_LAG)
        if r: ev_rows.append(r)
    for t in t_controls:
        r = run_window(t)
        if r: ct_rows.append(r)
    if not ev_rows or not ct_rows:
        return None
    ev_df = pd.DataFrame(ev_rows); ct_df = pd.DataFrame(ct_rows)
    return {
        'subject_id': sub_id,
        'n_ev': len(ev_rows), 'n_ct': len(ct_rows),
        'ev_mean': ev_df.mean().to_dict(),
        'ct_mean': ct_df.mean().to_dict(),
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
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    print(f"\n" + "=" * 64)
    print(f"{args.cohort} composite · Envelope cross-correlation lag, event vs control")
    print("=" * 64)
    pairs = [('sr1_b16', 'SR1 × β16', 'positive lag → β16 lags SR1'),
              ('sr1_sr3', 'SR1 × SR3', 'positive lag → SR3 lags SR1'),
              ('b16_sr3', 'β16 × SR3', 'positive lag → SR3 lags β16')]
    print(f"{'pair':<12} {'metric':<12} {'event':>10} {'control':>10} {'Δ':>10} {'p':>10}")
    rows = []
    for key, label, interp in pairs:
        print(f"\n  {label}  ({interp})")
        for metric in ['lag', 'rpk', 'r0']:
            ev = np.array([r['ev_mean'][f'{key}_{metric}'] for r in results])
            ct = np.array([r['ct_mean'][f'{key}_{metric}'] for r in results])
            d = ev - ct
            d = d[np.isfinite(d)]
            if len(d) >= 10 and np.any(d != 0):
                _, p = wilcoxon(d)
            else:
                p = np.nan
            print(f"    {metric:<10}  {np.nanmean(ev):>+10.3f} "
                  f"{np.nanmean(ct):>+10.3f} {np.nanmean(ev - ct):>+10.3f} "
                  f"{p:>10.3g}")
            rows.append({'pair': key, 'metric': metric,
                          'event_mean': float(np.nanmean(ev)),
                          'control_mean': float(np.nanmean(ct)),
                          'delta': float(np.nanmean(ev - ct)),
                          'p_wilcoxon': p})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'directed_coupling.csv'),
                                index=False)

    print(f"\n=== Per-subject lag (SR1 × β16), Q4 event windows ===")
    lags = np.array([r['ev_mean']['sr1_b16_lag'] for r in results])
    print(f"  n = {len(lags)}")
    print(f"  median {np.median(lags):+.2f}s   IQR [{np.percentile(lags,25):+.2f}, "
          f"{np.percentile(lags,75):+.2f}]")
    print(f"  fraction positive (β16 lags SR1): {(lags > 0).mean()*100:.0f}%")
    try:
        _, p = wilcoxon(lags)
        print(f"  Wilcoxon vs 0: p = {p:.3g}")
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for col, (key, label, _) in enumerate(pairs):
        ax = axes[col]
        ev_lags = np.array([r['ev_mean'][f'{key}_lag'] for r in results])
        ct_lags = np.array([r['ct_mean'][f'{key}_lag'] for r in results])
        ax.hist(ct_lags, bins=np.linspace(-3, 3, 30), color='gray', alpha=0.5,
                label=f'control (med {np.median(ct_lags):+.2f})')
        ax.hist(ev_lags, bins=np.linspace(-3, 3, 30), color='firebrick', alpha=0.6,
                label=f'event (med {np.median(ev_lags):+.2f})')
        ax.axvline(0, color='k', lw=0.6)
        ax.axvline(np.median(ev_lags), color='firebrick', ls='--', lw=1.0)
        ax.axvline(np.median(ct_lags), color='gray', ls='--', lw=1.0)
        ax.set_xlabel('envelope lag (s)\npositive = right band lags left band')
        ax.set_ylabel('subjects')
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f'B44 composite — Envelope directed coupling '
                 f'({args.cohort} Q4, n = {len(results)})',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'directed_coupling.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/directed_coupling.png")


if __name__ == '__main__':
    main()
