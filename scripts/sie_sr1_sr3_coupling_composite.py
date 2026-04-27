#!/usr/bin/env python3
"""
B34 re-run on composite v2 detector.

SR1 × SR3 envelope coupling + Tort PAC test. Per composite event (6-s window
at t0_net + 1 s):
  1. Envelope xcorr: |Hilbert(bp_SR1)| vs |Hilbert(bp_SR3)|, lag search
     [−1.5, +1.5] s → peak r + lag
  2. Tort MI: SR1 phase → SR3 amplitude (18 bins)
  3. Random-time control (≥30 s from any event)

Envelope B34: PAC MI ≈ 0.0001 (event) ≈ control; zero-lag env corr 0.04;
peak xcorr event 0.504 vs control 0.487 — SR1 and SR3 are INDEPENDENT modes
co-excited by a shared envelope, not phase-coupled.

Cohort-parameterized.

Usage:
    python scripts/sie_sr1_sr3_coupling_composite.py --cohort lemon
    python scripts/sie_sr1_sr3_coupling_composite.py --cohort lemon_EO
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

SR1_BAND = (7.0, 8.2)
SR3_BAND = (19.5, 20.4)

EVENT_WIN = 6.0
EVENT_LAG = 1.0
MIN_GAP_FROM_EVENT = 30.0

LAGS_S = np.arange(-1.5, 1.5 + 0.05, 0.05)
N_PHASE_BINS = 18

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events
    if cohort == 'srm':
        return load_srm, {}, events
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events
    if cohort == 'dortmund':
        return load_dortmund, {}, events
    if cohort == 'chbmp':
        return load_chbmp, {}, events
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events
    raise ValueError(f"unsupported cohort {cohort!r}")


def bandpass(x, fs, lo, hi, order=4):
    b, a = signal.butter(order, [lo, hi], btype='band', fs=fs)
    return signal.filtfilt(b, a, x, axis=-1)


def envelope_xcorr(env1, env2, fs, lags_s):
    env1 = (env1 - env1.mean()) / (env1.std() + 1e-12)
    env2 = (env2 - env2.mean()) / (env2.std() + 1e-12)
    out = np.zeros(len(lags_s))
    n = len(env1)
    for i, lag_s in enumerate(lags_s):
        lag_samp = int(round(lag_s * fs))
        if lag_samp >= 0:
            a = env1[:n - lag_samp]
            b = env2[lag_samp:]
        else:
            a = env1[-lag_samp:]
            b = env2[:n + lag_samp]
        if len(a) < 10:
            out[i] = np.nan; continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return out


def tort_mi(phase, amp, n_bins=N_PHASE_BINS):
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(phase, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    mean_amp = np.array([np.nanmean(amp[bin_idx == k]) if np.sum(bin_idx == k) > 0 else np.nan
                          for k in range(n_bins)])
    if np.any(np.isnan(mean_amp)) or np.sum(mean_amp) <= 0:
        return np.nan
    p = mean_amp / np.sum(mean_amp)
    kl = np.nansum(p * np.log(p * n_bins + 1e-12))
    return float(kl / np.log(n_bins))


def sample_control_times(t_events, n_target, t_end, seed=0):
    rng = np.random.default_rng(seed)
    lo = EVENT_WIN / 2 + 1
    hi = t_end - EVENT_WIN / 2 - 1
    out, tries = [], 0
    while len(out) < n_target and tries < n_target * 200:
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
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 3:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    t_end = raw.times[-1]

    bp1 = bandpass(y, fs, *SR1_BAND)
    bp3 = bandpass(y, fs, *SR3_BAND)
    z1 = signal.hilbert(bp1)
    z3 = signal.hilbert(bp3)
    env1 = np.abs(z1)
    env3 = np.abs(z3)
    phase1 = np.angle(z1)

    t_events = events['t0_net'].astype(float).values
    t_controls = sample_control_times(t_events, len(t_events), t_end,
                                        seed=abs(hash(sub_id)) % (2**31))

    def score_window(t_center):
        i0 = int(round((t_center - EVENT_WIN / 2) * fs))
        i1 = i0 + int(round(EVENT_WIN * fs))
        if i0 < 0 or i1 > len(y):
            return None
        env1_w = env1[i0:i1]
        env3_w = env3[i0:i1]
        phase1_w = phase1[i0:i1]
        xc = envelope_xcorr(env1_w, env3_w, fs, LAGS_S)
        pk_idx = np.nanargmax(xc)
        return {'r_peak': xc[pk_idx], 'lag_peak_s': LAGS_S[pk_idx],
                'r_at_zero': xc[len(LAGS_S)//2],
                'mi': tort_mi(phase1_w, env3_w)}

    ev_rows, ct_rows = [], []
    for t0 in t_events:
        s = score_window(t0 + EVENT_LAG)
        if s: ev_rows.append(s)
    for tc in t_controls:
        s = score_window(tc)
        if s: ct_rows.append(s)

    if len(ev_rows) < 2 or len(ct_rows) < 2:
        return None
    ev = pd.DataFrame(ev_rows)
    ct = pd.DataFrame(ct_rows)
    return {
        'subject_id': sub_id,
        'n_events': len(ev),
        'n_controls': len(ct),
        'ev_r_peak_mean': float(ev['r_peak'].mean()),
        'ev_lag_peak_s_median': float(ev['lag_peak_s'].median()),
        'ev_r_at_zero_mean': float(ev['r_at_zero'].mean()),
        'ev_mi_mean': float(ev['mi'].mean()),
        'ct_r_peak_mean': float(ct['r_peak'].mean()),
        'ct_lag_peak_s_median': float(ct['lag_peak_s'].median()),
        'ct_r_at_zero_mean': float(ct['r_at_zero'].mean()),
        'ct_mi_mean': float(ct['mi'].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'coupling', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(os.path.join(out_dir, 'sr1_sr3_coupling.csv'), index=False)
    print(f"Successful: {len(df)}")

    print(f"\n=== {args.cohort} composite · paired Wilcoxon event vs control ===")
    print(f"(envelope B34: MI 0.0001≈control; zero-lag env r 0.04; peak xcorr 0.504 vs 0.487)")
    for metric in ['r_peak', 'r_at_zero', 'mi']:
        ev = df[f'ev_{metric}_mean']
        ct = df[f'ct_{metric}_mean']
        d = ev - ct
        if (d != 0).sum() < 5:
            continue
        s, p = wilcoxon(d.dropna())
        print(f"  {metric:10s}  event {ev.mean():+.4f} ± {ev.std():.4f}   "
              f"control {ct.mean():+.4f} ± {ct.std():.4f}   "
              f"Δ {d.mean():+.4f}   p = {p:.3g}")

    print(f"\n=== Lag at peak envelope correlation ===")
    print(f"  event lag:   median {df['ev_lag_peak_s_median'].median():+.2f}s  "
          f"IQR [{df['ev_lag_peak_s_median'].quantile(.25):+.2f}, {df['ev_lag_peak_s_median'].quantile(.75):+.2f}]")
    print(f"  control lag: median {df['ct_lag_peak_s_median'].median():+.2f}s")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, metric, label in [
        (axes[0], 'r_peak', 'peak envelope correlation'),
        (axes[1], 'r_at_zero', 'envelope correlation @ lag 0'),
        (axes[2], 'mi', 'Tort PAC MI (SR1 phase → SR3 amp)'),
    ]:
        ev = df[f'ev_{metric}_mean']; ct = df[f'ct_{metric}_mean']
        lo_x = min(ev.min(), ct.min()); hi_x = max(ev.max(), ct.max())
        ax.hist(ct, bins=25, range=(lo_x, hi_x), color='gray', alpha=0.6,
                 label=f'control (mean {ct.mean():.4f})')
        ax.hist(ev, bins=25, range=(lo_x, hi_x), color='firebrick', alpha=0.6,
                 label=f'event (mean {ev.mean():.4f})')
        ax.axvline(ct.mean(), color='gray', ls='--', lw=1)
        ax.axvline(ev.mean(), color='firebrick', ls='--', lw=1)
        ax.set_xlabel(label); ax.set_ylabel('subjects')
        s, p = wilcoxon((ev - ct).dropna()) if (ev != ct).sum() > 5 else (np.nan, np.nan)
        ax.set_title(f'{label}\nΔ = {(ev-ct).mean():+.4f}  Wilcoxon p = {p:.3g}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(f'B34 · SR1 × SR3 coupling · {args.cohort} composite v2', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sr1_sr3_coupling.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/sr1_sr3_coupling.png")


if __name__ == '__main__':
    main()
