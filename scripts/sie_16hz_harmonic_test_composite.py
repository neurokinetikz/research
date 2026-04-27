#!/usr/bin/env python3
"""
B41 re-run on composite v2 detector.

Is the 16 Hz event-locked peak the 2f harmonic of SR1 or an independent
β-band generator?

Test 1 (frequency tracking): subject SR1 peak in [7, 8.3] vs β16 peak in
[14, 18] from event-averaged PSD. Harmonic predicts slope = 2, ρ ≈ 1.
Independent predicts slope ≈ 0.

Test 2 (amplitude tracking): within-subject Pearson r across Q4 events
(SR1 event amplitude × β16 event amplitude). Harmonic → r > 0.5.
Independent generator with shared envelope → intermediate r.

Envelope B41: OLS slope 0.47 (vs harmonic 2.0); frequency tracking ρ = +0.09
(ns); within-subject amplitude coupling +0.48. Separate β-band generator
sharing the event envelope.

Cohort-parameterized.

Usage:
    python scripts/sie_16hz_harmonic_test_composite.py --cohort lemon
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, pearsonr, wilcoxon
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

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
SR1_RANGE = (7.0, 8.3)
BETA16_RANGE = (14.0, 18.0)

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


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    w = signal.windows.hann(len(seg))
    wp = np.sum(w ** 2)
    X = np.fft.rfft(seg * w, nfft)
    psd = (np.abs(X) ** 2) / (fs * wp)
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
    if len(events) < 2:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 40:
        return None
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    sr1_m = (freqs >= SR1_RANGE[0]) & (freqs <= SR1_RANGE[1])
    b16_m = (freqs >= BETA16_RANGE[0]) & (freqs <= BETA16_RANGE[1])
    f_sr1 = freqs[sr1_m]
    f_b16 = freqs[b16_m]

    base_sr1 = []; base_b16 = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)
        base_sr1.append(psd[sr1_m])
        base_b16.append(psd[b16_m])
    if len(base_sr1) < 10:
        return None
    base_sr1 = np.nanmedian(np.array(base_sr1), axis=0)
    base_b16 = np.nanmedian(np.array(base_b16), axis=0)

    event_sr1_peak_f = []; event_sr1_amp = []
    event_b16_peak_f = []; event_b16_amp = []
    ev_psd_sr1 = []; ev_psd_b16 = []
    for _, ev in events.iterrows():
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)
        sr1_r = (psd[sr1_m] + 1e-20) / (base_sr1 + 1e-20)
        b16_r = (psd[b16_m] + 1e-20) / (base_b16 + 1e-20)
        event_sr1_peak_f.append(parabolic_peak(sr1_r, f_sr1))
        event_sr1_amp.append(float(sr1_r[int(np.argmax(sr1_r))]))
        event_b16_peak_f.append(parabolic_peak(b16_r, f_b16))
        event_b16_amp.append(float(b16_r[int(np.argmax(b16_r))]))
        ev_psd_sr1.append(sr1_r); ev_psd_b16.append(b16_r)
    if len(event_sr1_amp) < 2:
        return None

    sr1_avg = np.nanmean(np.array(ev_psd_sr1), axis=0)
    b16_avg = np.nanmean(np.array(ev_psd_b16), axis=0)
    subj_sr1_peak = parabolic_peak(sr1_avg, f_sr1)
    subj_b16_peak = parabolic_peak(b16_avg, f_b16)

    if len(event_sr1_amp) >= 3:
        r, _ = pearsonr(event_sr1_amp, event_b16_amp)
        r_sp, _ = spearmanr(event_sr1_amp, event_b16_amp)
    else:
        r, r_sp = np.nan, np.nan

    return {
        'subject_id': sub_id,
        'n_events': len(event_sr1_amp),
        'subj_sr1_peak_hz': subj_sr1_peak,
        'subj_b16_peak_hz': subj_b16_peak,
        '2x_sr1': 2.0 * subj_sr1_peak,
        'offset_b16_minus_2sr1': subj_b16_peak - 2.0 * subj_sr1_peak,
        'amp_corr_pearson': r,
        'amp_corr_spearman': r_sp,
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

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, '16hz_harmonic_test.csv'), index=False)

    x = df['subj_sr1_peak_hz'].values
    y = df['subj_b16_peak_hz'].values
    slope, intercept = np.polyfit(x, y, 1)
    r_freq, p_freq = pearsonr(x, y)
    rho_freq, p_rho = spearmanr(x, y)
    print(f"\n=== {args.cohort} composite · Test 1: Frequency tracking (n = {len(df)}) ===")
    print(f"  subject SR1 peak: mean {x.mean():.3f}  std {x.std():.3f}")
    print(f"  subject β16 peak: mean {y.mean():.3f}  std {y.std():.3f}")
    print(f"  2·SR1 would predict: mean {2*x.mean():.3f}")
    print(f"  Observed β16 − 2·SR1 offset: mean {(y - 2*x).mean():+.3f}  "
          f"std {(y - 2*x).std():.3f}")
    print(f"  OLS β16 = {slope:.3f}·SR1 + {intercept:.3f}")
    print(f"  Pearson r = {r_freq:+.3f}  p = {p_freq:.3g}")
    print(f"  Spearman ρ = {rho_freq:+.3f}  p = {p_rho:.3g}")
    print(f"  Harmonic (2f) predicts slope = 2.0, intercept = 0")
    print(f"  Independent predicts slope = 0")

    r_amp = df['amp_corr_pearson'].dropna()
    print(f"\n=== {args.cohort} composite · Test 2: Within-subject amplitude correlation ===")
    print(f"  n subjects with ≥ 3 events: {len(r_amp)}")
    print(f"  Within-subject Pearson r: mean {r_amp.mean():+.3f}  median {r_amp.median():+.3f}  "
          f"std {r_amp.std():.3f}")
    print(f"    IQR [{r_amp.quantile(.25):+.3f}, {r_amp.quantile(.75):+.3f}]")
    print(f"    fraction > 0: {(r_amp > 0).mean()*100:.0f}%")
    print(f"    fraction > 0.5: {(r_amp > 0.5).mean()*100:.0f}%")
    try:
        s, p = wilcoxon(r_amp)
        print(f"  Wilcoxon vs 0:  stat = {s:.1f}  p = {p:.3g}")
    except Exception as e:
        print(f"  Wilcoxon vs 0:  fail: {e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(x, y, s=30, alpha=0.6, color='steelblue', edgecolor='k', lw=0.3)
    rng = np.array([x.min() - 0.1, x.max() + 0.1])
    ax.plot(rng, 2.0 * rng, 'k--', lw=1.2, label='H1: β16 = 2·SR1 (harmonic)')
    ax.plot(rng, np.zeros_like(rng) + y.mean(), 'g-', lw=1.2,
            label=f'H2: β16 fixed ({y.mean():.2f})')
    ax.plot(rng, slope * rng + intercept, color='red', lw=1.5,
            label=f'OLS: {slope:.2f}·SR1 + {intercept:.2f}')
    ax.set_xlabel('subject SR1 peak (Hz)', fontsize=11)
    ax.set_ylabel('subject β16 peak (Hz)', fontsize=11)
    ax.set_title(f'Frequency tracking  ρ={rho_freq:+.2f} p={p_rho:.2g}  n={len(df)}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(r_amp, bins=25, color='firebrick', edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(r_amp.median(), color='blue', ls='--', lw=1.5,
                label=f'median {r_amp.median():+.2f}')
    ax.axvline(0.5, color='green', ls=':', lw=0.8, alpha=0.7,
                label='harmonic-like threshold 0.5')
    ax.set_xlabel('within-subject Pearson r (SR1 amp × β16 amp)', fontsize=11)
    ax.set_ylabel('subjects')
    ax.set_title(f'Amplitude tracking distribution  n={len(r_amp)}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(f'B41 composite — Is 16 Hz peak 2f harmonic of SR1 or independent? '
                 f'({args.cohort} Q4, n={len(df)})',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '16hz_harmonic_test.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/16hz_harmonic_test.png")


if __name__ == '__main__':
    main()
