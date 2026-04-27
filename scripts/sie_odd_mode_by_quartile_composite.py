#!/usr/bin/env python3
"""
B33 + B40 combined on composite v2 detector.

B33: event-locked aggregate spectrum pooled across all composite events,
     measure log-excess at SR1/SR2/SR3/SR4 landmarks. Odd-mode-only pattern?
B40: same analysis stratified by template_ρ quartile (Q1-Q4).

Envelope finding (B33): SR1 +0.27, SR3 +0.06, SR2/SR4 at floor — odd-only.
Envelope finding (B40): Q4 odd/even ratio 14.24 vs Q1 0.07 (inverted).

SR landmarks: SR1=7.82, SR2=13.97, SR3=19.95, SR4=25.44 (actual measured).

Cohort-parameterized.

Usage:
    python scripts/sie_odd_mode_by_quartile_composite.py --cohort lemon
    python scripts/sie_odd_mode_by_quartile_composite.py --cohort lemon_EO
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

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
FREQ_LO, FREQ_HI = 2.0, 30.0

SR_MODES = {
    'SR1': 7.82,
    'SR2': 13.97,
    'SR3': 19.95,
    'SR4': 25.44,
}
SR_HALF_WIDTH = 0.35

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
    win = signal.windows.hann(len(seg))
    wp = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * wp)
    psd[1:-1] *= 2.0
    return psd


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
        q_sub = qual[qual['subject_id'] == sub_id][['t0_net', 'rho_q', 'template_rho']].copy()
        q_sub['t0_round'] = q_sub['t0_net'].round(3)
        events['t0_round'] = events['t0_net'].round(3)
        events = events.merge(q_sub[['t0_round', 'rho_q', 'template_rho']],
                              on='t0_round', how='left')
        events = events.dropna(subset=['rho_q'])
    except Exception:
        return None
    if len(events) < 1:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 70:
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

    ratios_by_q = {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    all_ratios = []
    sw_weights = []
    for _, ev in events.iterrows():
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[mask]
        r = (psd + 1e-20) / (baseline + 1e-20)
        ratios_by_q[ev['rho_q']].append(r)
        all_ratios.append(r)
        # Soft weight = max(template_rho, 0)
        rho = float(ev['template_rho']) if pd.notna(ev['template_rho']) else 0.0
        sw_weights.append(max(rho, 0.0))

    out = {'subject_id': sub_id, 'freqs': f_band}
    out['ALL_mean'] = np.nanmean(np.array(all_ratios), axis=0) if all_ratios else None
    # Soft-weighted per-subject mean: weighted average of event-locked ratios
    # by max(template_rho, 0). Per-subject normalization happens at cohort
    # aggregation step (each subject contributes one summary). NaN if all
    # weights are 0.
    if all_ratios and sum(sw_weights) > 0:
        ev_arr = np.array(all_ratios)
        w_arr = np.array(sw_weights)
        out['SW_mean'] = np.average(ev_arr, axis=0, weights=w_arr)
    else:
        out['SW_mean'] = None
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        if ratios_by_q[q]:
            out[f'{q}_mean'] = np.nanmean(np.array(ratios_by_q[q]), axis=0)
        else:
            out[f'{q}_mean'] = None
    return out


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
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    common = np.arange(FREQ_LO, FREQ_HI + 0.005, 0.05)
    per_q = {}
    all_stack = []
    sw_stack = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        stack = []
        for r in results:
            if r[f'{q}_mean'] is not None:
                stack.append(np.interp(common, r['freqs'],
                                        np.log10(r[f'{q}_mean'] + 1e-20)))
        per_q[q] = np.array(stack)
    for r in results:
        if r['ALL_mean'] is not None:
            all_stack.append(np.interp(common, r['freqs'],
                                        np.log10(r['ALL_mean'] + 1e-20)))
        if r.get('SW_mean') is not None:
            sw_stack.append(np.interp(common, r['freqs'],
                                       np.log10(r['SW_mean'] + 1e-20)))
    all_stack = np.array(all_stack)
    sw_stack = np.array(sw_stack)

    floor_mask = (common >= 12) & (common <= 25) & ~(
        (common >= 19.5) & (common <= 20.5))

    def summarize(mat, label):
        if len(mat) < 5:
            return None
        grand = np.nanmean(mat, axis=0)
        floor = np.nanmedian(grand[floor_mask])
        vals = {}
        for name, f in SR_MODES.items():
            m = (common >= f - SR_HALF_WIDTH) & (common <= f + SR_HALF_WIDTH)
            v = float(np.nanmax(grand[m]))
            vals[name] = v - floor
        odd = (vals['SR1'] + vals['SR3']) / 2
        even = (vals['SR2'] + vals['SR4']) / 2
        return {
            'label': label, 'n_sub': len(mat), 'grand': grand,
            'floor': float(floor), **vals, 'odd_mean': odd, 'even_mean': even,
            'odd_over_even': (odd / even) if abs(even) > 1e-6 else np.inf,
        }

    print(f"\n=== {args.cohort} composite · B33 (all events pooled) ===")
    print(f"(envelope B33: SR1 +0.27 log, SR3 +0.06, SR2/SR4 at floor → odd-only)")
    s_all = summarize(all_stack, 'ALL')
    if s_all:
        print(f"  SR1 {1e2*s_all['SR1']:+.1f}%  SR2 {1e2*s_all['SR2']:+.1f}%  "
              f"SR3 {1e2*s_all['SR3']:+.1f}%  SR4 {1e2*s_all['SR4']:+.1f}%   "
              f"odd/even ratio = {s_all['odd_over_even']:.2f}")

    print(f"\n=== {args.cohort} composite · B40 (by template_ρ quartile) ===")
    print(f"(envelope B40: Q4 odd/even 14.24 vs Q1 0.07)")
    print(f"{'q':<4}{'n_sub':>8}   {'SR1':>10}{'SR2':>10}{'SR3':>10}{'SR4':>10}   {'odd/even':>12}")
    per_q_stats = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        s = summarize(per_q[q], q)
        if s:
            per_q_stats[q] = s
            print(f"{q:<4}{s['n_sub']:>8}   "
                  f"{1e2*s['SR1']:>+8.1f}%  {1e2*s['SR2']:>+8.1f}%  "
                  f"{1e2*s['SR3']:>+8.1f}%  {1e2*s['SR4']:>+8.1f}%   "
                  f"{s['odd_over_even']:>12.2f}")

    print(f"\n=== {args.cohort} composite · SOFT-WEIGHTED (template_rho-weighted) ===")
    print(f"(canonical-event scoping: each event weighted by max(template_rho,0))")
    s_sw = summarize(sw_stack, 'SW')
    if s_sw:
        print(f"  n_sub={s_sw['n_sub']}  "
              f"SR1 {1e2*s_sw['SR1']:+.1f}%  SR2 {1e2*s_sw['SR2']:+.1f}%  "
              f"SR3 {1e2*s_sw['SR3']:+.1f}%  SR4 {1e2*s_sw['SR4']:+.1f}%   "
              f"odd/even = {s_sw['odd_over_even']:.2f}")

    # Save CSV
    rows = []
    if s_all:
        rows.append({'q': 'ALL', **{k: v for k, v in s_all.items()
                                      if k not in ('grand', 'label')}})
    if s_sw:
        rows.append({'q': 'SW', **{k: v for k, v in s_sw.items()
                                     if k not in ('grand', 'label')}})
    for q, s in per_q_stats.items():
        rows.append({'q': q, **{k: v for k, v in s.items()
                                 if k not in ('grand', 'label')}})
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, 'odd_mode_by_quartile.csv'), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = {'Q1': '#4575b4', 'Q2': '#91bfdb', 'Q3': '#fc8d59', 'Q4': '#d73027'}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mat = per_q[q]
        if len(mat) < 5: continue
        grand = np.nanmean(mat, axis=0)
        ratio = 10 ** grand
        ax.plot(common, ratio, color=colors[q], lw=1.6, label=f'{q} n_sub={len(mat)}')
    for name, f in SR_MODES.items():
        c = '#1a9641' if name in ('SR1', 'SR3') else '#d7301f'
        ls = '--' if name in ('SR1', 'SR3') else ':'
        ax.axvline(f, color=c, ls=ls, lw=0.8, alpha=0.7)
    ax.axhline(1.0, color='k', lw=0.6)
    ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('event / baseline PSD (×)')
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_title(f'B33+B40 · event-locked spectrum by template_ρ · {args.cohort} composite v2')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'odd_mode_by_quartile.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/odd_mode_by_quartile.png")


if __name__ == '__main__':
    main()
