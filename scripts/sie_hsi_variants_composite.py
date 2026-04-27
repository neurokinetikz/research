#!/usr/bin/env python3
"""
B39 re-run on composite v2 detector.

Four HSI variants to disambiguate sign/direction against discovery paper:

  V1 (our): HSI = log10(P_SR1 / P_SR3)
  V2 (SR average): HSI = log10(P_SR1 / mean(P_SR2, P_SR3, P_SR4))
  V3 (invert): HSI = log10(mean(P_SR2, P_SR3, P_SR4) / P_SR1)
  V4 (φ-harmonic set): HSI = log10(P_SR1 / mean(P_φ1, P_φ2, P_φ3))

Peri-event (−15 s .. +15 s) time courses from frontal channels; ±95 % cluster
bootstrap. Per-subject rise/dip distribution at t = +2 s on V1.

Envelope B39: V1 rises +0.20 at events; V2 also rises (+0.24 direction, same
as V1); V3 drops (definitional inverse); V4 depends on SR1 vs φ-harmonic
competition. Conclusion: log(SR1/SR3) at events rises in LEMON — opposite to
discovery paper's DIP — and definition inversion does not reconcile the two.

Cohort-parameterized.

Usage:
    python scripts/sie_hsi_variants_composite.py --cohort lemon
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

FRONTAL_CHS = ['F3', 'F4', 'Fz', 'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8']

SR_BANDS = {
    'SR1': (7.2, 8.4),
    'SR2': (13.5, 14.5),
    'SR3': (19.5, 20.4),
    'SR4': (25.0, 26.0),
}

PHI = 1.6180339887
PHI_HARM_BANDS = {
    f'phi{n}': (7.82 * PHI ** n - 0.6, 7.82 * PHI ** n + 0.6)
    for n in [1, 2, 3]
}

WIN_SEC = 2.0
HOP_SEC = 0.5
TGRID = np.arange(-15.0, 15.0 + 0.25, 0.5)
MIN_GAP_FROM_EVENT = 30.0

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


def sliding_band_powers(y, fs, bands, win_sec=WIN_SEC, hop_sec=HOP_SEC):
    nperseg = int(round(win_sec * fs))
    nhop = int(round(hop_sec * fs))
    nfft = nperseg * 4
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    w = signal.windows.hann(nperseg)
    wp = np.sum(w ** 2)
    masks = {name: (freqs >= lo) & (freqs <= hi) for name, (lo, hi) in bands.items()}
    t_cent = []
    powers = {name: [] for name in bands}
    for i in range(0, len(y) - nperseg + 1, nhop):
        seg = y[i:i + nperseg] - np.mean(y[i:i + nperseg])
        X = np.fft.rfft(seg * w, nfft)
        psd = (np.abs(X) ** 2) / (fs * wp)
        psd[1:-1] *= 2.0
        for name, m in masks.items():
            powers[name].append(np.nanmean(psd[m]))
        t_cent.append((i + nperseg / 2) / fs)
    return np.array(t_cent), {k: np.array(v) for k, v in powers.items()}


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
    if fs < 130:
        return None
    present = [c for c in FRONTAL_CHS if c in raw.ch_names]
    if len(present) < 2:
        return None
    y = raw.get_data(picks=present).mean(axis=0) * 1e6
    t_end = raw.times[-1]

    all_bands = {**SR_BANDS, **PHI_HARM_BANDS}
    t_pow, pow_all = sliding_band_powers(y, fs, all_bands)

    eps = 1e-20
    hsi_V1 = np.log10(pow_all['SR1'] + eps) - np.log10(pow_all['SR3'] + eps)
    hsi_V2 = np.log10(pow_all['SR1'] + eps) - np.log10(
        (pow_all['SR2'] + pow_all['SR3'] + pow_all['SR4']) / 3 + eps)
    hsi_V3 = -hsi_V2
    phi_mean = (pow_all['phi1'] + pow_all['phi2'] + pow_all['phi3']) / 3
    hsi_V4 = np.log10(pow_all['SR1'] + eps) - np.log10(phi_mean + eps)

    def center(x):
        return x - np.nanmedian(x)
    V1c = center(hsi_V1); V2c = center(hsi_V2); V3c = center(hsi_V3); V4c = center(hsi_V4)

    t_events = events['t0_net'].astype(float).values
    t_controls = sample_control_times(t_events, len(t_events), t_end,
                                        seed=abs(hash(sub_id)) % (2**31))

    def align_all(t_center):
        rel = t_pow - t_center
        mask = (rel >= TGRID[0] - 1) & (rel <= TGRID[-1] + 1)
        if mask.sum() == 0:
            return None
        out = {}
        for name, arr in [('V1', V1c), ('V2', V2c), ('V3', V3c), ('V4', V4c)]:
            out[name] = np.interp(TGRID, rel[mask], arr[mask],
                                   left=np.nan, right=np.nan)
        return out

    ev_traces = {'V1': [], 'V2': [], 'V3': [], 'V4': []}
    ct_traces = {'V1': [], 'V2': [], 'V3': [], 'V4': []}
    for t0 in t_events:
        d = align_all(t0)
        if d:
            for k in ev_traces:
                ev_traces[k].append(d[k])
    for tc in t_controls:
        d = align_all(tc)
        if d:
            for k in ct_traces:
                ct_traces[k].append(d[k])
    if not ev_traces['V1'] or not ct_traces['V1']:
        return None
    return {
        'subject_id': sub_id,
        'n_ev': len(ev_traces['V1']),
        'n_ct': len(ct_traces['V1']),
        'ev_V1': np.nanmean(np.array(ev_traces['V1']), axis=0),
        'ev_V2': np.nanmean(np.array(ev_traces['V2']), axis=0),
        'ev_V3': np.nanmean(np.array(ev_traces['V3']), axis=0),
        'ev_V4': np.nanmean(np.array(ev_traces['V4']), axis=0),
        'ct_V1': np.nanmean(np.array(ct_traces['V1']), axis=0),
        'ct_V2': np.nanmean(np.array(ct_traces['V2']), axis=0),
        'ct_V3': np.nanmean(np.array(ct_traces['V3']), axis=0),
        'ct_V4': np.nanmean(np.array(ct_traces['V4']), axis=0),
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

    labels = {
        'V1': 'V1: log(SR1/SR3)',
        'V2': 'V2: log(SR1 / mean(SR2,SR3,SR4))',
        'V3': 'V3: log(mean(SR2,SR3,SR4)/SR1) — inverse of V2',
        'V4': 'V4: log(SR1 / mean(φ¹,φ²,φ³ × SR1))',
    }

    print(f"\n=== {args.cohort} composite · Peak ΔHSI at t=0-4s and dip for each variant ===")
    for v in ['V1', 'V2', 'V3', 'V4']:
        ev = np.array([r[f'ev_{v}'] for r in results])
        ct = np.array([r[f'ct_{v}'] for r in results])
        grand_ev, _, _ = bootstrap_ci(ev)
        grand_ct, _, _ = bootstrap_ci(ct)
        peri_mask = (TGRID >= 0) & (TGRID <= 4)
        ev_peak = np.nanmax(grand_ev[peri_mask])
        ev_peak_t = TGRID[peri_mask][np.nanargmax(grand_ev[peri_mask])]
        ev_dip = np.nanmin(grand_ev[peri_mask])
        ev_dip_t = TGRID[peri_mask][np.nanargmin(grand_ev[peri_mask])]
        ct_peak = np.nanmax(grand_ct[peri_mask])
        ct_dip = np.nanmin(grand_ct[peri_mask])
        print(f"  {labels[v]}")
        print(f"    peak: {ev_peak:+.4f} @ t={ev_peak_t:+.1f}s (ctl {ct_peak:+.4f})   "
              f"dip: {ev_dip:+.4f} @ t={ev_dip_t:+.1f}s (ctl {ct_dip:+.4f})")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax_idx, v in enumerate(['V1', 'V2', 'V3', 'V4']):
        ax = axes.flatten()[ax_idx]
        ev = np.array([r[f'ev_{v}'] for r in results])
        ct = np.array([r[f'ct_{v}'] for r in results])
        ge, loe, hie = bootstrap_ci(ev)
        gc, loc, hic = bootstrap_ci(ct)
        ax.plot(TGRID, ge, color='firebrick', lw=2, label='event')
        ax.fill_between(TGRID, loe, hie, color='firebrick', alpha=0.22)
        ax.plot(TGRID, gc, color='gray', lw=2, label='control')
        ax.fill_between(TGRID, loc, hic, color='gray', alpha=0.22)
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', ls='--', lw=0.6)
        ax.set_xlabel('time rel. t0_net (s)')
        ax.set_ylabel('ΔHSI (centered on own median)')
        ax.set_title(labels[v])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle(f'B39 composite — HSI direction under four definitions '
                 f'({args.cohort} Q4, n={len(results)})', y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hsi_variants.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\n=== {args.cohort} composite · Per-subject distribution of event ΔHSI_V1 at t=+2s ===")
    t2_idx = int(np.argmin(np.abs(TGRID - 2.0)))
    ev_V1_subj = np.array([r['ev_V1'][t2_idx] for r in results])
    n_rise = np.sum(ev_V1_subj > 0)
    n_dip = np.sum(ev_V1_subj < 0)
    print(f"  Subjects with event HSI RISE at t=+2: {n_rise} / {len(ev_V1_subj)} "
          f"({100*n_rise/len(ev_V1_subj):.0f}%)")
    print(f"  Subjects with event HSI DIP  at t=+2: {n_dip} / {len(ev_V1_subj)} "
          f"({100*n_dip/len(ev_V1_subj):.0f}%)")
    print(f"  Subject median: {np.nanmedian(ev_V1_subj):+.4f}")

    rows = []
    for r in results:
        for v in ['V1', 'V2', 'V3', 'V4']:
            for ti, t in enumerate(TGRID):
                rows.append({'subject_id': r['subject_id'], 'variant': v,
                              't': t, 'event': r[f'ev_{v}'][ti], 'control': r[f'ct_{v}'][ti]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'hsi_variants.csv'),
                               index=False)
    print(f"\nSaved: {out_dir}/hsi_variants.png")


if __name__ == '__main__':
    main()
