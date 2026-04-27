#!/usr/bin/env python3
"""B47-equivalent posterior-vs-anterior SR1 contrast for composite-extracted
cohorts. Runs per-cohort Q4 filter (if template_ρ available), computes
event-locked SR1 event/baseline ratio for posterior and anterior region
means, and reports per-subject contrast statistics.

Designed to run locally on cohorts with local raw EEG access.
"""
from __future__ import annotations
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import wilcoxon
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (load_lemon, load_dortmund,
                                          load_srm, load_chbmp)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

EVENTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
QUALITY_ROOT = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                             'schumann', 'images', 'quality')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
os.makedirs(OUT_DIR, exist_ok=True)

SR1_BAND = (7.0, 8.3)
EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
QUALITY_TOP_Q = 0.75


def is_posterior(ch):
    n = ch.upper()
    if n.startswith('FP'):
        return False
    return (any(n.startswith(p) for p in ('O', 'PO', 'P', 'TP')) or
            n in ('T7', 'T8', 'T5', 'T6'))


def is_anterior(ch):
    n = ch.upper()
    if is_posterior(ch):
        return False
    return any(n.startswith(p) for p in ('F', 'AF', 'FP'))


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
        denom = (y0 - 2*y1 + y2)
        delta = 0.5*(y0 - y2)/denom if denom != 0 else 0.0
        delta = max(-1.0, min(1.0, delta))
        return float(x[k] + delta*(x[1]-x[0]))
    return float(x[k])


def region_picks(raw):
    post = [i for i, ch in enumerate(raw.ch_names) if is_posterior(ch)]
    ant = [i for i, ch in enumerate(raw.ch_names) if is_anterior(ch)]
    return post, ant


def process_subject(args):
    cohort, sub_id, events_path, quality_csv = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 3:
        return None
    # Q4 filter
    if quality_csv and os.path.isfile(quality_csv):
        try:
            qual = pd.read_csv(quality_csv).dropna(subset=['template_rho'])
            qsub = qual[qual['subject_id'] == sub_id]
            if len(qsub) >= 4:
                thr = qsub['template_rho'].quantile(QUALITY_TOP_Q)
                q4t = set(qsub.loc[qsub['template_rho'] >= thr,
                                    't0_net'].round(3).values)
                events['t0r'] = events['t0_net'].round(3)
                events = events[events['t0r'].isin(q4t)]
        except Exception:
            pass
    if len(events) < 1:
        return None
    # Load raw
    try:
        if cohort.startswith('lemon_EO'):
            raw = load_lemon(sub_id, condition='EO')
        elif cohort.startswith('lemon'):
            raw = load_lemon(sub_id, condition='EC')
        elif cohort.startswith('dortmund'):
            task = 'EyesOpen' if '_EO' in cohort else 'EyesClosed'
            acq = 'post' if '_post' in cohort else 'pre'
            ses = '2' if '_ses2' in cohort else '1'
            raw = load_dortmund(sub_id, task=task, acq=acq, ses=ses)
        elif cohort.startswith('srm'):
            raw = load_srm(sub_id, ses='t1')
        elif cohort.startswith('chbmp'):
            raw = load_chbmp(sub_id)
        else:
            return None
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 40:
        return None
    post_idx, ant_idx = region_picks(raw)
    if len(post_idx) < 5 or len(ant_idx) < 5:
        return None
    X = raw.get_data() * 1e6
    post_sig = X[post_idx].mean(axis=0)
    ant_sig = X[ant_idx].mean(axis=0)
    nper = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nper * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0/fs)
    mask = (freqs >= SR1_BAND[0]) & (freqs <= SR1_BAND[1])

    def ratio(sig_1d):
        base_rows = []
        for i in range(0, len(sig_1d)-nper+1, nhop):
            base_rows.append(welch_one(sig_1d[i:i+nper], fs, nfft)[mask])
        if len(base_rows) < 10:
            return np.nan
        baseline = np.nanmedian(np.array(base_rows), axis=0)
        ev_rows = []
        for _, ev in events.iterrows():
            tc = float(ev['t0_net']) + EV_LAG_S
            i0 = int(round((tc - EV_WIN_SEC/2) * fs))
            i1 = i0 + nper
            if i0 < 0 or i1 > len(sig_1d): continue
            ev_rows.append(welch_one(sig_1d[i0:i1], fs, nfft)[mask])
        if not ev_rows:
            return np.nan
        ev_avg = np.nanmean(np.array(ev_rows), axis=0)
        r = (ev_avg + 1e-20) / (baseline + 1e-20)
        return float(np.max(r))

    p_r = ratio(post_sig); a_r = ratio(ant_sig)
    if np.isnan(p_r) or np.isnan(a_r):
        return None
    return {'cohort': cohort, 'subject_id': sub_id,
            'sr1_ratio_posterior': p_r, 'sr1_ratio_anterior': a_r,
            'sr1_contrast': p_r - a_r,
            'n_events': len(events)}


def build_tasks(cohort):
    events_dir = os.path.join(EVENTS_ROOT, cohort)
    if not os.path.isdir(events_dir):
        return []
    sum_path = os.path.join(events_dir, 'extraction_summary.csv')
    if not os.path.isfile(sum_path):
        return []
    summary = pd.read_csv(sum_path)
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=3)]
    quality_csv = os.path.join(QUALITY_ROOT,
                                f'per_event_quality_{cohort}.csv')
    if not os.path.isfile(quality_csv):
        # fallback name
        quality_csv = None
    tasks = []
    for _, r in ok.iterrows():
        sub = r['subject_id']
        ep = os.path.join(events_dir, f'{sub}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((cohort, sub, ep, quality_csv))
    return tasks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohorts', nargs='+', default=None,
                    help='Cohort directory names under exports_sie/')
    args = ap.parse_args()

    if args.cohorts is None:
        args.cohorts = [d for d in sorted(os.listdir(EVENTS_ROOT))
                        if d.endswith('_composite')]
    all_tasks = []
    for c in args.cohorts:
        t = build_tasks(c)
        print(f"  {c}: {len(t)} tasks")
        all_tasks += t
    print(f"Total: {len(all_tasks)}")

    n_workers = int(os.environ.get('SIE_WORKERS',
                                    min(8, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, all_tasks)
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR,
                            'composite_b47_posterior_sr1.csv'), index=False)

    print(f"\n{'cohort':<45}{'n':>5}{'post med':>10}{'ant med':>10}"
          f"{'contrast':>11}{'% p>a':>8}{'Wilcoxon p':>14}")
    print('-' * 105)
    for c in args.cohorts:
        sub = df[df['cohort'] == c]
        if len(sub) < 5:
            continue
        contrast = sub['sr1_contrast'].values
        pm = sub['sr1_ratio_posterior'].median()
        am = sub['sr1_ratio_anterior'].median()
        cm = np.median(contrast)
        pct = (contrast > 0).mean() * 100
        try:
            _, p = wilcoxon(contrast)
        except Exception:
            p = np.nan
        print(f"{c:<45}{len(sub):>5}{pm:>10.3f}{am:>10.3f}"
              f"{cm:>+11.3f}{pct:>7.0f}%{p:>14.2g}")


if __name__ == '__main__':
    main()
