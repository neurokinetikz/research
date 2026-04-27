#!/usr/bin/env python3
"""
A6 re-run on composite v2 detector.

Per-event dip and rebound-peak detection for env z, Kuramoto R, mean PLV on
composite v2 events. Reports:
  - Per-stream dip time distribution (median, IQR)
  - Per-stream rebound peak time distribution
  - Pairwise lead-lag (E−R, E−P, R−P): median, IQR, % within ±0.1 s
  - Dip depth (peak − nadir) per stream

Cohort-parameterized; reads composite v2 events. Outputs to
outputs/schumann/images/perionset/<cohort>_composite/.

Usage:
    python scripts/sie_dip_rebound_analysis_composite.py --cohort lemon
    python scripts/sie_dip_rebound_analysis_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)
from scripts.sie_perionset_triple_average_composite import (
    compute_streams, TGRID, PRE_SEC, POST_SEC, PAD_SEC,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

DIP_WINDOW = (-3.0, 0.4)
PEAK_WINDOW = (-0.5, 2.5)

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


def find_dip_peak(t_rel, y):
    dip_mask = (t_rel >= DIP_WINDOW[0]) & (t_rel <= DIP_WINDOW[1])
    peak_mask = (t_rel >= PEAK_WINDOW[0]) & (t_rel <= PEAK_WINDOW[1])
    if not dip_mask.any() or not peak_mask.any():
        return None
    y_d = y[dip_mask]
    t_d = t_rel[dip_mask]
    idx = int(np.nanargmin(y_d))
    t_dip = float(t_d[idx])
    y_dip = float(y_d[idx])
    y_p = y[peak_mask]
    t_p = t_rel[peak_mask]
    idx = int(np.nanargmax(y_p))
    t_peak = float(t_p[idx])
    y_peak = float(y_p[idx])
    return t_dip, y_dip, t_peak, y_peak


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
    if len(events) == 0:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X_all = raw.get_data() * 1e6
    t_end_rec = raw.times[-1]

    rows = []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end_rec:
            continue
        i0 = int(round(lo * fs))
        i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue
        try:
            (t_env, zenv), (tR, R), (tP, P) = compute_streams(X_seg, fs)
        except Exception:
            continue
        rel_env = t_env - PAD_SEC - PRE_SEC
        rel_R = tR - PAD_SEC - PRE_SEC
        rel_P = tP - PAD_SEC - PRE_SEC

        info_E = find_dip_peak(rel_env, zenv)
        info_R = find_dip_peak(rel_R, R)
        info_P = find_dip_peak(rel_P, P)
        if not info_E or not info_R or not info_P:
            continue
        tE_dip, yE_dip, tE_peak, yE_peak = info_E
        tR_dip, yR_dip, tR_peak, yR_peak = info_R
        tP_dip, yP_dip, tP_peak, yP_peak = info_P
        rows.append({
            'subject_id': sub_id, 't0_net': t0,
            'tE_dip': tE_dip, 'yE_dip': yE_dip,
            'tE_peak': tE_peak, 'yE_peak': yE_peak,
            'tR_dip': tR_dip, 'yR_dip': yR_dip,
            'tR_peak': tR_peak, 'yR_peak': yR_peak,
            'tP_dip': tP_dip, 'yP_dip': yP_dip,
            'tP_peak': tP_peak, 'yP_peak': yP_peak,
            'E_depth': yE_peak - yE_dip,
            'R_depth': yR_peak - yR_dip,
            'P_depth': yP_peak - yP_dip,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'perionset', f'{args.cohort}_composite')
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
    all_rows = []
    for r in results:
        if r:
            all_rows.extend(r)
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, 'dip_rebound_per_event.csv')
    df.to_csv(csv_path, index=False)
    print(f"Events: {len(df)}  Subjects: {df['subject_id'].nunique()}")

    print(f"\n=== {args.cohort} composite · dip times (median, IQR) ===")
    for col, label in [('tE_dip', 'envelope z'), ('tR_dip', 'Kuramoto R'), ('tP_dip', 'mean PLV')]:
        vals = df[col].dropna()
        print(f"  {label:12s}: {np.median(vals):+.3f} s   "
              f"IQR [{np.percentile(vals,25):+.3f}, {np.percentile(vals,75):+.3f}]   "
              f"std {np.std(vals):.3f}")
    print(f"\n=== {args.cohort} composite · peak times (median, IQR) ===")
    for col, label in [('tE_peak', 'envelope z'), ('tR_peak', 'Kuramoto R'), ('tP_peak', 'mean PLV')]:
        vals = df[col].dropna()
        print(f"  {label:12s}: {np.median(vals):+.3f} s   "
              f"IQR [{np.percentile(vals,25):+.3f}, {np.percentile(vals,75):+.3f}]")

    print(f"\n=== {args.cohort} composite · pairwise dip lead-lag ===")
    pairs = [
        ('E − R',  df['tE_dip'] - df['tR_dip']),
        ('E − P',  df['tE_dip'] - df['tP_dip']),
        ('R − P',  df['tR_dip'] - df['tP_dip']),
    ]
    for label, diffs in pairs:
        d = diffs.dropna()
        frac_left_first = float((d < 0).mean())
        frac_concurrent = float((d.abs() < 0.1).mean())
        print(f"  {label}: median {np.median(d):+.3f} s, "
              f"IQR [{np.percentile(d,25):+.3f}, {np.percentile(d,75):+.3f}]  "
              f"left-leads {frac_left_first*100:.1f}%  (|Δ|<0.1s {frac_concurrent*100:.1f}%)")

    print(f"\n=== {args.cohort} composite · dip depths (peak − nadir) ===")
    for col, label in [('E_depth', 'envelope z'), ('R_depth', 'Kuramoto R'), ('P_depth', 'mean PLV')]:
        vals = df[col].dropna()
        print(f"  {label:12s}: median {np.median(vals):.3f}   IQR [{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}]")

    # Save summary for report
    summary = {
        'tE_dip_median': float(np.median(df['tE_dip'])),
        'tR_dip_median': float(np.median(df['tR_dip'])),
        'tP_dip_median': float(np.median(df['tP_dip'])),
        'tE_peak_median': float(np.median(df['tE_peak'])),
        'tR_peak_median': float(np.median(df['tR_peak'])),
        'tP_peak_median': float(np.median(df['tP_peak'])),
        'ER_lag_median': float(np.median(df['tE_dip'] - df['tR_dip'])),
        'EP_lag_median': float(np.median(df['tE_dip'] - df['tP_dip'])),
        'RP_lag_median': float(np.median(df['tR_dip'] - df['tP_dip'])),
        'ER_concurrent_pct': float(((df['tE_dip'] - df['tR_dip']).abs() < 0.1).mean() * 100),
        'EP_concurrent_pct': float(((df['tE_dip'] - df['tP_dip']).abs() < 0.1).mean() * 100),
        'RP_concurrent_pct': float(((df['tR_dip'] - df['tP_dip']).abs() < 0.1).mean() * 100),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'dip_rebound_summary.csv'), index=False)
    print(f"\nSaved: {out_dir}/dip_rebound_per_event.csv")
    print(f"Saved: {out_dir}/dip_rebound_summary.csv")


if __name__ == '__main__':
    main()
