#!/usr/bin/env python3
"""
A6b MSC co-dip test — revisit under fixed vectorized MSC.

Original envelope A6b claimed that MSC co-dips with env/R/PLV at the joint
nadir (−1.3 s rel t0_net). A8 retracted this as a code bug: MSC input was
double-bandpassed, saturating at 1.0, making MSC a zero-information stream.

The MSC vectorized fix in lib/detect_ignition.py::_welch_msc_vectorized uses
RAW signals (not bandpassed) and produces a valid MSC at F0=7.83 Hz. This
script tests whether, under the FIXED MSC, the original A6b claim (MSC
co-dips with the other three streams) is actually true.

Uses composite v2 events and lib/detect_ignition.py::_composite_streams for
4-stream computation (env + R + PLV + MSC with vectorized Welch). Extract
peri-onset 4-stream traces aligned to t0_net. Grand-mean per stream with
subject-level bootstrap CI. Report dip time and magnitude per stream.

Cohort-parameterized.

Usage:
    python scripts/sie_a6b_msc_codip_composite.py --cohort lemon
    python scripts/sie_a6b_msc_codip_composite.py --cohort lemon_EO
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
from lib.detect_ignition import _composite_streams

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

F0 = 7.83
R_BAND = (7.2, 8.4)
PRE_SEC = 10.0
POST_SEC = 10.0
PAD_SEC = 2.0
STEP_SEC = 0.1
WIN_SEC = 1.0

TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC / 2, STEP_SEC)

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
    t_end = raw.times[-1]

    env_rows, R_rows, P_rows, M_rows = [], [], [], []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end:
            continue
        i0 = int(round(lo * fs))
        i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue
        try:
            t_c, env, R, P, M = _composite_streams(X_seg, fs,
                                                    f0=F0, half_bw=0.6,
                                                    R_band=R_BAND,
                                                    step_sec=STEP_SEC, win_sec=WIN_SEC)
        except Exception:
            continue
        rel = t_c - PAD_SEC - PRE_SEC
        env_rows.append(np.interp(TGRID, rel, env, left=np.nan, right=np.nan))
        R_rows.append(np.interp(TGRID, rel, R, left=np.nan, right=np.nan))
        P_rows.append(np.interp(TGRID, rel, P, left=np.nan, right=np.nan))
        M_rows.append(np.interp(TGRID, rel, M, left=np.nan, right=np.nan))

    if not env_rows:
        return None
    return {
        'subject_id': sub_id,
        'n_events': len(env_rows),
        'env': np.nanmean(np.array(env_rows), axis=0),
        'R': np.nanmean(np.array(R_rows), axis=0),
        'P': np.nanmean(np.array(P_rows), axis=0),
        'M': np.nanmean(np.array(M_rows), axis=0),
    }


def bootstrap_ci(mat, n_boot=1000, seed=0):
    mat = np.array(mat)
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    if len(mat) < 2:
        return np.nanmean(mat, axis=0), np.full(mat.shape[1], np.nan), np.full(mat.shape[1], np.nan)
    rng = np.random.default_rng(seed)
    grand = np.nanmean(mat, axis=0)
    boots = np.zeros((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, len(mat), size=len(mat))
        boots[b] = np.nanmean(mat[idx], axis=0)
    return grand, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def dip_peak(trace, t=TGRID, dip_win=(-3.0, 0.4), peak_win=(-0.5, 2.5)):
    dip_m = (t >= dip_win[0]) & (t <= dip_win[1])
    pk_m = (t >= peak_win[0]) & (t <= peak_win[1])
    if not np.any(dip_m) or not np.any(pk_m):
        return np.nan, np.nan, np.nan, np.nan
    di = int(np.nanargmin(trace[dip_m]))
    pi = int(np.nanargmax(trace[pk_m]))
    return (float(t[dip_m][di]), float(trace[dip_m][di]),
            float(t[pk_m][pi]), float(trace[pk_m][pi]))


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
    print(f"Workers: {args.workers}   (4-stream with fixed vectorized MSC)")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")
    n_events = sum(r['n_events'] for r in results)
    print(f"Total events: {n_events}")

    streams = {}
    for name in ['env', 'R', 'P', 'M']:
        mat = np.array([r[name] for r in results])
        grand, lo, hi = bootstrap_ci(mat)
        streams[name] = {'grand': grand, 'lo': lo, 'hi': hi, 'mat': mat}

    print(f"\n=== {args.cohort} composite · 4-stream peri-onset (fixed MSC) ===")
    print(f"(envelope A3: all 3 streams dip at −1.30 s, peak at +1.20 s)")
    print(f"(A6b original retracted: MSC saturated at 1.0 due to double-bandpass code bug)")
    print(f"(A6b fixed-MSC test: does MSC ACTUALLY co-dip at −1.30 s?)")
    print(f"{'stream':<8} {'nadir time':<12} {'nadir val':<12} {'peak time':<12} {'peak val':<12} {'range':<10}")
    stats = {}
    for name, label in [('env', 'env z'), ('R', 'Kuramoto R'),
                         ('P', 'mean PLV'), ('M', 'MSC (fixed)')]:
        nd_t, nd_v, pk_t, pk_v = dip_peak(streams[name]['grand'])
        stats[name] = {'nadir_t': nd_t, 'nadir_v': nd_v, 'peak_t': pk_t, 'peak_v': pk_v}
        print(f"{label:<12} {nd_t:+.2f} s     {nd_v:+.4f}      {pk_t:+.2f} s     {pk_v:+.4f}      {pk_v - nd_v:+.4f}")

    # Cross-check: does MSC co-dip with the other 3?
    msc_dip_t = stats['M']['nadir_t']
    other_dips = [stats['env']['nadir_t'], stats['R']['nadir_t'], stats['P']['nadir_t']]
    diffs = [msc_dip_t - o for o in other_dips]
    print(f"\n  MSC nadir @ {msc_dip_t:+.2f} s")
    print(f"  env/R/PLV nadirs @ {other_dips} s")
    print(f"  MSC − (env/R/PLV) = {diffs} s")
    within_500ms = all(abs(d) <= 0.5 for d in diffs)
    if within_500ms:
        print(f"  → All within ±0.5 s: MSC CO-DIPS with the other 3 streams ✓")
        print(f"     (A6b original claim resurrected under fixed MSC)")
    else:
        print(f"  → MSC does NOT co-dip: max Δ = {max(abs(d) for d in diffs):.2f} s")
        print(f"     (A6b claim still fails even under fixed MSC)")

    # Save + plot
    pd.DataFrame({
        't': TGRID,
        'env_grand': streams['env']['grand'],
        'R_grand':   streams['R']['grand'],
        'P_grand':   streams['P']['grand'],
        'M_grand':   streams['M']['grand'],
    }).to_csv(os.path.join(out_dir, 'a6b_msc_codip_composite.csv'), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panels = [
        ('env', 'envelope z', 'darkorange', (0, 0)),
        ('R',   'Kuramoto R', 'seagreen',   (0, 1)),
        ('P',   'mean PLV',   'purple',     (1, 0)),
        ('M',   'MSC (fixed)', 'steelblue', (1, 1)),
    ]
    for name, label, color, pos in panels:
        ax = axes[pos]
        g = streams[name]['grand']; lo = streams[name]['lo']; hi = streams[name]['hi']
        ax.plot(TGRID, g, color=color, lw=2)
        ax.fill_between(TGRID, lo, hi, color=color, alpha=0.22)
        ax.axvline(0, color='k', ls='--', lw=0.6)
        ax.axvline(stats[name]['nadir_t'], color='red', ls=':', lw=1)
        ax.axvline(stats[name]['peak_t'],  color='blue', ls=':', lw=1)
        ax.set_ylabel(label)
        ax.set_title(f'{label}: nadir @ {stats[name]["nadir_t"]:+.2f}s, peak @ {stats[name]["peak_t"]:+.2f}s')
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel('time rel t0_net (s)')

    plt.suptitle(f'A6b MSC co-dip test · fixed MSC · {args.cohort} composite v2\n'
                 f'{len(results)} subjects, {n_events} events',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'a6b_msc_codip_composite.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/a6b_msc_codip_composite.png")


if __name__ == '__main__':
    main()
