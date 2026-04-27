#!/usr/bin/env python3
"""
B10 re-run on composite v2 detector.

Pseudo-event null: sample n_events pseudo-times per subject that are ≥30 s from
any real composite-v2 t0_net. Compute phase-jump rate peri-pseudo-nadir
(approximated at pseudo_t − 1.30 s, matching composite A7 nadir offset).
Compare peri-nadir [-1, +1] s / baseline [-8, -4] s elevation to real events.

Uses `sample_pseudo_times` and `phase_jumps` helpers from the envelope scripts.

Cohort-parameterized.

Usage:
    python scripts/sie_phase_reset_null_composite.py --cohort lemon
    python scripts/sie_phase_reset_null_composite.py --cohort lemon_EO
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
from scripts.sie_mechanism_battery import phase_jumps
from scripts.sie_propagation_null_random import sample_pseudo_times
from scripts.sie_perionset_multistream import PRE_SEC, POST_SEC, PAD_SEC

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

PJ_TGRID = np.arange(-8.0, 8.0 + 0.05, 0.1)
NADIR_OFFSET = -1.30

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
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X_all = raw.get_data() * 1e6
    t_end = raw.times[-1]

    t_events = events['t0_net'].values.astype(float)
    n_target = int(len(t_events))
    if n_target == 0:
        return None
    t_pseudo = sample_pseudo_times(t_events, n_target, t_end,
                                    seed=abs(hash(sub_id)) % (2**31))

    traces = []
    for t0 in t_pseudo:
        lo_t = t0 - PRE_SEC - PAD_SEC
        hi_t = t0 + POST_SEC + PAD_SEC
        if lo_t < 0 or hi_t > t_end:
            continue
        i0 = int(round(lo_t * fs)); i1 = int(round(hi_t * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi_t - lo_t) * fs * 0.95)):
            continue
        try:
            t_pj, pj = phase_jumps(X_seg, fs)
            rel_pj = t_pj - PAD_SEC - PRE_SEC - NADIR_OFFSET
            traces.append(np.interp(PJ_TGRID, rel_pj, pj,
                                     left=np.nan, right=np.nan))
        except Exception:
            continue
    if not traces:
        return None
    return {
        'subject_id': sub_id,
        'n_pseudo': len(traces),
        'pj_mean': np.nanmean(np.array(traces), axis=0),
    }


def bootstrap_ci(mat, n_boot=1000, seed=0):
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


def elevation(traj, t=PJ_TGRID):
    peri = (t >= -1) & (t <= +1)
    base = (t >= -8) & (t <= -4)
    m_peri = np.nanmean(traj[peri]); m_base = np.nanmean(traj[base])
    return (m_peri / m_base) if m_base > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'mechanism_battery', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 2)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} composite  ·  subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")
    total_pseudo = sum(r['n_pseudo'] for r in results)
    print(f"Total pseudo-events: {total_pseudo}")

    pj_mat = np.array([r['pj_mean'] for r in results])
    grand, lo, hi = bootstrap_ci(pj_mat)

    per_sub_elev = np.array([elevation(r['pj_mean']) for r in results])
    per_sub_elev = per_sub_elev[np.isfinite(per_sub_elev)]

    print(f"\n=== {args.cohort} composite · pseudo-event phase-jump elevation ===")
    print(f"  Grand-mean elevation: {elevation(grand):.3f}×")
    print(f"  Per-subject elevation: median {np.median(per_sub_elev):.3f}   "
          f"IQR [{np.percentile(per_sub_elev,25):.3f}, {np.percentile(per_sub_elev,75):.3f}]")
    print(f"  Subjects with elevation > 1.2×: {(per_sub_elev > 1.2).mean()*100:.1f}%")

    print(f"\n  (Compare to envelope B10: pseudo-event grand 1.15×, real Q1 1.34×, real Q4 1.49×)")

    pd.DataFrame({
        'subject_id': [r['subject_id'] for r in results
                        if np.isfinite(elevation(r['pj_mean']))],
        'elevation': per_sub_elev,
    }).to_csv(os.path.join(out_dir, 'phase_reset_null_random.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(PJ_TGRID, grand, color='gray', lw=2,
            label=f'pseudo-event grand mean (elev {elevation(grand):.2f}×)')
    ax.fill_between(PJ_TGRID, lo, hi, color='gray', alpha=0.25)
    ax.axvspan(-8, -4, alpha=0.08, color='blue', label='baseline')
    ax.axvspan(-1, +1, alpha=0.08, color='red', label='peri-nadir')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('time rel pseudo-nadir (s)')
    ax.set_ylabel('phase-jump rate (counts/100ms)')
    ax.set_title(f'B10 · composite pseudo-event null · {len(pj_mat)} subjects')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(per_sub_elev, bins=30, color='gray', edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(1.0, color='k', lw=0.8, label='no elevation')
    ax.axvline(np.median(per_sub_elev), color='gray', ls='--', lw=1.5,
               label=f'pseudo median {np.median(per_sub_elev):.2f}×')
    ax.axvline(1.15, color='dimgray', ls=':', lw=1.5, label='envelope pseudo 1.15×')
    ax.axvline(1.34, color='#4575b4', ls=':', lw=1.5, label='envelope real Q1 1.34×')
    ax.axvline(1.49, color='#d73027', ls=':', lw=1.5, label='envelope real Q4 1.49×')
    ax.set_xlabel('peri-nadir / baseline ratio')
    ax.set_ylabel('subjects')
    ax.set_title('Per-subject pseudo-event elevation')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    plt.suptitle(f'B10 · phase-reset null · {args.cohort} composite v2', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phase_reset_null_random.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/phase_reset_null_random.png")


if __name__ == '__main__':
    main()
