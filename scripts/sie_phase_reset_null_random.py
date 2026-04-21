#!/usr/bin/env python3
"""
B10 — B5 phase-reset null: does peri-nadir phase-jump elevation appear in
random non-event windows too?

For each subject, sample n_events pseudo-events from random times ≥30 s from
any real t0_net (same as B9). For each pseudo-event:
  1. Extract ±12 s window
  2. compute_streams_4way → find joint nadir via find_nadir
  3. Count cross-channel phase jumps per 100-ms bin (narrowband 7.2-8.4 Hz)
  4. Align on nadir, aggregate subject-mean trace

Compute subject-level peri-nadir [-1, +1] s / baseline [-8, -4] s elevation.
Compare pseudo elevation to real-event B8 elevations:
  - Real Q1: 1.34×
  - Real Q4: 1.49×

If pseudo elevation ≈ 1 → phase-reset is event-specific (mechanism survives).
If pseudo elevation ≈ real → phase-reset is a baseline feature and retracts.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_dip_onset_and_narrow_fooof import compute_streams_4way
from scripts.sie_perionset_multistream import (
    PRE_SEC, POST_SEC, PAD_SEC, find_nadir,
)
from scripts.sie_mechanism_battery import phase_jumps
from scripts.sie_propagation_null_random import sample_pseudo_times

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'quality')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

PJ_TGRID = np.arange(-8.0, 8.0 + 0.05, 0.1)


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
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
            t_c, env, R, P, M = compute_streams_4way(X_seg, fs)
            rel = t_c - PAD_SEC - PRE_SEC
            nadir = find_nadir(rel, env, R, P, M)
            if not np.isfinite(nadir):
                continue
            t_pj, pj = phase_jumps(X_seg, fs)
            rel_pj = t_pj - PAD_SEC - PRE_SEC - nadir
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
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 2)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")
    total_pseudo = sum(r['n_pseudo'] for r in results)
    print(f"Total pseudo-events: {total_pseudo}")

    pj_mat = np.array([r['pj_mean'] for r in results])
    grand, lo, hi = bootstrap_ci(pj_mat)

    # Per-subject elevation
    per_sub_elev = np.array([elevation(r['pj_mean']) for r in results])
    per_sub_elev = per_sub_elev[np.isfinite(per_sub_elev)]

    print(f"\n=== Pseudo-event peri-nadir phase-jump elevation ===")
    print(f"  Grand-mean trace elevation: {elevation(grand):.3f}×")
    print(f"  Per-subject elevation: median {np.median(per_sub_elev):.3f}  "
          f"IQR [{np.percentile(per_sub_elev,25):.3f}, {np.percentile(per_sub_elev,75):.3f}]")
    print(f"  Fraction of subjects with elevation > 1.2×: {(per_sub_elev>1.2).mean()*100:.1f}%")

    print(f"\n=== Real-event elevations (from B8) ===")
    print(f"  Q1: 1.34×")
    print(f"  Q4: 1.49×")

    pd.DataFrame({'subject_id': [r['subject_id'] for r in results if np.isfinite(elevation(r['pj_mean']))],
                   'elevation': per_sub_elev}).to_csv(
        os.path.join(OUT_DIR, 'phase_reset_null_random.csv'), index=False)

    # Plot: pseudo grand-mean trace + CI with reference lines for Q1 and Q4
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(PJ_TGRID, grand, color='gray', lw=2,
            label=f'pseudo-event grand mean\nelevation {elevation(grand):.2f}×')
    ax.fill_between(PJ_TGRID, lo, hi, color='gray', alpha=0.25)
    # Shade baseline and peri-nadir windows
    ax.axvspan(-8, -4, alpha=0.08, color='blue', label='baseline [-8,-4]s')
    ax.axvspan(-1, +1, alpha=0.08, color='red', label='peri-nadir [-1,+1]s')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('time rel. nadir (s)')
    ax.set_ylabel('phase-jump rate (counts/100ms)')
    ax.set_title(f'B10 — Phase-reset null · {len(pj_mat)} subjects')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    # Distribution of per-subject elevations
    ax = axes[1]
    ax.hist(per_sub_elev, bins=30, color='gray', edgecolor='k', lw=0.3,
            alpha=0.85)
    ax.axvline(1.0, color='k', lw=0.8, label='no elevation')
    ax.axvline(np.median(per_sub_elev), color='gray', ls='--', lw=1.5,
               label=f'pseudo median {np.median(per_sub_elev):.2f}×')
    ax.axvline(1.34, color='#4575b4', ls='--', lw=1.5, label='real Q1 1.34×')
    ax.axvline(1.49, color='#d73027', ls='--', lw=1.5, label='real Q4 1.49×')
    ax.set_xlabel('peri-nadir / baseline ratio')
    ax.set_ylabel('subjects')
    ax.set_title('Per-subject pseudo-event elevations')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    plt.suptitle(f'B10 — Phase-reset null (pseudo-events)', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'phase_reset_null_random.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/phase_reset_null_random.png")
    print(f"Saved: {OUT_DIR}/phase_reset_null_random.csv")


if __name__ == '__main__':
    main()
