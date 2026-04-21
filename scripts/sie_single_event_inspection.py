#!/usr/bin/env python3
"""
B1 — Single-event inspection (robustness check for grand averages).

The A3/A9 peri-onset figures are grand averages over ~900 events. Averages
can produce structure no individual event shows (e.g., half with dip at -1s,
half at -3s → grand average shows flat curve or two bumps).

This script:
  1. Randomly samples 20 events from across subjects, overlays their
     nadir-aligned envelope z, R, PLV, MSC traces on one plot per stream.
  2. Computes per-subject means for 12 subjects and overlays them.
  3. Reports, per event, time of minimum in [-3, +0.4] for each stream,
     and correlation of this 'per-event dip time' across streams.

If the dip-rebound is robust, single events should show a visible dip at
roughly t=0, and per-subject means should also.
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
from scripts.sie_perionset_triple_average import bandpass
from scripts.sie_dip_onset_and_narrow_fooof import compute_streams_4way
from scripts.sie_perionset_multistream import (
    PRE_SEC, POST_SEC, PAD_SEC, STEP_SEC, find_nadir, TGRID,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'single_event')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

N_EVENTS_RANDOM = 24
N_SUBJECTS_MEAN = 12
SEED = 7


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) == 0:
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

    rows = []
    for i, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end:
            continue
        i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue
        try:
            t_c, env, R, P, M = compute_streams_4way(X_seg, fs)
        except Exception:
            continue
        rel = t_c - PAD_SEC - PRE_SEC
        nadir = find_nadir(rel, env, R, P, M)
        if not np.isfinite(nadir):
            continue
        rel_n = rel - nadir
        env_i = np.interp(TGRID, rel_n, env, left=np.nan, right=np.nan)
        R_i = np.interp(TGRID, rel_n, R, left=np.nan, right=np.nan)
        P_i = np.interp(TGRID, rel_n, P, left=np.nan, right=np.nan)
        M_i = np.interp(TGRID, rel_n, M, left=np.nan, right=np.nan)
        rows.append({
            'subject_id': sub_id, 'event_idx': int(i),
            'env': env_i, 'R': R_i, 'PLV': P_i, 'MSC': M_i,
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path))
    print(f"Subjects: {len(tasks)}")

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    # flatten
    all_events = []
    subject_of_event = []
    for r in results:
        if not r:
            continue
        for ev in r:
            all_events.append(ev)
            subject_of_event.append(ev['subject_id'])
    print(f"Total nadir-aligned events: {len(all_events)}")

    # Sample 24 random events
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(all_events), size=min(N_EVENTS_RANDOM, len(all_events)),
                             replace=False)

    # 12 subjects with most events, compute their per-subject means
    by_sub = {}
    for ev in all_events:
        by_sub.setdefault(ev['subject_id'], []).append(ev)
    top_subs = sorted(by_sub.keys(), key=lambda s: -len(by_sub[s]))[:N_SUBJECTS_MEAN]

    # Figure: 4 streams × 2 rows (events | subject means)
    streams = [('env', 'envelope z', 'darkorange'),
                ('R',   'Kuramoto R',  'seagreen'),
                ('PLV', 'mean PLV',    'purple'),
                ('MSC', 'mean MSC',    'steelblue')]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    for i, (key, label, color) in enumerate(streams):
        # Events column
        ax = axes[i, 0]
        for idx in sample_idx:
            ev = all_events[idx]
            ax.plot(TGRID, ev[key], color=color, alpha=0.3, lw=0.7)
        # Grand mean over sample
        arr = np.array([all_events[idx][key] for idx in sample_idx])
        ax.plot(TGRID, np.nanmean(arr, axis=0), color='black', lw=2,
                label=f'mean of {len(sample_idx)} events')
        ax.axvline(0, color='k', ls='--', lw=0.5)
        ax.set_ylabel(label)
        if i == 0:
            ax.set_title(f'{N_EVENTS_RANDOM} random events (thin) + their mean (thick)')
        ax.legend(fontsize=8)

        # Subject means column
        ax = axes[i, 1]
        for sub in top_subs:
            sub_evs = by_sub[sub]
            arr = np.array([ev[key] for ev in sub_evs])
            ax.plot(TGRID, np.nanmean(arr, axis=0), alpha=0.65, lw=1.0,
                    label=f'{sub} (n={len(sub_evs)})')
        ax.axvline(0, color='k', ls='--', lw=0.5)
        if i == 0:
            ax.set_title(f'{N_SUBJECTS_MEAN} subjects (mean over their events)')
        ax.legend(fontsize=6, loc='upper left', ncol=2)
    axes[-1, 0].set_xlabel('time relative to nadir (s)')
    axes[-1, 1].set_xlabel('time relative to nadir (s)')
    fig.suptitle(f'B1 — Single-event inspection (nadir-aligned)\n'
                  f'{len(all_events)} total events from {len(by_sub)} subjects',
                  fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'single_event_overlay.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # Per-event nadir times per stream + pairwise correlation
    dip_rows = []
    dip_win = (TGRID >= -3.0) & (TGRID <= 0.4)
    for ev in all_events:
        rec = {'subject_id': ev['subject_id']}
        for key, _, _ in streams:
            arr = ev[key]
            sub = np.where(dip_win, arr, np.inf)
            idx = int(np.nanargmin(sub))
            rec[f'{key}_dip_t'] = float(TGRID[idx])
        dip_rows.append(rec)
    dip_df = pd.DataFrame(dip_rows)
    dip_df.to_csv(os.path.join(OUT_DIR, 'per_event_dip_times.csv'), index=False)

    print(f"\nPer-event dip time std (should be small if robust):")
    for key, label, _ in streams:
        vals = dip_df[f'{key}_dip_t']
        print(f"  {label:12s}: median {vals.median():+.2f}s  "
              f"IQR [{vals.quantile(.25):+.2f}, {vals.quantile(.75):+.2f}]s  "
              f"std {vals.std():.2f}s")
    # Pairwise correlation
    corr = dip_df[[f'{s[0]}_dip_t' for s in streams]].corr()
    print(f"\nPair-wise correlation of per-event dip times:")
    print(corr.round(3).to_string())
    corr.to_csv(os.path.join(OUT_DIR, 'dip_time_cross_correlations.csv'))

    print(f"\nSaved: {OUT_DIR}/single_event_overlay.png")
    print(f"       {OUT_DIR}/per_event_dip_times.csv")


if __name__ == '__main__':
    main()
