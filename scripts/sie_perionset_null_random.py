#!/usr/bin/env python3
"""
A3b — Null check for A3 peri-onset peak.

Replicate A3 peri-onset triple average but with RANDOM onsets drawn per subject,
matched in count to the real events, constrained to:
  - Stay ≥12 s from recording edges
  - Stay ≥20 s away from any real detected event (so we sample true baseline)

If the +1.2 s joint peak observed in A3 is real, it should vanish here.
If it persists, it is an alignment/artifact signature.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import (
    compute_streams, bootstrap_ci, TGRID, PRE_SEC, POST_SEC, PAD_SEC,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'perionset')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')


def process_subject(args):
    sub_id, events_path, seed = args
    try:
        events = pd.read_csv(events_path)
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
    t_end_rec = raw.times[-1]

    # Real onsets, for exclusion
    real_t0 = events['t0_net'].dropna().values

    # Draw random onsets: count matched to real events, ≥12 s from edges,
    # ≥20 s from any real onset. Try up to 200 draws to fill.
    rng = np.random.default_rng(seed)
    n_target = len(real_t0)
    margin_edge = PRE_SEC + PAD_SEC
    exclusion = 20.0
    candidates = []
    for _ in range(n_target * 200):
        if len(candidates) >= n_target:
            break
        t = rng.uniform(margin_edge, t_end_rec - margin_edge)
        if len(real_t0) > 0 and np.min(np.abs(real_t0 - t)) < exclusion:
            continue
        if len(candidates) > 0 and np.min(np.abs(np.array(candidates) - t)) < 5.0:
            continue
        candidates.append(t)
    if not candidates:
        return None

    env_rows, R_rows, P_rows = [], [], []
    for t0 in candidates:
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
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
        env_rows.append(np.interp(TGRID, rel_env, zenv, left=np.nan, right=np.nan))
        R_rows.append(np.interp(TGRID, rel_R, R, left=np.nan, right=np.nan))
        P_rows.append(np.interp(TGRID, rel_P, P, left=np.nan, right=np.nan))

    if not env_rows:
        return None

    return {
        'subject_id': sub_id,
        'n_events': len(env_rows),
        'env': np.nanmean(np.array(env_rows), axis=0),
        'R': np.nanmean(np.array(R_rows), axis=0),
        'P': np.nanmean(np.array(P_rows), axis=0),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for i, (_, r) in enumerate(ok.iterrows()):
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path, i + 1))
    print(f"Subjects to process: {len(tasks)}")

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")
    n_events_total = sum(r['n_events'] for r in results)
    print(f"Total null events aggregated: {n_events_total}")

    env_arr = np.array([r['env'] for r in results])
    R_arr = np.array([r['R'] for r in results])
    P_arr = np.array([r['P'] for r in results])

    env_m, env_lo, env_hi = bootstrap_ci(env_arr)
    R_m, R_lo, R_hi = bootstrap_ci(R_arr)
    P_m, P_lo, P_hi = bootstrap_ci(P_arr)

    df = pd.DataFrame({
        't_rel': TGRID,
        'env_mean': env_m, 'env_ci_lo': env_lo, 'env_ci_hi': env_hi,
        'R_mean': R_m, 'R_ci_lo': R_lo, 'R_ci_hi': R_hi,
        'P_mean': P_m, 'P_ci_lo': P_lo, 'P_ci_hi': P_hi,
    })
    csv_path = os.path.join(OUT_DIR, 'perionset_null_random.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Load real for overlay
    real_csv = os.path.join(OUT_DIR, 'perionset_triple_average.csv')
    real = pd.read_csv(real_csv)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    specs = [
        ('envelope z (7.83 ± 0.6 Hz)', 'env', 'darkorange'),
        ('Kuramoto R(t) in 7.2–8.4 Hz', 'R', 'seagreen'),
        ('mean PLV to median', 'P', 'purple'),
    ]
    for ax, (label, key, color) in zip(axes, specs):
        # Real
        ax.fill_between(real['t_rel'], real[f'{key}_ci_lo'], real[f'{key}_ci_hi'],
                        color=color, alpha=0.20, label='real events 95% CI')
        ax.plot(real['t_rel'], real[f'{key}_mean'], color=color, lw=2, label='real mean')
        # Null
        ax.fill_between(df['t_rel'], df[f'{key}_ci_lo'], df[f'{key}_ci_hi'],
                        color='gray', alpha=0.30, label='random-onset null 95% CI')
        ax.plot(df['t_rel'], df[f'{key}_mean'], color='black', lw=1.2, ls='--',
                label='null mean')
        ax.axvline(0, color='k', ls=':', lw=0.6)
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
    axes[0].set_title(f'Peri-onset triple average — real events vs random-onset null\n'
                      f'LEMON EC · {len(results)} subjects · {n_events_total} null events')
    axes[2].set_xlabel('time relative to onset (s)')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'perionset_null_vs_real.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # Print peak info for null
    i0 = np.argmin(np.abs(TGRID))
    peak_env = np.argmax(env_m)
    peak_R = np.argmax(R_m)
    peak_P = np.argmax(P_m)
    print(f"\nNULL peak locations:")
    print(f"  envelope z peak at t = {TGRID[peak_env]:+.2f} s (value {env_m[peak_env]:.3f})")
    print(f"  R(t) peak at t = {TGRID[peak_R]:+.2f} s (value {R_m[peak_R]:.3f})")
    print(f"  PLV peak at t = {TGRID[peak_P]:+.2f} s (value {P_m[peak_P]:.3f})")
    print(f"\nNULL at t=0:")
    print(f"  env z = {env_m[i0]:.3f}  [{env_lo[i0]:.3f}, {env_hi[i0]:.3f}]")
    print(f"  R     = {R_m[i0]:.3f}  [{R_lo[i0]:.3f}, {R_hi[i0]:.3f}]")
    print(f"  PLV   = {P_m[i0]:.3f}  [{P_lo[i0]:.3f}, {P_hi[i0]:.3f}]")


if __name__ == '__main__':
    main()
