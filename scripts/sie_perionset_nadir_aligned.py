#!/usr/bin/env python3
"""
A3/A4a re-run with NADIR-ALIGNED onset.

For each SIE event:
  1. Extract ±15 s around t0_net (to have headroom for nadir that may be up to -3 s)
  2. Compute four streams (envelope z, Kuramoto R, mean PLV, MSC)
  3. Find joint-dip-minimum (nadir) = argmin of (zE + zR + zP + zM)
     in [t0_net - 3.0, t0_net + 0.4]
  4. Realign streams on the nadir (t=0 := nadir)
  5. Compute grand-mean per-stream over subjects (subject-level cluster bootstrap CI)

Output: nadir-aligned peri-onset figure and CSV, to replace t0_net-aligned
A3 as the canonical peri-onset characterization.
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
from scripts.sie_perionset_triple_average import bootstrap_ci
from scripts.sie_dip_onset_and_narrow_fooof import (
    compute_streams_4way, find_joint_dip,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'perionset')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

PRE_SEC = 10.0
POST_SEC = 10.0
PAD_SEC = 5.0   # extra pad for nadir search (up to -3s) + filter stability
STEP_SEC = 0.1

# Nadir-centered output grid: -8 to +8 s (10-2 buffer on pre, 10-2 on post)
TGRID = np.arange(-8.0, 8.0 + STEP_SEC/2, STEP_SEC)


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
    t_end_rec = raw.times[-1]

    env_rows, R_rows, P_rows, M_rows = [], [], [], []
    nadir_offsets = []

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
            t_c_rel, env, R, P, M = compute_streams_4way(X_seg, fs)
        except Exception:
            continue
        # rel: time relative to t0_net
        rel = t_c_rel - PAD_SEC - PRE_SEC
        dip_rel = find_joint_dip(rel, env, R, P, M)
        if not np.isfinite(dip_rel):
            continue
        nadir_offsets.append(dip_rel)
        # realign on nadir: rel_to_nadir = rel - dip_rel
        rel_nadir = rel - dip_rel
        env_rows.append(np.interp(TGRID, rel_nadir, env, left=np.nan, right=np.nan))
        R_rows.append(np.interp(TGRID, rel_nadir, R, left=np.nan, right=np.nan))
        P_rows.append(np.interp(TGRID, rel_nadir, P, left=np.nan, right=np.nan))
        M_rows.append(np.interp(TGRID, rel_nadir, M, left=np.nan, right=np.nan))

    if not env_rows:
        return None

    return {
        'subject_id': sub_id,
        'n_events': len(env_rows),
        'nadir_offsets': np.array(nadir_offsets),
        'env': np.nanmean(np.array(env_rows), axis=0),
        'R':   np.nanmean(np.array(R_rows), axis=0),
        'P':   np.nanmean(np.array(P_rows), axis=0),
        'M':   np.nanmean(np.array(M_rows), axis=0),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path))
    print(f"Subjects to process: {len(tasks)}")

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")
    n_events = sum(r['n_events'] for r in results)
    print(f"Total nadir-aligned events: {n_events}")
    all_offsets = np.concatenate([r['nadir_offsets'] for r in results])
    print(f"Nadir offsets vs t0_net: median {np.median(all_offsets):+.2f} s, "
          f"IQR [{np.percentile(all_offsets, 25):+.2f}, "
          f"{np.percentile(all_offsets, 75):+.2f}] s")

    env_arr = np.array([r['env'] for r in results])
    R_arr = np.array([r['R'] for r in results])
    P_arr = np.array([r['P'] for r in results])
    M_arr = np.array([r['M'] for r in results])

    env_m, env_lo, env_hi = bootstrap_ci(env_arr)
    R_m, R_lo, R_hi = bootstrap_ci(R_arr)
    P_m, P_lo, P_hi = bootstrap_ci(P_arr)
    M_m, M_lo, M_hi = bootstrap_ci(M_arr)

    df = pd.DataFrame({
        't_rel': TGRID,
        'env_mean': env_m, 'env_ci_lo': env_lo, 'env_ci_hi': env_hi,
        'R_mean': R_m, 'R_ci_lo': R_lo, 'R_ci_hi': R_hi,
        'P_mean': P_m, 'P_ci_lo': P_lo, 'P_ci_hi': P_hi,
        'M_mean': M_m, 'M_ci_lo': M_lo, 'M_ci_hi': M_hi,
    })
    csv_path = os.path.join(OUT_DIR, 'perionset_nadir_aligned.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    specs = [
        (axes[0], env_arr, env_m, env_lo, env_hi, 'envelope z', 'darkorange'),
        (axes[1], R_arr,   R_m,   R_lo,   R_hi,   'Kuramoto R', 'seagreen'),
        (axes[2], P_arr,   P_m,   P_lo,   P_hi,   'mean PLV',   'purple'),
        (axes[3], M_arr,   M_m,   M_lo,   M_hi,   'mean MSC',   'steelblue'),
    ]
    for ax, arr, m, lo, hi, label, color in specs:
        for i in range(arr.shape[0]):
            ax.plot(TGRID, arr[i], color='gray', alpha=0.07, lw=0.3)
        ax.fill_between(TGRID, lo, hi, color=color, alpha=0.25, label='95% CI')
        ax.plot(TGRID, m, color=color, lw=2, label='grand mean')
        ax.axvline(0, color='k', ls='--', lw=0.7, label='nadir (t₀ = onset)')
        peak_idx = int(np.argmax(m))
        ax.axvline(TGRID[peak_idx], color='red', ls=':', lw=0.7)
        ax.annotate(f'peak {TGRID[peak_idx]:+.2f}s', xy=(TGRID[peak_idx], m[peak_idx]),
                    xytext=(TGRID[peak_idx] + 0.3, m[peak_idx]), fontsize=9, color='red')
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
    axes[0].set_title(f'A3-nadir — Peri-onset grand averages, aligned on JOINT NADIR\n'
                       f'{len(results)} subjects · {n_events} events · '
                       f'nadir-offset median {np.median(all_offsets):+.2f}s vs t₀_net')
    axes[3].set_xlabel('time relative to nadir (s)')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'perionset_nadir_aligned.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # Peak times report
    print(f"\nPeak times relative to NADIR (t=0):")
    for ax_i, label, m in [(0, 'envelope z', env_m), (1, 'R', R_m),
                             (2, 'PLV', P_m), (3, 'MSC', M_m)]:
        idx = int(np.argmax(m))
        print(f"  {label:12s}: peak at {TGRID[idx]:+.2f} s (value {m[idx]:.3f})")


if __name__ == '__main__':
    main()
