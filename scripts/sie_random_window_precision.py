#!/usr/bin/env python3
"""
A5e — Ratio precision from RANDOM 20-s windows (no detector).

For each LEMON EC subject with ≥5 events in the current extraction:
  1. Load raw
  2. Draw N random 20-s window centers (N matched to current event count,
     capped at 10 per subject), uniformly between [10, duration-10] s
  3. Run the same FOOOF harmonic refinement per window as A5b
  4. Compute sr3/sr1, sr5/sr1, sr5/sr3, sr6/sr4 ratios

Compare MAE vs φⁿ to current-detector and composite-detector from A5b.

If random-window MAE ≈ detector MAE, the φⁿ ratio precision is NOT driven by
detection — it is a population-level feature of the EEG signal.
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
from scripts.sie_composite_vs_current_precision import (
    harmonics_in_window, RATIO_TARGETS, CAP_PER_SUB, WIN_SEC_EVT,
)
from lib.mne_to_ignition import mne_raw_to_ignition_df

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'composite_detector')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')


def process_subject(args):
    sub_id, events_path, seed = args
    try:
        cur = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(cur) < 5:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    duration = raw.times[-1]
    if duration < 40:
        return None

    df, eeg_channels = mne_raw_to_ignition_df(raw)

    n_target = min(CAP_PER_SUB, len(cur))
    rng = np.random.default_rng(seed)
    half = WIN_SEC_EVT / 2
    # Uniform random centers, at least half+1s from edges, ≥2s between consecutive
    centers = []
    for _ in range(n_target * 100):
        if len(centers) >= n_target:
            break
        c = rng.uniform(half + 1.0, duration - half - 1.0)
        if centers and np.min(np.abs(np.array(centers) - c)) < 2.0:
            continue
        centers.append(c)

    rows = []
    for t_c in centers:
        h = harmonics_in_window(df, eeg_channels, fs, t_c)
        if h is None:
            continue
        rec = {'subject_id': sub_id, 'source': 'random', 't_center': float(t_c), **h}
        for name in RATIO_TARGETS:
            num, den = name.split('/')
            try:
                v = rec[num] / rec[den]
            except Exception:
                v = np.nan
            if not np.isfinite(v):
                v = np.nan
            rec[name] = v
        rows.append(rec)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 5)]
    tasks = []
    for i, (_, r) in enumerate(ok.iterrows()):
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path, i + 1))
    print(f"Subjects: {len(tasks)}, cap {CAP_PER_SUB} random windows each")

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    all_rows = []
    for r in results:
        if r:
            all_rows.extend(r)
    df_rand = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, 'random_window_ratios.csv')
    df_rand.to_csv(csv_path, index=False)
    print(f"Random-window events with FOOOF fits: {len(df_rand)}")

    # Load current + composite from A5b
    detect_csv = os.path.join(OUT_DIR, 'composite_vs_current_ratios.csv')
    df_det = pd.read_csv(detect_csv)

    # Combine and summarize
    df_all = pd.concat([df_det, df_rand], ignore_index=True)
    summary_rows = []
    print(f"\n=== Ratio precision (MAE vs φⁿ) ===\n")
    for name, target in RATIO_TARGETS.items():
        for src in ['current', 'composite', 'random']:
            sub = df_all[df_all['source'] == src][name].dropna()
            if len(sub) == 0:
                continue
            mae = float(np.mean(np.abs(sub - target)))
            mean = float(np.mean(sub))
            std = float(np.std(sub))
            n = int(len(sub))
            summary_rows.append({
                'ratio': name, 'source': src, 'target': target, 'n': n,
                'mean': mean, 'std': std, 'MAE_vs_phi': mae, 'bias': mean - target,
            })
    summ = pd.DataFrame(summary_rows)
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(OUT_DIR, 'ratio_precision_3way_summary.csv'), index=False)

    # Figure: overlay histograms + MAE bars
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    colors = {'current': 'steelblue', 'composite': 'coral', 'random': 'gray'}
    for ci, (name, target) in enumerate(RATIO_TARGETS.items()):
        ax = axes[0, ci]
        for src, color in colors.items():
            vals = df_all[df_all['source'] == src][name].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=40, alpha=0.45, label=f'{src} (n={len(vals)})',
                    color=color, edgecolor='k', lw=0.2)
        ax.axvline(target, color='red', ls='--', lw=1, label=f'φⁿ = {target:.3f}')
        ax.set_title(name)
        ax.set_xlabel('ratio')
        ax.set_ylabel('count')
        ax.legend(fontsize=7)

        ax2 = axes[1, ci]
        src_names, maes = [], []
        for src in ['current', 'composite', 'random']:
            vals = df_all[df_all['source'] == src][name].dropna()
            if len(vals) == 0:
                continue
            src_names.append(src)
            maes.append(float(np.mean(np.abs(vals - target))))
        ax2.bar(src_names, maes, color=[colors[s] for s in src_names])
        ax2.set_ylabel(f'MAE vs φⁿ = {target:.3f}')
        ax2.set_title(name)
        for si, (s, m) in enumerate(zip(src_names, maes)):
            ax2.text(si, m, f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'A5e — Ratio precision: detector events vs RANDOM windows\n'
                 f'LEMON EC · {df_all["subject_id"].nunique()} subjects',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'ratio_precision_3way.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
