#!/usr/bin/env python3
"""
A5b — Harmonic ratio precision: composite events vs current Stage-1 events.

For each LEMON EC subject:
  1. Load raw; recompute composite events with same method as A5
  2. For current events (t0_net) AND composite events:
       - For each event, define 20-s window centered on event time
       - Run FOOOF across std-1020 subset PSD median in the window
       - Extract refined harmonic frequencies for sr1..sr6
       - Compute 4 ratios: sr3/sr1, sr5/sr1, sr5/sr3, sr6/sr4
  3. Aggregate per-event ratios for both event sets
  4. Compute MAE vs phi^n predictions:
       sr3/sr1 → φ² = 2.618
       sr5/sr1 → φ³ = 4.236
       sr5/sr3 → φ¹ = 1.618
       sr6/sr4 → φ¹ = 1.618
  5. Also: per-ratio std (within-ratio precision) and subject-level ICC proxies

Output: CSV of ratios per event (labeled current/composite), figure with
ratio histograms and MAE bars.

Note: to keep runtime reasonable, limit to subjects with n_events ≥ 5 and
cap events per subject at 10 per detector.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import bandpass
from scripts.sie_composite_detector import (
    compute_global_streams, robust_z, detect_composite_events,
    F0, HALF_BW, R_BAND, STEP_SEC, WIN_SEC, EDGE_SEC, MIN_ISI_SEC,
)
from lib.fooof_harmonics import detect_harmonics_fooof
from lib.mne_to_ignition import (
    mne_raw_to_ignition_df, CANON, HALF_BW as CANON_BW, FREQ_RANGES, LABELS,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'composite_detector')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

PHI = (1 + 5 ** 0.5) / 2
RATIO_TARGETS = {
    'sr3/sr1': PHI ** 2,     # 2.618
    'sr5/sr1': PHI ** 3,     # 4.236
    'sr5/sr3': PHI ** 1,     # 1.618
    'sr6/sr4': PHI ** 1,     # 1.618
}

CAP_PER_SUB = 10   # max events per subject per detector to control runtime
WIN_SEC_EVT = 20.0


def harmonics_in_window(df, eeg_channels, fs, t_center):
    """Return dict of refined harmonic freqs for the 20-s window centered on t_center."""
    half = WIN_SEC_EVT / 2
    window = [t_center - half, t_center + half]
    try:
        harmonics, _ = detect_harmonics_fooof(
            df, eeg_channels, fs=fs, window=window,
            f_can=tuple(CANON), freq_ranges=FREQ_RANGES,
            nperseg_sec=min(WIN_SEC_EVT, 10.0),
            peak_width_limits=(0.1, 4),
            max_n_peaks=10, min_peak_height=0.01, peak_threshold=0.01,
            search_halfband=tuple(CANON_BW), match_method='power',
            combine='median',
        )
    except Exception:
        return None
    out = {L: float(h) if h is not None and np.isfinite(h) else np.nan
           for L, h in zip(LABELS, harmonics)}
    return out


def process_subject(args):
    sub_id, events_path = args
    try:
        cur_events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(cur_events) < 5:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    df, eeg_channels = mne_raw_to_ignition_df(raw)

    # --- Composite event times (recompute as in A5) ---
    X = raw.get_data() * 1e6
    try:
        t, env, R, P = compute_global_streams(X, fs)
    except Exception:
        return None
    zE, zR, zP = robust_z(env), robust_z(R), robust_z(P)
    S = np.cbrt(np.clip(zE, 0, None) * np.clip(zR, 0, None) * np.clip(zP, 0, None))
    n_target = min(CAP_PER_SUB, len(cur_events))
    comp_times = detect_composite_events(t, S, n_target=n_target)

    # --- Current event times (capped) ---
    cur_times = np.sort(cur_events['t0_net'].values)[:CAP_PER_SUB]

    rows = []
    for src, times in [('current', cur_times), ('composite', comp_times)]:
        for t_c in times:
            h = harmonics_in_window(df, eeg_channels, fs, t_c)
            if h is None:
                continue
            rec = {'subject_id': sub_id, 'source': src, 't_center': float(t_c), **h}
            # Compute ratios
            for name, target in RATIO_TARGETS.items():
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
    for _, r in ok.iterrows():
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path))
    print(f"Subjects to process: {len(tasks)} (cap {CAP_PER_SUB} events/detector)")

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    all_rows = []
    for r in results:
        if r:
            all_rows.extend(r)
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, 'composite_vs_current_ratios.csv')
    df.to_csv(csv_path, index=False)
    print(f"Total events with FOOOF fits: {len(df)}")
    print(f"Saved: {csv_path}")

    # Summary stats: per source, per ratio
    print(f"\n=== Ratio precision (MAE vs φⁿ) ===\n")
    summary_rows = []
    for name, target in RATIO_TARGETS.items():
        for src in ['current', 'composite']:
            sub = df[(df['source'] == src)][name].dropna()
            if len(sub) == 0:
                continue
            mae = float(np.mean(np.abs(sub - target)))
            mean = float(np.mean(sub))
            std = float(np.std(sub))
            n = int(len(sub))
            summary_rows.append({
                'ratio': name, 'source': src, 'target': target,
                'n': n, 'mean': mean, 'std': std, 'MAE_vs_phi': mae,
                'bias': mean - target,
            })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUT_DIR, 'ratio_precision_summary.csv'),
                       index=False)

    # Figure: ratio histograms + MAE bars
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ci, (name, target) in enumerate(RATIO_TARGETS.items()):
        ax = axes[0, ci]
        for src, color in [('current', 'steelblue'), ('composite', 'coral')]:
            vals = df[df['source'] == src][name].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=40, alpha=0.55, label=f'{src} (n={len(vals)})',
                    color=color, edgecolor='k', lw=0.2)
        ax.axvline(target, color='red', ls='--', lw=1, label=f'φⁿ = {target:.3f}')
        ax.set_title(f'{name}')
        ax.set_xlabel('ratio')
        ax.set_ylabel('count')
        ax.legend(fontsize=7)

        ax2 = axes[1, ci]
        src_names = []
        maes = []
        for src, color in [('current', 'steelblue'), ('composite', 'coral')]:
            vals = df[df['source'] == src][name].dropna()
            if len(vals) == 0:
                continue
            mae = float(np.mean(np.abs(vals - target)))
            src_names.append(src)
            maes.append(mae)
        ax2.bar(src_names, maes, color=['steelblue', 'coral'])
        ax2.set_ylabel(f'MAE vs φⁿ = {target:.3f}')
        ax2.set_title(f'{name}')
        for si, (s, m) in enumerate(zip(src_names, maes)):
            ax2.text(si, m, f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'A5b — Harmonic ratio precision: composite vs current event sets\n'
                  f'LEMON EC · {df["subject_id"].nunique()} subjects · '
                  f'{(df["source"]=="current").sum()} current / '
                  f'{(df["source"]=="composite").sum()} composite events',
                  fontsize=12, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'composite_vs_current_precision.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
