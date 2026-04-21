#!/usr/bin/env python3
"""
A4a — Compute a principled ignition onset from the composite stream S(t).

For each real SIE event in LEMON EC:
  1. Extract ±12 s around t0_net (for padding)
  2. Compute envelope z, R(t), PLV(t) streams
  3. z-score each stream against its PRE-window (t0_net - 10 s to t0_net - 5 s) mean/SD
     (a local baseline, not session baseline — to isolate peri-event change)
  4. Build composite S(t) = (max(zE,0) * max(zR,0) * max(zP,0))^(1/3)
  5. Find peak of S on the [-5, +5] s window around t0_net
  6. Define onset = earliest time in [peak - 3s, peak] at which S first
     crosses 25% of peak value from below (rising edge)

Then realign all events on the computed onset and re-run the peri-onset
average (envelope z, R, PLV around the *computed onset*, not t0_net) with
bootstrap CIs. Compare timing of joint peak to t=0.

Report:
  - Distribution of (onset - t0_net) offsets across events
  - Distribution of (peak_time - onset) latencies (how long from onset to peak)
  - Peri-computed-onset grand averages for all three streams
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
from scripts.sie_perionset_triple_average import (
    compute_streams, bootstrap_ci, TGRID, PRE_SEC, POST_SEC, PAD_SEC,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'perionset')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

ONSET_FRAC = 0.25      # onset = earliest rising crossing of this fraction of peak
PEAK_WIN = (-5, 5)     # search for peak within this window (rel to t0_net)
SEARCH_BACK = 3.0      # search backwards this many seconds from peak to find onset
BASELINE_WIN = (-10, -5)  # baseline window (rel to t0_net) for local z-scoring


def local_zscore(x_series, t_series, baseline_win):
    """Z-score x against baseline subset defined by baseline_win bounds."""
    mask = (t_series >= baseline_win[0]) & (t_series < baseline_win[1])
    if mask.sum() < 5:
        return x_series - np.nanmean(x_series)
    mu = np.nanmean(x_series[mask])
    sd = np.nanstd(x_series[mask])
    if not np.isfinite(sd) or sd < 1e-9:
        sd = 1.0
    return (x_series - mu) / sd


def compute_event_onset(X_seg, fs, pre_pad, pre_sec):
    """
    Given a ± PRE_SEC+PAD segment (pre_pad = PAD_SEC), compute the composite
    onset time (relative to t0_net=0 in segment-relative time).
    Returns dict with onset_rel, peak_rel, S, t_S.
    """
    (t_env, zenv), (tR, R), (tP, P) = compute_streams(X_seg, fs)
    # convert to t0_net-relative time
    rel_env = t_env - pre_pad - pre_sec
    rel_R = tR - pre_pad - pre_sec
    rel_P = tP - pre_pad - pre_sec

    # Interpolate envelope z onto R/P grid (R and P share same grid via compute_streams)
    # Use common grid = R's grid
    zenv_i = np.interp(rel_R, rel_env, zenv, left=np.nan, right=np.nan)

    # Local z-score each stream against BASELINE_WIN
    zE_local = local_zscore(zenv_i, rel_R, BASELINE_WIN)
    zR_local = local_zscore(R, rel_R, BASELINE_WIN)
    zP_local = local_zscore(P, rel_R, BASELINE_WIN)

    # Composite S(t) = geometric mean of positive parts
    S = np.cbrt(
        np.clip(zE_local, 0, None) *
        np.clip(zR_local, 0, None) *
        np.clip(zP_local, 0, None)
    )

    # Find peak within PEAK_WIN
    peak_mask = (rel_R >= PEAK_WIN[0]) & (rel_R <= PEAK_WIN[1])
    if not np.any(peak_mask) or not np.any(np.isfinite(S[peak_mask])):
        return None
    S_pw = np.where(peak_mask, S, -np.inf)
    peak_idx = int(np.nanargmax(S_pw))
    peak_val = S[peak_idx]
    peak_rel = rel_R[peak_idx]
    if not np.isfinite(peak_val) or peak_val <= 0:
        return None

    # Onset = earliest rising crossing of ONSET_FRAC * peak_val, looking back
    threshold = ONSET_FRAC * peak_val
    back_mask = (rel_R >= peak_rel - SEARCH_BACK) & (rel_R <= peak_rel)
    idxs = np.where(back_mask)[0]
    if len(idxs) < 2:
        return None
    # Walk from peak backwards, find last idx where S < threshold.
    onset_idx = None
    for j in range(peak_idx, idxs[0] - 1, -1):
        if S[j] < threshold:
            onset_idx = j + 1
            break
    if onset_idx is None:
        onset_idx = idxs[0]
    onset_rel = rel_R[onset_idx]

    return {
        'onset_rel': float(onset_rel),
        'peak_rel': float(peak_rel),
        'peak_S': float(peak_val),
        'zE': zenv_i, 'zR_local': zR_local, 'zP_local': zP_local,
        'envr': zenv_i, 'R': R, 'P': P,  # raw streams (unscaled) for peri-onset avg
        't_rel': rel_R,
    }


def process_subject(args):
    sub_id, events_path = args
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

    onset_offsets = []
    peak_latencies = []  # peak - onset
    env_rows, R_rows, P_rows = [], [], []

    for _, ev in events.iterrows():
        t0 = float(ev.get('t0_net', np.nan))
        if not np.isfinite(t0):
            continue
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end_rec:
            continue
        i0 = int(round(lo * fs))
        i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue

        info = compute_event_onset(X_seg, fs, PAD_SEC, PRE_SEC)
        if info is None:
            continue
        onset_offsets.append(info['onset_rel'])  # relative to t0_net
        peak_latencies.append(info['peak_rel'] - info['onset_rel'])

        # Realign on computed onset: shift time axis so onset -> 0
        rel_onset_time = info['t_rel'] - info['onset_rel']
        # Interpolate onto TGRID (which is -10..+10)
        env_i = np.interp(TGRID, rel_onset_time, info['envr'], left=np.nan, right=np.nan)
        R_i = np.interp(TGRID, rel_onset_time, info['R'], left=np.nan, right=np.nan)
        P_i = np.interp(TGRID, rel_onset_time, info['P'], left=np.nan, right=np.nan)
        env_rows.append(env_i)
        R_rows.append(R_i)
        P_rows.append(P_i)

    if not env_rows:
        return None

    return {
        'subject_id': sub_id,
        'n_events': len(env_rows),
        'onset_offsets': np.array(onset_offsets),
        'peak_latencies': np.array(peak_latencies),
        'env': np.nanmean(np.array(env_rows), axis=0),
        'R': np.nanmean(np.array(R_rows), axis=0),
        'P': np.nanmean(np.array(P_rows), axis=0),
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
    print(f"Successful subjects: {len(results)}")
    n_events = sum(r['n_events'] for r in results)
    print(f"Total events with computed onset: {n_events}")

    all_offsets = np.concatenate([r['onset_offsets'] for r in results])
    all_latencies = np.concatenate([r['peak_latencies'] for r in results])
    print(f"\nOffset (computed_onset - t0_net):")
    print(f"  median = {np.median(all_offsets):+.2f} s, "
          f"[25%, 75%] = [{np.percentile(all_offsets, 25):+.2f}, "
          f"{np.percentile(all_offsets, 75):+.2f}] s")
    print(f"  mean   = {np.mean(all_offsets):+.2f} s ± {np.std(all_offsets):.2f}")
    print(f"\nPeak latency (peak_S - computed_onset):")
    print(f"  median = {np.median(all_latencies):.2f} s, "
          f"[25%, 75%] = [{np.percentile(all_latencies, 25):.2f}, "
          f"{np.percentile(all_latencies, 75):.2f}] s")

    env_arr = np.array([r['env'] for r in results])
    R_arr = np.array([r['R'] for r in results])
    P_arr = np.array([r['P'] for r in results])

    env_m, env_lo, env_hi = bootstrap_ci(env_arr)
    R_m, R_lo, R_hi = bootstrap_ci(R_arr)
    P_m, P_lo, P_hi = bootstrap_ci(P_arr)

    peak_env = TGRID[np.argmax(env_m)]
    peak_R = TGRID[np.argmax(R_m)]
    peak_P = TGRID[np.argmax(P_m)]
    print(f"\nAligned on computed onset, grand-mean peak locations:")
    print(f"  envelope z peak at t = {peak_env:+.2f} s")
    print(f"  R(t) peak at t = {peak_R:+.2f} s")
    print(f"  PLV peak at t = {peak_P:+.2f} s")

    # Save CSV
    df = pd.DataFrame({
        't_rel': TGRID,
        'env_mean': env_m, 'env_ci_lo': env_lo, 'env_ci_hi': env_hi,
        'R_mean': R_m, 'R_ci_lo': R_lo, 'R_ci_hi': R_hi,
        'P_mean': P_m, 'P_ci_lo': P_lo, 'P_ci_hi': P_hi,
    })
    csv_path = os.path.join(OUT_DIR, 'perionset_computed_onset.csv')
    df.to_csv(csv_path, index=False)

    # Figure
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1])
    ax_hist1 = fig.add_subplot(gs[0, 0])
    ax_hist2 = fig.add_subplot(gs[0, 1])
    ax_env = fig.add_subplot(gs[1, :])
    ax_R = fig.add_subplot(gs[2, :], sharex=ax_env)
    ax_P = fig.add_subplot(gs[3, :], sharex=ax_env)

    ax_hist1.hist(all_offsets, bins=40, color='steelblue', edgecolor='k', lw=0.3)
    ax_hist1.axvline(0, color='k', ls='--', lw=0.6, label='t0_net')
    ax_hist1.axvline(np.median(all_offsets), color='red', lw=1.2,
                    label=f'median {np.median(all_offsets):+.2f} s')
    ax_hist1.set_xlabel('computed onset − t0_net (s)')
    ax_hist1.set_ylabel('count')
    ax_hist1.set_title('Onset offset from t0_net')
    ax_hist1.legend(fontsize=8)

    ax_hist2.hist(all_latencies, bins=40, color='coral', edgecolor='k', lw=0.3)
    ax_hist2.axvline(np.median(all_latencies), color='red', lw=1.2,
                    label=f'median {np.median(all_latencies):.2f} s')
    ax_hist2.set_xlabel('S-peak − computed onset (s)')
    ax_hist2.set_ylabel('count')
    ax_hist2.set_title('Onset→peak latency')
    ax_hist2.legend(fontsize=8)

    for ax, arr, m, lo, hi, label, color in [
        (ax_env, env_arr, env_m, env_lo, env_hi, 'envelope z', 'darkorange'),
        (ax_R,   R_arr,   R_m,   R_lo,   R_hi,   'Kuramoto R', 'seagreen'),
        (ax_P,   P_arr,   P_m,   P_lo,   P_hi,   'mean PLV',   'purple'),
    ]:
        for i in range(arr.shape[0]):
            ax.plot(TGRID, arr[i], color='gray', alpha=0.07, lw=0.3)
        ax.fill_between(TGRID, lo, hi, color=color, alpha=0.25)
        ax.plot(TGRID, m, color=color, lw=2)
        ax.axvline(0, color='k', ls='--', lw=0.6, label='computed onset')
        ax.axvline(np.median(all_latencies), color='r', ls=':', lw=0.8, label='median peak')
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
    ax_P.set_xlabel('time relative to computed onset (s)')
    ax_env.set_title(f'Peri-onset averages aligned on COMPUTED ONSET\n'
                      f'{len(results)} subjects · {n_events} events')

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'perionset_computed_onset.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
