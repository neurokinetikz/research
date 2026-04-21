#!/usr/bin/env python3
"""
A6 — Dip-rebound onset analysis.

Hypothesis: the true ignition onset is a concurrent *dip* (desynchronization)
in envelope z, Kuramoto R, and mean PLV, followed by a rebound peak ~0.5 s
later. Test which stream dips first.

For each event (aligned on t0_net):
  - Compute streams on ±10 s peri-onset window (same as A3)
  - Find dip time per stream: argmin in [-3.0, +0.4] s relative to t0_net
  - Find rebound peak per stream: argmax in [-0.5, +2.5] s
  - Compute dip depth (peak value - dip value) and rebound latency

Compare:
  - Distribution of dip times across events for each stream
  - Pairwise lead-lag: dip_E - dip_R, dip_R - dip_P, dip_E - dip_P
  - Grand-average overlay showing dip alignment

If dips are concurrent (no consistent lead), "ignition" is a unified
desync-then-resync signature. If one stream reliably leads, it's the
driver of the cascade.
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
from scripts.sie_perionset_triple_average import (
    compute_streams, TGRID, PRE_SEC, POST_SEC, PAD_SEC,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'perionset')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

DIP_WINDOW = (-3.0, 0.4)
PEAK_WINDOW = (-0.5, 2.5)


def find_dip_peak(t_rel, y):
    """Given time and stream, return (t_dip, y_dip, t_peak, y_peak)."""
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
            # depths
            'E_depth': yE_peak - yE_dip,
            'R_depth': yR_peak - yR_dip,
            'P_depth': yP_peak - yP_dip,
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

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    all_rows = []
    for r in results:
        if r:
            all_rows.extend(r)
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, 'dip_rebound_per_event.csv')
    df.to_csv(csv_path, index=False)
    print(f"Events: {len(df)}  Subjects: {df['subject_id'].nunique()}")

    # Summary stats
    print(f"\nDip times (median, IQR) vs t0_net:")
    for pre, label in [('tE_dip', 'envelope z'), ('tR_dip', 'Kuramoto R'), ('tP_dip', 'mean PLV')]:
        vals = df[pre].dropna()
        print(f"  {label:12s}: {np.median(vals):+.3f} s   "
              f"IQR [{np.percentile(vals,25):+.3f}, {np.percentile(vals,75):+.3f}]")
    print(f"\nPeak times (median, IQR):")
    for pre, label in [('tE_peak', 'envelope z'), ('tR_peak', 'Kuramoto R'), ('tP_peak', 'mean PLV')]:
        vals = df[pre].dropna()
        print(f"  {label:12s}: {np.median(vals):+.3f} s   "
              f"IQR [{np.percentile(vals,25):+.3f}, {np.percentile(vals,75):+.3f}]")

    # Pairwise lead-lag (who dips first)
    print(f"\nPairwise dip lead-lag (median, IQR):")
    pairs = [
        ('E − R',  df['tE_dip'] - df['tR_dip']),
        ('E − P',  df['tE_dip'] - df['tP_dip']),
        ('R − P',  df['tR_dip'] - df['tP_dip']),
    ]
    for label, diffs in pairs:
        d = diffs.dropna()
        # sign test: fraction of events where left dips first (negative means left leads)
        frac_left_first = float((d < 0).mean())
        frac_concurrent = float((d.abs() < 0.1).mean())
        print(f"  {label}: median {np.median(d):+.3f} s, "
              f"IQR [{np.percentile(d,25):+.3f}, {np.percentile(d,75):+.3f}]  "
              f"left-leads {frac_left_first*100:.1f}%  (|Δ|<0.1s {frac_concurrent*100:.1f}%)")

    # Figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3)

    # Row 1: dip time histograms per stream
    for ci, (col, label, color) in enumerate([
        ('tE_dip', 'envelope z', 'darkorange'),
        ('tR_dip', 'Kuramoto R', 'seagreen'),
        ('tP_dip', 'mean PLV', 'purple'),
    ]):
        ax = fig.add_subplot(gs[0, ci])
        vals = df[col].dropna()
        ax.hist(vals, bins=40, color=color, edgecolor='k', lw=0.3, alpha=0.7)
        ax.axvline(0, color='k', ls='--', lw=0.6, label='t₀_net')
        ax.axvline(np.median(vals), color='red', lw=1.2,
                   label=f'median {np.median(vals):+.2f}')
        ax.set_xlabel(f'{label} dip time (s)')
        ax.set_ylabel('events')
        ax.set_title(f'{label} dip time distribution')
        ax.legend(fontsize=8)

    # Row 2: pairwise lead-lag histograms
    for ci, (left, right, label) in enumerate([
        ('tE_dip', 'tR_dip', 'E − R'),
        ('tE_dip', 'tP_dip', 'E − P'),
        ('tR_dip', 'tP_dip', 'R − P'),
    ]):
        ax = fig.add_subplot(gs[1, ci])
        d = (df[left] - df[right]).dropna()
        ax.hist(d, bins=40, color='slategray', edgecolor='k', lw=0.3, alpha=0.7)
        ax.axvline(0, color='k', ls='--', lw=0.6, label='simultaneous')
        ax.axvline(np.median(d), color='red', lw=1.2,
                   label=f'median {np.median(d):+.3f}')
        ax.set_xlabel(f'{label} dip time difference (s)')
        ax.set_ylabel('events')
        ax.set_title(f'{label}')
        ax.legend(fontsize=8)

    # Row 3: dip depth histograms
    for ci, (col, label, color) in enumerate([
        ('E_depth', 'envelope z', 'darkorange'),
        ('R_depth', 'Kuramoto R', 'seagreen'),
        ('P_depth', 'mean PLV', 'purple'),
    ]):
        ax = fig.add_subplot(gs[2, ci])
        vals = df[col].dropna()
        ax.hist(vals, bins=40, color=color, edgecolor='k', lw=0.3, alpha=0.7)
        ax.axvline(np.median(vals), color='red', lw=1.2,
                   label=f'median {np.median(vals):.3f}')
        ax.set_xlabel(f'{label} dip→peak excursion')
        ax.set_ylabel('events')
        ax.set_title(f'{label} rebound depth')
        ax.legend(fontsize=8)

    fig.suptitle(f'A6 — Dip-rebound onset analysis\n'
                 f'LEMON EC · {df["subject_id"].nunique()} subjects · {len(df)} events',
                 fontsize=12, y=1.0)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'dip_rebound_analysis.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
