#!/usr/bin/env python3
"""
B9 — B3 propagation null: does the per-channel nadir-timing gradient R² appear
in random non-event windows too?

For every subject, match the number of real events with pseudo-events drawn
from random times that are ≥ 30 s from any real t0_net and ≥ 15 s from
recording edges. Run the same B3 pipeline on each pseudo-event:

  1. Extract ±12 s window
  2. compute_streams_4way → find joint nadir via find_nadir
  3. Per-channel envelope nadir in [-3, +0.4] s rel to joint nadir
  4. Fit gradient to (x, y) channel positions; report R², slopes, nadir_std

Compares real-event R² / nadir_std / slope_y distributions (from B8 output) to
the pseudo-event distributions.

If pseudo R² ≈ real R² → propagation is a scalp-field baseline property, not
event-specific. If real R² > pseudo R² → there is a genuine event-locked
spatial pattern.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_dip_onset_and_narrow_fooof import compute_streams_4way
from scripts.sie_perionset_multistream import (
    PRE_SEC, POST_SEC, PAD_SEC, find_nadir,
)
from scripts.sie_mechanism_battery import (
    per_channel_nadir, channel_positions,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'quality')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
REAL_B3_CSV = os.path.join(OUT_DIR, 'mechanism_by_quality_B3.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

MIN_GAP_FROM_EVENT = 30.0
MIN_EDGE = PRE_SEC + PAD_SEC + 1.0


def sample_pseudo_times(t_events, n_target, t_end, seed=0):
    """Sample n_target times from [MIN_EDGE, t_end - MIN_EDGE] that are
    ≥ MIN_GAP_FROM_EVENT from every event time. Rejection sample."""
    rng = np.random.default_rng(seed)
    lo = MIN_EDGE
    hi = t_end - MIN_EDGE
    if hi <= lo:
        return np.array([])
    out = []
    tries = 0
    while len(out) < n_target and tries < n_target * 200:
        t = rng.uniform(lo, hi)
        if len(t_events) == 0 or np.min(np.abs(t - t_events)) >= MIN_GAP_FROM_EVENT:
            out.append(t)
        tries += 1
    return np.array(out)


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
    pos = channel_positions(raw.ch_names)

    t_events = events['t0_net'].values.astype(float)
    n_target = int(len(t_events))
    if n_target == 0:
        return None
    t_pseudo = sample_pseudo_times(t_events, n_target, t_end,
                                     seed=abs(hash(sub_id)) % (2**31))

    rows = []
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
            nad_ch = per_channel_nadir(X_seg, fs, nadir)
            std_ch = float(np.nanstd(nad_ch))
            good = (np.isfinite(nad_ch) &
                    np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1]))
            if good.sum() < 6:
                continue
            X_fit = np.column_stack([pos[good, 0], pos[good, 1],
                                     np.ones(good.sum())])
            y_fit = nad_ch[good]
            coefs, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
            y_pred = X_fit @ coefs
            ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rows.append({
                'subject_id': sub_id,
                't_pseudo': float(t0),
                'R2': float(r2),
                'nadir_std_s': std_ch,
                'slope_x': float(coefs[0]),
                'slope_y': float(coefs[1]),
            })
        except Exception:
            continue
    return rows


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
        out = pool.map(process_subject, tasks)
    rows = [r for sub_rows in out if sub_rows for r in sub_rows]
    print(f"Pseudo-events scored: {len(rows)}")
    pseudo = pd.DataFrame(rows)
    pseudo.to_csv(os.path.join(OUT_DIR, 'propagation_null_random.csv'), index=False)

    # Load real-event B3 stats
    real = pd.read_csv(REAL_B3_CSV)
    print("\n=== Real events (pooled across quartiles) ===")
    print(f"  R² median {real['R2_median'].mean():.3f}")

    # Recompute real per-event R² by loading the source quality-stratified
    # output — we need per-event R², not per-quartile medians.
    # Fall back: look for per-event output in the mechanism_by_quality script
    # (not saved); we pool the IQRs from the quartile table as an approx.
    # For the head-to-head, load pseudo vs quartile-median real:
    real_r2_representative = np.array([])

    # Compare pseudo to real-event R² distribution derived from raw output.
    # Use the pooled IQRs to recreate a rough real-event distribution — or
    # just state quartile medians. Here we just plot pseudo distribution
    # and the 4 quartile medians as reference lines.
    r2_pseudo = pseudo['R2'].dropna().values
    std_pseudo = pseudo['nadir_std_s'].dropna().values
    sy_pseudo = pseudo['slope_y'].dropna().values

    print(f"\n=== Pseudo events (null) ===")
    print(f"  R² median {np.median(r2_pseudo):.3f}  "
          f"IQR [{np.percentile(r2_pseudo,25):.3f}, {np.percentile(r2_pseudo,75):.3f}]")
    print(f"  nadir_std (s) median {np.median(std_pseudo):.3f}")
    print(f"  slope_y median {np.median(sy_pseudo):+.2f} s/m")
    print(f"  % nadir_std > 200ms: {(std_pseudo>0.20).mean()*100:.1f}%")

    print(f"\n=== Real-event quartile medians (from B8 csv) ===")
    for _, row in real.iterrows():
        print(f"  {row['q']}: R² {row['R2_median']:.3f}   "
              f"nadir_std {row['nadir_std_median_s']:.2f}s   "
              f"slope_y {row['slope_y_median']:+.2f}")

    # Compute per-subject median R² for real (approx via quartile) vs pseudo
    real_r2_median_overall = float(np.mean(real['R2_median']))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R² distributions
    ax = axes[0]
    bins = np.linspace(0, 1, 30)
    ax.hist(r2_pseudo, bins=bins, color='gray', alpha=0.7,
            label=f'pseudo (median {np.median(r2_pseudo):.2f})')
    for _, row in real.iterrows():
        ax.axvline(row['R2_median'], lw=1.2, ls='--',
                   color='steelblue' if row['q'] == 'Q1' else 'firebrick' if row['q'] == 'Q4' else 'darkgray',
                   label=f"real {row['q']} median {row['R2_median']:.2f}")
    ax.set_xlabel('gradient R²')
    ax.set_ylabel('events')
    ax.set_title('Propagation R² · pseudo vs real-event quartiles')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # nadir_std distribution
    ax = axes[1]
    bins2 = np.linspace(0, 1.8, 30)
    ax.hist(std_pseudo, bins=bins2, color='gray', alpha=0.7,
            label=f'pseudo (median {np.median(std_pseudo):.2f} s)')
    for _, row in real.iterrows():
        ax.axvline(row['nadir_std_median_s'], lw=1.2, ls='--',
                   color='steelblue' if row['q'] == 'Q1' else 'firebrick' if row['q'] == 'Q4' else 'darkgray',
                   label=f"real {row['q']} {row['nadir_std_median_s']:.2f}s")
    ax.axvline(0.20, color='red', ls='--', lw=1, label='200ms threshold')
    ax.set_xlabel('nadir std across channels (s)')
    ax.set_ylabel('events')
    ax.set_title('Channel-wise nadir dispersion')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # slope_y distribution
    ax = axes[2]
    bins3 = np.linspace(-15, 15, 30)
    ax.hist(sy_pseudo, bins=bins3, color='gray', alpha=0.7,
            label=f'pseudo (median {np.median(sy_pseudo):+.1f})')
    for _, row in real.iterrows():
        ax.axvline(row['slope_y_median'], lw=1.2, ls='--',
                   color='steelblue' if row['q'] == 'Q1' else 'firebrick' if row['q'] == 'Q4' else 'darkgray',
                   label=f"real {row['q']} {row['slope_y_median']:+.1f}")
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('slope_y (s/m, + = anterior later)')
    ax.set_ylabel('events')
    ax.set_title('A-P propagation direction')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(f'B9 — Propagation null · {len(r2_pseudo)} pseudo-events vs real-event quartiles',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'propagation_null_random.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/propagation_null_random.png")
    print(f"Saved: {OUT_DIR}/propagation_null_random.csv")


if __name__ == '__main__':
    main()
