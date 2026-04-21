#!/usr/bin/env python3
"""
A5 — Composite-detector on raw LEMON EC recordings.

For each subject:
  1. Compute envelope z(t), Kuramoto R(t), mean PLV(t) across the ENTIRE
     recording at 10 Hz resolution (100-ms step).
  2. Robust-z each stream (median / MAD) against the full recording.
  3. Build S(t) = cbrt(max(zE,0) * max(zR,0) * max(zP,0)).
  4. Detect events: local maxima of S(t) ≥ threshold, min ISI = 2 s,
     edge-masked 5 s.
  5. Choose threshold per-subject to match the number of current Stage-1 events
     for that subject (match by rank — top-n peaks).
  6. Compare composite event times to current Stage-1 t0_net:
       - For each composite event, find nearest current event
       - Bin into matched (Δt ≤ 2 s), shifted (2 < Δt ≤ 5), unique (> 5)

Outputs:
  - CSV of per-subject match statistics
  - CSV of per-event match table (composite → nearest current)
  - Figure: overlap rates, offset distributions
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

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'composite_detector')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)
STEP_SEC = 0.1
WIN_SEC = 1.0
EDGE_SEC = 5.0
MIN_ISI_SEC = 2.0


def robust_z(x):
    """Median/MAD-based z-score, robust to bursts."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-9:
        return x - med
    return (x - med) / mad


def compute_global_streams(X_uV, fs):
    """Return t, z_env, R, PLV sampled at STEP_SEC."""
    # Envelope z (full)
    y = X_uV.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))

    # Phase streams for R, PLV
    Xb = bandpass(X_uV, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    ref = np.median(Xb, axis=0)
    ph_ref = np.angle(signal.hilbert(ref))
    dphi = ph - ph_ref[None, :]

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    centers = []
    env_vals = []
    Rv = []
    Pv = []
    for i in range(0, X_uV.shape[1] - nwin + 1, nstep):
        seg_ph = ph[:, i:i+nwin]
        R_t = np.abs(np.mean(np.exp(1j * seg_ph), axis=0))
        Rv.append(float(np.mean(R_t)))
        pseg = dphi[:, i:i+nwin]
        plv = np.abs(np.mean(np.exp(1j * pseg), axis=1))
        Pv.append(float(np.mean(plv)))
        # env at window center
        env_vals.append(float(np.mean(env[i:i+nwin])))
        centers.append((i + nwin/2) / fs)
    return (np.array(centers), np.array(env_vals),
            np.array(Rv), np.array(Pv))


def detect_composite_events(t, S, n_target, edge_sec=EDGE_SEC, min_isi=MIN_ISI_SEC):
    """Top-n local maxima of S, with min-ISI enforcement, edge-masked."""
    mask = (t >= t[0] + edge_sec) & (t <= t[-1] - edge_sec)
    S_m = S.copy()
    S_m[~mask] = -np.inf
    # Find local maxima
    peak_idx, _ = signal.find_peaks(S_m, distance=int(round(min_isi / STEP_SEC)))
    if len(peak_idx) == 0:
        return np.array([])
    # Sort by amplitude, keep top-n
    order = peak_idx[np.argsort(-S_m[peak_idx])]
    top = order[:max(1, n_target)]
    return np.sort(t[top])


def process_subject(args):
    sub_id, events_path = args
    try:
        cur_events = pd.read_csv(events_path)
    except Exception:
        return None
    n_cur = len(cur_events.dropna(subset=['t0_net']))
    if n_cur == 0:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6

    try:
        t, env, R, P = compute_global_streams(X, fs)
    except Exception:
        return None

    zE = robust_z(env)
    zR = robust_z(R)
    zP = robust_z(P)
    S = np.cbrt(np.clip(zE, 0, None) * np.clip(zR, 0, None) * np.clip(zP, 0, None))

    # Top-n composite event times (n matched to current)
    comp_times = detect_composite_events(t, S, n_target=n_cur)

    # Current event times (t0_net)
    cur_times = np.sort(cur_events['t0_net'].dropna().values)

    # Match composite → nearest current
    offsets = []
    matched = []
    for ct in comp_times:
        diffs = cur_times - ct
        if len(diffs) == 0:
            offsets.append(np.nan)
            matched.append('none')
            continue
        nearest = diffs[np.argmin(np.abs(diffs))]
        offsets.append(nearest)
        if abs(nearest) <= 2.0:
            matched.append('matched')
        elif abs(nearest) <= 5.0:
            matched.append('shifted')
        else:
            matched.append('unique')

    matched = np.array(matched)
    n_matched = int((matched == 'matched').sum())
    n_shifted = int((matched == 'shifted').sum())
    n_unique = int((matched == 'unique').sum())

    return {
        'subject_id': sub_id,
        'n_cur': int(n_cur),
        'n_comp': int(len(comp_times)),
        'n_matched': n_matched,
        'n_shifted': n_shifted,
        'n_unique': n_unique,
        'frac_matched': n_matched / max(1, len(comp_times)),
        'median_offset': float(np.nanmedian(np.abs(offsets))) if offsets else np.nan,
        'offsets': np.array(offsets),
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

    rows = []
    all_offsets = []
    for r in results:
        rows.append({k: r[k] for k in
                     ['subject_id', 'n_cur', 'n_comp', 'n_matched',
                      'n_shifted', 'n_unique', 'frac_matched', 'median_offset']})
        all_offsets.extend(r['offsets'].tolist())
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'composite_detector_match_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    total_cur = df['n_cur'].sum()
    total_comp = df['n_comp'].sum()
    total_matched = df['n_matched'].sum()
    total_shifted = df['n_shifted'].sum()
    total_unique = df['n_unique'].sum()
    print(f"\nTotals across {len(df)} subjects:")
    print(f"  current events:    {total_cur:,}")
    print(f"  composite events:  {total_comp:,}")
    print(f"  matched (Δ≤2s):    {total_matched:,} ({total_matched/total_comp*100:.1f}%)")
    print(f"  shifted (2<Δ≤5s):  {total_shifted:,} ({total_shifted/total_comp*100:.1f}%)")
    print(f"  unique (Δ>5s):     {total_unique:,} ({total_unique/total_comp*100:.1f}%)")

    offsets_arr = np.array([o for o in all_offsets if np.isfinite(o)])
    print(f"\nOffset distribution (composite − current):")
    print(f"  median abs: {np.nanmedian(np.abs(offsets_arr)):.2f} s")
    print(f"  IQR abs:    [{np.nanpercentile(np.abs(offsets_arr),25):.2f}, "
          f"{np.nanpercentile(np.abs(offsets_arr),75):.2f}] s")
    print(f"  median:     {np.nanmedian(offsets_arr):+.2f} s")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) Per-subject match fraction
    ax = axes[0]
    ax.hist(df['frac_matched'], bins=25, color='steelblue', edgecolor='k', lw=0.3)
    ax.axvline(df['frac_matched'].median(), color='red', lw=1.2,
               label=f'median {df["frac_matched"].median():.2f}')
    ax.set_xlabel('per-subject matched fraction (Δ ≤ 2 s)')
    ax.set_ylabel('subjects')
    ax.set_title('Event-set agreement per subject')
    ax.legend(fontsize=8)

    # 2) Offset histogram (all events, composite - nearest current)
    ax = axes[1]
    clip = np.clip(offsets_arr, -30, 30)
    ax.hist(clip, bins=60, color='seagreen', edgecolor='k', lw=0.3)
    ax.axvline(0, color='k', ls='--', lw=0.6)
    ax.axvline(-2, color='red', ls=':', lw=0.6)
    ax.axvline(2, color='red', ls=':', lw=0.6, label='±2 s match window')
    ax.set_xlabel('composite − nearest current (s, clipped ±30)')
    ax.set_ylabel('events')
    ax.set_title('Onset-time offset distribution')
    ax.legend(fontsize=8)

    # 3) Stacked bar: matched/shifted/unique per subject
    ax = axes[2]
    totals = pd.DataFrame({
        'matched': [total_matched],
        'shifted': [total_shifted],
        'unique': [total_unique],
    })
    totals.plot(kind='bar', stacked=True, ax=ax,
                color=['#2ecc71', '#f39c12', '#e74c3c'], legend=True)
    ax.set_xticks([])
    ax.set_ylabel('event count')
    ax.set_title(f'Composite event classification\n(n={total_comp:,})')
    # annotations
    for cat, val in [('matched', total_matched), ('shifted', total_shifted),
                      ('unique', total_unique)]:
        pct = 100 * val / total_comp
        ax.text(0.5, val / 2 if cat == 'matched' else
                    total_matched + val / 2 if cat == 'shifted' else
                    total_matched + total_shifted + val / 2,
                f'{cat}\n{val:,} ({pct:.1f}%)',
                ha='center', va='center', fontsize=9)

    plt.suptitle(f'Composite detector vs current Stage 1 — LEMON EC\n'
                 f'{len(df)} subjects, event counts matched per subject',
                 fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'composite_vs_current.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
