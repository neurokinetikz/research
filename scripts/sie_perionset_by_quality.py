#!/usr/bin/env python3
"""
B7 — Peri-onset grand averages stratified by event quality (template_rho).

Uses the quality scores from sie_event_quality.py (per_event_quality.csv) to
split events into template_rho quartiles (Q1: worst fit, Q4: best fit), then
computes grand-average envelope, Kuramoto R, and PLV traces (with subject-level
cluster bootstrap 95% CIs) separately for Q1 and Q4.

Tests the prediction: if template_rho captures a real "clean ignition" axis,
Q4 events should show a sharper / deeper dip-rebound than Q1.

Also reports peak-rebound amplitude, nadir depth, and FWHM per quartile.
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
from scripts.sie_perionset_triple_average import compute_streams

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'quality')
QUALITY_CSV = os.path.join(OUT_DIR, 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

PRE_SEC, POST_SEC, PAD_SEC, STEP_SEC = 10.0, 10.0, 2.0, 0.1
TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC / 2, STEP_SEC)


def process_subject(args):
    sub_id, df_sub = args
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X_all = raw.get_data() * 1e6
    t_end = raw.times[-1]

    out = {q: {'env': [], 'R': [], 'P': []} for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    for _, ev in df_sub.iterrows():
        t0 = float(ev['t0_net'])
        q = ev['rho_q']
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end:
            continue
        i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
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
        out[q]['env'].append(np.interp(TGRID, rel_env, zenv, left=np.nan, right=np.nan))
        out[q]['R'].append(np.interp(TGRID, rel_R, R, left=np.nan, right=np.nan))
        out[q]['P'].append(np.interp(TGRID, rel_P, P, left=np.nan, right=np.nan))

    result = {'subject_id': sub_id}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        for stream in ['env', 'R', 'P']:
            if out[q][stream]:
                result[f'{q}_{stream}'] = np.nanmean(np.array(out[q][stream]), axis=0)
                result[f'{q}_{stream}_n'] = len(out[q][stream])
            else:
                result[f'{q}_{stream}'] = None
                result[f'{q}_{stream}_n'] = 0
    return result


def bootstrap_ci(mat, n_boot=1000, seed=0):
    """Cluster bootstrap (resample rows = subjects)."""
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


def nadir_and_rebound_stats(trace, t=TGRID):
    """Return (nadir_depth, nadir_time, rebound_peak, rebound_time)."""
    dip_win = (t >= -2) & (t <= +2)
    reb_win = (t >= +2) & (t <= +6)
    if not np.any(np.isfinite(trace[dip_win])) or not np.any(np.isfinite(trace[reb_win])):
        return np.nan, np.nan, np.nan, np.nan
    dip_idx = np.nanargmin(trace[dip_win])
    reb_idx = np.nanargmax(trace[reb_win])
    nadir_t = t[dip_win][dip_idx]
    reb_t = t[reb_win][reb_idx]
    return float(trace[dip_win][dip_idx]), float(nadir_t), float(trace[reb_win][reb_idx]), float(reb_t)


def main():
    q_df = pd.read_csv(QUALITY_CSV)
    q_df = q_df.dropna(subset=['template_rho']).reset_index(drop=True)
    q_df['rho_q'] = pd.qcut(q_df['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print(f"Events: {len(q_df)}  subjects: {q_df['subject_id'].nunique()}")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = q_df[q_df['rho_q'] == q]
        print(f"  {q}: n_events={len(sub)}  template_rho median={sub['template_rho'].median():.3f}")

    tasks = [(sid, g) for sid, g in q_df.groupby('subject_id')]
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Processing {len(tasks)} subjects  workers: {n_workers}")
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    # Stack per-subject averages (NaN for subjects with no events in that quartile)
    def stack(key):
        arr = []
        for r in results:
            v = r[key]
            arr.append(v if v is not None else np.full(len(TGRID), np.nan))
        return np.array(arr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    streams = [('env', 'envelope z'), ('R', 'Kuramoto R'), ('P', 'mean PLV')]
    colors = {'Q1': '#4575b4', 'Q4': '#d73027'}

    stats_rows = []
    for ax, (s, label) in zip(axes, streams):
        for q in ['Q1', 'Q4']:
            mat = stack(f'{q}_{s}')
            grand, lo, hi = bootstrap_ci(mat)
            n_sub = (~np.all(np.isnan(mat), axis=1)).sum()
            total_events = sum(r[f'{q}_{s}_n'] for r in results)
            ax.plot(TGRID, grand, color=colors[q], lw=2,
                    label=f'{q} (n_sub={n_sub}, n_ev={total_events})')
            ax.fill_between(TGRID, lo, hi, color=colors[q], alpha=0.2)
            if s == 'env':
                nd, nt, rp, rt = nadir_and_rebound_stats(grand)
                stats_rows.append({
                    'stream': s, 'quartile': q, 'n_sub': n_sub,
                    'n_events': total_events,
                    'nadir_depth_z': nd, 'nadir_time_s': nt,
                    'rebound_peak_z': rp, 'rebound_time_s': rt,
                    'dip_rebound_range': rp - nd if np.isfinite(rp) and np.isfinite(nd) else np.nan,
                })
        ax.axvline(0, color='k', ls='--', lw=0.5)
        ax.set_xlabel('time rel. t0_net (s)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} — Q1 vs Q4 by template_rho')
        ax.legend(fontsize=9, loc='lower right' if s == 'env' else 'upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim(-10, 10)

    plt.suptitle(f'B7 — Peri-onset grand averages stratified by template_rho',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'perionset_by_rho_quartile.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(os.path.join(OUT_DIR, 'perionset_by_quality_stats.csv'), index=False)

    print("\n=== envelope z: Q1 vs Q4 nadir/rebound ===")
    print(stats.to_string(index=False))
    print(f"\nSaved: {OUT_DIR}/perionset_by_rho_quartile.png")
    print(f"Saved: {OUT_DIR}/perionset_by_quality_stats.csv")


if __name__ == '__main__':
    main()
