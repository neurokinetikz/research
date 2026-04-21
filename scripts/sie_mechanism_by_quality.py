#!/usr/bin/env python3
"""
B8 — Mechanism battery (B3 propagation + B5 phase-reset) stratified by
template_rho quartile.

Tests whether filtering to high-quality events (Q4 template_rho) sharpens the
mechanism signatures:
  - Propagation: R² of the per-channel nadir-time fit on (x,y), slope direction
  - Phase reset: rate of peri-nadir phase jumps elevated vs baseline

Outputs Q1 vs Q4 comparisons for each metric and per-event distributions.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import mannwhitneyu
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import bandpass
from scripts.sie_dip_onset_and_narrow_fooof import compute_streams_4way
from scripts.sie_perionset_multistream import (
    F0, HALF_BW, R_BAND, PRE_SEC, POST_SEC, PAD_SEC, STEP_SEC, find_nadir,
)
from scripts.sie_mechanism_battery import (
    per_channel_nadir, channel_positions, phase_jumps,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'quality')
QUALITY_CSV = os.path.join(OUT_DIR, 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

PJ_TGRID = np.arange(-8.0, 8.0 + 0.05, 0.1)


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
    pos = channel_positions(raw.ch_names)

    # per-quartile accumulators
    buckets = {q: {'nadir_std': [], 'r2': [], 'slope_x': [], 'slope_y': [],
                    'pj_rows': []} for q in ['Q1', 'Q2', 'Q3', 'Q4']}

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
            t_c, env, R, P, M = compute_streams_4way(X_seg, fs)
            rel = t_c - PAD_SEC - PRE_SEC
            nadir = find_nadir(rel, env, R, P, M)
            if not np.isfinite(nadir):
                continue
        except Exception:
            continue

        # B3: per-channel nadir timing + gradient fit
        try:
            nad_ch = per_channel_nadir(X_seg, fs, nadir)
            std_ch = float(np.nanstd(nad_ch))
            buckets[q]['nadir_std'].append(std_ch)
            good = (np.isfinite(nad_ch) &
                    np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1]))
            if good.sum() >= 6:
                X_fit = np.column_stack([pos[good, 0], pos[good, 1],
                                         np.ones(good.sum())])
                y_fit = nad_ch[good]
                coefs, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
                y_pred = X_fit @ coefs
                ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
                ss_res = np.sum((y_fit - y_pred) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                buckets[q]['r2'].append(float(r2))
                buckets[q]['slope_x'].append(float(coefs[0]))
                buckets[q]['slope_y'].append(float(coefs[1]))
        except Exception:
            pass

        # B5: phase-jump rate aligned on nadir
        try:
            t_pj, pj = phase_jumps(X_seg, fs)
            rel_pj = t_pj - PAD_SEC - PRE_SEC - nadir
            buckets[q]['pj_rows'].append(
                np.interp(PJ_TGRID, rel_pj, pj, left=np.nan, right=np.nan))
        except Exception:
            pass

    # subject-level aggregates
    out = {'subject_id': sub_id}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        b = buckets[q]
        out[f'{q}_nadir_std']  = np.array(b['nadir_std'])
        out[f'{q}_r2']         = np.array(b['r2'])
        out[f'{q}_slope_x']    = np.array(b['slope_x'])
        out[f'{q}_slope_y']    = np.array(b['slope_y'])
        out[f'{q}_pj_mean']    = (np.nanmean(np.array(b['pj_rows']), axis=0)
                                   if b['pj_rows'] else None)
        out[f'{q}_n_events']   = len(b['nadir_std'])
    return out


def bootstrap_ci(mat, n_boot=1000, seed=0):
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


def main():
    q_df = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).reset_index(drop=True)
    q_df['rho_q'] = pd.qcut(q_df['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    tasks = [(sid, g) for sid, g in q_df.groupby('subject_id')]
    print(f"Events: {len(q_df)}  subjects: {len(tasks)}")

    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    # Flatten per-event arrays per quartile
    def cat(q, key):
        arr = [r[f'{q}_{key}'] for r in results if len(r[f'{q}_{key}']) > 0]
        return np.concatenate(arr) if arr else np.array([])

    print("\n=== B3 propagation — per-event stats by template_rho quartile ===")
    rows = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        r2 = cat(q, 'r2')
        sd = cat(q, 'nadir_std')
        sy = cat(q, 'slope_y')
        sx = cat(q, 'slope_x')
        rows.append({
            'q': q, 'n_events': len(r2),
            'R2_median': float(np.median(r2)) if len(r2) else np.nan,
            'R2_IQR_lo': float(np.percentile(r2, 25)) if len(r2) else np.nan,
            'R2_IQR_hi': float(np.percentile(r2, 75)) if len(r2) else np.nan,
            'nadir_std_median_s': float(np.median(sd)) if len(sd) else np.nan,
            'slope_y_median': float(np.median(sy)) if len(sy) else np.nan,
            'slope_x_median': float(np.median(sx)) if len(sx) else np.nan,
            'pct_prop_like_std_gt_200ms': float((sd > 0.20).mean() * 100) if len(sd) else np.nan,
        })
    b3 = pd.DataFrame(rows)
    print(b3.round(3).to_string(index=False))
    b3.to_csv(os.path.join(OUT_DIR, 'mechanism_by_quality_B3.csv'), index=False)

    # MWU tests Q4 vs Q1
    r2_Q1 = cat('Q1', 'r2'); r2_Q4 = cat('Q4', 'r2')
    sd_Q1 = cat('Q1', 'nadir_std'); sd_Q4 = cat('Q4', 'nadir_std')
    u_r2, p_r2 = mannwhitneyu(r2_Q4, r2_Q1, alternative='greater')
    u_sd, p_sd = mannwhitneyu(sd_Q4, sd_Q1, alternative='two-sided')
    print(f"\nMWU Q4 > Q1 on gradient R²: U={u_r2:.0f}  p={p_r2:.3g}")
    print(f"MWU Q4 vs Q1 on nadir_std (two-sided): U={u_sd:.0f}  p={p_sd:.3g}")

    # Phase jump subject means → grand mean + CI
    def pj_stack(q):
        arr = []
        for r in results:
            v = r[f'{q}_pj_mean']
            if v is not None:
                arr.append(v)
        return np.array(arr) if arr else np.empty((0, len(PJ_TGRID)))

    pj_Q1 = pj_stack('Q1')
    pj_Q4 = pj_stack('Q4')
    pj_Q1_m, pj_Q1_lo, pj_Q1_hi = bootstrap_ci(pj_Q1) if len(pj_Q1) else (None, None, None)
    pj_Q4_m, pj_Q4_lo, pj_Q4_hi = bootstrap_ci(pj_Q4) if len(pj_Q4) else (None, None, None)

    # Elevation: peri-nadir [-1, +1] vs baseline [-8, -4]
    def elevation(traj):
        if traj is None:
            return np.nan
        peri = (PJ_TGRID >= -1) & (PJ_TGRID <= +1)
        base = (PJ_TGRID >= -8) & (PJ_TGRID <= -4)
        m_peri = np.nanmean(traj[peri]); m_base = np.nanmean(traj[base])
        return (m_peri / m_base) if m_base > 0 else np.nan
    print(f"\n=== B5 phase-jump elevation (peri-nadir / baseline) ===")
    print(f"  Q1: {elevation(pj_Q1_m):.2f}×   n_subjects={len(pj_Q1)}")
    print(f"  Q4: {elevation(pj_Q4_m):.2f}×   n_subjects={len(pj_Q4)}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R² distribution Q1 vs Q4
    ax = axes[0]
    bins = np.linspace(0, 1, 25)
    ax.hist(r2_Q1, bins=bins, color='#4575b4', alpha=0.6, label=f'Q1 (n={len(r2_Q1)})')
    ax.hist(r2_Q4, bins=bins, color='#d73027', alpha=0.6, label=f'Q4 (n={len(r2_Q4)})')
    ax.axvline(np.median(r2_Q1), color='#4575b4', ls='--', lw=1.5)
    ax.axvline(np.median(r2_Q4), color='#d73027', ls='--', lw=1.5)
    ax.set_xlabel('gradient R² (per-channel nadir fit)')
    ax.set_ylabel('events')
    ax.set_title(f'B3 R² · Q4 median {np.median(r2_Q4):.2f} vs Q1 {np.median(r2_Q1):.2f}\np={p_r2:.2g}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Slope_y direction
    ax = axes[1]
    sy_Q1 = cat('Q1', 'slope_y'); sy_Q4 = cat('Q4', 'slope_y')
    bins2 = np.linspace(-15, 15, 30)
    ax.hist(sy_Q1, bins=bins2, color='#4575b4', alpha=0.6, label=f'Q1 median {np.median(sy_Q1):+.1f}')
    ax.hist(sy_Q4, bins=bins2, color='#d73027', alpha=0.6, label=f'Q4 median {np.median(sy_Q4):+.1f}')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('slope_y (s/m)  (+ = anterior later)')
    ax.set_ylabel('events')
    ax.set_title('B3 anterior→posterior slope')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Phase-jump traces
    ax = axes[2]
    if pj_Q1_m is not None:
        ax.plot(PJ_TGRID, pj_Q1_m, color='#4575b4', lw=2, label=f'Q1 n_sub={len(pj_Q1)}')
        ax.fill_between(PJ_TGRID, pj_Q1_lo, pj_Q1_hi, color='#4575b4', alpha=0.25)
    if pj_Q4_m is not None:
        ax.plot(PJ_TGRID, pj_Q4_m, color='#d73027', lw=2, label=f'Q4 n_sub={len(pj_Q4)}')
        ax.fill_between(PJ_TGRID, pj_Q4_lo, pj_Q4_hi, color='#d73027', alpha=0.25)
    ax.axvline(0, color='k', ls='--', lw=0.5, label='nadir')
    ax.set_xlabel('time rel. nadir (s)')
    ax.set_ylabel('phase-jump rate (counts/100ms)')
    ax.set_title(f'B5 phase jumps · Q4 {elevation(pj_Q4_m):.2f}× vs Q1 {elevation(pj_Q1_m):.2f}×')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    plt.suptitle('B8 — Mechanism battery stratified by template_rho quartile',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mechanism_by_quality.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/mechanism_by_quality.png")


if __name__ == '__main__':
    main()
