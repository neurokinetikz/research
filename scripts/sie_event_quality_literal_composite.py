#!/usr/bin/env python3
"""
B6 literal re-run on composite v2 detector.

Replicates envelope B6's five quality axes, per composite event, using the
composite's OWN 4-stream S(t) (env, R, PLV, MSC) computed via the vectorized
MSC estimator from lib/detect_ignition.py (~100× faster than per-channel
scipy.signal.coherence).

Five axes per event:
  1. peak_S         max of S(t) = cbrt(zE·zR·zP·zM) in [t0_net − 5, t0_net + 5] s
  2. S_fwhm_s       FWHM of S(t) around its peak (in seconds)
  3. template_ρ     Pearson correlation of this event's env z trajectory on
                    [−5, +5] s with cohort grand-average template
  4. spatial_coh    fraction of channels whose per-channel env nadir falls
                    within ±0.3 s of the median channel nadir
  5. baseline_calm  1 / std(env z) on [−10, −3] s

Cohort-parameterized. Runs LEMON EC + optionally EO. Only EC by default.

Outputs: outputs/schumann/images/quality/<cohort>_composite/
  - per_event_quality_literal.csv
  - event_quality_overview_literal.png
  - cross-axis Spearman correlation matrix

Usage:
    python scripts/sie_event_quality_literal_composite.py --cohort lemon
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)
from scripts.sie_perionset_triple_average_composite import bandpass
from lib.detect_ignition import _composite_streams, _composite_robust_z

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)
PRE_SEC = 10.0
POST_SEC = 10.0
STEP_SEC = 0.1
TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC / 2, STEP_SEC)

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events
    if cohort == 'srm':
        return load_srm, {}, events
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events
    if cohort == 'dortmund':
        return load_dortmund, {}, events
    if cohort == 'chbmp':
        return load_chbmp, {}, events
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events
    raise ValueError(f"unsupported cohort {cohort!r}")


def compute_full_4stream_S(X_uV, fs):
    """Full-recording 4-stream S(t) using vectorized MSC.
    Returns (t, zE, zR, zP, zM, S) at STEP_SEC resolution."""
    # _composite_streams returns (t, env, R, PLV, MSC) at step_sec
    t, env, R, P, M = _composite_streams(X_uV, fs,
                                          f0=F0, half_bw=HALF_BW,
                                          R_band=R_BAND,
                                          step_sec=STEP_SEC, win_sec=1.0)
    zE = _composite_robust_z(env)
    zR = _composite_robust_z(R)
    zP = _composite_robust_z(P)
    zM = _composite_robust_z(M)
    S = np.cbrt(np.clip(zE, 0, None) *
                np.clip(zR, 0, None) *
                np.clip(zP, 0, None) *
                np.clip(zM, 0, None))
    return t, zE, zR, zP, zM, S


def compute_chan_env(X_uV, fs):
    Xb = bandpass(X_uV, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(Xb, axis=-1))
    return env


def fwhm_samples(y, peak_idx):
    if peak_idx < 0 or peak_idx >= len(y):
        return np.nan
    half = y[peak_idx] / 2.0
    if half <= 0:
        return np.nan
    L = peak_idx
    while L > 0 and y[L] > half:
        L -= 1
    R = peak_idx
    while R < len(y) - 1 and y[R] > half:
        R += 1
    return R - L


_LOADER = None
_LOADER_KW = None


def _init_worker(loader_name, loader_kw):
    global _LOADER, _LOADER_KW
    _LOADER_KW = loader_kw
    _LOADER = {
        'load_lemon': load_lemon,
        'load_tdbrain': load_tdbrain,
        'load_srm': load_srm,
        'load_dortmund': load_dortmund,
        'load_chbmp': load_chbmp,
        'load_hbn_by_subject': load_hbn_by_subject,
    }[loader_name]


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) == 0:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X_all = raw.get_data() * 1e6
    t_end = raw.times[-1]

    try:
        t_full, zE_full, zR_full, zP_full, zM_full, S_full = compute_full_4stream_S(
            X_all, fs)
    except Exception:
        return None

    try:
        chan_env = compute_chan_env(X_all, fs)
    except Exception:
        return None
    n_ch = chan_env.shape[0]

    rows = []
    event_env_traj = []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        if t0 - PRE_SEC < 2 or t0 + POST_SEC > t_end - 2:
            continue

        sel = (t_full >= t0 - PRE_SEC) & (t_full <= t0 + POST_SEC)
        if sel.sum() < int(PRE_SEC + POST_SEC) * 5:
            continue
        t_rel = t_full[sel] - t0
        zE_seg = zE_full[sel]
        S_seg = S_full[sel]

        env_i = np.interp(TGRID, t_rel, zE_seg, left=np.nan, right=np.nan)
        S_i = np.interp(TGRID, t_rel, S_seg, left=np.nan, right=np.nan)

        # peak_S + S_fwhm_s on [-5, +5]
        m_core = (TGRID >= -5) & (TGRID <= +5)
        if not np.any(np.isfinite(S_i[m_core])):
            continue
        peak_idx_core = int(np.nanargmax(S_i[m_core]))
        core_idx = np.where(m_core)[0]
        peak_idx = core_idx[peak_idx_core]
        peak_S = float(S_i[peak_idx])
        peak_S_lat = float(TGRID[peak_idx])
        s_fwhm = float(fwhm_samples(S_i, peak_idx) * STEP_SEC)

        # baseline_calm
        m_pre = (TGRID >= -10) & (TGRID <= -3)
        pre_std = float(np.nanstd(env_i[m_pre]))
        baseline_calm = 1.0 / (pre_std + 1e-3)

        # spatial_coh
        i0_s = int(round((t0 - 3) * fs))
        i1_s = int(round((t0 + 3) * fs))
        if i0_s < 0 or i1_s > chan_env.shape[1]:
            spatial_coh = np.nan
            chan_nadir_jitter = np.nan
        else:
            ce = chan_env[:, i0_s:i1_s]
            ch_nadir_t = (np.argmin(ce, axis=1) / fs) - 3.0
            mean_ch_nadir = float(np.nanmedian(ch_nadir_t))
            spatial_coh = float(np.mean(np.abs(ch_nadir_t - mean_ch_nadir) <= 0.3))
            chan_nadir_jitter = float(np.nanstd(ch_nadir_t))

        rows.append({
            'subject_id': sub_id,
            't0_net': t0,
            'peak_S': peak_S,
            'peak_S_lat': peak_S_lat,
            'S_fwhm_s': s_fwhm,
            'baseline_calm': baseline_calm,
            'pre_env_std': pre_std,
            'spatial_coh': spatial_coh,
            'chan_nadir_jitter_s': chan_nadir_jitter,
            'n_ch': n_ch,
        })
        event_env_traj.append(env_i)

    if not rows:
        return None
    return {
        'subject_id': sub_id,
        'rows': rows,
        'env_traj': np.array(event_env_traj),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'quality', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 2)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}   (vectorized 4-stream S(t))")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    all_rows = []
    all_env = []
    for r in results:
        all_rows.extend(r['rows'])
        all_env.append(r['env_traj'])
    df = pd.DataFrame(all_rows)
    env_mat = np.vstack(all_env)
    print(f"Total scored events: {len(df)}")

    # template_ρ per event using cohort grand-average template
    m_core = (TGRID >= -5) & (TGRID <= +5)
    template = np.nanmean(env_mat, axis=0)
    tmpl_core = template[m_core]
    tmpl_core = tmpl_core - np.nanmean(tmpl_core)
    rhos = []
    for i in range(env_mat.shape[0]):
        ev = env_mat[i, m_core]
        if np.any(~np.isfinite(ev)):
            rhos.append(np.nan); continue
        ev_c = ev - np.nanmean(ev)
        denom = np.sqrt(np.nansum(ev_c**2) * np.nansum(tmpl_core**2))
        rhos.append(float(np.nansum(ev_c * tmpl_core) / denom) if denom > 0 else np.nan)
    df['template_rho_literal'] = rhos

    df.to_csv(os.path.join(out_dir, 'per_event_quality_literal.csv'), index=False)

    axes_cols = ['peak_S', 'template_rho_literal', 'spatial_coh', 'baseline_calm', 'S_fwhm_s']
    print(f"\n=== {args.cohort} composite literal · per-event quality axes (median [IQR]) ===")
    print(f"(envelope B6 medians: peak_S 0.93, template_ρ 0.59, spatial_coh 0.15, baseline_calm 1.16, S_fwhm 1.2)")
    for c in axes_cols:
        v = df[c].dropna()
        print(f"  {c:22s} median {v.median():.3f}  IQR [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]")

    # Spearman correlation matrix
    X = df[axes_cols].dropna()
    corr_mat = np.zeros((len(axes_cols), len(axes_cols)))
    for i, a in enumerate(axes_cols):
        for j, b in enumerate(axes_cols):
            rho, _ = spearmanr(X[a], X[b])
            corr_mat[i, j] = rho
    print(f"\n=== {args.cohort} composite literal · Spearman ρ ===")
    print(pd.DataFrame(corr_mat, index=axes_cols, columns=axes_cols).round(2).to_string())

    # Compute max |ρ| off-diagonal
    off = corr_mat - np.eye(len(axes_cols))
    max_abs = float(np.max(np.abs(off)))
    print(f"\nMax |ρ| off-diagonal: {max_abs:.3f}  (envelope B6: 0.28 — axes largely orthogonal)")

    # Multi-axis threshold passing: top-25% on each of 4 core axes
    # (peak_S, template_rho, spatial_coh, baseline_calm)
    core = ['peak_S', 'template_rho_literal', 'spatial_coh', 'baseline_calm']
    q75 = {a: df[a].quantile(0.75) for a in core}
    passes = np.ones(len(df), dtype=bool)
    for a in core:
        passes &= (df[a] >= q75[a]).values
    print(f"\n=== {args.cohort} composite literal · multi-axis threshold ===")
    print(f"(envelope B6: 6.2% of events pass all 4 thresholds)")
    print(f"Events passing all 4 top-25% axes: {passes.sum()} / {len(df)} = {passes.sum()/len(df)*100:.1f}%")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im = axes[0].imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(axes_cols))); axes[0].set_yticks(range(len(axes_cols)))
    axes[0].set_xticklabels(axes_cols, rotation=45, ha='right')
    axes[0].set_yticklabels(axes_cols)
    for i in range(len(axes_cols)):
        for j in range(len(axes_cols)):
            axes[0].text(j, i, f'{corr_mat[i,j]:.2f}', ha='center', va='center',
                         color='k' if abs(corr_mat[i,j]) < 0.5 else 'w', fontsize=9)
    axes[0].set_title(f'Spearman ρ, quality axes · max |ρ|={max_abs:.2f}')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    axes[1].scatter(df['peak_S'], df['template_rho_literal'], s=5, alpha=0.3,
                    color='steelblue')
    rho_st, p_st = spearmanr(df['peak_S'].dropna(),
                              df.loc[df['peak_S'].notna(), 'template_rho_literal'])
    axes[1].set_xlabel('peak_S')
    axes[1].set_ylabel('template_ρ')
    axes[1].set_title(f'peak_S vs template_ρ · ρ = {rho_st:.2f}  p = {p_st:.2g}')
    axes[1].axhline(0, color='k', lw=0.5); axes[1].grid(alpha=0.3)

    # Grand averages by peak_S quartile
    q = pd.qcut(df['peak_S'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    colors = ['#4575b4', '#91bfdb', '#fc8d59', '#d73027']
    for qi, lab in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        idx = (q == lab).values
        if idx.sum() == 0:
            continue
        mean_trace = np.nanmean(env_mat[idx], axis=0)
        axes[2].plot(TGRID, mean_trace, color=colors[qi],
                     label=f'{lab} (n={idx.sum()})', lw=2)
    axes[2].axvline(0, color='k', ls='--', lw=0.5)
    axes[2].set_xlabel('time rel. t0_net (s)')
    axes[2].set_ylabel('envelope z')
    axes[2].set_title('Grand-average env z by peak_S quartile')
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.suptitle(f'B6 literal · event quality 5-axis scoring · {args.cohort} composite v2 · '
                 f'{len(df)} events · {df["subject_id"].nunique()} subjects',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'event_quality_overview_literal.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/event_quality_overview_literal.png")
    print(f"Saved: {out_dir}/per_event_quality_literal.csv")


if __name__ == '__main__':
    main()
