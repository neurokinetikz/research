#!/usr/bin/env python3
"""
B6 — Event-quality scoring for SIE ignition events.

For every LEMON EC event, score five quality axes and test whether
high-quality events show a sharper dip-rebound grand average.

Axes:
  1. peak_S          — max of composite S(t) = cbrt(max(zE,0)·max(zR,0)·max(zP,0))
                       on [-5, +5] s around t0_net (streams robust-z'd over the
                       full recording).
  2. S_fwhm          — full-width-half-max of S(t) around its peak.
  3. template_rho    — Pearson correlation of this event's envelope z trajectory
                       (on [-5, +5] s) against the grand-average envelope template
                       computed from all events.
  4. spatial_coh     — fraction of channels whose per-channel envelope minimum
                       (on [-3, +3] s) falls within ±0.3 s of the event's mean-
                       channel nadir. Higher = more simultaneous.
  5. baseline_calm   — 1 / std of this event's envelope z on [-10, -3] s
                       (higher = cleaner pre-event baseline).

Outputs:
  - outputs/schumann/images/quality/per_event_quality.csv
  - Correlation heatmap of the five axes (Spearman).
  - Grand-average env / R / PLV stratified by peak_S quartile (sharpness test).
  - Scatter peak_S vs template_rho.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore, spearmanr
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import bandpass

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'quality')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)
PRE_SEC = 10.0
POST_SEC = 10.0
PAD_SEC = 2.0
STEP_SEC = 0.1
WIN_SEC = 1.0

TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC / 2, STEP_SEC)


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-9:
        return x - med
    return (x - med) / mad


def compute_full_streams(X_uV, fs):
    """Whole-recording z_env, R, PLV at STEP_SEC.  Returns robust-z'd streams."""
    y = X_uV.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))

    Xb = bandpass(X_uV, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    ref = np.median(Xb, axis=0)
    ph_ref = np.angle(signal.hilbert(ref))
    dphi = ph - ph_ref[None, :]

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    centers = []
    env_vals, Rv, Pv = [], [], []
    for i in range(0, X_uV.shape[1] - nwin + 1, nstep):
        env_vals.append(float(np.mean(env[i:i+nwin])))
        seg = ph[:, i:i+nwin]
        Rv.append(float(np.mean(np.abs(np.mean(np.exp(1j * seg), axis=0)))))
        pseg = dphi[:, i:i+nwin]
        Pv.append(float(np.mean(np.abs(np.mean(np.exp(1j * pseg), axis=1)))))
        centers.append((i + nwin / 2) / fs)
    t = np.array(centers)
    zE = robust_z(np.array(env_vals))
    zR = robust_z(np.array(Rv))
    zP = robust_z(np.array(Pv))
    S = np.cbrt(np.clip(zE, 0, None) *
                np.clip(zR, 0, None) *
                np.clip(zP, 0, None))
    return t, zE, zR, zP, S


def compute_chan_env(X_uV, fs):
    """Per-channel Hilbert envelope at native fs, narrowband F0."""
    Xb = bandpass(X_uV, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(Xb, axis=-1))
    return env


def fwhm(y, peak_idx):
    """Width at half maximum around peak_idx, in sample units."""
    if peak_idx < 0 or peak_idx >= len(y):
        return np.nan
    half = y[peak_idx] / 2.0
    L = peak_idx
    while L > 0 and y[L] > half:
        L -= 1
    R = peak_idx
    while R < len(y) - 1 and y[R] > half:
        R += 1
    return R - L


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

    try:
        t_full, zE_full, zR_full, zP_full, S_full = compute_full_streams(X_all, fs)
    except Exception:
        return None
    try:
        chan_env = compute_chan_env(X_all, fs)   # (n_ch, n_samples)
    except Exception:
        return None
    n_ch = chan_env.shape[0]
    t_end = raw.times[-1]

    rows = []
    event_env_traj = []   # z_env traj on TGRID per event
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        if t0 - PRE_SEC < 2 or t0 + POST_SEC > t_end - 2:
            continue

        # pull stream segments relative to t0
        sel = (t_full >= t0 - PRE_SEC) & (t_full <= t0 + POST_SEC)
        if sel.sum() < int(PRE_SEC + POST_SEC) * 5:
            continue
        t_rel = t_full[sel] - t0
        zE_seg = zE_full[sel]
        S_seg = S_full[sel]

        env_i = np.interp(TGRID, t_rel, zE_seg, left=np.nan, right=np.nan)
        S_i = np.interp(TGRID, t_rel, S_seg, left=np.nan, right=np.nan)

        # Peak_S on [-5, +5]
        m_core = (TGRID >= -5) & (TGRID <= +5)
        if not np.any(np.isfinite(S_i[m_core])):
            continue
        peak_idx_core = np.nanargmax(S_i[m_core])
        core_idx = np.where(m_core)[0]
        peak_idx = core_idx[peak_idx_core]
        peak_S = S_i[peak_idx]
        peak_S_lat = TGRID[peak_idx]
        s_fwhm = fwhm(S_i, peak_idx) * STEP_SEC  # seconds

        # Baseline calm (pre-event env std)
        m_pre = (TGRID >= -10) & (TGRID <= -3)
        pre_std = float(np.nanstd(env_i[m_pre]))
        baseline_calm = 1.0 / (pre_std + 1e-3)

        # Spatial coherence: per-channel env nadir on [-3, +3] s
        i0_s = int(round((t0 - 3) * fs))
        i1_s = int(round((t0 + 3) * fs))
        if i0_s < 0 or i1_s > chan_env.shape[1]:
            spatial_coh = np.nan
            chan_nadir_jitter = np.nan
        else:
            ce = chan_env[:, i0_s:i1_s]  # (n_ch, n_samples)
            # channel-level nadirs (in seconds rel to t0)
            ch_nadir_t = (np.argmin(ce, axis=1) / fs) - 3.0
            mean_ch_nadir = float(np.nanmedian(ch_nadir_t))
            # fraction of channels within ±0.3 s of median nadir
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
        'env_traj': np.array(event_env_traj),  # (n_events, len(TGRID))
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 2)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Subjects to process: {len(tasks)}  workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    all_rows = []
    all_env = []
    for r in results:
        all_rows.extend(r['rows'])
        all_env.append(r['env_traj'])
    df = pd.DataFrame(all_rows)
    env_mat = np.vstack(all_env)  # (n_events, len(TGRID))
    print(f"Total scored events: {len(df)}")

    # Grand-average template from all events, then template_rho per event
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
    df['template_rho'] = rhos

    df.to_csv(os.path.join(OUT_DIR, 'per_event_quality.csv'), index=False)

    # Summaries
    axes_cols = ['peak_S', 'template_rho', 'spatial_coh', 'baseline_calm', 'S_fwhm_s']
    print("\n=== Per-event quality axes (median [IQR]) ===")
    for c in axes_cols:
        v = df[c].dropna()
        print(f"  {c:16s} median {v.median():.3f}  IQR [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]")

    # Spearman correlation matrix
    X = df[axes_cols].dropna()
    corr_mat = np.zeros((len(axes_cols), len(axes_cols)))
    for i, a in enumerate(axes_cols):
        for j, b in enumerate(axes_cols):
            rho, _ = spearmanr(X[a], X[b])
            corr_mat[i, j] = rho
    print("\n=== Spearman correlation of quality axes ===")
    print(pd.DataFrame(corr_mat, index=axes_cols, columns=axes_cols).round(2))

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Corr heatmap
    ax = axes[0]
    im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(axes_cols))); ax.set_yticks(range(len(axes_cols)))
    ax.set_xticklabels(axes_cols, rotation=45, ha='right')
    ax.set_yticklabels(axes_cols)
    for i in range(len(axes_cols)):
        for j in range(len(axes_cols)):
            ax.text(j, i, f'{corr_mat[i,j]:.2f}', ha='center', va='center',
                    color='k' if abs(corr_mat[i,j]) < 0.5 else 'w', fontsize=9)
    ax.set_title('Spearman ρ, quality axes')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Scatter peak_S vs template_rho
    ax = axes[1]
    ax.scatter(df['peak_S'], df['template_rho'], s=5, alpha=0.3, color='steelblue')
    rho_st, p_st = spearmanr(df['peak_S'].dropna(),
                              df.loc[df['peak_S'].notna(), 'template_rho'])
    ax.set_xlabel('peak_S (composite magnitude)')
    ax.set_ylabel('template_rho (shape fidelity)')
    ax.set_title(f'peak_S vs template_rho · Spearman ρ = {rho_st:.2f}  p = {p_st:.2g}')
    ax.axhline(0, color='k', lw=0.5); ax.grid(alpha=0.3)

    # Grand averages stratified by peak_S quartile
    ax = axes[2]
    q = pd.qcut(df['peak_S'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    colors = ['#4575b4', '#91bfdb', '#fc8d59', '#d73027']
    for qi, lab in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        idx = (q == lab).values
        if idx.sum() == 0:
            continue
        mean_trace = np.nanmean(env_mat[idx], axis=0)
        ax.plot(TGRID, mean_trace, color=colors[qi],
                label=f'{lab} (n={idx.sum()})', lw=2)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('time rel. t0_net (s)')
    ax.set_ylabel('envelope z (peri-event)')
    ax.set_title('Grand-average envelope by peak_S quartile')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(f'B6 — Event quality scoring · {len(df)} events · '
                 f'{df["subject_id"].nunique()} subjects', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'event_quality_overview.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/event_quality_overview.png")
    print(f"Saved: {OUT_DIR}/per_event_quality.csv")


if __name__ == '__main__':
    main()
