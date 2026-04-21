#!/usr/bin/env python3
"""
B48 — Coherence-first verification (per-event R-vs-envelope onset lag).

Paper's anchor claim: "phase alignment precedes amplitude elevation by 2-3 s."
Tested here at the individual-subject level.

Method. For each Q4 event, extract z-envelope and Kuramoto R(t) in [-10, +5] s
around t0_net. Find onset time for each stream as the first sample in the
search window where the stream exceeds its prior-baseline mean + 2 * baseline
SD (baseline = [-10, -5] s). Lag = env_onset - R_onset. Negative lag =
coherence-first.

Report per-subject median lag (Wilcoxon vs 0), per-quartile stratification
(Q1 vs Q4), and distributions.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import wilcoxon
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import bandpass

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality',
                            'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)

F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)
PRE_SEC = 10.0
POST_SEC = 5.0
STEP_SEC = 0.1
WIN_SEC = 1.0
BASELINE_WIN = (-10.0, -5.0)    # baseline for computing onset threshold
SEARCH_WIN_R = (-8.0, 0.0)      # search window for R onset
SEARCH_WIN_ENV = (-5.0, 2.0)    # search window for env onset
Z_THRESH = 2.0                   # baseline SD multiplier for onset


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-9:
        return x - med
    return (x - med) / mad


def compute_streams(raw):
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))

    Xb = bandpass(X, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    env_vals, Rv, centers = [], [], []
    for i in range(0, X.shape[1] - nwin + 1, nstep):
        env_vals.append(float(np.mean(env[i:i+nwin])))
        seg = ph[:, i:i+nwin]
        Rv.append(float(np.mean(np.abs(np.mean(np.exp(1j * seg), axis=0)))))
        centers.append((i + nwin / 2) / fs)
    t = np.array(centers)
    zE = robust_z(np.array(env_vals))
    zR = robust_z(np.array(Rv))
    return t, zE, zR, raw.times[-1]


def onset_time(t_rel, trace, search_lo, search_hi, baseline_lo, baseline_hi,
                k=Z_THRESH):
    """First sample in [search_lo, search_hi] where trace > baseline_mean +
    k * baseline_sd. Returns NaN if never crossed."""
    base_mask = (t_rel >= baseline_lo) & (t_rel <= baseline_hi)
    if base_mask.sum() < 5:
        return np.nan
    base_mean = np.nanmean(trace[base_mask])
    base_sd = np.nanstd(trace[base_mask])
    if not np.isfinite(base_sd) or base_sd < 1e-6:
        return np.nan
    thr = base_mean + k * base_sd
    search_mask = (t_rel >= search_lo) & (t_rel <= search_hi)
    idx = np.where(search_mask & (trace > thr))[0]
    if len(idx) == 0:
        return np.nan
    return float(t_rel[idx[0]])


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    try:
        qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
    except Exception:
        return None
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4,
                             labels=['Q1','Q2','Q3','Q4'])
    q_sub = qual[qual['subject_id'] == sub_id].copy()
    q_sub['t0_round'] = q_sub['t0_net'].round(3)
    events['t0_round'] = events['t0_net'].round(3)
    events = events.merge(q_sub[['t0_round', 'rho_q']], on='t0_round', how='left')
    if len(events) == 0:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    try:
        t_full, zE_full, zR_full, t_end = compute_streams(raw)
    except Exception:
        return None

    rows = []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        if t0 - PRE_SEC < 2 or t0 + POST_SEC > t_end - 2:
            continue
        sel = (t_full >= t0 - PRE_SEC) & (t_full <= t0 + POST_SEC)
        if sel.sum() < 50:
            continue
        t_rel = t_full[sel] - t0
        zE_seg = zE_full[sel]
        zR_seg = zR_full[sel]
        R_onset = onset_time(t_rel, zR_seg, *SEARCH_WIN_R, *BASELINE_WIN)
        env_onset = onset_time(t_rel, zE_seg, *SEARCH_WIN_ENV, *BASELINE_WIN)
        if np.isnan(R_onset) or np.isnan(env_onset):
            continue
        rows.append({
            'subject_id': sub_id,
            't0_net': t0,
            'rho_q': ev.get('rho_q'),
            'R_onset_s': R_onset,
            'env_onset_s': env_onset,
            'lag_env_minus_R_s': env_onset - R_onset,
        })
    return rows if rows else None


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=2)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects: {len(tasks)}")

    n_workers = int(os.environ.get('SIE_WORKERS',
                                    min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    all_rows = []
    for r in results:
        if r is not None:
            all_rows.extend(r)
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, 'coherence_first_lag.csv')
    df.to_csv(out_csv, index=False)
    print(f"Successful events: {len(df)}  subjects: {df['subject_id'].nunique()}")

    # ===== STATS =====
    def _summ(label, sub):
        if len(sub) == 0:
            return
        # Per-event
        lag_all = sub['lag_env_minus_R_s'].dropna().values
        # Per-subject median
        per_sub = sub.groupby('subject_id')['lag_env_minus_R_s'].median()
        per_sub = per_sub.dropna().values
        if len(per_sub) < 3:
            return
        try:
            w_stat, p_val = wilcoxon(per_sub)
        except ValueError:
            p_val = np.nan
        pct_neg = (per_sub < 0).mean() * 100
        print(f"  {label:15s}  events={len(lag_all):5d}  "
              f"subjects={len(per_sub):3d}  "
              f"per-event median={np.median(lag_all):+.2f}s  "
              f"per-subject median={np.median(per_sub):+.2f}s  "
              f"% sub<0={pct_neg:.0f}%  Wilcoxon p={p_val:.2g}")
        return per_sub

    print(f"\n=== Lag (env onset − R onset), seconds ===")
    print(f"    Negative = coherence-first (R rises before env)")
    all_sub = _summ('All events', df)
    q1_sub = _summ('Q1', df[df['rho_q']=='Q1'])
    q4_sub = _summ('Q4', df[df['rho_q']=='Q4'])

    # ===== FIGURE =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A — per-event lag histogram
    ax = axes[0]
    lag_all = df['lag_env_minus_R_s'].dropna().values
    ax.hist(lag_all, bins=40, color='#2b5fb8', edgecolor='k', lw=0.3,
             alpha=0.75)
    ax.axvline(0, color='k', lw=0.8)
    med = np.median(lag_all)
    ax.axvline(med, color='red', ls='--', lw=1.5,
                label=f'median {med:+.2f}s')
    ax.set_xlabel('env onset − R onset (s)')
    ax.set_ylabel('events')
    ax.set_title(f'A — Per-event lag\n{len(lag_all)} events · '
                  f'median {med:+.2f}s',
                  loc='left', fontweight='bold', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel B — per-subject median lag
    ax = axes[1]
    per_sub = df.groupby('subject_id')['lag_env_minus_R_s'].median().dropna()
    ax.hist(per_sub.values, bins=25, color='#8c1a1a', edgecolor='k', lw=0.3,
             alpha=0.85)
    ax.axvline(0, color='k', lw=0.8)
    med_s = np.median(per_sub)
    ax.axvline(med_s, color='blue', ls='--', lw=1.5,
                label=f'median {med_s:+.2f}s')
    pct_neg = (per_sub < 0).mean() * 100
    ax.set_xlabel('per-subject median lag (s)')
    ax.set_ylabel('subjects')
    try:
        _, p = wilcoxon(per_sub.values)
        p_str = f'p={p:.1g}'
    except ValueError:
        p_str = ''
    ax.set_title(f'B — Per-subject median lag\n'
                  f'n={len(per_sub)} · median {med_s:+.2f}s · '
                  f'{pct_neg:.0f}% negative · {p_str}',
                  loc='left', fontweight='bold', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel C — by quartile
    ax = axes[2]
    colors = ['#4575b4', '#91bfdb', '#fc8d59', '#d73027']
    labels = ['Q1','Q2','Q3','Q4']
    positions = np.arange(4)
    data = []
    for lab in labels:
        g = df[df['rho_q'] == lab]
        ps = g.groupby('subject_id')['lag_env_minus_R_s'].median().dropna()
        data.append(ps.values)
    parts = ax.violinplot(data, positions=positions, showmedians=True,
                           widths=0.7)
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c); pc.set_alpha(0.6); pc.set_edgecolor('k')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{lab}\nn={len(d)}' for lab,d in zip(labels,data)])
    ax.set_ylabel('per-subject median lag (s)')
    ax.set_title('C — By template_ρ quartile',
                  loc='left', fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle('B48 — Coherence-first verification: lag between Kuramoto '
                  'R(t) and envelope z(t) onsets per event (LEMON Q-all)',
                  fontsize=12, y=1.02)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'coherence_first_lag.png')
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
