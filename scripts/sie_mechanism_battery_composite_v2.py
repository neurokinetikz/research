#!/usr/bin/env python3
"""
B5 + B8 + B10 re-run on composite v2 detector WITH per-event find_nadir.

Addresses the methodology caveat in §25: the fixed-nadir-offset approximation
(−1.30 s) introduces ±0.9 s alignment jitter that washes out peri-nadir phase-
jump elevation. This script uses per-event joint-dip nadir detection (3-stream
sum of normalized env z, R, PLV in [−3, +0.4] s window) to align each event to
its own nadir before computing peri-nadir elevation.

Outputs mirror §25; differences in elevation vs §25 isolate the alignment
effect. MSC is intentionally excluded from the nadir-detection sum (per A6b
audit — MSC stream was retracted due to double-bandpass saturation), which
also avoids the expensive per-channel coherence computation.

Cohort-parameterized. Runs B5 (all events), B8 (by quartile), AND B10 (pseudo-
events) in a single pass for efficiency.

Usage:
    python scripts/sie_mechanism_battery_composite_v2.py --cohort lemon
    python scripts/sie_mechanism_battery_composite_v2.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)
from scripts.sie_mechanism_battery import phase_jumps
from scripts.sie_propagation_null_random import sample_pseudo_times
from scripts.sie_perionset_triple_average_composite import (
    compute_streams, PRE_SEC, POST_SEC, PAD_SEC,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

DIP_WINDOW = (-3.0, 0.4)
NADIR_BASELINE = (-5.0, -3.0)   # z-scoring baseline for nadir detection
PJ_TGRID = np.arange(-8.0, 8.0 + 0.05, 0.1)

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    qual = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events, qual
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events, qual
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events, qual
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events, qual
    if cohort == 'srm':
        return load_srm, {}, events, qual
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events, qual
    if cohort == 'dortmund':
        return load_dortmund, {}, events, qual
    if cohort == 'chbmp':
        return load_chbmp, {}, events, qual
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events, qual
    raise ValueError(f"unsupported cohort {cohort!r}")


def find_joint_nadir_3stream(t_rel_env, env_z, t_rel_win, R, P):
    """Find joint-dip nadir in DIP_WINDOW using normalized env+R+PLV sum.
    env_z is at sample rate; R/P at 100-ms step on t_rel_win.
    Interpolate env onto t_rel_win and compute sum of baseline-z-scored streams.
    Returns nadir time (s, rel to t0_net), or nan.
    """
    env_interp = np.interp(t_rel_win, t_rel_env, env_z, left=np.nan, right=np.nan)
    base_mask = (t_rel_win >= NADIR_BASELINE[0]) & (t_rel_win < NADIR_BASELINE[1])
    srch_mask = (t_rel_win >= DIP_WINDOW[0]) & (t_rel_win <= DIP_WINDOW[1])
    if not base_mask.any() or not srch_mask.any():
        return np.nan

    def lz(x):
        mu = np.nanmean(x[base_mask]); sd = np.nanstd(x[base_mask])
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        return (x - mu) / sd

    s = lz(env_interp) + lz(R) + lz(P)
    s_srch = np.where(srch_mask, s, np.inf)
    idx = int(np.nanargmin(s_srch))
    if not np.isfinite(s_srch[idx]):
        return np.nan
    return float(t_rel_win[idx])


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


def per_event_phase_jump(X_seg, fs, t_window_center_rel_t0, is_pseudo=False):
    """Return (nadir_rel_t0, pj_interp) for one event window.

    X_seg is the [-12, +12] s segment around t0 (or pseudo-t0).
    t_window_center_rel_t0 helps convert absolute window times to t0-relative.
    Returns (nadir_time_rel_t0 or nan if not found, pj_trace on PJ_TGRID).
    """
    try:
        (t_env_abs, env_z), (t_c, R), (_, P) = compute_streams(X_seg, fs)
    except Exception:
        return np.nan, None
    # Convert window-internal times to t0-relative
    rel_env = t_env_abs - PAD_SEC - PRE_SEC
    rel_win = t_c - PAD_SEC - PRE_SEC
    nadir_rel_t0 = find_joint_nadir_3stream(rel_env, env_z, rel_win, R, P)
    if not np.isfinite(nadir_rel_t0):
        return np.nan, None
    try:
        t_pj_abs, pj = phase_jumps(X_seg, fs)
        rel_pj = t_pj_abs - PAD_SEC - PRE_SEC - nadir_rel_t0
        pj_interp = np.interp(PJ_TGRID, rel_pj, pj, left=np.nan, right=np.nan)
    except Exception:
        return nadir_rel_t0, None
    return nadir_rel_t0, pj_interp


def process_subject(args):
    sub_id, df_sub, ec_events = args
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X_all = raw.get_data() * 1e6
    t_end = raw.times[-1]

    # Real-event traces (all events and per-quartile)
    pj_all, pj_q = [], {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    nadir_rel_t0_list = []

    for _, ev in df_sub.iterrows():
        t0 = float(ev['t0_net'])
        q = ev.get('rho_q', None)
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end:
            continue
        i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue
        nadir_rel_t0, pj = per_event_phase_jump(X_seg, fs, t0)
        if pj is None:
            continue
        pj_all.append(pj)
        nadir_rel_t0_list.append(nadir_rel_t0)
        if isinstance(q, str) and q in pj_q:
            pj_q[q].append(pj)

    # Pseudo-events for B10
    t_events = ec_events
    n_target = len(ec_events)
    pj_pseudo = []
    if n_target > 0:
        t_pseudo = sample_pseudo_times(t_events, n_target, t_end,
                                        seed=abs(hash(sub_id)) % (2**31))
        for t0 in t_pseudo:
            lo = t0 - PRE_SEC - PAD_SEC
            hi = t0 + POST_SEC + PAD_SEC
            if lo < 0 or hi > t_end:
                continue
            i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
            X_seg = X_all[:, i0:i1]
            if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
                continue
            _, pj = per_event_phase_jump(X_seg, fs, t0, is_pseudo=True)
            if pj is None:
                continue
            pj_pseudo.append(pj)

    if not pj_all:
        return None

    out = {
        'subject_id': sub_id,
        'n_events_real': len(pj_all),
        'n_events_pseudo': len(pj_pseudo),
        'pj_all_mean': np.nanmean(np.array(pj_all), axis=0),
        'pj_pseudo_mean': np.nanmean(np.array(pj_pseudo), axis=0) if pj_pseudo else None,
        'nadir_times': np.array(nadir_rel_t0_list),
    }
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        if pj_q[q]:
            out[f'pj_{q}_mean'] = np.nanmean(np.array(pj_q[q]), axis=0)
            out[f'pj_{q}_n'] = len(pj_q[q])
        else:
            out[f'pj_{q}_mean'] = None
            out[f'pj_{q}_n'] = 0
    return out


def bootstrap_ci(mat, n_boot=1000, seed=0):
    mat = np.array(mat)
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


def elevation(traj):
    peri = (PJ_TGRID >= -1) & (PJ_TGRID <= +1)
    base = (PJ_TGRID >= -8) & (PJ_TGRID <= -4)
    m_peri = np.nanmean(traj[peri]); m_base = np.nanmean(traj[base])
    return (m_peri / m_base) if m_base > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'mechanism_battery', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Build per-subject task: (sub_id, events df, event times array for pseudo sampling)
    tasks = []
    for sid, g in qual.groupby('subject_id'):
        ec_events = g['t0_net'].astype(float).values
        tasks.append((sid, g, ec_events))
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}   (per-event find_nadir enabled)")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")
    n_real = sum(r['n_events_real'] for r in results)
    n_pseudo = sum(r['n_events_pseudo'] for r in results)
    print(f"Total events: real={n_real}  pseudo={n_pseudo}")

    # Per-event nadir distribution diagnostic
    all_nadirs = np.concatenate([r['nadir_times'] for r in results])
    print(f"\n=== {args.cohort} composite · per-event nadir distribution ===")
    print(f"  median {np.median(all_nadirs):.3f} s  std {np.std(all_nadirs):.3f} s")
    print(f"  IQR [{np.percentile(all_nadirs, 25):.3f}, {np.percentile(all_nadirs, 75):.3f}]")
    print(f"  (fixed-offset in §25 used −1.30 s; per-event std was ~0.9 s)")

    pj_all_mat = np.array([r['pj_all_mean'] for r in results])
    pj_pseudo_mat = np.array([r['pj_pseudo_mean'] for r in results
                                if r['pj_pseudo_mean'] is not None])

    grand_all, _, _ = bootstrap_ci(pj_all_mat)
    grand_pseudo, _, _ = bootstrap_ci(pj_pseudo_mat) if len(pj_pseudo_mat) > 1 else (np.nan * PJ_TGRID, None, None)

    elev_all = elevation(grand_all)
    elev_pseudo = elevation(grand_pseudo) if np.any(np.isfinite(grand_pseudo)) else np.nan

    per_sub_all = np.array([elevation(r['pj_all_mean']) for r in results])
    per_sub_all = per_sub_all[np.isfinite(per_sub_all)]
    per_sub_pseudo = np.array([elevation(r['pj_pseudo_mean']) for r in results
                                if r['pj_pseudo_mean'] is not None])
    per_sub_pseudo = per_sub_pseudo[np.isfinite(per_sub_pseudo)]

    print(f"\n=== {args.cohort} composite · B5 phase-reset WITH per-event nadir ===")
    print(f"  Grand-mean elevation (all events): {elev_all:.3f}×")
    print(f"  Per-subject median: {np.median(per_sub_all):.3f}   IQR [{np.percentile(per_sub_all,25):.3f}, {np.percentile(per_sub_all,75):.3f}]")
    print(f"  Subjects > 1.2×: {(per_sub_all > 1.2).mean()*100:.1f}%")
    print(f"  (envelope B5: 1.61× at nadir with per-event find_nadir)")

    print(f"\n=== {args.cohort} composite · B8 phase-reset by quartile ===")
    q_grand = {}
    for q in ['Q1', 'Q4']:
        mat = np.array([r[f'pj_{q}_mean'] for r in results
                        if r[f'pj_{q}_mean'] is not None])
        if len(mat) == 0:
            continue
        grand_q, _, _ = bootstrap_ci(mat)
        elev_q = elevation(grand_q)
        q_grand[q] = (elev_q, grand_q, mat)
        n_events_q = sum(r[f'pj_{q}_n'] for r in results)
        per_sub_q = np.array([elevation(r[f'pj_{q}_mean']) for r in results
                              if r[f'pj_{q}_mean'] is not None])
        per_sub_q = per_sub_q[np.isfinite(per_sub_q)]
        print(f"  {q}: grand {elev_q:.3f}×   per-sub median {np.median(per_sub_q):.3f}  "
              f"n_sub={len(per_sub_q)}  n_events={n_events_q}")
    print(f"  (envelope B8: Q1 1.34×, Q4 1.49×)")

    print(f"\n=== {args.cohort} composite · B10 phase-reset pseudo-event null ===")
    print(f"  Pseudo grand-mean elevation: {elev_pseudo:.3f}×")
    print(f"  Per-subject pseudo median: {np.median(per_sub_pseudo):.3f}")
    print(f"  (envelope B10: pseudo grand 1.15×, real Q4 1.49×, effective +0.3× above null)")
    if 'Q4' in q_grand:
        print(f"  Real Q4 − Pseudo: {q_grand['Q4'][0] - elev_pseudo:+.3f}")

    # Save outputs
    pd.DataFrame({
        't_rel_nadir': PJ_TGRID,
        'pj_all_grand': grand_all,
        'pj_Q1_grand': q_grand['Q1'][1] if 'Q1' in q_grand else np.nan,
        'pj_Q4_grand': q_grand['Q4'][1] if 'Q4' in q_grand else np.nan,
        'pj_pseudo_grand': grand_pseudo,
    }).to_csv(os.path.join(out_dir, 'mechanism_v2_grand_traces.csv'), index=False)

    pd.DataFrame([{
        'metric': 'B5_all_events_grand_elevation', 'value': float(elev_all),
    }, {
        'metric': 'B5_all_events_per_subject_median', 'value': float(np.median(per_sub_all)),
    }, {
        'metric': 'B8_Q1_grand_elevation', 'value': float(q_grand['Q1'][0]) if 'Q1' in q_grand else np.nan,
    }, {
        'metric': 'B8_Q4_grand_elevation', 'value': float(q_grand['Q4'][0]) if 'Q4' in q_grand else np.nan,
    }, {
        'metric': 'B10_pseudo_grand_elevation', 'value': float(elev_pseudo),
    }, {
        'metric': 'nadir_per_event_median', 'value': float(np.median(all_nadirs)),
    }, {
        'metric': 'nadir_per_event_std', 'value': float(np.std(all_nadirs)),
    }]).to_csv(os.path.join(out_dir, 'mechanism_v2_summary.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(PJ_TGRID, grand_all, color='purple', lw=2,
            label=f'all events (elev {elev_all:.2f}×)')
    if 'Q4' in q_grand:
        ax.plot(PJ_TGRID, q_grand['Q4'][1], color='#d73027', lw=1.5, ls='--',
                label=f'Q4 (elev {q_grand["Q4"][0]:.2f}×)')
    if 'Q1' in q_grand:
        ax.plot(PJ_TGRID, q_grand['Q1'][1], color='#4575b4', lw=1.5, ls='--',
                label=f'Q1 (elev {q_grand["Q1"][0]:.2f}×)')
    ax.plot(PJ_TGRID, grand_pseudo, color='gray', lw=1.5,
            label=f'pseudo (elev {elev_pseudo:.2f}×)')
    ax.axvline(0, color='k', ls='--', lw=0.5, label='nadir')
    ax.axvspan(-8, -4, alpha=0.08, color='blue')
    ax.axvspan(-1, +1, alpha=0.08, color='red')
    ax.set_xlabel('time rel nadir (s)')
    ax.set_ylabel('phase-jump rate (counts/100ms)')
    ax.set_title(f'B5/B8/B10 (per-event nadir) · {args.cohort} composite')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(all_nadirs, bins=40, color='purple', edgecolor='k', lw=0.3, alpha=0.75)
    ax.axvline(np.median(all_nadirs), color='red', lw=1.5,
                label=f"median {np.median(all_nadirs):+.2f}")
    ax.axvline(-1.30, color='gray', ls='--', lw=1,
                label='§25 fixed offset (−1.30)')
    ax.set_xlabel('per-event nadir time (rel t0_net, s)')
    ax.set_ylabel('events')
    ax.set_title(f'Per-event nadir distribution · std {np.std(all_nadirs):.2f} s')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.hist(per_sub_all, bins=30, color='purple', edgecolor='k', lw=0.3, alpha=0.7,
            label=f'real (median {np.median(per_sub_all):.2f})')
    if len(per_sub_pseudo) > 1:
        ax.hist(per_sub_pseudo, bins=30, color='gray', edgecolor='k', lw=0.3, alpha=0.6,
                label=f'pseudo (median {np.median(per_sub_pseudo):.2f})')
    ax.axvline(1.0, color='k', lw=0.8)
    ax.axvline(1.49, color='#d73027', ls=':', lw=1, label='envelope Q4 1.49×')
    ax.set_xlabel('per-subject peri-nadir elevation')
    ax.set_ylabel('subjects')
    ax.set_title('Per-subject elevation · real vs pseudo')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(f'B5+B8+B10 WITH per-event find_nadir · {args.cohort} composite v2',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mechanism_v2_battery.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/mechanism_v2_battery.png")


if __name__ == '__main__':
    main()
