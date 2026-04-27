#!/usr/bin/env python3
"""
A3 re-run on composite v2 detector.

Peri-onset triple average (envelope z, Kuramoto R(t), mean PLV to median)
aligned on composite v2's t0_net (which per A8 is the joint-dip nadir, not the
envelope crossing). Per A7/A8, the peri-onset structure at the nadir-anchored
t0 should show:
  - envelope z dip at t=0 (by detector construction)
  - all three streams peaking at ~+2.5 s (the canonical Q4 rebound)

Cohort-parameterized; reads composite v2 events and the corresponding raw.

Outputs to outputs/schumann/images/perionset/<cohort>_composite/.

Usage:
    python scripts/sie_perionset_triple_average_composite.py --cohort lemon
    python scripts/sie_perionset_triple_average_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)

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

TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC/2, STEP_SEC)

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


def bandpass(x, fs, f1, f2, order=4):
    ny = 0.5 * fs
    b, a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


def compute_streams(X_uV, fs):
    y_mean = X_uV.mean(axis=0)
    yb = bandpass(y_mean, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    z = zscore(env, nan_policy='omit')
    t_env = np.arange(len(z)) / fs

    Xb = bandpass(X_uV, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    ref = np.median(Xb, axis=0)
    ph_ref = np.angle(signal.hilbert(ref))
    dphi = ph - ph_ref[None, :]

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    centers = []
    Rv = []
    Pv = []
    for i in range(0, X_uV.shape[1] - nwin + 1, nstep):
        seg = ph[:, i:i+nwin]
        R_t = np.abs(np.mean(np.exp(1j * seg), axis=0))
        Rv.append(float(np.mean(R_t)))
        pseg = dphi[:, i:i+nwin]
        plv_per_ch = np.abs(np.mean(np.exp(1j * pseg), axis=1))
        Pv.append(float(np.mean(plv_per_ch)))
        centers.append((i + nwin/2) / fs)
    return (t_env, z), (np.array(centers), np.array(Rv)), (np.array(centers), np.array(Pv))


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
        events = pd.read_csv(events_path)
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
    t_end_rec = raw.times[-1]

    env_rows = []
    R_rows = []
    P_rows = []

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
        try:
            (t_env, zenv), (tR, R), (tP, P) = compute_streams(X_seg, fs)
        except Exception:
            continue
        rel_env = t_env - PAD_SEC - PRE_SEC
        rel_R = tR - PAD_SEC - PRE_SEC
        rel_P = tP - PAD_SEC - PRE_SEC
        env_i = np.interp(TGRID, rel_env, zenv, left=np.nan, right=np.nan)
        R_i = np.interp(TGRID, rel_R, R, left=np.nan, right=np.nan)
        P_i = np.interp(TGRID, rel_P, P, left=np.nan, right=np.nan)
        env_rows.append(env_i)
        R_rows.append(R_i)
        P_rows.append(P_i)

    if not env_rows:
        return None

    return {
        'subject_id': sub_id,
        'n_events': len(env_rows),
        'env': np.nanmean(np.array(env_rows), axis=0),
        'R': np.nanmean(np.array(R_rows), axis=0),
        'P': np.nanmean(np.array(P_rows), axis=0),
    }


def bootstrap_ci(subject_means, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n_sub = subject_means.shape[0]
    grand = np.nanmean(subject_means, axis=0)
    boots = np.zeros((n_boot, subject_means.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n_sub, size=n_sub)
        boots[b] = np.nanmean(subject_means[idx], axis=0)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return grand, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'perionset', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")
    n_events_total = sum(r['n_events'] for r in results)
    print(f"Total events aggregated: {n_events_total}")

    env_arr = np.array([r['env'] for r in results])
    R_arr = np.array([r['R'] for r in results])
    P_arr = np.array([r['P'] for r in results])

    env_m, env_lo, env_hi = bootstrap_ci(env_arr)
    R_m, R_lo, R_hi = bootstrap_ci(R_arr)
    P_m, P_lo, P_hi = bootstrap_ci(P_arr)

    df = pd.DataFrame({
        't_rel': TGRID,
        'env_mean': env_m, 'env_ci_lo': env_lo, 'env_ci_hi': env_hi,
        'R_mean': R_m, 'R_ci_lo': R_lo, 'R_ci_hi': R_hi,
        'P_mean': P_m, 'P_ci_lo': P_lo, 'P_ci_hi': P_hi,
    })
    csv_path = os.path.join(out_dir, 'perionset_triple_average.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, arr, m, lo, hi, label, color in [
        (axes[0], env_arr, env_m, env_lo, env_hi, 'envelope z (7.83 ± 0.6 Hz)', 'darkorange'),
        (axes[1], R_arr,   R_m,   R_lo,   R_hi,   'Kuramoto R(t) in 7.2–8.4 Hz', 'seagreen'),
        (axes[2], P_arr,   P_m,   P_lo,   P_hi,   'mean PLV to median',          'purple'),
    ]:
        for i in range(arr.shape[0]):
            ax.plot(TGRID, arr[i], color='gray', alpha=0.08, lw=0.3)
        ax.fill_between(TGRID, lo, hi, color=color, alpha=0.25, label='95% bootstrap CI')
        ax.plot(TGRID, m, color=color, lw=2, label='grand mean')
        ax.axvline(0, color='k', ls='--', lw=0.6)
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
    axes[0].set_title(f'Peri-onset triple average · {args.cohort} composite v2\n'
                      f'{len(results)} subjects, {n_events_total} events, aligned on composite t0_net')
    axes[2].set_xlabel('time relative to t0_net (s)')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'perionset_triple_average.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    i0 = int(np.argmin(np.abs(TGRID)))
    peak_idx_env = int(np.argmax(env_m))
    peak_idx_R = int(np.argmax(R_m))
    peak_idx_P = int(np.argmax(P_m))
    nadir_idx_env = int(np.argmin(env_m))
    nadir_idx_R = int(np.argmin(R_m))
    nadir_idx_P = int(np.argmin(P_m))
    print(f"\n=== {args.cohort} composite · peri-onset peaks & nadirs ===")
    print(f"  envelope z: nadir {env_m[nadir_idx_env]:+.3f} at t={TGRID[nadir_idx_env]:+.2f}s  |  peak {env_m[peak_idx_env]:+.3f} at t={TGRID[peak_idx_env]:+.2f}s")
    print(f"  R(t)      : nadir {R_m[nadir_idx_R]:+.3f} at t={TGRID[nadir_idx_R]:+.2f}s  |  peak {R_m[peak_idx_R]:+.3f} at t={TGRID[peak_idx_R]:+.2f}s")
    print(f"  PLV       : nadir {P_m[nadir_idx_P]:+.3f} at t={TGRID[nadir_idx_P]:+.2f}s  |  peak {P_m[peak_idx_P]:+.3f} at t={TGRID[peak_idx_P]:+.2f}s")
    print(f"\n  At t=0 (composite t0_net):")
    print(f"    env z = {env_m[i0]:+.3f}  CI [{env_lo[i0]:+.3f}, {env_hi[i0]:+.3f}]")
    print(f"    R     = {R_m[i0]:.3f}   CI [{R_lo[i0]:.3f}, {R_hi[i0]:.3f}]")
    print(f"    PLV   = {P_m[i0]:.3f}   CI [{P_lo[i0]:.3f}, {P_hi[i0]:.3f}]")


if __name__ == '__main__':
    main()
