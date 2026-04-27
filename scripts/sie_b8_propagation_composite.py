#!/usr/bin/env python3
"""
B8 propagation-sharpening-by-template_ρ under proper per-event nadir alignment.

§25 fixed-nadir already showed no propagation sharpening under composite, but
used the approximate −1.30 s offset rather than per-event find_nadir. This
script redoes the propagation-by-Q4 test with per-event 3-stream nadir
alignment (same as §25b/§26/§27/§28).

For each event:
  1. compute 3 streams (env, R, PLV) via compute_streams
  2. Per-event nadir via find_nadir_3stream (env + R + PLV normalized sum)
  3. Per-channel envelope nadir time in [-3, +0.4] s relative to per-event nadir
  4. Compute per-channel nadir std + spatial-gradient R² + slope_y

Stratify by template_ρ quartile, compare Q1 vs Q4:
  - std across channels (low = simultaneous; high = propagation)
  - R² of spatial regression (high = clean propagation pattern)
  - slope_y (direction: negative means anterior leads)

Cohort-parameterized.

Usage:
    python scripts/sie_b8_propagation_composite.py --cohort lemon
    python scripts/sie_b8_propagation_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
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
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)
from scripts.sie_perionset_triple_average_composite import bandpass, compute_streams
from scripts.sie_mechanism_battery import channel_positions
from scripts.sie_perionset_multistream import (
    F0, HALF_BW, PRE_SEC, POST_SEC, PAD_SEC,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

DIP_WINDOW = (-3.0, 0.4)
NADIR_BASELINE = (-5.0, -3.0)

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


def find_nadir_3stream(t_rel, env_z, R, P):
    base = (t_rel >= NADIR_BASELINE[0]) & (t_rel < NADIR_BASELINE[1])
    srch = (t_rel >= DIP_WINDOW[0]) & (t_rel <= DIP_WINDOW[1])
    if not base.any() or not srch.any():
        return np.nan

    def lz(x):
        mu = np.nanmean(x[base]); sd = np.nanstd(x[base])
        if not np.isfinite(sd) or sd < 1e-9: sd = 1.0
        return (x - mu) / sd

    s = lz(env_z) + lz(R) + lz(P)
    s_m = np.where(srch, s, np.inf)
    idx = int(np.nanargmin(s_m))
    if not np.isfinite(s_m[idx]):
        return np.nan
    return float(t_rel[idx])


def per_channel_nadir_local(X_uV, fs, t0_nadir_rel_t0):
    """Per-channel envelope nadir time relative to per-event nadir.
    t0_nadir_rel_t0 is the nadir time relative to t0_net.
    Window-internal nadir absolute time = PAD_SEC + PRE_SEC + t0_nadir_rel_t0.
    """
    n_ch = X_uV.shape[0]
    Xb = bandpass(X_uV, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(Xb, axis=-1))
    win_smooth = int(round(0.5 * fs))
    kernel = np.ones(win_smooth) / win_smooth
    env = np.array([np.convolve(env[i], kernel, mode='same') for i in range(n_ch)])

    t_grid = np.arange(env.shape[1]) / fs
    # rel-to-nadir: window-absolute time minus (PAD + PRE + nadir_rel_t0)
    rel = t_grid - PAD_SEC - PRE_SEC - t0_nadir_rel_t0
    mask = (rel >= -3.0) & (rel <= 0.4)
    nadir_t = np.full(n_ch, np.nan)
    for i in range(n_ch):
        e = env[i]
        e_m = np.where(mask, e, np.inf)
        idx = int(np.nanargmin(e_m))
        if np.isfinite(e_m[idx]):
            nadir_t[i] = rel[idx]
    return nadir_t


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
    sub_id, df_sub = args
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X_all = raw.get_data() * 1e6
    t_end = raw.times[-1]
    pos = channel_positions(raw.ch_names)

    buckets = {q: {'nadir_std': [], 'r2': [], 'slope_x': [], 'slope_y': []}
                for q in ['Q1', 'Q2', 'Q3', 'Q4']}

    for _, ev in df_sub.iterrows():
        t0 = float(ev['t0_net'])
        q = ev.get('rho_q', None)
        if not isinstance(q, str) or q not in buckets:
            continue
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end:
            continue
        i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue
        # 3-stream nadir
        try:
            (t_env_abs, env_z), (t_c, R), (_, P) = compute_streams(X_seg, fs)
        except Exception:
            continue
        rel_win = t_c - PAD_SEC - PRE_SEC
        rel_env = t_env_abs - PAD_SEC - PRE_SEC
        env_on_win = np.interp(rel_win, rel_env, env_z, left=np.nan, right=np.nan)
        nadir = find_nadir_3stream(rel_win, env_on_win, R, P)
        if not np.isfinite(nadir):
            continue

        # Per-channel nadir times
        try:
            nadir_per_ch = per_channel_nadir_local(X_seg, fs, nadir)
        except Exception:
            continue
        std_ch = float(np.nanstd(nadir_per_ch))
        # Spatial gradient fit
        good = np.isfinite(nadir_per_ch) & np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
        if good.sum() < 6:
            continue
        X_fit = np.column_stack([pos[good, 0], pos[good, 1], np.ones(good.sum())])
        y_fit = nadir_per_ch[good]
        coefs, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
        y_pred = X_fit @ coefs
        ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        buckets[q]['nadir_std'].append(std_ch)
        buckets[q]['r2'].append(float(r2))
        buckets[q]['slope_x'].append(float(coefs[0]))
        buckets[q]['slope_y'].append(float(coefs[1]))

    if not buckets['Q4']['nadir_std']:
        return None

    out = {'subject_id': sub_id}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        out[f'{q}_nadir_std'] = np.array(buckets[q]['nadir_std'])
        out[f'{q}_r2'] = np.array(buckets[q]['r2'])
        out[f'{q}_slope_y'] = np.array(buckets[q]['slope_y'])
        out[f'{q}_n'] = len(buckets[q]['nadir_std'])
    return out


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
    tasks = [(sid, g) for sid, g in qual.groupby('subject_id')]
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}   (per-event find_nadir enabled)")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    print(f"\n=== {args.cohort} composite · B8 propagation-by-Q4 (per-event nadir) ===")
    print(f"(envelope B8: Q1 R²=0.172, Q4 R²=0.174, p=0.41 — NO sharpening)")
    print(f"{'q':<4} {'n_ev':<6} {'nadir_std (median)':<22} {'R² (median)':<14} {'slope_y (median)':<18}")
    q_data = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        stds = np.concatenate([r[f'{q}_nadir_std'] for r in results if len(r[f'{q}_nadir_std'])])
        r2s  = np.concatenate([r[f'{q}_r2']        for r in results if len(r[f'{q}_r2'])])
        slys = np.concatenate([r[f'{q}_slope_y']   for r in results if len(r[f'{q}_slope_y'])])
        q_data[q] = {'stds': stds, 'r2s': r2s, 'slys': slys}
        print(f"{q:<4} {len(stds):<6} "
              f"{np.median(stds):.3f}s (IQR [{np.percentile(stds,25):.3f}, {np.percentile(stds,75):.3f}])   "
              f"{np.median(r2s):.3f}           {np.median(slys):+.3f} s/m")

    # Q1 vs Q4 sharpening tests
    if 'Q1' in q_data and 'Q4' in q_data:
        print(f"\n=== Q1 vs Q4 sharpening tests ===")
        for key, label in [('stds', 'nadir_std'), ('r2s', 'R²'), ('slys', 'slope_y')]:
            q1 = q_data['Q1'][key]; q4 = q_data['Q4'][key]
            u, p = mannwhitneyu(q4, q1)
            if key == 'stds':
                # For std, Q4 should be LOWER if propagation is sharper
                u_dir, p_dir = mannwhitneyu(q4, q1, alternative='less')
                print(f"  {label:12s}: Q1 median {np.median(q1):.3f}  Q4 median {np.median(q4):.3f}  "
                      f"MWU two-sided p={p:.3g}   Q4<Q1 one-sided p={p_dir:.3g}")
            elif key == 'r2s':
                u_dir, p_dir = mannwhitneyu(q4, q1, alternative='greater')
                print(f"  {label:12s}: Q1 median {np.median(q1):.3f}  Q4 median {np.median(q4):.3f}  "
                      f"MWU two-sided p={p:.3g}   Q4>Q1 one-sided p={p_dir:.3g}")
            else:
                print(f"  {label:12s}: Q1 median {np.median(q1):+.3f}  Q4 median {np.median(q4):+.3f}  "
                      f"MWU two-sided p={p:.3g}")

    # Save summary
    rows = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        if q in q_data:
            rows.append({
                'q': q,
                'n_events': int(len(q_data[q]['stds'])),
                'nadir_std_median': float(np.median(q_data[q]['stds'])),
                'r2_median': float(np.median(q_data[q]['r2s'])),
                'slope_y_median': float(np.median(q_data[q]['slys'])),
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'b8_propagation_per_event_nadir.csv'),
                               index=False)
    print(f"\nSaved: {out_dir}/b8_propagation_per_event_nadir.csv")


if __name__ == '__main__':
    main()
