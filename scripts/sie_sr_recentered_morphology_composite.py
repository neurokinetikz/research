#!/usr/bin/env python3
"""
B15 part 1 on composite v2 — t0_sr-centered morphology vs t0_net-centered.

Envelope B15 finding: re-centering events at t0_sr (SR-band peak time)
degrades dip-rebound morphology — Q4 env z range drops 3.22 → 2.43.
Interpretation: t0_net is the correct alignment anchor for canonical-event
morphology; t0_sr jitter smears the grand average.

This script runs peri-event grand averages of env z / R / PLV aligned to
t0_sr (from §31 per_event_t0_shift.csv) and compares Q4 dip-rebound range
to §30's t0_net-aligned Q4 result.

Per composite event:
  1. Look up t0_sr from §31 per_event_t0_shift.csv
  2. Extract env z / R / PLV in [t0_sr − 10, t0_sr + 10] s window
  3. Grand average per template_ρ quartile
  4. Measure dip-rebound range per quartile
  5. Compare to §30 (t0_net-aligned) output

Cohort-parameterized.

Usage:
    python scripts/sie_sr_recentered_morphology_composite.py --cohort lemon
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)
from scripts.sie_perionset_triple_average_composite import compute_streams

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

PRE_SEC, POST_SEC, PAD_SEC, STEP_SEC = 10.0, 10.0, 2.0, 0.1
TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC / 2, STEP_SEC)

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    qual = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    shift = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'psd_timelapse',
                         f'{cohort}_composite', 'per_event_t0_shift.csv')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events, qual, shift
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events, qual, shift
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events, qual, shift
    if cohort == 'srm':
        return load_srm, {}, events, qual, shift
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events, qual, shift
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events, qual, shift
    if cohort == 'dortmund':
        return load_dortmund, {}, events, qual, shift
    if cohort == 'chbmp':
        return load_chbmp, {}, events, qual, shift
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events, qual, shift
    raise ValueError(f"unsupported cohort {cohort!r}")


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

    out = {q: {'env': [], 'R': [], 'P': []} for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    for _, ev in df_sub.iterrows():
        t0_sr = float(ev['t0_sr'])  # re-centered anchor
        q = ev['rho_q']
        if q not in out:
            continue
        lo = t0_sr - PRE_SEC - PAD_SEC
        hi = t0_sr + POST_SEC + PAD_SEC
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
        rel_env = t_env - PAD_SEC - PRE_SEC  # rel to t0_sr
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
    # Same window as §30 B7: nadir in [-2, +2] s, rebound in [+2, +6] s
    dip_win = (t >= -2) & (t <= +2)
    reb_win = (t >= +2) & (t <= +6)
    if not np.any(np.isfinite(trace[dip_win])) or not np.any(np.isfinite(trace[reb_win])):
        return np.nan, np.nan, np.nan, np.nan
    dip_idx = np.nanargmin(trace[dip_win])
    reb_idx = np.nanargmax(trace[reb_win])
    nadir_t = t[dip_win][dip_idx]
    reb_t = t[reb_win][reb_idx]
    return (float(trace[dip_win][dip_idx]), float(nadir_t),
            float(trace[reb_win][reb_idx]), float(reb_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir, quality_csv, shift_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'quality', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    # Load t0_sr per event + template_ρ
    shifts = pd.read_csv(shift_csv)
    qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    shifts['key'] = shifts.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
    qual['key'] = qual.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
    merged = shifts.merge(qual[['key', 'rho_q']], on='key', how='left').dropna(subset=['rho_q'])
    print(f"Cohort: {args.cohort} composite · events with t0_sr + rho_q: {len(merged)}")

    tasks = [(sid, g) for sid, g in merged.groupby('subject_id')]
    print(f"Processing {len(tasks)} subjects  workers: {args.workers}")
    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    def stack(key):
        arr = []
        for r in results:
            v = r[key]
            arr.append(v if v is not None else np.full(len(TGRID), np.nan))
        return np.array(arr)

    print(f"\n=== {args.cohort} composite · t0_sr-centered morphology ===")
    print(f"(envelope B15: t0_sr re-centering WORSE — Q4 range 3.22 → 2.43)")
    print(f"(§30 t0_net-centered Q4 env z range: EC 1.249, EO 1.210)")

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
                    label=f'{q} n_sub={n_sub} n_ev={total_events}')
            ax.fill_between(TGRID, lo, hi, color=colors[q], alpha=0.2)
            nd, nt, rp, rt = nadir_and_rebound_stats(grand)
            stats_rows.append({
                'stream': s, 'quartile': q, 'n_sub': n_sub, 'n_events': total_events,
                'nadir_depth': nd, 'nadir_time_s': nt,
                'rebound_peak': rp, 'rebound_time_s': rt,
                'dip_rebound_range': rp - nd if np.isfinite(rp) and np.isfinite(nd) else np.nan,
            })
            print(f"  {label:12s} {q}: nadir {nd:+.3f} at {nt:+.2f}s  rebound {rp:+.3f} at {rt:+.2f}s  range {rp-nd:+.3f}")
        ax.axvline(0, color='k', ls='--', lw=0.5)
        ax.set_xlabel('time rel. t0_sr (s)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} · Q1 vs Q4 (t0_sr aligned)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-10, 10)

    plt.suptitle(f'B15 part 1 · t0_sr-centered morphology · {args.cohort} composite v2', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'b15_t0sr_centered_morphology.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(out_dir, 'b15_t0sr_centered_morphology.csv'), index=False)
    print(f"\nSaved: {out_dir}/b15_t0sr_centered_morphology.png")


if __name__ == '__main__':
    main()
