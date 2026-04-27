#!/usr/bin/env python3
"""
A9 re-run on composite v2 detector.

9-stream peri-onset characterization (nadir-aligned):
  env_z, R, PLV, MSC, wPLI, IFt, IFs, xPLV13, slope

Uses per-event 3-stream joint-dip nadir detection (env + R + PLV, same as §25b)
to align each composite event before interpolating onto a common TGRID.

Cohort-parameterized.

Note: slow due to per-channel MSC coherence + per-channel IF gradients + Welch
slope + cross-harmonic PLV per window. Expect ~2 min/subject at 4 workers on
LEMON.

Usage:
    python scripts/sie_perionset_multistream_composite.py --cohort lemon
    python scripts/sie_perionset_multistream_composite.py --cohort lemon_EO
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
from scripts.sie_perionset_triple_average_composite import bandpass
from scripts.sie_perionset_multistream import (
    F0, HALF_BW, R_BAND, SR3_BAND, BROADBAND, ALPHA_EXCLUDE,
    STEP_SEC, WIN_SEC, SLOPE_WIN_SEC, SLOPE_STEP_SEC,
    PRE_SEC, POST_SEC, PAD_SEC, TGRID, DIP_WINDOW,
    inst_freq_channel, wpli_to_ref_window, cross_harmonic_plv_window,
    slope_via_loglog, compute_all_streams, STREAMS,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

NADIR_BASELINE = (-5.0, -3.0)

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


def find_nadir_3stream(t_rel, env, R, P):
    """Joint-dip nadir using normalized sum of env/R/PLV (no MSC).
    Matches §25b nadir detection logic."""
    base = (t_rel >= NADIR_BASELINE[0]) & (t_rel < NADIR_BASELINE[1])
    srch = (t_rel >= DIP_WINDOW[0]) & (t_rel <= DIP_WINDOW[1])
    if not base.any() or not srch.any():
        return np.nan

    def lz(x):
        mu = np.nanmean(x[base]); sd = np.nanstd(x[base])
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        return (x - mu) / sd

    s = lz(env) + lz(R) + lz(P)
    s_m = np.where(srch, s, np.inf)
    idx = int(np.nanargmin(s_m))
    if not np.isfinite(s_m[idx]):
        return np.nan
    return float(t_rel[idx])


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
    """args = (sub_id, events_path) or (sub_id, events_path, weights)."""
    if len(args) == 3:
        sub_id, events_path, weights = args
    else:
        sub_id, events_path = args
        weights = None
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
    t_end_rec = raw.times[-1]

    event_rows = {s: [] for s in STREAMS}
    event_weights = []

    for k, (_, ev) in enumerate(events.iterrows()):
        t0 = float(ev['t0_net'])
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
            S = compute_all_streams(X_seg, fs)
        except Exception:
            continue
        rel = S['centers'] - PAD_SEC - PRE_SEC
        slope_rel = S['slope_centers'] - PAD_SEC - PRE_SEC

        nadir = find_nadir_3stream(rel, S['env'], S['R'], S['PLV'])
        if not np.isfinite(nadir):
            continue
        rel_to_nadir = rel - nadir
        slope_rel_to_nadir = slope_rel - nadir

        for s in STREAMS[:-1]:
            arr = np.interp(TGRID, rel_to_nadir, S[s], left=np.nan, right=np.nan)
            event_rows[s].append(arr)
        arr = np.interp(TGRID, slope_rel_to_nadir, S['slope'], left=np.nan, right=np.nan)
        event_rows['slope'].append(arr)
        if weights is not None and k < len(weights):
            event_weights.append(max(float(weights[k]), 0.0))
        else:
            event_weights.append(1.0)

    if not event_rows['env']:
        return None

    out = {'subject_id': sub_id, 'n_events': len(event_rows['env'])}
    w_arr = np.array(event_weights)
    if w_arr.sum() <= 0:
        return None
    for s in STREAMS:
        ev_arr = np.array(event_rows[s])
        out[s] = np.average(ev_arr, axis=0, weights=w_arr)
    return out


def bootstrap_ci(mat, n_boot=500, seed=0):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    ap.add_argument('--shape-weighted', action='store_true',
                    help='Soft-weight per-event peri-event traces by '
                         'max(template_rho, 0); requires per_event_quality CSV')
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'multistream', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    weights_by_subj = None
    if args.shape_weighted:
        quality_csv = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                                    'quality',
                                    f'per_event_quality_{args.cohort}_composite.csv')
        if not os.path.isfile(quality_csv):
            print(f"Shape-weighted: quality CSV not found: {quality_csv}")
            return
        qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
        qual['t0r'] = qual['t0_net'].round(3)
        weights_by_subj = {}
        for sid, g in qual.groupby('subject_id'):
            weights_by_subj[sid] = dict(zip(g['t0r'], g['template_rho']))
        print(f"Shape-weighted: loaded template_rho for "
              f"{len(weights_by_subj)} subjects, {len(qual)} events")

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if not os.path.isfile(ep):
            continue
        if args.shape_weighted:
            try:
                ev_df = pd.read_csv(ep).dropna(subset=['t0_net'])
            except Exception:
                continue
            sub_w = weights_by_subj.get(r['subject_id'], {})
            t0r = ev_df['t0_net'].round(3).values
            w_arr = np.array([sub_w.get(t, np.nan) for t in t0r])
            if not np.any(np.isfinite(w_arr)):
                continue
            w_arr = np.where(np.isfinite(w_arr), w_arr, 0.0)
            tasks.append((r['subject_id'], ep, list(w_arr)))
        else:
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} composite{' shape-weighted' if args.shape_weighted else ''}"
          f" · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    n_events = sum(r['n_events'] for r in results)
    print(f"Successful subjects: {len(results)}")
    print(f"Total events: {n_events}")

    gm = {}
    print(f"\n=== {args.cohort} composite · A9 multi-stream peaks & troughs ===")
    for s in STREAMS:
        arr = np.array([r[s] for r in results])
        m, lo, hi = bootstrap_ci(arr)
        gm[s] = {'arr': arr, 'mean': m, 'lo': lo, 'hi': hi}
        peak_t = TGRID[np.argmax(m)]; peak_v = m[np.argmax(m)]
        trough_t = TGRID[np.argmin(m)]; trough_v = m[np.argmin(m)]
        # value at t=0 (nadir)
        i0 = int(np.argmin(np.abs(TGRID)))
        print(f"  {s:8s}: peak {peak_v:+.4f} at {peak_t:+.2f}s   "
              f"trough {trough_v:+.4f} at {trough_t:+.2f}s   "
              f"@nadir {m[i0]:+.4f}")

    out_csv = {'t_rel': TGRID}
    for s in STREAMS:
        out_csv[f'{s}_mean'] = gm[s]['mean']
        out_csv[f'{s}_ci_lo'] = gm[s]['lo']
        out_csv[f'{s}_ci_hi'] = gm[s]['hi']
    _tag = "_sw" if args.shape_weighted else ""
    pd.DataFrame(out_csv).to_csv(
        os.path.join(out_dir, f'multistream_nadir_aligned{_tag}.csv'), index=False)

    labels = {
        'env': 'envelope z (7.83 Hz)',
        'R': 'Kuramoto R',
        'PLV': 'mean PLV to median',
        'MSC': 'mean MSC at F0',
        'wPLI': 'mean wPLI to median',
        'IFt': 'IF temporal dispersion (Hz)',
        'IFs': 'IF spatial dispersion (Hz)',
        'xPLV13': 'cross-harmonic PLV sr1↔sr3',
        'slope': 'aperiodic slope (log-log)',
    }
    colors = {
        'env': 'darkorange', 'R': 'seagreen', 'PLV': 'purple', 'MSC': 'steelblue',
        'wPLI': 'teal', 'IFt': 'brown', 'IFs': 'olive', 'xPLV13': 'crimson',
        'slope': 'slategray',
    }
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    for idx, s in enumerate(STREAMS):
        ax = axes.flat[idx]
        ax.fill_between(TGRID, gm[s]['lo'], gm[s]['hi'], color=colors[s], alpha=0.25)
        ax.plot(TGRID, gm[s]['mean'], color=colors[s], lw=2)
        ax.axvline(0, color='k', ls='--', lw=0.6, alpha=0.7)
        peak_t = TGRID[np.argmax(gm[s]['mean'])]
        trough_t = TGRID[np.argmin(gm[s]['mean'])]
        ax.axvline(peak_t, color='red', ls=':', lw=0.5, alpha=0.6)
        ax.set_title(f'{labels[s]}\npeak {peak_t:+.2f}s  trough {trough_t:+.2f}s',
                     fontsize=10)
        ax.set_ylabel(s)
    for ax in axes[-1]:
        ax.set_xlabel('time relative to nadir (s)')
    fig.suptitle(f'A9 · multi-stream peri-onset · {args.cohort} composite v2\n'
                 f'{len(results)} subjects · {n_events} events',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'multistream_nadir_aligned.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/multistream_nadir_aligned.png")


if __name__ == '__main__':
    main()
