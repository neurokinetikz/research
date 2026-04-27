#!/usr/bin/env python3
"""
A10 Part B re-run on composite v2 detector.

Peri-nadir wPLI and imaginary coherence split by channel-pair scalp distance
(near < median vs far >= median). Tests whether composite events replicate
envelope A10 Part B's "pre-nadir coupling is long-range, post-nadir short-range"
signature, or whether the §26 wPLI post-nadir inversion is driven by near- and
far-pairs equally (composite selects only synchronous events).

Per-event 3-stream nadir detection (env + R + PLV, same as §25b/§26). 50
near-pair + 50 far-pair random samples per subject for speed.

Cohort-parameterized.

Usage:
    python scripts/sie_wpli_near_far_composite.py --cohort lemon
    python scripts/sie_wpli_near_far_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import sys
import random
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
from scripts.sie_perionset_triple_average_composite import bandpass, compute_streams
from scripts.sie_perionset_multistream import (
    R_BAND, PRE_SEC, POST_SEC, PAD_SEC, STEP_SEC, WIN_SEC, TGRID, F0,
)
from scripts.sie_wpli_deepdive_and_if_mean import (
    pair_distances, wpli_window, icoh_window,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

DIP_WINDOW = (-3.0, 0.4)
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


def find_nadir_3stream(t_rel, env_z, R, P):
    """Joint-dip nadir using normalized env + R + PLV sum."""
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
    t_end_rec = raw.times[-1]
    ch_names = raw.ch_names

    D = pair_distances(ch_names)
    n_ch = len(ch_names)
    triu_i, triu_j = np.triu_indices(n_ch, k=1)
    pair_D = D[triu_i, triu_j]
    good_pairs = np.isfinite(pair_D)
    if good_pairs.sum() < 10:
        return None
    pair_D_g = pair_D[good_pairs]
    d_med = np.nanmedian(pair_D_g)
    near_mask = (pair_D < d_med) & good_pairs
    far_mask  = (pair_D >= d_med) & good_pairs
    near_pairs = list(zip(triu_i[near_mask], triu_j[near_mask]))
    far_pairs  = list(zip(triu_i[far_mask], triu_j[far_mask]))
    rng = random.Random(abs(hash(sub_id)) % (2**31))
    if len(near_pairs) > 50: near_pairs = rng.sample(near_pairs, 50)
    if len(far_pairs)  > 50: far_pairs  = rng.sample(far_pairs,  50)

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))

    wpli_near_rows, wpli_far_rows = [], []
    icoh_near_rows, icoh_far_rows = [], []

    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end_rec:
            continue
        i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue

        # Phase & bandpass for wPLI/ICoh
        try:
            Xb = bandpass(X_seg, fs, R_BAND[0], R_BAND[1])
            ph = np.angle(signal.hilbert(Xb, axis=-1))
        except Exception:
            continue

        # Nadir detection via 3-stream on same segment
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

        # Sliding windows for wPLI and ICoh
        centers = []
        wpli_near_win, wpli_far_win = [], []
        icoh_near_win, icoh_far_win = [], []
        for i in range(0, X_seg.shape[1] - nwin + 1, nstep):
            wpli_n = [wpli_window(ph[a, i:i+nwin], ph[b, i:i+nwin])
                      for a, b in near_pairs]
            wpli_f = [wpli_window(ph[a, i:i+nwin], ph[b, i:i+nwin])
                      for a, b in far_pairs]
            wpli_near_win.append(float(np.mean(wpli_n)) if wpli_n else np.nan)
            wpli_far_win.append(float(np.mean(wpli_f)) if wpli_f else np.nan)
            icoh_n = [icoh_window(Xb[a, i:i+nwin], Xb[b, i:i+nwin], fs)
                      for a, b in near_pairs]
            icoh_f = [icoh_window(Xb[a, i:i+nwin], Xb[b, i:i+nwin], fs)
                      for a, b in far_pairs]
            icoh_near_win.append(float(np.nanmean(icoh_n)) if icoh_n else np.nan)
            icoh_far_win.append(float(np.nanmean(icoh_f)) if icoh_f else np.nan)
            centers.append((i + nwin/2) / fs)

        centers = np.array(centers)
        rel_my = centers - PAD_SEC - PRE_SEC - nadir
        wpli_near_rows.append(np.interp(TGRID, rel_my, wpli_near_win, left=np.nan, right=np.nan))
        wpli_far_rows.append(np.interp(TGRID, rel_my, wpli_far_win, left=np.nan, right=np.nan))
        icoh_near_rows.append(np.interp(TGRID, rel_my, icoh_near_win, left=np.nan, right=np.nan))
        icoh_far_rows.append(np.interp(TGRID, rel_my, icoh_far_win, left=np.nan, right=np.nan))

    if not wpli_near_rows:
        return None

    return {
        'subject_id': sub_id,
        'n_events': len(wpli_near_rows),
        'median_pair_dist': float(d_med),
        'wPLI_near': np.nanmean(np.array(wpli_near_rows), axis=0),
        'wPLI_far':  np.nanmean(np.array(wpli_far_rows),  axis=0),
        'ICoh_near': np.nanmean(np.array(icoh_near_rows), axis=0),
        'ICoh_far':  np.nanmean(np.array(icoh_far_rows),  axis=0),
    }


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
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'multistream', f'{args.cohort}_composite')
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
    print(f"Successful: {len(results)}")
    n_events = sum(r['n_events'] for r in results)
    print(f"Total events: {n_events}")

    # Grand means + CI
    streams = {
        'wPLI_near': np.array([r['wPLI_near'] for r in results]),
        'wPLI_far':  np.array([r['wPLI_far']  for r in results]),
        'ICoh_near': np.array([r['ICoh_near'] for r in results]),
        'ICoh_far':  np.array([r['ICoh_far']  for r in results]),
    }
    gm = {}
    for k, arr in streams.items():
        m, lo, hi = bootstrap_ci(arr)
        gm[k] = (m, lo, hi)

    print(f"\n=== {args.cohort} composite · A10 Part B peri-nadir peaks ===")
    for k in ['wPLI_near', 'wPLI_far', 'ICoh_near', 'ICoh_far']:
        m, _, _ = gm[k]
        pk = int(np.nanargmax(m))
        tr = int(np.nanargmin(m))
        i0 = int(np.argmin(np.abs(TGRID)))
        print(f"  {k:10s}: peak {m[pk]:+.4f} at {TGRID[pk]:+.2f}s   "
              f"trough {m[tr]:+.4f} at {TGRID[tr]:+.2f}s   @nadir {m[i0]:+.4f}")

    # Envelope A10 Part B baseline (paraphrased):
    #   near-pair wPLI: peaks post-nadir (synchronous with PLV)
    #   far-pair  wPLI: peaks PRE-nadir at ~−1 s (long-range coupling leads)
    #   ICoh: similar pattern; troughs at PLV peak
    print(f"\n(envelope A10 Part B: far-pair wPLI/ICoh peak pre-nadir; near-pair post-nadir)")

    # Pre-nadir [-1.5, -0.5] s vs post-nadir [+0.5, +1.5] s window ratios
    pre_mask = (TGRID >= -1.5) & (TGRID <= -0.5)
    post_mask = (TGRID >= 0.5) & (TGRID <= 1.5)
    print(f"\n=== Pre-nadir [-1.5,-0.5] vs post-nadir [+0.5,+1.5] ===")
    for k in ['wPLI_near', 'wPLI_far', 'ICoh_near', 'ICoh_far']:
        m, _, _ = gm[k]
        pre = np.nanmean(m[pre_mask])
        post = np.nanmean(m[post_mask])
        print(f"  {k:10s}: pre {pre:+.4f}   post {post:+.4f}   post−pre {post-pre:+.4f}")

    out_csv = {'t_rel': TGRID}
    for k in ['wPLI_near', 'wPLI_far', 'ICoh_near', 'ICoh_far']:
        m, lo, hi = gm[k]
        out_csv[f'{k}_mean'] = m
        out_csv[f'{k}_lo'] = lo
        out_csv[f'{k}_hi'] = hi
    pd.DataFrame(out_csv).to_csv(
        os.path.join(out_dir, 'wpli_icoh_near_far.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax = axes[0]
    for k, col in [('wPLI_near', '#d73027'), ('wPLI_far', '#4575b4')]:
        m, lo, hi = gm[k]
        ax.plot(TGRID, m, color=col, lw=2, label=k)
        ax.fill_between(TGRID, lo, hi, color=col, alpha=0.2)
    ax.axvline(0, color='k', ls='--', lw=0.5, label='nadir')
    ax.axvspan(-1.5, -0.5, color='gray', alpha=0.1, label='pre')
    ax.axvspan(+0.5, +1.5, color='orange', alpha=0.1, label='post')
    ax.set_ylabel('wPLI')
    ax.set_title(f'A10 Part B · wPLI near vs far · {args.cohort} composite v2')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for k, col in [('ICoh_near', '#d73027'), ('ICoh_far', '#4575b4')]:
        m, lo, hi = gm[k]
        ax.plot(TGRID, m, color=col, lw=2, label=k)
        ax.fill_between(TGRID, lo, hi, color=col, alpha=0.2)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.axvspan(-1.5, -0.5, color='gray', alpha=0.1)
    ax.axvspan(+0.5, +1.5, color='orange', alpha=0.1)
    ax.set_ylabel('imaginary coherence')
    ax.set_xlabel('time relative to nadir (s)')
    ax.set_title('ICoh near vs far')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wpli_icoh_near_far.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/wpli_icoh_near_far.png")


if __name__ == '__main__':
    main()
