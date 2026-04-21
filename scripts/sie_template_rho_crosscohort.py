#!/usr/bin/env python3
"""
Compute per-event template_rho for HBN R4 and TDBRAIN cohorts, matching the
LEMON definition in sie_event_quality.py.

template_rho := Pearson correlation between this event's envelope-z trajectory
on [-5, +5] s and the cohort's grand-average envelope template.

Output: outputs/schumann/images/quality/per_event_quality_{cohort}.csv with
columns {subject_id, t0_net, template_rho}.
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon, load_hbn, load_tdbrain
from scripts.sie_perionset_triple_average import bandpass

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'quality')
EVENTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
HBN_DATA = '/Volumes/T9/hbn_data'
os.makedirs(OUT_DIR, exist_ok=True)

F0 = 7.83
HALF_BW = 0.6
PRE_SEC = 10.0
POST_SEC = 10.0
STEP_SEC = 0.1
WIN_SEC = 1.0
TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC / 2, STEP_SEC)


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-9:
        return x - med
    return (x - med) / mad


def compute_env_stream(raw):
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    env_vals, centers = [], []
    for i in range(0, X.shape[1] - nwin + 1, nstep):
        env_vals.append(float(np.mean(env[i:i+nwin])))
        centers.append((i + nwin / 2) / fs)
    t = np.array(centers)
    zE = robust_z(np.array(env_vals))
    return t, zE, raw.times[-1]


def process_subject(args):
    cohort, sub_id, events_path, load_arg = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) == 0:
        return None
    try:
        if cohort == 'hbn':
            raw = load_hbn(load_arg)
        elif cohort == 'tdbrain':
            raw = load_tdbrain(sub_id, condition='EC')
        elif cohort == 'lemon':
            raw = load_lemon(sub_id, condition='EC')
        else:
            return None
    except Exception:
        return None
    if raw is None:
        return None
    try:
        t_full, zE_full, t_end = compute_env_stream(raw)
    except Exception:
        return None

    rows = []
    trajs = []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        if t0 - PRE_SEC < 2 or t0 + POST_SEC > t_end - 2:
            continue
        sel = (t_full >= t0 - PRE_SEC) & (t_full <= t0 + POST_SEC)
        if sel.sum() < int(PRE_SEC + POST_SEC) * 5:
            continue
        t_rel = t_full[sel] - t0
        zE_seg = zE_full[sel]
        env_i = np.interp(TGRID, t_rel, zE_seg, left=np.nan, right=np.nan)
        rows.append({'subject_id': sub_id, 't0_net': t0})
        trajs.append(env_i)

    if not rows:
        return None
    return {'rows': rows, 'trajs': np.array(trajs)}


def build_tasks(cohort):
    if cohort == 'lemon':
        summary = pd.read_csv(os.path.join(EVENTS_ROOT, 'lemon',
                                            'extraction_summary.csv'))
        ok = summary[(summary['status']=='ok') & (summary['n_events']>=2)]
        tasks = []
        for _, r in ok.iterrows():
            ep = os.path.join(EVENTS_ROOT, 'lemon',
                              f'{r["subject_id"]}_sie_events.csv')
            if os.path.isfile(ep):
                tasks.append(('lemon', r['subject_id'], ep, None))
        return tasks
    if cohort.startswith('hbn'):
        # cohort e.g. 'hbn_R4' or 'hbn' (default R4)
        if cohort == 'hbn':
            release = 'R4'
        else:
            release = cohort.split('_', 1)[1]
        events_key = f'hbn_{release}'
        summary = pd.read_csv(os.path.join(EVENTS_ROOT, events_key,
                                            'extraction_summary.csv'))
        ok = summary[(summary['status']=='ok') & (summary['n_events']>=2)]
        release_dir = os.path.join(HBN_DATA, f'cmi_bids_{release}')
        tasks = []
        for _, r in ok.iterrows():
            sub = r['subject_id']
            ep = os.path.join(EVENTS_ROOT, events_key, f'{sub}_sie_events.csv')
            sp = os.path.join(release_dir, sub, 'eeg',
                              f'{sub}_task-RestingState_eeg.set')
            if os.path.isfile(ep) and os.path.isfile(sp):
                tasks.append(('hbn', sub, ep, sp))
        return tasks
    if cohort == 'tdbrain':
        summary = pd.read_csv(os.path.join(EVENTS_ROOT, 'tdbrain',
                                            'extraction_summary.csv'))
        ok = summary[(summary['status']=='ok') & (summary['n_events']>=2)]
        tasks = []
        for _, r in ok.iterrows():
            sub = r['subject_id']
            ep = os.path.join(EVENTS_ROOT, 'tdbrain', f'{sub}_sie_events.csv')
            if os.path.isfile(ep):
                tasks.append(('tdbrain', sub, ep, None))
        return tasks
    raise ValueError(cohort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True,
                    help='lemon, tdbrain, hbn (=R4), or hbn_R1/R2/R3/R4/R6')
    ap.add_argument('--save-trajectories', action='store_true',
                    help='Save per-event envelope trajectories for '
                         'cross-cohort shared-template re-scoring')
    args = ap.parse_args()
    tasks = build_tasks(args.cohort)
    n_workers = int(os.environ.get('SIE_WORKERS',
                                    min(4, os.cpu_count() or 4)))
    print(f"[{args.cohort}] subjects: {len(tasks)} workers: {n_workers}")
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"[{args.cohort}] subjects ok: {len(results)}")

    all_rows = []
    all_trajs = []
    for r in results:
        all_rows.extend(r['rows'])
        all_trajs.append(r['trajs'])
    df = pd.DataFrame(all_rows)
    traj_mat = np.vstack(all_trajs) if all_trajs else np.array([])
    print(f"[{args.cohort}] scored events: {len(df)}")
    if len(df) == 0:
        return

    m_core = (TGRID >= -5) & (TGRID <= +5)
    template = np.nanmean(traj_mat, axis=0)
    tmpl_core = template[m_core]
    tmpl_core = tmpl_core - np.nanmean(tmpl_core)
    rhos = []
    for i in range(traj_mat.shape[0]):
        ev = traj_mat[i, m_core]
        if np.any(~np.isfinite(ev)):
            rhos.append(np.nan); continue
        ev_c = ev - np.nanmean(ev)
        denom = np.sqrt(np.nansum(ev_c**2) * np.nansum(tmpl_core**2))
        rhos.append(float(np.nansum(ev_c * tmpl_core) / denom)
                     if denom > 0 else np.nan)
    df['template_rho'] = rhos
    out_csv = os.path.join(OUT_DIR, f'per_event_quality_{args.cohort}.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    if getattr(args, 'save_trajectories', False):
        out_npz = os.path.join(OUT_DIR, f'trajectories_{args.cohort}.npz')
        np.savez_compressed(out_npz,
                             trajs=traj_mat.astype(np.float32),
                             subject_id=df['subject_id'].values,
                             t0_net=df['t0_net'].values,
                             tgrid=TGRID.astype(np.float32))
        print(f"Saved trajectories: {out_npz} shape={traj_mat.shape}")
    print(f"[{args.cohort}] template_rho median {np.nanmedian(rhos):.3f} "
          f"IQR [{np.nanpercentile(rhos,25):.3f}, {np.nanpercentile(rhos,75):.3f}]")


if __name__ == '__main__':
    main()
