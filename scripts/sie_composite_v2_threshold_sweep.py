#!/usr/bin/env python3
"""
B61v2 — Composite v2 threshold sweep.

At S≥1.5 (B61), composite found 4× more events than envelope. Sweep
threshold to find the value where composite event count matches envelope
count, then re-check overlap + template_rho stratification at that
threshold.

For each subject and each threshold in THRESHOLDS:
  1. Run composite_v2 at that threshold
  2. Compare to envelope-detected events
  3. Record: n_composite, n_matched_envelope, n_matched_composite,
     median rho matched, median rho unmatched
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_composite_detector_v2 import (compute_streams, composite_S,
                                                 detect_events,
                                                 refine_onset_nadir)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'quality')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie',
                           'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality',
                            'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
ALIGN_TOL = 2.0


def process_one(sub_id):
    try:
        env_events = pd.read_csv(os.path.join(EVENTS_DIR,
                                               f'{sub_id}_sie_events.csv'))
    except Exception:
        return None
    env_events = env_events.dropna(subset=['t0_net'])
    if len(env_events) == 0:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    X = raw.get_data() * 1e6
    fs = raw.info['sfreq']
    try:
        t, env, R, P, M = compute_streams(X, fs)
        S = composite_S(env, R, P, M)
    except Exception as e:
        print(f"{sub_id} streams fail: {e}")
        return None
    env_times = np.array(sorted(env_events['t0_net'].values))

    # Template_rho for envelope events
    try:
        q = pd.read_csv(QUALITY_CSV)
        q_sub = q[q['subject_id']==sub_id].copy()
        q_sub['t0_round'] = q_sub['t0_net'].round(3)
        env_events['t0_round'] = env_events['t0_net'].round(3)
        env_joined = env_events.merge(q_sub[['t0_round','template_rho']],
                                       on='t0_round', how='left')
        env_rho = env_joined['template_rho'].values
    except Exception:
        env_rho = np.full(len(env_events), np.nan)

    rows = []
    for thr in THRESHOLDS:
        t_detects, S_vals = detect_events(t, S, threshold=thr)
        onsets = np.array([refine_onset_nadir(t, env, R, P, M, td)
                            for td in t_detects])
        # alignment
        env_matched = np.zeros(len(env_times), dtype=bool)
        comp_matched = np.zeros(len(onsets), dtype=bool)
        if len(onsets) and len(env_times):
            for i, ev_t in enumerate(env_times):
                if np.min(np.abs(onsets - ev_t)) <= ALIGN_TOL:
                    env_matched[i] = True
            for i, on in enumerate(onsets):
                if np.min(np.abs(env_times - on)) <= ALIGN_TOL:
                    comp_matched[i] = True
        rows.append({
            'subject_id': sub_id, 'threshold': thr,
            'n_env': len(env_times), 'n_comp': len(onsets),
            'n_match_env': int(env_matched.sum()),
            'n_match_comp': int(comp_matched.sum()),
            'pct_env_matched': (float(env_matched.mean()*100)
                                 if len(env_times) else np.nan),
            'pct_comp_matched': (float(comp_matched.mean()*100)
                                  if len(onsets) else np.nan),
            'median_rho_matched': (float(np.nanmedian(env_rho[env_matched]))
                                    if env_matched.any() else np.nan),
            'median_rho_unmatched': (float(np.nanmedian(env_rho[~env_matched]))
                                      if (~env_matched).any() else np.nan),
        })
    return rows


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=4)]
    rng = np.random.default_rng(42)
    n = int(os.environ.get('N_SUBJECTS', 15))
    subs = rng.choice(ok['subject_id'].values,
                       size=min(n, len(ok)), replace=False).tolist()
    print(f"Composite v2 threshold sweep: {len(subs)} subjects × "
          f"{len(THRESHOLDS)} thresholds = {len(subs)*len(THRESHOLDS)} combos")
    print(f"Thresholds: {THRESHOLDS}")

    n_workers = int(os.environ.get('SIE_WORKERS',
                                    min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        out_lists = pool.map(process_one, subs)
    all_rows = [row for rs in out_lists if rs for row in rs]
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, 'composite_v2_threshold_sweep.csv')
    df.to_csv(out_csv, index=False)

    print(f"\n=== Per-threshold aggregate (median across subjects) ===")
    print(f"{'thr':>6}{'n_env':>7}{'n_comp':>8}{'ratio':>8}"
          f"{'%env_m':>9}{'%comp_m':>10}"
          f"{'rho_m':>9}{'rho_u':>9}")
    print('-' * 70)
    for thr, sub in df.groupby('threshold'):
        print(f"{thr:>6.1f}{sub['n_env'].median():>7.1f}"
              f"{sub['n_comp'].median():>8.1f}"
              f"{(sub['n_comp']/sub['n_env']).median():>8.2f}"
              f"{sub['pct_env_matched'].median():>8.0f}%"
              f"{sub['pct_comp_matched'].median():>9.0f}%"
              f"{sub['median_rho_matched'].median():>+9.3f}"
              f"{sub['median_rho_unmatched'].median():>+9.3f}")
    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
