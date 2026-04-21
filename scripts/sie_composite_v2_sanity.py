#!/usr/bin/env python3
"""
Composite v2 sanity test — standalone, does NOT modify extraction pipeline.

Loads N LEMON subjects, runs composite detector v2 on each, and compares
the composite-detected event set against the envelope-detected (legacy)
event set already saved in exports_sie/lemon/.

Questions:
  1. How many events does each detector find per subject?
  2. Overlap rate: what fraction of composite events align (within ±2 s)
     with envelope events?
  3. Of envelope events NOT picked up by composite: are they low-template_rho
     (composite rejecting noise) or canonical (composite missing them)?
  4. Of composite events NOT picked up by envelope: stealth canonical events
     envelope missed?

Output: a per-subject comparison CSV + summary stats.
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
from scripts.sie_composite_detector_v2 import detect_sie_composite

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

ALIGN_TOL = 2.0   # seconds; composite & envelope events within this are "same"
THRESHOLD = 1.5   # composite S-peak threshold


def compare_one(args):
    sub_id, sample_seed = args
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
        comp_events = detect_sie_composite(X, fs, threshold=THRESHOLD)
    except Exception as e:
        print(f"{sub_id} composite failed: {e}")
        return None

    env_times = np.array(sorted(env_events['t0_net'].values))
    comp_onsets = np.array(sorted([e.t_onset for e in comp_events]))
    comp_detects = np.array(sorted([e.t_detect for e in comp_events]))

    def align(a, b):
        # For each item in a, is there an item in b within ALIGN_TOL?
        if len(a) == 0 or len(b) == 0:
            return np.zeros(len(a), dtype=bool)
        out = np.zeros(len(a), dtype=bool)
        for i, x in enumerate(a):
            if np.min(np.abs(b - x)) <= ALIGN_TOL:
                out[i] = True
        return out

    # envelope → composite alignment (check if envelope event has a
    # composite onset nearby)
    env_matched = align(env_times, comp_onsets)
    # composite → envelope alignment
    comp_matched = align(comp_onsets, env_times)

    # Template_rho for envelope events
    try:
        q = pd.read_csv(QUALITY_CSV)
        q_sub = q[q['subject_id'] == sub_id].copy()
        q_sub['t0_round'] = q_sub['t0_net'].round(3)
        env_events['t0_round'] = env_events['t0_net'].round(3)
        env_joined = env_events.merge(q_sub[['t0_round','template_rho']],
                                       on='t0_round', how='left')
        env_rho = env_joined['template_rho'].values
    except Exception:
        env_rho = np.full(len(env_events), np.nan)

    return {
        'subject_id': sub_id,
        'n_env': len(env_events),
        'n_comp': len(comp_events),
        'n_match_env_to_comp': int(env_matched.sum()),
        'n_match_comp_to_env': int(comp_matched.sum()),
        'pct_env_matched': float(env_matched.mean() * 100),
        'pct_comp_matched': float(comp_matched.mean() * 100),
        'median_rho_matched_env': (float(np.nanmedian(env_rho[env_matched]))
                                    if env_matched.any() else np.nan),
        'median_rho_unmatched_env': (float(np.nanmedian(env_rho[~env_matched]))
                                      if (~env_matched).any() else np.nan),
        'n_comp_stealth': int((~comp_matched).sum()),
        'mean_comp_S': float(np.mean([e.S_at_detect for e in comp_events])
                              if comp_events else np.nan),
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=4)]
    rng = np.random.default_rng(42)
    n_sample = int(os.environ.get('N_SUBJECTS', 15))
    subs = rng.choice(ok['subject_id'].values,
                       size=min(n_sample, len(ok)), replace=False).tolist()
    print(f"Composite v2 sanity on {len(subs)} random LEMON subjects")
    print(f"Threshold: S ≥ {THRESHOLD}  alignment tol: ±{ALIGN_TOL}s")

    n_workers = int(os.environ.get('SIE_WORKERS',
                                    min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(compare_one,
                            [(s, 42) for s in subs])
    df = pd.DataFrame([r for r in results if r is not None])
    out_csv = os.path.join(OUT_DIR, 'composite_v2_sanity_comparison.csv')
    df.to_csv(out_csv, index=False)

    print(f"\n=== Per-subject comparison (n={len(df)}) ===")
    print(df[['subject_id','n_env','n_comp',
               'pct_env_matched','pct_comp_matched',
               'median_rho_matched_env','median_rho_unmatched_env',
               'n_comp_stealth']].to_string(index=False))

    print(f"\n=== Aggregate ===")
    print(f"  envelope events median per subject: {df['n_env'].median():.0f}")
    print(f"  composite events median per subject: {df['n_comp'].median():.0f}")
    print(f"  ratio comp/env median: {(df['n_comp']/df['n_env']).median():.2f}")
    print(f"  % envelope events matched by composite (median subj): "
          f"{df['pct_env_matched'].median():.0f}%")
    print(f"  % composite events matched by envelope (median subj): "
          f"{df['pct_comp_matched'].median():.0f}%")
    print(f"  median template_ρ of MATCHED envelope events: "
          f"{df['median_rho_matched_env'].median():.3f}")
    print(f"  median template_ρ of UNMATCHED envelope events: "
          f"{df['median_rho_unmatched_env'].median():.3f}")
    print(f"\n  If unmatched-rho < matched-rho → composite correctly "
          f"rejects noise")
    print(f"  If comp_stealth events exist at moderate n → composite "
          f"finds events envelope missed")


if __name__ == '__main__':
    main()
