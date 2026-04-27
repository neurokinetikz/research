#!/usr/bin/env python3
"""Off-frequency detector control: peri-event four-stream architecture.

Companion to sie_off_frequency_detector_control.py. Tests whether
off-f0 events show the same canonical six-phase peri-event architecture
in the four detector streams (envelope, Kuramoto R, PLV, MSC) as canonical
f0=7.6 events do.

For each detector at f0 in {7.6, 8.6, 12.0}, on the same LEMON EC subjects:
  1. Run the four-stream composite-v2 detector at f0
  2. For each detected event, extract peri-event traces of (env, R, PLV, MSC)
     in a window [-3, +3] s around onset.
  3. Pool and average; report grand-mean peri-event trace per stream.

Outputs:
  outputs/schumann/images/psd_timelapse/lemon_composite/
    off_frequency_peri_event_streams.csv

Per-stream comparison: do off-f0 events show:
  - Pre-onset triple-stream nadir at ~-1.3 s?
  - Rebound peak at ~+1.1 s?
  - 1:4 rise:decay ratio?
"""
from __future__ import annotations
import os
import sys
import glob as globfn
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.detect_ignition import _composite_streams, _composite_S, _composite_refine_onset
from scripts.run_sie_extraction import load_lemon
from scipy import signal
import mne

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse', 'lemon_composite')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie',
                           'lemon_composite')
os.makedirs(OUT_DIR, exist_ok=True)

PERI_PRE = 3.0
PERI_POST = 3.0
COMP_THRESH = 1.5
MIN_ISI = 2.0
EDGE_S = 5.0

F0_LIST = [7.6, 8.6, 12.0]


def detect_offset(Y, fs, f0):
    """Run composite-v2 detector with f0 and R_band shifted symmetrically."""
    half_bw = 0.6
    R_band = (f0 - 0.6, f0 + 0.6)
    t, env, R, P, M = _composite_streams(Y, fs, f0=f0, half_bw=half_bw, R_band=R_band)
    S = _composite_S(env, R, P, M)
    mask = (t >= t[0] + EDGE_S) & (t <= t[-1] - EDGE_S)
    S_m = S.copy()
    S_m[~mask] = -np.inf
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    peak_idx, _ = signal.find_peaks(
        S_m, distance=max(1, int(round(MIN_ISI / dt))), height=COMP_THRESH,
    )
    onsets = []
    for pi in peak_idx:
        t_on = _composite_refine_onset(t, env, R, P, M, float(t[pi]))
        onsets.append(t_on)
    return np.array(sorted(onsets)), t, env, R, P, M


def baseline_z(x, t, onset, baseline=(-5.0, -3.0)):
    rel = t - onset
    bm = (rel >= baseline[0]) & (rel < baseline[1])
    if bm.sum() < 3:
        return None
    mu = np.nanmean(x[bm])
    sd = np.nanstd(x[bm])
    if not np.isfinite(sd) or sd < 1e-9:
        return None
    return (x - mu) / sd


def process_subject(args):
    sub_id, f0 = args
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    rec_dur = X.shape[1] / fs
    if rec_dur < 60:
        return None

    onsets, t_stream, env, R, P, M = detect_offset(X, fs, f0)
    if len(onsets) < 1:
        return None

    # 0.1-s peri-event grid
    rel_grid = np.arange(-PERI_PRE, PERI_POST + 1e-6, 0.1)

    def gather(stream_vals):
        rows = []
        for t_on in onsets:
            z = baseline_z(stream_vals, t_stream, t_on)
            if z is None:
                continue
            # Interpolate z onto rel_grid
            rel = t_stream - t_on
            mask = (rel >= -PERI_PRE - 0.2) & (rel <= PERI_POST + 0.2)
            if mask.sum() < 5:
                continue
            try:
                row = np.interp(rel_grid, rel[mask], z[mask],
                                 left=np.nan, right=np.nan)
                rows.append(row)
            except Exception:
                continue
        if not rows:
            return None
        return np.nanmean(np.array(rows), axis=0)

    out = {}
    for name, vals in (('env', env), ('R', R), ('PLV', P), ('MSC', M)):
        out[name] = gather(vals)

    return (sub_id, rel_grid, out, len(onsets))


def main():
    n_max = int(os.environ.get('SIE_OFF_N', '50'))
    n_workers = int(os.environ.get('SIE_OFF_WORKERS', '4'))

    csvs = sorted(globfn.glob(os.path.join(EVENTS_BASE, 'sub-*_sie_events.csv')))
    sub_ids = [os.path.basename(p).replace('_sie_events.csv', '') for p in csvs]
    sub_ids = sub_ids[:n_max]
    print(f'Off-frequency peri-event four-stream comparison')
    print(f'  Subjects = {len(sub_ids)} (LEMON EC)')
    print(f'  f0 list  = {F0_LIST} Hz')
    print()

    all_results = {}
    rel_grid = None
    for f0 in F0_LIST:
        print(f'--- f0 = {f0:.1f} Hz ---')
        args_list = [(s, f0) for s in sub_ids]
        results = []
        with Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_subject, args_list)):
                if res is not None:
                    results.append(res)
                if (i + 1) % 25 == 0:
                    print(f'  {i+1}/{len(args_list)} processed; {len(results)} valid')
        if not results:
            continue
        rel_grid = results[0][1]
        # Pool per-stream
        pooled = {name: [] for name in ('env', 'R', 'PLV', 'MSC')}
        n_evs = []
        for sub_id, rg, sub_streams, n_ev in results:
            n_evs.append(n_ev)
            for name in pooled:
                arr = sub_streams.get(name)
                if arr is not None:
                    pooled[name].append(arr)
        all_results[f0] = {
            'n_subj': len(results),
            'n_evs_total': sum(n_evs),
            'n_evs_mean': np.mean(n_evs),
            'streams': {name: np.nanmean(np.array(rows), axis=0)
                        for name, rows in pooled.items() if rows},
        }
        print(f'  Pooled {len(results)} subjects, mean events/subject {np.mean(n_evs):.1f}')

    # Build dataframe
    rows = []
    for i, t in enumerate(rel_grid):
        for f0 in F0_LIST:
            if f0 not in all_results:
                continue
            for name in ('env', 'R', 'PLV', 'MSC'):
                v = all_results[f0]['streams'].get(name)
                if v is None or i >= len(v):
                    continue
                rows.append(dict(t_rel=t, f0=f0, stream=name, z=float(v[i])))
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'off_frequency_peri_event_streams.csv')
    df.to_csv(out_path, index=False)
    print(f'\nWrote {out_path}')

    # Per-f0 summary: nadir time and value, peak time and value, per stream
    print()
    print('=' * 72)
    print('PERI-EVENT ARCHITECTURE BY f0')
    print('=' * 72)
    print(f'{"f0":>5}  {"stream":>5}  {"nadir t":>9}  {"nadir z":>9}  {"peak t":>9}  {"peak z":>9}')
    print('-' * 60)
    for f0 in F0_LIST:
        if f0 not in all_results:
            continue
        for name in ('env', 'R', 'PLV', 'MSC'):
            v = all_results[f0]['streams'].get(name)
            if v is None:
                continue
            ni = int(np.nanargmin(v))
            pi = int(np.nanargmax(v))
            print(f'{f0:>5.1f}  {name:>5}  {rel_grid[ni]:>+8.2f}s  {v[ni]:>+9.3f}  '
                  f'{rel_grid[pi]:>+8.2f}s  {v[pi]:>+9.3f}')
        print()


if __name__ == '__main__':
    main()
