#!/usr/bin/env python3
"""
SIE Window Enrichment Analysis
================================
For each subject with detected SIE events, extract pre-baseline, ignition,
and post-baseline windows, concatenate within subject per condition, run
FOOOF peak extraction, and compute enrichment profiles per band.

Tests whether SIE events correspond to shifts in the spectral enrichment
landscape -- connecting the SIE framework to the spectral differentiation
framework.

Primary hypothesis: alpha mountain sharpening during ignition
  (attractor enrichment ↑, noble_1 flanks ↑, boundary ↓)

Usage:
    python scripts/sie_window_enrichment.py --dataset eegmmidb --window 20
"""

import os
import sys
import argparse
import warnings
import gc
import time
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, permutation_test, wilcoxon
import mne

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Reuse existing infrastructure
from scripts.run_f0_760_extraction import (
    load_eegmmidb, load_lemon, load_dortmund, load_chbmp, load_hbn, load_tdbrain,
    extract_adaptive_subject, F0, TARGET_FS,
)
from scripts.run_all_f0_760_analyses import (
    per_subject_enrichment, OCTAVE_BAND, BAND_ORDER, POS_NAMES,
)

SIE_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sie_events_for_dataset(dataset_dir):
    """Load SIE events from per-subject CSVs, filter MSC artifacts."""
    LABELS = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']
    msc_cols = [f'msc_{lbl}_v' for lbl in LABELS]

    dfs = []
    for f in glob(os.path.join(SIE_BASE, dataset_dir, '*_sie_events.csv')):
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    events = pd.concat(dfs, ignore_index=True)
    for c in msc_cols + ['t_start', 't_end']:
        if c in events.columns:
            events[c] = pd.to_numeric(events[c], errors='coerce')
    events['msc_mean'] = events[msc_cols].mean(axis=1)
    # MSC artifact filter
    events = events[events['msc_mean'] < 0.9].copy()
    return events


def extract_windows(raw, event_windows, window_sec, buffer_sec=5.0):
    """Extract concatenated pre/ignition/post windows from a raw EEG recording.

    Parameters
    ----------
    raw : mne.io.Raw
    event_windows : list of (t_start, t_end) tuples in seconds
    window_sec : int, duration of each window
    buffer_sec : float, buffer between event and pre/post windows

    Returns
    -------
    pre_raw, ign_raw, post_raw : mne.io.Raw objects with concatenated data
    """
    fs = raw.info['sfreq']
    duration = raw.n_times / fs

    pre_segments = []
    ign_segments = []
    post_segments = []

    for t_start, t_end in event_windows:
        # Ignition: centered on event, window_sec long
        event_mid = (t_start + t_end) / 2
        ign_lo = max(0, event_mid - window_sec / 2)
        ign_hi = min(duration, event_mid + window_sec / 2)

        # Pre: window_sec before t_start - buffer
        pre_hi = t_start - buffer_sec
        pre_lo = pre_hi - window_sec

        # Post: window_sec after t_end + buffer
        post_lo = t_end + buffer_sec
        post_hi = post_lo + window_sec

        if pre_lo >= 0 and (ign_hi - ign_lo) > window_sec * 0.8 \
                and post_hi <= duration:
            try:
                pre = raw.copy().crop(tmin=pre_lo, tmax=pre_hi).get_data()
                ign = raw.copy().crop(tmin=ign_lo, tmax=ign_hi).get_data()
                post = raw.copy().crop(tmin=post_lo, tmax=post_hi).get_data()
                pre_segments.append(pre)
                ign_segments.append(ign)
                post_segments.append(post)
            except Exception:
                continue

    if not ign_segments:
        return None, None, None

    # Concatenate along time axis
    pre_data = np.concatenate(pre_segments, axis=1)
    ign_data = np.concatenate(ign_segments, axis=1)
    post_data = np.concatenate(post_segments, axis=1)

    info = raw.info
    pre_raw = mne.io.RawArray(pre_data, info, verbose=False)
    ign_raw = mne.io.RawArray(ign_data, info, verbose=False)
    post_raw = mne.io.RawArray(post_data, info, verbose=False)

    return pre_raw, ign_raw, post_raw


def process_subject(sub_id, raw_loader_fn, events_df, window_sec=20,
                     buffer_sec=5.0, min_events=3):
    """Process one subject: load raw, extract windows, compute enrichment."""
    sub_events = events_df[events_df['subject_id'] == sub_id]
    if len(sub_events) < min_events:
        return None

    try:
        raw = raw_loader_fn()
    except Exception as e:
        return {'subject_id': sub_id, 'status': 'load_error', 'error': str(e)[:100]}

    if raw is None:
        return {'subject_id': sub_id, 'status': 'no_data'}

    fs = raw.info['sfreq']
    event_windows = list(zip(sub_events['t_start'].values, sub_events['t_end'].values))

    pre_raw, ign_raw, post_raw = extract_windows(
        raw, event_windows, window_sec=window_sec, buffer_sec=buffer_sec)

    if pre_raw is None:
        del raw; gc.collect()
        return {'subject_id': sub_id, 'status': 'window_fail', 'n_events': len(sub_events)}

    n_used = pre_raw.n_times / fs / window_sec

    # Run FOOOF extraction on each condition's concatenated signal
    results = {}
    for condition, c_raw in [('pre', pre_raw), ('ignition', ign_raw), ('post', post_raw)]:
        try:
            peaks_df, _ = extract_adaptive_subject(c_raw, F0, fs, method='fooof')
            if len(peaks_df) > 0:
                enr = per_subject_enrichment(peaks_df, min_peaks=10)
                results[condition] = enr
                results[f'{condition}_n_peaks'] = len(peaks_df)
            else:
                results[condition] = None
        except Exception as e:
            results[condition] = None
            results[f'{condition}_error'] = str(e)[:100]

    del raw, pre_raw, ign_raw, post_raw; gc.collect()

    return {
        'subject_id': sub_id,
        'status': 'ok',
        'n_events': len(sub_events),
        'n_windows_used': int(n_used),
        **{f'{cond}_{k}': v for cond, res in results.items()
           if isinstance(res, dict) for k, v in res.items()},
        **{k: v for k, v in results.items() if not isinstance(v, dict)},
    }


def get_loader_for(dataset, condition=None, session='1', release='R1'):
    """Return a function that loads a given subject from a given dataset."""
    if dataset == 'eegmmidb':
        sie_dir = 'eegmmidb'
        def make_loader(sub_id):
            return lambda: load_eegmmidb(int(sub_id[1:]))
    elif dataset == 'lemon':
        sie_dir = 'lemon' if (not condition or condition == 'EC') else 'lemon_EO'
        cond = condition or 'EC'
        def make_loader(sub_id):
            return lambda: load_lemon(sub_id, condition=cond)
    elif dataset == 'dortmund':
        cond = condition or 'EC-pre'
        task = 'EyesOpen' if cond.startswith('EO') else 'EyesClosed'
        acq = 'post' if cond.endswith('post') else 'pre'
        suffix = '' if (cond == 'EC-pre' and session == '1') else f'_{cond.replace("-", "_")}'
        ses_suffix = f'_ses2' if session == '2' else ''
        sie_dir = f'dortmund{suffix}{ses_suffix}'
        def make_loader(sub_id):
            return lambda: load_dortmund(sub_id, task=task, acq=acq, ses=session)
    elif dataset == 'chbmp':
        sie_dir = 'chbmp'
        def make_loader(sub_id):
            return lambda: load_chbmp(sub_id)
    elif dataset == 'hbn':
        sie_dir = f'hbn_{release}'
        release_dir = f'/Volumes/T9/hbn_data/cmi_bids_{release}'
        def make_loader(sub_id):
            import glob as g
            pattern = os.path.join(release_dir, sub_id, 'eeg', '*RestingState_eeg.set')
            files = g.glob(pattern)
            if not files:
                return None
            return load_hbn(files[0])
    elif dataset == 'tdbrain':
        cond = condition or 'EC'
        sie_dir = 'tdbrain' if cond == 'EC' else 'tdbrain_EO'
        def make_loader(sub_id):
            return lambda: load_tdbrain(sub_id, condition=cond, session=session if session != '1' else None)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return sie_dir, make_loader


def run_dataset(dataset, condition=None, session='1', release='R1',
                 window_sec=20, buffer_sec=5.0, min_events=3):
    """Run window enrichment analysis on any dataset."""
    sie_dir, make_loader = get_loader_for(dataset, condition, session, release)

    label = f"{dataset}{f'_{condition}' if condition else ''}{f'_R{release[1:]}' if dataset=='hbn' else ''}"
    print(f"\n{'='*80}")
    print(f"SIE Window Enrichment: {label}")
    print(f"SIE dir: {sie_dir}, Window={window_sec}s, buffer={buffer_sec}s, min_events={min_events}")
    print(f"{'='*80}\n")

    events_df = load_sie_events_for_dataset(sie_dir)
    if len(events_df) == 0:
        print(f"No events found in {sie_dir}, aborting")
        return pd.DataFrame()

    print(f"Loaded {len(events_df):,} clean events from "
          f"{events_df['subject_id'].nunique()} subjects")

    event_counts = events_df.groupby('subject_id').size()
    subjects = event_counts[event_counts >= min_events].index.tolist()
    print(f"Subjects with ≥{min_events} events: {len(subjects)}")

    results = []
    t_start = time.time()
    for i, sub_id in enumerate(subjects):
        loader = make_loader(sub_id)
        if loader is None:
            continue

        result = process_subject(sub_id, loader, events_df, window_sec,
                                  buffer_sec, min_events)
        if result:
            results.append(result)
            if i % 10 == 0:
                n_ok = sum(1 for r in results if r.get('status') == 'ok')
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                print(f"  [{i+1}/{len(subjects)}] {sub_id}: {result.get('status','?')} "
                      f"({n_ok} ok, {rate:.1f}/min)")

    results_df = pd.DataFrame(results)
    out_name = f'sie_window_enrichment_{label}.csv'
    out_path = os.path.join(OUTPUT_DIR, out_name)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  N subjects processed: {len(results_df)}")
    print(f"  N with status=ok: {(results_df['status']=='ok').sum()}")

    return results_df


def analyze_results(results_df):
    """Compare ignition vs (pre + post)/2 across subjects for each enrichment metric."""
    ok = results_df[results_df['status'] == 'ok'].copy()
    if len(ok) < 10:
        print(f"Only {len(ok)} subjects with status=ok, too few for analysis")
        return

    print(f"\n{'='*80}")
    print(f"ANALYSIS: IGNITION vs BASELINE (pre/post average)")
    print(f"N = {len(ok)} subjects")
    print(f"{'='*80}\n")

    # Key scalar metrics per band
    metrics = ['attractor', 'noble_1', 'boundary', 'mountain', 'ushape',
                'peak_height', 'ramp_depth', 'center_depletion', 'asymmetry']

    results_rows = []
    for band in BAND_ORDER:
        print(f"\n--- {band.upper()} ---")
        for metric in metrics:
            ign_col = f'ignition_{band}_{metric}'
            pre_col = f'pre_{band}_{metric}'
            post_col = f'post_{band}_{metric}'

            if ign_col not in ok.columns:
                continue

            trip = ok[[ign_col, pre_col, post_col]].apply(
                pd.to_numeric, errors='coerce').dropna()
            if len(trip) < 10:
                continue

            baseline = (trip[pre_col] + trip[post_col]) / 2
            ignition = trip[ign_col]
            diff = ignition - baseline

            t, p = ttest_rel(ignition, baseline)
            d = diff.mean() / diff.std() if diff.std() > 0 else 0

            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {metric:>20s}: ign={ignition.mean():+7.2f}  "
                  f"base={baseline.mean():+7.2f}  diff={diff.mean():+6.2f}  "
                  f"t={t:+.2f} p={p:.3e} d={d:+.3f} N={len(trip)} {sig}")

            results_rows.append({
                'band': band, 'metric': metric, 'n': len(trip),
                'ign_mean': ignition.mean(), 'base_mean': baseline.mean(),
                'diff_mean': diff.mean(), 't': t, 'p': p, 'd': d,
            })

    # Save results
    results_out = pd.DataFrame(results_rows)
    out_path = os.path.join(OUTPUT_DIR, 'sie_window_enrichment_stats.csv')
    results_out.to_csv(out_path, index=False)

    # FDR correction
    from scipy.stats import false_discovery_control
    if len(results_out) > 0:
        pvals = results_out['p'].values
        qvals = false_discovery_control(pvals)
        results_out['q_fdr'] = qvals
        n_fdr = (qvals < 0.05).sum()
        print(f"\n{'='*80}")
        print(f"FDR-corrected: {n_fdr} / {len(results_out)} survive q<0.05")
        if n_fdr > 0:
            print(f"Survivors:")
            survivors = results_out[qvals < 0.05].sort_values('q_fdr')
            for _, row in survivors.iterrows():
                print(f"  {row['band']:>10s} {row['metric']:>20s}: "
                      f"d={row['d']:+.3f} q={row['q_fdr']:.3e}")
        print(f"{'='*80}")
        results_out.to_csv(out_path, index=False)

    return results_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['eegmmidb', 'lemon', 'dortmund', 'chbmp', 'hbn', 'tdbrain'])
    parser.add_argument('--condition', default=None)
    parser.add_argument('--session', default='1')
    parser.add_argument('--release', default='R1')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--buffer', type=float, default=5.0)
    parser.add_argument('--min-events', type=int, default=3)
    args = parser.parse_args()

    results_df = run_dataset(
        dataset=args.dataset, condition=args.condition,
        session=args.session, release=args.release,
        window_sec=args.window, buffer_sec=args.buffer,
        min_events=args.min_events)
    if len(results_df) > 0:
        analyze_results(results_df)


if __name__ == '__main__':
    main()
