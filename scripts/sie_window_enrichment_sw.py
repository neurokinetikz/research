#!/usr/bin/env python3
"""SIE Window Enrichment — Shape-Weighted (SW) version.

Differs from sie_window_enrichment.py in the per-subject aggregation:
  - Original: concatenate per-event 20-s windows, then Welch + FOOOF on the
              concatenated signal.  Filters by Q4 quartile when --q4 is set.
  - SW:       compute per-event Welch PSDs per channel; per-subject template_rho-
              weighted mean PSD per channel; FOOOF on the weighted-mean PSD.
              Every subject contributes weight; events are weighted continuously.

Output: outputs/sie_window_enrichment_sw_<label>.csv (same schema as Q4 mode).
"""
from __future__ import annotations
import os
import sys
import argparse
import gc
import time
import warnings
from glob import glob

import numpy as np
import pandas as pd
from scipy import signal
import mne

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_f0_760_extraction import (
    load_eegmmidb, load_lemon, load_dortmund, load_chbmp, load_hbn, load_tdbrain,
    F0, FREQ_CEIL, R2_MIN, MAX_N_PEAKS, FOOOF_BASE_PARAMS,
    build_adaptive_bands, merge_narrow_bands, _fit_channel_fooof,
)
from scripts.run_all_f0_760_analyses import (
    per_subject_enrichment,
)
from scripts.sie_window_enrichment import (
    get_loader_for, load_sie_events_for_dataset,
)
# Prefer specparam (newer, faster) if available; fall back to deprecated fooof.
try:
    from specparam import SpectralModel as FOOOF
    _BACKEND = 'specparam'
except ImportError:
    from fooof import FOOOF
    _BACKEND = 'fooof'

SIE_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
QUALITY_BASE = os.path.join(os.path.dirname(__file__), '..',
                              'outputs', 'schumann', 'images', 'quality')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def per_event_psds(raw, event_windows, window_sec, buffer_sec, fs,
                    welch_nperseg_sec=4.0):
    """Per-event, per-channel PSDs for pre/ignition/post windows.

    Within each event window (window_sec long), Welch is computed with
    nperseg = welch_nperseg_sec * fs (default 4 s) and 50%% overlap.  This
    yields a smaller PSD (Δf = 0.25 Hz) with multiple sub-segments averaged,
    matching the frequency resolution used by extract_adaptive_subject's
    band-specific Welch calls.  Smaller nperseg => fewer PSD bins =>
    much faster FOOOF fits downstream.

    Returns dict[condition] = list of per-event lists of (freqs, psd) tuples
    per channel; parallel list of valid event indices.
    """
    duration = raw.n_times / fs
    out = {'pre': [], 'ignition': [], 'post': []}
    valid_event_idx = []

    welch_n = int(round(welch_nperseg_sec * fs))
    welch_noverlap = welch_n // 2

    # Extract raw data ONCE, then slice into per-event windows.
    full_data = raw.get_data()
    n_ch = full_data.shape[0]
    n_samples = full_data.shape[1]

    for k, (t_start, t_end) in enumerate(event_windows):
        event_mid = (t_start + t_end) / 2
        ign_lo = max(0, event_mid - window_sec / 2)
        ign_hi = min(duration, event_mid + window_sec / 2)
        pre_hi = t_start - buffer_sec
        pre_lo = pre_hi - window_sec
        post_lo = t_end + buffer_sec
        post_hi = post_lo + window_sec

        if pre_lo < 0 or post_hi > duration or (ign_hi - ign_lo) <= window_sec * 0.8:
            continue

        windows = {
            'pre':      (int(round(pre_lo * fs)),  int(round(pre_hi * fs))),
            'ignition': (int(round(ign_lo * fs)),  int(round(ign_hi * fs))),
            'post':     (int(round(post_lo * fs)), int(round(post_hi * fs))),
        }
        bad = False
        for (i0, i1) in windows.values():
            if i0 < 0 or i1 > n_samples:
                bad = True
                break
        if bad:
            continue

        ev_out = {'pre': [], 'ignition': [], 'post': []}
        for cond, (i0, i1) in windows.items():
            for ch_idx in range(n_ch):
                ch = full_data[ch_idx, i0:i1]
                if len(ch) < welch_n:
                    ev_out[cond].append(None)
                    continue
                f, psd = signal.welch(ch, fs=fs, nperseg=welch_n,
                                       noverlap=welch_noverlap, scaling='density')
                ev_out[cond].append((f, psd))

        out['pre'].append(ev_out['pre'])
        out['ignition'].append(ev_out['ignition'])
        out['post'].append(ev_out['post'])
        valid_event_idx.append(k)

    return out, valid_event_idx


def weighted_mean_psd_per_channel(per_event_psds_for_cond, weights):
    """Average across events with weights, per channel.
    Returns dict[ch_idx] = (freqs, weighted_mean_psd) or None if no valid events.
    """
    if not per_event_psds_for_cond:
        return {}
    n_ch = len(per_event_psds_for_cond[0])
    out = {}
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        return {}

    for ch_idx in range(n_ch):
        # Collect non-None PSDs across events for this channel
        psds = []
        ws = []
        freqs = None
        for ev_idx, ev in enumerate(per_event_psds_for_cond):
            ps = ev[ch_idx] if ch_idx < len(ev) else None
            if ps is None:
                continue
            f, p = ps
            if freqs is None:
                freqs = f
            psds.append(p)
            ws.append(weights[ev_idx] if ev_idx < len(weights) else 0.0)
        if not psds:
            continue
        psds = np.array(psds)
        ws = np.array(ws)
        if ws.sum() <= 0:
            continue
        wmean = np.average(psds, axis=0, weights=ws)
        out[ch_idx] = (freqs, wmean)
    return out


def fooof_on_weighted_psds(weighted_psds_by_ch, ch_names, f0, fs, freq_ceil=FREQ_CEIL):
    """Run FOOOF per channel per band on weighted-mean PSDs.

    Returns peaks_df with columns: ['channel', 'CF', 'PW', 'BW', 'octave_name',
                                     'octave_n', 'r2', 'band'] matching the
    extract_adaptive_subject output.
    """
    raw_bands = build_adaptive_bands(f0, fs, freq_ceil)
    bands = merge_narrow_bands(raw_bands)
    rows = []
    for band in bands:
        bname = band['name']
        target_lo, target_hi = band['target_lo'], band['target_hi']
        fit_lo, fit_hi = band['fit_lo'], band['fit_hi']
        fit_width = fit_hi - fit_lo
        freq_res = band['freq_res']
        is_merged = '_split_at' in band
        split_freq = band.get('_split_at', None)
        theta_name = band.get('_theta_name', None)
        alpha_name = band.get('_alpha_name', None)
        max_peak_width = min(fit_width * 0.6, 12.0)
        peak_width_limits = [2 * freq_res, max_peak_width]
        fooof_params = {**FOOOF_BASE_PARAMS,
                        'max_n_peaks': MAX_N_PEAKS,
                        'peak_width_limits': peak_width_limits}

        for ch_idx, (freqs, psd) in weighted_psds_by_ch.items():
            ch_name = ch_names[ch_idx] if ch_idx < len(ch_names) else f'ch{ch_idx}'
            try:
                fm = FOOOF(**{k: v for k, v in fooof_params.items()
                              if k in ('peak_width_limits', 'max_n_peaks',
                                       'min_peak_height', 'peak_threshold',
                                       'aperiodic_mode')},
                           verbose=False)
                fm.fit(freqs, psd, [fit_lo, fit_hi])
            except Exception:
                continue
            try:
                r2 = float(fm.r_squared_) if hasattr(fm, 'r_squared_') else 1.0
            except Exception:
                r2 = 1.0
            if r2 < R2_MIN:
                continue
            peaks_arr = None
            if hasattr(fm, 'peak_params_') and fm.peak_params_ is not None:
                peaks_arr = fm.peak_params_
            elif hasattr(fm, 'get_params'):
                try:
                    peaks_arr = fm.get_params('peak')
                except Exception:
                    pass
            if peaks_arr is None:
                continue
            peaks_arr = np.atleast_2d(peaks_arr)
            for row in peaks_arr:
                if len(row) < 3:
                    continue
                cf, pw, bw = float(row[0]), float(row[1]), float(row[2])
                if not (target_lo <= cf < target_hi):
                    continue
                if is_merged:
                    if cf < split_freq:
                        octave_name = theta_name; octave_n = -1
                    else:
                        octave_name = alpha_name; octave_n = 0
                else:
                    octave_name = bname; octave_n = band['n']
                rows.append({
                    'channel': ch_name, 'freq': cf, 'power': pw,
                    'bandwidth': bw, 'phi_octave': octave_name,
                    'phi_octave_n': octave_n,
                    'r_squared': round(r2, 4),
                    'target_lo': target_lo, 'target_hi': target_hi,
                    'fit_lo': fit_lo, 'fit_hi': fit_hi,
                    'band': bname,
                })
    return pd.DataFrame(rows)


def process_subject_sw(sub_id, raw_loader_fn, events_df, weights_lookup,
                        window_sec=20, buffer_sec=5.0, min_events=3):
    """Per-subject SW: weighted-mean PSD across events, then FOOOF."""
    sub_events = events_df[events_df['subject_id'] == sub_id]
    if len(sub_events) < min_events:
        return None
    try:
        raw = raw_loader_fn()
    except Exception as e:
        return {'subject_id': sub_id, 'status': 'load_error',
                 'error': str(e)[:100]}
    if raw is None:
        return {'subject_id': sub_id, 'status': 'no_data'}

    fs = raw.info['sfreq']
    ch_names = raw.ch_names
    event_windows = list(zip(sub_events['t_start'].values,
                              sub_events['t_end'].values))
    t0_nets = sub_events['t0_net'].values

    # Per-event Welch PSDs per channel for pre/ign/post
    per_ev, valid_idx = per_event_psds(raw, event_windows, window_sec,
                                         buffer_sec, fs)
    if not valid_idx:
        del raw; gc.collect()
        return {'subject_id': sub_id, 'status': 'window_fail',
                 'n_events': len(sub_events)}
    n_used = len(valid_idx)

    # Build per-event template_rho weights aligned to valid events
    valid_t0s = [t0_nets[i] for i in valid_idx]
    weights = []
    for t0 in valid_t0s:
        w = weights_lookup.get((sub_id, round(t0, 3)), np.nan)
        weights.append(max(float(w), 0.0) if np.isfinite(w) else 0.0)
    if sum(weights) <= 0:
        del raw; gc.collect()
        return {'subject_id': sub_id, 'status': 'no_weight',
                 'n_events': n_used}

    results = {}
    for cond in ('pre', 'ignition', 'post'):
        wpsds = weighted_mean_psd_per_channel(per_ev[cond], weights)
        if not wpsds:
            results[cond] = None
            continue
        try:
            peaks_df = fooof_on_weighted_psds(wpsds, ch_names, F0, fs)
            if len(peaks_df) > 0:
                enr = per_subject_enrichment(peaks_df, min_peaks=10)
                results[cond] = enr
                results[f'{cond}_n_peaks'] = len(peaks_df)
            else:
                results[cond] = None
        except Exception as e:
            results[cond] = None
            results[f'{cond}_error'] = str(e)[:100]

    del raw, per_ev; gc.collect()

    return {
        'subject_id': sub_id,
        'status': 'ok',
        'n_events': len(sub_events),
        'n_windows_used': int(n_used),
        'mean_weight': float(np.mean(weights)),
        **{f'{cond}_{k}': v for cond, res in results.items()
           if isinstance(res, dict) for k, v in res.items()},
        **{k: v for k, v in results.items() if not isinstance(v, dict)},
    }


# Globals set by _init_worker for use in worker_process_subject_sw.
_DATASET_GLOBAL = None
_CONDITION_GLOBAL = None
_SESSION_GLOBAL = None
_RELEASE_GLOBAL = None
_EVENTS_DF_GLOBAL = None
_WEIGHTS_LOOKUP_GLOBAL = None
_WINDOW_SEC_GLOBAL = None
_BUFFER_SEC_GLOBAL = None
_MIN_EVENTS_GLOBAL = None


def _init_worker(dataset, condition, session, release, events_df,
                  weights_lookup, window_sec, buffer_sec, min_events):
    """Pool initializer: set globals so workers don't need to pickle closures."""
    global _DATASET_GLOBAL, _CONDITION_GLOBAL, _SESSION_GLOBAL, _RELEASE_GLOBAL
    global _EVENTS_DF_GLOBAL, _WEIGHTS_LOOKUP_GLOBAL
    global _WINDOW_SEC_GLOBAL, _BUFFER_SEC_GLOBAL, _MIN_EVENTS_GLOBAL
    _DATASET_GLOBAL = dataset
    _CONDITION_GLOBAL = condition
    _SESSION_GLOBAL = session
    _RELEASE_GLOBAL = release
    _EVENTS_DF_GLOBAL = events_df
    _WEIGHTS_LOOKUP_GLOBAL = weights_lookup
    _WINDOW_SEC_GLOBAL = window_sec
    _BUFFER_SEC_GLOBAL = buffer_sec
    _MIN_EVENTS_GLOBAL = min_events


def worker_process_subject_sw(sub_id):
    """Pool worker: build loader from globals, then process subject."""
    _, make_loader = get_loader_for(_DATASET_GLOBAL, _CONDITION_GLOBAL,
                                     _SESSION_GLOBAL, _RELEASE_GLOBAL,
                                     composite=True)
    loader = make_loader(sub_id)
    if loader is None:
        return None
    return process_subject_sw(sub_id, loader, _EVENTS_DF_GLOBAL,
                               _WEIGHTS_LOOKUP_GLOBAL, _WINDOW_SEC_GLOBAL,
                               _BUFFER_SEC_GLOBAL, _MIN_EVENTS_GLOBAL)


def run_dataset_sw(dataset, condition=None, session='1', release='R1',
                    window_sec=20, buffer_sec=5.0, min_events=3):
    """SW window enrichment on composite-v2 events with template_rho weights."""
    sie_dir, make_loader = get_loader_for(dataset, condition, session, release,
                                           composite=True)
    ses_suffix = '' if session == '1' else f'_ses{session}'
    label = (f"{dataset}"
             f"{f'_{condition}' if condition else ''}"
             f"{f'_R{release[1:]}' if dataset == 'hbn' else ''}"
             f"{ses_suffix}")
    print(f"\n{'='*80}")
    print(f"SIE Window Enrichment SW: {label}")
    print(f"SIE dir: {sie_dir}, Window={window_sec}s, buffer={buffer_sec}s")
    print(f"{'='*80}\n")

    # Load events with MSC filter
    events_df = load_sie_events_for_dataset(sie_dir)
    if len(events_df) == 0:
        print(f"No events in {sie_dir}; abort")
        return pd.DataFrame()
    print(f"Events loaded: {len(events_df):,} from "
          f"{events_df['subject_id'].nunique()} subjects")

    # Load template_rho lookup
    quality_csv = os.path.join(QUALITY_BASE, f'per_event_quality_{sie_dir}.csv')
    if not os.path.isfile(quality_csv):
        print(f"Quality CSV not found: {quality_csv}; abort")
        return pd.DataFrame()
    qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
    qual['t0r'] = qual['t0_net'].round(3)
    weights_lookup = {(sid, t): r for sid, t, r in
                      zip(qual['subject_id'], qual['t0r'], qual['template_rho'])}
    print(f"Template_rho available for "
          f"{qual['subject_id'].nunique()} subjects, {len(qual)} events")

    event_counts = events_df.groupby('subject_id').size()
    subjects = event_counts[event_counts >= min_events].index.tolist()
    print(f"Subjects with ≥{min_events} events: {len(subjects)}")

    n_workers = int(os.environ.get('SIE_WORKERS', 4))
    print(f"Pool workers: {n_workers}")
    results = []
    t_start = time.time()
    from multiprocessing import Pool
    with Pool(n_workers, initializer=_init_worker,
              initargs=(dataset, condition, session, release, events_df,
                        weights_lookup, window_sec, buffer_sec, min_events)) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_process_subject_sw,
                                                         subjects)):
            if result:
                results.append(result)
                if (i + 1) % 10 == 0 or i == 0:
                    n_ok = sum(1 for r in results if r.get('status') == 'ok')
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                    print(f"  [{i+1}/{len(subjects)}] {result.get('subject_id')}: "
                          f"{result.get('status','?')}  "
                          f"({n_ok} ok, {rate:.1f}/min)")

    results_df = pd.DataFrame(results)
    out_name = f'sie_window_enrichment_sw_{label}.csv'
    out_path = os.path.join(OUTPUT_DIR, out_name)
    if os.path.exists(out_path):
        os.rename(out_path, out_path + '.bak')
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    if len(results_df) > 0 and 'status' in results_df.columns:
        print(f"  N subjects processed: {len(results_df)}")
        print(f"  N with status=ok: {(results_df['status']=='ok').sum()}")
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['eegmmidb', 'lemon', 'dortmund', 'chbmp',
                                  'hbn', 'tdbrain'])
    parser.add_argument('--condition', default=None)
    parser.add_argument('--session', default='1')
    parser.add_argument('--release', default='R1')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--buffer', type=float, default=5.0)
    parser.add_argument('--min-events', type=int, default=2)
    args = parser.parse_args()
    run_dataset_sw(dataset=args.dataset, condition=args.condition,
                   session=args.session, release=args.release,
                   window_sec=args.window, buffer_sec=args.buffer,
                   min_events=args.min_events)


if __name__ == '__main__':
    main()
