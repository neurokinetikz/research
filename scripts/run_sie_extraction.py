#!/usr/bin/env python3
"""
Schumann Ignition Event (SIE) Detection -- Batch Runner
========================================================

Runs SIE detection across all research-grade EEG datasets using the same
GCP infrastructure as run_f0_760_extraction.py.

Uses detect_ignitions_session() with the notebook's exact parameters:
  - f₀=7.6 Hz, 9 phi-lattice harmonics, fooof_hybrid method
  - Per-harmonic bandwidths and FOOOF frequency ranges

Dataset loaders are copied verbatim from run_f0_760_extraction.py.

Usage:
    python scripts/run_sie_extraction.py --dataset lemon --parallel 28
    python scripts/run_sie_extraction.py --dataset dortmund --condition EO-pre --parallel 28
    python scripts/run_sie_extraction.py --dataset hbn --release R1 --parallel 28
"""

import os
import sys
import time
import argparse
import warnings
import gc
import logging
import tempfile
import shutil
from glob import glob as globfn_top

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.mne_to_ignition import (
    detect_ignitions_mne, summarize_session, STANDARD_1020,
)

# Suppress MNE and matplotlib noise
mne.set_log_level('ERROR')
import matplotlib
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# =========================================================================
# CONSTANTS
# =========================================================================
TARGET_FS = 250.0
FILTER_LO = 1.0

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')


# =========================================================================
# DATASET LOADERS (verbatim from run_f0_760_extraction.py)
# =========================================================================

def load_eegmmidb(sub_id, data_dir='/Volumes/T9/eegmmidb'):
    raws = []
    for run in range(1, 15):
        edf_path = os.path.join(data_dir, f'S{sub_id:03d}', f'S{sub_id:03d}R{run:02d}.edf')
        if os.path.exists(edf_path):
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                raws.append(raw)
            except Exception:
                continue
    if not raws:
        return None
    for i, raw in enumerate(raws):
        if abs(raw.info['sfreq'] - 160) > 0.1:
            raws[i] = raw.resample(160, verbose=False)
    raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
    raw.filter(FILTER_LO, 59, verbose=False)
    return raw


def load_lemon(sub_id, data_dir='/Volumes/T9/lemon_data/eeg_preprocessed/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed',
               condition='EC'):
    path = os.path.join(data_dir, f'{sub_id}_{condition}.set')
    if not os.path.isfile(path):
        return None
    try:
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    except Exception:
        return None
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(eeg_picks) == 0:
        return None
    raw.pick(eeg_picks)
    nyq = raw.info['sfreq'] / 2.0
    h_freq = min(59, nyq - 1)
    if nyq > 52:
        raw.notch_filter(50, verbose=False)
    raw.filter(FILTER_LO, h_freq, verbose=False)
    return raw


def load_dortmund(sub_id, data_dir='/Volumes/T9/dortmund_data_dl',
                  task='EyesClosed', acq='pre', ses='1'):
    edf_path = os.path.join(data_dir, sub_id, f'ses-{ses}', 'eeg',
                            f'{sub_id}_ses-{ses}_task-{task}_acq-{acq}_eeg.edf')
    if not os.path.isfile(edf_path):
        return None
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None
    raw.pick_types(eeg=True, exclude='bads')
    if raw.info['sfreq'] > TARGET_FS:
        raw.resample(TARGET_FS, verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.filter(FILTER_LO, 59, verbose=False)
    return raw


def load_chbmp(sub_id, data_dir='/Volumes/T9/CHBMP/BIDS_dataset'):
    edf_path = os.path.join(data_dir, sub_id, 'ses-V01', 'eeg',
                            f'{sub_id}_ses-V01_task-protmap_eeg.edf')
    if not os.path.isfile(edf_path):
        return None
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False,
                               misc=False, exclude='bads')
    raw.pick(eeg_picks)
    raw.filter(FILTER_LO, 59, verbose=False)
    try:
        events_path = os.path.join(data_dir, sub_id, 'ses-V01', 'eeg',
                                   f'{sub_id}_ses-V01_task-protmap_events.tsv')
        if os.path.isfile(events_path):
            events = pd.read_csv(events_path, sep='\t')
            if 'value' in events.columns:
                hv = events[events['value'] == 67]
                hv_onset = hv['onset'].values[0] if len(hv) > 0 else float('inf')
                ec_onsets = events[(events['value'] == 65) & (events['onset'] < hv_onset)]['onset'].values
                all_onsets = sorted(events[events['onset'] < hv_onset]['onset'].values)
                raws = []
                for onset in ec_onsets:
                    later = [t for t in all_onsets if t > onset]
                    offset = later[0] if later else hv_onset
                    if offset - onset >= 2.0:
                        try:
                            seg = raw.copy().crop(tmin=onset, tmax=min(offset, raw.times[-1]))
                            if len(seg.times) > 0:
                                raws.append(seg)
                        except Exception:
                            continue
                if raws:
                    raw = mne.concatenate_raws(raws)
    except Exception:
        raw.crop(tmin=0, tmax=min(1000.0, raw.times[-1]))
    return raw


def load_tdbrain(sub_id, data_dir='/Volumes/T9/tdbrain/derivatives',
                  condition='EC', session=None):
    import json as json_mod
    task = 'restEC' if condition == 'EC' else 'restEO'
    sessions = [session] if session else ['1', '2']
    csv_path = None
    json_path = None
    for ses in sessions:
        candidate = os.path.join(data_dir, sub_id, f'ses-{ses}', 'eeg',
                                 f'{sub_id}_ses-{ses}_task-{task}_eeg.csv')
        if os.path.isfile(candidate):
            csv_path = candidate
            json_path = candidate.replace('_eeg.csv', '_eeg.json')
            break
    if csv_path is None:
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if len(df) < 1000:
        return None

    df.columns = [c.strip() for c in df.columns]

    standard_eeg = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                    'FC3', 'FCz', 'FC4',
                    'T7', 'T3', 'C3', 'Cz', 'C4', 'T8', 'T4',
                    'CP3', 'CPz', 'CP4',
                    'P7', 'T5', 'P3', 'Pz', 'P4', 'P8', 'T6',
                    'O1', 'Oz', 'O2']
    eeg_cols = [c for c in df.columns if c in standard_eeg]

    if len(eeg_cols) < 10:
        return None

    fs = 500.0
    if json_path and os.path.isfile(json_path):
        try:
            with open(json_path) as jf:
                meta = json_mod.load(jf)
            fs = float(meta.get('SamplingFrequency', 500.0))
        except Exception:
            pass

    data = df[eeg_cols].values.T.astype(np.float64)
    data = data * 1e-9  # nanovolts to Volts

    ch_names = [c for c in eeg_cols]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)

    if raw.info['sfreq'] > TARGET_FS:
        raw.resample(TARGET_FS, verbose=False)

    nyq = raw.info['sfreq'] / 2.0
    if nyq > 52:
        raw.notch_filter(50, verbose=False)
    h_freq = min(59, nyq - 1)
    raw.filter(FILTER_LO, h_freq, verbose=False)

    return raw


def load_hbn(set_path):
    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception:
        return None
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(eeg_picks) == 0:
        return None
    raw.pick(eeg_picks)
    if raw.info['sfreq'] > TARGET_FS:
        raw.resample(TARGET_FS, verbose=False)
    raw.filter(FILTER_LO, 59, verbose=False)
    return raw


# =========================================================================
# WORKER
# =========================================================================

_WORKER_LOADER_NAME = None
_WORKER_LOADER_KWARGS = None
_WORKER_OUT_DIR = None


def _init_worker(loader_name, loader_kwargs, out_dir):
    global _WORKER_LOADER_NAME, _WORKER_LOADER_KWARGS, _WORKER_OUT_DIR
    _WORKER_LOADER_NAME = loader_name
    _WORKER_LOADER_KWARGS = loader_kwargs or {}
    _WORKER_OUT_DIR = out_dir


def _sie_one_subject(args):
    """Worker function for parallel SIE detection."""
    sub_id, load_arg = args
    events_path = os.path.join(_WORKER_OUT_DIR, f'{sub_id}_sie_events.csv')
    summary_path = os.path.join(_WORKER_OUT_DIR, f'{sub_id}_sie_summary.csv')

    if os.path.exists(summary_path):
        return {'subject_id': sub_id, 'status': 'skipped', 'n_events': 0}

    # Load raw EEG
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if _WORKER_LOADER_NAME == 'eegmmidb':
            raw = load_eegmmidb(int(sub_id[1:]))
        elif _WORKER_LOADER_NAME == 'lemon':
            raw = load_lemon(sub_id, condition=_WORKER_LOADER_KWARGS.get('condition', 'EC'))
        elif _WORKER_LOADER_NAME == 'dortmund':
            raw = load_dortmund(sub_id, task=_WORKER_LOADER_KWARGS.get('task', 'EyesClosed'),
                                acq=_WORKER_LOADER_KWARGS.get('acq', 'pre'),
                                ses=_WORKER_LOADER_KWARGS.get('ses', '1'))
        elif _WORKER_LOADER_NAME == 'chbmp':
            raw = load_chbmp(sub_id)
        elif _WORKER_LOADER_NAME == 'hbn':
            raw = load_hbn(load_arg)
        elif _WORKER_LOADER_NAME == 'tdbrain':
            raw = load_tdbrain(sub_id,
                               condition=_WORKER_LOADER_KWARGS.get('condition', 'EC'),
                               session=_WORKER_LOADER_KWARGS.get('session', None))
        else:
            return {'subject_id': sub_id, 'status': 'error', 'n_events': 0}

    if raw is None:
        return {'subject_id': sub_id, 'status': 'no_data', 'n_events': 0}

    fs = raw.info['sfreq']
    duration = raw.n_times / fs
    n_channels = len(raw.ch_names)

    # Check minimum recording length (nperseg_sec=100 needs at least 100s)
    if duration < 120:
        del raw; gc.collect()
        return {'subject_id': sub_id, 'status': 'too_short', 'n_events': 0}

    # Temp directory for detection outputs (cleaned up after)
    tmp_dir = os.path.join(tempfile.gettempdir(), f'sie_{sub_id}')
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        dataset = _WORKER_LOADER_KWARGS.get('_dataset_label', _WORKER_LOADER_NAME)
        condition = _WORKER_LOADER_KWARGS.get('condition', 'EC')

        result, ign_windows = detect_ignitions_mne(
            raw,
            session_name=sub_id,
            out_dir=tmp_dir,
        )

        # Save per-event CSV
        events_df = result.get('events', pd.DataFrame())
        if events_df is not None and not events_df.empty:
            # Add metadata columns
            events_df = events_df.copy()
            events_df.insert(0, 'subject_id', sub_id)
            events_df.insert(1, 'dataset', dataset)
            events_df.insert(2, 'condition', condition)

            # Add harmonic ratio columns
            for col in ['sr1', 'sr3', 'sr4', 'sr5', 'sr6']:
                if col in events_df.columns:
                    events_df[col] = pd.to_numeric(events_df[col], errors='coerce')
            if 'sr1' in events_df.columns and 'sr3' in events_df.columns:
                events_df['sr3/sr1'] = events_df['sr3'] / events_df['sr1']
            if 'sr1' in events_df.columns and 'sr5' in events_df.columns:
                events_df['sr5/sr1'] = events_df['sr5'] / events_df['sr1']
            if 'sr3' in events_df.columns and 'sr5' in events_df.columns:
                events_df['sr5/sr3'] = events_df['sr5'] / events_df['sr3']
            if 'sr4' in events_df.columns and 'sr6' in events_df.columns:
                events_df['sr6/sr4'] = events_df['sr6'] / events_df['sr4']

            events_df.to_csv(events_path, index=False)

        n_events = len(events_df) if events_df is not None and not events_df.empty else 0

        # Save per-subject summary
        summary_row = summarize_session(
            result, sub_id, dataset, condition, n_channels, fs, duration)
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    except Exception as e:
        del raw; gc.collect()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {'subject_id': sub_id, 'status': 'error', 'n_events': 0,
                'error': str(e)[:200]}

    # Cleanup temp directory
    shutil.rmtree(tmp_dir, ignore_errors=True)
    del raw; gc.collect()

    return {
        'subject_id': sub_id,
        'status': 'ok',
        'n_events': n_events,
        'duration_sec': round(duration, 1),
    }


# =========================================================================
# PROCESS SUBJECTS
# =========================================================================

def process_subjects(subjects, loader_name, out_dir, dataset_label,
                     loader_kwargs=None, parallel=1):
    global _WORKER_LOADER_NAME, _WORKER_LOADER_KWARGS, _WORKER_OUT_DIR

    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    loader_kwargs = loader_kwargs or {}
    loader_kwargs['_dataset_label'] = dataset_label

    log.info(f"\n{dataset_label} (SIE detection): {len(subjects)} subjects")
    log.info(f"  Output: {out_dir}")
    if parallel > 1:
        log.info(f"  Parallel workers: {parallel}")

    if parallel > 1:
        from multiprocessing import Pool
        with Pool(parallel, initializer=_init_worker,
                  initargs=(loader_name, loader_kwargs, out_dir)) as pool:
            results = pool.map(_sie_one_subject, subjects)
        summary_rows = results
        for r in results:
            if r['status'] == 'ok':
                log.info(f"  {r['subject_id']}: {r['n_events']} events")
    else:
        _WORKER_LOADER_NAME = loader_name
        _WORKER_LOADER_KWARGS = loader_kwargs
        _WORKER_OUT_DIR = out_dir
        summary_rows = []
        for i, (sub_id, load_arg) in enumerate(subjects):
            result = _sie_one_subject((sub_id, load_arg))
            summary_rows.append(result)

            if result['status'] == 'ok':
                log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: {result['n_events']} events")
            elif result['status'] == 'skipped':
                log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: skipping")
            elif result['status'] == 'too_short':
                log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: too short (<120s)")
            else:
                log.info(f"  [{i+1}/{len(subjects)}] {sub_id}: {result['status']}")

            if (i + 1) % 20 == 0:
                pd.DataFrame(summary_rows).to_csv(
                    os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    total = time.time() - t_start
    n_ok = (summary_df['status'] == 'ok').sum() if len(summary_df) > 0 else 0
    log.info(f"\n  {dataset_label} DONE: {n_ok}/{len(subjects)} in {total/60:.1f} min")
    if n_ok > 0:
        ok = summary_df[summary_df['status'] == 'ok']
        total_events = ok['n_events'].sum()
        log.info(f"  Total events: {total_events:,}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SIE detection across research-grade EEG datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eegmmidb', 'lemon', 'dortmund', 'chbmp', 'hbn', 'tdbrain'])
    parser.add_argument('--release', type=str, default='R1',
                        help='HBN release (R1-R6 or all)')
    parser.add_argument('--condition', type=str, default=None,
                        help='Condition (EO for LEMON; EC-pre/EO-pre/EC-post/EO-post for Dortmund)')
    parser.add_argument('--session', type=str, default='1',
                        help='Session number for Dortmund (1 or 2)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers (default: 1 = sequential)')
    args = parser.parse_args()

    n_parallel = args.parallel

    if args.dataset == 'eegmmidb':
        data_dir = '/Volumes/T9/eegmmidb'
        subs = sorted(set(
            d for d in os.listdir(data_dir)
            if d.startswith('S') and os.path.isdir(os.path.join(data_dir, d))))
        subjects = [(f'S{int(s[1:]):03d}', None) for s in subs]
        out_dir = os.path.join(OUTPUT_BASE, 'eegmmidb')
        process_subjects(subjects, 'eegmmidb', out_dir, 'EEGMMIDB',
                         parallel=n_parallel)

    elif args.dataset == 'lemon':
        data_dir = '/Volumes/T9/lemon_data/eeg_preprocessed/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed'
        cond = args.condition or 'EC'
        files = sorted(globfn_top(os.path.join(data_dir, f'sub-*_{cond}.set')))
        subjects = [(os.path.basename(f).replace(f'_{cond}.set', ''), None) for f in files]
        suffix = f'_EO' if cond == 'EO' else ''
        out_dir = os.path.join(OUTPUT_BASE, f'lemon{suffix}')
        process_subjects(subjects, 'lemon', out_dir, f'LEMON {cond}',
                         loader_kwargs={'condition': cond}, parallel=n_parallel)

    elif args.dataset == 'dortmund':
        data_dir = '/Volumes/T9/dortmund_data_dl'
        ses = args.session
        ses_dir = f'ses-{ses}'
        subs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and
                       os.path.isdir(os.path.join(data_dir, d, ses_dir, 'eeg'))])
        subjects = [(s, None) for s in subs]
        cond = args.condition or 'EC-pre'
        task = 'EyesOpen' if cond.startswith('EO') else 'EyesClosed'
        acq = 'post' if cond.endswith('post') else 'pre'
        suffix = '' if (cond == 'EC-pre' and ses == '1') else f'_{cond.replace("-", "_")}'
        ses_suffix = f'_ses2' if ses == '2' else ''
        out_dir = os.path.join(OUTPUT_BASE, f'dortmund{suffix}{ses_suffix}')
        process_subjects(subjects, 'dortmund', out_dir, f'Dortmund {cond} ses-{ses}',
                         loader_kwargs={'task': task, 'acq': acq, 'ses': ses},
                         parallel=n_parallel)

    elif args.dataset == 'chbmp':
        data_dir = '/Volumes/T9/CHBMP/BIDS_dataset'
        pattern = os.path.join(data_dir, 'sub-CBM*', 'ses-V01', 'eeg', '*_eeg.edf')
        edf_files = sorted(globfn_top(pattern))
        seen = set()
        subjects = []
        for f in edf_files:
            for p in f.split(os.sep):
                if p.startswith('sub-CBM') and p not in seen:
                    subjects.append((p, None))
                    seen.add(p)
                    break
        out_dir = os.path.join(OUTPUT_BASE, 'chbmp')
        process_subjects(subjects, 'chbmp', out_dir, 'CHBMP',
                         parallel=n_parallel)

    elif args.dataset == 'hbn':
        releases = ['R1', 'R2', 'R3', 'R4', 'R6'] if args.release.lower() == 'all' else [args.release]
        for release in releases:
            release_dir = os.path.join('/Volumes/T9/hbn_data', f'cmi_bids_{release}')
            pattern = os.path.join(release_dir, 'sub-*', 'eeg', '*RestingState_eeg.set')
            files = sorted(globfn_top(pattern))
            seen = set()
            subjects = []
            for f in files:
                sub_id = os.path.basename(f).split('_task-')[0]
                if sub_id not in seen:
                    subjects.append((sub_id, f))
                    seen.add(sub_id)
            out_dir = os.path.join(OUTPUT_BASE, f'hbn_{release}')
            process_subjects(subjects, 'hbn', out_dir, f'HBN {release}',
                             parallel=n_parallel)

    elif args.dataset == 'tdbrain':
        data_dir = '/Volumes/T9/tdbrain/derivatives'
        cond = args.condition or 'EC'
        ses = args.session
        task = 'restEC' if cond == 'EC' else 'restEO'
        sessions_to_try = [ses] if ses and ses != '1' else ['1', '2']
        subs = sorted(set(
            d for d in os.listdir(data_dir)
            if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))))
        subjects = []
        for s in subs:
            for ses_try in sessions_to_try:
                csv = os.path.join(data_dir, s, f'ses-{ses_try}', 'eeg',
                                   f'{s}_ses-{ses_try}_task-{task}_eeg.csv')
                if os.path.isfile(csv):
                    subjects.append((s, None))
                    break
        suffix = '_EO' if cond == 'EO' else ''
        ses_suffix = f'_ses{ses}' if ses and ses != '1' else ''
        out_dir = os.path.join(OUTPUT_BASE, f'tdbrain{suffix}{ses_suffix}')
        process_subjects(subjects, 'tdbrain', out_dir,
                         f'TDBRAIN {cond} ses-{ses or "1"}',
                         loader_kwargs={'condition': cond, 'session': ses},
                         parallel=n_parallel)


if __name__ == '__main__':
    main()
