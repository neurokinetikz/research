#!/usr/bin/env python3
"""
B60 (part 1) — Source localization on HBN (5-9 yr females).

B54 found the posterior-α SIE effect concentrates in 5-9 yr girls
(+0.95 contrast, p=0.001). B49 localized the LEMON adult effect to
posterior-temporoparietal cortex. Question: is the same anatomy active
in young girls?

Pipeline parallel to B49 but for HBN EGI-128 subjects. Applies GSN-
HydroCel-128 montage; uses fsaverage template BEM + src (same as B49).

Subject filter: HBN participants where sex == 'F' AND age in [5, 10).
Across HBN R1 + R2 + R3 + R4 + R6.
"""
from __future__ import annotations
import os
import sys
import glob as _glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_hbn

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'source')
STC_DIR_HBN = os.path.join(OUT_DIR, 'stcs_hbn_girls_5_9')
EVENTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
QUALITY_ROOT = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                             'schumann', 'images', 'quality')
HBN_DATA = '/Volumes/T9/hbn_data'
os.makedirs(STC_DIR_HBN, exist_ok=True)

SR1_BAND = (7.0, 8.3)
EVENT_WIN = (-2.0, 4.0)
BASELINE_WIN_N = 40
BASELINE_WIN_DUR = 6.0
EVENT_LAG = 1.0
METHOD = 'sLORETA'
LAMBDA2 = 1.0 / 9.0

_fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
SUBJECTS_DIR = os.path.dirname(_fs_dir)
FSAVG_SRC = os.path.join(_fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
FSAVG_BEM = os.path.join(_fs_dir, 'bem',
                          'fsaverage-5120-5120-5120-bem-sol.fif')
TRANS = 'fsaverage'


def _prep_raw_hbn(raw):
    """Apply GSN-HydroCel-128 montage. HBN channels are E1..E128 + Cz."""
    # Try the 129 montage first (includes Cz); fallback to 128
    for mname in ('GSN-HydroCel-129', 'GSN-HydroCel-128'):
        try:
            montage = mne.channels.make_standard_montage(mname)
            raw_c = raw.copy()
            raw_c.set_montage(montage, match_case=False, on_missing='ignore',
                               verbose=False)
            pos = raw_c.get_montage().get_positions()['ch_pos']
            keep = [ch for ch in raw_c.ch_names
                    if ch in pos and np.all(np.isfinite(pos.get(ch)))]
            if len(keep) >= 60:
                raw_c.pick(keep)
                raw_c.set_eeg_reference('average', projection=True,
                                         verbose=False)
                return raw_c
        except Exception:
            continue
    return None


def _compute_band_power_stc(inv, epochs, band):
    freqs = np.arange(band[0], band[1] + 0.25, 0.25)
    stcs = mne.minimum_norm.compute_source_psd_epochs(
        epochs, inv, method=METHOD, lambda2=LAMBDA2,
        fmin=band[0], fmax=band[1], pick_ori=None,
        bandwidth=1.0, adaptive=False, low_bias=True,
        return_generator=False, verbose=False)
    if len(stcs) == 0:
        return None
    arr = np.array([np.mean(s.data, axis=1) for s in stcs])
    mean_power = np.mean(arr, axis=0)
    tmpl = stcs[0].copy()
    tmpl.data = mean_power[:, np.newaxis]
    tmpl.tmin = 0.0
    return tmpl


def process_subject(args):
    release, sub_id, set_path = args
    events_path = os.path.join(EVENTS_ROOT, f'hbn_{release}',
                                f'{sub_id}_sie_events.csv')
    if not os.path.isfile(events_path):
        return None
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) == 0:
        return None

    # Q4 filter from per-release template_rho
    quality_path = os.path.join(QUALITY_ROOT,
                                 f'per_event_quality_hbn_{release}.csv')
    try:
        qual = pd.read_csv(quality_path).dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4,
                                 labels=['Q1','Q2','Q3','Q4'])
        q4 = qual[(qual['subject_id']==sub_id) & (qual['rho_q']=='Q4')]
        q4_times = set(q4['t0_net'].round(3).values)
        events['t0_round'] = events['t0_net'].round(3)
        events = events[events['t0_round'].isin(q4_times)]
    except Exception:
        return None
    if len(events) < 1:
        return None

    try:
        raw = load_hbn(set_path)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 100:
        return None
    try:
        raw = _prep_raw_hbn(raw)
    except Exception:
        return None
    if raw is None or len(raw.ch_names) < 60:
        return None
    t_end = raw.times[-1]

    ev_times = [float(ev['t0_net']) + EVENT_LAG
                 for _, ev in events.iterrows()
                 if EVENT_WIN[0] + float(ev['t0_net']) > 2
                 and EVENT_WIN[1] + float(ev['t0_net']) < t_end - 2]
    if not ev_times:
        return None
    ev_array = np.array([[int(t * fs), 0, 1] for t in ev_times])
    ev_epochs = mne.Epochs(raw, ev_array, event_id={'Q4':1},
                            tmin=EVENT_WIN[0], tmax=EVENT_WIN[1],
                            baseline=None, preload=True, proj=True,
                            verbose=False)
    if len(ev_epochs) == 0:
        return None

    rng = np.random.default_rng(42 + hash(sub_id) % 10000)
    ev_set = set([int(t) for t in ev_times])
    cand = [t for t in range(int(BASELINE_WIN_DUR)+5,
                              int(t_end-BASELINE_WIN_DUR-5))
            if not any(abs(t-e) < 10 for e in ev_set)]
    if len(cand) < 5:
        return None
    bl_times = rng.choice(cand, size=min(BASELINE_WIN_N, len(cand)),
                           replace=False)
    bl_array = np.array([[int(t*fs), 0, 2] for t in bl_times])
    bl_epochs = mne.Epochs(raw, bl_array, event_id={'BL':2},
                            tmin=-BASELINE_WIN_DUR/2, tmax=BASELINE_WIN_DUR/2,
                            baseline=None, preload=True, proj=True,
                            verbose=False)
    if len(bl_epochs) < 5:
        return None

    try:
        noise_cov = mne.compute_covariance(bl_epochs, tmin=-2, tmax=2,
                                            method='shrunk', rank='info',
                                            verbose=False)
    except Exception:
        return None

    try:
        fwd = mne.make_forward_solution(raw.info, trans=TRANS,
                                         src=FSAVG_SRC, bem=FSAVG_BEM,
                                         eeg=True, mindist=5.0, n_jobs=1,
                                         verbose=False)
    except Exception:
        return None

    try:
        inv = mne.minimum_norm.make_inverse_operator(
            raw.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)
    except Exception:
        return None

    ev_stc = _compute_band_power_stc(inv, ev_epochs, SR1_BAND)
    bl_stc = _compute_band_power_stc(inv, bl_epochs, SR1_BAND)
    if ev_stc is None or bl_stc is None:
        return None

    ratio_data = (ev_stc.data + 1e-25) / (bl_stc.data + 1e-25)
    ratio_stc = ev_stc.copy()
    ratio_stc.data = ratio_data
    ratio_stc.save(os.path.join(STC_DIR_HBN,
                                 f'{sub_id}_Q4_SR1_ratio'),
                    overwrite=True, verbose=False)
    return sub_id


def build_tasks():
    """Build tasks: only subjects who are female, aged 5-9, with Q4 events."""
    tasks = []
    for release in ['R1','R2','R3','R4','R6']:
        p = os.path.join(HBN_DATA, f'cmi_bids_{release}', 'participants.tsv')
        if not os.path.isfile(p): continue
        meta = pd.read_csv(p, sep='\t').rename(
            columns={'participant_id':'subject_id'})
        meta['age'] = pd.to_numeric(meta['age'], errors='coerce')
        # Filter: female, age 5-9
        sel = meta[(meta['sex']=='F') & (meta['age']>=5) & (meta['age']<10)]
        release_dir = os.path.join(HBN_DATA, f'cmi_bids_{release}')
        for _, r in sel.iterrows():
            sid = r['subject_id']
            sp = os.path.join(release_dir, sid, 'eeg',
                              f'{sid}_task-RestingState_eeg.set')
            if not os.path.isfile(sp):
                cand = _glob.glob(os.path.join(release_dir, sid, 'eeg',
                                                f'{sid}_task-RestingState_eeg.set'))
                if not cand: continue
                sp = cand[0]
            tasks.append((release, sid, sp))
    return tasks


def main():
    tasks = build_tasks()
    print(f"HBN 5-9 yr female tasks: {len(tasks)}")
    if len(tasks) < 10:
        print("Too few subjects, aborting")
        return
    n_workers = int(os.environ.get('SIE_WORKERS',
                                    min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    ok_ids = [r for r in results if r is not None]
    print(f"Successful STCs: {len(ok_ids)}")


if __name__ == '__main__':
    main()
