#!/usr/bin/env python3
"""
B49 — Source-space localization of the Q4 SIE posterior-α generator.

Goal: anatomically name the cortical region producing the posterior α
signature observed at canonical (template_rho Q4) ignition events.

Pipeline (per LEMON subject):
  1. Load raw EEG with standard_1020 montage.
  2. Build forward model against fsaverage template.
  3. Compute noise covariance from random baseline segments.
  4. Make inverse operator (sLORETA).
  5. For each Q4 event: compute event-window source-PSD and baseline-
     window source-PSD. Take SR1-band (7.0-8.3 Hz) band-power per vertex.
  6. Compute event/baseline power ratio per vertex, average across events.

Group: morph each subject's ratio map to fsaverage, average across
subjects. Extract top-N Desikan-Killiany labels ranked by mean ratio.

Output:
  - Per-subject ratio STCs saved as .stc
  - Group average ratio map saved as .stc (fsaverage)
  - ROI ranking CSV (label × mean ratio)
  - Cortical surface figure showing group map
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'source')
STC_DIR = os.path.join(OUT_DIR, 'stcs')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality',
                            'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STC_DIR, exist_ok=True)

SR1_BAND = (7.0, 8.3)
EVENT_WIN = (-2.0, 4.0)       # seconds relative to t0_net
BASELINE_WIN_N = 40            # number of random baseline windows
BASELINE_WIN_DUR = 6.0         # seconds
EVENT_LAG = 1.0                # lag to account for ignition after t0_net
METHOD = 'sLORETA'
SNR = 3.0
LAMBDA2 = 1.0 / SNR ** 2

# Fetch fsaverage (cached after first download)
_fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
SUBJECTS_DIR = os.path.dirname(_fs_dir)
FSAVG_SRC = os.path.join(_fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
FSAVG_BEM = os.path.join(_fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
TRANS = 'fsaverage'  # identity for template-head alignment


def _prep_raw(raw):
    """Apply standard_1020 montage, keep only channels with positions."""
    montage = mne.channels.make_standard_montage('standard_1020')
    raw = raw.copy()
    raw.set_montage(montage, match_case=False, on_missing='ignore',
                     verbose=False)
    keep = [ch for ch in raw.ch_names
            if raw.get_montage().get_positions()['ch_pos'].get(ch) is not None
            and np.all(np.isfinite(
                raw.get_montage().get_positions()['ch_pos'].get(ch,
                                                                 np.full(3, np.nan))))]
    raw.pick(keep)
    raw.set_eeg_reference('average', projection=True, verbose=False)
    return raw


def _compute_band_power_stc(raw, inv, epochs, band):
    """Compute mean band-power STC across epochs using cross-spectral approach.
    Returns one STC whose .data is (n_vertices, 1) with mean band power."""
    freqs = np.arange(band[0], band[1] + 0.25, 0.25)
    stcs = mne.minimum_norm.compute_source_psd_epochs(
        epochs, inv, method=METHOD, lambda2=LAMBDA2,
        fmin=band[0], fmax=band[1], pick_ori=None,
        bandwidth=1.0, adaptive=False, low_bias=True,
        return_generator=False, verbose=False)
    if len(stcs) == 0:
        return None
    # Each stc.data has shape (n_vertices, n_freqs); average over freqs & epochs
    arr = np.array([np.mean(s.data, axis=1) for s in stcs])  # (n_ep, n_verts)
    mean_power = np.mean(arr, axis=0)  # (n_verts,)
    # Wrap as an STC for morphing
    tmpl = stcs[0].copy()
    tmpl.data = mean_power[:, np.newaxis]
    tmpl.tmin = 0.0
    return tmpl


def process_subject(sub_id):
    try:
        events = pd.read_csv(os.path.join(EVENTS_DIR,
                                           f'{sub_id}_sie_events.csv'))
    except Exception:
        return None
    events = events.dropna(subset=['t0_net'])
    if len(events) == 0:
        return None
    # Q4-filter by template_rho
    try:
        qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4,
                                 labels=['Q1','Q2','Q3','Q4'])
        q4 = qual[(qual['subject_id'] == sub_id) & (qual['rho_q'] == 'Q4')]
        q4_times = set(q4['t0_net'].round(3).values)
        events['t0_round'] = events['t0_net'].round(3)
        events = events[events['t0_round'].isin(q4_times)]
    except Exception:
        return None
    if len(events) < 1:
        return None

    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 100:
        return None
    try:
        raw = _prep_raw(raw)
    except Exception:
        return None
    if len(raw.ch_names) < 20:
        return None
    t_end = raw.times[-1]

    # Build event epochs
    ev_times = [float(ev['t0_net']) + EVENT_LAG
                 for _, ev in events.iterrows()
                 if EVENT_WIN[0] + float(ev['t0_net']) > 2
                 and EVENT_WIN[1] + float(ev['t0_net']) < t_end - 2]
    if not ev_times:
        return None
    ev_array = np.array([[int(t * fs), 0, 1] for t in ev_times])
    ev_epochs = mne.Epochs(raw, ev_array, event_id={'Q4': 1},
                            tmin=EVENT_WIN[0], tmax=EVENT_WIN[1],
                            baseline=None, preload=True, proj=True,
                            verbose=False)
    if len(ev_epochs) == 0:
        return None

    # Baseline epochs: random windows at least 10s away from events
    rng = np.random.default_rng(42 + hash(sub_id) % 10000)
    ev_set = set([int(t) for t in ev_times])
    cand = np.arange(int(BASELINE_WIN_DUR) + 5,
                      int(t_end - BASELINE_WIN_DUR - 5))
    cand = [t for t in cand if not any(abs(t - e) < 10 for e in ev_set)]
    if len(cand) < 5:
        return None
    bl_times = rng.choice(cand, size=min(BASELINE_WIN_N, len(cand)),
                           replace=False)
    bl_array = np.array([[int(t * fs), 0, 2] for t in bl_times])
    bl_epochs = mne.Epochs(raw, bl_array, event_id={'BL': 2},
                            tmin=-BASELINE_WIN_DUR / 2,
                            tmax=BASELINE_WIN_DUR / 2,
                            baseline=None, preload=True, proj=True,
                            verbose=False)
    if len(bl_epochs) < 5:
        return None

    # Noise covariance from baseline epochs
    try:
        noise_cov = mne.compute_covariance(bl_epochs, tmin=-2, tmax=2,
                                            method='shrunk',
                                            rank='info', verbose=False)
    except Exception:
        return None

    # Forward model (cached per subject by MNE)
    try:
        fwd = mne.make_forward_solution(raw.info, trans=TRANS,
                                         src=FSAVG_SRC, bem=FSAVG_BEM,
                                         eeg=True, mindist=5.0,
                                         n_jobs=1, verbose=False)
    except Exception:
        return None

    # Inverse operator
    try:
        inv = mne.minimum_norm.make_inverse_operator(
            raw.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)
    except Exception:
        return None

    # Event-window and baseline-window source PSD averaged across epochs
    ev_stc = _compute_band_power_stc(raw, inv, ev_epochs, SR1_BAND)
    bl_stc = _compute_band_power_stc(raw, inv, bl_epochs, SR1_BAND)
    if ev_stc is None or bl_stc is None:
        return None

    # Event/baseline ratio per vertex
    ratio_data = (ev_stc.data + 1e-25) / (bl_stc.data + 1e-25)
    ratio_stc = ev_stc.copy()
    ratio_stc.data = ratio_data

    # Save per-subject ratio STC
    ratio_stc.save(os.path.join(STC_DIR, f'{sub_id}_Q4_SR1_ratio'),
                   overwrite=True, verbose=False)
    return sub_id


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=3)]
    tasks = [r['subject_id'] for _, r in ok.iterrows()
              if os.path.isfile(os.path.join(EVENTS_DIR,
                                              f'{r["subject_id"]}_sie_events.csv'))]
    print(f"Subjects: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS',
                                    min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    ok_ids = [r for r in results if r is not None]
    print(f"Successful: {len(ok_ids)}")
    if not ok_ids:
        return

    # Group average on fsaverage (already in fsaverage space since we used
    # fsaverage src)
    all_data = []
    ref_stc = None
    for sub_id in ok_ids:
        f = os.path.join(STC_DIR, f'{sub_id}_Q4_SR1_ratio-lh.stc')
        if not os.path.isfile(f):
            continue
        stc = mne.read_source_estimate(f.replace('-lh.stc', ''), 'fsaverage')
        all_data.append(stc.data.squeeze())
        ref_stc = stc
    if not all_data:
        return
    grand = np.median(np.array(all_data), axis=0)
    if ref_stc is None:
        return
    grand_stc = ref_stc.copy()
    grand_stc.data = grand[:, np.newaxis]
    grand_stc.save(os.path.join(OUT_DIR, 'group_Q4_SR1_ratio'),
                   overwrite=True, verbose=False)
    print(f"Group STC saved. n={len(all_data)} subjects")
    print(f"Per-vertex ratio: median {np.nanmedian(grand):.2f} "
          f"p90 {np.nanpercentile(grand, 90):.2f} "
          f"p99 {np.nanpercentile(grand, 99):.2f}")

    # Desikan-Killiany label aggregation
    labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                         subjects_dir=SUBJECTS_DIR,
                                         verbose=False)
    rows = []
    for lbl in labels:
        idx = lbl.get_vertices_used()
        hemi_data = grand_stc.lh_data if lbl.hemi == 'lh' else grand_stc.rh_data
        vert_idx_in_src = (ref_stc.lh_vertno if lbl.hemi == 'lh'
                           else ref_stc.rh_vertno)
        match = np.isin(vert_idx_in_src, idx)
        if match.sum() == 0:
            continue
        r_mean = float(np.nanmean(hemi_data[match, 0]))
        r_median = float(np.nanmedian(hemi_data[match, 0]))
        rows.append({'label': lbl.name,
                     'hemi': lbl.hemi,
                     'n_vertices': int(match.sum()),
                     'ratio_mean': r_mean,
                     'ratio_median': r_median})
    df = pd.DataFrame(rows).sort_values('ratio_median', ascending=False)
    df.to_csv(os.path.join(OUT_DIR, 'Q4_SR1_label_ranking.csv'), index=False)
    print(f"\nTop 12 labels by median ratio:")
    print(df.head(12).to_string(index=False))

    # Simple brain figure (medial + lateral views, LH + RH)
    try:
        brain = grand_stc.plot(
            subject='fsaverage', subjects_dir=SUBJECTS_DIR,
            hemi='split', views=['lat', 'med'],
            clim='auto', time_viewer=False, backend='matplotlib',
            show_traces=False, background='white', size=(1200, 600))
        fig = brain if isinstance(brain, plt.Figure) else brain.plotter.gcf()
        fig.suptitle('B49 — Source-space SR1 ratio (event/baseline) at '
                      'Q4 SIEs, group median on fsaverage', fontsize=12)
        fig.savefig(os.path.join(OUT_DIR, 'group_Q4_SR1_ratio_brain.png'),
                     dpi=120, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Brain render failed: {e} (STC saved, can be rendered later)")


if __name__ == '__main__':
    main()
