#!/usr/bin/env python3
"""
Arc 9.1 — Source-space localization of composite v2 SIE Q4 generator.

Composite v2 extension of B49 (originally LEMON-only, envelope pipeline).

Pipeline (per subject):
  1. Load raw EEG (cohort-specific loader).
  2. Apply standard_1020 montage, pick channels with positions, avg ref.
  3. Build forward model against fsaverage template.
  4. Compute noise covariance from random baseline segments (EVENT_LAG away from events).
  5. Make sLORETA inverse operator.
  6. For each Q4 event (template_rho quartile from per_event_quality_<cohort>_composite.csv):
     - Compute event-window source-PSD and baseline-window source-PSD.
     - Take SR1-band (7.0-8.3 Hz) band-power per vertex.
  7. Event/baseline ratio per vertex, averaged across events.

Group: morph to fsaverage, median across subjects.
Output: per-cohort ROI ranking (Desikan-Killiany).

Usage on VM:
  python3 scripts/sie_source_localization_composite.py --cohort lemon
  python3 scripts/sie_source_localization_composite.py --cohort dortmund
  python3 scripts/sie_source_localization_composite.py --cohort hbn_R4

Environment:
  SIE_WORKERS: Pool size (default 28 on VM)
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_chbmp, load_tdbrain,
    load_hbn_by_subject, load_srm, load_mpeng_concatenated,
    load_vep_concatenated,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

# ============ Cohort-specific configuration ============
# For each cohort: (loader_fn, loader_kwargs)
def _get_loader(cohort):
    """Return (loader_fn, events_dir) for a cohort.
    Events dir is where per-subject _sie_events.csv files live.
    """
    ROOT = os.path.join(os.path.dirname(__file__), '..')
    EVENTS_BASE = os.path.join(ROOT, 'exports_sie')

    # LEMON variants
    if cohort == 'lemon':
        return (lambda s: load_lemon(s, condition='EC'),
                os.path.join(EVENTS_BASE, 'lemon_composite'))
    if cohort == 'lemon_EO':
        return (lambda s: load_lemon(s, condition='EO'),
                os.path.join(EVENTS_BASE, 'lemon_EO_composite'))
    # Dortmund variants (task × acq × ses)
    if cohort == 'dortmund':  # alias for EC_pre_s1
        return (lambda s: load_dortmund(s, task='EyesClosed', acq='pre', ses='1'),
                os.path.join(EVENTS_BASE, 'dortmund_composite'))
    if cohort == 'dortmund_EC_pre_s2':
        return (lambda s: load_dortmund(s, task='EyesClosed', acq='pre', ses='2'),
                os.path.join(EVENTS_BASE, 'dortmund_EC_pre_s2_composite'))
    if cohort == 'dortmund_EC_post_s1':
        return (lambda s: load_dortmund(s, task='EyesClosed', acq='post', ses='1'),
                os.path.join(EVENTS_BASE, 'dortmund_EC_post_s1_composite'))
    if cohort == 'dortmund_EC_post_s2':
        return (lambda s: load_dortmund(s, task='EyesClosed', acq='post', ses='2'),
                os.path.join(EVENTS_BASE, 'dortmund_EC_post_s2_composite'))
    if cohort == 'dortmund_EO':  # alias for EO_pre_s1
        return (lambda s: load_dortmund(s, task='EyesOpen', acq='pre', ses='1'),
                os.path.join(EVENTS_BASE, 'dortmund_EO_composite'))
    if cohort == 'dortmund_EO_pre_s2':
        return (lambda s: load_dortmund(s, task='EyesOpen', acq='pre', ses='2'),
                os.path.join(EVENTS_BASE, 'dortmund_EO_pre_s2_composite'))
    if cohort == 'dortmund_EO_post_s1':
        return (lambda s: load_dortmund(s, task='EyesOpen', acq='post', ses='1'),
                os.path.join(EVENTS_BASE, 'dortmund_EO_post_s1_composite'))
    if cohort == 'dortmund_EO_post_s2':
        return (lambda s: load_dortmund(s, task='EyesOpen', acq='post', ses='2'),
                os.path.join(EVENTS_BASE, 'dortmund_EO_post_s2_composite'))
    # chbmp
    if cohort == 'chbmp':
        return (load_chbmp, os.path.join(EVENTS_BASE, 'chbmp_composite'))
    # tdbrain
    if cohort == 'tdbrain':
        return (lambda s: load_tdbrain(s, condition='EC'),
                os.path.join(EVENTS_BASE, 'tdbrain_composite'))
    if cohort == 'tdbrain_EO':
        return (lambda s: load_tdbrain(s, condition='EO'),
                os.path.join(EVENTS_BASE, 'tdbrain_EO_composite'))
    # HBN releases
    if cohort.startswith('hbn_R'):
        release = cohort.replace('hbn_', '')
        return (lambda s: load_hbn_by_subject(s, release=release),
                os.path.join(EVENTS_BASE, f'{cohort}_composite'))
    # SRM (Norway BioSemi 64-ch EC resting state)
    if cohort == 'srm':
        return (lambda s: load_srm(s, ses='t1'),
                os.path.join(EVENTS_BASE, 'srm_composite'))
    raise ValueError(f'Unsupported cohort: {cohort}')


# ============ Source-space parameters ============
SR1_BAND = (7.0, 8.3)
EVENT_WIN = (-2.0, 4.0)
BASELINE_WIN_N = 40
BASELINE_WIN_DUR = 6.0
EVENT_LAG = 1.0
METHOD = 'sLORETA'
SNR = 3.0
LAMBDA2 = 1.0 / SNR ** 2

_fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
SUBJECTS_DIR = os.path.dirname(_fs_dir)
FSAVG_SRC = os.path.join(_fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
FSAVG_BEM = os.path.join(_fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
TRANS = 'fsaverage'


def _prep_raw(raw):
    """Apply best-fitting standard montage (tries 1020, 1005, GSN-HydroCel-129),
    keep channels with positions, avg ref.
    Also strips common suffixes like '-REF', ' -REF', '-Ref', 'EEG ', etc.
    """
    raw = raw.copy()
    # Clean channel names: strip REF, whitespace, common suffixes
    import re
    rename = {}
    for ch in raw.ch_names:
        clean = re.sub(r'\s*-\s*REF\s*$', '', ch, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^EEG\s+', '', clean, flags=re.IGNORECASE).strip()
        # Normalize "Cz " → "Cz", "OZ" → "Oz" (capitalize first letter, lower rest if needed)
        if clean and clean != ch:
            rename[ch] = clean
    if rename:
        raw.rename_channels(rename)
    best_n = 0
    best_montage_name = None
    for mname in ['standard_1020', 'standard_1005',
                  'GSN-HydroCel-129', 'GSN-HydroCel-128', 'GSN-HydroCel-257']:
        try:
            m = mne.channels.make_standard_montage(mname)
            r_test = raw.copy()
            r_test.set_montage(m, match_case=False, on_missing='ignore', verbose=False)
            pos = r_test.get_montage().get_positions()['ch_pos']
            n_pos = sum(1 for ch in r_test.ch_names
                        if pos.get(ch) is not None
                        and np.all(np.isfinite(pos.get(ch, np.full(3, np.nan)))))
            if n_pos > best_n:
                best_n = n_pos
                best_montage_name = mname
        except Exception:
            continue
    if best_montage_name is None or best_n < 20:
        raise ValueError(f'No montage with >=20 positioned channels (best {best_n}).')
    montage = mne.channels.make_standard_montage(best_montage_name)
    raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
    pos = raw.get_montage().get_positions()['ch_pos']
    keep = [ch for ch in raw.ch_names
            if pos.get(ch) is not None
            and np.all(np.isfinite(pos.get(ch, np.full(3, np.nan))))]
    raw.pick(keep)
    raw.set_eeg_reference('average', projection=True, verbose=False)
    return raw


def _compute_band_power_stc(raw, inv, epochs, band):
    """Mean band-power STC across epochs."""
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
    sub_id, cohort, events_dir, quality_csv, stc_dir = args
    try:
        events = pd.read_csv(os.path.join(events_dir, f'{sub_id}_sie_events.csv'))
    except Exception:
        return None
    events = events.dropna(subset=['t0_net'])
    if len(events) == 0:
        return None

    # Q4-filter by template_rho if external quality CSV covers this subject;
    # else fall back to sr_score_canonical from the events CSV (per-cohort Q4 on the fly).
    try:
        qual_has_subject = False
        if os.path.isfile(quality_csv):
            qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
            if (qual['subject_id'] == sub_id).any():
                qual_has_subject = True
                qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1','Q2','Q3','Q4'])
                q4 = qual[(qual['subject_id'] == sub_id) & (qual['rho_q'] == 'Q4')]
                q4_times = set(q4['t0_net'].round(3).values)
                events['t0_round'] = events['t0_net'].round(3)
                events = events[events['t0_round'].isin(q4_times)]
        if not qual_has_subject:
            # Fallback: rank this subject's events by sr_score_canonical.
            # Most dortmund subjects have only 1-3 events total (short recordings),
            # so keep all events if <4, else keep top quartile.
            fallback_col = None
            for col in ['sr_score_canonical', 'sr_score', 'HSI_canonical', 'HSI']:
                if col in events.columns:
                    fallback_col = col
                    break
            if fallback_col is not None:
                ev = events.dropna(subset=[fallback_col]).copy()
                if len(ev) >= 4:
                    q75 = ev[fallback_col].quantile(0.75)
                    events = ev[ev[fallback_col] >= q75]
                else:
                    events = ev  # too few events to rank; keep all
            # If fallback_col is None, just use all events (no filter)
    except Exception:
        return None
    if len(events) < 1:
        return None

    # Load raw via cohort-specific loader
    loader_fn, _ = _get_loader(cohort)
    try:
        raw = loader_fn(sub_id)
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

    ev_times = [float(ev['t0_net']) + EVENT_LAG
                for _, ev in events.iterrows()
                if EVENT_WIN[0] + float(ev['t0_net']) > 2
                and EVENT_WIN[1] + float(ev['t0_net']) < t_end - 2]
    if not ev_times:
        return None
    ev_array = np.array([[int(t * fs), 0, 1] for t in ev_times])
    ev_epochs = mne.Epochs(raw, ev_array, event_id={'Q4': 1},
                           tmin=EVENT_WIN[0], tmax=EVENT_WIN[1],
                           baseline=None, preload=True, proj=True, verbose=False)
    if len(ev_epochs) == 0:
        return None

    rng = np.random.default_rng(42 + hash(sub_id) % 10000)
    ev_set = set([int(t) for t in ev_times])
    cand = np.arange(int(BASELINE_WIN_DUR) + 5, int(t_end - BASELINE_WIN_DUR - 5))
    cand = [t for t in cand if not any(abs(t - e) < 10 for e in ev_set)]
    if len(cand) < 5:
        return None
    bl_times = rng.choice(cand, size=min(BASELINE_WIN_N, len(cand)), replace=False)
    bl_array = np.array([[int(t * fs), 0, 2] for t in bl_times])
    bl_epochs = mne.Epochs(raw, bl_array, event_id={'BL': 2},
                           tmin=-BASELINE_WIN_DUR / 2, tmax=BASELINE_WIN_DUR / 2,
                           baseline=None, preload=True, proj=True, verbose=False)
    if len(bl_epochs) < 5:
        return None

    try:
        noise_cov = mne.compute_covariance(bl_epochs, tmin=-2, tmax=2,
                                           method='shrunk', rank='info', verbose=False)
    except Exception:
        return None

    try:
        fwd = mne.make_forward_solution(raw.info, trans=TRANS,
                                        src=FSAVG_SRC, bem=FSAVG_BEM,
                                        eeg=True, mindist=5.0, n_jobs=1, verbose=False)
    except Exception:
        return None

    try:
        inv = mne.minimum_norm.make_inverse_operator(
            raw.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)
    except Exception:
        return None

    ev_stc = _compute_band_power_stc(raw, inv, ev_epochs, SR1_BAND)
    bl_stc = _compute_band_power_stc(raw, inv, bl_epochs, SR1_BAND)
    if ev_stc is None or bl_stc is None:
        return None

    ratio_data = (ev_stc.data + 1e-25) / (bl_stc.data + 1e-25)
    ratio_stc = ev_stc.copy()
    ratio_stc.data = ratio_data

    os.makedirs(stc_dir, exist_ok=True)
    ratio_stc.save(os.path.join(stc_dir, f'{sub_id}_Q4_SR1_ratio'),
                   overwrite=True, verbose=False)
    return sub_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', required=True,
                        help='Cohort name (lemon, lemon_EO, dortmund, chbmp, tdbrain, '
                             'tdbrain_EO, hbn_R1..hbn_R11)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: outputs/schumann/images/source/<cohort>_composite/)')
    args = parser.parse_args()

    cohort = args.cohort
    ROOT = os.path.join(os.path.dirname(__file__), '..')
    loader_fn, events_dir = _get_loader(cohort)

    out_dir = args.out_dir or os.path.join(
        ROOT, 'outputs', 'schumann', 'images', 'source', f'{cohort}_composite')
    stc_dir = os.path.join(out_dir, 'stcs')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(stc_dir, exist_ok=True)

    quality_csv = os.path.join(
        ROOT, 'outputs', 'schumann', 'images', 'quality',
        f'per_event_quality_{cohort}_composite.csv')

    print(f'=== Source-space composite v2: {cohort} ===')
    print(f'Events dir: {events_dir}')
    print(f'Quality CSV: {quality_csv}')
    print(f'Output dir: {out_dir}')

    # Load subject list from extraction summary
    ext_summary = os.path.join(events_dir, 'extraction_summary.csv')
    if not os.path.isfile(ext_summary):
        print(f'ERROR: {ext_summary} not found')
        return
    summary = pd.read_csv(ext_summary)
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = [(r['subject_id'], cohort, events_dir, quality_csv, stc_dir)
             for _, r in ok.iterrows()
             if os.path.isfile(os.path.join(events_dir, f"{r['subject_id']}_sie_events.csv"))]
    print(f'Subjects to process: {len(tasks)}')

    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f'Workers: {n_workers}')

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    ok_ids = [r for r in results if r is not None]
    print(f'Successful: {len(ok_ids)}/{len(tasks)}')
    if not ok_ids:
        print('No successful subjects; aborting group aggregation.')
        return

    # Group aggregation on fsaverage
    all_data = []
    ref_stc = None
    for sub_id in ok_ids:
        f = os.path.join(stc_dir, f'{sub_id}_Q4_SR1_ratio-lh.stc')
        if not os.path.isfile(f):
            continue
        stc = mne.read_source_estimate(f.replace('-lh.stc', ''), 'fsaverage')
        all_data.append(stc.data.squeeze())
        ref_stc = stc
    if not all_data or ref_stc is None:
        print('No STC files to aggregate.')
        return
    grand = np.median(np.array(all_data), axis=0)
    grand_stc = ref_stc.copy()
    grand_stc.data = grand[:, np.newaxis]
    grand_stc.save(os.path.join(out_dir, 'group_Q4_SR1_ratio'),
                   overwrite=True, verbose=False)
    print(f'Group STC saved. n={len(all_data)} subjects')
    print(f'Per-vertex ratio: median {np.nanmedian(grand):.2f} '
          f'p90 {np.nanpercentile(grand, 90):.2f} '
          f'p99 {np.nanpercentile(grand, 99):.2f}')

    # Desikan-Killiany label aggregation
    labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                        subjects_dir=SUBJECTS_DIR, verbose=False)
    rows = []
    for lbl in labels:
        idx = lbl.get_vertices_used()
        hemi_data = grand_stc.lh_data if lbl.hemi == 'lh' else grand_stc.rh_data
        vert_idx_in_src = (ref_stc.lh_vertno if lbl.hemi == 'lh'
                           else ref_stc.rh_vertno)
        match = np.isin(vert_idx_in_src, idx)
        if match.sum() == 0:
            continue
        rows.append({
            'label': lbl.name, 'hemi': lbl.hemi,
            'n_vertices': int(match.sum()),
            'ratio_mean': float(np.nanmean(hemi_data[match, 0])),
            'ratio_median': float(np.nanmedian(hemi_data[match, 0])),
        })
    df = pd.DataFrame(rows).sort_values('ratio_median', ascending=False)
    df.to_csv(os.path.join(out_dir, 'Q4_SR1_label_ranking.csv'), index=False)
    print(f'\nTop 12 labels by median ratio ({cohort}):')
    print(df.head(12).to_string(index=False))

    # Per-subject summary for downstream use (subject-level source-strength metric)
    subj_rows = []
    for sub_id in ok_ids:
        f = os.path.join(stc_dir, f'{sub_id}_Q4_SR1_ratio-lh.stc')
        if not os.path.isfile(f):
            continue
        stc = mne.read_source_estimate(f.replace('-lh.stc', ''), 'fsaverage')
        d = stc.data.squeeze()
        subj_rows.append({
            'subject_id': sub_id,
            'median_ratio': float(np.nanmedian(d)),
            'p90_ratio': float(np.nanpercentile(d, 90)),
            'p99_ratio': float(np.nanpercentile(d, 99)),
            'max_ratio': float(np.nanmax(d)),
        })
    pd.DataFrame(subj_rows).to_csv(
        os.path.join(out_dir, 'per_subject_source_summary.csv'), index=False)
    print(f"Per-subject summary saved: {len(subj_rows)} subjects")


if __name__ == '__main__':
    main()
