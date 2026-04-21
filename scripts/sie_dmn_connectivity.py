#!/usr/bin/env python3
"""
B50 — DMN-node source-level connectivity at Q4 ignitions (SR1-band wPLI).

Directly tests B49's interpretation: if Q4 SIEs are spontaneous DMN
engagement events, then SR1-band wPLI between DMN-node sources should
rise at event windows relative to baseline. DMN-frontal and frontal-
frontal edges should NOT rise (control).

Pipeline:
  1. Same forward/inverse setup as B49 (sLORETA on fsaverage ico-5).
  2. Epoch Q4 events at [-2, +4] s; baseline at 40 random 6-s windows.
  3. Apply inverse to get STC per epoch.
  4. Extract mean source time course per Desikan-Killiany label.
  5. Compute wPLI at SR1 (7.0-8.3 Hz) across event epochs vs baseline
     epochs separately. Output: (n_labels, n_labels) wPLI matrix per
     condition per subject.
  6. Group: median event - baseline wPLI per edge; classify edges by
     category (DMN-DMN, DMN-frontal, frontal-frontal, other) and test
     rise distributions.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon

warnings.filterwarnings('ignore')
import mne
import mne_connectivity
mne.set_log_level('ERROR')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'source')
CONN_DIR = os.path.join(OUT_DIR, 'connectivity')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality',
                            'per_event_quality.csv')
os.makedirs(CONN_DIR, exist_ok=True)

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

# ROI selection based on B49 results
DMN_LABELS = {'precuneus', 'parahippocampal', 'bankssts', 'supramarginal',
              'inferiorparietal', 'fusiform', 'inferiortemporal',
              'middletemporal'}
FRONTAL_LABELS = {'rostralanteriorcingulate', 'frontalpole',
                   'medialorbitofrontal', 'parsopercularis',
                   'superiorfrontal', 'rostralmiddlefrontal',
                   'caudalanteriorcingulate'}
OCCIPITAL_LABELS = {'lingual', 'cuneus', 'pericalcarine', 'lateraloccipital'}


def _prep_raw(raw):
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


def process_subject(sub_id):
    try:
        events = pd.read_csv(os.path.join(EVENTS_DIR,
                                           f'{sub_id}_sie_events.csv'))
    except Exception:
        return None
    events = events.dropna(subset=['t0_net'])
    if len(events) == 0:
        return None
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
    if len(events) < 2:
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

    ev_times = [float(ev['t0_net']) + EVENT_LAG
                 for _, ev in events.iterrows()
                 if EVENT_WIN[0] + float(ev['t0_net']) > 2
                 and EVENT_WIN[1] + float(ev['t0_net']) < t_end - 2]
    if len(ev_times) < 2:
        return None
    ev_array = np.array([[int(t * fs), 0, 1] for t in ev_times])
    ev_epochs = mne.Epochs(raw, ev_array, event_id={'Q4': 1},
                            tmin=EVENT_WIN[0], tmax=EVENT_WIN[1],
                            baseline=None, preload=True, proj=True,
                            verbose=False)
    if len(ev_epochs) < 2:
        return None

    rng = np.random.default_rng(42 + hash(sub_id) % 10000)
    ev_set = [int(t) for t in ev_times]
    cand = [t for t in range(int(BASELINE_WIN_DUR) + 5,
                              int(t_end - BASELINE_WIN_DUR - 5))
            if not any(abs(t - e) < 10 for e in ev_set)]
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

    src = inv['src']
    labels = mne.read_labels_from_annot('fsaverage', 'aparc',
                                         subjects_dir=SUBJECTS_DIR,
                                         verbose=False)
    labels = [lbl for lbl in labels
              if not (lbl.name.startswith('unknown-')
                      or lbl.name.startswith('corpuscallosum-'))]

    # Per-epoch label time courses → wPLI
    def conn_from_epochs(epochs):
        stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs, inv, lambda2=LAMBDA2, method=METHOD,
            pick_ori='normal', return_generator=True, verbose=False)
        ltcs = mne.extract_label_time_course(
            stcs, labels, src, mode='mean_flip', return_generator=False,
            verbose=False)
        if len(ltcs) < 2:
            return None
        data = np.stack(ltcs, axis=0)  # (n_epochs, n_labels, n_times)
        c = mne_connectivity.spectral_connectivity_epochs(
            data, sfreq=epochs.info['sfreq'], method='wpli',
            mode='multitaper', fmin=SR1_BAND[0], fmax=SR1_BAND[1],
            faverage=True, mt_bandwidth=1.0, mt_adaptive=False,
            verbose=False)
        # returns (n_edges, 1) connectivity matrix (ravel-indexed)
        mat = c.get_data(output='dense')[..., 0]
        # Symmetrize (MNE only fills lower triangle)
        mat = mat + mat.T
        return mat

    try:
        ev_mat = conn_from_epochs(ev_epochs)
        bl_mat = conn_from_epochs(bl_epochs)
    except Exception as e:
        print(f"  {sub_id} connectivity error: {e}")
        return None
    if ev_mat is None or bl_mat is None:
        return None

    # Save matrices
    np.savez_compressed(os.path.join(CONN_DIR, f'{sub_id}_wpli.npz'),
                        ev=ev_mat, bl=bl_mat,
                        labels=[lbl.name for lbl in labels])
    return sub_id


def classify_edge(l1, l2):
    base1 = l1.replace('-lh', '').replace('-rh', '')
    base2 = l2.replace('-lh', '').replace('-rh', '')
    is_dmn1 = base1 in DMN_LABELS
    is_dmn2 = base2 in DMN_LABELS
    is_fr1 = base1 in FRONTAL_LABELS
    is_fr2 = base2 in FRONTAL_LABELS
    is_occ1 = base1 in OCCIPITAL_LABELS
    is_occ2 = base2 in OCCIPITAL_LABELS
    if is_dmn1 and is_dmn2:
        return 'DMN-DMN'
    if (is_fr1 and is_fr2):
        return 'Frontal-Frontal'
    if (is_dmn1 and is_fr2) or (is_dmn2 and is_fr1):
        return 'DMN-Frontal'
    if (is_dmn1 and is_occ2) or (is_dmn2 and is_occ1):
        return 'DMN-Occipital'
    if (is_occ1 and is_occ2):
        return 'Occipital-Occipital'
    return 'Other'


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

    # Aggregate per-edge event-baseline differences
    all_diffs = []
    label_names = None
    for sub_id in ok_ids:
        f = os.path.join(CONN_DIR, f'{sub_id}_wpli.npz')
        if not os.path.isfile(f):
            continue
        d = np.load(f, allow_pickle=True)
        if label_names is None:
            label_names = list(d['labels'])
        all_diffs.append(d['ev'] - d['bl'])
    diff_stack = np.stack(all_diffs, axis=0)  # (n_sub, n_lbl, n_lbl)
    median_diff = np.nanmedian(diff_stack, axis=0)

    # Classify every edge
    n = len(label_names)
    cat = np.full((n, n), '', dtype=object)
    for i in range(n):
        for j in range(n):
            cat[i, j] = classify_edge(label_names[i], label_names[j])

    # Per-category distribution
    print(f"\n=== Median event-baseline SR1 wPLI by edge category ===")
    rows = []
    for category in ['DMN-DMN', 'DMN-Occipital', 'Occipital-Occipital',
                     'DMN-Frontal', 'Frontal-Frontal', 'Other']:
        idx = (cat == category) & np.triu(np.ones((n, n), bool), k=1)
        vals = median_diff[idx]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        q25, med, q75 = np.nanpercentile(vals, [25, 50, 75])
        # Per-subject test: mean over edges in this category, per subject
        per_sub = []
        for d in diff_stack:
            edge_vals = d[idx]
            edge_vals = edge_vals[np.isfinite(edge_vals)]
            if len(edge_vals) == 0: continue
            per_sub.append(np.nanmean(edge_vals))
        per_sub = np.array(per_sub)
        try:
            from scipy.stats import wilcoxon
            p = wilcoxon(per_sub).pvalue
        except Exception:
            p = np.nan
        pct_pos = (per_sub > 0).mean() * 100
        rows.append({
            'category': category, 'n_edges': int(idx.sum()),
            'median_diff': med, 'q25': q25, 'q75': q75,
            'per_sub_mean': float(np.nanmean(per_sub)),
            'pct_pos_sub': pct_pos, 'Wilcoxon_p': p,
        })
        print(f"  {category:22s}  n_edges={int(idx.sum()):4d}  "
              f"median={med:+.4f}  IQR=[{q25:+.4f}, {q75:+.4f}]  "
              f"per-sub mean={np.nanmean(per_sub):+.4f}  "
              f"{pct_pos:.0f}% sub>0  p={p:.2g}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'dmn_connectivity_by_category.csv'),
                               index=False)

    # Figure: box plots by category
    fig, ax = plt.subplots(figsize=(10, 5))
    cats_order = ['DMN-DMN', 'DMN-Occipital', 'Occipital-Occipital',
                  'DMN-Frontal', 'Frontal-Frontal']
    per_sub_bycat = []
    for category in cats_order:
        idx = (cat == category) & np.triu(np.ones((n, n), bool), k=1)
        ps = []
        for d in diff_stack:
            vals = d[idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0: continue
            ps.append(np.nanmean(vals))
        per_sub_bycat.append(ps)
    colors = ['#8c1a1a', '#d73027', '#fdae61', '#4575b4', '#2b5fb8']
    bp = ax.boxplot(per_sub_bycat, labels=cats_order, patch_artist=True,
                    showfliers=False)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('per-subject mean ΔwPLI (event − baseline) at SR1 band')
    ax.set_title('B50 — SR1 wPLI change at Q4 SIEs by edge category '
                  f'(n = {len(ok_ids)} subjects)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'dmn_connectivity_by_category.png')
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
