#!/usr/bin/env python3
"""
B42 re-run on composite v2 detector.

Topographic localization of event-locked peaks in three bands:
  SR1 [7.0-8.3], β16 [14.5-17.5], SR3 [19.5-20.4]

Per composite-Q4 subject per channel: event-average PSD at t0_net + 1 s
(4-s window) / baseline PSD ratio at band peak. Grand-average per channel.
Topomaps + top-10 ranked channels + per-band pair Pearson correlations.

Envelope B42: SR1 posterior (PO10/TP7/P7), β16 centro-parietal, SR3
central-left; SR1 × SR3 topographic r = −0.44 — anti-correlated.

Cohort-parameterized.

Usage:
    python scripts/sie_16hz_topography_composite.py --cohort lemon
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0

BANDS = {
    'SR1': (7.0, 8.3),
    'β16': (14.5, 17.5),
    'SR3': (19.5, 20.4),
}

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    qual = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events, qual
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events, qual
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events, qual
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events, qual
    if cohort == 'srm':
        return load_srm, {}, events, qual
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events, qual
    if cohort == 'dortmund':
        return load_dortmund, {}, events, qual
    if cohort == 'chbmp':
        return load_chbmp, {}, events, qual
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events, qual
    raise ValueError(f"unsupported cohort {cohort!r}")


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    w = signal.windows.hann(len(seg))
    wp = np.sum(w ** 2)
    X = np.fft.rfft(seg * w, nfft)
    psd = (np.abs(X) ** 2) / (fs * wp)
    psd[1:-1] *= 2.0
    return psd


_LOADER = None
_LOADER_KW = None


def _init_worker(loader_name, loader_kw):
    global _LOADER, _LOADER_KW
    _LOADER_KW = loader_kw
    _LOADER = {
        'load_lemon': load_lemon,
        'load_tdbrain': load_tdbrain,
        'load_srm': load_srm,
        'load_dortmund': load_dortmund,
        'load_chbmp': load_chbmp,
        'load_hbn_by_subject': load_hbn_by_subject,
    }[loader_name]


def process_subject(args):
    sub_id, events_path, quality_csv = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    try:
        qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1','Q2','Q3','Q4'])
        q4 = qual[(qual['subject_id'] == sub_id) & (qual['rho_q'] == 'Q4')]
        q4_times = set(q4['t0_net'].round(3).values)
        events['t0_round'] = events['t0_net'].round(3)
        events = events[events['t0_round'].isin(q4_times)]
    except Exception:
        pass
    if len(events) < 1:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 40:
        return None
    ch_names = raw.ch_names
    n_ch = len(ch_names)
    X = raw.get_data() * 1e6

    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band_masks = {name: (freqs >= lo) & (freqs <= hi) for name, (lo, hi) in BANDS.items()}

    base_cnt = 0
    base_psds = np.zeros((n_ch, len(freqs)))
    for i in range(0, X.shape[1] - nperseg + 1, nhop):
        for c in range(n_ch):
            base_psds[c] += welch_one(X[c, i:i+nperseg], fs, nfft)
        base_cnt += 1
    if base_cnt < 10:
        return None
    base_psds /= base_cnt

    ev_psds = np.zeros((n_ch, len(freqs)))
    ev_cnt = 0
    for _, ev in events.iterrows():
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > X.shape[1]:
            continue
        for c in range(n_ch):
            ev_psds[c] += welch_one(X[c, i0:i1], fs, nfft)
        ev_cnt += 1
    if ev_cnt < 1:
        return None
    ev_psds /= ev_cnt

    ratio_maps = {}
    for name, mask in band_masks.items():
        ratios = np.zeros(n_ch)
        for c in range(n_ch):
            ev_m = np.nanmax(ev_psds[c, mask])
            base_m = np.nanmean(base_psds[c, mask])
            ratios[c] = ev_m / (base_m + 1e-20)
        ratio_maps[name] = ratios
    return {
        'subject_id': sub_id,
        'n_events': ev_cnt,
        'ch_names': ch_names,
        'ratios': ratio_maps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'coupling', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep, quality_csv))
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)} (Q4)")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    ch_counts = {}
    for r in results:
        for ch in r['ch_names']:
            ch_counts[ch] = ch_counts.get(ch, 0) + 1
    threshold = int(0.8 * len(results))
    common_chs = [ch for ch, n in ch_counts.items() if n >= threshold]
    print(f"Common channels (≥80% of subjects): {len(common_chs)}")

    grand = {name: {} for name in BANDS}
    for r in results:
        for name, ratios in r['ratios'].items():
            for c_idx, c in enumerate(r['ch_names']):
                if c in common_chs:
                    grand[name].setdefault(c, []).append(ratios[c_idx])
    grand_avg = {}
    for name in BANDS:
        grand_avg[name] = {c: np.nanmean(v) for c, v in grand[name].items()}

    print(f"\n=== {args.cohort} composite · Top 10 channels per band ===")
    for name in BANDS:
        print(f"\n  {name}:")
        ordered = sorted(grand_avg[name].items(), key=lambda kv: -kv[1])
        for ch, v in ordered[:10]:
            print(f"    {ch:<6} {v:.3f}×")

    # Topographic correlations between bands
    def vec(name):
        return np.array([grand_avg[name].get(c, np.nan) for c in common_chs])
    v_sr1 = vec('SR1'); v_b16 = vec('β16'); v_sr3 = vec('SR3')
    mask_all = np.isfinite(v_sr1) & np.isfinite(v_b16) & np.isfinite(v_sr3)
    print(f"\n=== {args.cohort} composite · Topographic correlations (Pearson r across channels) ===")
    r_sr1_b16, p_sr1_b16 = pearsonr(v_sr1[mask_all], v_b16[mask_all])
    r_sr1_sr3, p_sr1_sr3 = pearsonr(v_sr1[mask_all], v_sr3[mask_all])
    r_b16_sr3, p_b16_sr3 = pearsonr(v_b16[mask_all], v_sr3[mask_all])
    print(f"  SR1 × β16:  r = {r_sr1_b16:+.3f}  p = {p_sr1_b16:.3g}")
    print(f"  SR1 × SR3:  r = {r_sr1_sr3:+.3f}  p = {p_sr1_sr3:.3g}  "
          f"(envelope: r = −0.44, anti-correlated)")
    print(f"  β16 × SR3:  r = {r_b16_sr3:+.3f}  p = {p_b16_sr3:.3g}")

    rows = []
    for name in BANDS:
        for c, v in grand_avg[name].items():
            rows.append({'band': name, 'channel': c, 'ratio': v})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, '16hz_topography.csv'),
                               index=False)

    try:
        raw_ref = loader(results[0]['subject_id'], **loader_kw)
        fs_ref = raw_ref.info['sfreq']
        montage = mne.channels.make_standard_montage('standard_1020')
        info_common = mne.create_info(ch_names=common_chs, sfreq=fs_ref,
                                       ch_types='eeg')
        info_common.set_montage(montage, match_case=False, on_missing='ignore')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, name in enumerate(BANDS):
            data = np.array([grand_avg[name].get(c, np.nan) for c in common_chs])
            ax = axes[i]
            im, _ = mne.viz.plot_topomap(data, info_common, axes=ax, show=False,
                                          cmap='viridis', contours=6)
            ax.set_title(f'{name}\n'
                         f'range {data.min():.2f}-{data.max():.2f}×\n'
                         f'max @ {common_chs[int(np.nanargmax(data))]}',
                         fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.7)
        plt.suptitle(f'B42 composite — Event-locked topography by band '
                     f'({args.cohort} Q4, n={len(results)})',
                     y=1.02, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '16hz_topography.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {out_dir}/16hz_topography.png")
    except Exception as e:
        print(f"Topomap plot failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
