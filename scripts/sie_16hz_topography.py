#!/usr/bin/env python3
"""
B42 — Topographic localization of the 16 Hz event-locked peak.

Test 3: where on the scalp does the 16 Hz elevation peak?

  - Central-frontal (C3/Cz/C4) → sensorimotor β (classical mu-harmonic)
  - Frontal (Fz) → prefrontal β (attentional)
  - Posterior (O1/Pz/O2) → posterior β (alpha-band-adjacent)
  - Diffuse → no clear localization

For each LEMON Q4 subject, per channel:
  - Event-average PSD at t0+1s (4-s window)
  - Baseline PSD (all sliding windows)
  - Peak amplitude in SR1 [7.0-8.3], β16 [14-18], SR3 [19.5-20.4]
Grand-average across subjects per channel and per band.

Topomaps for SR1, β16, SR3 side by side.
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

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'coupling')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality', 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
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


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    w = signal.windows.hann(len(seg))
    wp = np.sum(w ** 2)
    X = np.fft.rfft(seg * w, nfft)
    psd = (np.abs(X) ** 2) / (fs * wp)
    psd[1:-1] *= 2.0
    return psd


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    try:
        qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
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
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    if fs < 40:
        return None
    ch_names = raw.ch_names
    n_ch = len(ch_names)
    X = raw.get_data() * 1e6     # (n_ch, n_samples)

    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band_masks = {name: (freqs >= lo) & (freqs <= hi) for name, (lo, hi) in BANDS.items()}

    # Baselines per channel
    base_cnt = 0
    base_psds = np.zeros((n_ch, len(freqs)))
    for i in range(0, X.shape[1] - nperseg + 1, nhop):
        for c in range(n_ch):
            base_psds[c] += welch_one(X[c, i:i+nperseg], fs, nfft)
        base_cnt += 1
    if base_cnt < 10:
        return None
    base_psds /= base_cnt

    # Event PSDs per channel
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

    # Per-band peak ratio per channel
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
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    # Find common channels (present in ≥80% of subjects)
    ch_counts = {}
    for r in results:
        for ch in r['ch_names']:
            ch_counts[ch] = ch_counts.get(ch, 0) + 1
    threshold = int(0.8 * len(results))
    common_chs = [ch for ch, n in ch_counts.items() if n >= threshold]
    print(f"Common channels (≥80% of subjects): {len(common_chs)}")

    # Grand-average per channel per band
    grand = {name: {} for name in BANDS}
    for r in results:
        for name, ratios in r['ratios'].items():
            for c_idx, c in enumerate(r['ch_names']):
                if c in common_chs:
                    grand[name].setdefault(c, []).append(ratios[c_idx])
    # Collapse
    grand_avg = {}
    for name in BANDS:
        grand_avg[name] = {c: np.nanmean(v) for c, v in grand[name].items()}

    # Print top 10 channels per band
    print(f"\n=== Top 10 channels per band (grand-average ratio) ===")
    for name in BANDS:
        print(f"\n  {name}:")
        ordered = sorted(grand_avg[name].items(), key=lambda kv: -kv[1])
        for ch, v in ordered[:10]:
            print(f"    {ch:<6} {v:.3f}×")

    # Save CSV
    rows = []
    for name in BANDS:
        for c, v in grand_avg[name].items():
            rows.append({'band': name, 'channel': c, 'ratio': v})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, '16hz_topography.csv'),
                               index=False)

    # Plot topomaps using MNE
    try:
        raw_ref = load_lemon(results[0]['subject_id'], condition='EC')
        montage = mne.channels.make_standard_montage('standard_1020')
        # Align reference
        info_common = mne.create_info(ch_names=common_chs, sfreq=250,
                                       ch_types='eeg')
        info_common.set_montage(montage, match_case=False, on_missing='ignore')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, name in enumerate(BANDS):
            data = np.array([grand_avg[name].get(c, np.nan) for c in common_chs])
            ax = axes[i]
            im, _ = mne.viz.plot_topomap(data, info_common, axes=ax, show=False,
                                          cmap='viridis',
                                          contours=6)
            ax.set_title(f'{name}\n'
                         f'range {data.min():.2f}-{data.max():.2f}×\n'
                         f'max @ {common_chs[int(np.nanargmax(data))]}',
                         fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.7)
        plt.suptitle(f'B42 — Event-locked topography by band  (LEMON Q4, n={len(results)})',
                     y=1.02, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, '16hz_topography.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {OUT_DIR}/16hz_topography.png")
    except Exception as e:
        print(f"Topomap plot failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
