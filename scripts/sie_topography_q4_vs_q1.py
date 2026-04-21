#!/usr/bin/env python3
"""
B43 — Q4 vs Q1 topography at SR1 / β16 / SR3.

Directly tests whether template_rho stratification spatially separates the
two-network engagement (posterior α + centro-parietal β) seen in B42:

  - Q4 events (canonical ignitions, high template_rho): expect both networks
    clearly visible — posterior SR1 + centro-parietal β16 + central SR3.
  - Q1 events (noise-like, low/negative template_rho): expect flat / absent /
    different spatial pattern.

Produces 6-panel topography figure (3 bands × 2 quartiles).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr
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
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4,
                                 labels=['Q1', 'Q2', 'Q3', 'Q4'])
        q_sub = qual[qual['subject_id'] == sub_id][['t0_net', 'rho_q']].copy()
        q_sub['t0_round'] = q_sub['t0_net'].round(3)
        events['t0_round'] = events['t0_net'].round(3)
        events = events.merge(q_sub[['t0_round', 'rho_q']], on='t0_round', how='left')
        events = events.dropna(subset=['rho_q'])
    except Exception:
        return None
    # Need BOTH Q1 and Q4 for this subject
    q1_events = events[events['rho_q'] == 'Q1']
    q4_events = events[events['rho_q'] == 'Q4']
    if len(q1_events) < 1 or len(q4_events) < 1:
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
    X = raw.get_data() * 1e6

    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band_masks = {k: (freqs >= lo) & (freqs <= hi) for k, (lo, hi) in BANDS.items()}

    # Baseline per channel
    base_cnt = 0
    base_psds = np.zeros((n_ch, len(freqs)))
    for i in range(0, X.shape[1] - nperseg + 1, nhop):
        for c in range(n_ch):
            base_psds[c] += welch_one(X[c, i:i+nperseg], fs, nfft)
        base_cnt += 1
    if base_cnt < 10:
        return None
    base_psds /= base_cnt

    def event_psd_for(sub_events):
        ev_psds = np.zeros((n_ch, len(freqs)))
        cnt = 0
        for _, ev in sub_events.iterrows():
            tc = float(ev['t0_net']) + EV_LAG_S
            i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
            i1 = i0 + nperseg
            if i0 < 0 or i1 > X.shape[1]:
                continue
            for c in range(n_ch):
                ev_psds[c] += welch_one(X[c, i0:i1], fs, nfft)
            cnt += 1
        if cnt < 1:
            return None
        return ev_psds / cnt

    q4_psd = event_psd_for(q4_events)
    q1_psd = event_psd_for(q1_events)
    if q4_psd is None or q1_psd is None:
        return None

    def per_band_ratio(ev_psd):
        out = {}
        for name, mask in band_masks.items():
            r = np.zeros(n_ch)
            for c in range(n_ch):
                r[c] = np.nanmax(ev_psd[c, mask]) / (np.nanmean(base_psds[c, mask]) + 1e-20)
            out[name] = r
        return out

    return {
        'subject_id': sub_id,
        'ch_names': ch_names,
        'q4_ratios': per_band_ratio(q4_psd),
        'q1_ratios': per_band_ratio(q1_psd),
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
    print(f"Successful (subjects with both Q1 and Q4 events): {len(results)}")

    # Common channels ≥80% of subjects
    ch_counts = {}
    for r in results:
        for ch in r['ch_names']:
            ch_counts[ch] = ch_counts.get(ch, 0) + 1
    threshold = int(0.8 * len(results))
    common_chs = [ch for ch, n in ch_counts.items() if n >= threshold]
    print(f"Common channels (≥80%): {len(common_chs)}")

    # Grand averages
    def grand_avg(quartile_key, band_name):
        vals = {ch: [] for ch in common_chs}
        for r in results:
            ratios = r[quartile_key][band_name]
            for i, ch in enumerate(r['ch_names']):
                if ch in common_chs:
                    vals[ch].append(ratios[i])
        return {ch: np.nanmean(v) for ch, v in vals.items()}

    maps = {
        'Q4': {b: grand_avg('q4_ratios', b) for b in BANDS},
        'Q1': {b: grand_avg('q1_ratios', b) for b in BANDS},
    }

    # Print summaries
    print(f"\n=== Max-channel per band × quartile ===")
    for q in ['Q4', 'Q1']:
        for b in BANDS:
            d = maps[q][b]
            top5 = sorted(d.items(), key=lambda kv: -kv[1])[:5]
            vmin = min(d.values()); vmax = max(d.values())
            print(f"  {q} {b}: range [{vmin:.3f}, {vmax:.3f}]  top: "
                  f"{', '.join([f'{k}={v:.2f}' for k, v in top5])}")

    # Q4 vs Q1 topographic correlation per band
    print(f"\n=== Q4 vs Q1 topographic correlation per band ===")
    for b in BANDS:
        q4_vec = np.array([maps['Q4'][b][ch] for ch in common_chs])
        q1_vec = np.array([maps['Q1'][b][ch] for ch in common_chs])
        rho, p = spearmanr(q4_vec, q1_vec)
        print(f"  {b}: ρ = {rho:+.3f} p = {p:.2g}")

    # Save CSV
    rows = []
    for q in ['Q4', 'Q1']:
        for b in BANDS:
            for ch, v in maps[q][b].items():
                rows.append({'quartile': q, 'band': b, 'channel': ch, 'ratio': v})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'topography_q4_vs_q1.csv'),
                                index=False)

    # Plot: 3 × 2 grid
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info_common = mne.create_info(ch_names=common_chs, sfreq=250, ch_types='eeg')
        info_common.set_montage(montage, match_case=False, on_missing='ignore')

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        for qi, q in enumerate(['Q4', 'Q1']):
            for bi, b in enumerate(BANDS):
                ax = axes[qi, bi]
                data = np.array([maps[q][b].get(ch, np.nan) for ch in common_chs])
                im, _ = mne.viz.plot_topomap(data, info_common, axes=ax, show=False,
                                              cmap='viridis', contours=6)
                vmin, vmax = data.min(), data.max()
                ax.set_title(f'{q} — {b}\nrange {vmin:.2f}-{vmax:.2f}×',
                              fontsize=10)
                plt.colorbar(im, ax=ax, shrink=0.6)
        plt.suptitle(f'B43 — Q4 vs Q1 topography (LEMON, n={len(results)} subjects with both)',
                     y=1.00, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'topography_q4_vs_q1.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {OUT_DIR}/topography_q4_vs_q1.png")
    except Exception as e:
        print(f"Topomap failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
