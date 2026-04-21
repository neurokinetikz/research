#!/usr/bin/env python3
"""
B46 — Posterior-restricted SR1 test: strengthen the surviving B45 claim.

After B45, the robust individual-level finding is SR1 (~7.82 Hz) engagement
at Q4 events with reliable posterior topography. This analysis re-runs
B20-style IAF-independence AND B45 reliability using only posterior channels
(PO/P/O/TP), to show the surviving claim holds in its actual spatial
substrate.

Three outputs:
  1. Posterior-restricted IAF × SR1 scatter (mirror of B20 Fig 1)
  2. Posterior-only subject-to-group reliability histogram
  3. Summary stats suitable for a clean revised Figure 3

Posterior channels (inclusive): any channel whose label starts with
  'O', 'PO', 'P', 'TP', 'POz' (and specific 'T7', 'T8' which often sit
  posteriorly on 10-20 montages).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, pearsonr, wilcoxon
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


def is_posterior(ch_name):
    prefixes_yes = ('O', 'PO', 'P', 'TP')
    # Exclude Fp-prefixed frontopolar ("FP1", "FP2", "Fpz")
    n = ch_name.upper()
    if n.startswith('FP'):
        return False
    return any(n.startswith(p) for p in prefixes_yes) or n in ('T7', 'T8', 'T5', 'T6')


# PSD params
EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
IAF_WIN_SEC = 8.0
IAF_HOP_SEC = 2.0
IAF_NFFT_MULT = 4
SR1_RANGE = (6.5, 9.0)
IAF_RANGE = (7.0, 13.0)
SCHUMANN_F = 7.83


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    w = signal.windows.hann(len(seg))
    wp = np.sum(w ** 2)
    X = np.fft.rfft(seg * w, nfft)
    psd = (np.abs(X) ** 2) / (fs * wp)
    psd[1:-1] *= 2.0
    return psd


def parabolic_peak(y, x):
    k = int(np.argmax(y))
    if 1 <= k < len(y) - 1 and y[k-1] > 0 and y[k+1] > 0:
        y0, y1, y2 = y[k-1], y[k], y[k+1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = max(-1.0, min(1.0, delta))
        return float(x[k] + delta * (x[1] - x[0]))
    return float(x[k])


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
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
    post_idx = [i for i, ch in enumerate(ch_names) if is_posterior(ch)]
    post_chs = [ch_names[i] for i in post_idx]
    if len(post_idx) < 5:
        return None
    X = raw.get_data() * 1e6
    post_signal = X[post_idx].mean(axis=0)

    # IAF on posterior-mean signal, same as B20 pipeline
    nper_iaf = int(round(IAF_WIN_SEC * fs))
    nhop_iaf = int(round(IAF_HOP_SEC * fs))
    nfft_iaf = nper_iaf * IAF_NFFT_MULT
    freqs_iaf = np.fft.rfftfreq(nfft_iaf, 1.0 / fs)
    mask_iaf = (freqs_iaf >= IAF_RANGE[0]) & (freqs_iaf <= IAF_RANGE[1])
    psds = []
    for i in range(0, len(post_signal) - nper_iaf + 1, nhop_iaf):
        psds.append(welch_one(post_signal[i:i+nper_iaf], fs, nfft_iaf)[mask_iaf])
    if len(psds) < 3:
        return None
    iaf_grand = np.nanmedian(np.array(psds), axis=0)
    iaf = parabolic_peak(iaf_grand, freqs_iaf[mask_iaf])

    # Event-locked posterior SR1 peak
    nper_ev = int(round(EV_WIN_SEC * fs))
    nhop_ev = int(round(1.0 * fs))
    nfft_ev = nper_ev * EV_NFFT_MULT
    freqs_ev = np.fft.rfftfreq(nfft_ev, 1.0 / fs)
    mask_sr1 = (freqs_ev >= SR1_RANGE[0]) & (freqs_ev <= SR1_RANGE[1])
    f_sr1 = freqs_ev[mask_sr1]

    base_rows = []
    for i in range(0, len(post_signal) - nper_ev + 1, nhop_ev):
        base_rows.append(welch_one(post_signal[i:i+nper_ev], fs, nfft_ev)[mask_sr1])
    if len(base_rows) < 10:
        return None
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    ev_rows = []
    for _, ev in events.iterrows():
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nper_ev
        if i0 < 0 or i1 > len(post_signal):
            continue
        ev_rows.append(welch_one(post_signal[i0:i1], fs, nfft_ev)[mask_sr1])
    if not ev_rows:
        return None
    ev_avg = np.nanmean(np.array(ev_rows), axis=0)
    ratio = (ev_avg + 1e-20) / (baseline + 1e-20)
    sr1_peak = parabolic_peak(ratio, f_sr1)
    sr1_peak_ratio = float(ratio[int(np.argmax(ratio))])

    # Per-channel posterior topography (Q4) for reliability test
    post_topo = np.zeros(len(post_idx))
    base_post_psd = np.zeros((len(post_idx), len(freqs_ev)))
    ev_post_psd = np.zeros((len(post_idx), len(freqs_ev)))
    n_base = 0
    for i in range(0, X.shape[1] - nper_ev + 1, nhop_ev):
        for k, c_idx in enumerate(post_idx):
            base_post_psd[k] += welch_one(X[c_idx, i:i+nper_ev], fs, nfft_ev)
        n_base += 1
    base_post_psd /= n_base
    n_ev = 0
    for _, ev in events.iterrows():
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nper_ev
        if i0 < 0 or i1 > X.shape[1]: continue
        for k, c_idx in enumerate(post_idx):
            ev_post_psd[k] += welch_one(X[c_idx, i0:i1], fs, nfft_ev)
        n_ev += 1
    if n_ev < 1:
        return None
    ev_post_psd /= n_ev
    sr1_band_mask = (freqs_ev >= 7.0) & (freqs_ev <= 8.3)
    for k in range(len(post_idx)):
        post_topo[k] = (np.nanmax(ev_post_psd[k, sr1_band_mask])
                         / (np.nanmean(base_post_psd[k, sr1_band_mask]) + 1e-20))

    return {
        'subject_id': sub_id,
        'n_events': n_ev,
        'n_post_chs': len(post_idx),
        'iaf_hz': iaf,
        'sr1_peak_hz': sr1_peak,
        'sr1_peak_ratio': sr1_peak_ratio,
        'posterior_chs': post_chs,
        'post_topo': post_topo,
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status']=='ok') & (summary['n_events']>=3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    iaf = np.array([r['iaf_hz'] for r in results])
    sr1 = np.array([r['sr1_peak_hz'] for r in results])
    ratio = np.array([r['sr1_peak_ratio'] for r in results])

    # Test 1: IAF-independence (posterior-restricted)
    print(f"\n{'='*64}")
    print("Test 1: Posterior-restricted IAF × SR1 peak correlation")
    print(f"{'='*64}")
    rho, p = spearmanr(iaf, sr1)
    r, _ = pearsonr(iaf, sr1)
    slope, intercept = np.polyfit(iaf, sr1, 1)
    print(f"  IAF (posterior): mean {iaf.mean():.3f} Hz, std {iaf.std():.3f}")
    print(f"  SR1 peak (posterior): mean {sr1.mean():.3f} Hz, std {sr1.std():.3f}")
    print(f"  Spearman ρ = {rho:+.3f} p = {p:.3g}   (H1 IAF-lock: ρ≈1; H2 fixed: ρ≈0)")
    print(f"  Pearson r  = {r:+.3f}")
    print(f"  OLS slope = {slope:+.3f}   intercept = {intercept:+.3f}")
    print(f"  Prior B20 result (mean-channel): ρ = 0.029, slope = 0.015")

    # Test 2: posterior-only subject-to-group reliability
    print(f"\n{'='*64}")
    print("Test 2: Posterior-only subject-to-group SR1 topographic reliability")
    print(f"{'='*64}")
    # Each subject has different posterior channels; use common ones (≥80%)
    ch_counts = {}
    for r in results:
        for ch in r['posterior_chs']:
            ch_counts[ch] = ch_counts.get(ch, 0) + 1
    thr = int(0.8 * len(results))
    common = [ch for ch, n in ch_counts.items() if n >= thr]
    print(f"  Common posterior channels (≥80%): {len(common)}  {common}")

    # Subject topo vectors restricted to common channels
    topos = {}
    for r in results:
        vec = np.array([
            r['post_topo'][r['posterior_chs'].index(ch)]
            if ch in r['posterior_chs'] else np.nan
            for ch in common
        ])
        topos[r['subject_id']] = vec

    all_vecs = np.array([topos[s] for s in topos])
    rhos = []
    for i, s in enumerate(topos):
        others = np.delete(all_vecs, i, axis=0)
        group_mean = np.nanmean(others, axis=0)
        v = all_vecs[i]
        good = np.isfinite(v) & np.isfinite(group_mean)
        if good.sum() < 4:
            continue
        rho_i, _ = spearmanr(v[good], group_mean[good])
        rhos.append(rho_i)
    rhos = np.array(rhos)
    print(f"  n = {len(rhos)}")
    print(f"  subject-to-group ρ: median {np.median(rhos):+.3f}  "
          f"IQR [{np.percentile(rhos,25):+.3f}, {np.percentile(rhos,75):+.3f}]")
    print(f"  pct > 0: {(rhos>0).mean()*100:.0f}%  pct > 0.5: {(rhos>0.5).mean()*100:.0f}%")
    print(f"  B45 all-channel result: median ρ = 0.52, pct > 0.5 = 55%")

    # Save CSV
    pd.DataFrame({
        'subject_id': [r['subject_id'] for r in results],
        'iaf_hz_posterior': iaf,
        'sr1_peak_hz_posterior': sr1,
        'sr1_peak_ratio_posterior': ratio,
        'n_events': [r['n_events'] for r in results],
        'n_post_chs': [r['n_post_chs'] for r in results],
    }).to_csv(os.path.join(OUT_DIR, 'posterior_sr1_tightened.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.scatter(iaf, sr1, s=30, alpha=0.55, color='steelblue', edgecolor='k', lw=0.3)
    rng = np.array([iaf.min() - 0.3, iaf.max() + 0.3])
    ax.plot(rng, rng, 'k--', lw=1.0, label='IAF lock (y = x)')
    ax.axhline(SCHUMANN_F, color='#1a9641', ls=':', lw=1.0,
                label=f'Schumann {SCHUMANN_F}')
    ax.plot(rng, slope * rng + intercept, color='red', lw=1.5,
             label=f'OLS {slope:+.2f}·IAF {intercept:+.2f}')
    ax.set_xlabel('IAF (posterior, Hz)')
    ax.set_ylabel('SR1 peak (posterior, Hz)')
    ax.set_title(f'Posterior IAF-independence  ρ={rho:+.2f} p={p:.2g}  n={len(results)}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(rhos, bins=np.linspace(-1, 1, 30), color='firebrick',
             edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(np.median(rhos), color='blue', ls='--', lw=1.5,
                label=f'median {np.median(rhos):+.2f}')
    ax.axvline(0, color='k', lw=0.6)
    ax.axvline(0.5, color='green', ls=':', lw=0.8, alpha=0.7,
                label='ρ = 0.5')
    ax.set_xlabel('subject-to-group ρ (posterior-only, SR1 topography)')
    ax.set_ylabel('subjects')
    ax.set_title(f'Per-subject SR1 posterior reliability  n={len(rhos)}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(iaf, ratio, s=30, alpha=0.55, color='seagreen',
                edgecolor='k', lw=0.3)
    ax.set_xlabel('IAF (posterior, Hz)')
    ax.set_ylabel('SR1 peak ratio × baseline')
    rho_r, p_r = spearmanr(iaf, ratio)
    ax.set_title(f'Posterior SR1 amplitude × IAF  ρ={rho_r:+.2f} p={p_r:.2g}')
    ax.grid(alpha=0.3)

    plt.suptitle(f'B46 — Posterior-restricted SR1 tightened '
                 f'(LEMON Q4, n={len(results)})', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'posterior_sr1_tightened.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/posterior_sr1_tightened.png")


if __name__ == '__main__':
    main()
