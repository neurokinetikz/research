#!/usr/bin/env python3
"""
B39 — Harmonic Stacking Index: definition disambiguation.

B38 showed ΔHSI = log10(P_SR1/P_SR3) RISES +0.20 at events, opposite direction
from the discovery paper's DIP. Possible explanations:

  (1) Definition inversion: if theirs is log(P_harmonic / P_fundamental),
      the sign flips and both findings agree.
  (2) Different harmonic set: their 6 harmonics include non-SR frequencies
      (7.72, 12.03, 13.86, 20.28, 24.92, 32.44); computing HSI against
      different harmonic sets might flip the observed direction.
  (3) EPOCX vs LEMON genuine difference.

This script computes FOUR HSI variants on the same LEMON Q4 events and
compares time courses:

  V1 (our): HSI = log10(P_SR1 / P_SR3)
  V2 (SR average): HSI = log10(P_SR1 / mean(P_SR2, P_SR3, P_SR4))
  V3 (invert): HSI = log10(mean(P_SR2, P_SR3, P_SR4) / P_SR1)
  V4 (φ-harmonic set): HSI = log10(P_SR1 / mean(P_φ1, P_φ2, P_φ3))
      where P_φn = power at 7.82 × φⁿ

Also: per-event heterogeneity — are all events rising together, or is it a
mix of rising/dipping events that grand-averages to a rise?
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

FRONTAL_CHS = ['F3', 'F4', 'Fz', 'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8']

# Actual measured Schumann modes (from seasonal/diurnal maps)
SR_BANDS = {
    'SR1': (7.2, 8.4),
    'SR2': (13.5, 14.5),
    'SR3': (19.5, 20.4),
    'SR4': (25.0, 26.0),
}

# φⁿ × SR1 bands
PHI = 1.6180339887
PHI_HARM_BANDS = {
    f'phi{n}': (7.82 * PHI ** n - 0.6, 7.82 * PHI ** n + 0.6)
    for n in [1, 2, 3]
}

WIN_SEC = 2.0
HOP_SEC = 0.5
TGRID = np.arange(-15.0, 15.0 + 0.25, 0.5)

MIN_GAP_FROM_EVENT = 30.0


def sliding_band_powers(y, fs, bands, win_sec=WIN_SEC, hop_sec=HOP_SEC):
    nperseg = int(round(win_sec * fs))
    nhop = int(round(hop_sec * fs))
    nfft = nperseg * 4
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    w = signal.windows.hann(nperseg)
    wp = np.sum(w ** 2)
    masks = {name: (freqs >= lo) & (freqs <= hi) for name, (lo, hi) in bands.items()}
    t_cent = []
    powers = {name: [] for name in bands}
    for i in range(0, len(y) - nperseg + 1, nhop):
        seg = y[i:i + nperseg] - np.mean(y[i:i + nperseg])
        X = np.fft.rfft(seg * w, nfft)
        psd = (np.abs(X) ** 2) / (fs * wp)
        psd[1:-1] *= 2.0
        for name, m in masks.items():
            powers[name].append(np.nanmean(psd[m]))
        t_cent.append((i + nperseg / 2) / fs)
    return np.array(t_cent), {k: np.array(v) for k, v in powers.items()}


def sample_control_times(t_events, n, t_end, seed=0):
    rng = np.random.default_rng(seed)
    lo = WIN_SEC / 2 + 1
    hi = t_end - WIN_SEC / 2 - 1
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        t = rng.uniform(lo, hi)
        if len(t_events) == 0 or np.min(np.abs(t - t_events)) >= MIN_GAP_FROM_EVENT:
            out.append(t)
        tries += 1
    return np.array(out)


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
    if fs < 130:
        return None
    present = [c for c in FRONTAL_CHS if c in raw.ch_names]
    if len(present) < 2:
        return None
    y = raw.get_data(picks=present).mean(axis=0) * 1e6
    t_end = raw.times[-1]

    all_bands = {**SR_BANDS, **PHI_HARM_BANDS}
    t_pow, pow_all = sliding_band_powers(y, fs, all_bands)

    # Compute 4 HSI variants as time series
    eps = 1e-20
    hsi_V1 = np.log10(pow_all['SR1'] + eps) - np.log10(pow_all['SR3'] + eps)
    hsi_V2 = np.log10(pow_all['SR1'] + eps) - np.log10(
        (pow_all['SR2'] + pow_all['SR3'] + pow_all['SR4']) / 3 + eps)
    hsi_V3 = -hsi_V2
    phi_mean = (pow_all['phi1'] + pow_all['phi2'] + pow_all['phi3']) / 3
    hsi_V4 = np.log10(pow_all['SR1'] + eps) - np.log10(phi_mean + eps)

    # Center each variant on its overall median
    def center(x):
        return x - np.nanmedian(x)
    V1c = center(hsi_V1); V2c = center(hsi_V2); V3c = center(hsi_V3); V4c = center(hsi_V4)

    t_events = events['t0_net'].astype(float).values
    t_controls = sample_control_times(t_events, len(t_events), t_end,
                                         seed=abs(hash(sub_id)) % (2**31))

    def align_all(t_center):
        rel = t_pow - t_center
        mask = (rel >= TGRID[0] - 1) & (rel <= TGRID[-1] + 1)
        if mask.sum() == 0:
            return None
        out = {}
        for name, arr in [('V1', V1c), ('V2', V2c), ('V3', V3c), ('V4', V4c)]:
            out[name] = np.interp(TGRID, rel[mask], arr[mask],
                                   left=np.nan, right=np.nan)
        return out

    ev_traces = {'V1': [], 'V2': [], 'V3': [], 'V4': []}
    ct_traces = {'V1': [], 'V2': [], 'V3': [], 'V4': []}
    for t0 in t_events:
        d = align_all(t0)
        if d:
            for k in ev_traces:
                ev_traces[k].append(d[k])
    for tc in t_controls:
        d = align_all(tc)
        if d:
            for k in ct_traces:
                ct_traces[k].append(d[k])
    if not ev_traces['V1'] or not ct_traces['V1']:
        return None
    return {
        'subject_id': sub_id,
        'n_ev': len(ev_traces['V1']),
        'n_ct': len(ct_traces['V1']),
        'ev_V1': np.nanmean(np.array(ev_traces['V1']), axis=0),
        'ev_V2': np.nanmean(np.array(ev_traces['V2']), axis=0),
        'ev_V3': np.nanmean(np.array(ev_traces['V3']), axis=0),
        'ev_V4': np.nanmean(np.array(ev_traces['V4']), axis=0),
        'ct_V1': np.nanmean(np.array(ct_traces['V1']), axis=0),
        'ct_V2': np.nanmean(np.array(ct_traces['V2']), axis=0),
        'ct_V3': np.nanmean(np.array(ct_traces['V3']), axis=0),
        'ct_V4': np.nanmean(np.array(ct_traces['V4']), axis=0),
    }


def bootstrap_ci(mat, n_boot=1000, seed=0):
    mat = np.array(mat)
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    if len(mat) < 2:
        return (np.nanmean(mat, axis=0),
                np.full(mat.shape[1], np.nan),
                np.full(mat.shape[1], np.nan))
    rng = np.random.default_rng(seed)
    grand = np.nanmean(mat, axis=0)
    boots = np.zeros((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, len(mat), size=len(mat))
        boots[b] = np.nanmean(mat[idx], axis=0)
    return grand, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


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

    labels = {
        'V1': 'V1: log(SR1/SR3)',
        'V2': 'V2: log(SR1 / mean(SR2,SR3,SR4))',
        'V3': 'V3: log(mean(SR2,SR3,SR4)/SR1) — inverse of V2',
        'V4': 'V4: log(SR1 / mean(φ¹,φ²,φ³ × SR1))',
    }

    # Print peak values around t=0-4s for each variant
    print(f"\n=== Peak ΔHSI at t=0-4s and dip for each variant ===")
    for v in ['V1', 'V2', 'V3', 'V4']:
        ev = np.array([r[f'ev_{v}'] for r in results])
        ct = np.array([r[f'ct_{v}'] for r in results])
        grand_ev, _, _ = bootstrap_ci(ev)
        grand_ct, _, _ = bootstrap_ci(ct)
        peri_mask = (TGRID >= 0) & (TGRID <= 4)
        ev_peak = np.nanmax(grand_ev[peri_mask])
        ev_peak_t = TGRID[peri_mask][np.nanargmax(grand_ev[peri_mask])]
        ev_dip = np.nanmin(grand_ev[peri_mask])
        ev_dip_t = TGRID[peri_mask][np.nanargmin(grand_ev[peri_mask])]
        ct_peak = np.nanmax(grand_ct[peri_mask])
        ct_dip = np.nanmin(grand_ct[peri_mask])
        print(f"  {labels[v]}")
        print(f"    peak: {ev_peak:+.4f} @ t={ev_peak_t:+.1f}s (ctl {ct_peak:+.4f})   "
              f"dip: {ev_dip:+.4f} @ t={ev_dip_t:+.1f}s (ctl {ct_dip:+.4f})")

    # Plot all 4 variants
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax_idx, v in enumerate(['V1', 'V2', 'V3', 'V4']):
        ax = axes.flatten()[ax_idx]
        ev = np.array([r[f'ev_{v}'] for r in results])
        ct = np.array([r[f'ct_{v}'] for r in results])
        ge, loe, hie = bootstrap_ci(ev)
        gc, loc, hic = bootstrap_ci(ct)
        ax.plot(TGRID, ge, color='firebrick', lw=2, label='event')
        ax.fill_between(TGRID, loe, hie, color='firebrick', alpha=0.22)
        ax.plot(TGRID, gc, color='gray', lw=2, label='control')
        ax.fill_between(TGRID, loc, hic, color='gray', alpha=0.22)
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', ls='--', lw=0.6)
        ax.set_xlabel('time rel. t0_net (s)')
        ax.set_ylabel('ΔHSI (centered on own median)')
        ax.set_title(labels[v])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle(f'B39 — HSI direction under four definitions  (n={len(results)})',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hsi_variants.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # Per-event heterogeneity on V1: how many events show rise vs dip at t=+2s?
    print(f"\n=== Per-subject distribution of event ΔHSI_V1 at t=+2s ===")
    t2_idx = int(np.argmin(np.abs(TGRID - 2.0)))
    ev_V1_subj = np.array([r['ev_V1'][t2_idx] for r in results])
    n_rise = np.sum(ev_V1_subj > 0)
    n_dip = np.sum(ev_V1_subj < 0)
    print(f"  Subjects with event HSI RISE at t=+2: {n_rise} / {len(ev_V1_subj)} "
          f"({100*n_rise/len(ev_V1_subj):.0f}%)")
    print(f"  Subjects with event HSI DIP  at t=+2: {n_dip} / {len(ev_V1_subj)} "
          f"({100*n_dip/len(ev_V1_subj):.0f}%)")
    print(f"  Subject median: {np.nanmedian(ev_V1_subj):+.4f}")

    # Which bands actually drove the result? Look at individual band log-changes
    print(f"\n=== Per-band log-ratio rise at t=+2s (event − baseline t=-6s) ===")
    # Re-extract per band mean event trace (recompute band-wise)
    # Skip — data not saved per-band. Can infer: V1 rise (+0.2), V2 rise (+0.24 expected),
    # V3 dip (-0.24 expected), V4 depends on how SR1 grows vs φⁿ harmonics

    # Save summary CSV
    rows = []
    for r in results:
        for v in ['V1', 'V2', 'V3', 'V4']:
            for ti, t in enumerate(TGRID):
                rows.append({'subject_id': r['subject_id'], 'variant': v,
                              't': t, 'event': r[f'ev_{v}'][ti], 'control': r[f'ct_{v}'][ti]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'hsi_variants.csv'),
                               index=False)
    print(f"\nSaved: {OUT_DIR}/hsi_variants.png")


if __name__ == '__main__':
    main()
