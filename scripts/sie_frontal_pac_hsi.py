#!/usr/bin/env python3
"""
B38 — Frontal-channel time-resolved PAC (MVL) + Harmonic Stacking Index (HSI).

Two analyses from the discovery paper's figures:

  (C) Frontal-channel PAC: restrict to F3/F4/Fz mean-signal (vs prior
      all-channel mean). Tests whether the B36 null was caused by spatial
      averaging that diluted frontal θ-γ PAC. Uses Canolty MVL (matching
      the discovery paper's metric).

  (A) Harmonic Stacking Index (HSI): sliding-window log ratio
      HSI(t) = log10(P_SR1(t) / P_SR3(t))
      Larger = SR1 fundamental more dominant. Discovery paper panel D shows
      ΔHSI dipping ~−0.15 at ignition — non-fundamental harmonics surge
      relative to fundamental during the event.

Windows: PAC computed in 6-s windows centered at −6, 0, +6, +12, +18, +24 s
rel. t0_net. HSI as continuous time course on same grid.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import wilcoxon
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

# Frontal channels (prefer these; fall back to what's available)
FRONTAL_PREFERRED = ['F3', 'F4', 'Fz', 'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8']

# PAC bands
THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (30.0, 60.0)

# HSI bands
SR1_BAND = (7.2, 8.4)
SR3_BAND = (19.5, 20.4)

# Analysis windows
WIN_SEC = 6.0
WIN_CENTERS = np.array([-6.0, 0.0, 6.0, 12.0, 18.0, 24.0])
MIN_GAP_FROM_EVENT = 30.0

# HSI time grid (for continuous time-course)
HSI_HOP_SEC = 0.5
HSI_WIN_SEC = 2.0
HSI_TGRID = np.arange(-15.0, 15.0 + 0.25, 0.5)


def bandpass(x, fs, lo, hi, order=4):
    ny = 0.5 * fs
    lo = max(0.1, lo); hi = min(ny - 1e-3, hi)
    if lo >= hi:
        return np.zeros_like(x)
    b, a = signal.butter(order, [lo / ny, hi / ny], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


def canolty_mvl(phase, amp):
    """Canolty mean-vector length PAC: |< A(t) · exp(i·φ(t)) >|"""
    return float(np.abs(np.mean(amp * np.exp(1j * phase))))


def window_pac_mvl(y_win, fs):
    phase = np.angle(signal.hilbert(bandpass(y_win, fs, *THETA_BAND)))
    amp = np.abs(signal.hilbert(bandpass(y_win, fs, *GAMMA_BAND)))
    return canolty_mvl(phase, amp)


def sliding_log_ratio(y, fs, band_a, band_b, win_sec=HSI_WIN_SEC,
                       hop_sec=HSI_HOP_SEC):
    """Sliding-window power in two bands, return log10(P_a / P_b) and times."""
    nperseg = int(round(win_sec * fs))
    nhop = int(round(hop_sec * fs))
    nfft = nperseg * 4
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    w = signal.windows.hann(nperseg)
    wp = np.sum(w ** 2)
    mask_a = (freqs >= band_a[0]) & (freqs <= band_a[1])
    mask_b = (freqs >= band_b[0]) & (freqs <= band_b[1])
    out_hsi, tc = [], []
    for i in range(0, len(y) - nperseg + 1, nhop):
        seg = y[i:i + nperseg] - np.mean(y[i:i + nperseg])
        X = np.fft.rfft(seg * w, nfft)
        psd = (np.abs(X) ** 2) / (fs * wp)
        psd[1:-1] *= 2.0
        pa = np.nanmean(psd[mask_a])
        pb = np.nanmean(psd[mask_b])
        out_hsi.append(np.log10(pa + 1e-20) - np.log10(pb + 1e-20))
        tc.append((i + nperseg / 2) / fs)
    return np.array(tc), np.array(out_hsi)


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
    # Q4 filter
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
    # Pick frontal channels
    present = [c for c in FRONTAL_PREFERRED if c in raw.ch_names]
    if len(present) < 2:
        return None
    data = raw.get_data(picks=present) * 1e6
    y_frontal = data.mean(axis=0)
    t_end = raw.times[-1]

    t_events = events['t0_net'].astype(float).values
    t_controls = sample_control_times(t_events, len(t_events), t_end,
                                         seed=abs(hash(sub_id)) % (2**31))

    nperseg = int(round(WIN_SEC * fs))

    # PAC at each time window
    def pac_at(t_center):
        i0 = int(round((t_center - WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y_frontal):
            return np.nan
        return window_pac_mvl(y_frontal[i0:i1], fs)

    pac_ev = np.full((len(t_events), len(WIN_CENTERS)), np.nan)
    for ei, t0 in enumerate(t_events):
        for wi, wc in enumerate(WIN_CENTERS):
            pac_ev[ei, wi] = pac_at(t0 + wc)
    pac_ct = np.full((len(t_controls), len(WIN_CENTERS)), np.nan)
    for ci, tc in enumerate(t_controls):
        for wi, wc in enumerate(WIN_CENTERS):
            pac_ct[ci, wi] = pac_at(tc + wc)
    pac_ev_mean = np.nanmean(pac_ev, axis=0)
    pac_ct_mean = np.nanmean(pac_ct, axis=0)

    # HSI time course — per-event, aligned on t0 in HSI_TGRID
    t_hsi, hsi = sliding_log_ratio(y_frontal, fs, SR1_BAND, SR3_BAND)
    if len(hsi) < 10:
        return None
    hsi_median = np.nanmedian(hsi)
    hsi_centered = hsi - hsi_median

    hsi_ev_traces = []
    for t0 in t_events:
        rel = t_hsi - t0
        mask = (rel >= HSI_TGRID[0] - 1) & (rel <= HSI_TGRID[-1] + 1)
        if mask.sum() == 0:
            continue
        hsi_ev_traces.append(np.interp(HSI_TGRID, rel[mask], hsi_centered[mask],
                                        left=np.nan, right=np.nan))
    hsi_ct_traces = []
    for tc in t_controls:
        rel = t_hsi - tc
        mask = (rel >= HSI_TGRID[0] - 1) & (rel <= HSI_TGRID[-1] + 1)
        if mask.sum() == 0:
            continue
        hsi_ct_traces.append(np.interp(HSI_TGRID, rel[mask], hsi_centered[mask],
                                        left=np.nan, right=np.nan))
    if not hsi_ev_traces or not hsi_ct_traces:
        return None

    return {
        'subject_id': sub_id,
        'n_ev': len(t_events),
        'n_ct': len(t_controls),
        'frontal_chs': present[:3],
        'pac_ev': pac_ev_mean,
        'pac_ct': pac_ct_mean,
        'hsi_ev': np.nanmean(np.array(hsi_ev_traces), axis=0),
        'hsi_ct': np.nanmean(np.array(hsi_ct_traces), axis=0),
    }


def bootstrap_ci(mat, n_boot=1000, seed=0):
    mat = np.array(mat)
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    if len(mat) < 2:
        return np.nanmean(mat, axis=0), np.full(mat.shape[1], np.nan), np.full(mat.shape[1], np.nan)
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
    if not results:
        return

    # PAC aggregate
    pac_ev = np.array([r['pac_ev'] for r in results])
    pac_ct = np.array([r['pac_ct'] for r in results])
    print(f"\n=== Frontal θ (4-8) × γ (30-60) PAC (Canolty MVL) time course ===")
    print(f"{'t (s)':>8}  {'event MVL':>10}  {'control':>10}  {'Δ':>10}  {'p':>10}")
    for wi, wc in enumerate(WIN_CENTERS):
        e = pac_ev[:, wi]; c = pac_ct[:, wi]; d = e - c
        d = d[np.isfinite(d)]
        _, p = wilcoxon(d) if len(d) >= 5 and np.any(d != 0) else (np.nan, np.nan)
        print(f"{wc:>+8.1f}  {np.nanmean(e):>10.4f}  {np.nanmean(c):>10.4f}  "
              f"{np.nanmean(e-c):>+10.4f}  {p:>10.3g}")

    # HSI aggregate
    hsi_ev = np.array([r['hsi_ev'] for r in results])
    hsi_ct = np.array([r['hsi_ct'] for r in results])
    hsi_ev_grand, hsi_ev_lo, hsi_ev_hi = bootstrap_ci(hsi_ev)
    hsi_ct_grand, hsi_ct_lo, hsi_ct_hi = bootstrap_ci(hsi_ct)

    # HSI dip metric: minimum ΔHSI in [-3, +3] s
    dip_win = (HSI_TGRID >= -3) & (HSI_TGRID <= 3)
    ev_dip = np.nanmin(hsi_ev[:, dip_win], axis=1)
    ct_dip = np.nanmin(hsi_ct[:, dip_win], axis=1)
    d = ev_dip - ct_dip
    d = d[np.isfinite(d)]
    _, p_dip = wilcoxon(d) if len(d) >= 5 and np.any(d != 0) else (np.nan, np.nan)
    print(f"\n=== Harmonic Stacking Index (SR1/SR3 log-ratio) — peri-event dip ===")
    print(f"  Event min ΔHSI in [-3,+3]s: {np.nanmean(ev_dip):.4f}")
    print(f"  Control min ΔHSI in [-3,+3]s: {np.nanmean(ct_dip):.4f}")
    print(f"  Δ (event − control): {np.nanmean(ev_dip - ct_dip):.4f}   p = {p_dip:.3g}")

    # Print some HSI landmark values
    print(f"\n=== HSI ΔHSI at key times ===")
    for t in [-6, -2, 0, 2, 6, 12]:
        idx = int(np.argmin(np.abs(HSI_TGRID - t)))
        print(f"  t = {t:>+4}s  event {hsi_ev_grand[idx]:+.4f}  control {hsi_ct_grand[idx]:+.4f}  "
              f"Δ {hsi_ev_grand[idx] - hsi_ct_grand[idx]:+.4f}")

    # Save
    out_rows = []
    for r in results:
        for wi, wc in enumerate(WIN_CENTERS):
            out_rows.append({'subject_id': r['subject_id'], 't_s': wc,
                              'pac_ev': r['pac_ev'][wi], 'pac_ct': r['pac_ct'][wi]})
    pd.DataFrame(out_rows).to_csv(os.path.join(OUT_DIR, 'frontal_pac_mvl.csv'), index=False)
    np.savez(os.path.join(OUT_DIR, 'frontal_hsi.npz'),
             t=HSI_TGRID, ev_grand=hsi_ev_grand, ct_grand=hsi_ct_grand,
             ev_lo=hsi_ev_lo, ev_hi=hsi_ev_hi, ct_lo=hsi_ct_lo, ct_hi=hsi_ct_hi)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    pac_ev_mean = np.nanmean(pac_ev, axis=0)
    pac_ev_sem  = np.nanstd(pac_ev, axis=0) / np.sqrt(np.sum(np.isfinite(pac_ev), axis=0))
    pac_ct_mean = np.nanmean(pac_ct, axis=0)
    pac_ct_sem  = np.nanstd(pac_ct, axis=0) / np.sqrt(np.sum(np.isfinite(pac_ct), axis=0))
    ax.errorbar(WIN_CENTERS, pac_ev_mean, yerr=pac_ev_sem, fmt='o-',
                 color='firebrick', label='event')
    ax.errorbar(WIN_CENTERS, pac_ct_mean, yerr=pac_ct_sem, fmt='o-',
                 color='gray', label='control')
    ax.axvline(0, color='k', ls='--', lw=0.6)
    ax.set_xlabel('time rel. t0_net (s)')
    ax.set_ylabel('Canolty MVL (θ × γ)')
    ax.set_title(f'Frontal θ(4-8) × γ(30-60) PAC (n={len(results)})')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(HSI_TGRID, hsi_ev_grand, color='firebrick', lw=2, label='event')
    ax.fill_between(HSI_TGRID, hsi_ev_lo, hsi_ev_hi, color='firebrick', alpha=0.22)
    ax.plot(HSI_TGRID, hsi_ct_grand, color='gray', lw=2, label='control')
    ax.fill_between(HSI_TGRID, hsi_ct_lo, hsi_ct_hi, color='gray', alpha=0.22)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', ls='--', lw=0.6)
    ax.set_xlabel('time rel. t0_net (s)')
    ax.set_ylabel('ΔHSI = log10(P_SR1 / P_SR3)')
    ax.set_title(f'Harmonic Stacking Index time course (n={len(results)})')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'B38 — Frontal PAC (MVL) + HSI time course · LEMON Q4',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'frontal_pac_hsi.png'),
                 dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/frontal_pac_hsi.png")


if __name__ == '__main__':
    main()
