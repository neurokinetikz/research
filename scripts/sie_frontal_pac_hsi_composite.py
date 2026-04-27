#!/usr/bin/env python3
"""
B38 re-run on composite v2 detector.

Per composite Q4 event:
  (C) Frontal θ(4-8) × γ(30-60) PAC via Canolty MVL at 6 time windows
      centered at −6, 0, +6, +12, +18, +24 s rel t0_net. Event vs control.
  (A) HSI(t) = log10(P_SR1(t) / P_SR3(t)) sliding window time course.
      Discovery paper showed peri-event dip; envelope B38 showed RISE
      (+0.20 at t=+2s — SR1 gains dominance over SR3 at events).

Envelope B38 findings:
  - Frontal PAC: null (Δ ≤ 0.003 at every window)
  - HSI: rises +0.20 at t=+2s (SR1 dominance ~2× over SR3 at events)

Composite context: §39 B33 showed SR3 is weaker under composite LEMON EC
(flat vs envelope's +0.06 log-excess). The HSI rise direction should still
hold, possibly amplified since composite EC has no SR3 rebound.

Cohort-parameterized. Uses Q4 events (canonical) per template_ρ.

Usage:
    python scripts/sie_frontal_pac_hsi_composite.py --cohort lemon
    python scripts/sie_frontal_pac_hsi_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
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
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

FRONTAL_PREFERRED = ['F3', 'F4', 'Fz', 'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8']

THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (30.0, 60.0)
SR1_BAND = (7.2, 8.4)
SR3_BAND = (19.5, 20.4)

WIN_SEC = 6.0
WIN_CENTERS = np.array([-6.0, 0.0, 6.0, 12.0, 18.0, 24.0])
MIN_GAP_FROM_EVENT = 30.0

HSI_HOP_SEC = 0.5
HSI_WIN_SEC = 2.0
HSI_TGRID = np.arange(-15.0, 15.0 + 0.25, 0.5)

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


def bandpass(x, fs, lo, hi, order=4):
    ny = 0.5 * fs
    lo = max(0.1, lo); hi = min(ny - 1e-3, hi)
    if lo >= hi:
        return np.zeros_like(x)
    b, a = signal.butter(order, [lo / ny, hi / ny], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


def canolty_mvl(phase, amp):
    return float(np.abs(np.mean(amp * np.exp(1j * phase))))


def window_pac_mvl(y_win, fs):
    phase = np.angle(signal.hilbert(bandpass(y_win, fs, *THETA_BAND)))
    amp = np.abs(signal.hilbert(bandpass(y_win, fs, *GAMMA_BAND)))
    return canolty_mvl(phase, amp)


def sliding_log_ratio(y, fs, band_a, band_b, win_sec=HSI_WIN_SEC,
                       hop_sec=HSI_HOP_SEC):
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
    if fs < 130:
        return None
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
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}  (Q4 only)")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")
    if not results:
        return

    pac_ev = np.array([r['pac_ev'] for r in results])
    pac_ct = np.array([r['pac_ct'] for r in results])
    print(f"\n=== {args.cohort} composite · Frontal θ × γ PAC (MVL) time course ===")
    print(f"(envelope B38: null — Δ ≤ 0.003 at every window)")
    print(f"{'t (s)':>8}  {'event MVL':>10}  {'control':>10}  {'Δ':>10}  {'p':>10}")
    for wi, wc in enumerate(WIN_CENTERS):
        e = pac_ev[:, wi]; c = pac_ct[:, wi]; d = e - c
        d_clean = d[np.isfinite(d)]
        _, p = wilcoxon(d_clean) if len(d_clean) >= 5 and np.any(d_clean != 0) else (np.nan, np.nan)
        print(f"{wc:>+8.1f}  {np.nanmean(e):>10.4f}  {np.nanmean(c):>10.4f}  "
              f"{np.nanmean(e-c):>+10.4f}  {p:>10.3g}")

    hsi_ev = np.array([r['hsi_ev'] for r in results])
    hsi_ct = np.array([r['hsi_ct'] for r in results])
    hsi_ev_grand, hsi_ev_lo, hsi_ev_hi = bootstrap_ci(hsi_ev)
    hsi_ct_grand, hsi_ct_lo, hsi_ct_hi = bootstrap_ci(hsi_ct)

    print(f"\n=== {args.cohort} composite · HSI (log10 SR1/SR3) time course ===")
    print(f"(envelope B38: HSI rises +0.20 at t=+2s — SR1 gains ~2× dominance over SR3)")
    print(f"{'t (s)':>6}  {'event':>10}  {'control':>10}  {'Δ':>10}")
    for t in [-6, -2, 0, 2, 6, 12]:
        idx = int(np.argmin(np.abs(HSI_TGRID - t)))
        e = hsi_ev_grand[idx]; c = hsi_ct_grand[idx]
        print(f"{t:>+6}  {e:>+10.4f}  {c:>+10.4f}  {e - c:>+10.4f}")

    # Peri-event RISE metric (argmax of ev − ct in [0, +4] s)
    peri_mask = (HSI_TGRID >= 0) & (HSI_TGRID <= 4)
    delta = hsi_ev_grand - hsi_ct_grand
    rise_idx = int(np.nanargmax(delta[peri_mask]))
    rise_t = float(HSI_TGRID[peri_mask][rise_idx])
    rise_delta = float(delta[peri_mask][rise_idx])
    print(f"\n  Peri-event HSI rise: max Δ(ev − ct) = {rise_delta:+.4f} at t = {rise_t:+.1f} s")
    print(f"  (envelope B38: +0.20 at t = +2 s)")

    # Save
    out_rows = []
    for r in results:
        for wi, wc in enumerate(WIN_CENTERS):
            out_rows.append({'subject_id': r['subject_id'], 't_s': wc,
                              'pac_ev': r['pac_ev'][wi], 'pac_ct': r['pac_ct'][wi]})
    pd.DataFrame(out_rows).to_csv(os.path.join(out_dir, 'frontal_pac_mvl.csv'), index=False)
    np.savez(os.path.join(out_dir, 'frontal_hsi.npz'),
             t=HSI_TGRID, ev_grand=hsi_ev_grand, ct_grand=hsi_ct_grand,
             ev_lo=hsi_ev_lo, ev_hi=hsi_ev_hi, ct_lo=hsi_ct_lo, ct_hi=hsi_ct_hi)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    pac_ev_mean = np.nanmean(pac_ev, axis=0)
    pac_ct_mean = np.nanmean(pac_ct, axis=0)
    ax.plot(WIN_CENTERS, pac_ev_mean, 'o-', color='#d73027', lw=2, label='event')
    ax.plot(WIN_CENTERS, pac_ct_mean, 's-', color='#4575b4', lw=2, label='control')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('time rel t0_net (s)')
    ax.set_ylabel('Canolty MVL (frontal θ-γ PAC)')
    ax.set_title(f'{args.cohort} composite · frontal PAC time course · n={len(results)} Q4')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(HSI_TGRID, hsi_ev_grand, color='#d73027', lw=2, label='event (Q4)')
    ax.fill_between(HSI_TGRID, hsi_ev_lo, hsi_ev_hi, color='#d73027', alpha=0.22)
    ax.plot(HSI_TGRID, hsi_ct_grand, color='#4575b4', lw=2, label='control')
    ax.fill_between(HSI_TGRID, hsi_ct_lo, hsi_ct_hi, color='#4575b4', alpha=0.22)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('time rel t0_net (s)')
    ax.set_ylabel('Δ HSI = log10(P_SR1 / P_SR3) − median')
    ax.set_title(f'HSI peri-event time course · peak Δ {rise_delta:+.3f} at t={rise_t:+.1f}s')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle(f'B38 · frontal PAC + HSI · {args.cohort} composite v2', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'frontal_pac_hsi.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/frontal_pac_hsi.png")


if __name__ == '__main__':
    main()
