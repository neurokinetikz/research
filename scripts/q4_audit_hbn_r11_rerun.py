#!/usr/bin/env python3
"""HBN R11 Q4 sensitivity check: re-run beta_peak_iaf_coupling on Q4 events
only, then compare cross-subject ρ(SR_peak_ratio, β_peak_ratio) to the
published all-events ρ=0.825.

Per-subject corpus event yield is low (median 4 events, Q4 median 1 event),
so per-subject Q4 metrics are computed from 1-2 events rather than 4-5.
This is a sensitivity check, not a replacement: if the Q4 ρ is materially
similar (within ~0.1) the claim is robust to scoping choice; if it falls
to null, the published claim depends on multi-event averaging.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import warnings
from glob import glob
from multiprocessing import Pool
from scipy import signal
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_hbn_by_subject, load_lemon
import mne
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

ROOT = os.path.join(os.path.dirname(__file__), '..')
COHORT = os.environ.get('COHORT', 'hbn_R1')  # 'hbn_R1', 'lemon', etc.
EVENTS_DIR = os.path.join(ROOT, 'exports_sie', f'{COHORT}_composite')
QUALITY_CSV = os.path.join(ROOT, 'outputs/schumann/images/quality/'
                                  f'per_event_quality_{COHORT}_composite.csv')
OUT_PATH = os.path.join(ROOT, 'outputs/schumann/images/psd_timelapse/'
                              f'{COHORT}_composite/beta_peak_iaf_coupling_q4.csv')

if COHORT.startswith('hbn_'):
    RELEASE = COHORT.replace('hbn_', '')
    LOADER = lambda sid: load_hbn_by_subject(sid, release=RELEASE)
elif COHORT == 'lemon':
    LOADER = lambda sid: load_lemon(sid, condition='EC')
else:
    raise ValueError(f'Unknown cohort: {COHORT}')

# IAF/SR/β bands matching sie_beta_peak_iaf_coupling_composite.py
SR_LO, SR_HI = 7.0, 8.3
BETA_LO, BETA_HI = 16.0, 24.0
IAF_LO, IAF_HI = 7.0, 13.0
EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
PSD_NPER_SEC = 4.0


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def compute_iaf(y, fs):
    nperseg = int(round(PSD_NPER_SEC * fs))
    f, psd = signal.welch(y, fs=fs, nperseg=nperseg)
    m = (f >= IAF_LO) & (f <= IAF_HI)
    if not m.any():
        return np.nan
    return float(f[m][np.argmax(psd[m])])


def event_peak_in_band(y, fs, t_events, lo, hi):
    """Event-locked aggregate PSD, return peak freq and ratio over baseline.
    Baseline = median of all sliding windows.
    """
    nperseg = int(round(EV_WIN_SEC * fs))
    nfft = nperseg * EV_NFFT_MULT
    nhop = int(round(1.0 * fs))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band_mask = (freqs >= lo) & (freqs <= hi)
    if not band_mask.any():
        return np.nan, np.nan
    f_band = freqs[band_mask]

    # Baseline
    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i+nperseg], fs, nfft)[band_mask]
        base_rows.append(psd)
    if len(base_rows) < 5:
        return np.nan, np.nan
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    # Event-locked
    ev_rows = []
    for t0 in t_events:
        tc = t0 + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[band_mask]
        ev_rows.append(psd)
    if not ev_rows:
        return np.nan, np.nan
    ev = np.nanmean(np.array(ev_rows), axis=0)
    ratio = ev / baseline
    pi = int(np.argmax(ratio))
    return float(f_band[pi]), float(ratio[pi])


def process_subject(args):
    sub_id, q4_t0s = args
    if len(q4_t0s) < 1:
        return None
    try:
        raw = LOADER(sub_id)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    iaf = compute_iaf(y, fs)
    sr_f, sr_r = event_peak_in_band(y, fs, q4_t0s, SR_LO, SR_HI)
    beta_f, beta_r = event_peak_in_band(y, fs, q4_t0s, BETA_LO, BETA_HI)
    return {
        'subject_id': sub_id,
        'iaf_hz': iaf,
        'sr_peak_hz': sr_f,
        'sr_peak_ratio': sr_r,
        'beta_peak_hz': beta_f,
        'beta_peak_ratio': beta_r,
        'n_q4_events': int(len(q4_t0s)),
    }


def main():
    qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = qual.groupby('subject_id')['template_rho'].transform(
        lambda x: pd.qcut(x, 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
        if x.nunique() >= 4 else None
    )
    q4 = qual[qual['rho_q'] == 'Q4']
    print(f'Quality CSV: {len(qual)} events, {qual["subject_id"].nunique()} subjects')
    print(f'Q4 events: {len(q4)} across {q4["subject_id"].nunique()} subjects')

    tasks = []
    for sid, g in q4.groupby('subject_id'):
        tasks.append((sid, g['t0_net'].values))
    print(f'Tasks: {len(tasks)} subjects')

    n_workers = int(os.environ.get('SIE_WORKERS', 4))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    df = pd.DataFrame([r for r in results if r is not None])
    print(f'Successful: {len(df)}')
    df.to_csv(OUT_PATH, index=False)
    print(f'Wrote {OUT_PATH}')

    good = df.dropna(subset=['sr_peak_ratio', 'beta_peak_ratio']).copy()
    print(f'Complete rows: {len(good)}')
    if len(good) < 10:
        print('Not enough subjects for correlation.')
        return

    rho, p = spearmanr(good['sr_peak_ratio'], good['beta_peak_ratio'])
    rp, pp = pearsonr(good['sr_peak_ratio'], good['beta_peak_ratio'])
    print()
    print('=' * 72)
    print('HBN R11 Q4 SENSITIVITY CHECK')
    print('=' * 72)
    print(f'  Q4 median events/subject: {good["n_q4_events"].median():.0f}')
    print(f'  Q4 ρ(SR_peak_ratio, β_peak_ratio) = {rho:+.3f}, p = {p:.3g}, n = {len(good)}')
    print(f'  Q4 r(SR_peak_ratio, β_peak_ratio) = {rp:+.3f}, p = {pp:.3g}')
    # Look up published all-events ρ for this release
    published_csv = OUT_PATH.replace('_q4', '')
    pub_rho = None
    try:
        pub = pd.read_csv(published_csv).dropna(subset=['sr_peak_ratio', 'beta_peak_ratio'])
        pub_rho, _ = spearmanr(pub['sr_peak_ratio'], pub['beta_peak_ratio'])
    except Exception:
        pass
    print()
    print(f'  Compare to published all-events for HBN {RELEASE}:')
    if pub_rho is not None:
        print(f'    ρ = {pub_rho:+.3f}, n = {len(pub)}')
    print()
    if pub_rho is not None and abs(rho - pub_rho) < 0.10:
        print('  --> Q4 ρ is within 0.10 of all-events ρ. Published claim is robust.')
    elif pub_rho is not None and abs(rho - pub_rho) < 0.25:
        print('  --> Q4 ρ is within 0.25 of all-events ρ. Modest attenuation but ')
        print('      same qualitative direction.')
    else:
        print('  --> Q4 ρ differs from all-events ρ by more than 0.25. The ')
        print('      published claim is sensitive to scoping choice.')


if __name__ == '__main__':
    main()
