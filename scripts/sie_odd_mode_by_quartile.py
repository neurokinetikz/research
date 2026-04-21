#!/usr/bin/env python3
"""
B40 — Odd-mode Schumann excitation pattern by template_rho quartile.

B33 found SR1 and SR3 elevated; SR2, SR4 at floor — the vertical-dipole
odd-only signature. But B33 used all LEMON events pooled. Does quality
stratification sharpen this pattern?

Computes event-locked aggregate spectrum separately for Q1, Q2, Q3, Q4
template_rho events in LEMON, then measures elevation at actual Schumann
landmarks (from sos70.ru-style maps: SR1=7.82, SR2=13.97, SR3=19.95,
SR4=25.44).

Tests:
  - Does Q4 show larger odd-mode elevation (SR1, SR3) than Q1?
  - Does Q4 show FLATTER even-mode response (SR2, SR4 still at floor)?
  - I.e., does morphological filtering sharpen the cavity-mode pattern?
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
FREQ_LO, FREQ_HI = 2.0, 30.0

SR_MODES = {
    'SR1': 7.82,
    'SR2': 13.97,
    'SR3': 19.95,
    'SR4': 25.44,
}
SR_HALF_WIDTH = 0.35


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    wp = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
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
        q_sub = qual[qual['subject_id'] == sub_id][['t0_net', 'rho_q']].copy()
        q_sub['t0_round'] = q_sub['t0_net'].round(3)
        events['t0_round'] = events['t0_net'].round(3)
        events = events.merge(q_sub[['t0_round', 'rho_q']], on='t0_round', how='left')
        events = events.dropna(subset=['rho_q'])
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
    if fs < 70:
        return None
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    f_band = freqs[mask]

    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        base_rows.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if len(base_rows) < 10:
        return None
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    ratios_by_q = {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    for _, ev in events.iterrows():
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[mask]
        r = (psd + 1e-20) / (baseline + 1e-20)
        ratios_by_q[ev['rho_q']].append(r)

    out = {'subject_id': sub_id, 'freqs': f_band}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        if ratios_by_q[q]:
            out[f'{q}_mean'] = np.nanmean(np.array(ratios_by_q[q]), axis=0)
        else:
            out[f'{q}_mean'] = None
    return out


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

    # Interpolate onto common grid
    common = np.arange(FREQ_LO, FREQ_HI + 0.005, 0.05)
    per_q = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        stack = []
        for r in results:
            if r[f'{q}_mean'] is not None:
                interp = np.interp(common, r['freqs'],
                                    np.log10(r[f'{q}_mean'] + 1e-20))
                stack.append(interp)
        per_q[q] = np.array(stack)

    print(f"\n=== Log10 event/baseline ratio at SR landmarks by quartile ===")
    print(f"{'q':<4}{'n_sub':>8}   {'SR1':>10}{'SR2':>10}{'SR3':>10}{'SR4':>10}   odd-only pattern?")
    # Use β-floor excluding SR3 region as reference
    floor_mask = (common >= 12) & (common <= 25) & ~(
        (common >= 19.5) & (common <= 20.5))
    odd_excess = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mat = per_q[q]
        if len(mat) < 5:
            continue
        grand = np.nanmean(mat, axis=0)  # log10 ratio
        floor = np.nanmedian(grand[floor_mask])
        vals = {}
        for name, f in SR_MODES.items():
            m = (common >= f - SR_HALF_WIDTH) & (common <= f + SR_HALF_WIDTH)
            v = np.nanmax(grand[m])
            vals[name] = v - floor  # log excess above floor
        odd = (vals['SR1'] + vals['SR3']) / 2
        even = (vals['SR2'] + vals['SR4']) / 2
        pattern = 'YES' if odd > 3 * max(0, even) else ('partial' if odd > even else 'no')
        odd_excess[q] = {**vals, 'odd_mean': odd, 'even_mean': even, 'pattern': pattern}
        print(f"{q:<4}{len(mat):>8}   "
              f"{1e2*vals['SR1']:>+8.1f}%  {1e2*vals['SR2']:>+8.1f}%  "
              f"{1e2*vals['SR3']:>+8.1f}%  {1e2*vals['SR4']:>+8.1f}%   {pattern}")

    # Odd vs even ratio by quartile
    print(f"\n=== Odd / Even mean excess ratio by quartile ===")
    print(f"{'q':<4}{'odd mean':>12}{'even mean':>12}{'ratio odd/even':>20}")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        if q not in odd_excess:
            continue
        o = odd_excess[q]['odd_mean']
        e = odd_excess[q]['even_mean']
        r = o / e if abs(e) > 1e-6 else np.inf
        print(f"{q:<4}{1e2*o:>+11.2f}%{1e2*e:>+11.2f}%{r:>20.2f}")

    # Save CSV
    rows = []
    for q, vals in odd_excess.items():
        rows.append({'quartile': q, **{f'{k}_log_excess': v for k, v in vals.items()}})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'odd_mode_by_quartile.csv'),
                               index=False)

    # Plot: 4 quartile spectra overlaid
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = {'Q1': '#4575b4', 'Q2': '#91bfdb', 'Q3': '#fc8d59', 'Q4': '#d73027'}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mat = per_q[q]
        if len(mat) < 5: continue
        grand = np.nanmean(mat, axis=0)
        ratio = 10 ** grand
        ax.plot(common, ratio, color=colors[q], lw=1.6, label=f'{q} n_sub={len(mat)}')
    for name, f in SR_MODES.items():
        c = '#1a9641' if name in ('SR1', 'SR3') else '#d7301f'
        ls = '--' if name in ('SR1', 'SR3') else ':'
        ax.axvline(f, color=c, ls=ls, lw=0.8, alpha=0.7)
        ax.text(f, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 6, name,
                ha='center', fontsize=8, color=c)
    ax.axhline(1.0, color='k', lw=0.6)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('event / baseline PSD (×)')
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_title(f'B40 — Event-locked spectrum by template_rho quartile (LEMON)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'odd_mode_by_quartile.png'),
                 dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/odd_mode_by_quartile.png")


if __name__ == '__main__':
    main()
