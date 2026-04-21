#!/usr/bin/env python3
"""
B25 — Does the φ-lattice anchor to subject (IAF), to cohort, or is it fixed?
Does the lattice shift during ignition events?

Three candidate anchorings to distinguish:

  A. Fixed lattice  — ignition peak is at 7.83 Hz regardless of subject IAF.
                      Ratio event_peak / IAF varies across subjects.
  B. IAF-anchored   — ignition peak = IAF × constant (e.g., 7.83 / 9.69 ≈ 0.81).
                      Ratio event_peak / IAF is stable, peak itself varies.
  C. Event-anchored — the lattice shifts during ignition vs baseline (e.g.,
                      alpha peak re-centers toward 7.83 during events).

For each LEMON subject ≥3 events:
  1. Subject IAF from aggregate 1/f-corrected spectrum (alpha peak in [7, 13]).
  2. Subject-mean event peak (B22-style, averaged across events).
  3. Baseline aggregate spectrum & event-time aggregate spectrum (in t0_net ±2 s
     windows, averaged across events). Compare alpha-peak and θ-α-boundary
     positions.

Metrics:
  - Ratio event_peak / IAF, distribution across subjects (A vs B).
  - SD of event_peak / IAF across subjects. Small SD → B; large SD → A.
  - Shift in alpha peak freq (baseline → event), shift in θ-α boundary nadir,
    averaged across subjects. Non-zero → C.
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
                        'images', 'psd_timelapse')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

# Aggregate PSD — full recording
AGG_WIN_SEC = 8.0
AGG_HOP_SEC = 2.0
AGG_NFFT_MULT = 4
AGG_LO, AGG_HI = 2.0, 20.0
APERIODIC_RANGES = [(2.0, 5.0), (9.0, 20.0)]

# Event-window PSD — 4 s centered on t0_net + 1 s
EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
EV_LO, EV_HI = 6.5, 9.0

# Baseline window: 4-s sliding windows ≥ 30 s from any event
BASELINE_GAP_S = 30.0

IAF_LO, IAF_HI = 7.0, 13.0
THETA_ALPHA_BOUNDARY_LO, THETA_ALPHA_BOUNDARY_HI = 6.5, 9.0  # search for nadir
SCHUMANN_F = 7.83
POP_THETA_ALPHA_BOUND = 7.60


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
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


def parabolic_nadir(y, x):
    """Argmin with parabolic refinement."""
    k = int(np.argmin(y))
    if 1 <= k < len(y) - 1 and y[k-1] > 0 and y[k+1] > 0:
        y0, y1, y2 = y[k-1], y[k], y[k+1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = max(-1.0, min(1.0, delta))
        return float(x[k] + delta * (x[1] - x[0]))
    return float(x[k])


def aperiodic_resid(freqs, psd):
    mask = np.zeros_like(freqs, dtype=bool)
    for lo, hi in APERIODIC_RANGES:
        mask |= (freqs >= lo) & (freqs <= hi)
    logf = np.log10(freqs)
    logp = np.log10(psd + 1e-20)
    good = mask & np.isfinite(logp) & (logp > -10)
    if good.sum() < 8:
        return psd.copy()
    A = np.column_stack([logf[good], np.ones(good.sum())])
    coefs, *_ = np.linalg.lstsq(A, logp[good], rcond=None)
    a, b = float(coefs[0]), float(coefs[1])
    return psd - 10 ** (a * logf + b)


def find_iaf(freqs, resid):
    m = (freqs >= IAF_LO) & (freqs <= IAF_HI)
    if not np.any(resid[m] > 0):
        return np.nan
    idx = np.where(m)[0]
    k = idx[int(np.argmax(resid[idx]))]
    return parabolic_peak(resid[idx], freqs[idx])


def find_theta_alpha_nadir(freqs, resid):
    """Nadir in 1/f-corrected residual between [6.5, 9] (the θ-α boundary dip)."""
    m = (freqs >= THETA_ALPHA_BOUNDARY_LO) & (freqs <= THETA_ALPHA_BOUNDARY_HI)
    idx = np.where(m)[0]
    return parabolic_nadir(resid[idx], freqs[idx])


def subject_aggregate(y, fs):
    nperseg = int(round(AGG_WIN_SEC * fs))
    nhop = int(round(AGG_HOP_SEC * fs))
    nfft = nperseg * AGG_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= AGG_LO) & (freqs <= AGG_HI)
    psds = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psds.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if len(psds) < 5:
        return None, None
    return freqs[mask], np.nanmedian(np.array(psds), axis=0)


def event_peak_from_avg(y, fs, t_events, lag=EV_LAG_S):
    """Return (freqs, event_avg_psd, baseline_avg_psd, peak_f)."""
    nperseg = int(round(EV_WIN_SEC * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    ev_mask = (freqs >= EV_LO) & (freqs <= EV_HI)
    nhop = int(round(1.0 * fs))

    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        base_rows.append(welch_one(y[i:i + nperseg], fs, nfft)[ev_mask])
    if len(base_rows) < 10:
        return None, None, None, np.nan
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    ev_rows = []
    for t0 in t_events:
        tc = t0 + lag
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        ev_rows.append(welch_one(y[i0:i1], fs, nfft)[ev_mask])
    if not ev_rows:
        return None, None, None, np.nan
    ev_avg = np.nanmean(np.array(ev_rows), axis=0)
    ratio = (ev_avg + 1e-20) / (baseline + 1e-20)
    return freqs[ev_mask], ev_avg, baseline, parabolic_peak(ratio, freqs[ev_mask])


def aggregate_event_vs_baseline(y, fs, t_events):
    """Full-spectrum aggregate PSDs: all windows in event segments vs all
    windows far from events."""
    nperseg = int(round(AGG_WIN_SEC * fs))
    nhop = int(round(AGG_HOP_SEC * fs))
    nfft = nperseg * AGG_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= AGG_LO) & (freqs <= AGG_HI)
    t_events = np.array(t_events)
    ev_rows, base_rows = [], []
    for i in range(0, len(y) - nperseg + 1, nhop):
        wc = (i + nperseg / 2) / fs
        # if any event within ±2 s of window center → event window
        if len(t_events) and np.min(np.abs(wc - t_events)) <= 2.0:
            ev_rows.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
        elif len(t_events) == 0 or np.min(np.abs(wc - t_events)) >= BASELINE_GAP_S:
            base_rows.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if len(ev_rows) < 2 or len(base_rows) < 5:
        return None, None, None
    ev_mean = np.nanmean(np.array(ev_rows), axis=0)
    base_mean = np.nanmean(np.array(base_rows), axis=0)
    return freqs[mask], ev_mean, base_mean


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 3:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    # Full aggregate → IAF
    freqs_a, agg = subject_aggregate(y, fs)
    if agg is None:
        return None
    resid_all = aperiodic_resid(freqs_a, agg)
    iaf = find_iaf(freqs_a, resid_all)
    if not np.isfinite(iaf):
        return None

    # Event-window fine-resolution peak
    _, _, _, peak_f = event_peak_from_avg(y, fs, events['t0_net'].astype(float).values)
    if not np.isfinite(peak_f):
        return None

    # Aggregate: event windows vs baseline windows; re-compute IAF and θ-α nadir in each
    freqs_b, agg_ev, agg_base = aggregate_event_vs_baseline(
        y, fs, events['t0_net'].astype(float).values)
    iaf_ev = iaf_base = nadir_ev = nadir_base = np.nan
    if agg_ev is not None:
        r_ev = aperiodic_resid(freqs_b, agg_ev)
        r_base = aperiodic_resid(freqs_b, agg_base)
        iaf_ev = find_iaf(freqs_b, r_ev)
        iaf_base = find_iaf(freqs_b, r_base)
        nadir_ev = find_theta_alpha_nadir(freqs_b, r_ev)
        nadir_base = find_theta_alpha_nadir(freqs_b, r_base)

    return {
        'subject_id': sub_id,
        'n_events': int(len(events)),
        'iaf_hz': iaf,
        'ignition_peak_hz': peak_f,
        'event_peak_over_iaf': peak_f / iaf if iaf > 0 else np.nan,
        'event_peak_minus_iaf': peak_f - iaf,
        # event-vs-baseline aggregate
        'iaf_event_window': iaf_ev,
        'iaf_baseline_window': iaf_base,
        'iaf_shift_event_minus_baseline': iaf_ev - iaf_base,
        'theta_alpha_nadir_event': nadir_ev,
        'theta_alpha_nadir_baseline': nadir_base,
        'theta_alpha_nadir_shift': nadir_ev - nadir_base,
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects ≥3 events: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        out = pool.map(process_subject, tasks)
    df = pd.DataFrame([r for r in out if r is not None])
    df.to_csv(os.path.join(OUT_DIR, 'lattice_anchoring.csv'), index=False)
    print(f"Successful: {len(df)}")

    # Hypothesis A (fixed lattice): event_peak is constant; ratio varies
    # Hypothesis B (IAF-anchored): ratio is constant; peak varies
    peak_mean = df['ignition_peak_hz'].mean()
    peak_std = df['ignition_peak_hz'].std()
    ratio_mean = df['event_peak_over_iaf'].mean()
    ratio_std = df['event_peak_over_iaf'].std()
    peak_cv = peak_std / peak_mean
    ratio_cv = ratio_std / ratio_mean
    print(f"\n=== A vs B test ===")
    print(f"  Ignition peak (Hz):  mean {peak_mean:.3f}   SD {peak_std:.3f}   CV {peak_cv*100:.2f}%")
    print(f"  event_peak / IAF:    mean {ratio_mean:.3f}   SD {ratio_std:.3f}   CV {ratio_cv*100:.2f}%")
    print(f"  Interpretation:")
    if peak_cv < ratio_cv * 0.8:
        print(f"    → Peak is MORE stable than ratio → FIXED lattice (A)")
    elif ratio_cv < peak_cv * 0.8:
        print(f"    → Ratio is MORE stable than peak → IAF-ANCHORED lattice (B)")
    else:
        print(f"    → Peak and ratio variability comparable — ambiguous")

    # Hypothesis C (event-anchored): θ-α nadir or IAF shifts during event windows
    print(f"\n=== C test — does the aggregate spectrum reshape during events? ===")
    shifts_iaf = df['iaf_shift_event_minus_baseline'].dropna()
    shifts_nadir = df['theta_alpha_nadir_shift'].dropna()
    print(f"  IAF shift (event − baseline):    median {shifts_iaf.median():+.3f}   "
          f"mean {shifts_iaf.mean():+.3f}   SD {shifts_iaf.std():.3f}   n={len(shifts_iaf)}")
    print(f"  θ-α nadir shift (event − base):  median {shifts_nadir.median():+.3f}   "
          f"mean {shifts_nadir.mean():+.3f}   SD {shifts_nadir.std():.3f}   n={len(shifts_nadir)}")
    # Wilcoxon test vs 0
    from scipy.stats import wilcoxon
    if len(shifts_iaf) > 5:
        s, p = wilcoxon(shifts_iaf.values)
        print(f"  Wilcoxon IAF shift vs 0:    stat={s:.1f}  p={p:.3g}")
    if len(shifts_nadir) > 5:
        s, p = wilcoxon(shifts_nadir.values)
        print(f"  Wilcoxon nadir shift vs 0:  stat={s:.1f}  p={p:.3g}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: peak vs IAF with three-hypothesis lines
    ax = axes[0, 0]
    ax.scatter(df['iaf_hz'], df['ignition_peak_hz'], s=30, alpha=0.6,
                color='steelblue', edgecolor='k', lw=0.3)
    rng = np.array([df['iaf_hz'].min() - 0.3, df['iaf_hz'].max() + 0.3])
    # A: horizontal at mean
    ax.plot(rng, [peak_mean, peak_mean], 'g-', lw=1.5,
             label=f'A: fixed {peak_mean:.2f} Hz')
    # B: slope-through-origin with cohort ratio
    ax.plot(rng, ratio_mean * rng, 'r-', lw=1.5,
             label=f'B: IAF × {ratio_mean:.2f}')
    # identity
    ax.plot(rng, rng, 'k--', lw=0.8, alpha=0.5, label='identity')
    ax.set_xlabel('IAF (Hz)'); ax.set_ylabel('ignition peak (Hz)')
    ax.set_title('A (fixed) vs B (IAF-anchored)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: ratio distribution
    ax = axes[0, 1]
    ax.hist(df['event_peak_over_iaf'].dropna(), bins=25,
             color='firebrick', edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(ratio_mean, color='blue', ls='--', lw=1.5,
                label=f'cohort ratio {ratio_mean:.3f}')
    ax.axvline(SCHUMANN_F / 9.69, color='green', ls=':', lw=1,
                label='7.83/9.69 (fixed pred.)')
    ax.set_xlabel('ignition peak / IAF')
    ax.set_ylabel('subjects')
    ax.set_title(f'ratio CV={ratio_cv*100:.1f}%  vs peak CV={peak_cv*100:.1f}%')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 3: IAF shift event-baseline
    ax = axes[1, 0]
    ax.hist(shifts_iaf, bins=25, color='darkorange', edgecolor='k',
             lw=0.3, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(shifts_iaf.mean(), color='blue', ls='--', lw=1.5,
                label=f'mean {shifts_iaf.mean():+.3f} Hz')
    ax.set_xlabel('IAF (event) − IAF (baseline)  Hz')
    ax.set_ylabel('subjects')
    ax.set_title('Does IAF move during events?')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 4: θ-α nadir shift
    ax = axes[1, 1]
    ax.hist(shifts_nadir, bins=25, color='purple', edgecolor='k',
             lw=0.3, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(shifts_nadir.mean(), color='blue', ls='--', lw=1.5,
                label=f'mean {shifts_nadir.mean():+.3f} Hz')
    ax.axvline(POP_THETA_ALPHA_BOUND - shifts_nadir.median(), color='gray', ls=':',
                lw=0.6)
    ax.set_xlabel('θ-α nadir (event) − nadir (baseline)  Hz')
    ax.set_ylabel('subjects')
    ax.set_title('Does θ-α lattice boundary move during events?')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle('B25 — Lattice anchoring: subject-IAF vs fixed vs event-shifted',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'lattice_anchoring.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/lattice_anchoring.png")


if __name__ == '__main__':
    main()
