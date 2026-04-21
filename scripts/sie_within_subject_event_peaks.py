#!/usr/bin/env python3
"""
B22 — Within-subject event-to-event variability of the ignition peak.

The Dortmund test-retest (B21) showed ICC ≈ 0 but only ~1 event per session,
making the low ICC consistent with pure measurement noise rather than a real
state-vs-trait dissociation. Here we compute per-event peak frequencies in
LEMON (longer recordings, ~5 events/subject) and ask:

  (a) How much do event peak frequencies vary within a single subject's
      session? (Gives expected single-event measurement noise.)
  (b) Is there a subject-specific mean that events cluster around? (Bootstrap:
      per-subject mean peak vs per-event peak distributions.)
  (c) If within-subject SD(event peak) ≈ 1.12 Hz (the Dortmund Δ SD), then
      Dortmund's ICC ≈ 0 is fully explained by measurement noise.
      If within-subject SD << 1.12 Hz, then Dortmund's retest variability
      contains additional state variance.

Uses all LEMON events with ≥3 events per subject. Computes per-event peak
frequency in [6.5, 9.0] Hz from a single 4-s window at t0_net + 1.0 s, divided
by the subject's baseline PSD.
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

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
ZOOM_LO, ZOOM_HI = 6.5, 9.0


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

    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= ZOOM_LO) & (freqs <= ZOOM_HI)
    f_band = freqs[mask]

    # Baseline PSD
    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)[mask]
        base_rows.append(psd)
    if len(base_rows) < 10:
        return None
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    # Per-event peak
    peaks = []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((t0 - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[mask]
        ratio = (psd + 1e-20) / (baseline + 1e-20)
        peaks.append(parabolic_peak(ratio, f_band))
    peaks = np.array(peaks)
    if len(peaks) < 3:
        return None
    return {
        'subject_id': sub_id,
        'n_events': int(len(peaks)),
        'peak_mean': float(np.mean(peaks)),
        'peak_median': float(np.median(peaks)),
        'peak_sd': float(np.std(peaks, ddof=1)),
        'peak_iqr': float(np.percentile(peaks, 75) - np.percentile(peaks, 25)),
        'peaks': peaks.tolist(),
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects with ≥3 events: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    # Summary of within-subject peak SDs
    sds = np.array([r['peak_sd'] for r in results])
    means = np.array([r['peak_mean'] for r in results])
    iqrs = np.array([r['peak_iqr'] for r in results])
    ns = np.array([r['n_events'] for r in results])

    print(f"\n=== Within-subject peak-frequency variability (LEMON, all events) ===")
    print(f"  n_subjects with ≥3 events: {len(results)}")
    print(f"  events per subject: median {np.median(ns):.0f}  range {ns.min()}-{ns.max()}")
    print(f"  Within-subject SD of event peaks: median {np.median(sds):.3f} Hz  "
          f"IQR [{np.percentile(sds,25):.3f}, {np.percentile(sds,75):.3f}]")
    print(f"  Within-subject IQR of event peaks: median {np.median(iqrs):.3f} Hz")
    print(f"  Between-subject SD of per-subject means: {np.std(means, ddof=1):.3f} Hz")
    print(f"  Per-subject mean distribution: mean {np.mean(means):.3f}   "
          f"std {np.std(means):.3f}")

    # Compare: pooled event-peak SD vs per-subject mean SD
    all_peaks = np.concatenate([r['peaks'] for r in results])
    print(f"\n  Pooled event peak frequency SD (all events): {np.std(all_peaks, ddof=1):.3f} Hz")
    print(f"  Pooled event peak frequency mean: {np.mean(all_peaks):.3f} Hz")

    # Within/between variance ratio → true trait ICC estimate
    within_var = np.mean(sds ** 2)
    between_var = np.var(means, ddof=1)
    icc_est = between_var / (between_var + within_var) if (between_var + within_var) > 0 else np.nan
    print(f"\n  Trait-ICC estimate = between_var / (between + within) = {icc_est:.3f}")
    print(f"  Reference: Dortmund retest ICC observed was −0.10 with ~1 event/session")
    print(f"  Within-subject SD ~ {np.median(sds):.2f} Hz; Dortmund session SD was 1.12 Hz.")
    if np.median(sds) < 1.0:
        print(f"  → LEMON within-subject SD ({np.median(sds):.2f}) << Dortmund retest SD (1.12)")
        print(f"    Suggests LEMON events DO cluster within subjects; Dortmund's high retest SD")
        print(f"    is primarily single-event measurement noise, not absence of a trait.")
    else:
        print(f"  → LEMON within-subject SD ~ Dortmund retest SD; ignition peak looks genuinely")
        print(f"    event-stochastic with weak or no subject trait.")

    # Save per-subject CSV
    rows = [{k: v for k, v in r.items() if k != 'peaks'} for r in results]
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, 'within_subject_event_peaks.csv'), index=False)

    # Plot: distribution of within-subject SDs + scatter of per-subject mean ± SD
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(sds, bins=25, color='firebrick', edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(np.median(sds), color='blue', ls='--', lw=1.5,
                label=f'median {np.median(sds):.2f} Hz')
    ax.axvline(1.12, color='gray', ls=':', lw=1.5,
                label='Dortmund retest Δ std (1.12)')
    ax.set_xlabel('within-subject SD of event peak freq (Hz)')
    ax.set_ylabel('subjects')
    ax.set_title(f'Within-subject event-peak SD (n={len(sds)})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    # Sort by mean
    order = np.argsort(means)
    for j, i in enumerate(order):
        peaks = results[i]['peaks']
        ax.scatter([j] * len(peaks), peaks, s=8, alpha=0.4,
                    color='steelblue')
        ax.errorbar(j, means[i], yerr=sds[i], fmt='o', color='firebrick',
                     capsize=0, markersize=3, elinewidth=0.5)
    ax.axhline(7.83, color='green', ls='--', lw=1, label='Schumann 7.83 Hz')
    ax.set_xlabel('subject (sorted by mean peak)')
    ax.set_ylabel('event peak freq (Hz)')
    ax.set_title(f'Per-subject event peaks · sorted by mean')
    ax.legend(fontsize=9)
    ax.set_ylim(ZOOM_LO, ZOOM_HI)
    ax.grid(alpha=0.3)

    plt.suptitle('B22 — Within-subject event-to-event peak variability',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'within_subject_event_peaks.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/within_subject_event_peaks.png")


if __name__ == '__main__':
    main()
