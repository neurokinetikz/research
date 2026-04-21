#!/usr/bin/env python3
"""
B11 (prototype) — Time-frequency timelapse of the 4-9 Hz theta/low-alpha band.

For each of 5 LEMON EC subjects:
  1. Sliding Welch PSD (4-s window, 75% overlap) in the 4-9 Hz band
  2. Instantaneous-peak tracker (argmax in 6-9 Hz per window)
  3. Overlay SIE event t0_net times
  4. Compare power at the tracked peak vs power in the fixed 7.83 ± 0.6 Hz band

Goal: see whether the long-window PSD plateau at 7-8 Hz is a time-average of
intermittent crisp narrow peaks (drifting in frequency), and whether events
cluster on "crisp-peak" epochs or peak-frequency jumps.

Prototype: 5 subjects, visual diagnostics only. No cross-subject stats.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
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

N_SUBJECTS = 5
WIN_SEC = 4.0
HOP_SEC = 1.0            # 75% overlap
FREQ_LO, FREQ_HI = 4.0, 9.0
PEAK_LO, PEAK_HI = 6.0, 9.0
SR_LO, SR_HI = 7.0, 8.2    # SR fundamental natural range
NFFT_MULT = 4            # zero-pad for finer frequency resolution
F0_FIXED = 7.83
HALF_BW = 0.6


def sliding_welch(x, fs, win_sec=WIN_SEC, hop_sec=HOP_SEC,
                   f_lo=FREQ_LO, f_hi=FREQ_HI):
    """Sliding Welch PSD. Return (t_centers, freqs, P) — P is (n_freq, n_time)."""
    nperseg = int(round(win_sec * fs))
    nhop = int(round(hop_sec * fs))
    nfft = nperseg * NFFT_MULT
    freqs_full = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_mask = (freqs_full >= f_lo) & (freqs_full <= f_hi)
    freqs = freqs_full[f_mask]

    win = signal.windows.hann(nperseg)
    win_pow = np.sum(win ** 2)

    t_centers, P_cols = [], []
    for i in range(0, len(x) - nperseg + 1, nhop):
        seg = x[i:i + nperseg]
        seg = seg - seg.mean()
        seg_w = seg * win
        X = np.fft.rfft(seg_w, nfft)
        psd = (np.abs(X) ** 2) / (fs * win_pow)
        psd[1:-1] *= 2.0
        P_cols.append(psd[f_mask])
        t_centers.append((i + nperseg / 2) / fs)
    return np.array(t_centers), freqs, np.array(P_cols).T  # (n_freq, n_time)


def track_peak(freqs, P, f_lo=PEAK_LO, f_hi=PEAK_HI):
    """Per-column peak frequency within [f_lo, f_hi], with quadratic refinement."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    idx_band = np.where(mask)[0]
    f_band = freqs[idx_band]
    peaks = np.full(P.shape[1], np.nan)
    crispness = np.full(P.shape[1], np.nan)   # peak / band mean
    peak_pow = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        col = P[idx_band, j]
        if not np.isfinite(col).any() or np.all(col == 0):
            continue
        k = int(np.argmax(col))
        # parabolic refinement
        if 1 <= k < len(col) - 1:
            y0, y1, y2 = col[k - 1], col[k], col[k + 1]
            denom = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            delta = max(-1.0, min(1.0, delta))
            f_k = f_band[k] + delta * (f_band[1] - f_band[0])
        else:
            f_k = f_band[k]
        peaks[j] = f_k
        peak_pow[j] = col[k]
        crispness[j] = col[k] / (np.nanmean(col) + 1e-12)
    return peaks, peak_pow, crispness


def fixed_band_power(freqs, P, f0=F0_FIXED, half_bw=HALF_BW):
    """Mean power in [f0 - hb, f0 + hb] per column."""
    mask = (freqs >= f0 - half_bw) & (freqs <= f0 + half_bw)
    return np.nanmean(P[mask, :], axis=0)


def render_subject(sub_id, events_path):
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception as e:
        print(f"  skip {sub_id}: {e}")
        return
    if raw is None:
        return
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
        t_events = events['t0_net'].values.astype(float)
    except Exception:
        t_events = np.array([])

    t, freqs, P = sliding_welch(y, fs)
    peak_f, peak_p, crisp = track_peak(freqs, P)
    band_p = fixed_band_power(freqs, P)

    # SR-band restricted tracker
    peak_f_sr, peak_p_sr, crisp_sr = track_peak(freqs, P, f_lo=SR_LO, f_hi=SR_HI)
    # Fraction of time the unrestricted peak falls inside the SR band
    in_sr_mask = (peak_f >= SR_LO) & (peak_f <= SR_HI)
    pct_in_sr = np.nanmean(in_sr_mask) * 100

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1.2, 1.2, 1.2]})

    # 1. Spectrogram (log10)
    ax = axes[0]
    P_log = np.log10(P + 1e-20)
    im = ax.imshow(P_log, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], freqs[0], freqs[-1]],
                   cmap='magma', interpolation='nearest')
    ax.plot(t, peak_f, color='cyan', lw=0.9, alpha=0.9, label='peak (6-9 Hz)')
    ax.axhline(F0_FIXED, color='white', ls='--', lw=0.5, alpha=0.7, label=f'{F0_FIXED} Hz')
    for te in t_events:
        ax.axvline(te, color='lime', alpha=0.6, lw=0.6)
    ax.set_ylabel('frequency (Hz)')
    ax.set_title(f'{sub_id} · log10 PSD timelapse (4-s Welch, 1-s hop) · green=event, cyan=tracked peak')
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax, label='log10 PSD', pad=0.01)

    # 2. Peak frequency over time
    ax = axes[1]
    ax.plot(t, peak_f, color='steelblue', lw=1)
    ax.axhline(F0_FIXED, color='red', ls='--', lw=0.6)
    for te in t_events:
        ax.axvline(te, color='lime', alpha=0.4, lw=0.6)
    ax.set_ylabel('peak freq (Hz)')
    ax.set_ylim(PEAK_LO, PEAK_HI)
    ax.grid(alpha=0.3)

    # 3. Peak power vs fixed-band power (our detector's band)
    ax = axes[2]
    ax.plot(t, peak_p, color='firebrick', lw=1, label='peak power')
    ax.plot(t, band_p, color='darkorange', lw=1, alpha=0.7,
            label=f'fixed band {F0_FIXED}±{HALF_BW} Hz')
    for te in t_events:
        ax.axvline(te, color='lime', alpha=0.4, lw=0.6)
    ax.set_ylabel('power')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Peak "crispness" — peak / band mean
    ax = axes[3]
    ax.plot(t, crisp, color='darkgreen', lw=1)
    ax.axhline(np.nanmedian(crisp), color='k', ls='--', lw=0.5,
                label=f'median {np.nanmedian(crisp):.2f}')
    for te in t_events:
        ax.axvline(te, color='lime', alpha=0.4, lw=0.6)
    ax.set_ylabel('peak / band mean\n(crispness)')
    ax.set_xlabel('time (s)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{sub_id}_timelapse.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # Event-aligned summary: crispness / band_p / peak_f at event times
    if len(t_events) > 0:
        evt_rows = []
        for te in t_events:
            j = int(np.argmin(np.abs(t - te)))
            if 0 <= j < len(t):
                evt_rows.append({
                    't_event': te,
                    'peak_f': peak_f[j],
                    'peak_p': peak_p[j],
                    'band_p': band_p[j],
                    'crispness': crisp[j],
                    'peak_f_sr': peak_f_sr[j],
                    'peak_p_sr': peak_p_sr[j],
                    'crisp_sr': crisp_sr[j],
                    'peak_in_sr': bool(in_sr_mask[j]),
                })
        evt = pd.DataFrame(evt_rows)
        all_cr = crisp[np.isfinite(crisp)]
        all_pf = peak_f[np.isfinite(peak_f)]
        all_cr_sr = crisp_sr[np.isfinite(crisp_sr)]
        all_pp_sr = peak_p_sr[np.isfinite(peak_p_sr)]
        print(f"{sub_id}  n_events={len(evt)}")
        print(f"   BROAD [6-9 Hz]  peak_f ev-med={evt['peak_f'].median():.2f} "
              f"all-med={np.median(all_pf):.2f}   "
              f"crisp ev-med={evt['crispness'].median():.2f} "
              f"all-med={np.median(all_cr):.2f}")
        print(f"   SR   [7-8.2]    peak_f ev-med={evt['peak_f_sr'].median():.2f}   "
              f"peak_p ev-med={evt['peak_p_sr'].median():.1f} "
              f"all-med={np.median(all_pp_sr):.1f}   "
              f"crisp ev-med={evt['crisp_sr'].median():.2f} "
              f"all-med={np.median(all_cr_sr):.2f}")
        print(f"   Fraction of time unrestricted peak IN SR band: {pct_in_sr:.1f}%  "
              f"  fraction of events with peak IN SR band: {evt['peak_in_sr'].mean()*100:.1f}%")

    print(f"  saved {OUT_DIR}/{sub_id}_timelapse.png")


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 5)]
    # pick 5 subjects spread across n_events quartiles for variety
    if len(ok) >= N_SUBJECTS:
        ok_sorted = ok.sort_values('n_events').reset_index(drop=True)
        idxs = np.linspace(0, len(ok_sorted) - 1, N_SUBJECTS).astype(int)
        picks = ok_sorted.iloc[idxs]
    else:
        picks = ok
    print(f"Rendering {len(picks)} subjects: "
          f"{list(picks['subject_id'])}")
    for _, row in picks.iterrows():
        sid = row['subject_id']
        ep = os.path.join(EVENTS_DIR, f'{sid}_sie_events.csv')
        render_subject(sid, ep)


if __name__ == '__main__':
    main()
