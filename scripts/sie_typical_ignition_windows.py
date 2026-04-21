#!/usr/bin/env python3
"""
Pick a handful of SIE events from extracted LEMON EC output and visualize
the 20-s window around each onset to see what a 'typical' ignition looks like.

For each chosen event, produces a 6-panel figure:
  (1) Raw EEG traces (subset, centered on onset)
  (2) SR1-bandpassed mean signal (7.83 ± 0.6 Hz)
  (3) Envelope z-score (what Stage 1 currently thresholds)
  (4) Kuramoto R(t) across channels in 7.2-8.4 Hz
  (5) Mean PLV to median reference (sliding window)
  (6) Spectrogram 1-20 Hz, showing harmonic structure during window

Also writes summary CSV of the per-event metrics at window level.

Outputs to outputs/schumann/images/typical_ignitions/.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'typical_ignitions')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)


def bandpass(x, fs, f1, f2, order=4):
    ny = 0.5 * fs
    b, a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


def envelope_z(y, fs, f0=F0, half_bw=HALF_BW):
    yb = bandpass(y, fs, f0 - half_bw, f0 + half_bw)
    env = np.abs(signal.hilbert(yb))
    return zscore(env, nan_policy='omit'), env


def kuramoto_R(X, fs, band=R_BAND, win_sec=1.0, step_sec=0.1):
    """Sliding Kuramoto order parameter across channels."""
    Xb = bandpass(X, fs, band[0], band[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    nwin = int(round(win_sec * fs))
    nstep = int(round(step_sec * fs))
    n_samples = X.shape[1]
    centers = []
    Rvals = []
    for i in range(0, n_samples - nwin + 1, nstep):
        # mean phase coherence across channels at each timepoint, then mean over window
        seg = ph[:, i:i+nwin]
        R_t = np.abs(np.mean(np.exp(1j * seg), axis=0))  # per timepoint across chans
        Rvals.append(float(np.mean(R_t)))
        centers.append((i + nwin/2) / fs)
    return np.array(centers), np.array(Rvals)


def mean_plv_to_median(X, fs, band=R_BAND, win_sec=1.0, step_sec=0.1):
    """Mean PLV of each channel to median reference, sliding."""
    Xb = bandpass(X, fs, band[0], band[1])
    ref = np.median(Xb, axis=0)
    ph_ref = np.angle(signal.hilbert(ref))
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    dphi = ph - ph_ref[None, :]
    nwin = int(round(win_sec * fs))
    nstep = int(round(step_sec * fs))
    centers = []
    plvs = []
    for i in range(0, X.shape[1] - nwin + 1, nstep):
        seg = dphi[:, i:i+nwin]
        plv_per_ch = np.abs(np.mean(np.exp(1j * seg), axis=1))
        plvs.append(float(np.mean(plv_per_ch)))
        centers.append((i + nwin/2) / fs)
    return np.array(centers), np.array(plvs)


def pick_events(sub_id, n=3, rng_seed=0):
    """Pick high/median/low quality events by sr_score."""
    path = os.path.join(EVENTS_DIR, f'{sub_id}_sie_events.csv')
    df = pd.read_csv(path)
    df = df.dropna(subset=['sr_score']).sort_values('sr_score').reset_index(drop=True)
    if len(df) < n:
        return df
    idx = [len(df) - 1, len(df)//2, 0][:n]  # top, median, bottom
    return df.iloc[idx].reset_index(drop=True)


def analyze_event(raw, event_row, out_path):
    fs = raw.info['sfreq']
    t_start = float(event_row['t_start'])
    t_end = float(event_row['t_end'])
    t0_net = float(event_row.get('t0_net', t_start + 10))
    z_peak_t = float(event_row.get('sr_z_peak_t', t0_net))

    # Crop ±2 s beyond window for filter stability
    pad = 2.0
    t_lo = max(0, t_start - pad)
    t_hi = min(raw.times[-1], t_end + pad)

    cropped = raw.copy().crop(tmin=t_lo, tmax=t_hi)
    X = cropped.get_data()  # (n_ch, n_samples), Volts
    X_uV = X * 1e6
    times = cropped.times + t_lo

    # Whole-window signals
    y_mean = X_uV.mean(axis=0)
    z, env = envelope_z(y_mean, fs)
    tR, R = kuramoto_R(X_uV, fs)
    tR_abs = tR + t_lo
    tP, P = mean_plv_to_median(X_uV, fs)
    tP_abs = tP + t_lo

    # Spectrogram
    f_sp, t_sp, Sxx = signal.spectrogram(y_mean, fs, nperseg=int(2*fs),
                                          noverlap=int(1.8*fs), nfft=int(8*fs))
    fmask = (f_sp >= 1) & (f_sp <= 22)
    Sxx = 10 * np.log10(Sxx[fmask] + 1e-20)
    f_sp = f_sp[fmask]

    # Figure
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    sub = event_row['subject_id']
    score = event_row.get('sr_score', np.nan)
    fig.suptitle(f"{sub} — event [{t_start:.1f}, {t_end:.1f}] s, sr_score={score:.2f}",
                 fontsize=12)

    # 1) raw traces — show 5 evenly-spaced channels
    ch_idx = np.linspace(0, X_uV.shape[0]-1, 5).astype(int)
    offset = 80
    for k, ci in enumerate(ch_idx):
        axes[0].plot(times, X_uV[ci] + k*offset, lw=0.5,
                     label=cropped.ch_names[ci])
    axes[0].set_ylabel('raw (µV, offset)')
    axes[0].legend(loc='upper right', fontsize=7, ncol=5)

    # 2) SR1 bandpassed mean
    y_sr1 = bandpass(y_mean, fs, F0-HALF_BW, F0+HALF_BW)
    axes[1].plot(times, y_sr1, lw=0.7, color='steelblue')
    axes[1].set_ylabel(f'{F0}±{HALF_BW} Hz\nmean (µV)')

    # 3) envelope z
    axes[2].plot(times, z, lw=0.8, color='darkorange')
    axes[2].axhline(3, color='k', ls='--', lw=0.6, label='z=3')
    axes[2].set_ylabel('envelope z')
    axes[2].legend(loc='upper right', fontsize=8)

    # 4) Kuramoto R(t)
    axes[3].plot(tR_abs, R, lw=0.8, color='seagreen')
    axes[3].set_ylabel(f'R(t) in\n{R_BAND[0]}-{R_BAND[1]} Hz')
    axes[3].set_ylim(0, 1)

    # 5) mean PLV
    axes[4].plot(tP_abs, P, lw=0.8, color='purple')
    axes[4].set_ylabel('mean PLV\nto median')
    axes[4].set_ylim(0, 1)

    # 6) spectrogram
    t_sp_abs = t_sp + t_lo
    im = axes[5].pcolormesh(t_sp_abs, f_sp, Sxx, shading='auto', cmap='viridis')
    axes[5].set_ylabel('freq (Hz)')
    axes[5].set_xlabel('time (s)')
    axes[5].axhline(F0, color='w', ls='--', lw=0.5, alpha=0.6)
    for fh in [10, 12, 13.75, 20]:
        axes[5].axhline(fh, color='w', ls=':', lw=0.4, alpha=0.4)

    # mark event boundaries + onsets on all axes
    for ax in axes[:-1]:
        ax.axvspan(t_start, t_end, color='red', alpha=0.08)
        ax.axvline(t0_net, color='red', ls='-', lw=0.8, alpha=0.7)
        ax.axvline(z_peak_t, color='darkred', ls=':', lw=0.8, alpha=0.7)
    axes[5].axvspan(t_start, t_end, color='red', alpha=0.08)
    axes[5].axvline(t0_net, color='white', ls='-', lw=0.8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()

    # Per-event summary
    win_mask_R = (tR_abs >= t_start) & (tR_abs <= t_end)
    win_mask_P = (tP_abs >= t_start) & (tP_abs <= t_end)
    pre_mask_R = (tR_abs >= t_start - 20) & (tR_abs < t_start)
    pre_mask_P = (tP_abs >= t_start - 20) & (tP_abs < t_start)
    # find time of envelope z peak
    t_z_samp = np.arange(len(z)) / fs + t_lo
    win_mask_z = (t_z_samp >= t_start) & (t_z_samp <= t_end)
    return {
        'subject_id': sub,
        't_start': t_start, 't_end': t_end, 't0_net': t0_net,
        'sr_score': score,
        'z_peak_in_window': float(np.nanmax(z[win_mask_z])) if win_mask_z.any() else np.nan,
        'R_mean_in_window': float(np.nanmean(R[win_mask_R])) if win_mask_R.any() else np.nan,
        'R_peak_in_window': float(np.nanmax(R[win_mask_R])) if win_mask_R.any() else np.nan,
        'R_mean_pre20s': float(np.nanmean(R[pre_mask_R])) if pre_mask_R.any() else np.nan,
        'PLV_mean_in_window': float(np.nanmean(P[win_mask_P])) if win_mask_P.any() else np.nan,
        'PLV_mean_pre20s': float(np.nanmean(P[pre_mask_P])) if pre_mask_P.any() else np.nan,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # pick 3 subjects with plenty of events
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    summary = summary[summary['status'] == 'ok'].sort_values('n_events', ascending=False)
    subjects = summary['subject_id'].head(3).tolist()
    print(f"Subjects: {subjects}")

    rows = []
    for sub in subjects:
        print(f"\n=== {sub} ===")
        raw = load_lemon(sub, condition='EC')
        if raw is None:
            print(f"  no data")
            continue
        print(f"  loaded {len(raw.ch_names)} ch, {raw.times[-1]:.1f}s, fs={raw.info['sfreq']}")
        ev = pick_events(sub, n=3)
        for i, row in ev.iterrows():
            tag = ['top', 'median', 'bottom'][i] if i < 3 else f'rank{i}'
            out_path = os.path.join(OUT_DIR, f'{sub}_{tag}_{row["t_start"]:.0f}s.png')
            print(f"  [{tag}] t=[{row['t_start']:.1f}, {row['t_end']:.1f}] "
                  f"sr_score={row['sr_score']:.2f} -> {os.path.basename(out_path)}")
            try:
                summary_row = analyze_event(raw, row, out_path)
                summary_row['rank'] = tag
                rows.append(summary_row)
            except Exception as e:
                print(f"    ERROR: {e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'window_metrics_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")
    print(df.to_string())


if __name__ == '__main__':
    main()
