#!/usr/bin/env python3
"""
A9 — Multi-stream peri-onset characterization (nadir-aligned).

Extends A3/A4a to 8 streams computed on a common time grid, aligned on the
joint-dip nadir:

  Core 4 streams:
    env_z    envelope z at F0 = 7.83 ± 0.6 Hz
    R        Kuramoto order parameter across channels in 7.2–8.4 Hz
    PLV      mean PLV to median (bandpassed) reference
    MSC      mean magnitude-squared coherence to raw median reference at F0
             (fixed: raw signal input, nperseg = 0.5 s)

  New 4 streams:
    wPLI     weighted phase-lag index to median (volume-conduction-robust)
    IF_tempo temporal dispersion: std of instantaneous frequency within
             a 1-s window, averaged across channels
    IF_spat  spatial dispersion: IQR of median(IF over window) across channels
    xPLV13   cross-harmonic PLV between sr1 and sr3 bands
             (phase of sr1 signal vs phase of sr3/2 — checks harmonic locking)
    slope    aperiodic slope via log-log linear regression on 2–45 Hz
             Welch PSD, excluding 6–14 Hz alpha region

For each event:
  - Extract ±15 s around t0_net
  - Compute all streams at 0.1 s grid (1 s sliding window)
  - Compute slope on 2 s sliding window at 0.2 s step (separate grid)
  - Joint-dip nadir = argmin(zE + zR + zP + zM) in [-3, +0.4] s
  - Realign on nadir, interpolate to -8..+8 s output grid

Aggregate: per-subject mean → grand mean with subject-level cluster bootstrap CI.

Output: 9-panel figure, CSV of all streams, text summary of peak/trough times.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import bootstrap_ci, bandpass

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'multistream')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)
SR3_BAND = (18, 22)
BROADBAND = (2, 45)
ALPHA_EXCLUDE = (6, 14)

STEP_SEC = 0.1
WIN_SEC = 1.0
SLOPE_WIN_SEC = 2.0
SLOPE_STEP_SEC = 0.2
PRE_SEC = 10.0
POST_SEC = 10.0
PAD_SEC = 5.0
TGRID = np.arange(-8.0, 8.0 + STEP_SEC/2, STEP_SEC)
DIP_WINDOW = (-3.0, 0.4)


# -------------------------------------------------------------------------
# Per-window computations
# -------------------------------------------------------------------------
def inst_freq_channel(sig_ch, fs):
    """Instantaneous frequency (Hz) from unwrapped Hilbert phase."""
    ph = np.unwrap(np.angle(signal.hilbert(sig_ch)))
    return np.gradient(ph, 1.0/fs) / (2 * np.pi)


def wpli_to_ref_window(ph_ch_win, ph_ref_win):
    """wPLI between one channel and reference for a window.
    ph_ch_win, ph_ref_win : (n_samples,) phase time series.
    """
    cdiff = np.exp(1j * (ph_ch_win - ph_ref_win))
    imag = np.imag(cdiff)
    num = np.abs(np.mean(imag))
    den = np.mean(np.abs(imag))
    return (num / den) if den > 1e-12 else 0.0


def cross_harmonic_plv_window(sr1_ph_win, sr3_ph_win):
    """Cross-harmonic PLV: sr1 phase vs sr3 phase/3 (n:m=1:3 locking)."""
    # Use n:m locking - 3*sr1 phase vs 1*sr3 phase (equivalent to sr1 vs sr3/3)
    dphi = 3 * sr1_ph_win - sr3_ph_win
    return float(np.abs(np.mean(np.exp(1j * dphi))))


def slope_via_loglog(freqs, psd, brange=BROADBAND, exclude=ALPHA_EXCLUDE):
    """Linear regression in log(freq) vs log(psd), excluding alpha region.
    Returns slope (negative for 1/f^alpha, alpha = -slope)."""
    mask = (freqs >= brange[0]) & (freqs <= brange[1]) & ~(
        (freqs >= exclude[0]) & (freqs <= exclude[1])
    )
    if mask.sum() < 5:
        return np.nan
    lf = np.log10(freqs[mask])
    lp = np.log10(psd[mask] + 1e-30)
    good = np.isfinite(lf) & np.isfinite(lp)
    if good.sum() < 5:
        return np.nan
    m, _ = np.polyfit(lf[good], lp[good], 1)
    return float(m)  # negative for usual 1/f


# -------------------------------------------------------------------------
# Stream extraction for an event segment
# -------------------------------------------------------------------------
def compute_all_streams(X_uV, fs):
    """Returns dict of streams + common time grid (seconds from segment start)."""
    n_ch, n_samp = X_uV.shape
    # Bandpassed signals
    y = X_uV.mean(axis=0)
    yb = bandpass(y, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))

    Xb = bandpass(X_uV, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    ref_b = np.median(Xb, axis=0)
    ph_ref = np.angle(signal.hilbert(ref_b))

    ref_raw = np.median(X_uV, axis=0)

    # sr3 band for cross-harmonic
    Xsr3 = bandpass(X_uV, fs, SR3_BAND[0], SR3_BAND[1])
    ph_sr3 = np.angle(signal.hilbert(Xsr3, axis=-1))

    # per-channel instantaneous frequency
    IF = np.zeros_like(Xb)
    for ci in range(n_ch):
        IF[ci] = inst_freq_channel(Xb[ci], fs)

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    nperseg_msc = max(int(round(0.5 * fs)), 32)

    centers = []
    env_v, R_v, P_v, M_v, W_v = [], [], [], [], []
    IFt_v, IFs_v, X13_v = [], [], []
    for i in range(0, n_samp - nwin + 1, nstep):
        seg_ph = ph[:, i:i+nwin]
        seg_ph_ref = ph_ref[i:i+nwin]
        seg_ph_sr3 = ph_sr3[:, i:i+nwin]
        seg_IF = IF[:, i:i+nwin]

        # Kuramoto R
        R_t = np.abs(np.mean(np.exp(1j * seg_ph), axis=0))
        R_v.append(float(np.mean(R_t)))
        # PLV to ref
        dphi = seg_ph - seg_ph_ref[None, :]
        plv_ch = np.abs(np.mean(np.exp(1j * dphi), axis=1))
        P_v.append(float(np.mean(plv_ch)))
        # envelope
        env_v.append(float(np.mean(env[i:i+nwin])))
        # MSC (raw signals)
        ref_seg_raw = ref_raw[i:i+nwin]
        msc = []
        for ci in range(n_ch):
            try:
                f_c, Cxy = signal.coherence(X_uV[ci, i:i+nwin], ref_seg_raw, fs=fs,
                                              nperseg=min(nperseg_msc, nwin))
                msc.append(float(Cxy[int(np.argmin(np.abs(f_c - F0)))]))
            except Exception:
                pass
        M_v.append(float(np.mean(msc)) if msc else np.nan)
        # wPLI
        wpli = [wpli_to_ref_window(seg_ph[ci], seg_ph_ref) for ci in range(n_ch)]
        W_v.append(float(np.mean(wpli)))
        # IF temporal: std of IF over window per channel, averaged across channels
        IFt = np.std(seg_IF, axis=1)
        IFt_v.append(float(np.mean(IFt)))
        # IF spatial: median IF per channel in window, then IQR across channels
        IFm = np.median(seg_IF, axis=1)
        IFs_v.append(float(np.percentile(IFm, 75) - np.percentile(IFm, 25)))
        # cross-harmonic PLV sr1-sr3 (per channel, mean)
        x13_per = [cross_harmonic_plv_window(seg_ph[ci], seg_ph_sr3[ci])
                    for ci in range(n_ch)]
        X13_v.append(float(np.mean(x13_per)))

        centers.append((i + nwin/2) / fs)

    # Slope (coarser grid)
    nwin_s = int(round(SLOPE_WIN_SEC * fs))
    nstep_s = int(round(SLOPE_STEP_SEC * fs))
    slope_centers = []
    slope_v = []
    for i in range(0, n_samp - nwin_s + 1, nstep_s):
        seg = y[i:i+nwin_s]
        try:
            f_c, Pxx = signal.welch(seg, fs=fs, nperseg=min(int(fs*1.0), nwin_s))
            slope_v.append(slope_via_loglog(f_c, Pxx))
        except Exception:
            slope_v.append(np.nan)
        slope_centers.append((i + nwin_s/2) / fs)

    return {
        'centers': np.array(centers),
        'env': np.array(env_v), 'R': np.array(R_v), 'PLV': np.array(P_v),
        'MSC': np.array(M_v), 'wPLI': np.array(W_v),
        'IFt': np.array(IFt_v), 'IFs': np.array(IFs_v), 'xPLV13': np.array(X13_v),
        'slope_centers': np.array(slope_centers),
        'slope': np.array(slope_v),
    }


# -------------------------------------------------------------------------
# Joint-dip detection (local z-scoring of 4 primary streams)
# -------------------------------------------------------------------------
def find_nadir(t_rel, env, R, P, M):
    base = (t_rel >= -5.0) & (t_rel < -3.0)
    srch = (t_rel >= DIP_WINDOW[0]) & (t_rel <= DIP_WINDOW[1])
    if not base.any() or not srch.any():
        return np.nan
    def lz(x):
        mu = np.nanmean(x[base]); sd = np.nanstd(x[base])
        if not np.isfinite(sd) or sd < 1e-9: sd = 1.0
        return (x - mu) / sd
    s = lz(env) + lz(R) + lz(P) + lz(M)
    s_m = np.where(srch, s, np.inf)
    return float(t_rel[int(np.nanargmin(s_m))])


# -------------------------------------------------------------------------
# Per-subject processing
# -------------------------------------------------------------------------
STREAMS = ['env', 'R', 'PLV', 'MSC', 'wPLI', 'IFt', 'IFs', 'xPLV13', 'slope']


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) == 0:
        return None
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X_all = raw.get_data() * 1e6
    t_end_rec = raw.times[-1]

    # Preallocate per-event matrices
    event_rows = {s: [] for s in STREAMS}

    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end_rec:
            continue
        i0 = int(round(lo * fs))
        i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue
        try:
            S = compute_all_streams(X_seg, fs)
        except Exception:
            continue
        rel = S['centers'] - PAD_SEC - PRE_SEC
        slope_rel = S['slope_centers'] - PAD_SEC - PRE_SEC

        nadir = find_nadir(rel, S['env'], S['R'], S['PLV'], S['MSC'])
        if not np.isfinite(nadir):
            continue
        rel_to_nadir = rel - nadir
        slope_rel_to_nadir = slope_rel - nadir

        for s in STREAMS[:-1]:  # all but slope (different grid)
            arr = np.interp(TGRID, rel_to_nadir, S[s], left=np.nan, right=np.nan)
            event_rows[s].append(arr)
        # slope on its own grid, still interp to common TGRID
        arr = np.interp(TGRID, slope_rel_to_nadir, S['slope'], left=np.nan, right=np.nan)
        event_rows['slope'].append(arr)

    if not event_rows['env']:
        return None

    # Per-subject mean over events
    out = {'subject_id': sub_id, 'n_events': len(event_rows['env'])}
    for s in STREAMS:
        out[s] = np.nanmean(np.array(event_rows[s]), axis=0)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path))
    print(f"Subjects: {len(tasks)}")

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    n_events = sum(r['n_events'] for r in results)
    print(f"Successful subjects: {len(results)}")
    print(f"Total events: {n_events}")

    # Grand-average + bootstrap CI per stream
    gm = {}
    for s in STREAMS:
        arr = np.array([r[s] for r in results])
        m, lo, hi = bootstrap_ci(arr, n_boot=500)
        gm[s] = {'arr': arr, 'mean': m, 'lo': lo, 'hi': hi}
        peak_t = TGRID[np.argmax(m)]
        trough_t = TGRID[np.argmin(m)]
        print(f"  {s:8s}: peak {peak_t:+.2f}s ({m[np.argmax(m)]:.4f}), "
              f"trough {trough_t:+.2f}s ({m[np.argmin(m)]:.4f})")

    # Save CSV
    out_csv = {'t_rel': TGRID}
    for s in STREAMS:
        out_csv[f'{s}_mean'] = gm[s]['mean']
        out_csv[f'{s}_ci_lo'] = gm[s]['lo']
        out_csv[f'{s}_ci_hi'] = gm[s]['hi']
    pd.DataFrame(out_csv).to_csv(
        os.path.join(OUT_DIR, 'multistream_nadir_aligned.csv'), index=False)

    # Figure
    labels = {
        'env': 'envelope z (7.83 Hz)',
        'R': 'Kuramoto R',
        'PLV': 'mean PLV to median',
        'MSC': 'mean MSC at F0',
        'wPLI': 'mean wPLI to median',
        'IFt': 'IF temporal dispersion (Hz)',
        'IFs': 'IF spatial dispersion (Hz)',
        'xPLV13': 'cross-harmonic PLV sr1↔sr3',
        'slope': 'aperiodic slope (log-log)',
    }
    colors = {
        'env': 'darkorange', 'R': 'seagreen', 'PLV': 'purple', 'MSC': 'steelblue',
        'wPLI': 'teal', 'IFt': 'brown', 'IFs': 'olive', 'xPLV13': 'crimson',
        'slope': 'slategray',
    }
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    for idx, s in enumerate(STREAMS):
        ax = axes.flat[idx]
        ax.fill_between(TGRID, gm[s]['lo'], gm[s]['hi'], color=colors[s], alpha=0.25)
        ax.plot(TGRID, gm[s]['mean'], color=colors[s], lw=2)
        ax.axvline(0, color='k', ls='--', lw=0.6, alpha=0.7)
        peak_t = TGRID[np.argmax(gm[s]['mean'])]
        trough_t = TGRID[np.argmin(gm[s]['mean'])]
        ax.axvline(peak_t, color='red', ls=':', lw=0.5, alpha=0.6)
        ax.set_title(f'{labels[s]}\npeak {peak_t:+.2f}s  trough {trough_t:+.2f}s',
                     fontsize=10)
        ax.set_ylabel(s)
    for ax in axes[-1]:
        ax.set_xlabel('time relative to nadir (s)')
    fig.suptitle(f'A9 — Multi-stream peri-onset (nadir-aligned)\n'
                 f'LEMON EC · {len(results)} subjects · {n_events} events',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'multistream_nadir_aligned.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/multistream_nadir_aligned.png")


if __name__ == '__main__':
    main()
