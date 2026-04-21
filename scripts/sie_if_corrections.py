#!/usr/bin/env python3
"""
A10-corrected — Three-way correction for the Hilbert-IF filter artifact.

Per event (nadir-aligned) computes:
  (1) peak_freq_FOOOF(t)  — sliding FOOOF peak location in 3-15 Hz fit range,
      restricted to the peak nearest 7.83 Hz within [6, 9.5] Hz.
      Data-driven, filter-independent.
  (2) IF_wide(t)  — Hilbert IF on a 4-12 Hz passband (10× wider than original)
  (3) IFt_wide(t) — temporal dispersion of the wide-passband IF
      (as in A9 but with the wide filter)
  (4) env_amplitude(t) — raw envelope at F0 ± 0.6 Hz (to use as SNR regressor)

After peri-nadir averaging per subject:
  - Compare peak_freq_FOOOF(t) and IF_wide(t) to the A10 narrow-band result
  - Regress IFt_wide on env amplitude across time per subject; report
    partial R² and whether the peri-nadir peak in IFt survives amplitude-regression
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
from scripts.sie_perionset_triple_average import bootstrap_ci, bandpass
from scripts.sie_dip_onset_and_narrow_fooof import compute_streams_4way
from scripts.sie_perionset_multistream import (
    F0, PRE_SEC, POST_SEC, PAD_SEC, STEP_SEC, WIN_SEC, find_nadir, TGRID,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'if_corrections')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

WIDE_BAND = (4.0, 12.0)
NARROW_BAND = (7.2, 8.4)
FOOOF_FIT_RANGE = (3.0, 15.0)
FOOOF_PEAK_SEARCH = (6.0, 9.5)

FOOOF_WIN_SEC = 2.0
FOOOF_STEP_SEC = 0.5   # fewer fits — expensive


def inst_freq(sig_ch, fs):
    ph = np.unwrap(np.angle(signal.hilbert(sig_ch)))
    return np.gradient(ph, 1.0/fs) / (2 * np.pi)


def sliding_fooof_peak(y, fs, win_sec=FOOOF_WIN_SEC, step_sec=FOOOF_STEP_SEC):
    from fooof import FOOOF
    nwin = int(round(win_sec * fs))
    nstep = int(round(step_sec * fs))
    centers, peaks = [], []
    for i in range(0, len(y) - nwin + 1, nstep):
        seg = y[i:i+nwin]
        try:
            f_w, Pxx = signal.welch(seg, fs=fs, nperseg=min(nwin, int(fs*1.0)))
            fm = FOOOF(peak_width_limits=(0.5, 3.0), max_n_peaks=6,
                        peak_threshold=0.01, min_peak_height=0.01,
                        verbose=False)
            fm.fit(f_w, Pxx, FOOOF_FIT_RANGE)
            gp = fm.get_params('peak_params')
            if gp is None or len(gp) == 0:
                peaks.append(np.nan)
            else:
                # nearest to F0 within search range
                cands = gp[(gp[:, 0] >= FOOOF_PEAK_SEARCH[0]) &
                             (gp[:, 0] <= FOOOF_PEAK_SEARCH[1])]
                if len(cands) == 0:
                    peaks.append(np.nan)
                else:
                    # peak with highest power
                    peaks.append(float(cands[np.argmax(cands[:, 1]), 0]))
        except Exception:
            peaks.append(np.nan)
        centers.append((i + nwin/2) / fs)
    return np.array(centers), np.array(peaks)


def compute_if_wide(y, fs):
    yb = bandpass(y, fs, WIDE_BAND[0], WIDE_BAND[1])
    return inst_freq(yb, fs)


def compute_ift_wide(X_uV, fs):
    """IF temporal dispersion using wide passband."""
    n_ch = X_uV.shape[0]
    IF = np.array([inst_freq(bandpass(X_uV[ci], fs, WIDE_BAND[0], WIDE_BAND[1]), fs)
                    for ci in range(n_ch)])
    IF = np.clip(IF, WIDE_BAND[0], WIDE_BAND[1])
    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    centers, IFt = [], []
    for i in range(0, X_uV.shape[1] - nwin + 1, nstep):
        seg = IF[:, i:i+nwin]
        IFt.append(float(np.mean(np.std(seg, axis=1))))
        centers.append((i + nwin/2) / fs)
    return np.array(centers), np.array(IFt)


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
    t_end = raw.times[-1]

    fooof_rows, if_wide_rows, ift_wide_rows, env_rows = [], [], [], []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end:
            continue
        i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue
        # Narrow streams needed for nadir alignment
        try:
            t_c, env, R, P, M = compute_streams_4way(X_seg, fs)
        except Exception:
            continue
        rel = t_c - PAD_SEC - PRE_SEC
        nadir = find_nadir(rel, env, R, P, M)
        if not np.isfinite(nadir):
            continue
        rel_n = rel - nadir  # narrow-stream relative time

        # Wide IF stream (same grid as narrow streams via WIN_SEC)
        y = X_seg.mean(axis=0)
        t_if_wide = rel_n  # same indexing
        try:
            # ift_wide uses same WIN_SEC/STEP_SEC as compute_streams_4way
            t_ift, ift_wide = compute_ift_wide(X_seg, fs)
            rel_ift = t_ift - PAD_SEC - PRE_SEC - nadir
            # IF_wide — just the mean-channel value at window centers
            IF_w_arr = compute_if_wide(y, fs)
            # window-averaged to match grid
            nwin = int(round(WIN_SEC * fs))
            nstep = int(round(STEP_SEC * fs))
            IF_w_win = []
            for i in range(0, len(IF_w_arr) - nwin + 1, nstep):
                seg = IF_w_arr[i:i+nwin]
                seg = seg[(seg >= WIDE_BAND[0]) & (seg <= WIDE_BAND[1])]
                IF_w_win.append(float(np.nanmean(seg)) if len(seg) else np.nan)
            rel_IF = (np.arange(len(IF_w_win)) * STEP_SEC + WIN_SEC/2) - PAD_SEC - PRE_SEC - nadir
        except Exception:
            continue

        # Sliding FOOOF peak location
        try:
            t_f, peaks = sliding_fooof_peak(y, fs)
            rel_f = t_f - PAD_SEC - PRE_SEC - nadir
        except Exception:
            continue

        # Interpolate to TGRID
        fooof_rows.append(np.interp(TGRID, rel_f, peaks, left=np.nan, right=np.nan))
        if_wide_rows.append(np.interp(TGRID, rel_IF, np.array(IF_w_win),
                                         left=np.nan, right=np.nan))
        ift_wide_rows.append(np.interp(TGRID, rel_ift, ift_wide,
                                          left=np.nan, right=np.nan))
        env_rows.append(np.interp(TGRID, rel_n, env, left=np.nan, right=np.nan))

    if not fooof_rows:
        return None
    return {
        'subject_id': sub_id,
        'n_events': len(fooof_rows),
        'fooof_peak': np.nanmean(np.array(fooof_rows), axis=0),
        'IF_wide': np.nanmean(np.array(if_wide_rows), axis=0),
        'IFt_wide': np.nanmean(np.array(ift_wide_rows), axis=0),
        'env': np.nanmean(np.array(env_rows), axis=0),
    }


def regress_out(y, x):
    """Return residuals of y on x (linear)."""
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 10:
        return y * np.nan, np.nan
    b, a = np.polyfit(x[mask], y[mask], 1)
    resid = y - (b * x + a)
    # partial R² explained
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    ss_res = np.sum(resid[mask] ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return resid, r2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path))
    import os as _os
    n_workers = int(_os.environ.get('SIE_WORKERS', min(30, _os.cpu_count() or 8)))
    print(f"Subjects: {len(tasks)}  workers: {n_workers}")
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    n_events = sum(r['n_events'] for r in results)
    print(f"Successful subjects: {len(results)}, events: {n_events}")

    fooof_arr = np.array([r['fooof_peak'] for r in results])
    if_wide_arr = np.array([r['IF_wide']  for r in results])
    ift_wide_arr = np.array([r['IFt_wide'] for r in results])
    env_arr = np.array([r['env']     for r in results])

    fooof_m, fooof_lo, fooof_hi = bootstrap_ci(fooof_arr, n_boot=500)
    ifw_m, ifw_lo, ifw_hi       = bootstrap_ci(if_wide_arr, n_boot=500)
    iftw_m, iftw_lo, iftw_hi    = bootstrap_ci(ift_wide_arr, n_boot=500)
    env_m, env_lo, env_hi       = bootstrap_ci(env_arr, n_boot=500)

    # SNR correction: regress IFt_wide on env per-subject, then grand-average residuals
    resid_rows = []
    r2_vals = []
    for r in results:
        resid, r2 = regress_out(r['IFt_wide'], r['env'])
        resid_rows.append(resid)
        if np.isfinite(r2):
            r2_vals.append(r2)
    resid_arr = np.array(resid_rows)
    res_m, res_lo, res_hi = bootstrap_ci(resid_arr, n_boot=500)

    print(f"\nSummary (nadir t=0):")
    i0 = int(np.argmin(np.abs(TGRID)))
    for name, m, lo, hi in [
        ('FOOOF peak (Hz)', fooof_m, fooof_lo, fooof_hi),
        ('IF wide (Hz)',    ifw_m,   ifw_lo,   ifw_hi),
        ('IFt wide (Hz)',   iftw_m,  iftw_lo,  iftw_hi),
        ('env',             env_m,   env_lo,   env_hi),
        ('IFt resid',       res_m,   res_lo,   res_hi),
    ]:
        i_peak = int(np.argmax(m)); i_trough = int(np.argmin(m))
        print(f"  {name:20s} value@t0 = {m[i0]:.4f} "
              f"peak {TGRID[i_peak]:+.2f}s={m[i_peak]:.4f} "
              f"trough {TGRID[i_trough]:+.2f}s={m[i_trough]:.4f}")
    print(f"\nMedian per-subject R² of IFt~env: {np.median(r2_vals):.3f}")
    print(f"  (if high, IFt is mostly explained by amplitude; the residual is the"
           f" non-SNR dispersion structure)")

    # Save CSV
    df = pd.DataFrame({
        't_rel': TGRID,
        'fooof_peak_mean': fooof_m, 'fooof_peak_lo': fooof_lo, 'fooof_peak_hi': fooof_hi,
        'IF_wide_mean': ifw_m, 'IF_wide_lo': ifw_lo, 'IF_wide_hi': ifw_hi,
        'IFt_wide_mean': iftw_m, 'IFt_wide_lo': iftw_lo, 'IFt_wide_hi': iftw_hi,
        'IFt_resid_mean': res_m, 'IFt_resid_lo': res_lo, 'IFt_resid_hi': res_hi,
        'env_mean': env_m,
    })
    df.to_csv(os.path.join(OUT_DIR, 'if_corrections.csv'), index=False)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    ax = axes[0, 0]
    ax.fill_between(TGRID, fooof_lo, fooof_hi, color='navy', alpha=0.25)
    ax.plot(TGRID, fooof_m, color='navy', lw=2)
    ax.axhline(F0, color='red', ls='--', lw=0.8, label=f'F₀ = {F0} Hz')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_ylim(6.0, 9.5)
    ax.set_ylabel('FOOOF peak frequency (Hz)')
    ax.set_title('Sliding FOOOF peak in 3–15 Hz fit, search 6–9.5 Hz')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.fill_between(TGRID, ifw_lo, ifw_hi, color='darkgreen', alpha=0.25)
    ax.plot(TGRID, ifw_m, color='darkgreen', lw=2)
    ax.axhline(F0, color='red', ls='--', lw=0.8, label=f'F₀ = {F0} Hz')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_ylim(4.0, 12.0)
    ax.set_ylabel('mean IF, wide passband 4–12 Hz (Hz)')
    ax.set_title('Hilbert IF with wide passband')
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.fill_between(TGRID, iftw_lo, iftw_hi, color='brown', alpha=0.25)
    ax.plot(TGRID, iftw_m, color='brown', lw=2, label='raw IFt wide')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('IFt wide (Hz std)')
    ax.set_title('Wide-band IF temporal dispersion')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.fill_between(TGRID, res_lo, res_hi, color='coral', alpha=0.25)
    ax.plot(TGRID, res_m, color='coral', lw=2,
            label=f'IFt residual (R²={np.median(r2_vals):.2f} explained by env)')
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('IFt residual after regressing on env')
    ax.set_title('SNR-corrected IFt (non-amplitude part)')
    ax.legend(fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel('time relative to nadir (s)')
    fig.suptitle(f'A10-corrected — filter-artifact-corrected IF analyses\n'
                 f'LEMON EC · {len(results)} subj · {n_events} events',
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'if_corrections.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/if_corrections.png")


if __name__ == '__main__':
    main()
