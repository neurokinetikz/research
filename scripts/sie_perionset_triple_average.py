#!/usr/bin/env python3
"""
Peri-onset triple average: envelope z, Kuramoto R(t), and mean PLV to median,
aligned on t0_net across all SIE events in LEMON EC.

For each event:
  - Extract ±12 s around t0_net (pad for filter stability; analyze ±10 s)
  - Compute the three streams on that segment
  - Interpolate onto a common relative time grid
  - Accumulate across events

Output:
  - 3-panel figure: mean ± 95% bootstrap CI for each stream, -10 to +10 s
  - Also plot per-subject-mean traces (thin) behind the grand mean
  - CSV of time x stream x {mean, ci_lo, ci_hi}

Use subject-level cluster bootstrap (1000 iterations): resample subjects,
recompute grand mean, take 2.5 / 97.5 percentiles across iterations.
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

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'perionset')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')

F0 = 7.83
HALF_BW = 0.6
R_BAND = (7.2, 8.4)
PRE_SEC = 10.0
POST_SEC = 10.0
PAD_SEC = 2.0
STEP_SEC = 0.1
WIN_SEC = 1.0

# Common time grid, -PRE to +POST at STEP_SEC
TGRID = np.arange(-PRE_SEC, POST_SEC + STEP_SEC/2, STEP_SEC)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')


def bandpass(x, fs, f1, f2, order=4):
    ny = 0.5 * fs
    b, a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


def compute_streams(X_uV, fs):
    """Return (t_env, env_z), (t_win, R), (t_win, PLV) for the segment."""
    y_mean = X_uV.mean(axis=0)
    yb = bandpass(y_mean, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    z = zscore(env, nan_policy='omit')
    t_env = np.arange(len(z)) / fs

    Xb = bandpass(X_uV, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    ref = np.median(Xb, axis=0)
    ph_ref = np.angle(signal.hilbert(ref))
    dphi = ph - ph_ref[None, :]

    nwin = int(round(WIN_SEC * fs))
    nstep = int(round(STEP_SEC * fs))
    centers = []
    Rv = []
    Pv = []
    for i in range(0, X_uV.shape[1] - nwin + 1, nstep):
        seg = ph[:, i:i+nwin]
        R_t = np.abs(np.mean(np.exp(1j * seg), axis=0))
        Rv.append(float(np.mean(R_t)))
        pseg = dphi[:, i:i+nwin]
        plv_per_ch = np.abs(np.mean(np.exp(1j * pseg), axis=1))
        Pv.append(float(np.mean(plv_per_ch)))
        centers.append((i + nwin/2) / fs)
    return (t_env, z), (np.array(centers), np.array(Rv)), (np.array(centers), np.array(Pv))


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path)
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
    X_all = raw.get_data() * 1e6  # (n_ch, n_samples), µV
    t_end_rec = raw.times[-1]

    env_rows = []
    R_rows = []
    P_rows = []

    for _, ev in events.iterrows():
        t0 = float(ev.get('t0_net', np.nan))
        if not np.isfinite(t0):
            continue
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
            (t_env, zenv), (tR, R), (tP, P) = compute_streams(X_seg, fs)
        except Exception:
            continue
        # abs time → relative time (t0 = 0)
        rel_env = t_env - PAD_SEC - PRE_SEC
        rel_R = tR - PAD_SEC - PRE_SEC
        rel_P = tP - PAD_SEC - PRE_SEC
        # interpolate onto TGRID
        env_i = np.interp(TGRID, rel_env, zenv, left=np.nan, right=np.nan)
        R_i = np.interp(TGRID, rel_R, R, left=np.nan, right=np.nan)
        P_i = np.interp(TGRID, rel_P, P, left=np.nan, right=np.nan)
        env_rows.append(env_i)
        R_rows.append(R_i)
        P_rows.append(P_i)

    if not env_rows:
        return None

    # subject-level average across its events
    return {
        'subject_id': sub_id,
        'n_events': len(env_rows),
        'env': np.nanmean(np.array(env_rows), axis=0),
        'R': np.nanmean(np.array(R_rows), axis=0),
        'P': np.nanmean(np.array(P_rows), axis=0),
    }


def bootstrap_ci(subject_means, n_boot=1000, seed=0):
    """Subject-level cluster bootstrap: resample subjects, compute grand mean."""
    rng = np.random.default_rng(seed)
    n_sub = subject_means.shape[0]
    grand = np.nanmean(subject_means, axis=0)
    boots = np.zeros((n_boot, subject_means.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n_sub, size=n_sub)
        boots[b] = np.nanmean(subject_means[idx], axis=0)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return grand, lo, hi


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Find all subjects with events
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        events_path = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(events_path):
            tasks.append((r['subject_id'], events_path))
    print(f"Subjects to process: {len(tasks)}")

    with Pool(8) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")
    n_events_total = sum(r['n_events'] for r in results)
    print(f"Total events aggregated: {n_events_total}")

    env_arr = np.array([r['env'] for r in results])
    R_arr = np.array([r['R'] for r in results])
    P_arr = np.array([r['P'] for r in results])

    env_m, env_lo, env_hi = bootstrap_ci(env_arr)
    R_m, R_lo, R_hi = bootstrap_ci(R_arr)
    P_m, P_lo, P_hi = bootstrap_ci(P_arr)

    # Save CSV
    df = pd.DataFrame({
        't_rel': TGRID,
        'env_mean': env_m, 'env_ci_lo': env_lo, 'env_ci_hi': env_hi,
        'R_mean': R_m, 'R_ci_lo': R_lo, 'R_ci_hi': R_hi,
        'P_mean': P_m, 'P_ci_lo': P_lo, 'P_ci_hi': P_hi,
    })
    csv_path = os.path.join(OUT_DIR, 'perionset_triple_average.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, arr, m, lo, hi, label, color in [
        (axes[0], env_arr, env_m, env_lo, env_hi, 'envelope z (7.83 ± 0.6 Hz)', 'darkorange'),
        (axes[1], R_arr,   R_m,   R_lo,   R_hi,   'Kuramoto R(t) in 7.2–8.4 Hz', 'seagreen'),
        (axes[2], P_arr,   P_m,   P_lo,   P_hi,   'mean PLV to median',          'purple'),
    ]:
        # thin per-subject traces
        for i in range(arr.shape[0]):
            ax.plot(TGRID, arr[i], color='gray', alpha=0.08, lw=0.3)
        ax.fill_between(TGRID, lo, hi, color=color, alpha=0.25, label='95% bootstrap CI')
        ax.plot(TGRID, m, color=color, lw=2, label='grand mean')
        ax.axvline(0, color='k', ls='--', lw=0.6)
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
    axes[0].set_title(f'Peri-onset triple average — LEMON EC\n'
                      f'{len(results)} subjects, {n_events_total} events, aligned on t0_net')
    axes[2].set_xlabel('time relative to t0_net (s)')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'perionset_triple_average.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # Print peak info
    i0 = np.argmin(np.abs(TGRID))
    peak_idx_env = np.argmax(env_m)
    peak_idx_R = np.argmax(R_m)
    peak_idx_P = np.argmax(P_m)
    print(f"\nPeak locations:")
    print(f"  envelope z peak at t = {TGRID[peak_idx_env]:+.2f} s (value {env_m[peak_idx_env]:.3f})")
    print(f"  R(t) peak at t = {TGRID[peak_idx_R]:+.2f} s (value {R_m[peak_idx_R]:.3f})")
    print(f"  PLV peak at t = {TGRID[peak_idx_P]:+.2f} s (value {P_m[peak_idx_P]:.3f})")
    print(f"\nAt t=0:")
    print(f"  env z = {env_m[i0]:.3f}  [{env_lo[i0]:.3f}, {env_hi[i0]:.3f}]")
    print(f"  R     = {R_m[i0]:.3f}  [{R_lo[i0]:.3f}, {R_hi[i0]:.3f}]")
    print(f"  PLV   = {P_m[i0]:.3f}  [{P_lo[i0]:.3f}, {P_hi[i0]:.3f}]")


if __name__ == '__main__':
    main()
