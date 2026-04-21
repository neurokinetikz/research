#!/usr/bin/env python3
"""
B15 — SR-band-peak re-centered detector.

For each existing Stage 1 event, redefine t0 as the SR-band peak-power maximum
inside [t0_net - 5, t0_net + 5] s (call it t0_sr). Then re-compute the two
peri-event signatures on the new alignment and compare:

  - Envelope z (narrowband 7.83 ± 0.6 Hz Hilbert envelope) — should sharpen
    if the SR-band peak is a better event center than the envelope-z threshold
    crossing.
  - SR-band peak-power boost — by construction tighter under t0_sr alignment
    (peak is now at t = 0), but amplitude and width give a cleaner spec.

Stratify by template_rho quartile. Cohort grand mean + subject-level bootstrap
95% CI. Compare Q4 signatures under t0_net alignment vs t0_sr alignment.

Outputs:
  - per_event_t0_shift.csv  : every event's t0_shift = t0_sr - t0_net
  - figure: 2x2 panel — envelope z (t0_net vs t0_sr) and SR boost (same)
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
from scripts.sie_perionset_triple_average import bandpass

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality', 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

WIN_SEC = 4.0
HOP_SEC = 1.0
FREQ_LO, FREQ_HI = 2.0, 20.0
SR_LO, SR_HI = 7.0, 8.2
F0_FIXED = 7.83
HALF_BW = 0.6
NFFT_MULT = 4

TGRID = np.arange(-15.0, 15.0 + 0.05, 0.1)   # envelope/finer grid
TGRID_PSD = np.arange(-15.0, 15.0 + 0.5, 1.0)  # PSD grid (1 s hop)
SR_SEARCH_HALF = 5.0                           # ± 5 s of t0_net to find peak


def sliding_welch(x, fs):
    nperseg = int(round(WIN_SEC * fs))
    nhop = int(round(HOP_SEC * fs))
    nfft = nperseg * NFFT_MULT
    freqs_full = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_mask = (freqs_full >= FREQ_LO) & (freqs_full <= FREQ_HI)
    freqs = freqs_full[f_mask]
    win = signal.windows.hann(nperseg)
    win_pow = np.sum(win ** 2)
    t_cent, cols = [], []
    for i in range(0, len(x) - nperseg + 1, nhop):
        seg = x[i:i + nperseg] - np.mean(x[i:i + nperseg])
        X = np.fft.rfft(seg * win, nfft)
        psd = (np.abs(X) ** 2) / (fs * win_pow)
        psd[1:-1] *= 2.0
        cols.append(psd[f_mask])
        t_cent.append((i + nperseg / 2) / fs)
    return np.array(t_cent), freqs, np.array(cols).T


def sr_peak_power(freqs, P):
    sr_m = (freqs >= SR_LO) & (freqs <= SR_HI)
    idx_sr = np.where(sr_m)[0]
    peak_p = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        col_sr = P[idx_sr, j]
        if not np.isfinite(col_sr).any() or np.all(col_sr <= 0):
            continue
        peak_p[j] = np.max(col_sr)
    return peak_p


def process_subject(args):
    sub_id, df_sub = args
    try:
        raw = load_lemon(sub_id, condition='EC')
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y_mean = X.mean(axis=0)
    t_end = raw.times[-1]

    # Full-recording sliding Welch (for SR peak tracker)
    t_psd, freqs, P = sliding_welch(y_mean, fs)
    sr_p = sr_peak_power(freqs, P)
    baseline_sr = np.nanmedian(sr_p)
    logboost = np.log10(sr_p + 1e-20) - np.log10(baseline_sr + 1e-20)

    # Full-recording narrowband envelope z
    yb = bandpass(y_mean, fs, F0_FIXED - HALF_BW, F0_FIXED + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    env_med = np.nanmedian(env); env_mad = np.nanmedian(np.abs(env - env_med)) * 1.4826
    env_z = (env - env_med) / (env_mad + 1e-9)
    t_env = np.arange(len(env_z)) / fs

    # Per-event
    buckets = {q: {'env_net': [], 'env_sr': [],
                   'boost_net': [], 'boost_sr': []}
                for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    shift_rows = []
    for _, ev in df_sub.iterrows():
        t0_net = float(ev['t0_net'])
        q = ev['rho_q']
        # Find SR peak in [t0_net - 5, t0_net + 5]
        in_win = (t_psd >= t0_net - SR_SEARCH_HALF) & (t_psd <= t0_net + SR_SEARCH_HALF)
        if in_win.sum() == 0 or not np.any(np.isfinite(sr_p[in_win])):
            continue
        t_psd_win = t_psd[in_win]
        sr_win = sr_p[in_win]
        idx_max = np.nanargmax(sr_win)
        t0_sr = float(t_psd_win[idx_max])
        shift_rows.append({
            'subject_id': sub_id,
            't0_net': t0_net, 't0_sr': t0_sr,
            't_shift_s': t0_sr - t0_net,
            'rho_q': q,
        })

        # Envelope traj (100 ms step)
        rel_env = t_env - t0_net
        rel_env_sr = t_env - t0_sr
        # Interpolate to TGRID (sub-100ms resolution → sample)
        mask_net = (rel_env >= TGRID[0] - 1) & (rel_env <= TGRID[-1] + 1)
        mask_sr  = (rel_env_sr >= TGRID[0] - 1) & (rel_env_sr <= TGRID[-1] + 1)
        if mask_net.sum() > 0 and mask_sr.sum() > 0:
            env_net_i = np.interp(TGRID, rel_env[mask_net], env_z[mask_net],
                                  left=np.nan, right=np.nan)
            env_sr_i  = np.interp(TGRID, rel_env_sr[mask_sr], env_z[mask_sr],
                                  left=np.nan, right=np.nan)
            buckets[q]['env_net'].append(env_net_i)
            buckets[q]['env_sr'].append(env_sr_i)

        # SR-boost traj (1s step)
        rel_psd = t_psd - t0_net
        rel_psd_sr = t_psd - t0_sr
        mask_net_p = (rel_psd >= TGRID_PSD[0] - 1) & (rel_psd <= TGRID_PSD[-1] + 1)
        mask_sr_p  = (rel_psd_sr >= TGRID_PSD[0] - 1) & (rel_psd_sr <= TGRID_PSD[-1] + 1)
        if mask_net_p.sum() > 0 and mask_sr_p.sum() > 0:
            bst_net = np.interp(TGRID_PSD, rel_psd[mask_net_p], logboost[mask_net_p],
                                 left=np.nan, right=np.nan)
            bst_sr  = np.interp(TGRID_PSD, rel_psd_sr[mask_sr_p], logboost[mask_sr_p],
                                 left=np.nan, right=np.nan)
            buckets[q]['boost_net'].append(bst_net)
            buckets[q]['boost_sr'].append(bst_sr)

    out = {'subject_id': sub_id, 'shifts': shift_rows}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        for key in ['env_net', 'env_sr', 'boost_net', 'boost_sr']:
            arr = buckets[q][key]
            out[f'{q}_{key}'] = np.nanmean(np.array(arr), axis=0) if arr else None
            out[f'{q}_{key}_n'] = len(arr)
    return out


def bootstrap_ci(mat, n_boot=1000, seed=0):
    mat = np.array(mat)
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    if len(mat) < 2:
        return (np.nanmean(mat, axis=0),
                np.full(mat.shape[1], np.nan),
                np.full(mat.shape[1], np.nan))
    rng = np.random.default_rng(seed)
    grand = np.nanmean(mat, axis=0)
    boots = np.zeros((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, len(mat), size=len(mat))
        boots[b] = np.nanmean(mat[idx], axis=0)
    return grand, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def nadir_rebound(trace, tgrid):
    dip_win = (tgrid >= -4) & (tgrid <= +4)
    reb_win = (tgrid >= +0) & (tgrid <= +8)
    if not np.any(np.isfinite(trace[dip_win])):
        return np.nan, np.nan, np.nan, np.nan
    dip_idx = np.nanargmin(trace[dip_win])
    reb_idx = np.nanargmax(trace[reb_win])
    return (float(trace[dip_win][dip_idx]), float(tgrid[dip_win][dip_idx]),
            float(trace[reb_win][reb_idx]), float(tgrid[reb_win][reb_idx]))


def main():
    qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    tasks = [(sid, g) for sid, g in qual.groupby('subject_id')]
    print(f"Subjects: {len(tasks)}  events: {len(qual)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    all_shifts = []
    for r in results:
        all_shifts.extend(r['shifts'])
    shifts = pd.DataFrame(all_shifts)
    shifts.to_csv(os.path.join(OUT_DIR, 'per_event_t0_shift.csv'), index=False)

    print(f"\n=== t0_sr - t0_net distribution (event-level) ===")
    print(f"  median {shifts['t_shift_s'].median():+.2f}s  IQR ["
          f"{shifts['t_shift_s'].quantile(.25):+.2f}, {shifts['t_shift_s'].quantile(.75):+.2f}]")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        s = shifts[shifts['rho_q'] == q]['t_shift_s']
        print(f"  {q}: median {s.median():+.2f}s  IQR [{s.quantile(.25):+.2f}, {s.quantile(.75):+.2f}]")

    def stack(q, key):
        arr = [r[f'{q}_{key}'] for r in results if r[f'{q}_{key}'] is not None]
        return np.array(arr) if arr else np.empty((0,
                len(TGRID) if key.startswith('env') else len(TGRID_PSD)))

    # Compute nadir/rebound for each alignment × quartile
    print(f"\n=== Envelope z nadir/rebound (grand-average, subject-level averaged) ===")
    print(f"{'q':<3}  {'align':<8} nadir        rebound       range")
    for q in ['Q1', 'Q4']:
        for align in ['net', 'sr']:
            mat = stack(q, f'env_{align}')
            grand, _, _ = bootstrap_ci(mat)
            nd, nt, rp, rt = nadir_rebound(grand, TGRID)
            print(f"{q:<3}  t0_{align:<5} {nd:+.2f} @ {nt:+.1f}s   {rp:+.2f} @ {rt:+.1f}s   {rp - nd:.2f}")

    print(f"\n=== SR-band log-boost peak (grand-average) ===")
    for q in ['Q1', 'Q4']:
        for align in ['net', 'sr']:
            mat = stack(q, f'boost_{align}')
            grand, _, _ = bootstrap_ci(mat)
            if np.all(np.isnan(grand)):
                continue
            pk = int(np.nanargmax(grand))
            ratio_pk = 10 ** grand[pk]
            t_pk = TGRID_PSD[pk]
            print(f"  {q} t0_{align}: peak {ratio_pk:.2f}× at t = {t_pk:+.1f}s")

    # Plot 2x2: envelope (t0_net vs t0_sr) × SR boost (same)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Row 0: envelope z
    for col_i, align in enumerate(['net', 'sr']):
        ax = axes[0, col_i]
        for q, color in [('Q1', '#4575b4'), ('Q4', '#d73027')]:
            mat = stack(q, f'env_{align}')
            grand, lo, hi = bootstrap_ci(mat)
            n_sub = len(mat)
            ax.plot(TGRID, grand, color=color, lw=2, label=f'{q} n_sub={n_sub}')
            ax.fill_between(TGRID, lo, hi, color=color, alpha=0.22)
        ax.axvline(0, color='k', ls='--', lw=0.5)
        ax.set_xlabel(f't rel. t0_{align} (s)')
        ax.set_ylabel('envelope z')
        ax.set_title(f"Envelope z · t0_{align} alignment")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Row 1: SR-band log-boost → ratio
    for col_i, align in enumerate(['net', 'sr']):
        ax = axes[1, col_i]
        for q, color in [('Q1', '#4575b4'), ('Q4', '#d73027')]:
            mat = stack(q, f'boost_{align}')
            grand, lo, hi = bootstrap_ci(mat)
            ratio = 10 ** grand
            lo_r = 10 ** lo; hi_r = 10 ** hi
            n_sub = len(mat)
            ax.plot(TGRID_PSD, ratio, color=color, lw=2, label=f'{q} n_sub={n_sub}')
            ax.fill_between(TGRID_PSD, lo_r, hi_r, color=color, alpha=0.22)
        ax.axhline(1.0, color='k', lw=0.6)
        ax.axvline(0, color='k', ls='--', lw=0.5)
        ax.set_xlabel(f't rel. t0_{align} (s)')
        ax.set_ylabel('SR-band peak power / subject baseline (×)')
        ax.set_title(f"SR-band boost · t0_{align} alignment")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('B15 — SR-band-peak re-centered detector: t0_net vs t0_sr alignment',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sr_recentered_detector.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/sr_recentered_detector.png")
    print(f"Saved: {OUT_DIR}/per_event_t0_shift.csv")


if __name__ == '__main__':
    main()
