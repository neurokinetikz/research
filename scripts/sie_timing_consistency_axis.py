#!/usr/bin/env python3
"""
B16 — Test timing-consistency axis as an alternative quality score.

New axis:  timing_distance = |t0_sr - t0_net - 1.2|
  where 1.2 s is the canonical Q4 shift (B15).

Stratify events into quartiles of timing_distance (T1 = tightest, T4 = loosest),
then re-run peri-event grand averages for envelope z and SR-band boost aligned
to t0_net. Compare T1 vs T4 to the template_rho Q1 vs Q4 results from B7/B14.

Hypothesis: if template_rho is primarily a timing-consistency axis (B15), then:
  (a) timing_distance should correlate strongly negatively with template_rho
  (b) T1 (tight timing) peri-event signatures should be as sharp or sharper
      than Q4 (high rho), and
  (c) T1 and Q4 should largely contain the same events

Outputs:
  - per_event_timing_stratification.csv
  - figure: peri-event envelope z and SR-band boost, T1 vs T4 vs template_rho Q1/Q4
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon
from scripts.sie_perionset_triple_average import bandpass

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
T0_SHIFT_CSV = os.path.join(OUT_DIR, 'per_event_t0_shift.csv')
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
CANONICAL_SHIFT = 1.2   # from B15 Q4 median

TGRID = np.arange(-15.0, 15.0 + 0.05, 0.1)
TGRID_PSD = np.arange(-15.0, 15.0 + 0.5, 1.0)


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

    # Full-recording Welch + SR peak + logboost
    t_psd, freqs, P = sliding_welch(y_mean, fs)
    sr_p = sr_peak_power(freqs, P)
    baseline_sr = np.nanmedian(sr_p)
    logboost = np.log10(sr_p + 1e-20) - np.log10(baseline_sr + 1e-20)

    # Envelope z
    yb = bandpass(y_mean, fs, F0_FIXED - HALF_BW, F0_FIXED + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    env_med = np.nanmedian(env); env_mad = np.nanmedian(np.abs(env - env_med)) * 1.4826
    env_z = (env - env_med) / (env_mad + 1e-9)
    t_env = np.arange(len(env_z)) / fs

    buckets_t = {q: {'env': [], 'boost': []} for q in ['T1', 'T2', 'T3', 'T4']}
    for _, ev in df_sub.iterrows():
        t0 = float(ev['t0_net'])
        tq = ev['timing_q']
        rel_env = t_env - t0
        rel_psd = t_psd - t0
        m_e = (rel_env >= TGRID[0] - 1) & (rel_env <= TGRID[-1] + 1)
        m_p = (rel_psd >= TGRID_PSD[0] - 1) & (rel_psd <= TGRID_PSD[-1] + 1)
        if m_e.sum() > 0 and m_p.sum() > 0:
            env_i = np.interp(TGRID, rel_env[m_e], env_z[m_e],
                              left=np.nan, right=np.nan)
            bst_i = np.interp(TGRID_PSD, rel_psd[m_p], logboost[m_p],
                              left=np.nan, right=np.nan)
            buckets_t[tq]['env'].append(env_i)
            buckets_t[tq]['boost'].append(bst_i)

    out = {'subject_id': sub_id}
    for q in ['T1', 'T2', 'T3', 'T4']:
        for key in ['env', 'boost']:
            arr = buckets_t[q][key]
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


def nadir_rebound(trace):
    dip_win = (TGRID >= -4) & (TGRID <= +4)
    reb_win = (TGRID >= 0) & (TGRID <= +8)
    if not np.any(np.isfinite(trace[dip_win])):
        return np.nan, np.nan, np.nan, np.nan
    di = np.nanargmin(trace[dip_win])
    ri = np.nanargmax(trace[reb_win])
    return (float(trace[dip_win][di]), float(TGRID[dip_win][di]),
            float(trace[reb_win][ri]), float(TGRID[reb_win][ri]))


def main():
    shifts = pd.read_csv(T0_SHIFT_CSV)
    shifts['timing_distance'] = np.abs(shifts['t_shift_s'] - CANONICAL_SHIFT)
    # quartile: T1 = tightest timing (small distance), T4 = loosest
    shifts['timing_q'] = pd.qcut(shifts['timing_distance'], 4,
                                   labels=['T1', 'T2', 'T3', 'T4'])
    # Merge template_rho for correlation
    qual = pd.read_csv(QUALITY_CSV)[['subject_id', 't0_net', 'template_rho']]
    merged = shifts.merge(qual, on=['subject_id', 't0_net'], how='left')
    rho_ok = merged.dropna(subset=['template_rho'])
    rho, p = spearmanr(rho_ok['timing_distance'], rho_ok['template_rho'])
    print(f"Spearman corr(timing_distance, template_rho): rho = {rho:.3f}  p = {p:.3g}")

    # Overlap of T1 with Q4(template_rho) — merged already has rho_q from shifts CSV
    # but rebuild it here from scratch so the quartiling uses the current event set.
    merged = merged.drop(columns=['rho_q'], errors='ignore')
    with_rho = merged.dropna(subset=['template_rho']).copy()
    with_rho['rho_q'] = pd.qcut(with_rho['template_rho'], 4,
                                 labels=['Q1', 'Q2', 'Q3', 'Q4'])
    if len(with_rho) > 0:
        ct = pd.crosstab(with_rho['timing_q'], with_rho['rho_q'])
        print(f"\nContingency: timing_q × rho_q (row-normalized %)")
        print((ct.T / ct.sum(axis=1)).T.round(2) * 100)
    overlap = merged

    print(f"\n=== Timing quartile distributions ===")
    for tq in ['T1', 'T2', 'T3', 'T4']:
        sub = shifts[shifts['timing_q'] == tq]
        print(f"  {tq}: n={len(sub)}  timing_dist median {sub['timing_distance'].median():.2f}s  "
              f"shift median {sub['t_shift_s'].median():+.2f}s")

    # Run peri-event per subject
    tasks = [(sid, g) for sid, g in shifts.groupby('subject_id')]
    print(f"\nProcessing {len(tasks)} subjects...")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    def stack(q, key):
        arr = [r[f'{q}_{key}'] for r in results if r[f'{q}_{key}'] is not None]
        return (np.array(arr) if arr else
                np.empty((0, len(TGRID) if key == 'env' else len(TGRID_PSD))))

    print(f"\n=== Envelope z nadir/rebound by timing quartile ===")
    print(f"{'q':<3}  nadir          rebound         range")
    for tq in ['T1', 'T2', 'T3', 'T4']:
        mat = stack(tq, 'env')
        grand, _, _ = bootstrap_ci(mat)
        nd, nt, rp, rt = nadir_rebound(grand)
        print(f"{tq:<3}  {nd:+.2f} @ {nt:+.1f}s   {rp:+.2f} @ {rt:+.1f}s   {rp - nd:.2f}")

    print(f"\n=== SR-band log-boost peak by timing quartile ===")
    for tq in ['T1', 'T2', 'T3', 'T4']:
        mat = stack(tq, 'boost')
        grand, _, _ = bootstrap_ci(mat)
        if len(grand) == 0 or np.all(np.isnan(grand)):
            continue
        pk = int(np.nanargmax(grand))
        ratio_pk = 10 ** grand[pk]
        print(f"  {tq}: peak {ratio_pk:.2f}× at t = {TGRID_PSD[pk]:+.1f}s")

    # Save merged event-level CSV
    overlap.to_csv(os.path.join(OUT_DIR, 'per_event_timing_stratification.csv'),
                    index=False)

    # Plot: T1 vs T4 envelope & boost
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for tq, color in [('T1', '#1a9641'), ('T4', '#d7191c')]:
        mat = stack(tq, 'env')
        grand, lo, hi = bootstrap_ci(mat)
        n_sub = len(mat); n_ev = sum(r[f'{tq}_env_n'] for r in results)
        ax.plot(TGRID, grand, color=color, lw=2,
                label=f'{tq} n_sub={n_sub} n_ev={n_ev}')
        ax.fill_between(TGRID, lo, hi, color=color, alpha=0.22)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('t rel. t0_net (s)')
    ax.set_ylabel('envelope z')
    ax.set_title('Envelope z · tightest (T1) vs loosest (T4) timing')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for tq, color in [('T1', '#1a9641'), ('T4', '#d7191c')]:
        mat = stack(tq, 'boost')
        grand, lo, hi = bootstrap_ci(mat)
        ratio = 10 ** grand
        lo_r = 10 ** lo; hi_r = 10 ** hi
        n_sub = len(mat)
        ax.plot(TGRID_PSD, ratio, color=color, lw=2, label=f'{tq} n_sub={n_sub}')
        ax.fill_between(TGRID_PSD, lo_r, hi_r, color=color, alpha=0.22)
    ax.axhline(1.0, color='k', lw=0.6)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('t rel. t0_net (s)')
    ax.set_ylabel('SR-band peak power (×)')
    ax.set_title('SR-band boost · T1 vs T4')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(f'B16 — Timing-consistency axis (|t_shift − {CANONICAL_SHIFT}s|)',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'timing_consistency_axis.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/timing_consistency_axis.png")


if __name__ == '__main__':
    main()
