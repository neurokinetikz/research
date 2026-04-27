#!/usr/bin/env python3
"""
B15 + B16 combined on composite v2 detector.

Computes per-composite-event t0_sr = argmax SR-band [7.0, 8.2] Hz peak power in
[t0_net − 5, t0_net + 5] s, then computes timing_distance = |t_shift − 1.2|
(canonical Q4 shift from envelope B14/B15), stratifies events into timing
quartiles, and tests:

  (a) Spearman correlation between timing_distance and composite template_ρ
      (envelope finding: ρ = −0.54, p = 10⁻⁷⁰)
  (b) Contingency table T_q × Q_q
  (c) Peri-event env z + SR-band log-boost stratified by timing quartile

Envelope B16 found timing_q and rho_q are ~54% correlated; T1 captures most of
Q4 canonical events. Under composite we check whether the same relationship
holds.

Cohort-parameterized.

Usage:
    python scripts/sie_timing_consistency_axis_composite.py --cohort lemon
    python scripts/sie_timing_consistency_axis_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
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
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)
from scripts.sie_perionset_triple_average_composite import bandpass

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
CANONICAL_SHIFT = 1.2
T0_SR_WINDOW = 5.0  # ±5 s around t0_net

TGRID = np.arange(-15.0, 15.0 + 0.05, 0.1)
TGRID_PSD = np.arange(-15.0, 15.0 + 0.5, 1.0)

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    qual = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events, qual
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events, qual
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events, qual
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events, qual
    if cohort == 'srm':
        return load_srm, {}, events, qual
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events, qual
    if cohort == 'dortmund':
        return load_dortmund, {}, events, qual
    if cohort == 'chbmp':
        return load_chbmp, {}, events, qual
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events, qual
    raise ValueError(f"unsupported cohort {cohort!r}")


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


_LOADER = None
_LOADER_KW = None


def _init_worker(loader_name, loader_kw):
    global _LOADER, _LOADER_KW
    _LOADER_KW = loader_kw
    _LOADER = {
        'load_lemon': load_lemon,
        'load_tdbrain': load_tdbrain,
        'load_srm': load_srm,
        'load_dortmund': load_dortmund,
        'load_chbmp': load_chbmp,
        'load_hbn_by_subject': load_hbn_by_subject,
    }[loader_name]


def compute_t0_sr_for_subject(args):
    """Compute t0_sr per event for a subject.
    Returns per-event rows with subject_id, t0_net, t0_sr, t_shift_s, and
    pre-computed per-subject env z + logboost streams for peri-event analysis.
    """
    sub_id, events = args
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y_mean = X.mean(axis=0)

    # Full-recording sliding Welch → SR peak power per window
    t_psd, freqs, P = sliding_welch(y_mean, fs)
    sr_p = sr_peak_power(freqs, P)
    baseline_sr = np.nanmedian(sr_p)
    logboost = np.log10(sr_p + 1e-20) - np.log10(baseline_sr + 1e-20)

    # Envelope z at F0
    yb = bandpass(y_mean, fs, F0_FIXED - HALF_BW, F0_FIXED + HALF_BW)
    env = np.abs(signal.hilbert(yb))
    env_med = np.nanmedian(env)
    env_mad = np.nanmedian(np.abs(env - env_med)) * 1.4826
    env_z = (env - env_med) / (env_mad + 1e-9)
    t_env = np.arange(len(env_z)) / fs

    # Per-event t0_sr: argmax SR peak power in [t0_net - 5, t0_net + 5] s
    t0_rows = []
    for _, ev in events.iterrows():
        t0 = float(ev['t0_net'])
        win_mask = (t_psd >= t0 - T0_SR_WINDOW) & (t_psd <= t0 + T0_SR_WINDOW)
        if win_mask.sum() == 0 or not np.isfinite(sr_p[win_mask]).any():
            continue
        win_t = t_psd[win_mask]
        win_sr = sr_p[win_mask]
        good = np.isfinite(win_sr)
        if not good.any():
            continue
        idx = int(np.nanargmax(win_sr))
        t0_sr = float(win_t[idx])
        t0_rows.append({
            'subject_id': sub_id,
            't0_net': t0,
            't0_sr': t0_sr,
            't_shift_s': t0_sr - t0,
        })

    return {
        'subject_id': sub_id,
        't0_rows': t0_rows,
        't_psd': t_psd,
        'logboost': logboost,
        't_env': t_env,
        'env_z': env_z,
    }


def peri_event_streams(subject_result, events_with_tq):
    """For a subject's streams, compute peri-event env_z and logboost
    interpolated onto TGRID and TGRID_PSD, per timing quartile."""
    t_env = subject_result['t_env']
    env_z = subject_result['env_z']
    t_psd = subject_result['t_psd']
    logboost = subject_result['logboost']
    buckets = {q: {'env': [], 'boost': []} for q in ['T1', 'T2', 'T3', 'T4']}
    for _, ev in events_with_tq.iterrows():
        t0 = float(ev['t0_net'])
        tq = ev['timing_q']
        if tq not in buckets:
            continue
        rel_env = t_env - t0
        rel_psd = t_psd - t0
        m_e = (rel_env >= TGRID[0] - 1) & (rel_env <= TGRID[-1] + 1)
        m_p = (rel_psd >= TGRID_PSD[0] - 1) & (rel_psd <= TGRID_PSD[-1] + 1)
        if m_e.sum() > 0 and m_p.sum() > 0:
            env_i = np.interp(TGRID, rel_env[m_e], env_z[m_e],
                                left=np.nan, right=np.nan)
            bst_i = np.interp(TGRID_PSD, rel_psd[m_p], logboost[m_p],
                                left=np.nan, right=np.nan)
            buckets[tq]['env'].append(env_i)
            buckets[tq]['boost'].append(bst_i)
    out = {}
    for q in ['T1', 'T2', 'T3', 'T4']:
        for key in ['env', 'boost']:
            arr = buckets[q][key]
            out[f'{q}_{key}'] = np.nanmean(np.array(arr), axis=0) if arr else None
            out[f'{q}_{key}_n'] = len(arr)
    return out


def bootstrap_ci(mat, n_boot=500, seed=0):
    mat = np.array(mat)
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    if len(mat) < 2:
        return (np.nanmean(mat, axis=0), np.full(mat.shape[1], np.nan),
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'psd_timelapse', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    # Events with template_rho quartile
    qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print(f"Cohort: {args.cohort} composite · events: {len(qual)}  subjects: {qual['subject_id'].nunique()}")

    tasks = [(sid, g) for sid, g in qual.groupby('subject_id')]
    print(f"Processing {len(tasks)} subjects · workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        subject_results = pool.map(compute_t0_sr_for_subject, tasks)
    subject_results = [r for r in subject_results if r is not None]
    print(f"Successful: {len(subject_results)}")

    # Aggregate per-event t0_sr rows
    all_rows = []
    for r in subject_results:
        all_rows.extend(r['t0_rows'])
    shifts = pd.DataFrame(all_rows)
    shifts['timing_distance'] = np.abs(shifts['t_shift_s'] - CANONICAL_SHIFT)
    shifts['timing_q'] = pd.qcut(shifts['timing_distance'], 4,
                                   labels=['T1', 'T2', 'T3', 'T4'])
    shifts.to_csv(os.path.join(out_dir, 'per_event_t0_shift.csv'), index=False)
    print(f"\nSaved t0_shift CSV: {out_dir}/per_event_t0_shift.csv ({len(shifts)} events)")

    # Merge with template_rho
    merged = shifts.merge(qual[['subject_id', 't0_net', 'template_rho', 'rho_q']],
                           on=['subject_id', 't0_net'], how='left')
    rho_ok = merged.dropna(subset=['template_rho'])

    rho, p = spearmanr(rho_ok['timing_distance'], rho_ok['template_rho'])
    print(f"\n=== Spearman corr(timing_distance, template_rho) ===")
    print(f"  ρ = {rho:+.3f}  p = {p:.3g}  n = {len(rho_ok)}")
    print(f"  (envelope B16: ρ = −0.54, p = 10⁻⁷⁰)")

    print(f"\n=== Contingency: timing_q × rho_q (row %) ===")
    ct = pd.crosstab(rho_ok['timing_q'], rho_ok['rho_q'])
    pct = (ct.T / ct.sum(axis=1)).T.round(3) * 100
    print(pct.to_string())

    # Timing-quartile stats
    print(f"\n=== Timing quartile distributions ===")
    for tq in ['T1', 'T2', 'T3', 'T4']:
        sub = shifts[shifts['timing_q'] == tq]
        print(f"  {tq}: n={len(sub)}  "
              f"timing_dist median {sub['timing_distance'].median():.2f}s  "
              f"shift median {sub['t_shift_s'].median():+.2f}s")

    # Per-subject peri-event
    per_sub = {}
    sr_results = {r['subject_id']: r for r in subject_results}
    for sid, g in merged.dropna(subset=['timing_q']).groupby('subject_id'):
        if sid not in sr_results:
            continue
        out = peri_event_streams(sr_results[sid], g)
        per_sub[sid] = out

    def stack(tq, key):
        arr = [per_sub[s][f'{tq}_{key}'] for s in per_sub if per_sub[s][f'{tq}_{key}'] is not None]
        return (np.array(arr) if arr else
                np.empty((0, len(TGRID) if key == 'env' else len(TGRID_PSD))))

    print(f"\n=== {args.cohort} composite · envelope z nadir/rebound by timing quartile ===")
    print(f"{'tq':<3}  nadir          rebound         range")
    for tq in ['T1', 'T2', 'T3', 'T4']:
        mat = stack(tq, 'env')
        grand, _, _ = bootstrap_ci(mat)
        nd, nt, rp, rt = nadir_rebound(grand)
        print(f"{tq:<3}  {nd:+.2f} @ {nt:+.1f}s   {rp:+.2f} @ {rt:+.1f}s   {rp - nd:.2f}")

    print(f"\n=== {args.cohort} composite · SR-band log-boost peak by timing quartile ===")
    for tq in ['T1', 'T2', 'T3', 'T4']:
        mat = stack(tq, 'boost')
        if len(mat) == 0:
            continue
        grand, _, _ = bootstrap_ci(mat)
        if np.all(np.isnan(grand)):
            continue
        pk = int(np.nanargmax(grand))
        ratio_pk = 10 ** grand[pk]
        print(f"  {tq}: peak {ratio_pk:.2f}× at t = {TGRID_PSD[pk]:+.1f}s")

    # Save figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    for tq, color in [('T1', '#1a9641'), ('T4', '#d7191c')]:
        mat = stack(tq, 'env')
        grand, lo, hi = bootstrap_ci(mat)
        n_sub = len(mat); n_ev = sum(per_sub[s][f'{tq}_env_n'] for s in per_sub)
        ax.plot(TGRID, grand, color=color, lw=2, label=f'{tq} n_sub={n_sub} n_ev={n_ev}')
        ax.fill_between(TGRID, lo, hi, color=color, alpha=0.22)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('t rel. t0_net (s)'); ax.set_ylabel('envelope z')
    ax.set_title(f'Envelope z · T1 vs T4 · {args.cohort} composite')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for tq, color in [('T1', '#1a9641'), ('T4', '#d7191c')]:
        mat = stack(tq, 'boost')
        if len(mat) == 0:
            continue
        grand, lo, hi = bootstrap_ci(mat)
        ax.plot(TGRID_PSD, 10 ** grand, color=color, lw=2, label=tq)
        ax.fill_between(TGRID_PSD, 10 ** lo, 10 ** hi, color=color, alpha=0.22)
    ax.axhline(1.0, color='k', lw=0.6)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('t rel. t0_net (s)'); ax.set_ylabel('SR-band peak power (×)')
    ax.set_title(f'SR-band boost · T1 vs T4 · {args.cohort} composite')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(f'B16 · timing-consistency axis · {args.cohort} composite v2 · ρ(t_dist, template_ρ)={rho:+.2f}',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'timing_consistency_axis.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/timing_consistency_axis.png")


if __name__ == '__main__':
    main()
