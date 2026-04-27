#!/usr/bin/env python3
"""
A10-corrected re-run on composite v2 detector.

Per composite event (nadir-aligned via 3-stream find_nadir):
  (1) sliding-FOOOF peak location in 3-15 Hz fit range, restricted to nearest
      peak to F0 within [6, 9.5] Hz — filter-independent
  (2) IF_wide(t): Hilbert IF on 4-12 Hz (10× wider passband than narrow)
  (3) IFt_wide(t): temporal IF dispersion in wide passband
  (4) env amplitude as SNR regressor
  (5) IFt_wide residual after regressing out env amplitude

Envelope A10-corrected found: empirical alpha peak ≈8 Hz (not 7.83), small
~0.15 Hz peri-event downward drift in FOOOF peak. Composite test: does this
pattern reproduce? Does composite select events with less peri-event IF drift
(consistent with §25b/§26/§27 smoother-event framing)?

To manage runtime, FOOOF_STEP_SEC is 1.0 s (vs envelope's 0.5 s) — halves
FOOOF fits per event with minimal information loss.

Cohort-parameterized.

Usage:
    python scripts/sie_if_corrections_composite.py --cohort lemon
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)
from scripts.sie_perionset_triple_average_composite import bandpass, compute_streams
from scripts.sie_perionset_multistream import (
    F0, PRE_SEC, POST_SEC, PAD_SEC, STEP_SEC, WIN_SEC, TGRID,
)
from scripts.sie_if_corrections import (
    WIDE_BAND, FOOOF_FIT_RANGE, FOOOF_PEAK_SEARCH,
    inst_freq, compute_if_wide, compute_ift_wide,
)

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

FOOOF_WIN_SEC = 2.0
FOOOF_STEP_SEC = 1.0  # wider step than envelope's 0.5 for speed
DIP_WINDOW = (-3.0, 0.4)
NADIR_BASELINE = (-5.0, -3.0)

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cohort_config(cohort):
    events = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, events
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, events
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, events
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, events
    if cohort == 'srm':
        return load_srm, {}, events
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, events
    if cohort == 'dortmund':
        return load_dortmund, {}, events
    if cohort == 'chbmp':
        return load_chbmp, {}, events
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, events
    raise ValueError(f"unsupported cohort {cohort!r}")


def find_nadir_3stream(t_rel, env_z, R, P):
    base = (t_rel >= NADIR_BASELINE[0]) & (t_rel < NADIR_BASELINE[1])
    srch = (t_rel >= DIP_WINDOW[0]) & (t_rel <= DIP_WINDOW[1])
    if not base.any() or not srch.any():
        return np.nan

    def lz(x):
        mu = np.nanmean(x[base]); sd = np.nanstd(x[base])
        if not np.isfinite(sd) or sd < 1e-9: sd = 1.0
        return (x - mu) / sd

    s = lz(env_z) + lz(R) + lz(P)
    s_m = np.where(srch, s, np.inf)
    idx = int(np.nanargmin(s_m))
    if not np.isfinite(s_m[idx]):
        return np.nan
    return float(t_rel[idx])


def sliding_fooof_peak_local(y, fs, win_sec=FOOOF_WIN_SEC, step_sec=FOOOF_STEP_SEC):
    # specparam 2.x (formerly FOOOF)
    from specparam import SpectralModel
    nwin = int(round(win_sec * fs))
    nstep = int(round(step_sec * fs))
    centers, peaks = [], []
    for i in range(0, len(y) - nwin + 1, nstep):
        seg = y[i:i+nwin]
        try:
            f_w, Pxx = signal.welch(seg, fs=fs, nperseg=min(nwin, int(fs*1.0)))
            fm = SpectralModel(peak_width_limits=(0.5, 3.0), max_n_peaks=6,
                                peak_threshold=0.01, min_peak_height=0.01,
                                verbose=False)
            fm.fit(f_w, Pxx, FOOOF_FIT_RANGE)
            gp = fm.results.get_params('peak')
            if gp is None or (isinstance(gp, np.ndarray) and gp.size == 0):
                peaks.append(np.nan)
            else:
                gp = np.atleast_2d(np.asarray(gp))
                if gp.ndim == 1:
                    gp = gp.reshape(1, -1)
                cands = gp[(gp[:, 0] >= FOOOF_PEAK_SEARCH[0]) &
                             (gp[:, 0] <= FOOOF_PEAK_SEARCH[1])]
                if len(cands) == 0:
                    peaks.append(np.nan)
                else:
                    peaks.append(float(cands[np.argmax(cands[:, 1]), 0]))
        except Exception:
            peaks.append(np.nan)
        centers.append((i + nwin/2) / fs)
    return np.array(centers), np.array(peaks)


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


def process_subject(args):
    sub_id, events_path = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) == 0:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
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

        # 3-stream nadir
        try:
            (t_env_abs, env_z), (t_c, R), (_, P) = compute_streams(X_seg, fs)
        except Exception:
            continue
        rel_win = t_c - PAD_SEC - PRE_SEC
        rel_env = t_env_abs - PAD_SEC - PRE_SEC
        env_on_win = np.interp(rel_win, rel_env, env_z, left=np.nan, right=np.nan)
        nadir = find_nadir_3stream(rel_win, env_on_win, R, P)
        if not np.isfinite(nadir):
            continue
        rel_n = rel_win - nadir

        y = X_seg.mean(axis=0)
        # IFt_wide & IF_wide
        try:
            t_ift, ift_wide = compute_ift_wide(X_seg, fs)
            rel_ift = t_ift - PAD_SEC - PRE_SEC - nadir
            IF_w_arr = compute_if_wide(y, fs)
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

        # Sliding FOOOF peak
        try:
            t_f, peaks = sliding_fooof_peak_local(y, fs)
            rel_f = t_f - PAD_SEC - PRE_SEC - nadir
        except Exception:
            continue

        fooof_rows.append(np.interp(TGRID, rel_f, peaks, left=np.nan, right=np.nan))
        if_wide_rows.append(np.interp(TGRID, rel_IF, np.array(IF_w_win),
                                         left=np.nan, right=np.nan))
        ift_wide_rows.append(np.interp(TGRID, rel_ift, ift_wide,
                                          left=np.nan, right=np.nan))
        env_rows.append(np.interp(TGRID, rel_n, env_on_win, left=np.nan, right=np.nan))

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


def bootstrap_ci(mat, n_boot=500, seed=0):
    mat = np.array(mat)
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    if len(mat) < 2:
        return np.nanmean(mat, axis=0), np.full(mat.shape[1], np.nan), np.full(mat.shape[1], np.nan)
    rng = np.random.default_rng(seed)
    grand = np.nanmean(mat, axis=0)
    boots = np.zeros((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, len(mat), size=len(mat))
        boots[b] = np.nanmean(mat[idx], axis=0)
    return grand, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def regress_out(y, x):
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 10:
        return y * np.nan, np.nan
    b, a = np.polyfit(x[mask], y[mask], 1)
    resid = y - (b * x + a)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    ss_res = np.sum(resid[mask] ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return resid, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'if_corrections', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} composite · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}   (FOOOF_STEP_SEC = {FOOOF_STEP_SEC}s)")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    n_events = sum(r['n_events'] for r in results)
    print(f"Successful: {len(results)}  events: {n_events}")

    fooof_arr = np.array([r['fooof_peak'] for r in results])
    if_wide_arr = np.array([r['IF_wide']  for r in results])
    ift_wide_arr = np.array([r['IFt_wide'] for r in results])
    env_arr = np.array([r['env']     for r in results])

    fooof_m, fooof_lo, fooof_hi = bootstrap_ci(fooof_arr)
    ifw_m, ifw_lo, ifw_hi       = bootstrap_ci(if_wide_arr)
    iftw_m, iftw_lo, iftw_hi    = bootstrap_ci(ift_wide_arr)
    env_m, env_lo, env_hi       = bootstrap_ci(env_arr)

    resid_rows = []
    r2_vals = []
    for r in results:
        resid, r2 = regress_out(r['IFt_wide'], r['env'])
        resid_rows.append(resid)
        if np.isfinite(r2):
            r2_vals.append(r2)
    resid_arr = np.array(resid_rows)
    res_m, res_lo, res_hi = bootstrap_ci(resid_arr)

    i0 = int(np.argmin(np.abs(TGRID)))
    # Find pre-nadir and post-peak FOOOF windows
    pre_mask = (TGRID >= -6) & (TGRID <= -3)
    peri_mask = (TGRID >= -1) & (TGRID <= +2.5)

    print(f"\n=== {args.cohort} composite · A10-corrected ===")
    for name, m in [('FOOOF peak (Hz)', fooof_m), ('IF wide (Hz)', ifw_m),
                     ('IFt wide (Hz)', iftw_m), ('env', env_m),
                     ('IFt resid', res_m)]:
        i_peak = int(np.nanargmax(m)); i_trough = int(np.nanargmin(m))
        print(f"  {name:20s} @nadir {m[i0]:+.4f}   "
              f"peak {m[i_peak]:+.4f} at {TGRID[i_peak]:+.2f}s   "
              f"trough {m[i_trough]:+.4f} at {TGRID[i_trough]:+.2f}s")

    # Peri-event FOOOF drift: pre-nadir [-6,-3] vs peri-nadir [-1,+2.5]
    pre_f = np.nanmean(fooof_m[pre_mask])
    peri_f = np.nanmean(fooof_m[peri_mask])
    print(f"\n  FOOOF peak pre-nadir [-6,-3]: {pre_f:.3f} Hz")
    print(f"  FOOOF peak peri-nadir [-1,+2.5]: {peri_f:.3f} Hz")
    print(f"  Peri-event FOOOF drift (peri − pre): {peri_f - pre_f:+.3f} Hz")
    print(f"  (envelope A10-corrected: ~−0.15 Hz peri-event downward drift)")

    print(f"\n  Median per-subject R² of IFt~env: {np.median(r2_vals):.3f}")
    print(f"    (if high, IFt is mostly amplitude-driven)")

    pd.DataFrame({
        't_rel': TGRID,
        'fooof_peak_mean': fooof_m, 'fooof_peak_lo': fooof_lo, 'fooof_peak_hi': fooof_hi,
        'IF_wide_mean': ifw_m, 'IF_wide_lo': ifw_lo, 'IF_wide_hi': ifw_hi,
        'IFt_wide_mean': iftw_m, 'IFt_wide_lo': iftw_lo, 'IFt_wide_hi': iftw_hi,
        'IFt_resid_mean': res_m, 'IFt_resid_lo': res_lo, 'IFt_resid_hi': res_hi,
        'env_mean': env_m,
    }).to_csv(os.path.join(out_dir, 'if_corrections.csv'), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    ax = axes[0, 0]
    ax.fill_between(TGRID, fooof_lo, fooof_hi, color='navy', alpha=0.25)
    ax.plot(TGRID, fooof_m, color='navy', lw=2)
    ax.axhline(F0, color='red', ls='--', lw=0.8, label=f'F₀ = {F0} Hz')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.axvspan(-6, -3, color='blue', alpha=0.08, label='pre [-6,-3]s')
    ax.axvspan(-1, +2.5, color='orange', alpha=0.08, label='peri [-1,+2.5]s')
    ax.set_ylim(6.0, 9.5)
    ax.set_ylabel('FOOOF peak frequency (Hz)')
    ax.set_title(f'Sliding FOOOF peak · {args.cohort} composite')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.fill_between(TGRID, ifw_lo, ifw_hi, color='darkgreen', alpha=0.25)
    ax.plot(TGRID, ifw_m, color='darkgreen', lw=2)
    ax.axhline(F0, color='red', ls='--', lw=0.8)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_ylim(4.0, 12.0)
    ax.set_ylabel('mean IF, wide passband 4–12 Hz (Hz)')
    ax.set_title('Hilbert IF, wide passband')

    ax = axes[1, 0]
    ax.fill_between(TGRID, iftw_lo, iftw_hi, color='brown', alpha=0.25)
    ax.plot(TGRID, iftw_m, color='brown', lw=2)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('IFt wide (Hz)')
    ax.set_xlabel('time relative to nadir (s)')
    ax.set_title('IFt wide (temporal IF dispersion)')

    ax = axes[1, 1]
    ax.fill_between(TGRID, res_lo, res_hi, color='slategray', alpha=0.25)
    ax.plot(TGRID, res_m, color='slategray', lw=2)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('IFt resid (env-regressed)')
    ax.set_xlabel('time relative to nadir (s)')
    ax.set_title(f'IFt wide after regressing out env\n'
                 f'median per-subj R²(IFt~env)={np.median(r2_vals):.3f}')

    plt.suptitle(f'A10-corrected · filter-independent IF analyses · {args.cohort} composite v2',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'if_corrections.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/if_corrections.png")


if __name__ == '__main__':
    main()
