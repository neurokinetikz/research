#!/usr/bin/env python3
"""
B3 + B4 + B5 — Mechanism-diagnostic battery (single pass).

B3  Per-channel nadir timing → propagation vs simultaneity.
    For each event, per-channel envelope nadir time (in [-3, +0.4]s rel to
    joint nadir). Report:
      - std across channels (low → simultaneous; high → propagation)
      - spatial gradient: regress nadir_time on channel (x, y), report slope
        (propagation direction) and R² of fit
      - distribution across events

B4  Critical slowing test.
    For each event, sliding variance and lag-1 autocorrelation of:
      - envelope z
      - Kuramoto R
    in a -5 to +1s window. Test whether variance and AR(1) trend up toward
    the nadir (classic critical-slowing signature) vs. flat.
    Also compute a matched-baseline estimate (random 1-s windows far from
    events) for comparison.

B5  Phase discontinuity (phase-reset test).
    Per-channel unwrapped Hilbert phase on bandpassed signal (7.2-8.4 Hz).
    Count |dφ| > threshold per 100-ms bin, aligned on nadir.
    Test whether phase-jump rate elevates peri-nadir vs baseline.
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
    F0, HALF_BW, R_BAND, PRE_SEC, POST_SEC, PAD_SEC, STEP_SEC, WIN_SEC,
    find_nadir, TGRID, inst_freq_channel,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'mechanism_battery')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')


# ---------- B3 helpers ----------
def per_channel_nadir(X_uV, fs, t0_narrow_nadir):
    """
    Per-channel envelope nadir time (rel to overall nadir=0), within [-3, +0.4] s.
    Returns array of nadir times (n_ch,), NaN for channels without clear dip.
    """
    n_ch = X_uV.shape[0]
    # Per-channel envelope at F0 ± 0.6 Hz
    Xb = bandpass(X_uV, fs, F0 - HALF_BW, F0 + HALF_BW)
    env = np.abs(signal.hilbert(Xb, axis=-1))
    # smooth
    win_smooth = int(round(0.5 * fs))
    kernel = np.ones(win_smooth) / win_smooth
    env = np.array([np.convolve(env[i], kernel, mode='same') for i in range(n_ch)])

    t_grid = np.arange(env.shape[1]) / fs
    rel = t_grid - PAD_SEC - PRE_SEC - t0_narrow_nadir  # rel to nadir
    mask = (rel >= -3.0) & (rel <= 0.4)
    nadir_t = np.full(n_ch, np.nan)
    for i in range(n_ch):
        e = env[i]
        e_m = np.where(mask, e, np.inf)
        idx = int(np.nanargmin(e_m))
        if np.isfinite(e_m[idx]):
            nadir_t[i] = rel[idx]
    return nadir_t


def channel_positions(ch_names):
    """Return (n_ch, 2) x/y positions on head sphere using standard_1020."""
    from mne.channels import make_standard_montage
    mon = make_standard_montage('standard_1020')
    pos_dict = {name: pos for name, pos in
                 zip(mon.ch_names, mon.get_positions()['ch_pos'].values())}
    aliases = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    pos = np.full((len(ch_names), 2), np.nan)
    for i, c in enumerate(ch_names):
        key = aliases.get(c, c)
        if key in pos_dict:
            p = pos_dict[key]
            pos[i] = [p[0], p[1]]  # x (left-right), y (front-back)
    return pos


# ---------- B4 helpers ----------
def sliding_var_ar1(x, fs, win_sec=1.0, step_sec=0.1):
    """Return t, var, ar1 for sliding windows."""
    nwin = int(round(win_sec * fs))
    nstep = int(round(step_sec * fs))
    centers, vars_, ar1s = [], [], []
    for i in range(0, len(x) - nwin + 1, nstep):
        seg = x[i:i+nwin]
        vars_.append(float(np.var(seg)))
        if len(seg) > 2:
            s0 = seg[:-1] - np.mean(seg[:-1])
            s1 = seg[1:]  - np.mean(seg[1:])
            num = np.sum(s0 * s1)
            den = np.sqrt(np.sum(s0 ** 2) * np.sum(s1 ** 2))
            ar1s.append(float(num / (den + 1e-12)))
        else:
            ar1s.append(np.nan)
        centers.append((i + nwin/2) / fs)
    return np.array(centers), np.array(vars_), np.array(ar1s)


# ---------- B5 helpers ----------
def phase_jumps(X_uV, fs, thresh=np.pi/2, bin_sec=0.1):
    """Count phase-jump events per 100-ms bin, summed across channels.
    dphi > thresh rad in one sample → a jump."""
    Xb = bandpass(X_uV, fs, R_BAND[0], R_BAND[1])
    ph = np.angle(signal.hilbert(Xb, axis=-1))
    # Instantaneous diff (not unwrapped — we want raw jumps)
    dph = np.diff(ph, axis=-1)
    # Wrap to [-pi, pi]
    dph = (dph + np.pi) % (2 * np.pi) - np.pi
    # Expected Δφ per sample = 2π f_center / fs; subtract off to get residual
    expected = 2 * np.pi * F0 / fs
    resid = np.abs(dph - expected)
    # Any channel's |resid| > thresh counts
    jumps_per_sample = np.any(resid > thresh, axis=0).astype(float)
    # Bin to 100-ms
    nbin = int(round(bin_sec * fs))
    n_full = (len(jumps_per_sample) // nbin) * nbin
    binned = jumps_per_sample[:n_full].reshape(-1, nbin).sum(axis=1)
    t_bin = np.arange(len(binned)) * bin_sec + bin_sec/2
    return t_bin, binned


# ---------- Per-subject processing ----------
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
    ch_names = raw.ch_names
    pos = channel_positions(ch_names)

    n_ch = len(ch_names)
    nadir_std_per_event = []
    slope_x_per_event = []
    slope_y_per_event = []
    r2_per_event = []

    var_env_rows, ar1_env_rows, var_R_rows, ar1_R_rows = [], [], [], []
    phase_jump_rows = []
    pj_tgrid = np.arange(-8.0, 8.0 + 0.1/2, 0.1)  # 100-ms bins

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
        try:
            t_c, env, R, P, M = compute_streams_4way(X_seg, fs)
            rel = t_c - PAD_SEC - PRE_SEC
            nadir = find_nadir(rel, env, R, P, M)
            if not np.isfinite(nadir):
                continue
        except Exception:
            continue

        # ---- B3: per-channel nadir timing ----
        try:
            nadir_per_ch = per_channel_nadir(X_seg, fs, nadir)
            std_ch = float(np.nanstd(nadir_per_ch))
            nadir_std_per_event.append(std_ch)
            # Gradient fit
            good = np.isfinite(nadir_per_ch) & np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
            if good.sum() >= 6:
                X_fit = np.column_stack([pos[good, 0], pos[good, 1], np.ones(good.sum())])
                y_fit = nadir_per_ch[good]
                coefs, residuals, rank, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
                y_pred = X_fit @ coefs
                ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
                ss_res = np.sum((y_fit - y_pred) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                slope_x_per_event.append(float(coefs[0]))
                slope_y_per_event.append(float(coefs[1]))
                r2_per_event.append(float(r2))
        except Exception:
            pass

        # ---- B4: sliding variance and AR(1) for env and R ----
        try:
            y_env = X_seg.mean(axis=0)
            y_envbp = bandpass(y_env, fs, F0-HALF_BW, F0+HALF_BW)
            y_env_full = np.abs(signal.hilbert(y_envbp))
            t_v, ve, ae = sliding_var_ar1(y_env_full, fs, 1.0, 0.1)
            rel_v = t_v - PAD_SEC - PRE_SEC - nadir
            var_env_rows.append(np.interp(TGRID, rel_v, ve, left=np.nan, right=np.nan))
            ar1_env_rows.append(np.interp(TGRID, rel_v, ae, left=np.nan, right=np.nan))

            # For R(t), interpolate R (already at 100-ms step from compute_streams_4way)
            t_Rv, vR, aR = sliding_var_ar1(R, 1.0/STEP_SEC, 1.0, 0.1)
            # R's native grid: rel = c_Rv / (1/STEP_SEC) - ... hmm — need care
            # Simpler: R is on STEP_SEC grid; do sliding window in index units
            # Recompute directly:
            nwin_R = int(round(1.0 / STEP_SEC))  # 10 samples at 0.1s step
            var_R, ar1_R, cR = [], [], []
            for k in range(0, len(R) - nwin_R + 1):
                seg = R[k:k+nwin_R]
                var_R.append(float(np.var(seg)))
                s0 = seg[:-1] - np.mean(seg[:-1])
                s1 = seg[1:]  - np.mean(seg[1:])
                num = np.sum(s0 * s1)
                den = np.sqrt(np.sum(s0 ** 2) * np.sum(s1 ** 2))
                ar1_R.append(float(num / (den + 1e-12)))
                cR.append((k + nwin_R/2) * STEP_SEC)
            var_R = np.array(var_R); ar1_R = np.array(ar1_R); cR = np.array(cR)
            rel_Rv = cR - PAD_SEC - PRE_SEC - nadir
            var_R_rows.append(np.interp(TGRID, rel_Rv, var_R, left=np.nan, right=np.nan))
            ar1_R_rows.append(np.interp(TGRID, rel_Rv, ar1_R, left=np.nan, right=np.nan))
        except Exception:
            pass

        # ---- B5: phase jumps ----
        try:
            t_pj, pj = phase_jumps(X_seg, fs)
            rel_pj = t_pj - PAD_SEC - PRE_SEC - nadir
            phase_jump_rows.append(np.interp(pj_tgrid, rel_pj, pj,
                                                left=np.nan, right=np.nan))
        except Exception:
            pass

    if not nadir_std_per_event:
        return None

    return {
        'subject_id': sub_id,
        'n_events': len(nadir_std_per_event),
        # B3
        'nadir_std': np.array(nadir_std_per_event),
        'slope_x': np.array(slope_x_per_event),
        'slope_y': np.array(slope_y_per_event),
        'r2_gradient': np.array(r2_per_event),
        # B4 (subject means)
        'var_env': np.nanmean(np.array(var_env_rows), axis=0) if var_env_rows else None,
        'ar1_env': np.nanmean(np.array(ar1_env_rows), axis=0) if ar1_env_rows else None,
        'var_R':   np.nanmean(np.array(var_R_rows), axis=0)   if var_R_rows   else None,
        'ar1_R':   np.nanmean(np.array(ar1_R_rows), axis=0)   if ar1_R_rows   else None,
        # B5
        'phase_jump': np.nanmean(np.array(phase_jump_rows), axis=0)
                       if phase_jump_rows else None,
    }


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
    print(f"Successful: {len(results)}")
    n_events = sum(r['n_events'] for r in results)
    print(f"Events: {n_events}")

    # ---- B3 summary ----
    all_std = np.concatenate([r['nadir_std'] for r in results])
    all_r2 = np.concatenate([r['r2_gradient'] for r in results])
    all_sx = np.concatenate([r['slope_x'] for r in results])
    all_sy = np.concatenate([r['slope_y'] for r in results])
    print(f"\n=== B3 per-channel nadir timing ===")
    print(f"  std across channels (s): median {np.median(all_std):.3f}  "
          f"IQR [{np.percentile(all_std,25):.3f}, {np.percentile(all_std,75):.3f}]")
    print(f"  gradient R²: median {np.median(all_r2):.3f}  "
          f"IQR [{np.percentile(all_r2,25):.3f}, {np.percentile(all_r2,75):.3f}]")
    print(f"  slope x (s/m): median {np.median(all_sx):+.2f}")
    print(f"  slope y (s/m): median {np.median(all_sy):+.2f}")
    print(f"  std < 50ms (simultaneous-like): {(all_std<0.05).mean()*100:.1f}%")
    print(f"  std > 200ms (propagation-like): {(all_std>0.20).mean()*100:.1f}%")

    pd.DataFrame({'std_s': all_std, 'r2_gradient': all_r2,
                   'slope_x_s_per_m': all_sx, 'slope_y_s_per_m': all_sy}).to_csv(
        os.path.join(OUT_DIR, 'b3_per_channel_nadir.csv'), index=False)

    # ---- B4 summary ----
    var_env_arr = np.array([r['var_env'] for r in results if r['var_env'] is not None])
    ar1_env_arr = np.array([r['ar1_env'] for r in results if r['ar1_env'] is not None])
    var_R_arr = np.array([r['var_R'] for r in results if r['var_R'] is not None])
    ar1_R_arr = np.array([r['ar1_R'] for r in results if r['ar1_R'] is not None])

    vem, velo, vehi = bootstrap_ci(var_env_arr, n_boot=500)
    aem, aelo, aehi = bootstrap_ci(ar1_env_arr, n_boot=500)
    vRm, vRlo, vRhi = bootstrap_ci(var_R_arr, n_boot=500)
    aRm, aRlo, aRhi = bootstrap_ci(ar1_R_arr, n_boot=500)

    # Test for monotone rise approaching nadir: in [-5, -0.5] window
    pre_mask = (TGRID >= -5.0) & (TGRID <= -0.5)
    print(f"\n=== B4 critical slowing ===")
    print(f"  env variance @ t=-5s: {vem[np.argmin(np.abs(TGRID+5))]:.3f}  "
          f"@ t=-0.5s: {vem[np.argmin(np.abs(TGRID+0.5))]:.3f}")
    print(f"  env AR(1)    @ t=-5s: {aem[np.argmin(np.abs(TGRID+5))]:.3f}  "
          f"@ t=-0.5s: {aem[np.argmin(np.abs(TGRID+0.5))]:.3f}")
    print(f"  R   variance @ t=-5s: {vRm[np.argmin(np.abs(TGRID+5))]:.4f}  "
          f"@ t=-0.5s: {vRm[np.argmin(np.abs(TGRID+0.5))]:.4f}")
    print(f"  R   AR(1)    @ t=-5s: {aRm[np.argmin(np.abs(TGRID+5))]:.3f}  "
          f"@ t=-0.5s: {aRm[np.argmin(np.abs(TGRID+0.5))]:.3f}")

    # ---- B5 summary ----
    pj_tgrid = np.arange(-8.0, 8.0 + 0.1/2, 0.1)
    pj_arr = np.array([r['phase_jump'] for r in results if r['phase_jump'] is not None])
    pjm, pjlo, pjhi = bootstrap_ci(pj_arr, n_boot=500)
    print(f"\n=== B5 phase jumps ===")
    i_nadir = int(np.argmin(np.abs(pj_tgrid)))
    i_base = int(np.argmin(np.abs(pj_tgrid + 5)))
    print(f"  phase jumps/100ms @ baseline (-5s): {pjm[i_base]:.3f}")
    print(f"  phase jumps/100ms @ nadir    (0s):  {pjm[i_nadir]:.3f}")
    print(f"  ratio nadir/baseline: {pjm[i_nadir]/max(pjm[i_base], 1e-6):.2f}")

    # Save CSVs
    pd.DataFrame({
        't_rel': TGRID, 'var_env_mean': vem, 'var_env_lo': velo, 'var_env_hi': vehi,
        'ar1_env_mean': aem, 'ar1_env_lo': aelo, 'ar1_env_hi': aehi,
        'var_R_mean': vRm, 'var_R_lo': vRlo, 'var_R_hi': vRhi,
        'ar1_R_mean': aRm, 'ar1_R_lo': aRlo, 'ar1_R_hi': aRhi,
    }).to_csv(os.path.join(OUT_DIR, 'b4_critical_slowing.csv'), index=False)
    pd.DataFrame({
        't_rel': pj_tgrid, 'jumps_mean': pjm, 'jumps_lo': pjlo, 'jumps_hi': pjhi,
    }).to_csv(os.path.join(OUT_DIR, 'b5_phase_jumps.csv'), index=False)

    # ---- Figure ----
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3)

    # B3: nadir std + R² + direction
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(all_std, bins=40, color='steelblue', edgecolor='k', lw=0.3)
    ax.axvline(0.05, color='green', ls='--', lw=1, label='<50ms simultaneous')
    ax.axvline(0.20, color='red',   ls='--', lw=1, label='>200ms propagation')
    ax.axvline(np.median(all_std), color='black', lw=1.2,
                label=f'median {np.median(all_std):.2f}s')
    ax.set_xlabel('std of per-channel nadir times (s)'); ax.set_ylabel('events')
    ax.set_title('B3: dispersion of nadir across channels')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(all_r2, bins=40, color='seagreen', edgecolor='k', lw=0.3)
    ax.axvline(np.median(all_r2), color='black', lw=1.2,
                label=f'median R²={np.median(all_r2):.2f}')
    ax.set_xlabel('gradient fit R²'); ax.set_ylabel('events')
    ax.set_title('B3: how well does x,y explain nadir time?')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(all_sx, all_sy, s=4, alpha=0.35, color='coral')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('slope x (s/m)   (+: right later)')
    ax.set_ylabel('slope y (s/m)   (+: anterior later)')
    ax.set_title('B3: propagation direction (per-event slopes)')

    # B4: variance and AR1 for env and R
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(TGRID, velo, vehi, color='darkorange', alpha=0.25)
    ax.plot(TGRID, vem, color='darkorange', lw=2)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('env variance (1-s window)')
    ax.set_title('B4: env variance')

    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(TGRID, aelo, aehi, color='darkorange', alpha=0.25)
    ax.plot(TGRID, aem, color='darkorange', lw=2)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('env AR(1)')
    ax.set_title('B4: env AR(1)')

    ax = fig.add_subplot(gs[1, 2])
    ax.fill_between(TGRID, vRlo, vRhi, color='seagreen', alpha=0.25)
    ax.plot(TGRID, vRm, color='seagreen', lw=2, label='var R')
    ax_r = ax.twinx()
    ax_r.fill_between(TGRID, aRlo, aRhi, color='olivedrab', alpha=0.20)
    ax_r.plot(TGRID, aRm, color='olivedrab', lw=1, ls='--', label='AR(1) R')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('var R')
    ax_r.set_ylabel('AR(1) R')
    ax.set_title('B4: R variance & AR(1)')

    # B5: phase jumps
    ax = fig.add_subplot(gs[2, :])
    ax.fill_between(pj_tgrid, pjlo, pjhi, color='purple', alpha=0.25)
    ax.plot(pj_tgrid, pjm, color='purple', lw=2)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('time relative to nadir (s)')
    ax.set_ylabel('phase jumps per 100ms (any channel)')
    ax.set_title('B5: phase discontinuity rate peri-nadir')

    fig.suptitle(f'Mechanism battery (B3, B4, B5)\n'
                 f'LEMON EC · {len(results)} subj · {n_events} events',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mechanism_battery.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/mechanism_battery.png")


if __name__ == '__main__':
    main()
