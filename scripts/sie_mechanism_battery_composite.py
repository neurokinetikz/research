#!/usr/bin/env python3
"""
B4 + B5 + B8 re-run on composite v2 detector.

Covers three Arc-2 mechanism tests in a single pass:

  B4  Critical slowing test
      Sliding variance + AR(1) of env z and R(t), interpolated onto
      [-8, +8] s grid relative to composite nadir. Test whether variance
      and AR(1) trend upward toward the nadir.

  B5  Phase discontinuity (phase-reset) test
      Per-100-ms bin count of cross-channel phase jumps (|dφ − expected| > π/2)
      in the narrowband 7.2-8.4 Hz signal. Peri-nadir [-1, +1] s vs baseline
      [-8, -4] s elevation.

  B8  Template_ρ stratification of B5 and B3 metrics (per-event channel
      nadir std + propagation slope). Compare Q1 vs Q4.

Nadir is taken as t0_net − 1.30 s (composite A3/A7 invariant; same in both
LEMON states). This saves running find_nadir per event.

Cohort-parameterized.

Usage:
    python scripts/sie_mechanism_battery_composite.py --cohort lemon
    python scripts/sie_mechanism_battery_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import mannwhitneyu
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import (
    load_lemon, load_dortmund, load_srm, load_tdbrain,
    load_chbmp, load_hbn_by_subject,
)
from scripts.sie_mechanism_battery import (
    per_channel_nadir, channel_positions, phase_jumps,
    sliding_var_ar1,
)
from scripts.sie_perionset_multistream import (
    F0, HALF_BW, PRE_SEC, POST_SEC, PAD_SEC,
)
from scripts.sie_perionset_triple_average_composite import bandpass

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

# Grids
TGRID_B4 = np.arange(-8.0, 8.0 + 0.05, 0.1)     # 100-ms bins for var/ar1
TGRID_B5 = np.arange(-8.0, 8.0 + 0.05, 0.1)     # 100-ms bins for phase jumps

# Composite nadir offset (from §23c A7 composite: nadir at −1.30 s rel t0_net)
NADIR_OFFSET = -1.30

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
    sub_id, df_sub = args
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X_all = raw.get_data() * 1e6
    t_end = raw.times[-1]
    pos = channel_positions(raw.ch_names)

    # Overall accumulators (grand-mean)
    var_env_rows, ar1_env_rows = [], []
    phase_jump_rows = []
    # B8: per-quartile accumulators
    buckets = {q: {'pj': [], 'nadir_std': [], 'r2': [], 'slope_y': []}
                for q in ['Q1', 'Q2', 'Q3', 'Q4']}

    for _, ev in df_sub.iterrows():
        t0 = float(ev['t0_net'])
        q = ev.get('rho_q', None)
        lo = t0 - PRE_SEC - PAD_SEC
        hi = t0 + POST_SEC + PAD_SEC
        if lo < 0 or hi > t_end:
            continue
        i0 = int(round(lo * fs)); i1 = int(round(hi * fs))
        X_seg = X_all[:, i0:i1]
        if X_seg.shape[1] < int(round((hi - lo) * fs * 0.95)):
            continue
        # Approximate nadir = NADIR_OFFSET relative to t0_net (window-internal
        # nadir offset from window start is PAD+PRE+NADIR_OFFSET)
        # For stream alignment: rel_to_nadir = absolute_time_in_segment
        #                                       - (PAD + PRE + NADIR_OFFSET)

        # B4: sliding var/ar1 on env z
        try:
            y_mean = X_seg.mean(axis=0)
            y_envbp = bandpass(y_mean, fs, F0 - HALF_BW, F0 + HALF_BW)
            y_env_full = np.abs(signal.hilbert(y_envbp))
            t_v, ve, ae = sliding_var_ar1(y_env_full, fs, 1.0, 0.1)
            rel_v = t_v - PAD_SEC - PRE_SEC - NADIR_OFFSET
            var_env_rows.append(np.interp(TGRID_B4, rel_v, ve, left=np.nan, right=np.nan))
            ar1_env_rows.append(np.interp(TGRID_B4, rel_v, ae, left=np.nan, right=np.nan))
        except Exception:
            pass

        # B5: phase jumps
        try:
            t_pj, pj = phase_jumps(X_seg, fs)
            rel_pj = t_pj - PAD_SEC - PRE_SEC - NADIR_OFFSET
            pj_row = np.interp(TGRID_B5, rel_pj, pj, left=np.nan, right=np.nan)
            phase_jump_rows.append(pj_row)
            if isinstance(q, str) and q in buckets:
                buckets[q]['pj'].append(pj_row)
        except Exception:
            pass

        # B8: B3-style per-channel nadir std + gradient (only bother for quartile groups)
        if isinstance(q, str) and q in buckets:
            try:
                nadir_per_ch = per_channel_nadir(X_seg, fs, NADIR_OFFSET)
                std_ch = float(np.nanstd(nadir_per_ch))
                good = np.isfinite(nadir_per_ch) & np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
                if good.sum() >= 6:
                    X_fit = np.column_stack([pos[good, 0], pos[good, 1], np.ones(good.sum())])
                    y_fit = nadir_per_ch[good]
                    coefs, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
                    y_pred = X_fit @ coefs
                    ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
                    ss_res = np.sum((y_fit - y_pred) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    buckets[q]['nadir_std'].append(std_ch)
                    buckets[q]['r2'].append(float(r2))
                    buckets[q]['slope_y'].append(float(coefs[1]))
            except Exception:
                pass

    if not phase_jump_rows:
        return None

    out = {
        'subject_id': sub_id,
        'n_events': len(phase_jump_rows),
        'var_env': np.nanmean(np.array(var_env_rows), axis=0) if var_env_rows else None,
        'ar1_env': np.nanmean(np.array(ar1_env_rows), axis=0) if ar1_env_rows else None,
        'pj':      np.nanmean(np.array(phase_jump_rows), axis=0),
    }
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        if buckets[q]['pj']:
            out[f'{q}_pj'] = np.nanmean(np.array(buckets[q]['pj']), axis=0)
            out[f'{q}_pj_n'] = len(buckets[q]['pj'])
            out[f'{q}_nadir_std'] = np.array(buckets[q]['nadir_std'])
            out[f'{q}_r2'] = np.array(buckets[q]['r2'])
            out[f'{q}_slope_y'] = np.array(buckets[q]['slope_y'])
        else:
            out[f'{q}_pj'] = None
            out[f'{q}_pj_n'] = 0
    return out


def bootstrap_ci(mat, n_boot=1000, seed=0):
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


def elevation(traj, t=TGRID_B5):
    peri = (t >= -1) & (t <= +1)
    base = (t >= -8) & (t <= -4)
    m_peri = np.nanmean(traj[peri]); m_base = np.nanmean(traj[base])
    return (m_peri / m_base) if m_base > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'mechanism_battery', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
    qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    tasks = [(sid, g) for sid, g in qual.groupby('subject_id')]
    print(f"Cohort: {args.cohort} composite  ·  subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    # B4: critical slowing
    pj_mat = np.array([r['pj'] for r in results])
    ve_mat = np.array([r['var_env'] for r in results if r['var_env'] is not None])
    ae_mat = np.array([r['ar1_env'] for r in results if r['ar1_env'] is not None])

    ve_grand, _, _ = bootstrap_ci(ve_mat)
    ae_grand, _, _ = bootstrap_ci(ae_mat)
    pj_grand, pj_lo, pj_hi = bootstrap_ci(pj_mat)

    # Does env var/ar1 trend up toward nadir? Compare baseline [-8,-4] vs peri [-2,0]
    peri_mask = (TGRID_B4 >= -2) & (TGRID_B4 <= 0)
    base_mask = (TGRID_B4 >= -8) & (TGRID_B4 <= -4)
    var_peri = np.nanmean(ve_grand[peri_mask]); var_base = np.nanmean(ve_grand[base_mask])
    ar1_peri = np.nanmean(ae_grand[peri_mask]); ar1_base = np.nanmean(ae_grand[base_mask])

    print(f"\n=== {args.cohort} composite · B4 critical slowing ===")
    print(f"  env variance: baseline {var_base:.4f}  peri-nadir {var_peri:.4f}  ratio {var_peri/var_base:.3f}")
    print(f"  env AR(1):    baseline {ar1_base:.4f}  peri-nadir {ar1_peri:.4f}  Δ {ar1_peri - ar1_base:+.4f}")
    print(f"  (envelope B4: no critical slowing; AR(1) flat)")

    # B5: phase reset
    pj_elev_grand = elevation(pj_grand)
    per_sub_elev = np.array([elevation(r['pj']) for r in results])
    per_sub_elev = per_sub_elev[np.isfinite(per_sub_elev)]
    print(f"\n=== {args.cohort} composite · B5 phase reset (all events) ===")
    print(f"  Grand-mean elevation: {pj_elev_grand:.3f}×")
    print(f"  Per-subject elevation: median {np.median(per_sub_elev):.3f}   "
          f"IQR [{np.percentile(per_sub_elev,25):.3f}, {np.percentile(per_sub_elev,75):.3f}]")
    print(f"  Subjects > 1.2×: {(per_sub_elev > 1.2).mean()*100:.1f}%")
    print(f"  (envelope B5 all-events: 1.61× at nadir)")

    # B8: per-quartile
    print(f"\n=== {args.cohort} composite · B8 phase-reset by template_ρ quartile ===")
    q_elevations = {}
    for q in ['Q1', 'Q4']:
        traj = np.array([r[f'{q}_pj'] for r in results if r[f'{q}_pj'] is not None])
        if len(traj) == 0:
            continue
        grand_q, _, _ = bootstrap_ci(traj)
        elev_q = elevation(grand_q)
        # per-subject Q-elevation
        per_sub_q = np.array([elevation(r[f'{q}_pj']) for r in results
                              if r[f'{q}_pj'] is not None])
        per_sub_q = per_sub_q[np.isfinite(per_sub_q)]
        q_elevations[q] = (elev_q, grand_q)
        n_events_q = sum(r[f'{q}_pj_n'] for r in results)
        print(f"  {q}: grand {elev_q:.3f}×  per-sub median {np.median(per_sub_q):.3f}   n_sub={len(per_sub_q)}  n_events={n_events_q}")

    # Nadir std and gradient R² per quartile
    print(f"\n=== {args.cohort} composite · B8 nadir std + propagation R² per quartile ===")
    for q in ['Q1', 'Q4']:
        stds = np.concatenate([r[f'{q}_nadir_std'] for r in results
                                if f'{q}_nadir_std' in r and r[f'{q}_nadir_std'] is not None
                                and len(r[f'{q}_nadir_std'])])
        r2s  = np.concatenate([r[f'{q}_r2'] for r in results
                                if f'{q}_r2' in r and r[f'{q}_r2'] is not None
                                and len(r[f'{q}_r2'])])
        slys = np.concatenate([r[f'{q}_slope_y'] for r in results
                                if f'{q}_slope_y' in r and r[f'{q}_slope_y'] is not None
                                and len(r[f'{q}_slope_y'])])
        if len(stds) > 0:
            print(f"  {q}: nadir_std median {np.median(stds):.3f}s  R² median {np.median(r2s):.3f}  "
                  f"slope_y median {np.median(slys):.3f} s/m  n={len(stds)}")

    # Save outputs
    pd.DataFrame({
        't_rel': TGRID_B4,
        'var_env_grand': ve_grand,
        'ar1_env_grand': ae_grand,
        'pj_grand': pj_grand,
        'pj_ci_lo': pj_lo,
        'pj_ci_hi': pj_hi,
    }).to_csv(os.path.join(out_dir, 'mechanism_grand_average.csv'), index=False)

    summary_rows = [{
        'metric': 'B4_env_var_ratio_peri_over_baseline', 'value': float(var_peri/var_base) if var_base > 0 else np.nan,
    }, {
        'metric': 'B4_env_ar1_delta_peri_minus_baseline', 'value': float(ar1_peri - ar1_base),
    }, {
        'metric': 'B5_phase_jump_grand_elevation_all', 'value': float(pj_elev_grand),
    }, {
        'metric': 'B5_phase_jump_per_subject_median_all', 'value': float(np.median(per_sub_elev)),
    }]
    for q in ['Q1', 'Q4']:
        if q in q_elevations:
            summary_rows.append({'metric': f'B8_phase_jump_grand_elevation_{q}',
                                  'value': float(q_elevations[q][0])})
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, 'mechanism_summary.csv'),
                                       index=False)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(TGRID_B4, ve_grand, color='darkorange', lw=2, label='env variance')
    ax.axvline(0, color='k', ls='--', lw=0.5, label='nadir')
    ax.axvspan(-8, -4, alpha=0.1, color='blue', label='baseline')
    ax.axvspan(-2, 0, alpha=0.1, color='red', label='peri-nadir')
    ax.set_title(f'B4 env variance · ratio peri/base {var_peri/var_base:.2f}')
    ax.set_xlabel('time rel nadir (s)'); ax.set_ylabel('variance')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(TGRID_B4, ae_grand, color='seagreen', lw=2, label='env AR(1)')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.axvspan(-8, -4, alpha=0.1, color='blue')
    ax.axvspan(-2, 0, alpha=0.1, color='red')
    ax.set_title(f'B4 env AR(1) · Δ(peri − base) {ar1_peri - ar1_base:+.3f}')
    ax.set_xlabel('time rel nadir (s)'); ax.set_ylabel('AR(1)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(TGRID_B5, pj_grand, color='purple', lw=2, label=f'all events (elev {pj_elev_grand:.2f}×)')
    ax.fill_between(TGRID_B5, pj_lo, pj_hi, color='purple', alpha=0.2)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.axvspan(-8, -4, alpha=0.1, color='blue')
    ax.axvspan(-1, +1, alpha=0.1, color='red')
    ax.set_title(f'B5 phase-jump rate · all events')
    ax.set_xlabel('time rel nadir (s)'); ax.set_ylabel('jumps/100ms')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for q, color in [('Q1', '#4575b4'), ('Q4', '#d73027')]:
        if q in q_elevations:
            e, trace = q_elevations[q]
            ax.plot(TGRID_B5, trace, color=color, lw=2, label=f'{q} (elev {e:.2f}×)')
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.axvspan(-8, -4, alpha=0.1, color='blue')
    ax.axvspan(-1, +1, alpha=0.1, color='red')
    ax.set_title(f'B8 phase-jump rate by template_ρ quartile')
    ax.set_xlabel('time rel nadir (s)'); ax.set_ylabel('jumps/100ms')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(f'B4+B5+B8 mechanism battery · {args.cohort} composite v2', y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mechanism_battery.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/mechanism_battery.png")


if __name__ == '__main__':
    main()
