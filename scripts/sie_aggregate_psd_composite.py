#!/usr/bin/env python3
"""
B26 + B27 + Canonical paper figure on composite v2 detector.

Per subject (composite v2 events):
  Panel A (B26 standing aggregate PSD):
    - 8-s Welch on mean-channel, 2-s hop, full recording
    - Median across windows → subject aggregate PSD on [2, 25] Hz
    - 1/f aperiodic subtraction (log-linear fit on 2-5 ∪ 9-22 Hz, excluded 5-9)
    - Log-space residual
  Panel B (B27 event-locked ratio):
    - Per-event 4-s Welch centered at t0_net + 1 s (B14 Q4 peak lag)
    - Per-subject baseline PSD = median over all 4-s sliding windows (1-s hop)
    - Per-subject log10(event/baseline) ratio
  Pooled grand-averages per panel with subject-level cluster bootstrap 95% CI.

B26 is detector-INDEPENDENT (no events used). Composite Panel A on LEMON =
envelope Panel A exactly. Only B27 Panel B depends on detector.

Canonical paper figure: two-panel, Panel A above Panel B, on same subjects.

Cohort-parameterized.

Usage:
    python scripts/sie_aggregate_psd_composite.py --cohort lemon
    python scripts/sie_aggregate_psd_composite.py --cohort lemon_EO
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

warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

# Panel A (B26) params
AGG_WIN_SEC = 8.0
AGG_HOP_SEC = 2.0
AGG_NFFT_MULT = 4
APERIODIC_RANGES = [(2.0, 5.0), (9.0, 22.0)]

# Panel B (B27) params
EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0

FREQ_LO, FREQ_HI = 2.0, 25.0
COMMON_FREQ = np.arange(FREQ_LO, FREQ_HI + 0.05, 0.05)

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


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def aperiodic_log_resid(freqs, psd):
    mask = np.zeros_like(freqs, dtype=bool)
    for lo, hi in APERIODIC_RANGES:
        mask |= (freqs >= lo) & (freqs <= hi)
    logf = np.log10(freqs)
    logp = np.log10(psd + 1e-20)
    good = mask & np.isfinite(logp) & (logp > -10)
    if good.sum() < 8:
        return np.zeros_like(psd)
    A = np.column_stack([logf[good], np.ones(good.sum())])
    coefs, *_ = np.linalg.lstsq(A, logp[good], rcond=None)
    a, b = float(coefs[0]), float(coefs[1])
    return logp - (a * logf + b)


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
    """args = (sub_id, events_path) or (sub_id, events_path, weights_per_event)."""
    if len(args) == 3:
        sub_id, events_path, weights = args
    else:
        sub_id, events_path = args
        weights = None
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 1:
        return None
    try:
        raw = _LOADER(sub_id, **_LOADER_KW)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    # ---- Panel A: standing aggregate PSD (B26) ----
    nperseg_A = int(round(AGG_WIN_SEC * fs))
    nhop_A = int(round(AGG_HOP_SEC * fs))
    nfft_A = nperseg_A * AGG_NFFT_MULT
    freqs_A = np.fft.rfftfreq(nfft_A, 1.0 / fs)
    mask_A = (freqs_A >= FREQ_LO) & (freqs_A <= FREQ_HI)
    fA = freqs_A[mask_A]
    psds_A = []
    for i in range(0, len(y) - nperseg_A + 1, nhop_A):
        psds_A.append(welch_one(y[i:i + nperseg_A], fs, nfft_A)[mask_A])
    if len(psds_A) < 5:
        return None
    agg_psd = np.nanmedian(np.array(psds_A), axis=0)
    log_resid_A = aperiodic_log_resid(fA, agg_psd)
    log_resid_A_on_common = np.interp(COMMON_FREQ, fA, log_resid_A)

    # ---- Panel B: event-locked ratio (B27) ----
    nperseg_B = int(round(EV_WIN_SEC * fs))
    nhop_B = int(round(1.0 * fs))
    nfft_B = nperseg_B * EV_NFFT_MULT
    freqs_B = np.fft.rfftfreq(nfft_B, 1.0 / fs)
    mask_B = (freqs_B >= FREQ_LO) & (freqs_B <= FREQ_HI)
    fB = freqs_B[mask_B]

    # Baseline PSD (median over all sliding windows)
    base_rows = []
    for i in range(0, len(y) - nperseg_B + 1, nhop_B):
        base_rows.append(welch_one(y[i:i + nperseg_B], fs, nfft_B)[mask_B])
    if len(base_rows) < 10:
        return None
    baseline_psd = np.nanmedian(np.array(base_rows), axis=0)

    # Event PSDs (4-s window centered at t0_net + EV_LAG_S)
    ev_rows = []
    ev_weights = []
    for k, (_, ev) in enumerate(events.iterrows()):
        tc = float(ev['t0_net']) + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg_B
        if i0 < 0 or i1 > len(y):
            continue
        ev_rows.append(welch_one(y[i0:i1], fs, nfft_B)[mask_B])
        if weights is not None and k < len(weights):
            ev_weights.append(max(float(weights[k]), 0.0))
        else:
            ev_weights.append(1.0)
    if not ev_rows:
        return None
    ev_arr = np.array(ev_rows)
    w_arr = np.array(ev_weights)
    if w_arr.sum() <= 0:
        return None
    event_psd = np.average(ev_arr, axis=0, weights=w_arr)
    log_ratio_B = np.log10(event_psd + 1e-20) - np.log10(baseline_psd + 1e-20)
    log_ratio_B_on_common = np.interp(COMMON_FREQ, fB, log_ratio_B)

    return {
        'subject_id': sub_id,
        'n_events': len(ev_rows),
        'log_resid_A': log_resid_A_on_common,
        'log_ratio_B': log_ratio_B_on_common,
    }


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


def find_peak_near(curve, freqs, target, half_window=0.30):
    m = (freqs >= target - half_window) & (freqs <= target + half_window)
    if not np.any(np.isfinite(curve[m])):
        return np.nan, np.nan
    idx = np.where(m)[0]
    k = idx[int(np.nanargmax(curve[m]))]
    return float(freqs[k]), float(curve[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    ap.add_argument('--shape-weighted', action='store_true',
                    help='Weight Panel B per-subject event-locked PSD by '
                         'max(template_rho, 0); requires per_event_quality CSV')
    args = ap.parse_args()

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'psd_timelapse', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    # Optional shape-weighted mode: load per-event template_rho lookup
    weights_by_subj = None
    if args.shape_weighted:
        quality_csv = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                                    'quality',
                                    f'per_event_quality_{args.cohort}_composite.csv')
        if not os.path.isfile(quality_csv):
            print(f"Shape-weighted: quality CSV not found: {quality_csv}")
            return
        qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
        qual['t0r'] = qual['t0_net'].round(3)
        weights_by_subj = {}
        for sid, g in qual.groupby('subject_id'):
            weights_by_subj[sid] = dict(zip(g['t0r'], g['template_rho']))
        print(f"Shape-weighted: loaded template_rho for "
              f"{len(weights_by_subj)} subjects, {len(qual)} events")

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if not os.path.isfile(ep):
            continue
        if args.shape_weighted:
            try:
                ev_df = pd.read_csv(ep).dropna(subset=['t0_net'])
            except Exception:
                continue
            sub_w = weights_by_subj.get(r['subject_id'], {})
            t0r = ev_df['t0_net'].round(3).values
            w_arr = np.array([sub_w.get(t, np.nan) for t in t0r])
            if not np.any(np.isfinite(w_arr)):
                continue
            w_arr = np.where(np.isfinite(w_arr), w_arr, 0.0)
            tasks.append((r['subject_id'], ep, list(w_arr)))
        else:
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} composite{' shape-weighted' if args.shape_weighted else ''}"
          f" · subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")
    n_events = sum(r['n_events'] for r in results)
    print(f"Total events in Panel B: {n_events}")

    mat_A = np.array([r['log_resid_A'] for r in results])
    mat_B = np.array([r['log_ratio_B'] for r in results])

    grand_A, lo_A, hi_A = bootstrap_ci(mat_A)
    grand_B, lo_B, hi_B = bootstrap_ci(mat_B)

    # Check peaks
    targets = [7.60, 7.83, 14.3, 20.8]
    print(f"\n=== {args.cohort} composite · Panel A (standing, B26-equivalent) ===")
    print(f"(envelope B26: no 7.83 peak; only alpha at 9.45 Hz)")
    iaf_mask = (COMMON_FREQ >= 7) & (COMMON_FREQ <= 13)
    iaf_idx = int(np.argmax(grand_A[iaf_mask]))
    iaf_f = float(COMMON_FREQ[iaf_mask][iaf_idx])
    iaf_v = float(grand_A[iaf_mask][iaf_idx])
    print(f"  IAF-like peak @ {iaf_f:.2f} Hz (log-resid {iaf_v:+.3f})")
    for t in targets:
        pf, pv = find_peak_near(grand_A, COMMON_FREQ, t)
        print(f"  near {t:.2f} Hz: peak at {pf:.3f} (log-resid {pv:+.3f})")

    print(f"\n=== {args.cohort} composite · Panel B (event-locked ratio, B27-equivalent) ===")
    print(f"(envelope B27: 7.85 Hz peak 5.92× across 3 cohorts)")
    # Peak in SR range
    sr_mask = (COMMON_FREQ >= 7) & (COMMON_FREQ <= 8.5)
    sr_idx = int(np.argmax(grand_B[sr_mask]))
    sr_f = float(COMMON_FREQ[sr_mask][sr_idx])
    sr_logv = float(grand_B[sr_mask][sr_idx])
    sr_ratio = 10 ** sr_logv
    print(f"  SR-band peak @ {sr_f:.3f} Hz  log-ratio {sr_logv:+.3f}  ratio {sr_ratio:.2f}×")
    for t in [7.60, 7.83, 14.3, 20.8]:
        pf, pv = find_peak_near(grand_B, COMMON_FREQ, t, half_window=0.5)
        print(f"  near {t:.2f} Hz: peak at {pf:.3f}  log-ratio {pv:+.3f}  ratio {10**pv:.2f}×")

    # Save CSVs
    out_csv_name = f'aggregate_psd_B26_B27{"_sw" if args.shape_weighted else ""}.csv'
    out_csv_path = os.path.join(out_dir, out_csv_name)
    pd.DataFrame({
        'freq_hz': COMMON_FREQ,
        'panel_A_log_resid_grand': grand_A, 'panel_A_lo': lo_A, 'panel_A_hi': hi_A,
        'panel_B_log_ratio_grand': grand_B, 'panel_B_lo': lo_B, 'panel_B_hi': hi_B,
    }).to_csv(out_csv_path, index=False)

    # Canonical paper figure
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(COMMON_FREQ, grand_A, color='#2166ac', lw=2,
            label=f'{args.cohort} composite (n={len(mat_A)})')
    ax.fill_between(COMMON_FREQ, lo_A, hi_A, color='#2166ac', alpha=0.2)
    ax.axvline(7.83, color='red', ls=':', lw=1, label='Schumann 7.83')
    ax.axvline(9.45, color='orange', ls=':', lw=1, label='α 9.45')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('log₁₀(PSD / 1/f fit)')
    ax.set_title(f'Panel A (B26 equivalent) — standing aggregate PSD · {args.cohort} composite\n'
                 f'IAF peak {iaf_f:.2f} Hz  ·  no 7.83 Hz peak')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(FREQ_LO, FREQ_HI)

    ax = axes[1]
    ax.plot(COMMON_FREQ, grand_B, color='#d7301f', lw=2,
            label=f'{args.cohort} composite (n={len(mat_B)})')
    ax.fill_between(COMMON_FREQ, lo_B, hi_B, color='#d7301f', alpha=0.2)
    ax.axvline(7.83, color='red', ls=':', lw=1, label='Schumann 7.83')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('log₁₀(event / baseline)')
    ax.set_title(f'Panel B (B27 equivalent) — event-locked ratio\n'
                 f'SR-band peak {sr_f:.3f} Hz  ratio {sr_ratio:.2f}×   (envelope 7.85 Hz 5.92×)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(f'Canonical paper figure · {args.cohort} composite v2 · '
                 f'{len(mat_A)} subjects · {n_events} events',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'canonical_paper_figure_composite.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/canonical_paper_figure_composite.png")
    print(f"Saved: {out_dir}/aggregate_psd_B26_B27{"_sw" if args.shape_weighted else ""}.csv")


if __name__ == '__main__':
    main()
