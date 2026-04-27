#!/usr/bin/env python3
"""
B28 re-run on composite v2 detector.

Per subject (composite events):
  - IAF: aggregate PSD peak in [7, 13] Hz (full-recording median Welch)
  - β event-locked peak: argmax in [16, 24] Hz of event/baseline ratio
    (4-s window at t0_net + 1 s)
  - SR event-locked peak: argmax in [7, 8.3] Hz (positive control, expected
    IAF-independent)

Tests:
  Spearman ρ(IAF, β_peak) and OLS slope.
  H1 (2×IAF harmonic): slope ≈ 2
  H2 (fixed β ≈ 20 Hz): slope ≈ 0

Envelope B28: slope 0.14 (not harmonic); cohort mean β peak 19.89 Hz;
SR and β amplitudes co-vary subject-level ρ = +0.55.

Cohort-parameterized.

Usage:
    python scripts/sie_beta_peak_iaf_coupling_composite.py --cohort lemon
    python scripts/sie_beta_peak_iaf_coupling_composite.py --cohort lemon_EO
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, pearsonr
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

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0

IAF_WIN_SEC = 8.0
IAF_HOP_SEC = 2.0
IAF_NFFT_MULT = 4
IAF_LO, IAF_HI = 7.0, 13.0

BETA_LO, BETA_HI = 16.0, 24.0
SR_LO, SR_HI = 7.0, 8.3

SCHUMANN_F = 7.83

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


def parabolic_peak(y, x):
    k = int(np.argmax(y))
    if 1 <= k < len(y) - 1 and y[k-1] > 0 and y[k+1] > 0:
        y0, y1, y2 = y[k-1], y[k], y[k+1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = max(-1.0, min(1.0, delta))
        return float(x[k] + delta * (x[1] - x[0]))
    return float(x[k])


def compute_iaf(y, fs):
    nperseg = int(round(IAF_WIN_SEC * fs))
    nhop = int(round(IAF_HOP_SEC * fs))
    nfft = nperseg * IAF_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= IAF_LO) & (freqs <= IAF_HI)
    psds = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psds.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if not psds:
        return np.nan
    return parabolic_peak(np.nanmedian(np.array(psds), axis=0), freqs[mask])


def event_peak_in_band(y, fs, t_events, f_lo, f_hi, weights=None):
    """Event-locked PSD peak in band, optionally template_rho-weighted.

    weights : array of len(t_events), per-event weight (typically max(template_rho, 0)).
              If None, all events weighted equally (unweighted mean).
              Negative weights are clipped to 0.
    """
    nperseg = int(round(EV_WIN_SEC * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    f_band = freqs[mask]
    nhop = int(round(1.0 * fs))
    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        base_rows.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if len(base_rows) < 10:
        return np.nan, np.nan
    baseline = np.nanmedian(np.array(base_rows), axis=0)
    ev_rows = []
    ev_weights = []
    for k, t0 in enumerate(t_events):
        tc = t0 + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        ev_rows.append(welch_one(y[i0:i1], fs, nfft)[mask])
        if weights is not None:
            w = max(float(weights[k]), 0.0)
        else:
            w = 1.0
        ev_weights.append(w)
    if not ev_rows:
        return np.nan, np.nan
    ev_arr = np.array(ev_rows)
    w_arr = np.array(ev_weights)
    if w_arr.sum() <= 0:
        return np.nan, np.nan
    # Weighted mean PSD across events (uses np.average for proper weighting)
    ev_avg = np.average(ev_arr, axis=0, weights=w_arr)
    ratio = (ev_avg + 1e-20) / (baseline + 1e-20)
    pk_f = parabolic_peak(ratio, f_band)
    pk_r = float(ratio[int(np.argmax(ratio))])
    return pk_f, pk_r


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
    """args = (sub_id, events_path) or (sub_id, events_path, weights_per_event).
    If weights_per_event provided (length matches events rows), computes
    template_rho-weighted PSD aggregation.
    """
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
    iaf = compute_iaf(y, fs)
    t_events = events['t0_net'].astype(float).values
    if weights is not None and len(weights) == len(t_events):
        w = np.asarray(weights, dtype=float)
    else:
        w = None
    sr_f, sr_r = event_peak_in_band(y, fs, t_events, SR_LO, SR_HI, weights=w)
    beta_f, beta_r = event_peak_in_band(y, fs, t_events, BETA_LO, BETA_HI, weights=w)
    return {
        'subject_id': sub_id,
        'iaf_hz': iaf,
        'sr_peak_hz': sr_f,
        'sr_peak_ratio': sr_r,
        'beta_peak_hz': beta_f,
        'beta_peak_ratio': beta_r,
        'beta_over_iaf': beta_f / iaf if iaf > 0 else np.nan,
        'n_events': int(len(t_events)),
        'mean_weight': float(np.mean(w)) if w is not None else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    ap.add_argument('--q4', action='store_true',
                    help='Filter events to per-subject template_rho Q4 '
                         '(requires per_event_quality CSV to exist)')
    ap.add_argument('--shape-weighted', action='store_true',
                    help='Weight every event by max(template_rho, 0) when '
                         'aggregating per-subject PSDs; do not drop events '
                         '(requires per_event_quality CSV to exist).')
    args = ap.parse_args()
    if args.q4 and args.shape_weighted:
        print("Cannot use --q4 and --shape-weighted together; pick one.")
        return

    loader, loader_kw, events_dir = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'psd_timelapse', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]

    # Q4 filter: build per-subject Q4 events, write temp Q4 events files
    q4_dir = None
    out_csv_name = 'beta_peak_iaf_coupling.csv'
    if args.q4:
        quality_csv = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                                    'quality',
                                    f'per_event_quality_{args.cohort}_composite.csv')
        if not os.path.isfile(quality_csv):
            print(f"Q4 mode: quality CSV not found at {quality_csv}")
            print(f"Q4 mode requires per-subject template_rho. Aborting.")
            return
        qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
        # Per-subject quartile assignment (only for subjects with >=4 events)
        def _qcut_per_subj(g):
            if g.nunique() < 4:
                return pd.Series(['NA'] * len(g), index=g.index)
            return pd.qcut(g, 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
        qual['rho_q'] = qual.groupby('subject_id')['template_rho'].transform(_qcut_per_subj)
        q4 = qual[qual['rho_q'] == 'Q4']
        print(f"Q4 mode: {len(q4)} Q4 events across "
              f"{q4['subject_id'].nunique()} subjects "
              f"(of {qual['subject_id'].nunique()} total)")
        q4_dir = os.path.join(out_dir, '_q4_events_tmp')
        os.makedirs(q4_dir, exist_ok=True)
        # Build per-subject Q4 events files (only t0_net column needed)
        q4_subs = set()
        for sid, g in q4.groupby('subject_id'):
            tmp = os.path.join(q4_dir, f'{sid}_sie_events.csv')
            g[['subject_id', 't0_net']].to_csv(tmp, index=False)
            q4_subs.add(sid)
        out_csv_name = 'beta_peak_iaf_coupling_q4.csv'

    # Shape-weighted mode: load template_rho per event, pass weights through
    weights_by_subj = None
    if args.shape_weighted:
        quality_csv = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                                    'quality',
                                    f'per_event_quality_{args.cohort}_composite.csv')
        if not os.path.isfile(quality_csv):
            print(f"Shape-weighted: quality CSV not found at {quality_csv}")
            return
        qual = pd.read_csv(quality_csv).dropna(subset=['template_rho']).copy()
        qual['t0r'] = qual['t0_net'].round(3)
        weights_by_subj = {}
        for sid, g in qual.groupby('subject_id'):
            weights_by_subj[sid] = dict(zip(g['t0r'], g['template_rho']))
        print(f"Shape-weighted: loaded template_rho for "
              f"{len(weights_by_subj)} subjects, "
              f"{len(qual)} events total. "
              f"Distribution: median {qual['template_rho'].median():.3f}, "
              f"5-95% [{qual['template_rho'].quantile(0.05):.3f}, "
              f"{qual['template_rho'].quantile(0.95):.3f}]")
        out_csv_name = 'beta_peak_iaf_coupling_sw.csv'

    tasks = []
    for _, r in ok.iterrows():
        if args.q4:
            if r['subject_id'] not in q4_subs:
                continue
            ep = os.path.join(q4_dir, f'{r["subject_id"]}_sie_events.csv')
        else:
            ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if not os.path.isfile(ep):
            continue
        if args.shape_weighted:
            # Load events, look up template_rho per t0_net, build weights aligned
            # to the events file's row order.
            try:
                ev_df = pd.read_csv(ep).dropna(subset=['t0_net'])
            except Exception:
                continue
            sub_w_map = weights_by_subj.get(r['subject_id'], {})
            t0r = ev_df['t0_net'].round(3).values
            w_arr = np.array([sub_w_map.get(t, np.nan) for t in t0r])
            # Subjects with no template_rho data at all are skipped
            if not np.any(np.isfinite(w_arr)):
                continue
            # Replace NaN weights with 0 so unmatched events count as zero contribution
            w_arr = np.where(np.isfinite(w_arr), w_arr, 0.0)
            tasks.append((r['subject_id'], ep, list(w_arr)))
        else:
            tasks.append((r['subject_id'], ep))
    label = ''
    if args.q4: label = ' Q4'
    elif args.shape_weighted: label = ' shape-weighted'
    print(f"Cohort: {args.cohort} composite{label} · "
          f"subjects: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader.__name__, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(os.path.join(out_dir, out_csv_name), index=False)

    # Cleanup temp Q4 events files
    if q4_dir and os.path.isdir(q4_dir):
        for f in os.listdir(q4_dir):
            try:
                os.remove(os.path.join(q4_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(q4_dir)
        except Exception:
            pass
    print(f"Successful: {len(df)}")

    good = df.dropna(subset=['iaf_hz', 'beta_peak_hz', 'sr_peak_hz']).copy()
    print(f"Complete rows: {len(good)}")

    print(f"\n=== {args.cohort} composite · distributions ===")
    print(f"(envelope B28: IAF 9.69 mean; β peak 19.89 ± 2.51; SR peak 7.84)")
    for c in ['iaf_hz', 'sr_peak_hz', 'beta_peak_hz', 'beta_over_iaf']:
        v = good[c]
        print(f"  {c:18s}  mean {v.mean():.3f}  median {v.median():.3f}  SD {v.std():.3f}")

    rho_b, p_b = spearmanr(good['iaf_hz'], good['beta_peak_hz'])
    r_b, _ = pearsonr(good['iaf_hz'], good['beta_peak_hz'])
    slope_b, intercept_b = np.polyfit(good['iaf_hz'].values,
                                       good['beta_peak_hz'].values, 1)
    print(f"\n=== {args.cohort} composite · IAF × β_peak ===")
    print(f"(envelope B28: slope 0.14, H1 harmonic predicts 2.0, H2 fixed predicts 0)")
    print(f"  Spearman ρ = {rho_b:+.3f}  p = {p_b:.3g}")
    print(f"  Pearson  r = {r_b:+.3f}")
    print(f"  OLS: β_peak = {slope_b:+.3f} × IAF + {intercept_b:+.3f}")

    # Positive control: IAF × SR_peak (should be near-zero, §18 B20)
    rho_s, p_s = spearmanr(good['iaf_hz'], good['sr_peak_hz'])
    slope_s, _ = np.polyfit(good['iaf_hz'].values, good['sr_peak_hz'].values, 1)
    print(f"\n=== {args.cohort} composite · IAF × SR_peak (positive control) ===")
    print(f"  Spearman ρ = {rho_s:+.3f}  p = {p_s:.3g}   OLS slope = {slope_s:+.3f}")

    # SR amplitude × β amplitude (envelope B28: ρ = +0.55)
    rho_amp, p_amp = spearmanr(good['sr_peak_ratio'], good['beta_peak_ratio'])
    print(f"\n=== {args.cohort} composite · SR × β amplitude (subject-level) ===")
    print(f"(envelope B28: ρ = +0.55)")
    print(f"  Spearman ρ = {rho_amp:+.3f}  p = {p_amp:.3g}")

    # Save
    pd.DataFrame([{
        'cohort': args.cohort, 'n': len(good),
        'iaf_mean': float(good['iaf_hz'].mean()),
        'beta_peak_mean': float(good['beta_peak_hz'].mean()),
        'beta_peak_sd': float(good['beta_peak_hz'].std()),
        'sr_peak_mean': float(good['sr_peak_hz'].mean()),
        'spearman_IAFxBeta': float(rho_b), 'p_IAFxBeta': float(p_b),
        'OLS_slope_IAFxBeta': float(slope_b),
        'OLS_intercept_IAFxBeta': float(intercept_b),
        'spearman_IAFxSR': float(rho_s), 'OLS_slope_IAFxSR': float(slope_s),
        'spearman_SRxBeta_amp': float(rho_amp), 'p_SRxBeta_amp': float(p_amp),
    }]).to_csv(os.path.join(out_dir, 'beta_peak_iaf_summary.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(good['iaf_hz'], good['beta_peak_hz'], s=30, alpha=0.6,
                color='steelblue', edgecolor='k', lw=0.3)
    rng = np.array([good['iaf_hz'].min() - 0.5, good['iaf_hz'].max() + 0.5])
    ax.plot(rng, 2 * rng, 'r--', lw=1, label='H1: β = 2×IAF')
    ax.axhline(20.0, color='green', lw=1, ls=':', label='H2: fixed 20 Hz')
    ax.plot(rng, slope_b * rng + intercept_b, color='orange', lw=1.5,
             label=f'OLS: {slope_b:+.2f}×IAF + {intercept_b:+.2f}')
    ax.set_xlabel('IAF (Hz)')
    ax.set_ylabel('β event-locked peak (Hz)')
    ax.set_title(f'{args.cohort} composite: IAF × β-peak · ρ={rho_b:+.2f} p={p_b:.2g} n={len(good)}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(good['sr_peak_ratio'], good['beta_peak_ratio'], s=30, alpha=0.6,
                color='firebrick', edgecolor='k', lw=0.3)
    ax.set_xlabel('SR peak ratio (event / baseline)')
    ax.set_ylabel('β peak ratio')
    ax.set_title(f'SR × β amplitude · ρ={rho_amp:+.2f} p={p_amp:.2g}')
    ax.grid(alpha=0.3)

    plt.suptitle(f'B28 · β-peak fixed or 2×IAF · {args.cohort} composite v2',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'beta_peak_iaf_coupling.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/beta_peak_iaf_coupling.png")


if __name__ == '__main__':
    main()
