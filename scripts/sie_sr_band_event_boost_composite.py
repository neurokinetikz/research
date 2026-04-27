#!/usr/bin/env python3
"""
B11-B12 re-run on composite v2 detector.

Identical analysis to scripts/sie_sr_band_event_boost.py (the envelope-detector
cohort run), but reads events from exports_sie/<cohort>_composite/ and uses the
composite-extraction template_rho at
outputs/schumann/images/quality/per_event_quality_<cohort>_composite.csv.

Per subject and event:
  1. Sliding Welch (4-s window, 1-s hop) on the mean-channel signal
  2. SR-band [7.0, 8.2] Hz peak power per window
  3. Event boost = peak power at event-window / all-window median peak power
  4. Broadband [6.0, 9.0] Hz peak → % time peak in SR

Outputs go to outputs/schumann/images/psd_timelapse/<cohort>_composite/ so they
sit alongside (not overwrite) the envelope baseline CSVs.

Usage:
    python scripts/sie_sr_band_event_boost_composite.py --cohort lemon
    python scripts/sie_sr_band_event_boost_composite.py --cohort lemon_EO
    python scripts/sie_sr_band_event_boost_composite.py --cohort tdbrain
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import wilcoxon, mannwhitneyu
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

WIN_SEC = 4.0
HOP_SEC = 1.0
FREQ_LO, FREQ_HI = 4.0, 9.0
BROAD_LO, BROAD_HI = 6.0, 9.0
SR_LO, SR_HI = 7.0, 8.2
NFFT_MULT = 4

ROOT = os.path.join(os.path.dirname(__file__), '..')


# Cohort → (loader, extra loader kwargs, events_dir, quality_csv)
def cohort_config(cohort):
    base = os.path.join(ROOT, 'exports_sie', f'{cohort}_composite')
    qual = os.path.join(ROOT, 'outputs', 'schumann', 'images', 'quality',
                        f'per_event_quality_{cohort}_composite.csv')
    if cohort == 'lemon':
        return load_lemon, {'condition': 'EC'}, base, qual
    if cohort == 'lemon_EO':
        return load_lemon, {'condition': 'EO'}, base, qual
    if cohort == 'tdbrain':
        return load_tdbrain, {'condition': 'EC'}, base, qual
    if cohort == 'tdbrain_EO':
        return load_tdbrain, {'condition': 'EO'}, base, qual
    if cohort == 'srm':
        return load_srm, {}, base, qual
    if cohort.startswith('dortmund_'):
        return load_dortmund, {}, base, qual
    if cohort == 'dortmund':
        return load_dortmund, {}, base, qual
    if cohort == 'chbmp':
        return load_chbmp, {}, base, qual
    if cohort.startswith('hbn_'):
        return load_hbn_by_subject, {'release': cohort.split('_', 1)[1]}, base, qual
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
    t_cent, psd_cols = [], []
    for i in range(0, len(x) - nperseg + 1, nhop):
        seg = x[i:i + nperseg] - np.mean(x[i:i + nperseg])
        X = np.fft.rfft(seg * win, nfft)
        psd = (np.abs(X) ** 2) / (fs * win_pow)
        psd[1:-1] *= 2.0
        psd_cols.append(psd[f_mask])
        t_cent.append((i + nperseg / 2) / fs)
    return np.array(t_cent), freqs, np.array(psd_cols).T


def band_peak(freqs, P, f_lo, f_hi):
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    idx_band = np.where(mask)[0]
    f_band = freqs[idx_band]
    peaks = np.full(P.shape[1], np.nan)
    peak_pow = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        col = P[idx_band, j]
        if not np.isfinite(col).any() or np.all(col == 0):
            continue
        k = int(np.argmax(col))
        if 1 <= k < len(col) - 1:
            y0, y1, y2 = col[k - 1], col[k], col[k + 1]
            denom = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            delta = max(-1.0, min(1.0, delta))
            f_k = f_band[k] + delta * (f_band[1] - f_band[0])
        else:
            f_k = f_band[k]
        peaks[j] = f_k
        peak_pow[j] = col[k]
    return peaks, peak_pow


_LOADER = None
_LOADER_KW = None


def _init_worker(loader_name, loader_kw):
    global _LOADER, _LOADER_KW
    _LOADER_KW = loader_kw
    if loader_name == 'load_lemon':
        _LOADER = load_lemon
    elif loader_name == 'load_tdbrain':
        _LOADER = load_tdbrain
    elif loader_name == 'load_srm':
        _LOADER = load_srm
    elif loader_name == 'load_dortmund':
        _LOADER = load_dortmund
    elif loader_name == 'load_chbmp':
        _LOADER = load_chbmp
    elif loader_name == 'load_hbn_by_subject':
        _LOADER = load_hbn_by_subject
    else:
        raise ValueError(loader_name)


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
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    t, freqs, P = sliding_welch(y, fs)
    peak_f_broad, _ = band_peak(freqs, P, BROAD_LO, BROAD_HI)
    in_sr = (peak_f_broad >= SR_LO) & (peak_f_broad <= SR_HI)
    pct_peak_in_sr = float(np.nanmean(in_sr) * 100)

    peak_f_sr, peak_p_sr = band_peak(freqs, P, SR_LO, SR_HI)
    baseline_median = float(np.nanmedian(peak_p_sr))

    rows = []
    for _, ev in events.iterrows():
        te = float(ev['t0_net'])
        j = int(np.argmin(np.abs(t - te)))
        if 0 <= j < len(t):
            boost = (peak_p_sr[j] / baseline_median) if baseline_median > 0 else np.nan
            rows.append({
                'subject_id': sub_id,
                't0_net': te,
                'sr_peak_f': float(peak_f_sr[j]),
                'sr_peak_p': float(peak_p_sr[j]),
                'broad_peak_f': float(peak_f_broad[j]),
                'broad_in_sr': bool(in_sr[j]),
                'sr_boost': float(boost),
            })

    return {
        'subject_id': sub_id,
        'rows': rows,
        'pct_peak_in_sr': pct_peak_in_sr,
        'sr_baseline_median_power': baseline_median,
        'n_windows': int(len(t)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='lemon',
                    help='composite cohort name (e.g. lemon, lemon_EO, tdbrain)')
    ap.add_argument('--min-events', type=int, default=3,
                    help='min events/subject (matches envelope run default)')
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SIE_WORKERS',
                                                min(4, os.cpu_count() or 4))))
    args = ap.parse_args()

    loader, loader_kw, events_dir, quality_csv = cohort_config(args.cohort)
    out_dir = os.path.join(ROOT, 'outputs', 'schumann', 'images',
                            'psd_timelapse', f'{args.cohort}_composite')
    os.makedirs(out_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') &
                 (summary['n_events'] >= args.min_events)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(events_dir, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Cohort: {args.cohort} (composite v2)")
    print(f"Subjects with >= {args.min_events} events: {len(tasks)}")
    print(f"Workers: {args.workers}")

    loader_name = loader.__name__
    with Pool(args.workers, initializer=_init_worker,
              initargs=(loader_name, loader_kw)) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    all_rows = []
    sub_rows = []
    for r in results:
        all_rows.extend(r['rows'])
        boosts = np.array([row['sr_boost'] for row in r['rows']
                            if np.isfinite(row['sr_boost'])])
        sub_rows.append({
            'subject_id': r['subject_id'],
            'n_events': len(r['rows']),
            'pct_peak_in_sr': r['pct_peak_in_sr'],
            'sr_baseline_median_power': r['sr_baseline_median_power'],
            'event_boost_median': float(np.median(boosts)) if len(boosts) else np.nan,
            'event_boost_mean':   float(np.mean(boosts)) if len(boosts) else np.nan,
        })
    ev = pd.DataFrame(all_rows)
    sub = pd.DataFrame(sub_rows)

    try:
        qual = pd.read_csv(quality_csv)
        qual = qual.dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        ev['key'] = ev.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        qual['key'] = qual.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        ev = ev.merge(qual[['key', 'template_rho', 'rho_q']], on='key', how='left')
        ev = ev.drop(columns=['key'])
        print(f"Events matched to quality CSV: "
              f"{ev['template_rho'].notna().sum()}/{len(ev)}")
    except Exception as e:
        print(f"Quality merge failed: {e}")

    ev.to_csv(os.path.join(out_dir, 'per_event_sr_boost.csv'), index=False)
    sub.to_csv(os.path.join(out_dir, 'per_subject_sr_summary.csv'), index=False)

    print(f"\n=== {args.cohort} composite — % time broadband peak in SR [7.0, 8.2] Hz ===")
    vals = sub['pct_peak_in_sr'].dropna()
    print(f"  median {vals.median():.1f}%  IQR [{vals.quantile(.25):.1f}, {vals.quantile(.75):.1f}]")
    print(f"  pct of subjects with <25% time in SR: {(vals<25).mean()*100:.1f}%")

    print(f"\n=== {args.cohort} composite — per-subject event boost ===")
    b = sub['event_boost_median'].dropna()
    print(f"  median {b.median():.2f}×   IQR [{b.quantile(.25):.2f}, {b.quantile(.75):.2f}]")
    print(f"  pct of subjects with boost >= 1.5: {(b>=1.5).mean()*100:.1f}%")
    print(f"  pct of subjects with boost <= 1.0: {(b<=1.0).mean()*100:.1f}%")

    stat, p = wilcoxon(b - 1.0)
    print(f"  Wilcoxon event-boost-median vs 1.0: stat={stat:.1f}, p={p:.3g}")

    if 'rho_q' in ev.columns and ev['rho_q'].notna().any():
        ev_q = ev.dropna(subset=['rho_q', 'sr_boost'])
        print(f"\n=== {args.cohort} composite — by template_rho quartile (pooled) ===")
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            sub_q = ev_q[ev_q['rho_q'] == q]['sr_boost']
            print(f"  {q}: n={len(sub_q)}  median {sub_q.median():.2f}×  "
                  f"IQR [{sub_q.quantile(.25):.2f}, {sub_q.quantile(.75):.2f}]")
        q1_vals = ev_q[ev_q['rho_q'] == 'Q1']['sr_boost']
        q4_vals = ev_q[ev_q['rho_q'] == 'Q4']['sr_boost']
        u, pval = mannwhitneyu(q4_vals, q1_vals, alternative='greater')
        print(f"\n  MWU Q4 > Q1 one-sided: U={u:.0f}  p={pval:.3g}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(sub['pct_peak_in_sr'].dropna(), bins=30, color='steelblue',
             edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(sub['pct_peak_in_sr'].median(), color='firebrick', ls='--',
                lw=1.5, label=f"median {sub['pct_peak_in_sr'].median():.1f}%")
    ax.set_xlabel('% time broadband peak is inside [7.0, 8.2] Hz')
    ax.set_ylabel('subjects')
    ax.set_title(f'{args.cohort} composite · n={len(sub)}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(sub['event_boost_median'].dropna(),
             bins=np.linspace(0, 6, 40),
             color='seagreen', edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(1.0, color='k', ls='-', lw=0.8, label='no boost')
    ax.axvline(sub['event_boost_median'].median(), color='firebrick', ls='--',
                lw=1.5, label=f"median {sub['event_boost_median'].median():.2f}×")
    ax.set_xlabel('per-subject median SR-band event boost')
    ax.set_ylabel('subjects')
    ax.set_title('Event boost (cohort)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if 'rho_q' in ev.columns and ev['rho_q'].notna().any():
        ev_q = ev.dropna(subset=['rho_q', 'sr_boost'])
        bins = np.linspace(0, 6, 40)
        for q, color in [('Q1', '#4575b4'), ('Q4', '#d73027')]:
            vals = ev_q[ev_q['rho_q'] == q]['sr_boost'].values
            ax.hist(vals, bins=bins, color=color, alpha=0.55,
                    label=f'{q} median {np.median(vals):.2f}× (n={len(vals)})')
        ax.axvline(1.0, color='k', lw=0.6)
        ax.set_xlabel('per-event SR-band boost')
        ax.set_ylabel('events')
        ax.set_title('Boost by template_rho quartile')
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(f'B12 · SR-band [7.0, 8.2 Hz] event boost · {args.cohort} composite v2',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sr_band_event_boost.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir}/sr_band_event_boost.png")
    print(f"Saved: {out_dir}/per_event_sr_boost.csv")
    print(f"Saved: {out_dir}/per_subject_sr_summary.csv")


if __name__ == '__main__':
    main()
