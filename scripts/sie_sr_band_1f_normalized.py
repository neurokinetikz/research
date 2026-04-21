#!/usr/bin/env python3
"""
B13 — SR-band event-boost after 1/f aperiodic normalization.

Confirms B12 is a narrowband-specific enhancement, not a broadband (1/f)
power shift. For each 4-s Welch window:

  1. Compute PSD on 2-20 Hz
  2. Fit a power-law aperiodic (log-linear: log P = a*log f + b) on
     [2-5] ∪ [9-20] Hz, excluding the 5-9 Hz theta/alpha region
  3. Find the peak in the SR band [7.0, 8.2] Hz
  4. Narrowband excess = log10(peak_power) − fit.predict(log10(peak_f))
     (log-ratio over aperiodic expectation)

Per subject:
  - All-window median of narrowband excess
  - Per-event narrowband excess
  - Event excess / all-time median excess
  - Per-subject summary + event-level CSV with template_rho quartile merge

Tests whether the B12 Q1 1.27× → Q4 1.97× gradient survives 1/f normalization.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import mannwhitneyu, wilcoxon
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
EVENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports_sie', 'lemon')
QUALITY_CSV = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                            'schumann', 'images', 'quality', 'per_event_quality.csv')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

WIN_SEC = 4.0
HOP_SEC = 1.0
FREQ_LO, FREQ_HI = 2.0, 20.0
APERIODIC_LO = [(2.0, 5.0), (9.0, 20.0)]   # exclude 5-9 Hz peak region
SR_LO, SR_HI = 7.0, 8.2
NFFT_MULT = 4


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
    return np.array(t_cent), freqs, np.array(cols).T  # (n_f, n_t)


def aperiodic_fit_mask(freqs, ranges):
    mask = np.zeros_like(freqs, dtype=bool)
    for lo, hi in ranges:
        mask |= (freqs >= lo) & (freqs <= hi)
    return mask


def per_window_sr_excess(freqs, P):
    """For each time column, compute:
       - peak freq & power in SR band
       - log10 of aperiodic at peak freq (via log-linear fit on aperiodic mask)
       - narrowband_excess = log10(peak_p) - log10(aperiodic_at_peak)
    Returns dict of arrays (peak_f, peak_p, excess)."""
    mask_ap = aperiodic_fit_mask(freqs, APERIODIC_LO)
    sr_mask = (freqs >= SR_LO) & (freqs <= SR_HI)
    idx_sr = np.where(sr_mask)[0]
    f_sr = freqs[idx_sr]
    logf = np.log10(freqs)
    logf_ap = logf[mask_ap]

    n_t = P.shape[1]
    peak_f = np.full(n_t, np.nan)
    peak_p = np.full(n_t, np.nan)
    excess = np.full(n_t, np.nan)   # log10 ratio
    for j in range(n_t):
        col = P[:, j]
        if not np.isfinite(col).any() or np.any(col <= 0):
            # skip columns with zero or nan entries
            pass
        # Peak in SR
        col_sr = col[idx_sr]
        if not np.isfinite(col_sr).any() or np.all(col_sr <= 0):
            continue
        k = int(np.argmax(col_sr))
        if 1 <= k < len(col_sr) - 1 and col_sr[k-1] > 0 and col_sr[k+1] > 0:
            y0, y1, y2 = col_sr[k-1], col_sr[k], col_sr[k+1]
            denom = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            delta = max(-1.0, min(1.0, delta))
            df = f_sr[1] - f_sr[0]
            f_k = f_sr[k] + delta * df
        else:
            f_k = f_sr[k]
        peak_f[j] = f_k
        peak_p[j] = col_sr[k]

        # Aperiodic fit (log-log linear) on mask
        logp_ap = np.log10(col[mask_ap] + 1e-20)
        good_ap = np.isfinite(logp_ap) & np.isfinite(logf_ap) & (logp_ap > -10)
        if good_ap.sum() < 8:
            continue
        # OLS: logp = a * logf + b
        A = np.column_stack([logf_ap[good_ap], np.ones(good_ap.sum())])
        try:
            coefs, *_ = np.linalg.lstsq(A, logp_ap[good_ap], rcond=None)
        except Exception:
            continue
        a, b = float(coefs[0]), float(coefs[1])
        log_ap_at_peak = a * np.log10(f_k) + b
        excess[j] = np.log10(col_sr[k] + 1e-20) - log_ap_at_peak
    return peak_f, peak_p, excess


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
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)

    t, freqs, P = sliding_welch(y, fs)
    peak_f, peak_p, excess = per_window_sr_excess(freqs, P)
    baseline_excess = float(np.nanmedian(excess))
    baseline_peak_p = float(np.nanmedian(peak_p))

    rows = []
    for _, ev in events.iterrows():
        te = float(ev['t0_net'])
        j = int(np.argmin(np.abs(t - te)))
        if 0 <= j < len(t) and np.isfinite(excess[j]):
            rows.append({
                'subject_id': sub_id,
                't0_net': te,
                'sr_peak_f': float(peak_f[j]),
                'sr_peak_p': float(peak_p[j]),
                'excess_log10': float(excess[j]),
                'excess_over_baseline_log10': float(excess[j] - baseline_excess),
                'raw_boost': float(peak_p[j] / baseline_peak_p) if baseline_peak_p > 0 else np.nan,
            })

    return {
        'subject_id': sub_id,
        'rows': rows,
        'baseline_excess_log10': baseline_excess,
        'baseline_peak_p': baseline_peak_p,
        'n_windows': int(len(t)),
    }


def main():
    summary = pd.read_csv(os.path.join(EVENTS_DIR, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    tasks = []
    for _, r in ok.iterrows():
        ep = os.path.join(EVENTS_DIR, f'{r["subject_id"]}_sie_events.csv')
        if os.path.isfile(ep):
            tasks.append((r['subject_id'], ep))
    print(f"Subjects: {len(tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, tasks)
    results = [r for r in results if r is not None]
    print(f"Successful subjects: {len(results)}")

    all_rows = []
    sub_rows = []
    for r in results:
        all_rows.extend(r['rows'])
        excesses = np.array([row['excess_over_baseline_log10'] for row in r['rows']])
        raws = np.array([row['raw_boost'] for row in r['rows']])
        sub_rows.append({
            'subject_id': r['subject_id'],
            'n_events': len(r['rows']),
            'baseline_excess_log10': r['baseline_excess_log10'],
            'event_excess_over_baseline_median_log10':
                float(np.nanmedian(excesses)) if len(excesses) else np.nan,
            'event_excess_over_baseline_median_x':
                float(10 ** np.nanmedian(excesses)) if len(excesses) else np.nan,
            'raw_boost_median': float(np.nanmedian(raws)) if len(raws) else np.nan,
        })
    ev = pd.DataFrame(all_rows)
    sub = pd.DataFrame(sub_rows)

    # Merge in template_rho quartile
    try:
        qual = pd.read_csv(QUALITY_CSV).dropna(subset=['template_rho']).copy()
        qual['rho_q'] = pd.qcut(qual['template_rho'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        ev['key'] = ev.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        qual['key'] = qual.apply(lambda r: f"{r['subject_id']}__{r['t0_net']:.3f}", axis=1)
        ev = ev.merge(qual[['key', 'template_rho', 'rho_q']], on='key', how='left')
        ev = ev.drop(columns=['key'])
        print(f"Events matched to quality CSV: {ev['template_rho'].notna().sum()}/{len(ev)}")
    except Exception as e:
        print(f"Quality merge failed: {e}")

    ev.to_csv(os.path.join(OUT_DIR, 'per_event_sr_1f_norm.csv'), index=False)
    sub.to_csv(os.path.join(OUT_DIR, 'per_subject_sr_1f_norm.csv'), index=False)

    print(f"\n=== Per-subject 1/f-normalized boost (event excess / baseline excess) ===")
    b_log = sub['event_excess_over_baseline_median_log10'].dropna()
    b_x   = sub['event_excess_over_baseline_median_x'].dropna()
    print(f"  log10: median {b_log.median():.3f}  IQR [{b_log.quantile(.25):.3f}, {b_log.quantile(.75):.3f}]")
    print(f"  ratio: median {b_x.median():.2f}×   IQR [{b_x.quantile(.25):.2f}, {b_x.quantile(.75):.2f}]")
    print(f"  pct subjects with boost >= 1.5×: {(b_x>=1.5).mean()*100:.1f}%")
    print(f"  pct subjects with boost <= 1.0×: {(b_x<=1.0).mean()*100:.1f}%")
    stat, pval = wilcoxon(b_log)
    print(f"  Wilcoxon per-subject excess vs 0 (log10): stat={stat:.1f} p={pval:.3g}")

    print(f"\n=== Raw (unnormalized) boost for reference ===")
    print(f"  median {sub['raw_boost_median'].median():.2f}×")

    if 'rho_q' in ev.columns:
        print(f"\n=== 1/f-normalized boost by template_rho quartile (pooled) ===")
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            sq = ev.dropna(subset=['rho_q'])
            sq = sq[sq['rho_q'] == q]['excess_over_baseline_log10']
            ratio = 10 ** sq
            print(f"  {q}: n={len(sq)}  median boost {ratio.median():.2f}×   "
                  f"IQR [{ratio.quantile(.25):.2f}, {ratio.quantile(.75):.2f}]")
        q1 = ev.dropna(subset=['rho_q'])
        q1 = q1[q1['rho_q'] == 'Q1']['excess_over_baseline_log10']
        q4 = ev.dropna(subset=['rho_q'])
        q4 = q4[q4['rho_q'] == 'Q4']['excess_over_baseline_log10']
        u, p = mannwhitneyu(q4, q1, alternative='greater')
        print(f"\n  MWU Q4 > Q1 (log-excess) one-sided: U={u:.0f}  p={p:.3g}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(sub['event_excess_over_baseline_median_x'].dropna(),
             bins=np.linspace(0, 6, 40), color='seagreen',
             edgecolor='k', lw=0.3, alpha=0.85)
    ax.axvline(1.0, color='k', lw=0.8)
    ax.axvline(sub['event_excess_over_baseline_median_x'].median(),
                color='firebrick', ls='--', lw=1.5,
                label=f"median {sub['event_excess_over_baseline_median_x'].median():.2f}×")
    ax.set_xlabel('per-subject 1/f-normalized event boost (×)')
    ax.set_ylabel('subjects')
    ax.set_title(f'1/f-normalized (n={len(sub)})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Compare raw vs 1/f-normalized
    ax = axes[1]
    bins = np.linspace(0, 6, 40)
    ax.hist(sub['raw_boost_median'].dropna(), bins=bins, color='gray',
             alpha=0.55, label=f"raw boost median {sub['raw_boost_median'].median():.2f}×")
    ax.hist(sub['event_excess_over_baseline_median_x'].dropna(), bins=bins,
             color='seagreen', alpha=0.55,
             label=f"1/f-norm median {sub['event_excess_over_baseline_median_x'].median():.2f}×")
    ax.axvline(1.0, color='k', lw=0.8)
    ax.set_xlabel('per-subject boost (×)')
    ax.set_ylabel('subjects')
    ax.set_title('Raw vs 1/f-normalized')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Q1 vs Q4 for 1/f-normalized
    ax = axes[2]
    if 'rho_q' in ev.columns:
        bins = np.linspace(0, 6, 40)
        ev_q = ev.dropna(subset=['rho_q', 'excess_over_baseline_log10'])
        for q, color in [('Q1', '#4575b4'), ('Q4', '#d73027')]:
            sq = ev_q[ev_q['rho_q'] == q]
            vals = (10 ** sq['excess_over_baseline_log10']).values
            ax.hist(vals, bins=bins, color=color, alpha=0.55,
                    label=f'{q} median {np.median(vals):.2f}× (n={len(vals)})')
        ax.axvline(1.0, color='k', lw=0.6)
        ax.set_xlabel('per-event 1/f-normalized boost (×)')
        ax.set_ylabel('events')
        ax.set_title('Boost by template_rho')
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(f'B13 — SR-band event-boost · 1/f-aperiodic normalized',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sr_band_1f_normalized.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/sr_band_1f_normalized.png")


if __name__ == '__main__':
    main()
