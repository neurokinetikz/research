#!/usr/bin/env python3
"""
B26 — Population-aggregate 1/f-corrected PSD across all three cohorts.

Does the 7.83 Hz attractor show up as a standing spectral bump in the
population average, separate from the individual alpha peak?

For each of LEMON (N=192), HBN R4 (N=219), TDBRAIN (N=51):
  1. Per subject, compute aggregate PSD (8-s Welch on mean-channel, full
     recording, median across windows) on [2, 25] Hz.
  2. 1/f aperiodic subtraction in log-space (fit log-linear on 2-5 ∪ 9-20 Hz).
  3. Log-space residual = log10(PSD) − log10(aperiodic fit). Dimensionless.

Grand-average the log-space residuals across subjects per cohort, and across
all 405 subjects pooled. Subject-level bootstrap 95% CI.

Check for standing bumps at:
  - 7.83 Hz  (Schumann fundamental)
  - 7.60 Hz  (φ-lattice θ-α boundary)
  - 14.3 Hz  (Schumann 2nd harmonic)
  - 20.8 Hz  (Schumann 3rd harmonic)
  - IAF (subject's alpha peak — should dominate)
"""
from __future__ import annotations
import argparse
import glob as globfn
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from multiprocessing import Pool
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.run_sie_extraction import load_lemon, load_hbn, load_tdbrain

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

AGG_WIN_SEC = 8.0
AGG_HOP_SEC = 2.0
AGG_NFFT_MULT = 4
FREQ_LO, FREQ_HI = 2.0, 25.0
APERIODIC_RANGES = [(2.0, 5.0), (9.0, 22.0)]  # exclude 5-9 Hz peak region


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def locate_hbn_set(sub_id, release):
    pat = f'/Volumes/T9/hbn_data/cmi_bids_{release}/{sub_id}/eeg/*RestingState_eeg.set'
    files = sorted(globfn.glob(pat))
    return files[0] if files else None


def load_recording(sub_id, dataset, release=None):
    if dataset == 'lemon':
        return load_lemon(sub_id, condition='EC')
    if dataset == 'hbn':
        set_path = locate_hbn_set(sub_id, release)
        if not set_path:
            return None
        return load_hbn(set_path)
    if dataset == 'tdbrain':
        return load_tdbrain(sub_id, condition='EC')
    raise ValueError(f"Unknown dataset: {dataset}")


def aperiodic_log_resid(freqs, psd):
    """Return log10(psd) − log10(fitted 1/f) on the full grid."""
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
    return logp - (a * logf + b)   # log10-residual


def subject_aggregate(y, fs):
    nperseg = int(round(AGG_WIN_SEC * fs))
    nhop = int(round(AGG_HOP_SEC * fs))
    nfft = nperseg * AGG_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    psds = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psds.append(welch_one(y[i:i + nperseg], fs, nfft)[mask])
    if len(psds) < 5:
        return None, None
    return freqs[mask], np.nanmedian(np.array(psds), axis=0)


def process_subject(args):
    sub_id, dataset, release = args
    try:
        raw = load_recording(sub_id, dataset, release)
    except Exception:
        return None
    if raw is None:
        return None
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    y = X.mean(axis=0)
    freqs, agg = subject_aggregate(y, fs)
    if freqs is None:
        return None
    log_resid = aperiodic_log_resid(freqs, agg)
    return {
        'subject_id': sub_id,
        'dataset': dataset,
        'freqs': freqs,
        'log_resid': log_resid,
    }


def bootstrap_ci(mat, n_boot=1000, seed=0):
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


def gather_subjects(dataset, release=None):
    if dataset == 'hbn':
        events_dir = os.path.join(EVENTS_BASE, f'hbn_{release}')
    elif dataset == 'tdbrain':
        events_dir = os.path.join(EVENTS_BASE, 'tdbrain')
    else:
        events_dir = os.path.join(EVENTS_BASE, 'lemon')
    summary = pd.read_csv(os.path.join(events_dir, 'extraction_summary.csv'))
    ok = summary[(summary['status'] == 'ok') & (summary['n_events'] >= 3)]
    return [(r['subject_id'], dataset, release) for _, r in ok.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='lemon,hbn,tdbrain',
                     help='comma-separated')
    ap.add_argument('--hbn-release', default='R4')
    args = ap.parse_args()

    datasets = args.datasets.split(',')
    all_tasks = []
    for ds in datasets:
        rel = args.hbn_release if ds == 'hbn' else None
        all_tasks.extend(gather_subjects(ds, rel))
    print(f"Total subjects: {len(all_tasks)}")
    n_workers = int(os.environ.get('SIE_WORKERS', min(4, os.cpu_count() or 4)))
    print(f"Workers: {n_workers}")

    with Pool(n_workers) as pool:
        results = pool.map(process_subject, all_tasks)
    results = [r for r in results if r is not None]
    print(f"Successful: {len(results)}")

    # Interpolate all residuals onto a common grid
    common = np.arange(FREQ_LO, FREQ_HI + 0.05, 0.05)
    per_dataset = {}
    for r in results:
        ds = r['dataset']
        per_dataset.setdefault(ds, []).append(
            np.interp(common, r['freqs'], r['log_resid']))

    all_mat = np.vstack([np.array(v) for v in per_dataset.values()])
    print(f"\nAll cohorts pooled: n_subjects = {len(all_mat)}, grid {len(common)} freqs")

    # Peak detection on grand mean
    def peaks_near(mean_curve, freqs, targets, half_window=0.30):
        out = {}
        for t in targets:
            m = (freqs >= t - half_window) & (freqs <= t + half_window)
            if np.any(mean_curve[m] > 0):
                idx = np.where(m)[0]
                k = idx[int(np.argmax(mean_curve[m]))]
                out[t] = {'peak_f': float(freqs[k]), 'peak_val': float(mean_curve[k])}
            else:
                out[t] = {'peak_f': np.nan, 'peak_val': np.nan}
        return out

    targets = [7.60, 7.83, 14.3, 20.8]
    print(f"\n=== Per-cohort grand means ===")
    grand_per_ds = {}
    for ds, arr in per_dataset.items():
        mat = np.array(arr)
        grand, lo, hi = bootstrap_ci(mat)
        grand_per_ds[ds] = (grand, lo, hi, mat)
        pks = peaks_near(grand, common, targets)
        iaf_mask = (common >= 7) & (common <= 13)
        iaf_idx = int(np.argmax(grand[iaf_mask]))
        iaf_f = float(common[iaf_mask][iaf_idx])
        iaf_v = float(grand[iaf_mask][iaf_idx])
        print(f"  {ds}  n={len(mat)}  IAF-like peak @ {iaf_f:.2f} Hz (v={iaf_v:.3f})")
        for t, d in pks.items():
            print(f"    near {t:.2f} Hz: peak {d['peak_f']:.2f} (log-resid {d['peak_val']:+.3f})")

    # Pooled grand mean
    grand_all, lo_all, hi_all = bootstrap_ci(all_mat)
    print(f"\n=== All-cohort pooled (n={len(all_mat)}) ===")
    pks = peaks_near(grand_all, common, targets)
    iaf_mask = (common >= 7) & (common <= 13)
    iaf_idx = int(np.argmax(grand_all[iaf_mask]))
    iaf_f = float(common[iaf_mask][iaf_idx])
    iaf_v = float(grand_all[iaf_mask][iaf_idx])
    print(f"  IAF-like peak @ {iaf_f:.2f} Hz (v={iaf_v:.3f})")
    for t, d in pks.items():
        print(f"    near {t:.2f} Hz: peak {d['peak_f']:.2f} (log-resid {d['peak_val']:+.3f})")

    # Save CSVs
    pd.DataFrame({'freq_hz': common, 'log_resid_grand_mean': grand_all,
                   'ci_lo': lo_all, 'ci_hi': hi_all}).to_csv(
        os.path.join(OUT_DIR, 'population_aggregate_psd_pooled.csv'), index=False)
    rows = []
    for ds, (grand, lo, hi, mat) in grand_per_ds.items():
        for i, f in enumerate(common):
            rows.append({'dataset': ds, 'freq_hz': f,
                          'log_resid_mean': grand[i],
                          'ci_lo': lo[i], 'ci_hi': hi[i],
                          'n_subjects': len(mat)})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, 'population_aggregate_psd_per_dataset.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # Panel 1: per-dataset
    ax = axes[0]
    colors = {'lemon': '#2166ac', 'hbn': '#d7301f', 'tdbrain': '#2d8659'}
    for ds, (grand, lo, hi, mat) in grand_per_ds.items():
        ax.plot(common, grand, color=colors.get(ds, 'k'), lw=1.6,
                label=f'{ds.upper()} (n={len(mat)})')
        ax.fill_between(common, lo, hi, color=colors.get(ds, 'k'), alpha=0.2)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(7.83, color='green', ls='--', lw=0.7, alpha=0.7,
                label='Schumann 7.83 Hz')
    ax.axvline(7.60, color='gray', ls=':', lw=0.7, alpha=0.7,
                label='φ-boundary 7.60')
    for h in [14.3, 20.8]:
        ax.axvline(h, color='green', ls=':', lw=0.5, alpha=0.5)
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_ylabel('log10 residual (aperiodic-corrected)')
    ax.set_title('Per-cohort population-aggregate 1/f-corrected PSD')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # Panel 2: pooled all
    ax = axes[1]
    ax.plot(common, grand_all, color='k', lw=2, label=f'All pooled (n={len(all_mat)})')
    ax.fill_between(common, lo_all, hi_all, color='gray', alpha=0.3)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(7.83, color='green', ls='--', lw=1, alpha=0.7,
                label='Schumann 7.83 Hz')
    ax.axvline(7.60, color='gray', ls=':', lw=0.7, alpha=0.7,
                label='φ-boundary 7.60 Hz')
    for h in [14.3, 20.8]:
        ax.axvline(h, color='green', ls=':', lw=0.5, alpha=0.5)
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('log10 residual (aperiodic-corrected)')
    ax.set_title(f'Pooled all-cohort aggregate ({len(all_mat)} subjects)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # Zoom inset into 6.5-9 Hz on pooled panel
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_in = inset_axes(ax, width='35%', height='45%', loc='upper left',
                         bbox_to_anchor=(0.05, 0.0, 1, 1),
                         bbox_transform=ax.transAxes)
    zoom = (common >= 6.5) & (common <= 9.0)
    ax_in.plot(common[zoom], grand_all[zoom], color='k', lw=1.5)
    ax_in.fill_between(common[zoom], lo_all[zoom], hi_all[zoom],
                        color='gray', alpha=0.3)
    ax_in.axvline(7.83, color='green', ls='--', lw=0.7)
    ax_in.axvline(7.60, color='gray', ls=':', lw=0.6)
    ax_in.axhline(0, color='k', lw=0.4)
    ax_in.set_title('zoom 6.5-9 Hz', fontsize=8)
    ax_in.grid(alpha=0.3)

    plt.suptitle('B26 — Population-aggregate 1/f-corrected PSD',
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'population_aggregate_psd.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUT_DIR}/population_aggregate_psd.png")


if __name__ == '__main__':
    main()
