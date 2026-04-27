#!/usr/bin/env python3
"""
B26-B27 random-window null: per-subject log10(random/baseline) ratio.

Companion to sie_event_locked_aggregate_psd.py. For each subject:
  1. Load the per-event CSV (event timestamps).
  2. Generate matched random window onsets:
     - Same count as detected events
     - >= 20 s away from any detected event onset
     - >= 12 s from recording edges
  3. Compute per-subject random-window PSD = mean over 4-s Welch windows
     centered at random_onset + 1 s (matching the event-locked +1 s lag in B27).
  4. Compute per-subject baseline PSD = median over all sliding 4-s windows.
  5. Compute log10(random/baseline) ratio spectrum (wide band [2, 25] Hz).

Pool across subjects, with subject-level bootstrap 95% CI. The hypothesis
the SIE paper makes is that the SR1 peak appears in event-locked aggregates
(B27, panel B of aggregate_psd_B26_B27.csv) but is ABSENT in random-window
aggregates and standing aggregates (B26, panel A). This script computes
panel C (random-window aggregate) for the three-panel comparison figure.

Output:
  outputs/schumann/images/psd_timelapse/lemon_composite/random_window_null_psd.csv
"""
from __future__ import annotations
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
from scripts.run_sie_extraction import load_lemon

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'schumann',
                        'images', 'psd_timelapse')
EVENTS_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_sie')
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

EV_WIN_SEC = 4.0
EV_NFFT_MULT = 16
EV_LAG_S = 1.0
FREQ_LO, FREQ_HI = 2.0, 25.0
RAND_MIN_GAP_S = 20.0  # min distance from any real event
RAND_EDGE_S = 12.0     # min distance from recording edges
RAND_SEED_BASE = 42


def welch_one(seg, fs, nfft):
    seg = seg - np.mean(seg)
    win = signal.windows.hann(len(seg))
    win_pow = np.sum(win ** 2)
    X = np.fft.rfft(seg * win, nfft)
    psd = (np.abs(X) ** 2) / (fs * win_pow)
    psd[1:-1] *= 2.0
    return psd


def make_random_onsets(real_t0s, recording_dur_s, n_random, rng):
    """Random onsets matched in count, ≥20 s from real events, ≥12 s from edges."""
    real_t0s = np.sort(np.array(real_t0s, dtype=float))
    candidates = []
    max_iter = n_random * 200
    iters = 0
    edge_lo = RAND_EDGE_S
    edge_hi = recording_dur_s - RAND_EDGE_S
    if edge_hi <= edge_lo:
        return np.array([])
    while len(candidates) < n_random and iters < max_iter:
        t = rng.uniform(edge_lo, edge_hi)
        iters += 1
        if real_t0s.size == 0:
            candidates.append(t)
            continue
        # Reject if within RAND_MIN_GAP of any real event
        if np.min(np.abs(real_t0s - t)) >= RAND_MIN_GAP_S:
            candidates.append(t)
    return np.array(candidates)


def process_subject(args):
    sub_id, events_path, seed = args
    try:
        events = pd.read_csv(events_path).dropna(subset=['t0_net'])
    except Exception:
        return None
    if len(events) < 1:
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
    rec_dur = len(y) / fs

    nperseg = int(round(EV_WIN_SEC * fs))
    nhop = int(round(1.0 * fs))
    nfft = nperseg * EV_NFFT_MULT
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    f_band = freqs[mask]

    # Baseline (all sliding windows)
    base_rows = []
    for i in range(0, len(y) - nperseg + 1, nhop):
        psd = welch_one(y[i:i + nperseg], fs, nfft)[mask]
        base_rows.append(psd)
    if len(base_rows) < 10:
        return None
    baseline = np.nanmedian(np.array(base_rows), axis=0)

    # Random window onsets
    rng = np.random.default_rng(seed)
    real_t0s = events['t0_net'].values
    rand_t0s = make_random_onsets(real_t0s, rec_dur, len(events), rng)
    if len(rand_t0s) == 0:
        return None

    rand_rows = []
    for tc_base in rand_t0s:
        tc = tc_base + EV_LAG_S
        i0 = int(round((tc - EV_WIN_SEC / 2) * fs))
        i1 = i0 + nperseg
        if i0 < 0 or i1 > len(y):
            continue
        psd = welch_one(y[i0:i1], fs, nfft)[mask]
        rand_rows.append(psd)
    if not rand_rows:
        return None
    rand_avg = np.nanmean(np.array(rand_rows), axis=0)

    log_ratio = np.log10(rand_avg / baseline)
    return (sub_id, f_band, log_ratio, len(rand_rows))


def main():
    """Run on all available LEMON composite events."""
    lemon_dir = os.path.join(EVENTS_BASE, 'lemon_composite')
    csvs = sorted(globfn.glob(os.path.join(lemon_dir, 'sub-*_sie_events.csv')))
    args_list = []
    for i, csv_path in enumerate(csvs):
        sub_id = os.path.basename(csv_path).replace('_sie_events.csv', '')
        args_list.append((sub_id, csv_path, RAND_SEED_BASE + i))

    # For local pilot, restrict to 50 subjects
    n_max = int(os.environ.get('SIE_RAND_N_MAX', '50'))
    args_list = args_list[:n_max]
    print(f"Processing {len(args_list)} LEMON EC composite subjects (random-window null)")

    n_workers = int(os.environ.get('SIE_RAND_WORKERS', '4'))
    results = []
    with Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_subject, args_list)):
            if res is not None:
                results.append(res)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(args_list)} subjects processed; {len(results)} valid")

    if not results:
        print("No valid results.")
        return

    sub_ids, f_bands, ratios, n_rands = zip(*results)
    f_band = f_bands[0]
    R = np.array(ratios)
    print(f"Pooled {len(results)} subjects, mean random windows per subject: {np.mean(n_rands):.1f}")

    # Grand mean + bootstrap CI
    grand_mean = np.nanmean(R, axis=0)
    n_boot = 1000
    rng = np.random.default_rng(2026)
    boot_means = np.zeros((n_boot, R.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, R.shape[0], R.shape[0])
        boot_means[b] = np.nanmean(R[idx], axis=0)
    ci_lo = np.nanpercentile(boot_means, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_means, 97.5, axis=0)

    # Save CSV
    out_dir = os.path.join(OUT_DIR, 'lemon_composite')
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, 'random_window_null_psd.csv')
    df = pd.DataFrame({
        'freq_hz': f_band,
        'panel_C_random_window_log_ratio_grand': grand_mean,
        'panel_C_lo': ci_lo,
        'panel_C_hi': ci_hi,
    })
    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")
    print(f"  N subjects: {len(results)}")
    print(f"  Peak in 6-9 Hz range: {f_band[6 <= f_band].min():.2f} Hz")
    sr_band = (f_band >= 7.0) & (f_band <= 8.5)
    if sr_band.any():
        peak_idx = np.argmax(grand_mean[sr_band])
        sr_freqs = f_band[sr_band]
        print(f"  Random-window log-ratio peak in SR band: {sr_freqs[peak_idx]:.2f} Hz, "
              f"value = {grand_mean[sr_band][peak_idx]:.4f}")


if __name__ == "__main__":
    main()
