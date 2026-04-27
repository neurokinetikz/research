#!/usr/bin/env python3
"""
Event-Locked IRASA Centroid on LEMON EC (Supplementary Table S2)
=================================================================

Tests whether the 0.14 Hz centroid-vs-cavity gap (FOOOF event-locked
centroid 7.687 Hz vs SR1 = 7.83 Hz) is methodological FOOOF leading-edge
bias or a residual neurophysiological offset.

For each LEMON EC subject with ≥3 detected SIE events under composite-v2:
1. Load raw EEG (BrainVision .vhdr).
2. Preprocess: HP 1 Hz, notch 50 Hz, downsample 250 Hz.
3. For each event, extract a 4-s window centered on t0_net.
4. Compute IRASA (lib/shape_vs_resonance.irasa_psd) on the posterior-
   channel-mean signal (O/PO/P/Oz channels).
5. Find the peak frequency of P_osc in the 7.0–8.2 Hz SR1 envelope window.
6. Aggregate: per-subject median (SW-weighted), then cohort-mean across
   subjects.

Compares the IRASA centroid to the FOOOF baseline of 7.687 Hz.
Expected outcome under the FOOOF-leading-edge-bias account: IRASA centroid
shifts upward toward 7.83 Hz.

Usage:
    /opt/anaconda3/envs/brainwaves/bin/python3 \
        scripts/sie_event_locked_irasa_centroid_lemon_ec.py [--subset N]

Outputs:
    outputs/schumann/2026-04-27-irasa-centroid-lemon-ec/
        - per_event_centroids.csv  (one row per event)
        - per_subject_centroids.csv  (one row per subject, SW-weighted median)
        - cohort_summary.txt  (cohort-mean centroid, comparison to FOOOF)
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import mne

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))
from shape_vs_resonance import irasa_psd  # noqa: E402

LEMON_RAW = Path("/Volumes/T9/lemon_data/eeg_raw/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID")
LEMON_EVENTS = BASE_DIR / "exports_sie" / "lemon_composite"
OUT_DIR = BASE_DIR / "outputs" / "schumann" / "2026-04-27-irasa-centroid-lemon-ec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Posterior channels expected in LEMON 59-channel montage
POSTERIOR_CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "POz", "PO7", "PO8",
                      "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "Pz"]

# Event-locked window (centered on event onset)
WINDOW_SEC = 4.0
HALF_WIN = WINDOW_SEC / 2

# SR1 envelope band for centroid extraction
SR1_BAND_LO, SR1_BAND_HI = 7.0, 8.2
FOOOF_BASELINE_CENTROID = 7.687  # cohort-mean centroid under FOOOF
SR1_CAVITY = 7.83


def list_lemon_ec_subjects():
    """Return list of LEMON EC subjects with ≥3 events from composite-v2 SIE CSVs."""
    csvs = sorted(LEMON_EVENTS.glob("sub-*_sie_events.csv"))
    out = []
    for path in csvs:
        sid = path.stem.replace("_sie_events", "")
        df = pd.read_csv(path)
        if "condition" not in df.columns:
            continue
        ec = df[df["condition"] == "EC"]
        if len(ec) >= 3:
            out.append((sid, ec))
    return out


def load_raw_eeg(sid):
    """Load LEMON raw BrainVision EEG; return MNE Raw or None."""
    vhdr = LEMON_RAW / sid / "RSEEG" / f"{sid}.vhdr"
    if not vhdr.exists():
        return None
    try:
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="error")
    except Exception as exc:
        print(f"  [{sid}] read_raw failed: {exc}")
        return None
    raw.filter(l_freq=1.0, h_freq=None, fir_design="firwin", verbose="error")
    if 50.0 < raw.info["sfreq"] / 2:
        raw.notch_filter(freqs=[50.0], verbose="error")
    if raw.info["sfreq"] > 250:
        raw.resample(250.0, verbose="error")
    return raw


def posterior_channel_mean_signal(raw, t_start_sec, t_end_sec):
    """Extract a posterior-channel-mean signal between t_start and t_end (seconds)."""
    available = [c for c in POSTERIOR_CHANNELS if c in raw.ch_names]
    if not available:
        return None, None
    s_start = int(t_start_sec * raw.info["sfreq"])
    s_end = int(t_end_sec * raw.info["sfreq"])
    if s_end > raw.n_times or s_start < 0:
        return None, None
    data, _ = raw[available, s_start:s_end]
    sig = data.mean(axis=0)
    return sig, raw.info["sfreq"]


def event_locked_irasa_centroid(sig, fs):
    """Compute IRASA periodic spectrum and find the peak in [7.0, 8.2] Hz."""
    f, P, P_frac, P_osc = irasa_psd(sig, fs=fs, fmax=20.0, nperseg=int(2 * fs))
    band_mask = (f >= SR1_BAND_LO) & (f <= SR1_BAND_HI)
    if not np.any(band_mask):
        return np.nan, np.nan
    f_band = f[band_mask]
    P_band = P_osc[band_mask]
    if np.all(P_band == 0):
        return np.nan, np.nan
    idx = int(np.argmax(P_band))
    return float(f_band[idx]), float(P_band[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None,
                    help="Run on first N subjects only (for testing).")
    args = ap.parse_args()

    subjects = list_lemon_ec_subjects()
    print(f"Found {len(subjects)} LEMON EC subjects with ≥3 events.")
    if args.subset:
        subjects = subjects[: args.subset]
        print(f"Running on first {len(subjects)} subjects (--subset).")

    per_event_rows = []
    per_subject_rows = []

    for i, (sid, events_df) in enumerate(subjects):
        print(f"[{i+1:>3}/{len(subjects)}] {sid}: {len(events_df)} events")
        raw = load_raw_eeg(sid)
        if raw is None:
            print(f"  raw missing, skip")
            continue

        sub_centroids = []
        sub_weights = []  # use sr1_z_max as proxy for canonicality if template-rho not in events
        for _, ev in events_df.iterrows():
            t0 = ev.get("t0_net", np.nan)
            if not np.isfinite(t0):
                t0 = ev.get("t_start", np.nan)
            if not np.isfinite(t0):
                continue
            sig, fs = posterior_channel_mean_signal(raw, t0 - HALF_WIN, t0 + HALF_WIN)
            if sig is None or len(sig) < int(2 * 250):
                continue
            centroid, peak_power = event_locked_irasa_centroid(sig, fs)
            if not np.isfinite(centroid):
                continue
            per_event_rows.append({
                "subject_id": sid,
                "t0_net": t0,
                "centroid_hz": centroid,
                "peak_power": peak_power,
                "sr1_z_max": ev.get("sr1_z_max", np.nan),
            })
            sub_centroids.append(centroid)
            w = float(ev.get("sr1_z_max", 0.0)) if np.isfinite(ev.get("sr1_z_max", np.nan)) else 0.0
            sub_weights.append(max(w, 0.0))

        if sub_centroids:
            arr = np.array(sub_centroids)
            wts = np.array(sub_weights)
            # SW-weighted median: events with sr1_z_max > 0 contribute proportionally
            if wts.sum() > 0:
                # weighted median via sorted weights
                order = np.argsort(arr)
                sorted_arr = arr[order]
                sorted_wts = wts[order]
                cum = np.cumsum(sorted_wts) / sorted_wts.sum()
                med_idx = int(np.searchsorted(cum, 0.5))
                med_idx = min(med_idx, len(sorted_arr) - 1)
                sw_median = float(sorted_arr[med_idx])
            else:
                sw_median = float(np.median(arr))
            per_subject_rows.append({
                "subject_id": sid,
                "n_events": len(arr),
                "centroid_median_hz": float(np.median(arr)),
                "centroid_sw_median_hz": sw_median,
            })

    pd.DataFrame(per_event_rows).to_csv(OUT_DIR / "per_event_centroids.csv", index=False)
    sub_df = pd.DataFrame(per_subject_rows)
    sub_df.to_csv(OUT_DIR / "per_subject_centroids.csv", index=False)

    if len(sub_df) == 0:
        print("\nNo subject results; aborting.")
        return

    cohort_mean_median = float(sub_df["centroid_median_hz"].mean())
    cohort_mean_sw = float(sub_df["centroid_sw_median_hz"].mean())
    cohort_sd_median = float(sub_df["centroid_median_hz"].std(ddof=1))
    cohort_sd_sw = float(sub_df["centroid_sw_median_hz"].std(ddof=1))

    summary = f"""Event-Locked IRASA Centroid on LEMON EC
========================================
Subjects: {len(sub_df)}
Total events: {len(per_event_rows)}

Cohort-mean centroid (per-subject median, IRASA): {cohort_mean_median:.4f} ± {cohort_sd_median:.4f} Hz
Cohort-mean centroid (SW-weighted median, IRASA): {cohort_mean_sw:.4f} ± {cohort_sd_sw:.4f} Hz

FOOOF baseline (paper headline):                  {FOOOF_BASELINE_CENTROID:.4f} Hz
SR1 cavity fundamental:                           {SR1_CAVITY:.4f} Hz

Gap to cavity (FOOOF):                            {SR1_CAVITY - FOOOF_BASELINE_CENTROID:+.4f} Hz
Gap to cavity (IRASA, median):                    {SR1_CAVITY - cohort_mean_median:+.4f} Hz
Gap to cavity (IRASA, SW):                        {SR1_CAVITY - cohort_mean_sw:+.4f} Hz

Interpretation:
- If IRASA centroid shifts UPWARD from FOOOF (closer to 7.83 Hz),
  the 0.14 Hz gap is methodological FOOOF leading-edge bias.
- If IRASA centroid stays at ~7.69 Hz, the gap is a residual
  neurophysiological offset that the methodological accounts cannot
  fully explain.
"""
    print("\n" + summary)
    with open(OUT_DIR / "cohort_summary.txt", "w") as f:
        f.write(summary)
    print(f"\nResults written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
