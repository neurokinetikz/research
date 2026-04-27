#!/usr/bin/env python3
"""Empirical zR-zP stream-correlation on LEMON EC continuous traces.

Computes the Spearman correlation between the Kuramoto-R z-stream
zR(t) and the PLV-to-median-reference z-stream zP(t) over the full
LEMON EC continuous recording for each subject, then aggregates the
per-subject correlations to a cohort-level statistic.

The PLV stream uses a robust median-reference phase across channels;
the Kuramoto stream uses the Kuramoto order parameter R(t) on the
band-limited Hilbert phases. Both are z-scored via the same robust
estimator the composite-v2 detector uses (`_composite_robust_z`).

The point of this analysis is to convert the *conceptual* argument that
zR and zP measure different things into an empirical defence: if
ρ(zR, zP) is below 1.0 by a meaningful margin, the streams genuinely
add information rather than being numerically redundant.

Output:
    outputs/schumann/2026-04-27-stream-correlation-lemon-ec/
        per_subject.csv
        cohort_summary.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import mne
mne.set_log_level("ERROR")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from lib.detect_ignition import _composite_streams, _composite_robust_z  # noqa: E402

T9 = Path("/Volumes/T9")
if not T9.exists():
    T9 = Path("/mnt/eeg-data/T9")

LEMON_DIR = T9 / "lemon_data" / "eeg_preprocessed" / "EEG_MPILMBB_LEMON" / \
    "EEG_Preprocessed_BIDS_ID" / "EEG_Preprocessed"

OUT_DIR = BASE_DIR / "outputs" / "schumann" / \
    "2026-04-27-stream-correlation-lemon-ec"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lemon_raw(sid):
    path = LEMON_DIR / f"{sid}_EC.set"
    if not path.is_file():
        return None
    try:
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
    except Exception:
        return None
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_picks) == 0:
        return None
    raw.pick(eeg_picks)
    if raw.info["sfreq"] > 250:
        raw.resample(250.0, verbose="error")
    nyq = raw.info["sfreq"] / 2.0
    if nyq > 52.0:
        raw.notch_filter(50.0, verbose="error")
    h_freq = min(59.0, nyq - 1.0)
    raw.filter(l_freq=1.0, h_freq=h_freq, fir_design="firwin", verbose="error")
    return raw


def process_subject(sid):
    raw = load_lemon_raw(sid)
    if raw is None:
        return None
    Y = raw.get_data() * 1e6  # V → µV
    fs = float(raw.info["sfreq"])
    if Y.shape[1] < int(fs * 60):  # require at least 60 s of recording
        return None
    try:
        t, env, R, P, M = _composite_streams(Y, fs, f0=7.6)
    except Exception:
        return None
    if R.size < 30:
        return None
    zE = _composite_robust_z(env)
    zR = _composite_robust_z(R)
    zP = _composite_robust_z(P)
    zM = _composite_robust_z(M)
    finite = (np.isfinite(zR) & np.isfinite(zP) &
              np.isfinite(zE) & np.isfinite(zM))
    if finite.sum() < 30:
        return None

    def sp(a, b):
        try:
            r, _ = scipy_stats.spearmanr(a[finite], b[finite])
            return float(r)
        except Exception:
            return float("nan")

    def pe(a, b):
        try:
            r, _ = scipy_stats.pearsonr(a[finite], b[finite])
            return float(r)
        except Exception:
            return float("nan")

    return {
        "sid": sid,
        "n_samples": int(finite.sum()),
        "duration_sec": float(t[-1] - t[0]),
        "spearman_zR_zP": sp(zR, zP),
        "spearman_zR_zM": sp(zR, zM),
        "spearman_zP_zM": sp(zP, zM),
        "spearman_zE_zR": sp(zE, zR),
        "spearman_zE_zP": sp(zE, zP),
        "spearman_zE_zM": sp(zE, zM),
        "pearson_zR_zP": pe(zR, zP),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args()

    n_workers = int(os.environ.get("SIE_WORKERS", "28"))
    n_workers = min(n_workers, max(1, cpu_count() - 1))

    files = sorted(glob(str(LEMON_DIR / "sub-*_EC.set")))
    sids = [Path(f).stem.replace("_EC", "") for f in files]
    if args.max_subjects:
        sids = sids[: args.max_subjects]
    print(f"LEMON EC: {len(sids)} subjects (workers={n_workers})", flush=True)

    t0 = time.time()
    rows = []
    with Pool(processes=n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_subject, sids,
                                                     chunksize=1)):
            if res is not None:
                rows.append(res)
            if (i + 1) % 25 == 0 or (i + 1) == len(sids):
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(sids)} subjects done "
                      f"({len(rows)} usable), {elapsed:.0f}s elapsed", flush=True)

    if not rows:
        print("no usable subjects", flush=True)
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_subject.csv", index=False)

    def stat(col):
        v = df[col].dropna().values
        return v.mean(), np.median(v), v.std(ddof=1), v.min(), v.max(), len(v)

    lines = [
        "Empirical stream correlations on LEMON EC continuous traces",
        "=" * 64,
        f"N subjects: {len(df)}  (LEMON EC sample)",
        f"Mean recording duration: {df['duration_sec'].mean():.1f} s",
        "",
        f"{'pair':<14}{'mean':>8}{'median':>9}{'sd':>7}"
        f"{'min':>8}{'max':>8}{'N':>5}",
        "-" * 64,
    ]
    for col in ["spearman_zR_zP", "spearman_zR_zM", "spearman_zP_zM",
                "spearman_zE_zR", "spearman_zE_zP", "spearman_zE_zM",
                "pearson_zR_zP"]:
        m, md, sd, mn, mx, n = stat(col)
        label = col.replace("spearman_", "ρ_").replace("pearson_", "r_")
        lines.append(f"{label:<14}{m:>8.3f}{md:>9.3f}{sd:>7.3f}"
                     f"{mn:>8.3f}{mx:>8.3f}{n:>5}")
    lines.append("")
    headline_m, headline_md, _, _, _, _ = stat("spearman_zR_zP")
    lines.append(f"Headline: cohort-mean Spearman ρ(zR, zP) = "
                 f"{headline_m:.3f}, median = {headline_md:.3f}.")
    txt = "\n".join(lines) + "\n"
    (OUT_DIR / "cohort_summary.txt").write_text(txt)
    print("\n" + txt, flush=True)
    print(f"Wrote: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
